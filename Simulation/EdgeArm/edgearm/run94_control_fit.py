"""Action-only adaptation to frozen image-keypoint/spatial-memory inputs.

Replay preserves the previous learned action function on old training images.
New labels are DAgger queries, not guaranteed successful demonstrations or RL.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .constrained_recovery_run40 import summarize
from .run34_repeat_eval import deterministic_runtime
from .run63_control_probe import TinyTarget
from .run68_visual_control import estimated_inputs
from .run82_spatial_model import SparseSpatialPolicy
from .run82_train_spatial import Sequences, sequence_chunk, encode_chunk
from .run88_keypoint_model import WristKeypoints
from .run91_static_memory import StaticSurveyMemory
from .run93_fast_observation import episode as evaluate
from .run94_collect import episode as collect
from .train_staged_hybrid_contact_sac import _atomic_json


def replay_encode(manifest, map_path, keypoint_path, anchor, publish, state, deadline, limit=None):
    dataset = Sequences(manifest)
    rows = [r for r in dataset.records if r["split"] == "train"]
    if limit is not None:
        rows = [next(r for r in rows if r["route"] == k and r["variant"] == -1) for k in range(9)]
    spatial = SparseSpatialPolicy().cuda().eval()
    spatial.load_state_dict(torch.load(map_path, map_location="cpu", weights_only=True)["model"])
    keypoint = WristKeypoints().cuda().eval()
    keypoint.load_state_dict(torch.load(keypoint_path, map_location="cpu", weights_only=True)["model"])
    xs, ys, routes = [], [], []
    with torch.inference_mode():
        for count, row in enumerate(rows):
            if time.time() > deadline - 300:
                raise TimeoutError("reserve control/evaluation time")
            item = dataset.load(row)
            n = len(item[0]["time_step"])
            ins, _, _ = sequence_chunk([item], 0, n)
            observed, hidden = encode_chunk(spatial, ins)
            measurements = keypoint(ins["rgb"][0, :, -1], ins["pose"][0, :, -1], ins["K"][0])
            memory = None
            correction = StaticSurveyMemory()
            for t, step in enumerate(item[0]["time_step"]):
                current = {k: v[:, t] for k, v in ins.items()}
                _, memory = spatial.step(**current, state=memory, encoded=(observed[:, t], hidden[:, t]))
                measurement = {
                    k: v[t : t + 1] for k, v in measurements.items() if k in ("xyz", "confidence", "valid")
                }
                memory, _ = correction.update(memory, measurement, int(step))
                if step < 220:
                    continue
                x = estimated_inputs(current["proprio"], memory["xyz"][..., :2], current["selected"])
                xs.append(x[0].cpu().numpy())
                # Functional replay deliberately does not call the old teacher.
                ys.append(anchor(x)[0].cpu().numpy())
                routes.append(row["route"])
            state.update(replay_sequences_done=count + 1, replay_sequences_total=len(rows))
            if count % 10 == 0 or count + 1 == len(rows):
                publish("encode_old_training_replay")
    state.update(
        replay_physical_episodes=len({r["seed"] for r in rows}),
        replay_sequences=len(rows),
        replay_states=len(xs),
        replay_old_frame_stride=8,
        replay_labels="frozen_anchor_predictions_on_training_images",
    )
    del spatial, keypoint
    torch.cuda.empty_cache()
    return tuple(np.asarray(v) for v in (xs, ys, routes))


def route_pools(routes, device):
    pools = [torch.as_tensor(np.flatnonzero(routes == r), device=device) for r in range(9)]
    if any(len(p) == 0 for p in pools):
        raise ValueError("all nine routes must be present")
    return pools


def train_step(model, anchor, optimizer, replay, online, rng=None):
    old_x, old_y, old_pools = replay
    new_x, new_y, new_pools = online
    old_ids = torch.cat([p[torch.randint(len(p), (32,), device=p.device)] for p in old_pools])
    new_ids = torch.cat([p[torch.randint(len(p), (64,), device=p.device)] for p in new_pools])
    scale = model.yscale.clamp_min(0.025)
    x = new_x[new_ids]
    # Small spatial jitter trains tolerance, without falsifying action history.
    noisy = x.clone()
    noisy[:, 108:112] += torch.randn_like(noisy[:, 108:112]) * 0.002
    prediction = model(noisy)
    supervised = F.smooth_l1_loss(prediction / scale, new_y[new_ids] / scale, beta=0.2)
    retention = F.smooth_l1_loss(model(old_x[old_ids]) / scale, old_y[old_ids] / scale, beta=0.2)
    with torch.no_grad():
        reference = anchor(x)
    trust = F.smooth_l1_loss(model(x) / scale, reference / scale, beta=0.2)
    loss = supervised + 0.2 * retention + 0.02 * trust
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return dict(
        loss=float(loss.detach()),
        supervised=float(supervised.detach()),
        retention=float(retention.detach()),
        trust=float(trust.detach()),
        grad_norm=float(norm),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "checkpoint", "vision", "control", "keypoint", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--rounds", type=int, choices=(1, 2, 3), default=3)
    p.add_argument("--updates", type=int, choices=(10, 600, 1200), default=1200)
    p.add_argument("--workers", type=int, choices=range(1, 10), default=9)
    p.add_argument("--wall-seconds", type=int, default=3300)
    p.add_argument("--smoke-only", action="store_true")
    args = p.parse_args()
    if not 120 <= args.wall_seconds <= 3300 or (args.updates == 10 and not args.smoke_only):
        raise ValueError("bounded experiment required")
    deterministic_runtime()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=False)
    state = dict(
        run="Run94",
        status="running",
        phase="initializing",
        started=time.time(),
        step=0,
        total_updates=args.rounds * args.updates,
        evaluations=[],
        collections=[],
        checkpoint_hashes={
            k: hashlib.sha256(getattr(args, k).read_bytes()).hexdigest()
            for k in ("checkpoint", "vision", "control", "keypoint")
        },
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        training_kind="frozen_perception_onpolicy_DAgger_and_functional_replay_not_RL",
        actor_uses_simulator_state=False,
        independent_acceptance=False,
        target_rate=0.6,
        fixed_survey_steps=220,
        max_steps=900,
        teacher_betas=[0.5, 0.25, 0.0],
        production_admission=False,
        export_admission=False,
        final_vla_acceptance=False,
    )
    source = args.output / "source_snapshot"
    source.mkdir()
    for name in (
        "run94_control_fit.py",
        "run94_collect.py",
        "run93_fast_observation.py",
        "run91_static_memory.py",
        "run89_geometric_memory.py",
        "run82_spatial_model.py",
        "run88_keypoint_model.py",
        "run78_completion_probe.py",
    ):
        shutil.copy2(Path(__file__).with_name(name), source / name)
    state["source_hashes"] = {x.name: hashlib.sha256(x.read_bytes()).hexdigest() for x in source.iterdir()}

    def publish(phase=None):
        if phase:
            state["phase"] = phase
        state.update(updated=time.time(), elapsed_seconds=time.time() - state["started"])
        _atomic_json(args.output / "run_state.json", state)

    publish()
    deadline = state["started"] + args.wall_seconds
    try:
        anchor = TinyTarget(118, 512).cuda().eval()
        anchor.load_state_dict(torch.load(args.control, map_location="cpu", weights_only=True)["model"])
        anchor.requires_grad_(False)
        replay = replay_encode(
            args.manifest,
            args.checkpoint,
            args.keypoint,
            anchor,
            publish,
            state,
            deadline,
            9 if args.smoke_only else None,
        )
        np.savez_compressed(
            args.output / "functional_replay.npz", x=replay[0], command=replay[1], route=replay[2]
        )
        if args.smoke_only:
            tensor = lambda d: (
                torch.tensor(d[0], device="cuda"),
                torch.tensor(d[1], device="cuda"),
                route_pools(d[2], "cuda"),
            )
            model = deepcopy(anchor).requires_grad_(True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-5, fused=True)
            for _ in range(args.updates):
                state["metrics"] = train_step(model, anchor, optimizer, tensor(replay), tensor(replay))
                state["step"] += 1
            publish("collection_smoke")
            smoke = collect(
                (
                    100500020 * 9 + 6,
                    str(args.checkpoint),
                    str(args.vision),
                    str(args.control),
                    str(args.keypoint),
                    str(args.output / "collection_smoke"),
                    "memory_fast_keypoints",
                    0.5,
                )
            )
            if smoke["teacher_queries"] < 1:
                raise ValueError("smoke failed to reach post-survey label collection")
            state.update(status="smoke_complete", phase="finished", smoke_collection=smoke)
            publish()
            return
        current = args.control
        accumulated = []
        best = None
        for rnd in range(args.rounds):
            if time.time() > deadline - 360:
                raise TimeoutError("reserve full development time")
            state.update(round=rnd + 1, phase_episodes_completed=0, phase_episodes_total=36)
            publish("on_policy_teacher_collection")
            folder = args.output / f"collection_{rnd + 1}"
            results = []
            jobs = [
                (
                    g * 9 + r,
                    str(args.checkpoint),
                    str(args.vision),
                    str(current),
                    str(args.keypoint),
                    str(folder),
                    "memory_fast_keypoints",
                    state["teacher_betas"][rnd],
                )
                for g in range(100500000 + rnd * 4, 100500004 + rnd * 4)
                for r in range(9)
            ]
            with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as pool:
                for future in as_completed([pool.submit(collect, j) for j in jobs]):
                    results.append(future.result())
                    state.update(phase_episodes_completed=len(results), partial=summarize(results))
                    publish()
            collection = summarize(results) | dict(
                round=rnd + 1,
                teacher_assisted=state["teacher_betas"][rnd] > 0,
                beta=state["teacher_betas"][rnd],
                teacher_queries=sum(r["teacher_queries"] for r in results),
            )
            state["collections"].append(collection)
            _atomic_json(folder / "summary.json", dict(summary=collection, results=results))
            for row in results:
                path = folder / "memory_fast_keypoints" / f"episode_{row['seed']}" / "action_labels.npz"
                with np.load(path, allow_pickle=False) as z:
                    accumulated.append(
                        (z["x"].copy(), z["command"].copy(), np.full(len(z["x"]), row["seed"] % 9))
                    )
            online = tuple(np.concatenate([d[i] for d in accumulated]) for i in range(3))
            tensor = lambda d: (
                torch.tensor(d[0], device="cuda"),
                torch.tensor(d[1], device="cuda"),
                route_pools(d[2], "cuda"),
            )
            old, new = tensor(replay), tensor(online)
            model = TinyTarget(118, 512).cuda()
            model.load_state_dict(torch.load(current, map_location="cpu", weights_only=True)["model"])
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-5, fused=True)
            state.update(
                online_states=len(online[0]),
                actual_online_route_counts=np.bincount(online[2], minlength=9).tolist(),
            )
            for update in range(args.updates):
                if time.time() > deadline - 180:
                    raise TimeoutError("reserve evaluation time")
                state["metrics"] = train_step(model, anchor, optimizer, old, new)
                state["step"] += 1
                if update % 50 == 0:
                    publish("action_only_supervised_fit")
            candidate = args.output / f"control_round_{rnd + 1}.pt"
            torch.save(
                dict(
                    model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    actor_uses_simulator_state=False,
                    original_ACT_checkpoint=False,
                    training_kind=state["training_kind"],
                    perception_hashes=state["checkpoint_hashes"],
                    production_admission=False,
                    export_admission=False,
                    final_vla_acceptance=False,
                ),
                candidate,
            )
            del model, optimizer, old, new
            torch.cuda.empty_cache()
            results = []
            dev = args.output / f"development_{rnd + 1}"
            state.update(phase_episodes_completed=0, phase_episodes_total=36)
            publish("autonomous_development")
            jobs = [
                (
                    g * 9 + r,
                    str(args.checkpoint),
                    str(args.vision),
                    str(candidate),
                    str(args.keypoint),
                    str(dev),
                    "memory_fast_keypoints",
                )
                for g in range(97100000, 97100004)
                for r in range(9)
            ]
            with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as pool:
                for future in as_completed([pool.submit(evaluate, j) for j in jobs]):
                    results.append(future.result())
                    state.update(phase_episodes_completed=len(results), partial=summarize(results))
                    publish()
            summary = summarize(results) | dict(
                round=rnd + 1,
                checkpoint=str(candidate),
                block_out_of_bounds=sum(r["terminal_reason"] == "block_out_of_bounds" for r in results),
                pairs={str(r): summarize([x for x in results if x["seed"] % 9 == r]) for r in range(9)},
            )
            state["evaluations"].append(summary)
            _atomic_json(dev / "summary.json", dict(summary=summary, results=results))
            eligible = summary["hard_failures"] <= 1 and summary["block_out_of_bounds"] == 0
            rank = (eligible, summary["successes"], -summary["hard_failures"], summary["mean_coverage"])
            if best is None or rank > best:
                best = rank
                state.update(best_control=str(candidate), best_development_rank=list(rank))
            # The initial anchor remains available; do not continue a clearly degraded head.
            current = Path(state["best_control"]) if best[1] >= 22 else args.control
            state.update(
                current_control=str(current), development_gate_passed=bool(best[0] and best[1] >= 26)
            )
            publish()
            if state["development_gate_passed"]:
                break
        state.update(status="complete_pending_review", phase="finished")
        publish()
    except BaseException as exc:
        state.update(status="failed", error=repr(exc))
        publish()
        raise


if __name__ == "__main__":
    main()
