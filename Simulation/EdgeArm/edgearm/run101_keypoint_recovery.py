"""Refresh image keypoints on replay-verified on-policy training-state images.

Old/new training sources and all routes are sampled equally. Frozen spatial
memory, action model, joint constraints, survey and hold scripts are unchanged.
This is supervised perception training, not RL or independent acceptance.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import multiprocessing as mp
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .constrained_recovery_run40 import summarize
from .run34_repeat_eval import deterministic_runtime
from .run82_train_spatial import Sequences
from .run88_keypoint_model import WristKeypoints, keypoint_loss
from .run88_train_keypoints import load_frames, validate
from .run99_joint_probe import episode as evaluate
from .run101_replay_keypoint_labels import prepare, training_split
from .train_staged_hybrid_contact_sac import _atomic_json


def load_new(records):
    rows = {k: [] for k in ("rgb", "pose", "K", "xyz", "present", "visible", "validation", "route")}
    for r in records:
        if r["split"] != training_split(r["seed"]):
            raise ValueError("perception group split mismatch")
        with np.load(Path(r["folder"]) / "inputs.npz", allow_pickle=False) as z:
            if set(z.files) != {"rgb", "pose", "K", "time_step"}:
                raise ValueError("privileged field in input packet")
            n = len(z["rgb"])
            for k in ("rgb", "pose"):
                rows[k].append(z[k].copy())
            rows["K"].append(np.broadcast_to(z["K"], (n, 3, 3)))
        with np.load(Path(r["folder"]) / "labels.npz", allow_pickle=False) as z:
            for k in ("xyz", "present", "visible"):
                rows[k].append(z[k].copy())
        rows["validation"].append(np.full(n, r["split"] == "validation", bool))
        rows["route"].append(np.full(n, r["route"], np.int64))
    return {k: torch.from_numpy(np.concatenate(v)).cuda() for k, v in rows.items()}


def draw_balanced(data, per_route=8):
    pools = [torch.nonzero((~data["validation"]) & (data["route"] == r)).flatten() for r in range(9)]
    if any(len(p) == 0 for p in pools):
        raise ValueError("each training source must cover every route")
    return torch.cat([p[torch.randint(len(p), (per_route,), device=p.device)] for p in pools])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "old-manifest", "checkpoint", "vision", "control", "keypoint", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--workers", type=int, choices=range(1, 10), default=9)
    p.add_argument("--smoke-only", action="store_true")
    a = p.parse_args()
    deterministic_runtime()
    torch.set_num_threads(2)
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    deadline = started + 2400
    state = dict(run="Run101", status="running", phase="replay_training_images", started=started,
                 step=0, total_updates=8 if a.smoke_only else 1800, evaluations=[],
                 training_kind="supervised_keypoint_recovery_not_RL", model_updated=True,
                 action_policy_changed=False, actor_uses_simulator_state=False,
                 independent_acceptance=False, production_admission=False, export_admission=False,
                 final_vla_acceptance=False, max_steps=900, fixed_survey_steps=220,
                 checkpoint_hashes={k: hashlib.sha256(getattr(a, k).read_bytes()).hexdigest()
                                    for k in ("checkpoint", "vision", "control", "keypoint")})
    snapshot = a.output / "source_snapshot"
    snapshot.mkdir()
    for name in ("run101_keypoint_recovery.py", "run101_replay_keypoint_labels.py", "run88_keypoint_model.py",
                 "run88_train_keypoints.py", "run99_joint_probe.py", "run99_joint_feasibility.py",
                 "run96_clearance_probe.py", "run96_camera_clearance.py", "run93_fast_observation.py",
                 "run91_static_memory.py", "run89_geometric_memory.py", "run78_completion_probe.py"):
        shutil.copy2(Path(__file__).with_name(name), snapshot / name)
    state["source_hashes"] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in snapshot.iterdir()}

    def publish(phase=None):
        if phase:
            state["phase"] = phase
        state.update(updated=time.time(), elapsed_seconds=time.time() - started)
        _atomic_json(a.output / "run_state.json", state)

    def progress(n, total):
        state.update(recode_completed=n, recode_total=total)
        publish()

    publish()
    try:
        paths = sorted(a.source.glob("collection_[123]/memory_fast_keypoints/episode_*/trace.npz"))
        if len(paths) != 108 or len({int(f.parent.name[8:]) for f in paths}) != 108:
            raise ValueError("all 108 original training sources required")
        for f in paths:
            training_split(int(f.parent.name[8:]))
        if a.smoke_only:
            paths = paths[:9] + paths[-9:]
        records = []
        state.update(replay_total=len(paths), replay_completed=0)
        publish()
        with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
            jobs = [(str(f.parent), str(a.output / "perception_data")) for f in paths]
            for f in as_completed([pool.submit(prepare, j) for j in jobs]):
                if time.time() > deadline - 600:
                    raise TimeoutError("reserve training and full development time")
                records.append(f.result())
                state.update(replay_completed=len(records), replay_frames=sum(r["frames"] for r in records))
                publish()
        records.sort(key=lambda r: r["seed"])
        _atomic_json(a.output / "replay_manifest.json", dict(records=records, labels_never_actor_inputs=True,
            replay_not_new_collection=True, perception_validation_not_independent=True))
        dataset = Sequences(a.old_manifest)
        if a.smoke_only:
            dataset.records = [r for r in dataset.records if r["variant"] == -1][:9] + [
                r for r in dataset.records if r["split"] == "validation" and r["variant"] == -1][:9]
        old = load_frames(dataset, progress)
        new = load_new(records)
        state["source_counts"] = {name: dict(train=int((~data["validation"]).sum()),
            validation=int(data["validation"].sum()),
            route_frames=torch.bincount(data["route"][~data["validation"]], minlength=9).tolist())
            for name, data in (("old", old), ("new", new))}
        model = WristKeypoints().cuda()
        model.load_state_dict(torch.load(a.keypoint, map_location="cpu", weights_only=True)["model"])
        state["perception_before"] = {"old": validate(model, old), "new": validate(model, new)}
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4, fused=True)
        actual_counts = np.zeros((2, 9), np.int64)
        best = None
        for step in range(1, state["total_updates"] + 1):
            if time.time() > deadline - 180:
                raise TimeoutError("reserve complete development time")
            model.train()
            ids = [draw_balanced(data) for data in (old, new)]
            batches = {k: torch.cat([data[k][idx] for data, idx in zip((old, new), ids)])
                       for k in ("rgb", "pose", "K", "xyz", "present", "visible")}
            for source, (data, idx) in enumerate(zip((old, new), ids)):
                actual_counts[source] += torch.bincount(data["route"][idx], minlength=9).cpu().numpy()
            output = model(batches["rgb"], batches["pose"], batches["K"])
            loss, metrics = keypoint_loss(output, batches["xyz"], batches["present"], batches["visible"],
                                         batches["pose"], batches["K"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            state.update(step=step, metrics=dict(loss=float(loss.detach()), **metrics))
            if step % 20 == 0:
                publish("keypoint_recovery_fit")
            if step % 600 == 0 or step == state["total_updates"]:
                model.eval()
                perception = {"old": validate(model, old), "new": validate(model, new)}
                candidate = a.output / f"keypoints_step_{step}.pt"
                torch.save(dict(model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    policy_kind=model.kind, actor_uses_simulator_state=False, nominal_height_prior=True,
                    perception_validation=perception, production_admission=False), candidate)
                state.update(perception_latest=perception, actual_source_route_counts=actual_counts.tolist())
                if a.smoke_only:
                    break
                folder = a.output / f"development_{step}"
                results = []
                state.update(phase_episodes_completed=0, phase_episodes_total=36)
                publish("frozen_action_autonomous_development")
                jobs = [(g * 9 + r, str(a.checkpoint), str(a.vision), str(a.control), str(candidate),
                    str(folder), "joint_feasible") for g in range(97100000, 97100004) for r in range(9)]
                with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
                    for f in as_completed([pool.submit(evaluate, j) for j in jobs]):
                        results.append(f.result())
                        state.update(phase_episodes_completed=len(results), partial=summarize(results))
                        publish()
                summary = summarize(results) | dict(step=step, keypoint_checkpoint=str(candidate),
                    keypoint_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
                    block_out_of_bounds=sum(r["terminal_reason"] == "block_out_of_bounds" for r in results),
                    pairs={str(r): summarize([v for v in results if v["seed"] % 9 == r]) for r in range(9)})
                _atomic_json(folder / "summary.json", dict(summary=summary, results=results))
                state["evaluations"].append(summary)
                eligible = summary["hard_failures"] <= 1 and summary["block_out_of_bounds"] == 0
                rank = (eligible, summary["successes"], -summary["hard_failures"], summary["mean_coverage"])
                if best is None or rank > best:
                    best = rank
                    state.update(best_keypoint=str(candidate), best_development_rank=list(rank))
                state["development_gate_passed"] = bool(best[0] and best[1] >= 26)
                publish()
                if state["development_gate_passed"]:
                    break
        state.update(status="smoke_complete" if a.smoke_only else "complete_pending_review", phase="finished")
        publish()
    except BaseException as exc:
        state.update(status="failed", error=repr(exc))
        publish()
        raise


if __name__ == "__main__":
    main()
