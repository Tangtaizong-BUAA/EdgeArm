"""Bounded action-label alignment to the unchanged joint-feasible controller.

Only existing Run94 training queries are relabelled, using reported joints and
static geometry. These remain queried teacher labels, including failed-source
episodes, not certified successful demonstrations or reinforcement learning.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import multiprocessing as mp
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .candidate_command_contract_v2 import ACTION_CONTRACT
from .constrained_recovery_run40 import summarize
from .run34_repeat_eval import deterministic_runtime
from .run42.domain import sample_domain
from .run42.session import DomainSession
from .run63_control_probe import TinyTarget
from .run94_control_fit import route_pools, train_step
from .run99_joint_feasibility import JointFeasibilityProjection
from .run99_joint_probe import episode as evaluate
from .sparse_4d_vla_act_v26 import Sparse4DVLAConfigV26
from .train_staged_hybrid_contact_sac import _atomic_json


def validate_training_file(path):
    path = Path(path)
    if path.name != "action_labels.npz" or not path.parent.name.startswith("episode_"):
        raise ValueError("Run94 training query file required")
    seed = int(path.parent.name.removeprefix("episode_"))
    if not 100500000 <= seed // 9 < 100500012:
        raise ValueError("development and independent queries are never training labels")
    return seed


def align_episode(path):
    path = Path(path)
    seed = validate_training_file(path)
    with np.load(path, allow_pickle=False) as z:
        x, y, steps = z["x"].copy(), z["command"].copy(), z["time_step"].copy()
    if x.shape != (len(y), 118) or y.shape[1:] != (6,) or np.any(steps < 220):
        raise ValueError("causal 118-dimensional post-survey labels required")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("nonfinite labels")
    session = DomainSession(
        Sparse4DVLAConfigV26(language_max_tokens=128, visual_memory_mode="episode_anchors_v54"),
        ACTION_CONTRACT, seed, sample_domain(seed + 6001, 0), render=False)
    accepted, targets, changed = [], [], []
    try:
        projection = JointFeasibilityProjection(session.env.model, session.env._ids["tool_site"])
        for i, label in enumerate(y):
            command, audit = projection.project(x[i, :12], label)
            if audit["feasible"]:
                accepted.append(i)
                targets.append(command)
                changed.append(float(np.max(np.abs(command - label)) * .055))
    finally:
        session.close()
    ids = np.asarray(accepted, dtype=np.int64)
    audit = dict(seed=seed, source=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 original_rows=len(y), accepted_rows=len(ids), rejected_rows=len(y) - len(ids),
                 changed_over_1mrad=sum(v > .001 for v in changed),
                 correction_max_rad=max(changed, default=0),
                 certified_success_demonstrations=False, uses_heldout=False)
    return x[ids], np.asarray(targets, np.float32).reshape(-1, 6), np.full(len(ids), seed % 9), audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "checkpoint", "vision", "control", "keypoint", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--workers", type=int, choices=range(1, 10), default=9)
    a = p.parse_args()
    deterministic_runtime()
    torch.set_num_threads(2)
    a.output.mkdir(parents=True, exist_ok=False)
    deadline = time.time() + 1500
    state = dict(run="Run100", status="running", phase="feasible_training_label_alignment",
                 started=time.time(), step=0, total_updates=1200, evaluations=[], completed=0,
                 training_kind="supervised_kinematic_label_alignment_not_RL", actor_uses_simulator_state=False,
                 independent_acceptance=False, production_admission=False, export_admission=False,
                 final_vla_acceptance=False, checkpoint_hashes={k: hashlib.sha256(getattr(a, k).read_bytes()).hexdigest()
                     for k in ("checkpoint", "vision", "control", "keypoint")})
    source = a.output / "source_snapshot"
    source.mkdir()
    for name in ("run100_feasible_labels.py", "run94_control_fit.py", "run99_joint_probe.py",
                 "run99_joint_feasibility.py", "run96_clearance_probe.py", "run96_camera_clearance.py",
                 "run93_fast_observation.py", "run91_static_memory.py", "run89_geometric_memory.py",
                 "run78_completion_probe.py"):
        shutil.copy2(Path(__file__).with_name(name), source / name)
    state["source_hashes"] = {x.name: hashlib.sha256(x.read_bytes()).hexdigest() for x in source.iterdir()}

    def publish(phase=None):
        if phase:
            state["phase"] = phase
        state.update(updated=time.time(), elapsed_seconds=time.time() - state["started"])
        _atomic_json(a.output / "run_state.json", state)

    publish()
    try:
        files = sorted(a.source.glob("collection_[123]/memory_fast_keypoints/episode_*/action_labels.npz"))
        if len(files) != 108 or len({validate_training_file(f) for f in files}) != 108:
            raise ValueError("all 108 distinct original training episodes required")
        aligned = []
        with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
            for future in as_completed([pool.submit(align_episode, f) for f in files]):
                if time.time() > deadline - 400:
                    raise TimeoutError("reserve complete model evaluation time")
                aligned.append(future.result())
                state.update(label_episodes_completed=len(aligned), label_episodes_total=len(files))
                publish()
        aligned.sort(key=lambda row: row[3]["seed"])
        x, y, routes = (np.concatenate([v[i] for v in aligned]) for i in range(3))
        audits = [v[3] for v in aligned]
        _atomic_json(a.output / "label_audit.json", dict(episodes=audits))
        np.savez_compressed(a.output / "aligned_labels.npz", x=x, command=y, route=routes)
        state.update(online_states=len(x), actual_online_route_counts=np.bincount(routes, minlength=9).tolist(),
                     label_changed_over_1mrad=sum(r["changed_over_1mrad"] for r in audits),
                     label_rejected=sum(r["rejected_rows"] for r in audits))
        anchor = TinyTarget(118, 512).cuda().eval().requires_grad_(False)
        anchor.load_state_dict(torch.load(a.control, map_location="cpu", weights_only=True)["model"])
        model = deepcopy(anchor).requires_grad_(True)
        with np.load(a.source / "functional_replay.npz", allow_pickle=False) as z:
            old_x = torch.as_tensor(z["x"].copy(), device="cuda")
            old_routes = z["route"].copy()
        with torch.no_grad():
            old_y = torch.cat([anchor(v) for v in old_x.split(2048)])
        old = (old_x, old_y, route_pools(old_routes, "cuda"))
        new = (torch.as_tensor(x, device="cuda"), torch.as_tensor(y, device="cuda"), route_pools(routes, "cuda"))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5, fused=True)
        best = None
        for rnd in (1, 2):
            state.update(round=rnd)
            for update in range(600):
                if time.time() > deadline - 180:
                    raise TimeoutError("reserve full development evaluation")
                state["metrics"] = train_step(model, anchor, optimizer, old, new)
                state["step"] += 1
                if update % 50 == 0:
                    publish("feasible_action_supervised_fit")
            candidate = a.output / f"control_round_{rnd}.pt"
            torch.save(dict(model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                            actor_uses_simulator_state=False, original_ACT_checkpoint=False,
                            training_kind=state["training_kind"], production_admission=False,
                            export_admission=False, final_vla_acceptance=False), candidate)
            results = []
            folder = a.output / f"development_{rnd}"
            state.update(phase_episodes_completed=0, phase_episodes_total=36)
            publish("autonomous_joint_feasible_development")
            jobs = [(g * 9 + r, str(a.checkpoint), str(a.vision), str(candidate), str(a.keypoint),
                     str(folder), "joint_feasible") for g in range(97100000, 97100004) for r in range(9)]
            with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
                for future in as_completed([pool.submit(evaluate, j) for j in jobs]):
                    results.append(future.result())
                    state.update(phase_episodes_completed=len(results), partial=summarize(results))
                    publish()
            summary = summarize(results) | dict(round=rnd, checkpoint=str(candidate),
                checkpoint_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
                block_out_of_bounds=sum(r["terminal_reason"] == "block_out_of_bounds" for r in results),
                pairs={str(r): summarize([v for v in results if v["seed"] % 9 == r]) for r in range(9)})
            _atomic_json(folder / "summary.json", dict(summary=summary, results=results))
            state["evaluations"].append(summary)
            eligible = summary["hard_failures"] <= 1 and summary["block_out_of_bounds"] == 0
            rank = (eligible, summary["successes"], -summary["hard_failures"], summary["mean_coverage"])
            if best is None or rank > best:
                best = rank
                state.update(best_control=str(candidate), best_development_rank=list(rank))
            state["development_gate_passed"] = bool(best[0] and best[1] >= 26)
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
