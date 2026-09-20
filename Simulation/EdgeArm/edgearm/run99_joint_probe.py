"""Frozen visual-policy contrast with joint camera/workspace constraints."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import multiprocessing as mp
from pathlib import Path
import shutil
import time
from unittest.mock import patch

import numpy as np

from . import run96_clearance_probe as original
from .constrained_recovery_run40 import summarize
from .run42.session import DomainSession
from .run99_joint_feasibility import JointFeasibilityProjection, NOMINAL_WORKSPACE
from .train_staged_hybrid_contact_sac import _atomic_json


def episode(job):
    seed, checkpoint, vision, control, keypoint, output, condition = job
    if condition not in ("baseline", "joint_feasible") or not 97100000 <= seed // 9 < 97100004:
        raise ValueError("registered development contrast required")

    class CheckedSession(DomainSession):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            cfg = self.env.config
            if not np.array_equal([cfg.workspace_x, cfg.workspace_y, cfg.workspace_z], NOMINAL_WORKSPACE):
                raise ValueError("controller calibration must match unchanged plant workspace")

    output = Path(output) / condition
    underlying = "baseline" if condition == "baseline" else "camera_clearance"
    with (patch.object(original, "CameraClearanceProjection", JointFeasibilityProjection),
          patch.object(original, "DomainSession", CheckedSession)):
        result = original.episode((seed, checkpoint, vision, control, keypoint, str(output), underlying))
    result.update(condition=condition, joint_workspace_camera_constraint=condition == "joint_feasible",
                  workspace_inset_m=.0005 if condition == "joint_feasible" else None,
                  controller_is_learned=False, plant_unchanged=True)
    folder = output / underlying / "memory_fast_keypoints" / f"episode_{seed}"
    _atomic_json(folder / "result.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "vision", "control", "keypoint", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--workers", type=int, choices=range(1, 10), default=9)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    source = a.output / "source_snapshot"
    source.mkdir()
    for name in ("run99_joint_probe.py", "run99_joint_feasibility.py", "run96_clearance_probe.py",
                 "run96_camera_clearance.py", "run93_fast_observation.py", "run91_static_memory.py",
                 "run89_geometric_memory.py", "run78_completion_probe.py"):
        shutil.copy2(Path(__file__).with_name(name), source / name)
    state = dict(run="Run99", status="running", started=time.time(), evaluations=[],
                 phase="joint_camera_workspace_development", completed=0, total=72,
                 model_updated=False, independent_acceptance=False, actor_uses_simulator_state=False,
                 projection_is_learned=False, plant_unchanged=True, max_steps=900, fixed_survey_steps=220,
                 production_admission=False, export_admission=False, final_vla_acceptance=False,
                 checkpoint_hashes={k: hashlib.sha256(getattr(a, k).read_bytes()).hexdigest()
                                    for k in ("checkpoint", "vision", "control", "keypoint")},
                 source_hashes={x.name: hashlib.sha256(x.read_bytes()).hexdigest() for x in source.iterdir()})

    def publish():
        state.update(updated=time.time(), elapsed_seconds=time.time() - state["started"])
        _atomic_json(a.output / "run_state.json", state)

    publish()
    try:
        for condition in ("baseline", "joint_feasible"):
            state.update(condition=condition, phase_episodes_completed=0, phase_episodes_total=36)
            publish()
            jobs = [(g * 9 + r, str(a.checkpoint), str(a.vision), str(a.control), str(a.keypoint),
                     str(a.output), condition) for g in range(97100000, 97100004) for r in range(9)]
            results = []
            with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
                for f in as_completed([pool.submit(episode, j) for j in jobs]):
                    results.append(f.result())
                    state.update(completed=state["completed"] + 1, phase_episodes_completed=len(results),
                                 partial=summarize(results))
                    publish()
            summary = summarize(results) | dict(condition=condition,
                block_out_of_bounds=sum(r["terminal_reason"] == "block_out_of_bounds" for r in results),
                pairs={str(r): summarize([v for v in results if v["seed"] % 9 == r]) for r in range(9)},
                projection_active=sum(r["projection_active"] for r in results))
            _atomic_json(a.output / condition / "summary.json", dict(summary=summary, results=results))
            state["evaluations"].append(summary)
            publish()
        state.update(status="complete_pending_review", phase="finished")
        publish()
    except BaseException as exc:
        state.update(status="failed", error=repr(exc))
        publish()
        raise


if __name__ == "__main__":
    main()
