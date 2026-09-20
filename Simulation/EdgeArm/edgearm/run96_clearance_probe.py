"""Paired development test of a static-kinematic camera clearance constraint."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import multiprocessing as mp
from pathlib import Path
import shutil
import time
from unittest.mock import patch

import torch

from . import run93_fast_observation as policy
from .constrained_recovery_run40 import summarize
from .run42.session import DomainSession
from .run96_camera_clearance import CameraClearanceProjection
from .train_staged_hybrid_contact_sac import _atomic_json


def episode(job):
    seed, checkpoint, vision, control, keypoint, output, condition = job
    if condition not in ("baseline", "camera_clearance") or not 97100000 <= seed // 9 < 97100004:
        raise ValueError("declared development contrast required")
    holder = {}
    changes = []
    original_factor = policy.factor_command

    class CalibratedSession(DomainSession):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            holder["projection"] = CameraClearanceProjection(self.env.model, self.env._ids["tool_site"])

    def constrained_factor(mode, proprio, selected, world, memory, anchor, learned):
        prediction = original_factor(mode, proprio, selected, world, memory, anchor, learned)
        if condition == "baseline":
            return prediction
        if prediction.shape != (1, 6):
            raise ValueError("single causal command required")
        action, audit = holder["projection"].project(
            proprio[0, :12].detach().cpu().numpy(), prediction[0].detach().cpu().numpy()
        )
        changes.append(audit)
        return torch.from_numpy(action).to(prediction.device)[None]

    directory = Path(output) / condition
    with (
        patch.object(policy, "DomainSession", CalibratedSession),
        patch.object(policy, "factor_command", constrained_factor),
    ):
        result = policy.episode(
            (seed, checkpoint, vision, control, keypoint, str(directory), "memory_fast_keypoints")
        )
    result.update(
        condition=condition,
        camera_clearance_constraint=condition == "camera_clearance",
        clearance_m=0.002 if condition == "camera_clearance" else None,
        projection_calls=len(changes),
        projection_active=sum(r["active"] for r in changes),
        projection_only_uses_reported_joints_and_static_calibration=True,
        projection_is_learned=False,
        projection_counts_are_proposals_before_scripted_completion_override=True,
    )
    folder = directory / "memory_fast_keypoints" / f"episode_{seed}"
    _atomic_json(folder / "result.json", result)
    _atomic_json(folder / "clearance_projection.json", dict(predictions_only=True, events=changes))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "vision", "control", "keypoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 10), default=9)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.output / "source_snapshot"
    source.mkdir()
    for name in (
        "run96_clearance_probe.py",
        "run96_camera_clearance.py",
        "run93_fast_observation.py",
        "run91_static_memory.py",
        "run89_geometric_memory.py",
        "run78_completion_probe.py",
    ):
        shutil.copy2(Path(__file__).with_name(name), source / name)
    state = dict(
        run="Run96",
        status="running",
        started=time.time(),
        evaluations=[],
        phase="camera_clearance_development",
        completed=0,
        total=72,
        step=0,
        total_updates=0,
        model_updated=False,
        independent_acceptance=False,
        actor_uses_simulator_state=False,
        projection_is_learned=False,
        clearance_m=0.002,
        max_steps=900,
        fixed_survey_steps=220,
        production_admission=False,
        export_admission=False,
        final_vla_acceptance=False,
        checkpoint_hashes={
            k: hashlib.sha256(getattr(args, k).read_bytes()).hexdigest()
            for k in ("checkpoint", "vision", "control", "keypoint")
        },
        source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()},
    )

    def publish():
        state.update(updated=time.time(), elapsed_seconds=time.time() - state["started"])
        _atomic_json(args.output / "run_state.json", state)

    publish()
    try:
        for condition in ("baseline", "camera_clearance"):
            state.update(condition=condition, phase_episodes_completed=0, phase_episodes_total=36)
            publish()
            jobs = [
                (
                    g * 9 + r,
                    str(args.checkpoint),
                    str(args.vision),
                    str(args.control),
                    str(args.keypoint),
                    str(args.output),
                    condition,
                )
                for g in range(97100000, 97100004)
                for r in range(9)
            ]
            results = []
            with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as pool:
                for future in as_completed([pool.submit(episode, j) for j in jobs]):
                    results.append(future.result())
                    state.update(
                        completed=state["completed"] + 1,
                        phase_episodes_completed=len(results),
                        partial=summarize(results),
                    )
                    publish()
            summary = summarize(results) | dict(
                condition=condition,
                block_out_of_bounds=sum(r["terminal_reason"] == "block_out_of_bounds" for r in results),
                pairs={str(r): summarize([v for v in results if v["seed"] % 9 == r]) for r in range(9)},
                projection_active=sum(r["projection_active"] for r in results),
            )
            _atomic_json(args.output / condition / "summary.json", dict(summary=summary, results=results))
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
