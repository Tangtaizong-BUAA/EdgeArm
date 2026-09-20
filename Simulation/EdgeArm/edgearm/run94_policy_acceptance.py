"""One-shot independent evaluation of the spatial-map/preserved-action policy."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import time

from .constrained_recovery_run40 import summarize
from .run94_frozen_fast_policy import episode
from .train_staged_hybrid_contact_sac import _atomic_json
from .visual_policy_acceptance import claim_groups, report


def development_gate(state, hashes, control_path):
    if state.get("run") != "Run94" or state.get("status") != "complete_pending_review":
        raise ValueError("complete Run94 development required")
    for name in ("checkpoint", "vision", "keypoint"):
        if state.get("checkpoint_hashes", {}).get(name) != hashes[name]:
            raise ValueError("perception weights differ from development")
    control_path = Path(control_path).resolve()
    if hashlib.sha256(control_path.read_bytes()).hexdigest() != hashes["control"]:
        raise ValueError("selected control checksum mismatch")
    matches = [
        r for r in state.get("evaluations", [])
        if Path(r.get("checkpoint", "")).resolve() == control_path
    ]
    if not any(
        r["episodes"] == 36
        and r["successes"] >= 26
        and r["hard_failures"] <= 1
        and r.get("block_out_of_bounds", -1) == 0
        for r in matches
    ):
        raise ValueError("predeclared 26/36 development gate not passed by selected control")


def fingerprints(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*.py"))
        if "__pycache__" not in p.parts
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "vision", "control", "keypoint", "development-state", "registry", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--mode", choices=("memory_fast_keypoints",), default="memory_fast_keypoints")
    p.add_argument("--group-start", type=int, default=98000100)
    p.add_argument("--groups", type=int, choices=range(8, 17), default=8)
    p.add_argument("--workers", type=int, choices=range(1, 10), default=9)
    a = p.parse_args()
    groups = list(range(a.group_start, a.group_start + a.groups))
    if not all(98000000 <= g < 99000000 for g in groups):
        raise ValueError("independent seed groups required")
    hashes = {
        k: hashlib.sha256(getattr(a, k).read_bytes()).hexdigest()
        for k in ("checkpoint", "vision", "control", "keypoint")
    }
    development = json.loads(a.development_state.read_text())
    development_gate(development, hashes, a.control)
    for name, digest in development["source_hashes"].items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != digest:
            raise ValueError("development inference source was changed")
    a.output.mkdir(parents=True, exist_ok=False)
    frozen = {}
    for name in ("checkpoint", "vision", "control", "keypoint"):
        path = getattr(a, name)
        dest = a.output / (name + ".pt")
        shutil.copy2(path, dest)
        frozen[name] = dict(
            source=str(path), frozen=str(dest), sha256=hashlib.sha256(dest.read_bytes()).hexdigest()
        )
    root = Path(__file__).parent
    before = fingerprints(root)
    for name in before:
        dest = a.output / "frozen_source" / "edgearm" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, dest)
    freeze = dict(
        checkpoints=frozen,
        source_hashes=before,
        groups=groups,
        mode=a.mode,
        independent_acceptance=True,
        no_training=True,
        target_rate=0.6,
        strict_target_comparison="greater_than",
        fixed_wrist_survey=True,
        fixed_survey_steps=220,
        scripted_completion=True,
        max_steps=900,
        current_teacher_action_input=False,
        actor_uses_simulator_state=False,
        nominal_height_prior=True,
        learned_depth=False,
        learned_keypoints=True,
        confidence_gated_completion=True,
        static_survey_memory=True,
        keypoint_update_stride_after_survey=1,
        latent_memory_update_stride=8,
        selected_control_development_source=str(a.control),
        control_hash_recorded_after_development=True,
    )
    _atomic_json(a.output / "candidate_freeze.json", freeze)
    claim_groups(a.registry, groups, a.output / "candidate_freeze.json")
    state = dict(
        run="Run94Independent",
        status="running",
        phase="independent_acceptance",
        started=time.time(),
        groups=groups,
        completed=0,
        total=9 * len(groups),
        target_rate=0.6,
        summary=None,
        independent_acceptance=True,
        original_ACT_checkpoint=False,
        actor_uses_simulator_state=False,
        production_admission=False,
        export_admission=False,
        final_vla_acceptance=False,
    )

    def publish():
        state.update(updated=time.time(), elapsed_seconds=time.time() - state["started"])
        _atomic_json(a.output / "run_state.json", state)

    publish()
    results = []
    try:
        jobs = [
            (
                g * 9 + r,
                frozen["checkpoint"]["frozen"],
                frozen["vision"]["frozen"],
                frozen["control"]["frozen"],
                frozen["keypoint"]["frozen"],
                str(a.output / "episodes"),
                a.mode,
                tuple(groups),
            )
            for g in groups
            for r in range(9)
        ]
        with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
            for future in as_completed([pool.submit(episode, job) for job in jobs]):
                result = future.result()
                result["independent_acceptance"] = True
                _atomic_json(
                    a.output / "episodes" / a.mode / f"episode_{result['seed']}" / "result.json", result
                )
                results.append(result)
                state.update(completed=len(results), partial=summarize(results))
                publish()
        if fingerprints(root) != before:
            raise ValueError("inference sources changed during frozen acceptance")
        summary = report(results, groups)
        summary.pop("statistical_lower_bound_at_least_80", None)
        summary.update(
            target_rate=0.6,
            target_point_estimate_met=summary["success_rate"] > 0.6,
            statistical_lower_bound_above_60=summary["wilson_95_episode_approximation"][0] > 0.6,
            learned_sparse_spatial_memory=True,
            scripted_completion=True,
            mode=a.mode,
        )
        _atomic_json(a.output / "summary.json", dict(summary=summary, results=results))
        state.update(
            status="complete_pending_review",
            phase="finished",
            summary=summary,
            target_point_estimate_met=summary["target_point_estimate_met"],
        )
        publish()
    except BaseException as exc:
        state.update(status="failed", error=repr(exc))
        publish()
        raise


if __name__ == "__main__":
    main()

