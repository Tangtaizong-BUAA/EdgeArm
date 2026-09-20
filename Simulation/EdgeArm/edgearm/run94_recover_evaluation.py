"""Finish the interrupted Run94 development evaluation without retraining.

Preserve completed episodes and original artifacts. Missing episodes execute the
unchanged development policy in a new directory; no independent seeds are used.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import time

from .constrained_recovery_run40 import summarize
from .run93_fast_observation import episode
from .train_staged_hybrid_contact_sac import _atomic_json


MODE = "memory_fast_keypoints"
SEEDS = tuple(g * 9 + r for g in range(97100000, 97100004) for r in range(9))


def completed_results(folder):
    results = []
    for seed in SEEDS:
        directory = folder / MODE / f"episode_{seed}"
        path = directory / "result.json"
        if not path.is_file():
            continue
        row = json.loads(path.read_text())
        if (
            row.get("seed") != seed
            or row.get("mode") != MODE
            or row.get("actor_uses_simulator_state") is not False
            or row.get("teacher_assisted") is not False
            or row.get("max_steps") != 900
            or not (directory / "trace.npz").is_file()
        ):
            raise ValueError("incomplete or mismatched preserved development episode")
        results.append(row)
    return results


def select_candidate(evaluations):
    def rank(row):
        eligible = row["hard_failures"] <= 1 and row["block_out_of_bounds"] == 0
        return eligible, row["successes"], -row["hard_failures"], row["mean_coverage"]

    if not evaluations or any(row["episodes"] != 36 for row in evaluations):
        raise ValueError("only complete 36-episode rounds may be selected")
    best = max(evaluations, key=rank)
    key = rank(best)
    return best["checkpoint"], list(key), bool(key[0] and key[1] >= 26)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "checkpoint", "vision", "keypoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 10), default=9)
    args = parser.parse_args()
    original_path = args.original / "run_state.json"
    original = json.loads(original_path.read_text())
    if (
        original.get("run") != "Run94"
        or original.get("round") != 3
        or original.get("step") != 3600
        or len(original.get("evaluations", [])) != 2
    ):
        raise ValueError("expected interrupted third evaluation after all 3600 updates")
    root = Path(__file__).parent
    for name, digest in original["source_hashes"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"original inference source changed: {name}")
    for name in ("checkpoint", "vision", "keypoint"):
        if (
            hashlib.sha256(getattr(args, name).read_bytes()).hexdigest()
            != original["checkpoint_hashes"][name]
        ):
            raise ValueError(f"original perception weight changed: {name}")
    control = args.original / "control_round_3.pt"
    control_hash = hashlib.sha256(control.read_bytes()).hexdigest()
    results = completed_results(args.original / "development_3")
    preserved = {row["seed"] for row in results}
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(original_path, args.output / "original_interrupted_run_state.json")
    shutil.copy2(Path(__file__), args.output / "recovery_driver.py")
    state = deepcopy(original)
    state.update(
        status="running",
        phase="resume_autonomous_development",
        started=time.time(),
        step=3600,
        recovery_of=str(args.original),
        training_resumed=False,
        no_gradient_updates=True,
        preserved_completed_episodes=len(results),
        newly_evaluated_episodes=0,
        selected_control_sha256_at_recovery=control_hash,
        original_interrupted_state_sha256=hashlib.sha256(original_path.read_bytes()).hexdigest(),
    )

    def publish():
        state.update(
            updated=time.time(),
            elapsed_seconds=time.time() - state["started"],
            phase_episodes_completed=len(results),
            phase_episodes_total=36,
            partial=summarize(results),
        )
        _atomic_json(args.output / "run_state.json", state)

    publish()
    try:
        jobs = [
            (
                seed,
                str(args.checkpoint),
                str(args.vision),
                str(control),
                str(args.keypoint),
                str(args.output / "episodes"),
                MODE,
            )
            for seed in SEEDS
            if seed not in preserved
        ]
        with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as pool:
            for future in as_completed([pool.submit(episode, job) for job in jobs]):
                results.append(future.result())
                state["newly_evaluated_episodes"] += 1
                publish()
        if sorted(row["seed"] for row in results) != sorted(SEEDS):
            raise ValueError("development cohort incomplete or duplicated")
        if hashlib.sha256(control.read_bytes()).hexdigest() != control_hash:
            raise ValueError("control weights changed during recovery")
        for name, digest in original["source_hashes"].items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
                raise ValueError("original inference source changed during recovery")
        summary = summarize(results) | dict(
            round=3,
            checkpoint=str(control),
            block_out_of_bounds=sum(r["terminal_reason"] == "block_out_of_bounds" for r in results),
            pairs={str(r): summarize([x for x in results if x["seed"] % 9 == r]) for r in range(9)},
            preserved_completed_episodes=len(preserved),
            newly_evaluated_episodes=len(jobs),
            resumed_after_instance_interruption=True,
        )
        _atomic_json(args.output / "summary.json", dict(summary=summary, results=results))
        state["evaluations"].append(summary)
        best, rank, passed = select_candidate(state["evaluations"])
        state.update(
            status="complete_pending_review",
            phase="finished",
            best_control=best,
            best_development_rank=rank,
            development_gate_passed=passed,
        )
        publish()
    except BaseException as exc:
        state.update(status="failed", error=repr(exc))
        publish()
        raise


if __name__ == "__main__":
    main()
