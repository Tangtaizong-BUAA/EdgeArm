"""Recoverably materialize strict 3-second wrist RGB-D for keyboard episodes.

The batch accepts successful obstacle-free V12 keyboard motion traces, writes
one immutable derived H5 per logical human episode, and updates an atomic
manifest after every episode.  Failed or ineligible traces are never admitted
to the output inventory.  Existing complete outputs may be reused only when
their parent hash and strict V40 contract still match.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

import h5py

from .materialize_keyboard_strict_rgbd_v40 import (
    STRICT_KEYBOARD_RGBD_FORMAT_V40,
    _sha256_file,
    materialize_keyboard_strict_rgbd_v40,
)
from .sim2real_env_v12 import MOUNTED_WRIST_CAMERA_DYNAMICS_PROFILE_V12


STRICT_KEYBOARD_RGBD_BATCH_FORMAT_V40 = (
    "edgearm-v40-derived-keyboard-strict-rgbd-batch-v1"
)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _eligible(path: Path) -> tuple[bool, int | None, str]:
    try:
        with h5py.File(path, "r", swmr=True) as stream:
            seed = int(stream.attrs["episode_seed"])
            environment_profile = str(stream.attrs.get("environment_profile", ""))
            control_source = str(stream.attrs.get("operator_input_source", ""))
            episode_success = bool(stream.attrs.get("episode_success", False))
            termination_reason = str(stream.attrs.get("termination_reason", ""))
            language_present = bool(
                stream.attrs.get("language_condition_present", False)
                and stream.attrs.get("task_contract_json", "")
            )
            obstacle = bool(stream.attrs.get("obstacle", True))
            rows = int(stream.attrs.get("rows", 0))
    except Exception as error:
        return False, None, f"unreadable:{type(error).__name__}:{error}"
    if environment_profile != MOUNTED_WRIST_CAMERA_DYNAMICS_PROFILE_V12:
        return False, seed, "not_v12_mounted_wrist"
    if control_source != "keyboard":
        return False, seed, "not_keyboard"
    if not episode_success or termination_reason != "strict_success":
        return False, seed, "legacy_episode_not_successful"
    if not language_present:
        return False, seed, "language_task_missing"
    if obstacle:
        return False, seed, "obstacle_enabled"
    if rows < 1:
        return False, seed, "empty_motion_trace"
    return True, seed, "eligible"


def _existing_summary(
    output: Path,
    *,
    parent_sha256: str,
) -> dict[str, Any] | None:
    if not output.is_file():
        return None
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    if not summary_path.is_file():
        raise ValueError(f"V40 existing strict H5 lacks summary: {output}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError(f"V40 existing strict summary is malformed: {summary_path}")
    with h5py.File(output, "r", swmr=True) as stream:
        valid = bool(
            str(stream.attrs.get("format", "")) == STRICT_KEYBOARD_RGBD_FORMAT_V40
            and bool(stream.attrs.get("finalized", False))
            and bool(stream.attrs.get("strict_success_trace_verified", False))
            and int(stream.attrs.get("strict_success_hold_steps", -1)) == 90
            and str(stream.attrs.get("parent_live_episode_sha256", ""))
            == parent_sha256
        )
    if not valid or summary.get("parent_motion_sha256") != parent_sha256:
        raise ValueError(f"V40 existing strict output does not match its parent: {output}")
    return summary


def materialize_keyboard_strict_rgbd_batch_v40(
    motion_paths: Sequence[Path],
    output_directory: Path,
    *,
    width: int = 224,
    height: int = 168,
    maximum_extension_steps: int = 180,
    exact_state_tolerance_rad: float = 1.0e-12,
    fail_fast: bool = False,
) -> dict[str, Any]:
    paths = tuple(sorted(Path(path).expanduser().resolve() for path in motion_paths))
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("V40 strict RGB-D batch requires unique input motion paths")
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "run_state.json"
    manifest_path = output / "manifest.json"

    eligible: list[tuple[Path, int]] = []
    excluded: list[dict[str, Any]] = []
    for path in paths:
        accepted, seed, reason = _eligible(path)
        if accepted and seed is not None:
            eligible.append((path, seed))
        else:
            excluded.append(
                {"motion_path": str(path), "episode_seed": seed, "reason": reason}
            )
    seeds = [seed for _path, seed in eligible]
    if len(set(seeds)) != len(seeds):
        raise ValueError("V40 strict RGB-D batch contains duplicate logical episode seeds")
    if not eligible:
        raise ValueError("V40 strict RGB-D batch found no eligible keyboard episodes")

    complete: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    reused = 0

    def snapshot(status: str, phase: str) -> dict[str, Any]:
        return {
            "format": STRICT_KEYBOARD_RGBD_BATCH_FORMAT_V40,
            "status": status,
            "phase": phase,
            "input_motion_count": len(paths),
            "eligible_logical_human_episode_count": len(eligible),
            "excluded_motion_count": len(excluded),
            "derived_artifact_complete_count": len(complete),
            "derived_artifact_failure_count": len(failures),
            "reused_complete_artifact_count": reused,
            "newly_acquired_live_human_episode_count": 0,
            "unique_human_episode_increment": 0,
            "appended_sensor_frames_materialized": bool(complete),
            "width": width,
            "height": height,
            "strict_success_hold_steps": 90,
            "strict_success_hold_seconds": 3.0,
            "episodes": complete,
            "failures": failures,
            "excluded": excluded,
            "physical_samples": 0,
            "production_admission": False,
        }

    _atomic_json(state_path, snapshot("running", "materializing"))
    for index, (motion, seed) in enumerate(eligible):
        stem = motion.name.removesuffix(".motion.h5")
        episode_output = output / f"{stem}.strict_rgbd.h5"
        parent_hash = _sha256_file(motion)
        try:
            existing = _existing_summary(
                episode_output,
                parent_sha256=parent_hash,
            )
            if existing is None:
                episode = materialize_keyboard_strict_rgbd_v40(
                    motion,
                    episode_output,
                    width=width,
                    height=height,
                    maximum_extension_steps=maximum_extension_steps,
                    exact_state_tolerance_rad=exact_state_tolerance_rad,
                )
                episode["reused_existing_artifact"] = False
            else:
                episode = dict(existing)
                episode["reused_existing_artifact"] = True
                reused += 1
            episode["batch_episode_index"] = index
            episode["episode_seed"] = seed
            complete.append(episode)
        except Exception as error:
            failure = {
                "batch_episode_index": index,
                "motion_path": str(motion),
                "episode_seed": seed,
                "error_type": type(error).__name__,
                "error": str(error),
            }
            failures.append(failure)
            _atomic_json(
                output / f"{stem}.failure.json",
                {"format": STRICT_KEYBOARD_RGBD_BATCH_FORMAT_V40, **failure},
            )
            if fail_fast:
                result = snapshot("failed", "failed")
                _atomic_json(manifest_path, result)
                _atomic_json(state_path, result)
                raise
        _atomic_json(manifest_path, snapshot("running", "materializing"))
        _atomic_json(state_path, snapshot("running", "materializing"))

    status = "complete" if not failures else "complete_with_failures"
    result = snapshot(status, status)
    _atomic_json(manifest_path, result)
    _atomic_json(state_path, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, action="append")
    parser.add_argument("--motion-directory", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=168)
    parser.add_argument("--maximum-extension-steps", type=int, default=180)
    parser.add_argument("--exact-state-tolerance-rad", type=float, default=1.0e-12)
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = list(args.motion or ())
    if args.motion_directory is not None:
        paths.extend(
            sorted(args.motion_directory.expanduser().resolve().glob("*.motion.h5"))
        )
    result = materialize_keyboard_strict_rgbd_batch_v40(
        paths,
        args.output_directory,
        width=args.width,
        height=args.height,
        maximum_extension_steps=args.maximum_extension_steps,
        exact_state_tolerance_rad=args.exact_state_tolerance_rad,
        fail_fast=args.fail_fast,
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key not in {"episodes", "excluded"}},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "STRICT_KEYBOARD_RGBD_BATCH_FORMAT_V40",
    "materialize_keyboard_strict_rgbd_batch_v40",
]
