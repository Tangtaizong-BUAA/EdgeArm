"""Audit whether legacy V12 keyboard demos extend to the current 3 s gate.

This is a dynamics audit, not a new sensor capture.  Each successful legacy
trajectory is re-executed with the same seeded V12 plant and transport.  Only
the episode horizon and strict hold count change from six to ninety steps.
The original action prefix must reproduce every reported arm state exactly;
neutral commands are then appended until the object has remained contained
and settled for 90 consecutive 30 Hz transitions.

An extended pass remains one derived simulation re-execution of an existing
human episode.  It does not increment the count of independent human episodes
and does not claim that the old deferred RGB-D contains the appended frames.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

import mujoco
import numpy as np

from .phone_deferred_capture_v1 import PhoneMotionTrace, read_phone_motion_trace
from .sim2real_env_v12 import (
    MOUNTED_WRIST_CAMERA_DYNAMICS_PROFILE_V12,
    RealisticEdgeArmEnvV12,
    RealisticEnvV12Config,
)


KEYBOARD_STRICT_EXTENSION_AUDIT_FORMAT_V40 = (
    "edgearm-v40-keyboard-derived-90-step-extension-audit-v1"
)
STRICT_HOLD_STEPS_V40 = 90
STRICT_HOLD_SECONDS_V40 = 3.0


@dataclass(frozen=True, slots=True)
class KeyboardStrictExtensionEpisodeAuditV40:
    motion_path: str
    episode_seed: int
    source_rows: int
    extension_rows: int
    total_reexecuted_rows: int
    maximum_q_before_error_rad: float
    maximum_q_after_error_rad: float
    first_prefix_divergence_row: int | None
    maximum_contained_settled_run_steps: int
    terminal_reason: str
    strict_success: bool
    original_prefix_exact: bool
    derived_reexecution: bool = True
    unique_human_episode_increment: int = 0
    appended_sensor_frames_materialized: bool = False
    physical_samples: int = 0
    production_admission: bool = False
    format: str = KEYBOARD_STRICT_EXTENSION_AUDIT_FORMAT_V40


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _strict_flags(info: dict[str, Any]) -> tuple[bool, bool]:
    realism = info.get("realism_v6")
    if not isinstance(realism, dict):
        raise ValueError("V40 strict extension lost realism_v6 evidence")
    return bool(realism.get("strict_contained", False)), bool(
        realism.get("strict_settled", False)
    )


def _reset_from_trace(trace: PhoneMotionTrace, *, maximum_extension_steps: int) -> RealisticEdgeArmEnvV12:
    if trace.environment_profile != MOUNTED_WRIST_CAMERA_DYNAMICS_PROFILE_V12:
        raise ValueError("V40 strict extension accepts V12 mounted-wrist traces only")
    if trace.control_source != "keyboard":
        raise ValueError("V40 strict extension source must be keyboard controlled")
    if not trace.episode_success or trace.termination_reason != "strict_success":
        raise ValueError("V40 strict extension requires one legacy successful trace")
    if trace.task_metadata is None:
        raise ValueError("V40 strict extension requires language task metadata")
    if not trace.cycles:
        raise ValueError("V40 strict extension source is empty")
    config = RealisticEnvV12Config(
        max_steps=len(trace.cycles) + maximum_extension_steps,
        strict_success_hold_steps=STRICT_HOLD_STEPS_V40,
    )
    env = RealisticEdgeArmEnvV12(config, seed=trace.episode_seed)
    env.reset(seed=trace.episode_seed, obstacle=False)
    if trace.initial_arm_pose_overridden:
        if (
            trace.initial_arm_qpos_rad is None
            or trace.initial_arm_qvel_rad_s is None
            or trace.initial_arm_ctrl_rad is None
        ):
            raise ValueError("V40 overridden trace omitted initial plant state")
        env.data.qpos[:6] = trace.initial_arm_qpos_rad
        env.data.qvel[:6] = trace.initial_arm_qvel_rad_s
        env.data.ctrl[:6] = trace.initial_arm_ctrl_rad
        mujoco.mj_forward(env.model, env.data)
    return env


def audit_keyboard_strict_extension_episode_v40(
    motion_path: Path,
    *,
    maximum_extension_steps: int = 180,
    exact_state_tolerance_rad: float = 1.0e-12,
) -> KeyboardStrictExtensionEpisodeAuditV40:
    if type(maximum_extension_steps) is not int or maximum_extension_steps < STRICT_HOLD_STEPS_V40:
        raise ValueError("V40 maximum extension must cover the 90-step gate")
    if (
        not np.isfinite(exact_state_tolerance_rad)
        or exact_state_tolerance_rad < 0.0
        or exact_state_tolerance_rad > 2.0e-5
    ):
        raise ValueError("V40 prefix state tolerance is invalid")
    path = Path(motion_path).expanduser().resolve()
    trace = read_phone_motion_trace(path)
    env = _reset_from_trace(trace, maximum_extension_steps=maximum_extension_steps)
    maximum_before = 0.0
    maximum_after = 0.0
    first_divergence: int | None = None
    contained_settled_run = 0
    maximum_run = 0
    terminal_reason = "not_terminated"
    strict_success = False
    total_rows = 0
    info: dict[str, Any] = {}

    for row, cycle in enumerate(trace.cycles):
        current = np.asarray(env.observation()["joint_state"], dtype=np.float64)
        before_error = float(
            np.max(np.abs(current[:6] - cycle.step_result.q_before_rad))
        )
        maximum_before = max(maximum_before, before_error)
        observation, _reward, terminated, truncated, info = env.step(
            cycle.step_result.submitted_normalized_action
        )
        total_rows += 1
        after_error = float(
            np.max(
                np.abs(
                    np.asarray(observation["joint_state"], dtype=np.float64)[:6]
                    - cycle.step_result.q_after_rad
                )
            )
        )
        maximum_after = max(maximum_after, after_error)
        if first_divergence is None and max(before_error, after_error) > exact_state_tolerance_rad:
            first_divergence = row
        contained, settled = _strict_flags(info)
        contained_settled_run = contained_settled_run + 1 if contained and settled else 0
        maximum_run = max(maximum_run, contained_settled_run)
        if terminated or truncated:
            terminal_reason = str(info.get("terminal_reason", "unknown"))
            strict_success = bool(info.get("success", False))
            break

    if total_rows != len(trace.cycles):
        raise RuntimeError("V40 extended environment terminated inside the legacy action prefix")
    if first_divergence is not None:
        return KeyboardStrictExtensionEpisodeAuditV40(
            motion_path=str(path),
            episode_seed=trace.episode_seed,
            source_rows=len(trace.cycles),
            extension_rows=0,
            total_reexecuted_rows=total_rows,
            maximum_q_before_error_rad=maximum_before,
            maximum_q_after_error_rad=maximum_after,
            first_prefix_divergence_row=first_divergence,
            maximum_contained_settled_run_steps=maximum_run,
            terminal_reason="prefix_state_divergence",
            strict_success=False,
            original_prefix_exact=False,
        )

    neutral = np.zeros(6, dtype=np.float64)
    extension_rows = 0
    for _ in range(maximum_extension_steps):
        _observation, _reward, terminated, truncated, info = env.step(neutral)
        total_rows += 1
        extension_rows += 1
        contained, settled = _strict_flags(info)
        contained_settled_run = contained_settled_run + 1 if contained and settled else 0
        maximum_run = max(maximum_run, contained_settled_run)
        if terminated or truncated:
            terminal_reason = str(info.get("terminal_reason", "unknown"))
            strict_success = bool(info.get("success", False))
            break
    qualified = bool(
        strict_success
        and terminal_reason == "strict_success"
        and maximum_run >= STRICT_HOLD_STEPS_V40
    )
    return KeyboardStrictExtensionEpisodeAuditV40(
        motion_path=str(path),
        episode_seed=trace.episode_seed,
        source_rows=len(trace.cycles),
        extension_rows=extension_rows,
        total_reexecuted_rows=total_rows,
        maximum_q_before_error_rad=maximum_before,
        maximum_q_after_error_rad=maximum_after,
        first_prefix_divergence_row=None,
        maximum_contained_settled_run_steps=maximum_run,
        terminal_reason=terminal_reason,
        strict_success=qualified,
        original_prefix_exact=True,
    )


def audit_keyboard_strict_extension_batch_v40(
    motion_paths: Sequence[Path],
    *,
    output_path: Path | None = None,
    maximum_extension_steps: int = 180,
    exact_state_tolerance_rad: float = 1.0e-12,
) -> dict[str, Any]:
    paths = tuple(Path(path).expanduser().resolve() for path in motion_paths)
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("V40 strict extension batch requires unique input paths")
    episodes = [
        audit_keyboard_strict_extension_episode_v40(
            path,
            maximum_extension_steps=maximum_extension_steps,
            exact_state_tolerance_rad=exact_state_tolerance_rad,
        )
        for path in paths
    ]
    result = {
        "format": KEYBOARD_STRICT_EXTENSION_AUDIT_FORMAT_V40,
        "status": "complete",
        "episode_count": len(episodes),
        "original_prefix_exact_count": sum(item.original_prefix_exact for item in episodes),
        "strict_extension_success_count": sum(item.strict_success for item in episodes),
        "strict_extension_success_rate": sum(item.strict_success for item in episodes)
        / len(episodes),
        "required_hold_steps": STRICT_HOLD_STEPS_V40,
        "required_hold_seconds": STRICT_HOLD_SECONDS_V40,
        "derived_reexecution_count": len(episodes),
        "unique_human_episode_increment": 0,
        "appended_sensor_frames_materialized": False,
        "episodes": [asdict(item) for item in episodes],
        "physical_samples": 0,
        "production_admission": False,
    }
    if output_path is not None:
        _atomic_json(Path(output_path).expanduser().resolve(), result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-extension-steps", type=int, default=180)
    parser.add_argument("--exact-state-tolerance-rad", type=float, default=1.0e-12)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = audit_keyboard_strict_extension_batch_v40(
        args.motion,
        output_path=args.output,
        maximum_extension_steps=args.maximum_extension_steps,
        exact_state_tolerance_rad=args.exact_state_tolerance_rad,
    )
    print(
        json.dumps(
            {
                key: value for key, value in result.items() if key != "episodes"
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "KEYBOARD_STRICT_EXTENSION_AUDIT_FORMAT_V40",
    "KeyboardStrictExtensionEpisodeAuditV40",
    "audit_keyboard_strict_extension_batch_v40",
    "audit_keyboard_strict_extension_episode_v40",
]
