"""Materialize exact wrist RGB-D for a legacy keyboard trajectory extended to 3 s.

The legacy six-step action prefix is re-executed with the same seeded V12 plant
and transport and must reproduce every reported arm state exactly.  The only
environment contract changes are a longer horizon and a 90-step success hold.
Neutral commands are appended until strict success, while wrist RGB, metric
depth, camera pose, state/action history, language, and command-chain admission
are captured live from that derived simulation execution.

The result is ``sim_human`` data, not scratch RL and not a new independent
human episode.  The programmed hold tail is explicit provenance metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Sequence

import h5py
import mujoco
import numpy as np

from .audit_keyboard_strict_extension_v40 import STRICT_HOLD_STEPS_V40
from .multimodal import CameraCaptureConfig, TrueMultimodalRenderer
from .phone_deferred_capture_v1 import read_phone_motion_trace
from .phone_task_language_v1 import PhoneTaskMetadata
from .phone_teleop_dataset_v1 import HDF5PhoneSimWriter, SimWristObservation
from .phone_teleop_runtime_v1 import (
    PhoneControlCycle,
    PhoneSafetyDecision,
    SimStepResult,
    StopReason,
)
from .sim2real_env_v12 import (
    MOUNTED_WRIST_CAMERA_DYNAMICS_PROFILE_V12,
    RealisticEdgeArmEnvV12,
    RealisticEnvV12Config,
)
from .trisource_contract_v26 import SIM_HUMAN_SOURCE_V26


STRICT_KEYBOARD_RGBD_FORMAT_V40 = "edgearm-v40-derived-keyboard-strict-rgbd-v1"
STRICT_KEYBOARD_RUNTIME_V40 = "edgearm-keyboard-derived-90-step-hold-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _camera_pose(env: RealisticEdgeArmEnvV12) -> np.ndarray:
    camera_id = int(
        mujoco.mj_name2id(
            env.model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            "edgearm_wrist",
        )
    )
    if camera_id < 0:
        raise ValueError("V40 V12 scene has no wrist camera")
    mujoco.mj_forward(env.model, env.data)
    return np.concatenate(
        (env.data.cam_xpos[camera_id], env.data.cam_xmat[camera_id])
    ).astype(np.float32)


def _submission_unchanged(info: dict[str, Any], raw_action: np.ndarray) -> tuple[bool, np.ndarray]:
    transport = info.get("sim2real_v2")
    if not isinstance(transport, dict):
        raise ValueError("V40 strict RGB-D lost sim2real_v2 command evidence")
    submitted = np.asarray(transport.get("submitted_action"), dtype=np.float64)
    applied = np.asarray(
        transport.get("actually_applied_delayed_action"),
        dtype=np.float64,
    )
    if submitted.shape != (6,) or applied.shape != (6,):
        raise ValueError("V40 strict RGB-D command chain is malformed")
    ingress_lost = bool(transport.get("submitted_command_ingress_lost", False))
    if not ingress_lost and not np.allclose(submitted, raw_action, rtol=0.0, atol=2.0e-7):
        raise RuntimeError("V40 raw/submitted action changed during strict materialization")
    unchanged = bool(
        not ingress_lost
        and not any(bool(value) for value in transport.get("submitted_action_changed_mask", ()))
        and not any(
            bool(value)
            for value in transport.get("submitted_command_target_changed_mask", ())
        )
    )
    return unchanged, applied.astype(np.float32)


def _strict_flags(info: dict[str, Any]) -> tuple[bool, bool]:
    realism = info.get("realism_v6")
    if not isinstance(realism, dict):
        raise ValueError("V40 strict RGB-D lost realism_v6 evidence")
    return bool(realism.get("strict_contained", False)), bool(
        realism.get("strict_settled", False)
    )


def _create_policy_layout_v40(
    path: Path,
    *,
    parent_motion_path: Path,
    parent_motion_sha256: str,
    episode_seed: int,
    task: PhoneTaskMetadata,
    poses: np.ndarray,
    intrinsics: np.ndarray,
    previous_actions: np.ndarray,
    submission_unchanged: np.ndarray,
    strict_contained: np.ndarray,
    strict_settled: np.ndarray,
    strict_streak: np.ndarray,
    prefix_rows: int,
    extension_rows: int,
    maximum_q_before_error: float,
    maximum_q_after_error: float,
    action_history_preclip_max_abs: float,
    action_history_clipped_value_count: int,
) -> None:
    with h5py.File(path, "r+") as stream:
        steps = stream["steps"]
        rows = int(stream.attrs["committed_rows"])
        if any(
            value.shape[0] != rows
            for value in (
                poses,
                previous_actions,
                submission_unchanged,
                strict_contained,
                strict_settled,
                strict_streak,
            )
        ):
            raise RuntimeError("V40 strict materialization arrays are not row aligned")
        if intrinsics.shape != (3, 3):
            raise ValueError("V40 wrist intrinsics must be 3x3")
        success_trace = strict_streak >= STRICT_HOLD_STEPS_V40
        if not bool(np.any(success_trace)) or not bool(success_trace[-1]):
            raise RuntimeError("V40 strict materialization lacks terminal 90-step evidence")
        depth = np.asarray(steps["depth_wrist_mm"], dtype=np.uint16)
        depth_valid = depth > 0
        reconstructable = np.mean(depth_valid, axis=(1, 2)) >= 0.05
        q = np.asarray(steps["q_before_rad"], dtype=np.float32)
        dq = np.asarray(steps["dq_before_rad_s"], dtype=np.float32)
        joint_state = np.concatenate((q, dq), axis=1).astype(np.float32)
        action = np.asarray(steps["submitted_normalized_action"], dtype=np.float32)
        if np.any(np.abs(action) > 1.0 + 1.0e-6):
            raise RuntimeError("V40 strict materialization action escaped normalized bounds")
        action_admission = submission_unchanged.copy()

        observation = stream.create_group("observation")
        observation["rgb_wrist"] = steps["wrist_rgb"]
        observation["depth_wrist_mm"] = steps["depth_wrist_mm"]
        observation.create_dataset(
            "depth_valid_mask_wrist",
            data=depth_valid,
            compression="lzf",
            shuffle=True,
            chunks=(1, *depth_valid.shape[1:]),
        )
        observation.create_dataset("camera_pose_wrist", data=poses.astype(np.float32))
        observation.create_dataset(
            "camera_intrinsics_wrist",
            data=np.repeat(intrinsics[None], rows, axis=0).astype(np.float32),
        )
        observation.create_dataset("camera_4d_reconstructable_mask", data=reconstructable)
        observation.create_dataset(
            "camera_geometry_alignment_exact",
            data=np.ones(rows, dtype=bool),
        )
        observation.create_dataset("joint_state", data=joint_state)
        observation.create_dataset("previous_executed_action", data=previous_actions)
        observation["previous_executed_action"].attrs["semantics"] = (
            "previous reported q_after-minus-q_before normalized by max_joint_delta"
        )

        action_group = stream.create_group("action")
        action_ds = action_group.create_dataset("submitted_joint_action", data=action)
        action_ds.attrs.update(
            {
                "policy_target_semantics": (
                    "current post-task-guard, pre-transport normalized command"
                ),
                "action_supervision_requires_current_submission_safety_unchanged": True,
            }
        )
        timing = stream.create_group("timing")
        clock = np.arange(rows, dtype=np.float64) / 30.0
        timing.create_dataset("camera_device_time_seconds", data=clock)
        timing.create_dataset("camera_host_receive_time_seconds", data=clock)
        timing.attrs["clock_semantics"] = "deterministic_simulation_control_clock_30hz"

        string_dtype = h5py.string_dtype("utf-8")
        language = stream.create_group("language")
        language.create_dataset(
            "row_instruction_en",
            data=np.asarray([task.task_text_en] * rows, dtype=object),
            dtype=string_dtype,
        )
        language.create_dataset(
            "row_instruction_zh",
            data=np.asarray([task.task_text_zh] * rows, dtype=object),
            dtype=string_dtype,
        )
        index = stream.create_group("index")
        index.create_dataset("episode_id", data=np.zeros(rows, dtype=np.int64))
        index.create_dataset(
            "source_episode_id",
            data=np.full(rows, episode_seed, dtype=np.int64),
        )
        index.create_dataset("episode_step_id", data=np.arange(rows, dtype=np.int64))

        admission = stream.create_group("admission")
        admission.create_dataset("act_action_supervision_mask", data=action_admission)
        admission.create_dataset("strict_success_trace", data=success_trace)
        admission.create_dataset("strict_success_streak_steps", data=strict_streak)
        admission.attrs["strict_success_trace_audit_json"] = json.dumps(
            {
                "all_recurrences_exact": True,
                "exact_three_second_evidence": True,
                "required_hold_steps": STRICT_HOLD_STEPS_V40,
                "maximum_streak_steps": int(np.max(strict_streak)),
                "terminal_success_trace": bool(success_trace[-1]),
                "source_prefix_exact": True,
                "programmed_hold_tail": True,
                "production_admission": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        outcome = stream.create_group("outcome")
        outcome.create_dataset("execution_attempted", data=np.ones(rows, dtype=bool))
        outcome.create_dataset(
            "current_submission_safety_unchanged",
            data=submission_unchanged,
        )
        outcome.create_dataset("strict_contained", data=strict_contained)
        outcome.create_dataset("strict_settled", data=strict_settled)
        outcome.create_dataset(
            "programmed_hold_tail",
            data=np.arange(rows, dtype=np.int64) >= prefix_rows,
        )

        stream.attrs.update(
            {
                "format": STRICT_KEYBOARD_RGBD_FORMAT_V40,
                "source_type": SIM_HUMAN_SOURCE_V26,
                "derived_reexecution": True,
                "parent_live_episode_path": str(parent_motion_path),
                "parent_live_episode_sha256": parent_motion_sha256,
                "unique_live_episode_increment": 0,
                "operator_action_prefix_rows": prefix_rows,
                "programmed_hold_tail_rows": extension_rows,
                "programmed_hold_tail": True,
                "strict_success_hold_steps": STRICT_HOLD_STEPS_V40,
                "strict_success_hold_seconds": 3.0,
                "strict_success_trace_verified": True,
                "camera_geometry_alignment_exact": True,
                "camera_4d_reconstructable": bool(np.all(reconstructable)),
                "camera_pose_source": "live_pre_action_mujoco_wrist_pose",
                "all_state_replays_exact": True,
                "original_action_prefix_q_before_max_error_rad": maximum_q_before_error,
                "original_action_prefix_q_after_max_error_rad": maximum_q_after_error,
                "action_history_semantics": (
                    "previous reported q_after-minus-q_before normalized by max_joint_delta, "
                    "then saturated to [-1,1] as a history feature"
                ),
                "action_history_preclip_max_abs": action_history_preclip_max_abs,
                "action_history_clipped_value_count": action_history_clipped_value_count,
                "segmentation_is_policy_input": False,
                "expert_calls": 0,
                "behavior_cloning_steps_used_to_generate_actions": 0,
                "physical_samples": 0,
                "production_admission": False,
            }
        )
        stream.flush()


def materialize_keyboard_strict_rgbd_v40(
    motion_path: Path,
    output_path: Path,
    *,
    width: int = 224,
    height: int = 168,
    maximum_extension_steps: int = 180,
    exact_state_tolerance_rad: float = 1.0e-12,
) -> dict[str, Any]:
    motion = Path(motion_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    if type(width) is not int or type(height) is not int or min(width, height) < 32:
        raise ValueError("V40 strict RGB-D dimensions are invalid")
    if type(maximum_extension_steps) is not int or maximum_extension_steps < 90:
        raise ValueError("V40 strict RGB-D extension budget is too short")
    if not 0.0 <= exact_state_tolerance_rad <= 2.0e-5:
        raise ValueError("V40 strict RGB-D state tolerance is invalid")
    trace = read_phone_motion_trace(motion)
    if (
        trace.environment_profile != MOUNTED_WRIST_CAMERA_DYNAMICS_PROFILE_V12
        or trace.control_source != "keyboard"
        or not trace.episode_success
        or trace.termination_reason != "strict_success"
        or not isinstance(trace.task_metadata, PhoneTaskMetadata)
        or trace.task_metadata.obstacle_enabled
    ):
        raise ValueError("V40 strict RGB-D requires one successful obstacle-free V12 keyboard trace")

    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        raise FileExistsError(partial)
    output.parent.mkdir(parents=True, exist_ok=True)
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
            raise ValueError("V40 strict RGB-D source omitted overridden initial state")
        env.data.qpos[:6] = trace.initial_arm_qpos_rad
        env.data.qvel[:6] = trace.initial_arm_qvel_rad_s
        env.data.ctrl[:6] = trace.initial_arm_ctrl_rad
        mujoco.mj_forward(env.model, env.data)

    renderer = TrueMultimodalRenderer(
        env,
        CameraCaptureConfig(
            width=width,
            height=height,
            cameras=("wrist",),
        ),
    )
    renderer.begin_episode(trace.episode_seed)
    calibration = renderer.calibration_metadata()
    intrinsics = np.asarray(calibration["wrist"]["intrinsics"], dtype=np.float32)
    parent_hash = _sha256_file(motion)
    poses: list[np.ndarray] = []
    previous_actions: list[np.ndarray] = []
    unchanged_rows: list[bool] = []
    contained_rows: list[bool] = []
    settled_rows: list[bool] = []
    streak_rows: list[int] = []
    previous_executed = np.zeros(6, dtype=np.float32)
    strict_run = 0
    maximum_before = 0.0
    maximum_after = 0.0
    extension_rows = 0
    action_history_preclip_max_abs = 0.0
    action_history_clipped_value_count = 0
    success = False
    try:
        with HDF5PhoneSimWriter(
            partial,
            image_shape=(height, width, 3),
            episode_seed=trace.episode_seed,
            environment_profile=env.profile_version,
            camera_metadata=calibration,
            task_metadata=trace.task_metadata,
            control_source="keyboard",
            control_runtime_version=STRICT_KEYBOARD_RUNTIME_V40,
            overwrite=False,
        ) as writer:
            for row in range(len(trace.cycles) + maximum_extension_steps):
                prefix = row < len(trace.cycles)
                original = trace.cycles[row] if prefix else None
                raw_action = (
                    np.asarray(
                        original.step_result.submitted_normalized_action,
                        dtype=np.float64,
                    )
                    if original is not None
                    else np.zeros(6, dtype=np.float64)
                )
                current = np.asarray(env.observation()["joint_state"], dtype=np.float64)
                if original is not None:
                    before_error = float(
                        np.max(np.abs(current[:6] - original.step_result.q_before_rad))
                    )
                    maximum_before = max(maximum_before, before_error)
                    if before_error > exact_state_tolerance_rad:
                        raise RuntimeError(
                            f"V40 strict RGB-D q_before diverged at prefix row {row}: "
                            f"{before_error:.9g}"
                        )
                pose = _camera_pose(env)
                capture_ns = time.monotonic_ns()
                capture = renderer.capture()
                queued_ns = time.monotonic_ns()
                observation, reward, terminated, truncated, info = env.step(raw_action)
                completed_ns = time.monotonic_ns()
                after = np.asarray(observation["joint_state"], dtype=np.float64)
                if original is not None:
                    after_error = float(
                        np.max(np.abs(after[:6] - original.step_result.q_after_rad))
                    )
                    maximum_after = max(maximum_after, after_error)
                    if after_error > exact_state_tolerance_rad:
                        raise RuntimeError(
                            f"V40 strict RGB-D q_after diverged at prefix row {row}: "
                            f"{after_error:.9g}"
                        )
                unchanged, _applied_target_delta = _submission_unchanged(info, raw_action)
                executed = (after[:6] - current[:6]) / float(env.config.max_joint_delta)
                if not np.all(np.isfinite(executed)):
                    raise RuntimeError(
                        f"V40 strict RGB-D reported executed action is non-finite at row {row}"
                    )
                row_maximum = float(np.max(np.abs(executed), initial=0.0))
                action_history_preclip_max_abs = max(
                    action_history_preclip_max_abs,
                    row_maximum,
                )
                action_history_clipped_value_count += int(
                    np.count_nonzero(np.abs(executed) > 1.0)
                )
                if row_maximum > 1.5:
                    raise RuntimeError(
                        f"V40 strict RGB-D reported joint motion is implausible at row {row}"
                    )
                executed = np.clip(executed, -1.0, 1.0).astype(np.float32)
                contained, settled = _strict_flags(info)
                strict_run = strict_run + 1 if contained and settled else 0
                poses.append(pose)
                previous_actions.append(previous_executed.copy())
                unchanged_rows.append(unchanged)
                contained_rows.append(contained)
                settled_rows.append(settled)
                streak_rows.append(strict_run)
                previous_executed = executed
                target_q = (
                    np.asarray(original.step_result.target_q_rad, dtype=np.float64)
                    if original is not None
                    else current[:6].copy()
                )
                queued_q = (
                    np.asarray(original.step_result.queued_target_q_rad, dtype=np.float64)
                    if original is not None
                    else current[:6].copy()
                )
                step = SimStepResult(
                    q_before_rad=current[:6],
                    dq_before_rad_s=current[6:],
                    target_q_rad=target_q,
                    queued_target_q_rad=queued_q,
                    submitted_normalized_action=raw_action,
                    q_after_rad=after[:6],
                    dq_after_rad_s=after[6:],
                    reward=float(reward),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    info=info,
                    command_queued_monotonic_ns=queued_ns,
                    completed_monotonic_ns=completed_ns,
                )
                cycle = PhoneControlCycle(
                    sample=(original.sample if original is not None else None),
                    decision=(
                        original.decision
                        if original is not None
                        else PhoneSafetyDecision(False, StopReason.NO_SAMPLE, 0, False)
                    ),
                    cartesian_target=(
                        original.cartesian_target if original is not None else None
                    ),
                    ik_result=(original.ik_result if original is not None else None),
                    step_result=step,
                )
                writer.append(
                    SimWristObservation(
                        wrist_rgb=capture["rgb_wrist"],
                        depth_wrist_mm=capture["depth_wrist_mm"],
                        segmentation_wrist=capture["segmentation_wrist"],
                        capture_monotonic_ns=capture_ns,
                    ),
                    cycle,
                )
                if not prefix:
                    extension_rows += 1
                if terminated or truncated:
                    if prefix:
                        raise RuntimeError(
                            "V40 strict RGB-D environment terminated inside the legacy prefix"
                        )
                    success = bool(
                        terminated
                        and info.get("success", False)
                        and str(info.get("terminal_reason", "")) == "strict_success"
                        and strict_run >= STRICT_HOLD_STEPS_V40
                    )
                    break
            if not success:
                raise RuntimeError("V40 strict RGB-D extension did not reach the 90-step gate")
            writer.finalize(
                success=True,
                termination_reason="strict_success_stable_3s",
                replay_audit={
                    "requested_state_match_mode": "exact_original_action_prefix",
                    "sensor_materialization_mode": "human_command_sequence_reexecution",
                    "state_exact": True,
                    "joint_tolerance_rad": exact_state_tolerance_rad,
                    "first_state_divergence_row": None,
                    "maximum_q_before_error_rad": maximum_before,
                    "maximum_q_after_error_rad": maximum_after,
                    "human_command_sequence_unchanged": True,
                    "programmed_neutral_hold_tail": True,
                    "supports_exact_dynamic_resume": False,
                    "production_admission": False,
                },
            )
    finally:
        renderer.close()

    _create_policy_layout_v40(
        partial,
        parent_motion_path=motion,
        parent_motion_sha256=parent_hash,
        episode_seed=trace.episode_seed,
        task=trace.task_metadata,
        poses=np.asarray(poses, dtype=np.float32),
        intrinsics=intrinsics,
        previous_actions=np.asarray(previous_actions, dtype=np.float32),
        submission_unchanged=np.asarray(unchanged_rows, dtype=bool),
        strict_contained=np.asarray(contained_rows, dtype=bool),
        strict_settled=np.asarray(settled_rows, dtype=bool),
        strict_streak=np.asarray(streak_rows, dtype=np.int64),
        prefix_rows=len(trace.cycles),
        extension_rows=extension_rows,
        maximum_q_before_error=maximum_before,
        maximum_q_after_error=maximum_after,
        action_history_preclip_max_abs=action_history_preclip_max_abs,
        action_history_clipped_value_count=action_history_clipped_value_count,
    )
    os.replace(partial, output)
    summary = {
        "format": STRICT_KEYBOARD_RGBD_FORMAT_V40,
        "status": "complete",
        "output_path": str(output),
        "output_sha256": _sha256_file(output),
        "parent_motion_path": str(motion),
        "parent_motion_sha256": parent_hash,
        "episode_seed": trace.episode_seed,
        "operator_action_prefix_rows": len(trace.cycles),
        "programmed_hold_tail_rows": extension_rows,
        "total_rows": len(poses),
        "maximum_q_before_error_rad": maximum_before,
        "maximum_q_after_error_rad": maximum_after,
        "action_history_preclip_max_abs": action_history_preclip_max_abs,
        "action_history_clipped_value_count": action_history_clipped_value_count,
        "strict_success_hold_steps": STRICT_HOLD_STEPS_V40,
        "strict_success_hold_seconds": 3.0,
        "source_type": SIM_HUMAN_SOURCE_V26,
        "derived_reexecution": True,
        "unique_live_episode_increment": 0,
        "simulated_wrist_rgbd": True,
        "camera_geometry_alignment_exact": True,
        "physical_samples": 0,
        "production_admission": False,
    }
    _atomic_json(output.with_suffix(output.suffix + ".summary.json"), summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=168)
    parser.add_argument("--maximum-extension-steps", type=int, default=180)
    parser.add_argument("--exact-state-tolerance-rad", type=float, default=1.0e-12)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = materialize_keyboard_strict_rgbd_v40(
        args.motion,
        args.output,
        width=args.width,
        height=args.height,
        maximum_extension_steps=args.maximum_extension_steps,
        exact_state_tolerance_rad=args.exact_state_tolerance_rad,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "STRICT_KEYBOARD_RGBD_FORMAT_V40",
    "STRICT_KEYBOARD_RUNTIME_V40",
    "materialize_keyboard_strict_rgbd_v40",
]
