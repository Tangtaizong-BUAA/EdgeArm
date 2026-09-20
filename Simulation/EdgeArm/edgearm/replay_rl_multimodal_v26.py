"""Deterministically replay V26 scratch-RL actions into multiview RGB-D VLA data.

The online PPO rollout stays intentionally small (four 32x32 RGB views).  This
separate pass reconstructs each exact reset, replays the recorded six-joint
command submitted to the plant, verifies every pre-action simulator state,
and captures a production-shaped wrist stream plus lower-resolution simulated
teacher views.  It never turns failed RL
episodes into imitation labels: only strict three-second, contact-bearing,
safe, exactly replayed episodes receive ACT/VLA action-supervision masks.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence

import h5py
import mujoco
import numpy as np

from .asymmetric_multiview_ppo_v1 import (
    MultiViewRGBRendererV1,
    canonical_sha256_v1,
    sha256_file_v1,
)
from .dynamic_language_v26 import verify_dynamic_language_v26
from .multimodal import CameraCaptureConfig, TrueMultimodalRenderer
from .sim2real_env_v10 import RealisticEdgeArmEnvV10
from .stock_gripper_reward_v22 import StockGripperPotentialRewardV22
from .stock_gripper_rollout_kernel_v22 import StockGripperRolloutKernelV22
from .stock_gripper_taskframe_v22 import StockGripperTaskFrameAdapterV22
from .train_feasible_multiview_ppo_v22 import (
    _derive_exact_configs,
    verify_shared_collection_sources_v25,
    verify_v26_collection_contracts,
)
from .v22_rollout_h5 import (
    load_feasible_multiview_rollout_v22,
    load_verified_run_plan_v22,
)
from .v23_checkpoint_lineage import load_collection_parent_actor_critic_v23
from .trajectory_contract_v1 import METRIC_DEPTH_CONTRACT_FORMAT


RL_MULTIMODAL_REPLAY_FORMAT_V26 = "edgearm-v26-scratch-rl-wrist-rgbd-replay-v1"
RL_MULTIMODAL_REPLAY_CAUSAL_ORDER_V26 = (
    "pre_action_wrist_rgb_depth_segmentation_pose_and_reported_joint_state_then_"
    "submitted_joint_action_then_next_state"
)
AQ16_REFERENCE_V26 = {
    "model": "AQ16",
    "sensor": "1/2.8-inch IMX298-A",
    "native_active_pixels": [4656, 3496],
    "nominal_frame_rate_hz": 30,
    "interface": "USB2.0 High Speed; vendor states USB3.0 support",
    "transport_formats": ["MJPG", "YUY2"],
    "focus": "autofocus 4mm approximately 80 degrees",
    "reference_source": "user_supplied_specification_photo",
    "physical_intrinsics_measured": False,
    "physical_extrinsics_measured": False,
}
AUXILIARY_SIM_VIEW_NAMES_V26 = ("front", "angled", "overhead")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def _attribute_text(attributes: h5py.AttributeManager, name: str) -> str:
    if name not in attributes:
        raise ValueError(f"V26 replay source is missing H5 attribute: {name}")
    value = attributes[name]
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _decode_vector(dataset: h5py.Dataset) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in dataset[...]]


def _camera_id_v26(env: RealisticEdgeArmEnvV10, camera: str) -> int:
    if not isinstance(camera, str) or not camera:
        raise ValueError("V26 replay camera alias must be a nonempty string")
    camera_id = int(
        mujoco.mj_name2id(
            env.model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            f"edgearm_{camera}",
        )
    )
    if camera_id < 0:
        raise ValueError(f"V26 replay scene is missing camera alias: {camera}")
    return camera_id


def _camera_world_pose_v26(
    env: RealisticEdgeArmEnvV10,
    camera: str,
) -> np.ndarray:
    camera_id = _camera_id_v26(env, camera)
    mujoco.mj_forward(env.model, env.data)
    return np.concatenate(
        (
            env.data.cam_xpos[camera_id],
            env.data.cam_xmat[camera_id],
        )
    ).astype(np.float32)


def _simulated_camera_calibration_v26(
    renderer: TrueMultimodalRenderer,
    camera: str,
) -> dict[str, Any]:
    """Build calibration for fixed teacher cameras absent from env._ids."""

    env = renderer.env
    camera_id = _camera_id_v26(env, camera)
    if camera not in renderer.episode_noise:
        raise ValueError(f"V26 replay camera noise is not initialized: {camera}")
    noise = renderer.episode_noise[camera]
    width = int(renderer.config.width)
    height = int(renderer.config.height)
    fovy = float(env.model.cam_fovy[camera_id])
    fy = 0.5 * height / np.tan(np.deg2rad(fovy) / 2.0)
    body_id = int(env.model.cam_bodyid[camera_id])
    body_name = mujoco.mj_id2name(
        env.model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id,
    )
    return {
        "width": width,
        "height": height,
        "intrinsics": [
            [float(fy), 0.0, (width - 1) / 2.0],
            [0.0, float(fy), (height - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        "distortion_k1_k2_p1_p2_k3": [
            noise["k1"],
            noise["k2"],
            noise["p1"],
            noise["p2"],
            0.0,
        ],
        "world_position": env.data.cam_xpos[camera_id].copy().tolist(),
        "world_rotation": env.data.cam_xmat[camera_id].reshape(3, 3).copy().tolist(),
        "fovy_degrees": fovy,
        "mount_parent_body": body_name or "world",
        "mount_type": "fixed_simulation_teacher_camera",
        "mount_profile_version": "edgearm-simulation-teacher-camera-v1",
        "mount_parameter_source": "scene-authored synthetic camera",
        "physically_calibrated": False,
        "simulation_teacher_view": True,
        "local_position": env.model.cam_pos[camera_id].copy().tolist(),
        "local_quaternion_wxyz": env.model.cam_quat[camera_id].copy().tolist(),
        "sensor_model": noise,
        "source_resolution": [
            int(renderer.config.source_width),
            int(renderer.config.source_height),
        ],
        "source_resolution_semantics": (
            "reference sensor resolution metadata; renderer emits the persisted tensor "
            "directly at capture width/height"
        ),
        "rendered_resolution": [width, height],
        "stored_resolution": [width, height],
        "metric_depth_contract": {
            "format": METRIC_DEPTH_CONTRACT_FORMAT,
            "clip_near_m": renderer._depth_clip_near_m,
            "clip_far_m": renderer._depth_clip_far_m,
            "valid_min_exclusive_m": renderer._depth_clip_near_m,
            "valid_max_exclusive_m": renderer._depth_valid_max_exclusive_m,
            "invalid_depth_mm": 0,
            "far_plane_no_hit_is_zeroed_before_sensor_noise": True,
        },
    }


def _create_dataset(
    group: h5py.Group,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
    *,
    image_like: bool = False,
) -> h5py.Dataset:
    return group.create_dataset(
        name,
        shape=shape,
        dtype=dtype,
        compression="lzf" if image_like else None,
        shuffle=image_like,
        chunks=(1, *shape[1:]) if image_like else None,
    )


def episode_supervision_admission_v26(
    *,
    episode_ids: np.ndarray,
    strict_success: np.ndarray,
    terminal_failure: np.ndarray,
    safety_stop: np.ndarray,
    shield_rejected: np.ndarray,
    invalid_contact: np.ndarray,
    valid_contact: np.ndarray,
    execution_attempted: np.ndarray,
    current_submission_safety_unchanged: np.ndarray,
    replay_exact: np.ndarray,
    exact_three_second_contract: bool,
) -> dict[str, Any]:
    arrays = tuple(
        np.asarray(value)
        for value in (
            episode_ids,
            strict_success,
            terminal_failure,
            safety_stop,
            shield_rejected,
            invalid_contact,
            valid_contact,
            execution_attempted,
            current_submission_safety_unchanged,
            replay_exact,
        )
    )
    count = len(arrays[0])
    if any(value.shape != (count,) for value in arrays):
        raise ValueError("V26 replay admission arrays must be aligned vectors")
    if arrays[0].dtype != np.int64 or any(value.dtype != np.bool_ for value in arrays[1:]):
        raise TypeError("V26 replay admission identities/masks have invalid dtypes")
    if type(exact_three_second_contract) is not bool:
        raise TypeError("V26 replay three-second selector must be boolean")
    unique = np.unique(arrays[0])
    if not np.array_equal(unique, np.arange(len(unique), dtype=np.int64)):
        raise ValueError("V26 replay episode ids must be contiguous from zero")
    action_mask = np.zeros(count, dtype=bool)
    world_mask = np.zeros(count, dtype=bool)
    episode_records: list[dict[str, Any]] = []
    for episode_id in unique:
        selected = arrays[0] == episode_id
        strict = bool(np.any(arrays[1][selected]))
        safe = bool(
            not np.any(arrays[2][selected])
            and not np.any(arrays[3][selected])
            and not np.any(arrays[4][selected])
            and not np.any(arrays[5][selected])
        )
        contact = bool(np.any(arrays[6][selected]))
        exact = bool(np.all(arrays[9][selected]))
        eligible_rows = arrays[7][selected] & arrays[8][selected]
        imitation = bool(strict and safe and contact and exact and exact_three_second_contract)
        action_mask[selected] = imitation & eligible_rows
        world_mask[selected] = safe & exact
        episode_records.append(
            {
                "episode_id": int(episode_id),
                "strict_success": strict,
                "safe": safe,
                "valid_contact_present": contact,
                "replay_exact": exact,
                "exact_three_second_contract": exact_three_second_contract,
                "current_safe_submission_row_count": int(np.sum(eligible_rows)),
                "act_action_supervision_eligible": imitation,
                "vla_action_supervision_eligible": imitation,
                "world_model_eligible": bool(safe and exact),
                "production_admission": False,
            }
        )
    return {
        "episode_records": episode_records,
        "act_action_supervision_mask": action_mask,
        "vla_action_supervision_mask": action_mask.copy(),
        "world_model_training_mask": world_mask,
        "strict_success_episode_count": sum(int(record["strict_success"]) for record in episode_records),
        "act_vla_eligible_episode_count": sum(
            int(record["act_action_supervision_eligible"]) for record in episode_records
        ),
        "production_admission": False,
    }


def strict_success_trace_audit_v26(
    *,
    episode_ids: np.ndarray,
    execution_attempted: np.ndarray,
    strict_contained: np.ndarray,
    strict_settled: np.ndarray,
    strict_success_streak: np.ndarray,
    raw_strict_success: np.ndarray,
    strict_success: np.ndarray,
    terminal_failure: np.ndarray,
    evidence_observed: np.ndarray,
    required_hold_steps: int,
) -> dict[str, Any]:
    """Recompute every V26 three-second success streak from replay evidence.

    Shield-rejected rows do not execute a simulator step and therefore carry no
    post-action success evidence.  Every executed row must carry evidence, and
    its stored streak must equal the recurrence over contained-and-settled
    frames within that episode.  This prevents a lone or stale success bit from
    becoming ACT/VLA action supervision.
    """

    identities = np.asarray(episode_ids)
    attempted = np.asarray(execution_attempted)
    contained = np.asarray(strict_contained)
    settled = np.asarray(strict_settled)
    streak = np.asarray(strict_success_streak)
    raw = np.asarray(raw_strict_success)
    success = np.asarray(strict_success)
    failure = np.asarray(terminal_failure)
    observed = np.asarray(evidence_observed)
    count = len(identities)
    bool_vectors = (attempted, contained, settled, raw, success, failure, observed)
    if (
        identities.shape != (count,)
        or streak.shape != (count,)
        or any(value.shape != (count,) for value in bool_vectors)
    ):
        raise ValueError("V26 strict-success trace vectors must be row aligned")
    if identities.dtype != np.int64 or streak.dtype != np.int64:
        raise TypeError("V26 strict-success trace identities/streak must be int64")
    if any(value.dtype != np.bool_ for value in bool_vectors):
        raise TypeError("V26 strict-success trace flags must be boolean")
    if type(required_hold_steps) is not int or required_hold_steps < 1:
        raise ValueError("V26 strict-success required hold steps must be positive")
    if not np.array_equal(observed, attempted):
        raise ValueError("V26 strict-success evidence must exist exactly for executed rows")

    unique = np.unique(identities)
    if not np.array_equal(unique, np.arange(len(unique), dtype=np.int64)):
        raise ValueError("V26 strict-success trace episode ids must be contiguous")
    maximum_streak = 0
    success_rows = 0
    for episode_id in unique:
        rows = np.flatnonzero(identities == episode_id)
        if not np.array_equal(rows, np.arange(rows[0], rows[-1] + 1)):
            raise ValueError("V26 strict-success episode rows must be contiguous")
        expected_streak = 0
        for row in rows:
            if not observed[row]:
                if contained[row] or settled[row] or raw[row] or success[row] or streak[row] != -1:
                    raise ValueError("V26 unexecuted row contains fabricated success evidence")
                continue
            expected_streak = expected_streak + 1 if contained[row] and settled[row] else 0
            if int(streak[row]) != expected_streak:
                raise ValueError("V26 replay strict-success streak recurrence diverged")
            expected_raw = expected_streak >= required_hold_steps
            if bool(raw[row]) != expected_raw:
                raise ValueError("V26 replay raw strict-success threshold diverged")
            expected_success = bool(expected_raw and not failure[row])
            if bool(success[row]) != expected_success:
                raise ValueError("V26 replay strict-success terminal gate diverged")
            maximum_streak = max(maximum_streak, expected_streak)
            success_rows += int(expected_success)
    return {
        "format": "edgearm-v26-strict-success-trace-audit-v1",
        "row_count": count,
        "episode_count": len(unique),
        "required_hold_steps": required_hold_steps,
        "executed_evidence_rows": int(np.count_nonzero(observed)),
        "maximum_observed_streak": maximum_streak,
        "strict_success_row_count": success_rows,
        "all_recurrences_exact": True,
        "exact_three_second_evidence": True,
    }


def _selected_rows(
    episode_ids: np.ndarray,
    strict_success: np.ndarray,
    selector: str,
) -> np.ndarray:
    if selector == "all":
        return np.arange(len(episode_ids), dtype=np.int64)
    if selector != "strict_success":
        raise ValueError("V26 replay episode selector is unsupported")
    successful = {
        int(episode_id)
        for episode_id in np.unique(episode_ids)
        if np.any(strict_success[episode_ids == episode_id])
    }
    rows = np.asarray(
        [index for index, episode_id in enumerate(episode_ids) if int(episode_id) in successful],
        dtype=np.int64,
    )
    if not len(rows):
        raise ValueError("V26 rollout contains no strict-success episode to replay")
    return rows


def _selected_episode_reset_prefixes_v26(
    selected_episode_ids: np.ndarray,
) -> tuple[tuple[int, ...], ...]:
    """Return skipped reset prefixes needed to preserve command epochs."""

    selected = np.asarray(selected_episode_ids)
    if (
        selected.ndim != 1
        or selected.dtype != np.int64
        or len(selected) < 1
        or np.any(selected < 0)
        or np.any(np.diff(selected) <= 0)
    ):
        raise ValueError("V26 replay selected episode ids must be increasing int64")
    prefixes: list[tuple[int, ...]] = []
    next_episode_id = 0
    for episode_id in selected:
        current = int(episode_id)
        prefixes.append(tuple(range(next_episode_id, current + 1)))
        next_episode_id = current + 1
    return tuple(prefixes)


def replay_rl_multimodal_v26(
    *,
    rollout_path: Path,
    collection_run_plan_path: Path,
    output_path: Path,
    width: int = 640,
    height: int = 480,
    auxiliary_width: int = 320,
    auxiliary_height: int = 240,
    episode_selector: str = "all",
    state_tolerance: float = 2.0e-5,
) -> dict[str, Any]:
    dimensions = (width, height, auxiliary_width, auxiliary_height)
    if any(type(value) is not int for value in dimensions):
        raise TypeError("V26 replay resolutions must be integers")
    if width < 64 or height < 48 or auxiliary_width < 64 or auxiliary_height < 48:
        raise ValueError("V26 replay resolution must be at least 64x48")
    if not np.isfinite(state_tolerance) or not 0.0 < state_tolerance <= 1.0e-3:
        raise ValueError("V26 replay state tolerance is outside (0,1e-3]")
    rollout = Path(rollout_path).expanduser().resolve()
    plan_path = Path(collection_run_plan_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    partial = output.with_suffix(output.suffix + ".partial")
    state_path = output.with_suffix(output.suffix + ".state.json")
    failure_path = output.with_suffix(output.suffix + ".failure.json")
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    if output.exists() or partial.exists():
        raise FileExistsError(f"V26 replay output already exists: {output}")
    if not rollout.is_file() or not plan_path.is_file():
        raise FileNotFoundError("V26 replay rollout or run plan is missing")
    source_rollout_sha256 = sha256_file_v1(rollout)
    with h5py.File(rollout, "r") as source:
        plan_sha256 = _attribute_text(source.attrs, "run_plan_sha256")
        parent_checkpoint = (
            Path(_attribute_text(source.attrs, "parent_checkpoint_path")).expanduser().resolve()
        )
        source_language_audit = verify_dynamic_language_v26(source)
        source_strict_contract = json.loads(_attribute_text(source.attrs, "strict_success_contract_json"))
        language_rows = {
            name: _decode_vector(source[f"language_v26/{name}"])
            for name in ("row_instruction_en", "row_instruction_zh")
        }
    plan = load_verified_run_plan_v22(plan_path, expected_sha256=plan_sha256)
    _actor, _critic, root_plan, lineage = load_collection_parent_actor_critic_v23(parent_checkpoint)
    if canonical_sha256_v1(lineage) != canonical_sha256_v1(plan.get("lineage")):
        raise ValueError("V26 replay checkpoint lineage differs from collection")
    environment_config, policy_config, action_config, scene_path = _derive_exact_configs(
        plan,
        root_plan,
    )
    verified_sources = verify_shared_collection_sources_v25(plan)
    batch = load_feasible_multiview_rollout_v22(rollout)
    v26_contract_audit = verify_v26_collection_contracts(
        plan,
        rollout,
        environment_config,
        batch,
    )
    selected_source_rows = _selected_rows(
        batch.episode_ids,
        batch.strict_success,
        episode_selector,
    )
    selected_episode_ids = np.unique(batch.episode_ids[selected_source_rows])
    reset_prefixes = _selected_episode_reset_prefixes_v26(selected_episode_ids)
    episode_id_remap = {
        int(source_episode_id): output_episode_id
        for output_episode_id, source_episode_id in enumerate(selected_episode_ids)
    }
    selected_output_episode_ids = np.asarray(
        [episode_id_remap[int(value)] for value in batch.episode_ids[selected_source_rows]],
        dtype=np.int64,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        state_path,
        {
            "format": RL_MULTIMODAL_REPLAY_FORMAT_V26,
            "status": "running",
            "phase": "preflight_complete",
            "updated_at_utc": _utc_now(),
            "source_rows": len(batch.rewards),
            "selected_rows": len(selected_source_rows),
            "committed_rows": 0,
            "production_admission": False,
        },
    )
    env = RealisticEdgeArmEnvV10(
        environment_config,
        seed=batch.rollout_seed,
        model_scene_path=scene_path,
    )
    reset_renderer: MultiViewRGBRendererV1 | None = None
    wrist_renderer: TrueMultimodalRenderer | None = None
    auxiliary_renderer: TrueMultimodalRenderer | None = None
    replay_exact = np.zeros(len(selected_source_rows), dtype=bool)
    qpos_errors = np.zeros(len(selected_source_rows), dtype=np.float64)
    qvel_errors = np.zeros(len(selected_source_rows), dtype=np.float64)
    strict_contained = np.zeros(len(selected_source_rows), dtype=bool)
    strict_settled = np.zeros(len(selected_source_rows), dtype=bool)
    strict_success_streak = np.full(len(selected_source_rows), -1, dtype=np.int64)
    raw_strict_success = np.zeros(len(selected_source_rows), dtype=bool)
    strict_evidence_observed = np.zeros(len(selected_source_rows), dtype=bool)
    current_submission_safety_unchanged = np.all(
        np.isclose(
            batch.requested_joint_target[selected_source_rows],
            batch.queued_safe_joint_target[selected_source_rows],
            rtol=0.0,
            atol=1.0e-7,
        ),
        axis=1,
    )
    episode_audits: list[dict[str, Any]] = []
    source_to_output = {
        int(source_row): output_row for output_row, source_row in enumerate(selected_source_rows)
    }
    try:
        reset_renderer = MultiViewRGBRendererV1(
            env,
            height=policy_config.image_height,
            width=policy_config.image_width,
        )
        wrist_renderer = TrueMultimodalRenderer(
            env,
            CameraCaptureConfig(
                width=width,
                height=height,
                source_width=AQ16_REFERENCE_V26["native_active_pixels"][0],
                source_height=AQ16_REFERENCE_V26["native_active_pixels"][1],
                cameras=("wrist",),
                apply_lens_distortion=False,
            ),
        )
        auxiliary_renderer = TrueMultimodalRenderer(
            env,
            CameraCaptureConfig(
                width=auxiliary_width,
                height=auxiliary_height,
                source_width=auxiliary_width,
                source_height=auxiliary_height,
                cameras=AUXILIARY_SIM_VIEW_NAMES_V26,
                apply_lens_distortion=False,
            ),
        )
        adapter = StockGripperTaskFrameAdapterV22(env, action_config)
        kernel = StockGripperRolloutKernelV22()
        kernel.validate(env, adapter, StockGripperPotentialRewardV22())
        with h5py.File(partial, "w") as destination:
            destination.attrs.update(
                {
                    "format": RL_MULTIMODAL_REPLAY_FORMAT_V26,
                    "created_at_utc": _utc_now(),
                    "source_type": "sim_rl_scratch",
                    "source_rollout_path": str(rollout),
                    "source_rollout_sha256": source_rollout_sha256,
                    "collection_run_plan_path": str(plan_path),
                    "collection_run_plan_sha256": plan_sha256,
                    "parent_checkpoint_path": str(parent_checkpoint),
                    "parent_checkpoint_sha256": sha256_file_v1(parent_checkpoint),
                    "scene_path": str(scene_path),
                    "scene_sha256": sha256_file_v1(scene_path),
                    "causal_order": RL_MULTIMODAL_REPLAY_CAUSAL_ORDER_V26,
                    "wrist_rgb_primary": True,
                    "simulated_wrist_rgbd": True,
                    "simulated_auxiliary_multiview_rgbd": True,
                    "auxiliary_view_names_json": json.dumps(AUXILIARY_SIM_VIEW_NAMES_V26),
                    "auxiliary_views_are_simulation_teacher_inputs": True,
                    "auxiliary_views_required_at_physical_deployment": False,
                    "all_simulated_views_share_pre_action_state_snapshot": True,
                    "physical_multicamera_simultaneity_claimed": False,
                    "depth_source": "mujoco_metric_zbuffer_plus_versioned_sensor_noise",
                    "segmentation_is_policy_input": False,
                    "segmentation_is_audit_only": True,
                    "lens_distortion_applied": False,
                    "camera_geometry_alignment_exact": True,
                    "camera_4d_reconstructable": True,
                    "aq16_reference_json": json.dumps(
                        AQ16_REFERENCE_V26,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "uno_q_yolo_depth_executed": False,
                    "physical_capture_claimed": False,
                    "physical_hardware_connected": False,
                    "physical_samples": 0,
                    "physical_trials": 0,
                    "wrist_camera_physically_calibrated": False,
                    "expert_calls": 0,
                    "behavior_cloning_steps_used_to_generate_actions": 0,
                    "strict_success_contract_json": json.dumps(
                        source_strict_contract,
                        sort_keys=True,
                    ),
                    "source_rows": len(batch.rewards),
                    "selected_rows": len(selected_source_rows),
                    "episode_selector": episode_selector,
                    "state_tolerance": state_tolerance,
                    "committed_rows": 0,
                    "finalized": False,
                    "production_admission": False,
                }
            )
            rows = len(selected_source_rows)
            observation = destination.create_group("observation")
            rgb_ds = _create_dataset(
                observation,
                "rgb_wrist",
                (rows, height, width, 3),
                np.uint8,
                image_like=True,
            )
            depth_ds = _create_dataset(
                observation,
                "depth_wrist_mm",
                (rows, height, width),
                np.uint16,
                image_like=True,
            )
            segmentation_ds = _create_dataset(
                observation,
                "segmentation_wrist",
                (rows, height, width),
                np.uint8,
                image_like=True,
            )
            pose_ds = _create_dataset(observation, "camera_pose_wrist", (rows, 12), np.float32)
            intrinsics_ds = _create_dataset(
                observation,
                "camera_intrinsics_wrist",
                (rows, 3, 3),
                np.float32,
            )
            depth_valid_ds = _create_dataset(
                observation,
                "depth_valid_mask_wrist",
                (rows, height, width),
                np.bool_,
                image_like=True,
            )
            reconstructable_ds = _create_dataset(
                observation,
                "camera_4d_reconstructable_mask",
                (rows,),
                np.bool_,
            )
            alignment_ds = _create_dataset(
                observation,
                "camera_geometry_alignment_exact",
                (rows,),
                np.bool_,
            )
            auxiliary_rgb_ds: dict[str, h5py.Dataset] = {}
            auxiliary_depth_ds: dict[str, h5py.Dataset] = {}
            auxiliary_segmentation_ds: dict[str, h5py.Dataset] = {}
            auxiliary_pose_ds: dict[str, h5py.Dataset] = {}
            auxiliary_intrinsics_ds: dict[str, h5py.Dataset] = {}
            auxiliary_depth_valid_ds: dict[str, h5py.Dataset] = {}
            auxiliary_reconstructable_ds: dict[str, h5py.Dataset] = {}
            auxiliary_alignment_ds: dict[str, h5py.Dataset] = {}
            for view in AUXILIARY_SIM_VIEW_NAMES_V26:
                auxiliary_rgb_ds[view] = _create_dataset(
                    observation,
                    f"rgb_{view}",
                    (rows, auxiliary_height, auxiliary_width, 3),
                    np.uint8,
                    image_like=True,
                )
                auxiliary_depth_ds[view] = _create_dataset(
                    observation,
                    f"depth_{view}_mm",
                    (rows, auxiliary_height, auxiliary_width),
                    np.uint16,
                    image_like=True,
                )
                auxiliary_segmentation_ds[view] = _create_dataset(
                    observation,
                    f"segmentation_{view}",
                    (rows, auxiliary_height, auxiliary_width),
                    np.uint8,
                    image_like=True,
                )
                auxiliary_pose_ds[view] = _create_dataset(
                    observation,
                    f"camera_pose_{view}",
                    (rows, 12),
                    np.float32,
                )
                auxiliary_intrinsics_ds[view] = _create_dataset(
                    observation,
                    f"camera_intrinsics_{view}",
                    (rows, 3, 3),
                    np.float32,
                )
                auxiliary_depth_valid_ds[view] = _create_dataset(
                    observation,
                    f"depth_valid_mask_{view}",
                    (rows, auxiliary_height, auxiliary_width),
                    np.bool_,
                    image_like=True,
                )
                auxiliary_reconstructable_ds[view] = _create_dataset(
                    observation,
                    f"camera_4d_reconstructable_mask_{view}",
                    (rows,),
                    np.bool_,
                )
                auxiliary_alignment_ds[view] = _create_dataset(
                    observation,
                    f"camera_geometry_alignment_exact_{view}",
                    (rows,),
                    np.bool_,
                )
            observation.create_dataset(
                "joint_state",
                data=batch.joint_state[selected_source_rows],
            )
            observation.create_dataset(
                "previous_executed_action",
                data=batch.previous_executed_action[selected_source_rows],
            )
            observation.create_dataset(
                "policy_joint_history",
                data=batch.policy_joint_history[selected_source_rows],
                compression="lzf",
                shuffle=True,
            )
            observation.create_dataset(
                "policy_action_history",
                data=batch.policy_action_history[selected_source_rows],
                compression="lzf",
                shuffle=True,
            )
            observation.create_dataset(
                "history_valid",
                data=batch.history_valid[selected_source_rows],
            )
            actions = destination.create_group("action")
            for name, values in (
                ("policy_intent_task_action", batch.policy_action),
                ("applied_task_action", batch.applied_task_action),
                ("submitted_joint_action", batch.submitted_joint_action),
                ("actually_executed_joint_action", batch.executed_action),
                ("requested_joint_target", batch.requested_joint_target),
                ("queued_safe_joint_target", batch.queued_safe_joint_target),
                ("applied_joint_target", batch.applied_joint_target),
            ):
                dataset = actions.create_dataset(name, data=values[selected_source_rows])
                if name == "submitted_joint_action":
                    dataset.attrs["policy_target_semantics"] = (
                        "current post-task-guard, pre-transport normalized command"
                    )
                    dataset.attrs["action_supervision_requires_current_submission_safety_unchanged"] = True
            outcome = destination.create_group("outcome")
            for name, values in (
                ("execution_attempted", batch.execution_attempted),
                ("shield_rejected_before_step", batch.shield_rejected_before_step),
                ("valid_push_side_contact", batch.valid_push_side_contact_any),
                ("invalid_tool_block_contact", batch.invalid_tool_block_contact_any),
                ("strict_success", batch.strict_success),
                ("terminal_failure", batch.terminal_failure),
                ("safety_stop", batch.safety_stop),
                ("terminated", batch.terminated),
                ("truncated", batch.truncated),
            ):
                outcome.create_dataset(name, data=values[selected_source_rows])
            outcome.create_dataset(
                "current_submission_safety_unchanged",
                data=current_submission_safety_unchanged,
            )
            strict_contained_ds = outcome.create_dataset(
                "strict_contained_after_step",
                data=strict_contained,
            )
            strict_settled_ds = outcome.create_dataset(
                "strict_settled_after_step",
                data=strict_settled,
            )
            strict_streak_ds = outcome.create_dataset(
                "strict_success_streak_after_step",
                data=strict_success_streak,
            )
            raw_strict_success_ds = outcome.create_dataset(
                "raw_strict_success_after_step",
                data=raw_strict_success,
            )
            strict_evidence_ds = outcome.create_dataset(
                "strict_success_evidence_observed",
                data=strict_evidence_observed,
            )
            index_group = destination.create_group("index")
            index_group.create_dataset("source_row", data=selected_source_rows)
            index_group.create_dataset(
                "episode_id",
                data=selected_output_episode_ids,
            )
            index_group.create_dataset(
                "source_episode_id",
                data=batch.episode_ids[selected_source_rows],
            )
            index_group.create_dataset(
                "episode_step_id",
                data=batch.episode_step_ids[selected_source_rows],
            )
            index_group.create_dataset(
                "control_time_seconds",
                data=(batch.episode_step_ids[selected_source_rows] / float(environment_config.fps)),
            )
            timing = destination.create_group("timing")
            simulation_control_time = (
                batch.episode_step_ids[selected_source_rows] / float(environment_config.fps)
            ).astype(np.float64)
            timing.create_dataset(
                "camera_device_time_seconds",
                data=simulation_control_time,
            )
            timing.create_dataset(
                "camera_host_receive_time_seconds",
                data=simulation_control_time,
            )
            for view in ("wrist", *AUXILIARY_SIM_VIEW_NAMES_V26):
                timing.create_dataset(
                    f"camera_device_time_seconds_{view}",
                    data=simulation_control_time,
                )
                timing.create_dataset(
                    f"camera_host_receive_time_seconds_{view}",
                    data=simulation_control_time,
                )
            timing.attrs.update(
                {
                    "clock_source": "deterministic_simulation_control_clock",
                    "device_host_clock_identity": True,
                    "per_view_time_fields_present": True,
                    "simulated_view_exposure_offset_seconds": 0.0,
                    "simulated_same_state_snapshot": True,
                    "physical_multicamera_sync_claimed": False,
                    "physical_usb_capture_latency_measured": False,
                    "rolling_shutter_row_timing_measured": False,
                    "control_fps": int(environment_config.fps),
                }
            )
            language = destination.create_group("language")
            string_dtype = h5py.string_dtype("utf-8")
            for name, values in language_rows.items():
                language.create_dataset(
                    name,
                    data=np.asarray(
                        [values[int(row)] for row in selected_source_rows],
                        dtype=string_dtype,
                    ),
                )
            language.attrs.update(
                {
                    "source_language_audit_json": json.dumps(
                        source_language_audit,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "policy_input_eligible_for_future_vla": True,
                    "current_scratch_rl_actor_consumed_language": False,
                }
            )
            replay_group = destination.create_group("replay_audit")
            replay_group.create_dataset("qpos_max_abs_error", data=qpos_errors)
            replay_group.create_dataset("qvel_max_abs_error", data=qvel_errors)
            replay_group.create_dataset("state_replay_exact", data=replay_exact)

            committed_rows = 0
            for episode_id, reset_prefix in zip(
                selected_episode_ids,
                reset_prefixes,
                strict=True,
            ):
                record: dict[str, Any] | None = None
                reset_audit: dict[str, Any] | None = None
                for reset_episode_id in reset_prefix:
                    prefix_record = batch.episode_records[reset_episode_id]
                    prefix_reset_audit = kernel.reset_episode(
                        env,
                        reset_renderer,
                        adapter,
                        requested_seed=int(prefix_record["requested_reset_seed"]),
                        obstacle=bool(prefix_record["obstacle_enabled"]),
                        stress=bool(prefix_record["stress_enabled"]),
                    )
                    if (
                        int(prefix_reset_audit["selected_seed"])
                        != int(prefix_record["selected_reset_seed"])
                        or int(prefix_reset_audit["selected_attempt_index"])
                        != int(prefix_record["reset_attempt_index"])
                    ):
                        raise RuntimeError(
                            "V26 replay reset identity diverged at source episode "
                            f"{reset_episode_id}"
                        )
                    realized_prefix_domain = json.loads(
                        json.dumps(
                            env.episode_domain,
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    )
                    if (
                        canonical_sha256_v1(realized_prefix_domain)
                        != prefix_record["episode_domain_sha256"]
                    ):
                        raise RuntimeError(
                            "V26 replay domain randomization diverged at source episode "
                            f"{reset_episode_id}"
                        )
                    if reset_episode_id == int(episode_id):
                        record = prefix_record
                        reset_audit = prefix_reset_audit
                if record is None or reset_audit is None:
                    raise RuntimeError("V26 replay reset prefix omitted selected episode")
                episode_source_rows = np.flatnonzero(batch.episode_ids == episode_id)
                episode_source_rows = np.intersect1d(
                    episode_source_rows,
                    selected_source_rows,
                    assume_unique=True,
                )
                requested_seed = int(record["requested_reset_seed"])
                wrist_renderer.begin_episode(int(reset_audit["selected_seed"]))
                auxiliary_renderer.begin_episode(int(reset_audit["selected_seed"]))
                calibration = wrist_renderer.calibration_metadata()["wrist"]
                intrinsics = np.asarray(calibration["intrinsics"], dtype=np.float32)
                auxiliary_calibration = {
                    view: _simulated_camera_calibration_v26(
                        auxiliary_renderer,
                        view,
                    )
                    for view in AUXILIARY_SIM_VIEW_NAMES_V26
                }
                auxiliary_intrinsics = {
                    view: np.asarray(
                        auxiliary_calibration[view]["intrinsics"],
                        dtype=np.float32,
                    )
                    for view in AUXILIARY_SIM_VIEW_NAMES_V26
                }
                episode_max_qpos = 0.0
                episode_max_qvel = 0.0
                for source_row in episode_source_rows:
                    output_row = source_to_output[int(source_row)]
                    qpos_error = float(
                        np.max(
                            np.abs(
                                np.asarray(env.data.qpos, dtype=np.float64)
                                - batch.sim_qpos[source_row].astype(np.float64)
                            )
                        )
                    )
                    qvel_error = float(
                        np.max(
                            np.abs(
                                np.asarray(env.data.qvel, dtype=np.float64)
                                - batch.sim_qvel[source_row].astype(np.float64)
                            )
                        )
                    )
                    qpos_errors[output_row] = qpos_error
                    qvel_errors[output_row] = qvel_error
                    episode_max_qpos = max(episode_max_qpos, qpos_error)
                    episode_max_qvel = max(episode_max_qvel, qvel_error)
                    if max(qpos_error, qvel_error) > state_tolerance:
                        raise RuntimeError(
                            "V26 deterministic replay state diverged at source row "
                            f"{source_row}: qpos={qpos_error:.7g}, qvel={qvel_error:.7g}"
                        )
                    mujoco.mj_forward(env.model, env.data)
                    capture = wrist_renderer.capture()
                    auxiliary_capture = auxiliary_renderer.capture()
                    camera_pose = _camera_world_pose_v26(env, "wrist")
                    depth = np.asarray(capture["depth_wrist_mm"], dtype=np.uint16)
                    valid_depth = depth > 0
                    rgb_ds[output_row] = capture["rgb_wrist"]
                    depth_ds[output_row] = depth
                    segmentation_ds[output_row] = capture["segmentation_wrist"]
                    pose_ds[output_row] = camera_pose
                    intrinsics_ds[output_row] = intrinsics
                    depth_valid_ds[output_row] = valid_depth
                    reconstructable_ds[output_row] = bool(np.mean(valid_depth) >= 0.05)
                    alignment_ds[output_row] = True
                    for view in AUXILIARY_SIM_VIEW_NAMES_V26:
                        auxiliary_camera_pose = _camera_world_pose_v26(env, view)
                        auxiliary_depth = np.asarray(
                            auxiliary_capture[f"depth_{view}_mm"],
                            dtype=np.uint16,
                        )
                        auxiliary_valid_depth = auxiliary_depth > 0
                        auxiliary_rgb_ds[view][output_row] = auxiliary_capture[f"rgb_{view}"]
                        auxiliary_depth_ds[view][output_row] = auxiliary_depth
                        auxiliary_segmentation_ds[view][output_row] = auxiliary_capture[
                            f"segmentation_{view}"
                        ]
                        auxiliary_pose_ds[view][output_row] = auxiliary_camera_pose
                        auxiliary_intrinsics_ds[view][output_row] = auxiliary_intrinsics[view]
                        auxiliary_depth_valid_ds[view][output_row] = auxiliary_valid_depth
                        auxiliary_reconstructable_ds[view][output_row] = bool(
                            np.mean(auxiliary_valid_depth) >= 0.05
                        )
                        auxiliary_alignment_ds[view][output_row] = True
                    replay_exact[output_row] = True
                    if bool(batch.execution_attempted[source_row]):
                        _observation, _reward, terminated, truncated, info = env.step(
                            batch.submitted_joint_action[source_row]
                        )
                        if bool(terminated) != bool(batch.terminated[source_row]):
                            raise RuntimeError("V26 replay termination flag diverged")
                        if bool(truncated) != bool(batch.truncated[source_row]):
                            raise RuntimeError("V26 replay truncation flag diverged")
                        if bool(info.get("success", False)) != bool(batch.strict_success[source_row]):
                            raise RuntimeError("V26 replay strict-success flag diverged")
                        realism = info.get("realism_v6")
                        if not isinstance(realism, dict):
                            raise RuntimeError("V26 replay lacks strict-success realism evidence")
                        required_fields = (
                            "strict_contained",
                            "strict_settled",
                            "strict_success_streak",
                            "raw_strict_success",
                        )
                        if any(name not in realism for name in required_fields):
                            raise RuntimeError("V26 replay strict-success evidence is incomplete")
                        strict_contained[output_row] = bool(realism["strict_contained"])
                        strict_settled[output_row] = bool(realism["strict_settled"])
                        strict_success_streak[output_row] = int(realism["strict_success_streak"])
                        raw_strict_success[output_row] = bool(realism["raw_strict_success"])
                        strict_evidence_observed[output_row] = True
                    elif not (
                        bool(batch.shield_rejected_before_step[source_row])
                        and bool(batch.terminated[source_row])
                        and bool(batch.terminal_failure[source_row])
                    ):
                        raise RuntimeError("V26 non-executed replay row lacks shield terminal")
                    committed_rows += 1
                    destination.attrs["committed_rows"] = committed_rows
                episode_audits.append(
                    {
                        "episode_id": episode_id_remap[int(episode_id)],
                        "source_episode_id": int(episode_id),
                        "requested_reset_seed": requested_seed,
                        "selected_reset_seed": int(reset_audit["selected_seed"]),
                        "selected_attempt_index": int(reset_audit["selected_attempt_index"]),
                        "row_count": len(episode_source_rows),
                        "maximum_qpos_absolute_error": episode_max_qpos,
                        "maximum_qvel_absolute_error": episode_max_qvel,
                        "camera_calibration": calibration,
                        "auxiliary_camera_calibration": auxiliary_calibration,
                        "simulated_same_state_snapshot": True,
                        "physical_multicamera_simultaneity_claimed": False,
                        "state_replay_exact": True,
                        "production_admission": False,
                    }
                )
                replay_group["qpos_max_abs_error"][...] = qpos_errors
                replay_group["qvel_max_abs_error"][...] = qvel_errors
                replay_group["state_replay_exact"][...] = replay_exact
                strict_contained_ds[...] = strict_contained
                strict_settled_ds[...] = strict_settled
                strict_streak_ds[...] = strict_success_streak
                raw_strict_success_ds[...] = raw_strict_success
                strict_evidence_ds[...] = strict_evidence_observed
                destination.flush()
                _atomic_json(
                    state_path,
                    {
                        "format": RL_MULTIMODAL_REPLAY_FORMAT_V26,
                        "status": "running",
                        "phase": "replaying",
                        "updated_at_utc": _utc_now(),
                        "completed_episodes": len(episode_audits),
                        "selected_episodes": len(selected_episode_ids),
                        "committed_rows": committed_rows,
                        "selected_rows": rows,
                        "production_admission": False,
                    },
                )
            strict_trace_audit = strict_success_trace_audit_v26(
                episode_ids=selected_output_episode_ids,
                execution_attempted=batch.execution_attempted[selected_source_rows],
                strict_contained=strict_contained,
                strict_settled=strict_settled,
                strict_success_streak=strict_success_streak,
                raw_strict_success=raw_strict_success,
                strict_success=batch.strict_success[selected_source_rows],
                terminal_failure=batch.terminal_failure[selected_source_rows],
                evidence_observed=strict_evidence_observed,
                required_hold_steps=int(source_strict_contract["hold_steps"]),
            )
            admission = episode_supervision_admission_v26(
                episode_ids=selected_output_episode_ids,
                strict_success=batch.strict_success[selected_source_rows],
                terminal_failure=batch.terminal_failure[selected_source_rows],
                safety_stop=batch.safety_stop[selected_source_rows],
                shield_rejected=batch.shield_rejected_before_step[selected_source_rows],
                invalid_contact=batch.invalid_tool_block_contact_any[selected_source_rows],
                valid_contact=batch.valid_push_side_contact_any[selected_source_rows],
                execution_attempted=batch.execution_attempted[selected_source_rows],
                current_submission_safety_unchanged=(current_submission_safety_unchanged),
                replay_exact=replay_exact,
                exact_three_second_contract=bool(source_strict_contract["exact_three_second_contract"]),
            )
            admission_group = destination.create_group("admission")
            for name in (
                "act_action_supervision_mask",
                "vla_action_supervision_mask",
                "world_model_training_mask",
            ):
                admission_group.create_dataset(name, data=admission[name])
            admission_group.create_dataset(
                "episode_audit_json",
                data=np.asarray(
                    [
                        json.dumps(record, ensure_ascii=False, sort_keys=True)
                        for record in admission["episode_records"]
                    ],
                    dtype=string_dtype,
                ),
            )
            admission_group.attrs["strict_success_trace_audit_json"] = json.dumps(
                strict_trace_audit,
                sort_keys=True,
            )
            destination.create_dataset(
                "episode_replay_audit_json",
                data=np.asarray(
                    [json.dumps(record, ensure_ascii=False, sort_keys=True) for record in episode_audits],
                    dtype=string_dtype,
                ),
            )
            destination.attrs.update(
                {
                    "act_vla_eligible_episode_count": admission["act_vla_eligible_episode_count"],
                    "strict_success_episode_count": admission["strict_success_episode_count"],
                    "maximum_qpos_absolute_error": float(np.max(qpos_errors)),
                    "maximum_qvel_absolute_error": float(np.max(qvel_errors)),
                    "all_state_replays_exact": bool(np.all(replay_exact)),
                    "strict_success_trace_verified": True,
                    "finalized": True,
                }
            )
            destination.flush()
        partial.replace(output)
    except Exception as error:
        _atomic_json(
            failure_path,
            {
                "format": RL_MULTIMODAL_REPLAY_FORMAT_V26,
                "status": "failed",
                "failed_at_utc": _utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "partial_path": str(partial),
                "production_admission": False,
            },
        )
        _atomic_json(
            state_path,
            {
                "format": RL_MULTIMODAL_REPLAY_FORMAT_V26,
                "status": "failed",
                "phase": "failed",
                "updated_at_utc": _utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "production_admission": False,
            },
        )
        raise
    finally:
        if auxiliary_renderer is not None:
            auxiliary_renderer.close()
        if wrist_renderer is not None:
            wrist_renderer.close()
        if reset_renderer is not None:
            reset_renderer.close()
    summary = {
        "format": RL_MULTIMODAL_REPLAY_FORMAT_V26,
        "status": "complete",
        "completed_at_utc": _utc_now(),
        "output_path": str(output),
        "output_sha256": sha256_file_v1(output),
        "source_rollout_path": str(rollout),
        "source_rollout_sha256": source_rollout_sha256,
        "selected_rows": len(selected_source_rows),
        "selected_episodes": len(selected_episode_ids),
        "maximum_qpos_absolute_error": float(np.max(qpos_errors)),
        "maximum_qvel_absolute_error": float(np.max(qvel_errors)),
        "all_state_replays_exact": bool(np.all(replay_exact)),
        "strict_success_trace_audit": strict_trace_audit,
        "strict_success_episode_count": admission["strict_success_episode_count"],
        "act_vla_eligible_episode_count": admission["act_vla_eligible_episode_count"],
        "current_submission_safety_unchanged_row_count": int(np.sum(current_submission_safety_unchanged)),
        "source_language_audit": source_language_audit,
        "v26_collection_contract_audit": v26_contract_audit,
        "verified_behavior_source_file_count": len(verified_sources),
        "aq16_reference": AQ16_REFERENCE_V26,
        "simulated_wrist_rgbd": True,
        "simulated_auxiliary_multiview_rgbd": True,
        "auxiliary_view_names": list(AUXILIARY_SIM_VIEW_NAMES_V26),
        "wrist_resolution": [width, height],
        "auxiliary_view_resolution": [auxiliary_width, auxiliary_height],
        "physical_multicamera_simultaneity_claimed": False,
        "physical_calibration_pending": True,
        "uno_q_yolo_depth_executed": False,
        "expert_calls": 0,
        "behavior_cloning_steps_used_to_generate_actions": 0,
        "production_admission": False,
    }
    _atomic_json(summary_path, summary)
    _atomic_json(
        state_path,
        {
            "format": RL_MULTIMODAL_REPLAY_FORMAT_V26,
            "status": "complete",
            "phase": "complete",
            "updated_at_utc": _utc_now(),
            "output_path": str(output),
            "output_sha256": summary["output_sha256"],
            "committed_rows": len(selected_source_rows),
            "production_admission": False,
        },
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--collection-run-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--auxiliary-width", type=int, default=320)
    parser.add_argument("--auxiliary-height", type=int, default=240)
    parser.add_argument(
        "--episode-selector",
        choices=("all", "strict_success"),
        default="all",
    )
    parser.add_argument("--state-tolerance", type=float, default=2.0e-5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = replay_rl_multimodal_v26(
        rollout_path=args.rollout,
        collection_run_plan_path=args.collection_run_plan,
        output_path=args.output,
        width=args.width,
        height=args.height,
        auxiliary_width=args.auxiliary_width,
        auxiliary_height=args.auxiliary_height,
        episode_selector=args.episode_selector,
        state_tolerance=args.state_tolerance,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "AQ16_REFERENCE_V26",
    "AUXILIARY_SIM_VIEW_NAMES_V26",
    "RL_MULTIMODAL_REPLAY_CAUSAL_ORDER_V26",
    "RL_MULTIMODAL_REPLAY_FORMAT_V26",
    "episode_supervision_admission_v26",
    "replay_rl_multimodal_v26",
    "strict_success_trace_audit_v26",
]
