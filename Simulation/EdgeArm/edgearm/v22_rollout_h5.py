"""Materialize and load causal V22 multiview PPO rollout artifacts.

The first admitted V22 RGB rollout was written before the shared H5 writer
persisted the already-collected policy history tensors.  The causal row index
table is sufficient to reconstruct those tensors exactly from the per-row
observations, but only after strict episode-boundary and right-alignment
checks.  This module creates a new artifact and never overwrites its source.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Sequence

import h5py
import numpy as np

from .asymmetric_multiview_ppo_v1 import (
    AsymmetricMultiViewRolloutBatchV1,
    canonical_sha256_v1,
    sha256_file_v1,
)
from .collect_feasible_multiview_rollout_v22 import (
    CAUSAL_HISTORY_DIRECT_PERSISTENCE_FORMAT_V22,
    FEASIBLE_MULTIVIEW_H5_FORMAT_V22,
)


CAUSAL_HISTORY_MATERIALIZATION_FORMAT_V22 = "edgearm-v22-causal-history-materialization-v1"
CAUSAL_HISTORY_DATASET_NAMES_V22 = (
    "policy_rgb_history",
    "policy_joint_history",
    "policy_action_history",
    "history_valid",
    "policy_view_history_valid",
)
CAUSAL_HISTORY_PERSISTENCE_FORMATS_V22 = frozenset(
    {
        CAUSAL_HISTORY_MATERIALIZATION_FORMAT_V22,
        CAUSAL_HISTORY_DIRECT_PERSISTENCE_FORMAT_V22,
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _attribute_text(attributes: h5py.AttributeManager, name: str) -> str:
    if name not in attributes:
        raise ValueError(f"V22 rollout is missing required attribute: {name}")
    value = attributes[name]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _read_string_vector(dataset: h5py.Dataset) -> np.ndarray:
    raw = np.asarray(dataset[...])
    decoded = [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in raw.reshape(-1)]
    return np.asarray(decoded, dtype=np.str_).reshape(raw.shape)


def _read_run_plan_v22(path: Path, expected_sha256: str) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise ValueError("V22 run plan must be a JSON object")
    embedded_sha256 = plan.get("run_plan_sha256")
    if embedded_sha256 != expected_sha256:
        raise ValueError("V22 run plan hash does not match the rollout attribute")
    unhashed = dict(plan)
    unhashed.pop("run_plan_sha256", None)
    if canonical_sha256_v1(unhashed) != expected_sha256:
        raise ValueError("V22 run plan canonical hash verification failed")
    ppo_config = plan.get("ppo_config")
    if not isinstance(ppo_config, dict):
        raise ValueError("V22 run plan is missing ppo_config")
    gamma = ppo_config.get("gamma")
    if type(gamma) not in {int, float} or not np.isfinite(gamma):
        raise ValueError("V22 run plan is missing a finite PPO gamma")
    if not 0.0 < float(gamma) <= 1.0:
        raise ValueError("V22 PPO gamma escaped (0,1]")
    return plan


def load_verified_run_plan_v22(
    path: Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    """Load a V22 run plan only after its embedded canonical hash verifies."""

    return _read_run_plan_v22(
        Path(path).expanduser().resolve(),
        expected_sha256,
    )


def reconstruct_causal_histories_v22(
    *,
    rgb_frames: np.ndarray,
    joint_state: np.ndarray,
    previous_executed_action: np.ndarray,
    view_valid: np.ndarray,
    history_row_indices: np.ndarray,
    episode_ids: np.ndarray,
    episode_step_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    """Reconstruct exact causal histories after validating every lineage row."""

    frames = np.asarray(rgb_frames)
    joints = np.asarray(joint_state)
    actions = np.asarray(previous_executed_action)
    views = np.asarray(view_valid)
    indices = np.asarray(history_row_indices)
    episodes = np.asarray(episode_ids)
    steps = np.asarray(episode_step_ids)
    if frames.ndim != 5 or frames.dtype != np.uint8:
        raise ValueError("V22 current RGB rows must be uint8 [N,V,H,W,3]")
    row_count = frames.shape[0]
    if indices.ndim != 2 or indices.shape[0] != row_count:
        raise ValueError("V22 causal row index table has an invalid shape")
    if indices.dtype != np.int64:
        raise ValueError("V22 causal row indices must be int64")
    history_steps = indices.shape[1]
    if history_steps < 1:
        raise ValueError("V22 causal history must contain at least one slot")
    if joints.shape[0] != row_count or actions.shape[0] != row_count:
        raise ValueError("V22 causal source rows have inconsistent leading dimensions")
    if views.shape != (row_count, frames.shape[1]) or views.dtype != np.bool_:
        raise ValueError("V22 current view-valid rows have an invalid shape or dtype")
    if episodes.shape != (row_count,) or episodes.dtype != np.int64:
        raise ValueError("V22 episode ids must be an int64 row vector")
    if steps.shape != (row_count,) or steps.dtype != np.int64:
        raise ValueError("V22 episode step ids must be an int64 row vector")

    history_valid = indices >= 0
    rgb_history = np.zeros(
        (row_count, history_steps, *frames.shape[1:]),
        dtype=frames.dtype,
    )
    joint_history = np.zeros(
        (row_count, history_steps, *joints.shape[1:]),
        dtype=joints.dtype,
    )
    action_history = np.zeros(
        (row_count, history_steps, *actions.shape[1:]),
        dtype=actions.dtype,
    )
    view_history_valid = np.zeros(
        (row_count, history_steps, frames.shape[1]),
        dtype=np.bool_,
    )
    for row in range(row_count):
        valid_positions = np.flatnonzero(history_valid[row])
        if valid_positions.size < 1:
            raise ValueError(f"V22 causal history row {row} is empty")
        expected_positions = np.arange(
            history_steps - valid_positions.size,
            history_steps,
            dtype=np.int64,
        )
        if not np.array_equal(valid_positions, expected_positions):
            raise ValueError(f"V22 causal history row {row} is not right-aligned")
        source_rows = indices[row, valid_positions]
        if np.any(source_rows < 0) or np.any(source_rows >= row_count):
            raise ValueError(f"V22 causal history row {row} has an out-of-range source")
        if int(source_rows[-1]) != row:
            raise ValueError(f"V22 causal history row {row} does not end at itself")
        episode_start = row - int(steps[row])
        if episode_start < 0 or int(steps[row]) < 0:
            raise ValueError(f"V22 episode step lineage is invalid at row {row}")
        expected_sources = np.arange(
            max(episode_start, row - history_steps + 1),
            row + 1,
            dtype=np.int64,
        )
        if not np.array_equal(source_rows, expected_sources):
            raise ValueError(f"V22 causal history lineage changed at row {row}")
        if np.any(episodes[source_rows] != episodes[row]):
            raise ValueError(f"V22 causal history crossed an episode at row {row}")
        expected_step_ids = np.arange(
            int(steps[row]) - source_rows.size + 1,
            int(steps[row]) + 1,
            dtype=np.int64,
        )
        if not np.array_equal(steps[source_rows], expected_step_ids):
            raise ValueError(f"V22 episode step ids are discontinuous at row {row}")
        rgb_history[row, valid_positions] = frames[source_rows]
        joint_history[row, valid_positions] = joints[source_rows]
        action_history[row, valid_positions] = actions[source_rows]
        view_history_valid[row, valid_positions] = views[source_rows]
    return {
        "policy_rgb_history": rgb_history,
        "policy_joint_history": joint_history,
        "policy_action_history": action_history,
        "history_valid": history_valid,
        "policy_view_history_valid": view_history_valid,
    }


def _write_policy_history_dataset(
    group: h5py.Group,
    name: str,
    value: np.ndarray,
) -> None:
    array = np.asarray(value)
    kwargs: dict[str, Any] = {}
    if array.nbytes >= 1024:
        kwargs.update(compression="lzf", shuffle=True)
    dataset = group.create_dataset(name, data=array, **kwargs)
    dataset.attrs["policy_input_eligible"] = True


def _read_required_dataset(stream: h5py.File, path: str) -> np.ndarray:
    if path not in stream:
        raise ValueError(f"V22 rollout is missing required dataset: {path}")
    return np.asarray(stream[path][...])


def _causal_source_arrays(stream: h5py.File) -> dict[str, np.ndarray]:
    return {
        "rgb_frames": _read_required_dataset(stream, "policy_observation/rgb_frames"),
        "joint_state": _read_required_dataset(stream, "policy_observation/joint_state"),
        "previous_executed_action": _read_required_dataset(
            stream, "policy_observation/previous_executed_action"
        ),
        "view_valid": _read_required_dataset(stream, "policy_observation/view_valid"),
        "history_row_indices": _read_required_dataset(stream, "policy_observation/history_row_indices"),
        "episode_ids": _read_required_dataset(stream, "execution/episode_ids"),
        "episode_step_ids": _read_required_dataset(stream, "execution/episode_step_ids"),
    }


def load_feasible_multiview_rollout_v22(
    path: Path,
    *,
    require_training_eligible: bool = True,
) -> AsymmetricMultiViewRolloutBatchV1:
    """Load one exact, causal, update-eligible V22 rollout batch."""

    resolved = Path(path).expanduser().resolve()
    with h5py.File(resolved, "r") as stream:
        if _attribute_text(stream.attrs, "format") != FEASIBLE_MULTIVIEW_H5_FORMAT_V22:
            raise ValueError("rollout is not an exact V22 feasible multiview artifact")
        if require_training_eligible and not bool(stream.attrs.get("ppo_training_eligible", False)):
            raise ValueError("V22 rollout did not pass its frozen PPO eligibility gates")
        if not bool(stream.attrs.get("ppo_batch_reconstructable", False)):
            raise ValueError("V22 rollout has no admitted causal-history materialization")
        if (
            _attribute_text(stream.attrs, "causal_history_materialization_format")
            not in CAUSAL_HISTORY_PERSISTENCE_FORMATS_V22
        ):
            raise ValueError("V22 rollout has an unknown causal-history persistence format")
        policy = stream.get("policy_observation")
        execution = stream.get("execution")
        reward = stream.get("reward_and_outcome")
        training = stream.get("training_only")
        if not all(isinstance(group, h5py.Group) for group in (policy, execution, reward, training)):
            raise ValueError("V22 rollout group structure is incomplete")
        assert isinstance(policy, h5py.Group)
        assert isinstance(execution, h5py.Group)
        assert isinstance(reward, h5py.Group)
        assert isinstance(training, h5py.Group)
        for name in CAUSAL_HISTORY_DATASET_NAMES_V22:
            if name not in policy:
                raise ValueError(f"V22 rollout is missing causal policy dataset: {name}")
            if not bool(policy[name].attrs.get("policy_input_eligible", False)):
                raise ValueError(f"V22 causal policy dataset lost eligibility: {name}")
        neutral_clearance = "minimum_executed_safety_only_block_clearance_m"
        compatibility_clearance = "minimum_executed_94_safety_only_block_clearance_m"
        if neutral_clearance not in execution or compatibility_clearance not in execution:
            raise ValueError("V22 rollout lost its safety-only clearance evidence")
        if not np.array_equal(
            execution[neutral_clearance][...],
            execution[compatibility_clearance][...],
        ):
            raise ValueError("V22 neutral and compatibility clearance evidence diverged")
        if "episode_randomization_json" not in stream:
            raise ValueError("V22 rollout is missing episode randomization records")
        encoded_records = _read_string_vector(stream["episode_randomization_json"])
        records: list[dict[str, Any]] = []
        for encoded in encoded_records:
            record = json.loads(str(encoded))
            if not isinstance(record, dict):
                raise ValueError("V22 episode randomization record is not an object")
            records.append(record)

        batch = AsymmetricMultiViewRolloutBatchV1(
            rgb_frames=np.asarray(policy["rgb_frames"][...]),
            policy_rgb_history=np.asarray(policy["policy_rgb_history"][...]),
            joint_state=np.asarray(policy["joint_state"][...]),
            policy_joint_history=np.asarray(policy["policy_joint_history"][...]),
            previous_executed_action=np.asarray(policy["previous_executed_action"][...]),
            policy_action_history=np.asarray(policy["policy_action_history"][...]),
            history_valid=np.asarray(policy["history_valid"][...]),
            view_valid=np.asarray(policy["view_valid"][...]),
            policy_view_history_valid=np.asarray(policy["policy_view_history_valid"][...]),
            history_row_indices=np.asarray(policy["history_row_indices"][...]),
            camera_pose=np.asarray(policy["camera_pose"][...]),
            privileged_state=np.asarray(training["privileged_state"][...]),
            next_privileged_state=np.asarray(training["next_privileged_state"][...]),
            visual_geometry_target=np.asarray(training["visual_geometry_target"][...]),
            policy_action=np.asarray(execution["policy_action"][...]),
            previous_policy_pre_tanh=np.asarray(policy["previous_policy_pre_tanh"][...]),
            applied_task_action=np.asarray(execution["applied_task_action"][...]),
            executed_action=np.asarray(execution["executed_action"][...]),
            submitted_joint_action=np.asarray(execution["submitted_joint_action"][...]),
            execution_attempted=np.asarray(execution["execution_attempted"][...]),
            shield_rejected_before_step=np.asarray(execution["shield_rejected_before_step"][...]),
            ik_target_position_world_m=np.asarray(execution["ik_target_position_world_m"][...]),
            ik_target_face_normal_world=np.asarray(execution["ik_target_face_normal_world"][...]),
            ik_application_scale=np.asarray(execution["ik_application_scale"][...]),
            ik_converged=np.asarray(execution["ik_converged"][...]),
            ik_face_label=_read_string_vector(execution["ik_face_label"]),
            ik_failure_reason=_read_string_vector(execution["ik_failure_reason"]),
            guard_safe_candidate=np.asarray(execution["guard_safe_candidate"][...]),
            guard_selected_scale=np.asarray(execution["guard_selected_scale"][...]),
            guard_minimum_one_step_clearance_m=np.asarray(
                execution["guard_minimum_one_step_clearance_m"][...]
            ),
            guard_minimum_braking_clearance_m=np.asarray(execution["guard_minimum_braking_clearance_m"][...]),
            guard_float32_execution_identity=np.asarray(execution["guard_float32_execution_identity"][...]),
            tool_block_contact_any=np.asarray(execution["tool_block_contact_any"][...]),
            tool_block_contact_substep_count=np.asarray(execution["tool_block_contact_substep_count"][...]),
            valid_push_side_contact_any=np.asarray(execution["valid_push_side_contact_any"][...]),
            valid_push_side_contact_substep_count=np.asarray(
                execution["valid_push_side_contact_substep_count"][...]
            ),
            valid_push_side_contact_transient_only=np.asarray(
                execution["valid_push_side_contact_transient_only"][...]
            ),
            invalid_tool_block_contact_any=np.asarray(execution["invalid_tool_block_contact_any"][...]),
            valid_push_side_peak_normal_force_n=np.asarray(
                execution["valid_push_side_peak_normal_force_n"][...]
            ),
            valid_push_side_normal_impulse_discrete_ns=np.asarray(
                execution["valid_push_side_normal_impulse_discrete_ns"][...]
            ),
            minimum_executed_94_safety_only_block_clearance_m=np.asarray(execution[neutral_clearance][...]),
            step_block_displacement_m=np.asarray(execution["step_block_displacement_m"][...]),
            pre_tanh=np.asarray(training["pre_tanh"][...]),
            old_log_probs=np.asarray(training["old_log_probs"][...]),
            raw_environment_reward=np.asarray(reward["raw_environment_reward"][...]),
            reward_before_potential=np.asarray(reward["reward_before_potential"][...]),
            potential_before=np.asarray(reward["potential_before"][...]),
            potential_after=np.asarray(reward["potential_after"][...]),
            potential_next_for_shaping=np.asarray(reward["potential_next_for_shaping"][...]),
            shaped_rewards=np.asarray(reward["shaped_rewards"][...]),
            values=np.asarray(training["values"][...]),
            next_values=np.asarray(training["next_values"][...]),
            terminated=np.asarray(reward["terminated"][...]),
            truncated=np.asarray(reward["truncated"][...]),
            strict_success=np.asarray(reward["strict_success"][...]),
            terminal_failure=np.asarray(reward["terminal_failure"][...]),
            safety_stop=np.asarray(reward["safety_stop"][...]),
            terminal_reason=_read_string_vector(reward["terminal_reason"]),
            episode_ids=np.asarray(execution["episode_ids"][...]),
            episode_step_ids=np.asarray(execution["episode_step_ids"][...]),
            task_ids=np.asarray(policy["task_ids"][...]),
            requested_joint_target=np.asarray(execution["requested_joint_target"][...]),
            queued_safe_joint_target=np.asarray(execution["queued_safe_joint_target"][...]),
            applied_joint_target=np.asarray(execution["applied_joint_target"][...]),
            sim_qpos=np.asarray(training["sim_qpos"][...]),
            sim_qvel=np.asarray(training["sim_qvel"][...]),
            episode_records=tuple(records),
            shaping_gamma=float(stream.attrs["shaping_gamma"]),
            potential_reward_config_sha256=_attribute_text(stream.attrs, "potential_reward_config_sha256"),
            rollout_seed=int(stream.attrs["rollout_seed"]),
            execution_kernel_format=_attribute_text(stream.attrs, "execution_kernel_format"),
            safety_only_geom_count=int(stream.attrs["safety_only_geom_count"]),
            contact_candidate_geom_count=int(stream.attrs["contact_candidate_geom_count"]),
            contact_identity_format=_attribute_text(stream.attrs, "rollout_contact_identity_format"),
            safety_guard_format=_attribute_text(stream.attrs, "safety_guard_format"),
            source_type=_attribute_text(stream.attrs, "source_type"),
            rollout_format=_attribute_text(stream.attrs, "rollout_format"),
        )
    batch.validate()
    return batch


def materialize_causal_rollout_v22(
    source_path: Path,
    destination_path: Path,
    *,
    run_plan_path: Path,
) -> dict[str, Any]:
    """Create a separately hashed, causal-history-complete V22 artifact."""

    source = Path(source_path).expanduser().resolve()
    destination = Path(destination_path).expanduser().resolve()
    run_plan = Path(run_plan_path).expanduser().resolve()
    partial = destination.with_suffix(destination.suffix + ".partial")
    if destination.exists() or partial.exists():
        raise FileExistsError(f"V22 causal rollout output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_sha256 = sha256_file_v1(source)
    with h5py.File(source, "r") as stream:
        if _attribute_text(stream.attrs, "format") != FEASIBLE_MULTIVIEW_H5_FORMAT_V22:
            raise ValueError("causal materialization source is not an exact V22 artifact")
        if not bool(stream.attrs.get("ppo_training_eligible", False)):
            raise ValueError("causal materialization source failed PPO eligibility gates")
        expected_plan_sha256 = _attribute_text(stream.attrs, "run_plan_sha256")
        plan = _read_run_plan_v22(run_plan, expected_plan_sha256)
        reconstructed = reconstruct_causal_histories_v22(**_causal_source_arrays(stream))
        policy = stream.get("policy_observation")
        if not isinstance(policy, h5py.Group):
            raise ValueError("causal materialization source lost policy observations")
        present = {name for name in CAUSAL_HISTORY_DATASET_NAMES_V22 if name in policy}
        if present and present != set(CAUSAL_HISTORY_DATASET_NAMES_V22):
            raise ValueError("causal materialization source contains a partial history set")
        for name in present:
            if not np.array_equal(policy[name][...], reconstructed[name]):
                raise ValueError(f"persisted causal history disagrees with lineage: {name}")
    shutil.copyfile(source, partial)
    try:
        with h5py.File(partial, "r+") as stream:
            policy = stream["policy_observation"]
            for name, value in reconstructed.items():
                if name not in policy:
                    _write_policy_history_dataset(policy, name, value)
                elif not bool(policy[name].attrs.get("policy_input_eligible", False)):
                    policy[name].attrs["policy_input_eligible"] = True
            gamma = float(plan["ppo_config"]["gamma"])
            if "shaping_gamma" in stream.attrs and np.float32(stream.attrs["shaping_gamma"]) != np.float32(
                gamma
            ):
                raise ValueError("rollout shaping gamma disagrees with its signed run plan")
            stream.attrs.update(
                {
                    "shaping_gamma": gamma,
                    "ppo_batch_reconstructable": True,
                    "causal_history_materialization_format": (CAUSAL_HISTORY_MATERIALIZATION_FORMAT_V22),
                    "causal_history_reconstructed_from_current_rows": True,
                    "causal_history_source_sha256": source_sha256,
                    "causal_history_materializer_sha256": sha256_file_v1(Path(__file__).resolve()),
                    "causal_history_dataset_names_json": json.dumps(CAUSAL_HISTORY_DATASET_NAMES_V22),
                    "causal_history_materialized_at_utc": _utc_now(),
                }
            )
            stream.flush()
        loaded = load_feasible_multiview_rollout_v22(partial)
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return {
        "format": CAUSAL_HISTORY_MATERIALIZATION_FORMAT_V22,
        "source_path": str(source),
        "source_sha256": source_sha256,
        "path": str(destination),
        "sha256": sha256_file_v1(destination),
        "byte_count": destination.stat().st_size,
        "rows": len(loaded.rewards),
        "episodes": loaded.completed_episode_count,
        "history_steps": int(loaded.history_row_indices.shape[1]),
        "ppo_batch_reconstructable": True,
        "ppo_training_eligible": True,
        "optimizer_steps": 0,
        "production_admission": False,
    }


def _atomic_new_json(path: Path, payload: dict[str, Any]) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    if path.exists() or partial.exists():
        raise FileExistsError(f"V22 causal receipt already exists: {path}")
    partial.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--run-plan", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = materialize_causal_rollout_v22(
        args.source,
        args.destination,
        run_plan_path=args.run_plan,
    )
    receipt = args.receipt or args.destination.with_name("causal_materialization.json")
    _atomic_new_json(Path(receipt).expanduser().resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CAUSAL_HISTORY_DATASET_NAMES_V22",
    "CAUSAL_HISTORY_MATERIALIZATION_FORMAT_V22",
    "CAUSAL_HISTORY_PERSISTENCE_FORMATS_V22",
    "load_feasible_multiview_rollout_v22",
    "load_verified_run_plan_v22",
    "materialize_causal_rollout_v22",
    "reconstruct_causal_histories_v22",
]
