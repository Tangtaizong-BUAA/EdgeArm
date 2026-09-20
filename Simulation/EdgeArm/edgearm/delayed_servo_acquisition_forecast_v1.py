"""Exact-state delayed-queue and servo forecast for V14 acquisition.

The forecast snapshots the live V9 simulator, appends one candidate command to
an exact copy of the current delay queue, and applies every queued target with
the plant's real execution-time filter, stateful servo shaper and V7 physics
substep integrator.  The complete live state is restored in ``finally``.

This helper is simulator-privileged and intentionally isolated from V9 so the
V7--V13 public environment behavior remains byte-for-byte unchanged.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .joint_path_planner_v1 import JointPathPlannerConfig
from .sim2real_env import _QueuedCommand
from .sim2real_env_v9 import RealisticEdgeArmEnvV9


DELAYED_SERVO_ACQUISITION_FORECAST_FORMAT_V1 = "edgearm-v14-delayed-queue-servo-physics-forecast-v1"
_JOINTS = 6


@dataclass(frozen=True)
class DelayedServoAcquisitionForecastConfigV1:
    minimum_safety_only_block_clearance_m: float = JointPathPlannerConfig().minimum_tool_block_clearance_m

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.minimum_safety_only_block_clearance_m)
            or self.minimum_safety_only_block_clearance_m <= 0.0
        ):
            raise ValueError("minimum_safety_only_block_clearance_m must be finite and positive")


class DelayedServoAcquisitionForecastV1:
    """Preview one newly submitted action behind the real queued commands."""

    def __init__(
        self,
        env: RealisticEdgeArmEnvV9,
        config: DelayedServoAcquisitionForecastConfigV1 | None = None,
    ) -> None:
        if type(env) is not RealisticEdgeArmEnvV9:
            raise TypeError("DelayedServoAcquisitionForecastV1 requires exact RealisticEdgeArmEnvV9")
        if config is not None and type(config) is not DelayedServoAcquisitionForecastConfigV1:
            raise TypeError("forecast requires exact DelayedServoAcquisitionForecastConfigV1")
        self.env = env
        self.config = config or DelayedServoAcquisitionForecastConfigV1()
        safety = tuple(int(value) for value in env._ids["tool_safety_geoms"])
        contacts = tuple(int(value) for value in env._ids["tool_contact_geoms"])
        roles = tuple(str(value) for value in env._ids["tool_contact_geom_roles"])
        if (
            len(safety) != 96
            or len(contacts) != 2
            or len(set(contacts)) != 2
            or not set(contacts).issubset(safety)
            or roles != ("fixed_tip", "moving_tip")
        ):
            raise RuntimeError("V14 forecast requires the exact V9 96/2 geometry contract")
        self._safety_geoms = safety
        self._contact_geoms = contacts
        self._contact_roles = roles
        self._contact_columns = np.asarray(
            [safety.index(geom_id) for geom_id in contacts],
            dtype=np.int64,
        )
        self._safety_only_geoms = tuple(geom_id for geom_id in safety if geom_id not in contacts)
        if len(self._safety_only_geoms) != 94:
            raise RuntimeError("V14 forecast requires exactly 94 safety-only parts")
        self._safety_only_columns = np.asarray(
            [safety.index(geom_id) for geom_id in self._safety_only_geoms],
            dtype=np.int64,
        )
        self._penetration_tolerance_m = float(env.contact_feasible_config.reset_penetration_tolerance_m)
        if not 0.0 <= self._penetration_tolerance_m <= 1.0e-3:
            raise RuntimeError("V14 forecast penetration tolerance is out of contract")

    @staticmethod
    def _count_matrix(
        trace: Mapping[str, Any],
        key: str,
        *,
        rows: int,
        columns: int,
    ) -> np.ndarray:
        values = np.asarray(trace.get(key, ()))
        if (
            values.shape != (rows, columns)
            or not np.issubdtype(values.dtype, np.number)
            or not np.all(np.isfinite(values))
            or np.any(values < 0)
            or np.any(values != np.floor(values))
        ):
            raise RuntimeError(f"V14 trace has invalid {key}")
        return values.astype(np.int64, copy=False)

    def audit_trace(self, trace: Mapping[str, Any]) -> dict[str, Any]:
        """Apply the Causal-V3 stock-gripper integrity rules to one trace.

        A configured contact-candidate part may penetrate the block only when
        MuJoCo reports a raw contact for that same role, all contacts for that
        role are geometrically valid, and the invalid-contact count is zero.
        Every remaining CoACD part retains the stricter positive-clearance
        requirement.  Historical V14 construction still binds this generic
        audit to its exact 96/2/94 identity.
        """

        ids = tuple(int(value) for value in trace.get("tool_safety_geom_ids", ()))
        if ids != self._safety_geoms:
            raise RuntimeError("V14 trace safety geometry order drifted")
        names = tuple(str(value) for value in trace.get("tool_safety_geom_names", ()))
        safety_count = len(self._safety_geoms)
        contact_count = len(self._contact_geoms)
        if len(names) != safety_count or any(not value for value in names):
            raise RuntimeError("V14 trace safety geometry names are incomplete")
        contact_ids = tuple(int(value) for value in trace.get("tool_contact_geom_ids", ()))
        roles = tuple(str(value) for value in trace.get("tool_contact_role_names", ()))
        if contact_ids != self._contact_geoms or roles != self._contact_roles:
            raise RuntimeError("V14 trace contact-part role identity drifted")

        distances = np.asarray(
            trace.get("tool_safety_block_signed_distance_m", ()),
            dtype=np.float64,
        )
        if (
            distances.ndim != 2
            or distances.shape[0] <= 0
            or distances.shape[1] != safety_count
            or not np.all(np.isfinite(distances))
        ):
            raise RuntimeError("V14 trace lacks finite 96-part substep distances")
        rows = int(distances.shape[0])
        if int(trace.get("physics_substeps", -1)) != rows:
            raise RuntimeError("V14 trace physics-substep count drifted")

        raw = self._count_matrix(
            trace,
            "tool_block_contact_count_by_role",
            rows=rows,
            columns=contact_count,
        )
        invalid = self._count_matrix(
            trace,
            "invalid_tool_block_contact_count_by_role",
            rows=rows,
            columns=contact_count,
        )
        all_valid_values = np.asarray(trace.get("all_tool_block_contacts_geometrically_valid_by_role", ()))
        if (
            all_valid_values.shape != (rows, contact_count)
            or not np.issubdtype(all_valid_values.dtype, np.number)
            or not np.all(np.isfinite(all_valid_values))
            or not np.all(np.isin(all_valid_values, (0, 1)))
        ):
            raise RuntimeError("V14 trace has invalid all-tool-contacts-valid role flags")
        all_valid = all_valid_values.astype(bool, copy=False)
        if np.any(invalid > raw) or not np.array_equal(all_valid, invalid == 0):
            raise RuntimeError("V14 trace per-role contact identity is inconsistent")

        safety_only = distances[:, self._safety_only_columns]
        safety_only_minimum = float(np.min(safety_only))
        safety_only_argmin = np.unravel_index(int(np.argmin(safety_only)), safety_only.shape)
        safety_only_substep = int(safety_only_argmin[0])
        safety_only_column = int(self._safety_only_columns[int(safety_only_argmin[1])])
        safety_only_location = {
            "substep_index": safety_only_substep,
            "safety_geometry_column": safety_only_column,
            "geom_id": int(self._safety_geoms[safety_only_column]),
            "geom_name": names[safety_only_column],
            "signed_distance_m": safety_only_minimum,
        }
        safety_only_valid = bool(
            safety_only_minimum >= self.config.minimum_safety_only_block_clearance_m - 1.0e-12
        )

        contact_distances = distances[:, self._contact_columns]
        contact_penetration = contact_distances < -self._penetration_tolerance_m
        authorized = (raw > 0) & all_valid & (invalid == 0)
        unauthorized = contact_penetration & ~authorized
        unauthorized_rows = np.argwhere(unauthorized)
        first_unauthorized: dict[str, Any] = {}
        if unauthorized_rows.size:
            substep_index, role_index = (
                int(unauthorized_rows[0, 0]),
                int(unauthorized_rows[0, 1]),
            )
            reasons: list[str] = []
            if int(raw[substep_index, role_index]) <= 0:
                reasons.append("same_role_raw_contact_missing")
            if not bool(all_valid[substep_index, role_index]):
                reasons.append("same_role_contacts_not_all_geometrically_valid")
            if int(invalid[substep_index, role_index]) > 0:
                reasons.append("same_role_invalid_contact_present")
            first_unauthorized = {
                "substep_index": substep_index,
                "role_index": role_index,
                "role": self._contact_roles[role_index],
                "contact_geom_id": int(self._contact_geoms[role_index]),
                "safety_geometry_column": int(self._contact_columns[role_index]),
                "signed_distance_m": float(contact_distances[substep_index, role_index]),
                "raw_contact_count": int(raw[substep_index, role_index]),
                "all_contacts_geometrically_valid": bool(all_valid[substep_index, role_index]),
                "invalid_contact_count": int(invalid[substep_index, role_index]),
                "reasons": reasons,
            }

        minimum_by_role = {
            role: float(np.min(contact_distances[:, role_index]))
            for role_index, role in enumerate(self._contact_roles)
        }
        penetration_count_by_role = {
            role: int(np.sum(contact_penetration[:, role_index]))
            for role_index, role in enumerate(self._contact_roles)
        }
        unauthorized_count_by_role = {
            role: int(np.sum(unauthorized[:, role_index]))
            for role_index, role in enumerate(self._contact_roles)
        }
        unauthorized_count = int(np.sum(unauthorized))
        report = {
            "safety_only_geom_count": len(self._safety_only_geoms),
            "minimum_safety_only_block_distance_m": safety_only_minimum,
            "minimum_required_safety_only_block_clearance_m": (
                self.config.minimum_safety_only_block_clearance_m
            ),
            "minimum_safety_only_block_distance_location": (safety_only_location),
            "safety_only_clearance_valid": safety_only_valid,
            "contact_part_penetration_tolerance_m": self._penetration_tolerance_m,
            "minimum_contact_part_block_distance_by_role_m": minimum_by_role,
            "contact_part_penetration_substep_count_by_role": (penetration_count_by_role),
            "unauthorized_contact_part_penetration_count_by_role": (unauthorized_count_by_role),
            "unauthorized_contact_part_penetration_count": unauthorized_count,
            "unauthorized_contact_part_penetration_substeps": int(np.sum(np.any(unauthorized, axis=1))),
            "unauthorized_contact_part_penetration_any": bool(unauthorized_count > 0),
            "first_unauthorized_contact_part_penetration": first_unauthorized,
            "valid": bool(safety_only_valid and unauthorized_count == 0),
        }
        if len(self._safety_only_geoms) == 94:
            report.update(
                {
                    "minimum_94_safety_only_block_distance_m": (safety_only_minimum),
                    "minimum_required_94_safety_only_block_clearance_m": (
                        self.config.minimum_safety_only_block_clearance_m
                    ),
                    "minimum_94_safety_only_block_distance_location": (safety_only_location),
                }
            )
        return report

    @staticmethod
    def _action(value: np.ndarray) -> np.ndarray:
        action = np.asarray(value, dtype=np.float64)
        if action.shape != (_JOINTS,) or not np.all(np.isfinite(action)):
            raise ValueError("candidate action must be a finite six-vector")
        return np.clip(action, -1.0, 1.0)

    def _candidate_command(self, action: np.ndarray) -> _QueuedCommand:
        env = self.env
        reported = np.asarray(env._command_reference_reported_position(), dtype=np.float64)
        requested_physical = (
            reported
            + action * float(env.config.max_joint_delta)
            - np.asarray(env._zero_offset, dtype=np.float64)
        )
        queued_target, reason = env._safety_filter(requested_physical)
        changed = env._target_changed_mask(requested_physical, queued_target)
        return _QueuedCommand(
            queued_target,
            command_id=int(env.next_command_id),
            original_action=action,
            submitted_action=action,
            send_step=int(env.step_count),
            send_time_seconds=float(env.data.time),
            submission_safety_reason=str(reason),
            submission_target_changed_mask=changed,
            ingress_lost=False,
        )

    @staticmethod
    def _counter_snapshot(env: RealisticEdgeArmEnvV9) -> dict[str, Any]:
        return {
            "gate_evaluated": int(env._terminal_viability_gate_evaluated_decisions),
            "gate_skipped": int(env._terminal_viability_gate_skipped_decisions),
            "gate_reasons": deepcopy(env._terminal_viability_gate_reason_counts),
            "exact_count": int(env._terminal_viability_exact_evaluation_count),
            "exact_wall": float(env._terminal_viability_exact_evaluation_wall_seconds),
        }

    @staticmethod
    def _restore_counters(
        env: RealisticEdgeArmEnvV9,
        snapshot: dict[str, Any],
    ) -> None:
        env._terminal_viability_gate_evaluated_decisions = int(snapshot["gate_evaluated"])
        env._terminal_viability_gate_skipped_decisions = int(snapshot["gate_skipped"])
        env._terminal_viability_gate_reason_counts = deepcopy(snapshot["gate_reasons"])
        env._terminal_viability_exact_evaluation_count = int(snapshot["exact_count"])
        env._terminal_viability_exact_evaluation_wall_seconds = float(snapshot["exact_wall"])

    def forecast(self, candidate_action: np.ndarray) -> dict[str, Any]:
        """Forecast all existing queued effects followed by the candidate."""

        env = self.env
        action = self._action(candidate_action)
        candidate = self._candidate_command(action)
        existing = [deepcopy(value) for value in env._command_queue]
        sequence = [*existing, candidate]
        snapshot = env._snapshot_terminal_viability_state()
        counters = self._counter_snapshot(env)
        reports: list[dict[str, Any]] = []
        try:
            env._command_queue = deque(deepcopy(sequence))
            while env._command_queue:
                application_index = len(reports)
                queue_depth_before_application = len(env._command_queue)
                applied = env._command_queue.popleft()
                env._runtime_pusher_desk_safety_stop_requested = False
                env._controller_preflight_v1 = {}
                delayed_before = np.asarray(applied, dtype=np.float64).copy()
                preflight_reason = env._preflight_applied_delayed_command(applied)
                delayed_target, execution_reason = env._safety_filter(np.asarray(applied, dtype=np.float64))
                start = env.data.qpos[:_JOINTS].copy()
                (
                    endpoint,
                    physical_velocity,
                    runtime_reason,
                    _tracking_noise,
                    _runtime_changed_mask,
                ) = env._servo_step(delayed_target)
                env._advance_physics(
                    start,
                    endpoint,
                    physical_velocity,
                    delayed_target,
                )
                env._last_actual_velocity = (env.data.qpos[:_JOINTS] - start) / max(
                    float(env.control_dt), 1.0e-9
                )
                env._refresh_encoder_noise()
                trace = env._physics_substep_contact_v1
                if trace is None:  # pragma: no cover
                    raise RuntimeError("V14 delayed forecast produced no substep trace")
                trace_audit = self.audit_trace(trace)
                block_joint_id = int(env._ids["block_joint"])
                block_qpos_address = int(
                    env.model.jnt_qposadr[block_joint_id]
                )
                block_dof_address = int(
                    env.model.jnt_dofadr[block_joint_id]
                )
                application = {
                    "applied_command_id": int(applied.command_id),
                    "command_send_step": int(applied.send_step),
                    "forecast_apply_step": int(env.step_count + application_index),
                    "forecast_actual_delay_steps": int(
                        env.step_count + application_index - int(applied.send_step)
                    ),
                    "queue_depth_before_application": int(queue_depth_before_application),
                    "candidate_command": bool(int(applied.command_id) == int(candidate.command_id)),
                    "submitted_action_linf": float(np.max(np.abs(applied.submitted_action))),
                    "queued_target_before_preflight_rad": delayed_before.tolist(),
                    "applied_target_after_preflight_rad": np.asarray(
                        delayed_target, dtype=np.float64
                    ).tolist(),
                    "servo_endpoint_rad": np.asarray(endpoint, dtype=np.float64).tolist(),
                    "physics_endpoint_rad": np.asarray(env.data.qpos[:_JOINTS], dtype=np.float64).tolist(),
                    "block_xy_m": np.asarray(env.block_xy(), dtype=np.float64).tolist(),
                    "block_pose_xyz_quaternion_wxyz": np.asarray(
                        env.data.qpos[
                            block_qpos_address : block_qpos_address + 7
                        ],
                        dtype=np.float64,
                    ).tolist(),
                    "block_velocity_linear_angular": np.asarray(
                        env.data.qvel[
                            block_dof_address : block_dof_address + 6
                        ],
                        dtype=np.float64,
                    ).tolist(),
                    "block_linear_angular_speed": [
                        float(value) for value in env._block_speeds()
                    ],
                    "simulation_time_seconds": float(env.data.time),
                    "safety_only_geom_count": int(trace_audit["safety_only_geom_count"]),
                    "minimum_safety_only_block_distance_m": trace_audit[
                        "minimum_safety_only_block_distance_m"
                    ],
                    "safety_only_clearance_valid": trace_audit["safety_only_clearance_valid"],
                    "minimum_safety_only_block_distance_location": (
                        trace_audit["minimum_safety_only_block_distance_location"]
                    ),
                    "minimum_contact_part_block_distance_by_role_m": (
                        trace_audit["minimum_contact_part_block_distance_by_role_m"]
                    ),
                    "unauthorized_contact_part_penetration_any": trace_audit[
                        "unauthorized_contact_part_penetration_any"
                    ],
                    "unauthorized_contact_part_penetration_count": trace_audit[
                        "unauthorized_contact_part_penetration_count"
                    ],
                    "first_unauthorized_contact_part_penetration": trace_audit[
                        "first_unauthorized_contact_part_penetration"
                    ],
                    "trace_audit": trace_audit,
                    "valid": trace_audit["valid"],
                    "preflight_reason": str(preflight_reason),
                    "execution_filter_reason": str(execution_reason),
                    "runtime_filter_reason": str(runtime_reason),
                }
                if int(application["safety_only_geom_count"]) == 94:
                    application.update(
                        {
                            "minimum_94_safety_only_block_distance_m": (
                                application["minimum_safety_only_block_distance_m"]
                            ),
                            "minimum_94_safety_only_block_distance_location": (
                                application["minimum_safety_only_block_distance_location"]
                            ),
                        }
                    )
                reports.append(application)
        finally:
            env._restore_terminal_viability_state(snapshot)
            self._restore_counters(env, counters)

        if not reports or not bool(reports[-1]["candidate_command"]):
            raise RuntimeError("V14 delayed forecast did not apply the candidate last")
        minimum_application_index = min(
            range(len(reports)),
            key=lambda index: float(reports[index]["minimum_safety_only_block_distance_m"]),
        )
        minimum = float(reports[minimum_application_index]["minimum_safety_only_block_distance_m"])
        minimum_location = {
            "forecast_application_index": minimum_application_index,
            "applied_command_id": int(reports[minimum_application_index]["applied_command_id"]),
            **dict(reports[minimum_application_index]["minimum_safety_only_block_distance_location"]),
        }
        minimum_contact_by_role = {
            role: min(float(row["minimum_contact_part_block_distance_by_role_m"][role]) for row in reports)
            for role in self._contact_roles
        }
        unauthorized_count = sum(int(row["unauthorized_contact_part_penetration_count"]) for row in reports)
        failure_reasons: list[str] = []
        if any(not bool(row["safety_only_clearance_valid"]) for row in reports):
            failure_reasons.append(
                "safety_only_block_clearance_below_0p25mm"
                if len(self._safety_only_geoms) == 94
                else "safety_only_block_clearance_below_required_minimum"
            )
        if unauthorized_count > 0:
            failure_reasons.append("unauthorized_contact_part_penetration")
        report = {
            "format": DELAYED_SERVO_ACQUISITION_FORECAST_FORMAT_V1,
            "live_state_written": False,
            "live_state_restored": True,
            "physically_calibrated": False,
            "physical_samples": 0,
            "existing_queue_length": len(existing),
            "forecast_application_count": len(reports),
            "candidate_command_id": int(candidate.command_id),
            "safety_only_geom_count": len(self._safety_only_geoms),
            "minimum_required_safety_only_block_clearance_m": (
                self.config.minimum_safety_only_block_clearance_m
            ),
            "minimum_forecast_safety_only_block_distance_m": minimum,
            "minimum_forecast_safety_only_block_distance_location": (minimum_location),
            "contact_part_penetration_tolerance_m": (self._penetration_tolerance_m),
            "minimum_forecast_contact_part_block_distance_by_role_m": (minimum_contact_by_role),
            "unauthorized_contact_part_penetration_count": unauthorized_count,
            "unauthorized_contact_part_penetration_any": bool(unauthorized_count > 0),
            "failure_reasons": failure_reasons,
            "valid": bool(all(bool(row["valid"]) for row in reports)),
            "applications": reports,
        }
        if len(self._safety_only_geoms) == 94:
            report.update(
                {
                    "minimum_required_94_safety_only_block_clearance_m": (
                        report["minimum_required_safety_only_block_clearance_m"]
                    ),
                    "minimum_forecast_94_safety_only_block_distance_m": (
                        report["minimum_forecast_safety_only_block_distance_m"]
                    ),
                    "minimum_forecast_94_safety_only_block_distance_location": (
                        report["minimum_forecast_safety_only_block_distance_location"]
                    ),
                }
            )
        return report


__all__ = [
    "DELAYED_SERVO_ACQUISITION_FORECAST_FORMAT_V1",
    "DelayedServoAcquisitionForecastConfigV1",
    "DelayedServoAcquisitionForecastV1",
]
