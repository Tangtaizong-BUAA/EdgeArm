"""Execution-derived twelve-phase tracker for causal EdgeArm trajectories.

The tracker consumes only post-``env.step`` effects, physics-substep contact
telemetry, and explicit intervention events.  Teacher labels and planned phases
are intentionally absent from the input contract, preventing plan/effect label
leakage in ACT/VLA datasets.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Final, Mapping

import numpy as np


EXECUTION_PHASE_FORMAT = "edgearm-effect-execution-phase-v1"
EXECUTION_GEOMETRY_FORMAT = "edgearm-source-neutral-effect-geometry-v1"
RECOVERY_EVIDENCE_FORMAT = "edgearm-explicit-recovery-evidence-v1"
EXECUTION_PHASE_NAMES: Final[tuple[str, ...]] = (
    "approach",
    "pre_contact_alignment",
    "first_contact",
    "sustained_push",
    "fine_correction",
    "settle_hold",
    "avoid",
    "perturb_contact_loss",
    "reposition",
    "re_contact",
    "re_push",
    "verified_recovery_settle",
)
EXECUTION_PHASE_IDS: Final[Mapping[str, int]] = MappingProxyType(
    {name: index for index, name in enumerate(EXECUTION_PHASE_NAMES)}
)

# Terminal episode outcomes and recoverable, non-terminal interventions are
# deliberately different namespaces.  A terminal collision/timeout cannot be
# relabelled as a recovery origin in the same episode.
TERMINAL_FAILURE_CODES: Final[tuple[str, ...]] = (
    "none",
    "collision",
    "workspace_out",
    "timeout",
    "estop",
    "other_terminated",
)
RECOVERY_INTERVENTION_TYPES: Final[tuple[str, ...]] = (
    "none",
    "scripted_retract_contact_loss",
    "scripted_command_drop_burst",
    "scripted_runtime_clearance_hold",
)
RECOVERY_ORIGIN_INTERVENTION_CODES: Final[tuple[str, ...]] = (
    "none",
    "scripted_retract_contact_loss",
    "scripted_command_drop_burst",
    "scripted_runtime_clearance_hold",
)
RECOVERY_INTERVENTION_TYPE_IDS: Final[Mapping[str, int]] = MappingProxyType(
    {name: index for index, name in enumerate(RECOVERY_INTERVENTION_TYPES)}
)
RECOVERY_ORIGIN_INTERVENTION_CODE_IDS: Final[Mapping[str, int]] = MappingProxyType(
    {name: index for index, name in enumerate(RECOVERY_ORIGIN_INTERVENTION_CODES)}
)
RECOVERY_INTERVENTION_ORIGIN_BY_TYPE: Final[Mapping[str, str]] = MappingProxyType(
    {name: name for name in RECOVERY_INTERVENTION_TYPES if name != "none"}
)

EVIDENCE_CONTACT: Final[int] = 1 << 0
EVIDENCE_CONTACT_ONSET: Final[int] = 1 << 1
EVIDENCE_CONTACT_SUSTAINED: Final[int] = 1 << 2
EVIDENCE_BLOCK_MOTION: Final[int] = 1 << 3
EVIDENCE_TARGET_PROGRESS: Final[int] = 1 << 4
EVIDENCE_NEAR_TARGET: Final[int] = 1 << 5
EVIDENCE_CONTAINED: Final[int] = 1 << 6
EVIDENCE_SETTLED: Final[int] = 1 << 7
EVIDENCE_ALIGNMENT: Final[int] = 1 << 8
EVIDENCE_OBSTACLE_DETOUR: Final[int] = 1 << 9
EVIDENCE_CONTACT_LOSS: Final[int] = 1 << 10
EVIDENCE_EXPLICIT_INTERVENTION: Final[int] = 1 << 11
EVIDENCE_RECOVERY_ORDER: Final[int] = 1 << 12
EVIDENCE_DESK_CLEARANCE: Final[int] = 1 << 13
EVIDENCE_VALID_PUSH_SIDE_CONTACT: Final[int] = 1 << 14
EVIDENCE_INVALID_RAW_CONTACT: Final[int] = 1 << 15
EVIDENCE_DIRECTIONAL_BLOCK_MOTION: Final[int] = 1 << 16


@dataclass(frozen=True)
class ExecutionPhaseThresholds:
    """Synthetic numerical thresholds; none are physically calibrated."""

    block_motion_epsilon_m: float = 0.0002
    target_progress_epsilon_m: float = 0.0001
    directional_block_motion_epsilon_m: float = 0.0002
    fine_correction_max_motion_m: float = 0.004
    near_target_coverage: float = 0.70
    # These defaults intentionally match ``RealisticEnvV6Config``'s strict
    # success contract.  A trajectory must not be labelled as an execution
    # settle under a second, drifting definition of success.
    strict_coverage_threshold: float = 0.95
    strict_linear_speed_m_s: float = 0.025
    strict_angular_speed_rad_s: float = 0.55
    settle_hold_steps: int = 6
    sustained_contact_transition_steps: int = 2
    alignment_min_behind_m: float = 0.035
    alignment_max_behind_m: float = 0.120
    alignment_lateral_m: float = 0.025
    alignment_vertical_error_m: float = 0.030
    # Matches the V7 dynamic desk barrier hard floor.  Static path clearance
    # alone is insufficient once delayed commands and joint momentum act.
    minimum_pusher_desk_clearance_m: float = 0.002
    obstacle_lateral_detour_m: float = 0.005

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if name in {"settle_hold_steps", "sustained_contact_transition_steps"}:
                if not isinstance(value, int) or value <= 0:
                    raise ValueError(f"{name} must be a positive integer")
            elif not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0.0 <= self.near_target_coverage < self.strict_coverage_threshold <= 1.0:
            raise ValueError("coverage thresholds must satisfy 0 <= near < strict <= 1")
        if self.alignment_min_behind_m >= self.alignment_max_behind_m:
            raise ValueError("alignment behind interval is empty")

    @property
    def profile_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExecutionEffect:
    """Post-transition evidence required by the phase state machine."""

    contact_any: bool
    force_bearing_any: bool
    contact_substep_count: int
    block_xy_displacement_m: tuple[float, float]
    progress_toward_target_m: float
    strict_target_coverage: float
    block_linear_speed_m_s: float
    block_angular_speed_rad_s: float
    tool_block_along_m: float
    tool_block_lateral_m: float
    tool_height_error_m: float
    minimum_pusher_desk_clearance_m: float
    obstacle_route_required: bool = False
    obstacle_lateral_detour_m: float = 0.0
    obstacle_collision_any: bool = False
    intervention_event_id: str = ""
    intervention_kind: str = ""
    origin_failure_code: str = ""
    recovery_intervention_event_id: str = ""
    recovery_intervention_kind: str = ""
    recovery_origin_intervention_code: str = ""
    valid_push_side_contact_any: bool = True
    valid_push_side_contact_substep_count: int = -1
    push_directional_block_displacement_m: float | None = None

    def __post_init__(self) -> None:
        if self.contact_substep_count < 0:
            raise ValueError("contact_substep_count must be non-negative")
        if self.valid_push_side_contact_substep_count < -1:
            raise ValueError("valid_push_side_contact_substep_count must be at least -1")
        displacement = np.asarray(self.block_xy_displacement_m, dtype=np.float64)
        if displacement.shape != (2,) or not np.all(np.isfinite(displacement)):
            raise ValueError("block_xy_displacement_m must be a finite two-vector")
        finite_names = (
            "progress_toward_target_m",
            "strict_target_coverage",
            "block_linear_speed_m_s",
            "block_angular_speed_rad_s",
            "tool_block_along_m",
            "tool_block_lateral_m",
            "tool_height_error_m",
            "minimum_pusher_desk_clearance_m",
            "obstacle_lateral_detour_m",
        )
        for name in finite_names:
            if not np.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if self.push_directional_block_displacement_m is not None and not np.isfinite(
            self.push_directional_block_displacement_m
        ):
            raise ValueError("push_directional_block_displacement_m must be finite when provided")
        if not 0.0 <= self.strict_target_coverage <= 1.0:
            raise ValueError("strict_target_coverage must be in [0, 1]")
        formal_fields = (
            self.recovery_intervention_event_id,
            self.recovery_intervention_kind,
            self.recovery_origin_intervention_code,
        )
        if any(formal_fields) and not all(formal_fields):
            raise ValueError(
                "formal recovery intervention id, type, and origin code must be declared together"
            )
        if self.recovery_intervention_kind:
            expected_origin = RECOVERY_INTERVENTION_ORIGIN_BY_TYPE.get(self.recovery_intervention_kind)
            if expected_origin is None:
                raise ValueError(
                    f"unknown formal recovery intervention type: {self.recovery_intervention_kind}"
                )
            if self.recovery_origin_intervention_code != expected_origin:
                raise ValueError("formal recovery intervention origin does not match its type")


@dataclass(frozen=True)
class SourceNeutralEffectGeometry:
    """Post-effect geometry used for phase evidence, never a policy input."""

    tool_block_direction_xy: tuple[float, float]
    tool_block_along_m: float
    tool_block_lateral_m: float
    tool_height_error_m: float
    operational_tool_height_reference_m: float
    desk_top_m: float
    orientation_invariant_tool_half_diagonal_m: float
    declared_runtime_clearance_m: float


def source_neutral_effect_geometry(
    *,
    tool_geom_center_xyz_m: np.ndarray | tuple[float, float, float],
    block_xy_m: np.ndarray | tuple[float, float],
    target_xy_m: np.ndarray | tuple[float, float],
    desk_geom_center_xyz_m: np.ndarray | tuple[float, float, float],
    desk_geom_rotation_matrix: np.ndarray,
    desk_geom_half_extents_m: np.ndarray | tuple[float, float, float],
    tool_geom_half_extents_m: np.ndarray | tuple[float, float, float],
    declared_runtime_clearance_m: float,
) -> SourceNeutralEffectGeometry:
    """Derive effect geometry without consulting a teacher/expert object.

    The height reference is exactly desk top + the tool box's
    orientation-invariant half diagonal + the runtime clearance declared by
    the environment.  It is deliberately not the expert's adaptive executed
    pre-contact height.
    """

    tool = np.asarray(tool_geom_center_xyz_m, dtype=np.float64)
    block = np.asarray(block_xy_m, dtype=np.float64)
    target = np.asarray(target_xy_m, dtype=np.float64)
    desk_center = np.asarray(desk_geom_center_xyz_m, dtype=np.float64)
    desk_rotation = np.asarray(desk_geom_rotation_matrix, dtype=np.float64)
    desk_half = np.asarray(desk_geom_half_extents_m, dtype=np.float64)
    tool_half = np.asarray(tool_geom_half_extents_m, dtype=np.float64)
    shapes = {
        "tool_geom_center_xyz_m": (tool, (3,)),
        "block_xy_m": (block, (2,)),
        "target_xy_m": (target, (2,)),
        "desk_geom_center_xyz_m": (desk_center, (3,)),
        "desk_geom_rotation_matrix": (desk_rotation, (3, 3)),
        "desk_geom_half_extents_m": (desk_half, (3,)),
        "tool_geom_half_extents_m": (tool_half, (3,)),
    }
    for name, (value, expected_shape) in shapes.items():
        if value.shape != expected_shape or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be finite with shape {expected_shape}")
    if np.any(desk_half <= 0.0) or np.any(tool_half <= 0.0):
        raise ValueError("effect geometry half extents must be positive")
    if not np.allclose(
        desk_rotation @ desk_rotation.T,
        np.eye(3, dtype=np.float64),
        atol=2.0e-5,
        rtol=2.0e-5,
    ):
        raise ValueError("desk_geom_rotation_matrix must be orthonormal")
    runtime_clearance = float(declared_runtime_clearance_m)
    if not np.isfinite(runtime_clearance) or runtime_clearance < 0.0:
        raise ValueError("declared_runtime_clearance_m must be finite and non-negative")

    target_delta = target - block
    target_distance = float(np.linalg.norm(target_delta))
    if target_distance > 1.0e-12:
        direction = target_delta / target_distance
    else:
        approach = block - tool[:2]
        approach_distance = float(np.linalg.norm(approach))
        direction = (
            approach / approach_distance
            if approach_distance > 1.0e-12
            else np.asarray([1.0, 0.0], dtype=np.float64)
        )
    offset_to_block = block - tool[:2]
    along = float(np.dot(offset_to_block, direction))
    lateral = float(abs(direction[0] * offset_to_block[1] - direction[1] * offset_to_block[0]))
    desk_vertical_radius = float(np.dot(np.abs(desk_rotation[2]), desk_half))
    desk_top = float(desk_center[2] + desk_vertical_radius)
    tool_half_diagonal = float(np.linalg.norm(tool_half))
    reference = float(desk_top + tool_half_diagonal + runtime_clearance)
    height_error = float(abs(tool[2] - reference))
    return SourceNeutralEffectGeometry(
        tool_block_direction_xy=(float(direction[0]), float(direction[1])),
        tool_block_along_m=along,
        tool_block_lateral_m=lateral,
        tool_height_error_m=height_error,
        operational_tool_height_reference_m=reference,
        desk_top_m=desk_top,
        orientation_invariant_tool_half_diagonal_m=tool_half_diagonal,
        declared_runtime_clearance_m=runtime_clearance,
    )


class ExecutionPhaseTrackerV1:
    """Stateful effect-only derivation of the formal twelve phases."""

    def __init__(self, thresholds: ExecutionPhaseThresholds | None = None) -> None:
        self.thresholds = thresholds or ExecutionPhaseThresholds()
        self.reset()

    def reset(self) -> None:
        self._previous_contact = False
        self._contact_transition_streak = 0
        self._settle_streak = 0
        self._had_sustained_contact = False
        self._previous_phase = ""
        self._recovery_state = "none"
        self._recovery_epoch_id = 0
        self._recovery_epoch_start_transition_index = -1
        self._recovery_intervention_event_index = -1
        self._recovery_intervention_event_id = ""
        self._recovery_intervention_kind = ""
        self._recovery_origin_intervention_code = ""
        self._pending_intervention_event_index = -1
        self._pending_intervention_event_id = ""
        self._pending_intervention_kind = ""
        self._pending_origin_intervention_code = ""
        self._transition_index = -1

    def update(self, effect: ExecutionEffect) -> dict[str, object]:
        self._transition_index += 1
        thresholds = self.thresholds
        raw_contact = bool(effect.contact_any and effect.force_bearing_any)
        contact = bool(raw_contact and effect.valid_push_side_contact_any)
        semantic_contact_substeps = (
            effect.contact_substep_count
            if effect.valid_push_side_contact_substep_count < 0
            else effect.valid_push_side_contact_substep_count
        )
        contact_onset = bool(contact and not self._previous_contact)
        contact_loss = bool(not contact and self._previous_contact and self._had_sustained_contact)
        self._contact_transition_streak = self._contact_transition_streak + 1 if contact else 0
        contact_sustained = bool(
            contact
            and (
                self._contact_transition_streak >= thresholds.sustained_contact_transition_steps
                or semantic_contact_substeps >= thresholds.sustained_contact_transition_steps
            )
        )
        displacement_norm = float(
            np.linalg.norm(np.asarray(effect.block_xy_displacement_m, dtype=np.float64))
        )
        block_motion = displacement_norm >= thresholds.block_motion_epsilon_m
        target_progress = effect.progress_toward_target_m >= thresholds.target_progress_epsilon_m
        directional_displacement = (
            effect.progress_toward_target_m
            if effect.push_directional_block_displacement_m is None
            else effect.push_directional_block_displacement_m
        )
        directional_block_motion = bool(
            directional_displacement >= thresholds.directional_block_motion_epsilon_m
        )
        near_target = effect.strict_target_coverage >= thresholds.near_target_coverage
        contained = effect.strict_target_coverage >= thresholds.strict_coverage_threshold
        settled = bool(
            effect.block_linear_speed_m_s <= thresholds.strict_linear_speed_m_s
            and effect.block_angular_speed_rad_s <= thresholds.strict_angular_speed_rad_s
        )
        self._settle_streak = self._settle_streak + 1 if contained and settled else 0
        aligned = bool(
            not contact
            and thresholds.alignment_min_behind_m
            <= effect.tool_block_along_m
            <= thresholds.alignment_max_behind_m
            and effect.tool_block_lateral_m <= thresholds.alignment_lateral_m
            and effect.tool_height_error_m <= thresholds.alignment_vertical_error_m
            and effect.minimum_pusher_desk_clearance_m >= thresholds.minimum_pusher_desk_clearance_m
        )
        obstacle_detour = bool(
            effect.obstacle_route_required
            and not effect.obstacle_collision_any
            and abs(effect.obstacle_lateral_detour_m) >= thresholds.obstacle_lateral_detour_m
        )
        diagnostic_intervention = bool(effect.intervention_event_id and effect.intervention_kind)
        explicit_intervention = bool(effect.recovery_intervention_event_id)
        if explicit_intervention:
            if (
                self._pending_intervention_event_id
                and self._pending_intervention_event_id != effect.recovery_intervention_event_id
            ):
                raise ValueError("a second formal intervention cannot replace pending recovery evidence")
            self._pending_intervention_event_index = self._transition_index
            self._pending_intervention_event_id = effect.recovery_intervention_event_id
            self._pending_intervention_kind = effect.recovery_intervention_kind
            self._pending_origin_intervention_code = effect.recovery_origin_intervention_code
        evidence = 0
        evidence |= EVIDENCE_CONTACT if contact else 0
        evidence |= EVIDENCE_VALID_PUSH_SIDE_CONTACT if contact else 0
        evidence |= EVIDENCE_INVALID_RAW_CONTACT if raw_contact and not contact else 0
        evidence |= EVIDENCE_CONTACT_ONSET if contact_onset else 0
        evidence |= EVIDENCE_CONTACT_SUSTAINED if contact_sustained else 0
        evidence |= EVIDENCE_BLOCK_MOTION if block_motion else 0
        evidence |= EVIDENCE_TARGET_PROGRESS if target_progress else 0
        evidence |= EVIDENCE_DIRECTIONAL_BLOCK_MOTION if directional_block_motion else 0
        evidence |= EVIDENCE_NEAR_TARGET if near_target else 0
        evidence |= EVIDENCE_CONTAINED if contained else 0
        evidence |= EVIDENCE_SETTLED if settled else 0
        evidence |= EVIDENCE_ALIGNMENT if aligned else 0
        evidence |= EVIDENCE_OBSTACLE_DETOUR if obstacle_detour else 0
        evidence |= EVIDENCE_CONTACT_LOSS if contact_loss else 0
        evidence |= EVIDENCE_EXPLICIT_INTERVENTION if explicit_intervention else 0
        evidence |= (
            EVIDENCE_DESK_CLEARANCE
            if effect.minimum_pusher_desk_clearance_m >= thresholds.minimum_pusher_desk_clearance_m
            else 0
        )

        recovery_started = bool(contact_loss and self._pending_intervention_event_id)
        if recovery_started:
            self._recovery_epoch_id += 1
            self._recovery_state = "lost"
            self._recovery_epoch_start_transition_index = self._transition_index
            self._recovery_intervention_event_index = self._pending_intervention_event_index
            self._recovery_intervention_event_id = self._pending_intervention_event_id
            self._recovery_intervention_kind = self._pending_intervention_kind
            self._recovery_origin_intervention_code = self._pending_origin_intervention_code
            self._pending_intervention_event_index = -1
            self._pending_intervention_event_id = ""
            self._pending_intervention_kind = ""
            self._pending_origin_intervention_code = ""

        phase = "approach"
        valid = True
        recovery_active_for_output = self._recovery_state != "none"
        if self._recovery_state != "none":
            evidence |= EVIDENCE_RECOVERY_ORDER
            evidence |= EVIDENCE_EXPLICIT_INTERVENTION
            if (
                self._recovery_state == "repush"
                and self._settle_streak >= thresholds.settle_hold_steps
                and self._recovery_origin_intervention_code
            ):
                phase = "verified_recovery_settle"
                self._recovery_state = "none"
            elif recovery_started:
                phase = "perturb_contact_loss"
            elif contact_onset:
                phase = "re_contact"
                self._recovery_state = "recontact"
            elif contact_sustained and block_motion and target_progress and directional_block_motion:
                phase = "re_push"
                self._recovery_state = "repush"
            elif not contact:
                phase = "reposition"
                self._recovery_state = "reposition"
            elif self._recovery_state == "recontact":
                phase = "re_contact"
            elif self._recovery_state == "repush" and contact:
                phase = "re_push"
            else:
                phase = "reposition"
                valid = False
        elif self._settle_streak >= thresholds.settle_hold_steps:
            phase = "settle_hold"
        elif obstacle_detour:
            phase = "avoid"
        elif contact_onset:
            phase = "first_contact"
        elif contact_sustained and block_motion and target_progress and directional_block_motion:
            if near_target and displacement_norm <= thresholds.fine_correction_max_motion_m:
                phase = "fine_correction"
            else:
                phase = "sustained_push"
            self._had_sustained_contact = True
        elif aligned:
            phase = "pre_contact_alignment"
        else:
            phase = "approach"

        transition = phase != self._previous_phase
        self._previous_phase = phase
        self._previous_contact = contact
        recovery_epoch_step_index = (
            self._transition_index - self._recovery_epoch_start_transition_index
            if recovery_active_for_output
            else -1
        )
        intervention_visible = bool(recovery_active_for_output or self._pending_intervention_event_id)
        visible_event_index = (
            self._recovery_intervention_event_index
            if recovery_active_for_output
            else self._pending_intervention_event_index
        )
        visible_intervention_kind = (
            self._recovery_intervention_kind
            if recovery_active_for_output
            else self._pending_intervention_kind
        )
        visible_origin_code = (
            self._recovery_origin_intervention_code
            if recovery_active_for_output
            else self._pending_origin_intervention_code
        )
        return {
            "format": EXECUTION_PHASE_FORMAT,
            "execution_phase_names": EXECUTION_PHASE_NAMES,
            "effect_execution_phase_id": EXECUTION_PHASE_IDS[phase],
            "effect_execution_phase_name": phase,
            "effect_execution_phase_valid": valid,
            "effect_execution_phase_transition": transition,
            "effect_execution_phase_evidence_mask": evidence,
            "raw_force_bearing_contact": raw_contact,
            "valid_push_side_contact": contact,
            "valid_push_side_contact_substep_count": semantic_contact_substeps,
            "push_directional_block_displacement_m": float(directional_displacement),
            "recovery_epoch_id": self._recovery_epoch_id,
            "recovery_epoch_start_transition_index": (
                self._recovery_epoch_start_transition_index if recovery_active_for_output else -1
            ),
            "recovery_epoch_step_index": recovery_epoch_step_index,
            "recovery_intervention_event_index": (visible_event_index if intervention_visible else -1),
            "recovery_intervention_type_id": (
                RECOVERY_INTERVENTION_TYPE_IDS[visible_intervention_kind]
                if intervention_visible
                else RECOVERY_INTERVENTION_TYPE_IDS["none"]
            ),
            "recovery_origin_intervention_code_id": (
                RECOVERY_ORIGIN_INTERVENTION_CODE_IDS[visible_origin_code]
                if intervention_visible
                else RECOVERY_ORIGIN_INTERVENTION_CODE_IDS["none"]
            ),
            "recovery_state_after_transition": self._recovery_state,
            "recovery_origin_intervention_code": (visible_origin_code if intervention_visible else ""),
            "diagnostic_intervention_present": diagnostic_intervention,
            "threshold_profile_hash": thresholds.profile_hash,
            "thresholds_physically_calibrated": False,
            "physical_samples": 0,
        }


def execution_effect_from_info(
    info: Mapping[str, object],
    *,
    tool_block_along_m: float,
    tool_block_lateral_m: float,
    tool_height_error_m: float,
    progress_toward_target_m: float,
    obstacle_route_required: bool = False,
    obstacle_lateral_detour_m: float = 0.0,
    intervention_event_id: str = "",
    intervention_kind: str = "",
    origin_failure_code: str = "",
    recovery_intervention_event_id: str = "",
    recovery_intervention_kind: str = "",
    recovery_origin_intervention_code: str = "",
) -> ExecutionEffect:
    """Build an effect contract from V7 post-step telemetry."""

    trace = info["physics_substep_contact_v1"]
    realism = info["realism_v7"]
    if not isinstance(trace, Mapping) or not isinstance(realism, Mapping):
        raise TypeError("V7 info is missing mapping telemetry")
    displacement = np.asarray(trace["block_xy_displacement_m"], dtype=np.float64)
    intended_push_direction = np.asarray(trace["intended_push_direction_xy"], dtype=np.float64)
    obstacle_counts = np.asarray(trace["obstacle_contact_counts"])
    return ExecutionEffect(
        contact_any=bool(trace["contact_any"]),
        force_bearing_any=bool(trace["force_bearing_any"]),
        contact_substep_count=int(trace["contact_substep_count"]),
        block_xy_displacement_m=(float(displacement[0]), float(displacement[1])),
        progress_toward_target_m=float(progress_toward_target_m),
        strict_target_coverage=float(realism["strict_target_coverage"]),
        block_linear_speed_m_s=float(realism["block_linear_speed_m_s"]),
        block_angular_speed_rad_s=float(realism["block_angular_speed_rad_s"]),
        tool_block_along_m=float(tool_block_along_m),
        tool_block_lateral_m=float(tool_block_lateral_m),
        tool_height_error_m=float(tool_height_error_m),
        minimum_pusher_desk_clearance_m=float(trace["minimum_tool_desk_signed_distance_m"]),
        obstacle_route_required=bool(obstacle_route_required),
        obstacle_lateral_detour_m=float(obstacle_lateral_detour_m),
        obstacle_collision_any=bool(np.any(obstacle_counts > 0)),
        intervention_event_id=intervention_event_id,
        intervention_kind=intervention_kind,
        origin_failure_code=origin_failure_code,
        recovery_intervention_event_id=recovery_intervention_event_id,
        recovery_intervention_kind=recovery_intervention_kind,
        recovery_origin_intervention_code=recovery_origin_intervention_code,
        valid_push_side_contact_any=bool(trace["valid_push_side_contact_any"]),
        valid_push_side_contact_substep_count=int(trace["valid_push_side_contact_substep_count"]),
        push_directional_block_displacement_m=float(np.dot(displacement, intended_push_direction)),
    )


__all__ = [
    "EXECUTION_GEOMETRY_FORMAT",
    "EXECUTION_PHASE_FORMAT",
    "EXECUTION_PHASE_IDS",
    "EXECUTION_PHASE_NAMES",
    "ExecutionEffect",
    "ExecutionPhaseThresholds",
    "ExecutionPhaseTrackerV1",
    "RECOVERY_EVIDENCE_FORMAT",
    "RECOVERY_INTERVENTION_ORIGIN_BY_TYPE",
    "RECOVERY_INTERVENTION_TYPE_IDS",
    "RECOVERY_INTERVENTION_TYPES",
    "RECOVERY_ORIGIN_INTERVENTION_CODE_IDS",
    "RECOVERY_ORIGIN_INTERVENTION_CODES",
    "SourceNeutralEffectGeometry",
    "TERMINAL_FAILURE_CODES",
    "execution_effect_from_info",
    "source_neutral_effect_geometry",
]
