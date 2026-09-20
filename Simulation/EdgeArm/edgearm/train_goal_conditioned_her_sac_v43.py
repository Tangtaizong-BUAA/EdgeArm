"""Train the persistent V43 goal-conditioned HER-SAC RL data generator.

The run begins from previously collected scratch-RL MuJoCo transitions, keeps
all failed episodes in replay, performs learning-only future-HER relabelling,
and then alternates fresh V22 safety-shielded interaction with off-policy
updates.  It does not train ACT and does not treat HER goals as successful
generated demonstrations.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from .acquisition_rollout_audit_v642 import acquisition_rollout_audit_v642
from .asymmetric_multiview_ppo_v1 import (
    SOURCE_TYPE,
    VIEW_NAMES,
    StockGripperTaskSpaceActionConfigV12,
    StockTaskFrameNoSafeRecoveryV13,
    canonical_sha256_v1,
    sha256_file_v1,
)
from .goal_conditioned_her_sac_v43 import (
    GOAL_CONDITIONED_HER_CHECKPOINT_FORMAT_V43,
    GOAL_CONDITIONED_HER_SAC_FORMAT_V43,
    GoalConditionedHerReplayV43,
    GoalConditionedHerSACBundleV43,
    GoalConditionedHerSACConfigV43,
    achieved_goal_from_privileged_v43,
    desired_goal_from_privileged_v43,
    goal_conditioned_her_sac_update_v43,
    goal_neutral_privileged_state_v43,
    initialize_goal_conditioned_her_sac_v43,
    observation_with_goal_v43,
)
from .privileged_effect_state_v1 import (
    build_privileged_effect_state_v1,
    privileged_effect_state_slices_v1,
)
from .reverse_curriculum_v26 import wilson_lower_bound_v26
from .se_rl_safeguard_projection_v720 import (
    safeguard_projected_task_action_v720,
)
from .sim2real_env_v10 import RealisticEdgeArmEnvV10, RealisticEnvV10Config
from .stock_gripper_action_guard_v3 import StockGripperActionGuardConfigV3
from .stock_gripper_rollout_kernel_v22 import StockGripperRolloutKernelV22
from .stock_gripper_taskframe_v22 import (
    StockGripperTaskFrameAdapterV22,
    reset_stock_taskframe_episode_v22,
)
from .task_independent_home_reset_v597 import (
    StockGripperHomeTaskFrameAdapterV597,
    reset_stock_home_taskframe_episode_v597,
)


GOAL_CONDITIONED_HER_TRAIN_RUN_FORMAT_V43 = "edgearm-v43-goal-conditioned-her-sac-online-training-run-v1"
GOAL_CONDITIONED_HER_EVALUATION_FORMAT_V43 = "edgearm-v43-goal-conditioned-her-sac-heldout-evaluation-v1"
_PRIVILEGED_SLICES_V43 = privileged_effect_state_slices_v1()
# Production demonstrations must represent a genuine full push.  The easier
# 15--17 cm reset band remains useful for RL curriculum/replay, but it must not
# be promoted into the final imitation/VLA corpus.  These thresholds match the
# geometry-aware full-push contract (V506): at least 17 cm initial centre
# separation and at least 13 cm of net object travel.
MINIMUM_FULL_TASK_INITIAL_DISTANCE_M_V607 = 0.170
MINIMUM_FULL_TASK_NET_PROGRESS_M_V607 = 0.130


def _trajectory_data_admission_v607(
    record: dict[str, Any],
) -> dict[str, Any]:
    """Fail closed against trivial or curriculum-reset data promotion."""

    initial_distance = float(record["initial_block_target_distance_m"])
    net_progress = float(record["net_target_progress_m"])
    initial_coverage = float(record["initial_target_coverage"])
    full_task_initial_state = bool(
        initial_coverage == 0.0 and initial_distance >= MINIMUM_FULL_TASK_INITIAL_DISTANCE_M_V607
    )
    meaningful_full_push = bool(net_progress >= MINIMUM_FULL_TASK_NET_PROGRESS_M_V607)
    strict_geometric_success = bool(
        record["strict_success"]
        and float(record["final_strict_hold_fraction"]) >= 1.0
        and int(record["invalid_contact_steps"]) == 0
        and int(record["safety_steps"]) == 0
        and full_task_initial_state
        and meaningful_full_push
    )
    final_home_data_admission = bool(
        strict_geometric_success
        and record.get("task_aligned_privileged_reset") is False
        and record.get("task_independent_final_home_reset") is True
    )
    return {
        "format": "edgearm-v607-nontrivial-trajectory-admission-v1",
        "minimum_initial_object_target_distance_m": (MINIMUM_FULL_TASK_INITIAL_DISTANCE_M_V607),
        "minimum_net_object_target_progress_m": (MINIMUM_FULL_TASK_NET_PROGRESS_M_V607),
        "initial_target_coverage_zero": initial_coverage == 0.0,
        "full_task_initial_state": full_task_initial_state,
        "meaningful_full_push": meaningful_full_push,
        "strict_geometric_success": strict_geometric_success,
        "task_independent_home_reset": bool(record.get("task_independent_final_home_reset") is True),
        "final_home_data_admission": final_home_data_admission,
        "bulk_vla_data_admission": False,
    }


def _training_selection_score_v610(
    evaluation: dict[str, Any],
) -> tuple[int, float, int, int]:
    """Rank curriculum checkpoints without turning selection into admission."""

    episodes = list(evaluation["episodes"])
    return (
        int(evaluation["strict_success_count"]),
        float(evaluation["mean_net_target_progress_m"]),
        -sum(int(row["safety_steps"]) for row in episodes),
        -sum(int(row["ik_failure_steps"]) for row in episodes),
    )


def _precontact_geometry_audit_v604(
    privileged: np.ndarray,
    desired_goal: np.ndarray,
) -> tuple[float, float]:
    """Measure approach quality without supplying an action or route."""

    state = np.asarray(privileged, dtype=np.float64)
    desired = np.asarray(desired_goal, dtype=np.float64)
    block = state[_PRIVILEGED_SLICES_V43["block_pose_xyz_quaternion_wxyz"]][:2]
    tool_pose = state[_PRIVILEGED_SLICES_V43["tool_pose_position_rotation"]]
    direction = desired - block
    direction /= max(float(np.linalg.norm(direction)), 1.0e-12)
    precontact = np.r_[block - 0.055 * direction, 0.055]
    distance = float(np.linalg.norm(tool_pose[:3] - precontact))
    rotation = tool_pose[3:12].reshape(3, 3)
    broad_face_xy = rotation[:, 1][:2]
    alignment = abs(float(np.dot(broad_face_xy, direction))) / max(
        float(np.linalg.norm(broad_face_xy)),
        1.0e-12,
    )
    return distance, float(np.clip(alignment, 0.0, 1.0))


def _automatic_failure_audit_v604(record: dict[str, Any]) -> dict[str, Any]:
    """Classify the first unmet end-to-end phase after every rollout."""

    if bool(record["strict_success"]):
        phase = "strict_success_stable_three_seconds"
    elif bool(record["failure_terminal"]):
        phase = "terminal_or_safety_failure"
    elif int(record["valid_contact_steps"]) == 0:
        acquisition_audit = record.get("acquisition_motion_audit_v642")
        if isinstance(acquisition_audit, dict) and isinstance(acquisition_audit.get("diagnosis"), str):
            phase = f"v642_{acquisition_audit['diagnosis']}"
        else:
            initial_gap = record.get("initial_tool_block_tip_gap_m")
            if initial_gap is not None and float(initial_gap) <= 0.020:
                phase = "near_contact_reset_not_converted_to_valid_contact"
            elif float(record["minimum_tool_precontact_distance_m"]) > 0.120:
                phase = "home_acquisition_did_not_reach_precontact_region"
            elif float(record["maximum_precontact_face_alignment"]) < 0.90:
                phase = "home_acquisition_reached_region_but_not_face_alignment"
            else:
                phase = "precontact_reached_without_valid_contact"
    elif float(record["net_target_progress_m"]) <= 0.001:
        phase = "valid_contact_without_productive_transport"
    elif float(record["maximum_target_coverage"]) <= 0.0:
        phase = "productive_transport_did_not_enter_target"
    elif float(record["maximum_target_coverage"]) < 0.95:
        phase = "partial_target_entry_below_strict_coverage"
    else:
        phase = "target_entry_not_stable_for_three_seconds"
    admission = _trajectory_data_admission_v607(record)
    return {
        "format": "edgearm-v604-automatic-rollout-failure-audit-v1",
        "first_unmet_phase": phase,
        "strict_success": bool(record["strict_success"]),
        "initial_object_target_distance_m": float(record["initial_block_target_distance_m"]),
        "object_target_net_progress_m": float(record["net_target_progress_m"]),
        "valid_contact_steps": int(record["valid_contact_steps"]),
        "invalid_contact_steps": int(record["invalid_contact_steps"]),
        "ik_failure_steps": int(record["ik_failure_steps"]),
        "safety_steps": int(record["safety_steps"]),
        "minimum_tool_precontact_distance_m": float(record["minimum_tool_precontact_distance_m"]),
        "maximum_precontact_face_alignment": float(record["maximum_precontact_face_alignment"]),
        "maximum_target_coverage": float(record["maximum_target_coverage"]),
        "final_strict_hold_fraction": float(record["final_strict_hold_fraction"]),
        "full_task_geometric_admission": bool(admission["strict_geometric_success"]),
        "successful_data_admission": bool(admission["final_home_data_admission"]),
        "bulk_vla_data_admission": False,
        "trajectory_admission_v607": admission,
    }


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


def _append_jsonl(path: Path, payload: object) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()


def _atomic_torch_save(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, partial)
    partial.replace(path)


def _resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("V43 requested CUDA but it is unavailable")
        return "cuda"
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("V43 requested MPS but it is unavailable")
        return "mps"
    if requested != "auto":
        raise ValueError("V43 device must be auto, cpu, cuda, or mps")
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _load_collection_contract(
    path: Path,
) -> tuple[dict[str, Any], RealisticEnvV10Config, StockGripperTaskSpaceActionConfigV12, Path]:
    plan_path = Path(path).expanduser().resolve()
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    if type(payload) is not dict:
        raise TypeError("V43 collection plan must be a JSON dictionary")
    stored_hash = payload.get("run_plan_sha256")
    unhashed = dict(payload)
    unhashed.pop("run_plan_sha256", None)
    if not isinstance(stored_hash, str) or stored_hash != canonical_sha256_v1(unhashed):
        raise ValueError("V43 seed collection run-plan hash is invalid")
    environment_payload = payload.get("environment_config")
    action_payload = payload.get("action_adapter_config")
    scene_value = payload.get("scene_path")
    if (
        type(environment_payload) is not dict
        or type(action_payload) is not dict
        or not isinstance(scene_value, str)
    ):
        raise TypeError("V43 collection contract is incomplete")
    action_values = dict(action_payload)
    guard_payload = action_values.pop("guard", None)
    if type(guard_payload) is not dict:
        raise TypeError("V43 collection guard contract is missing")
    for name in (
        "reset_height_candidates_m",
        "ik_backtracking_scales",
        "curriculum_reset_tip_gap_band_m",
    ):
        if isinstance(action_values.get(name), list):
            action_values[name] = tuple(action_values[name])
    guard_values = dict(guard_payload)
    if isinstance(guard_values.get("candidate_scales"), list):
        guard_values["candidate_scales"] = tuple(guard_values["candidate_scales"])
    environment = RealisticEnvV10Config(**environment_payload)
    action = StockGripperTaskSpaceActionConfigV12(
        **action_values,
        guard=StockGripperActionGuardConfigV3(**guard_values),
    )
    action.validate()
    scene = Path(scene_value).expanduser().resolve()
    if not scene.is_file():
        raise FileNotFoundError("V43 MuJoCo scene is missing")
    return payload, environment, action, scene


class _ResetOnlyRendererV43:
    """The V22 reset protocol needs view identity, not rendered pixels."""

    view_names = VIEW_NAMES

    def __init__(self) -> None:
        self.episode_seed = 0

    def begin_episode(self, seed: int) -> None:
        self.episode_seed = int(seed)


@dataclass(frozen=True)
class AxisScaledColoredExplorationConfigV605:
    """Zero-mean AR(1) scratch exploration in task-action coordinates."""

    standard_deviation: tuple[float, float, float] = (0.65, 0.22, 0.12)
    autoregressive_rho: float = 0.95
    minimum_burst_steps_v748: int = 1
    force_initial_burst_v748: bool = False
    stationary_initialization_v748: bool = False
    acquisition_gate_minimum_v748: float = 0.0

    def validate(self) -> None:
        deviation = np.asarray(self.standard_deviation, dtype=np.float64)
        if (
            deviation.shape != (3,)
            or not np.all(np.isfinite(deviation))
            or np.any(deviation < 0.03)
            or np.any(deviation > 1.0)
        ):
            raise ValueError("V605 exploration deviation must be [3] in [0.03,1]")
        if not np.isfinite(self.autoregressive_rho) or not 0.0 <= self.autoregressive_rho < 0.99:
            raise ValueError("V605 exploration rho must lie in [0,0.99)")
        if (
            type(self.minimum_burst_steps_v748) is not int
            or not 1 <= self.minimum_burst_steps_v748 <= 128
            or type(self.force_initial_burst_v748) is not bool
            or type(self.stationary_initialization_v748) is not bool
            or not np.isfinite(self.acquisition_gate_minimum_v748)
            or not 0.0 <= self.acquisition_gate_minimum_v748 <= 1.0
        ):
            raise ValueError("V748 exploration burst configuration is invalid")


class AxisScaledColoredExplorerV605:
    """Generate temporally coherent actions without a task rule or path."""

    def __init__(
        self,
        config: AxisScaledColoredExplorationConfigV605,
        *,
        seed: int,
    ) -> None:
        config.validate()
        if type(seed) is not int or seed < 0:
            raise ValueError("V605 exploration seed must be non-negative")
        self.config = config
        self.rng = np.random.default_rng(seed)
        deviation = np.asarray(
            self.config.standard_deviation,
            dtype=np.float32,
        )
        self.latent = (
            deviation * self.rng.standard_normal(3).astype(np.float32)
            if self.config.stationary_initialization_v748
            else np.zeros(3, dtype=np.float32)
        )
        self.step_index = 0
        self.selection_call_index_v748 = 0
        self.remaining_burst_steps_v748 = 0
        self.burst_count_v748 = 0

    def select_v748(
        self,
        random_action_probability: float,
        *,
        rng: np.random.Generator,
    ) -> bool:
        """Turn independent epsilon events into bounded coherent motor bursts."""

        if (
            not np.isfinite(random_action_probability)
            or not 0.0 <= random_action_probability <= 1.0
            or not isinstance(rng, np.random.Generator)
        ):
            raise ValueError("V748 exploration selection inputs are invalid")
        force_initial = bool(self.config.force_initial_burst_v748 and self.selection_call_index_v748 == 0)
        continuing = self.remaining_burst_steps_v748 > 0
        start = bool(force_initial or (not continuing and rng.random() < random_action_probability))
        if start:
            self.remaining_burst_steps_v748 = self.config.minimum_burst_steps_v748
            self.burst_count_v748 += 1
        selected = bool(self.remaining_burst_steps_v748 > 0)
        if selected:
            self.remaining_burst_steps_v748 -= 1
        self.selection_call_index_v748 += 1
        return selected

    def sample(self) -> np.ndarray:
        rho = float(self.config.autoregressive_rho)
        innovation_scale = np.asarray(
            self.config.standard_deviation,
            dtype=np.float32,
        ) * np.float32(np.sqrt(1.0 - rho * rho))
        self.latent = np.add(
            np.float32(rho) * self.latent,
            innovation_scale * self.rng.standard_normal(3).astype(np.float32),
            dtype=np.float32,
        )
        self.step_index += 1
        return np.clip(self.latent, -1.0, 1.0).astype(np.float32)


def _policy_action(
    bundle: GoalConditionedHerSACBundleV43,
    privileged_state: np.ndarray,
    desired_goal: np.ndarray,
    *,
    deterministic: bool,
    rng: np.random.Generator,
    random_action_probability: float,
    random_explorer: AxisScaledColoredExplorerV605 | None = None,
) -> np.ndarray:
    random_selected = bool(
        not deterministic
        and (
            random_explorer.select_v748(
                random_action_probability,
                rng=rng,
            )
            if random_explorer is not None
            else rng.random() < random_action_probability
        )
    )
    if random_selected:
        # Zero-mean scratch exploration; no expert direction or path label.
        if random_explorer is not None:
            return random_explorer.sample()
        return rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
    observation = observation_with_goal_v43(
        goal_neutral_privileged_state_v43(privileged_state),
        np.asarray(desired_goal, dtype=np.float32),
    )
    device = next(bundle.actor.parameters()).device
    bundle.actor.eval()
    with torch.no_grad():
        action, _log_probability = bundle.actor.sample(
            torch.from_numpy(observation).to(device).unsqueeze(0),
            deterministic=deterministic,
        )
    return action.squeeze(0).cpu().numpy().astype(np.float32)


def _translated_action_feasible_v43(
    translated: Any,
    selected_action: np.ndarray,
    filter_audit: dict[str, Any],
) -> tuple[bool, bool]:
    """Separate a proven fixed-target hold from a failed motion request."""

    action = np.asarray(selected_action, dtype=np.float32)
    intentional_latched_hold = bool(
        filter_audit.get("selected_source") == "strict_settle_latched_hold"
        and float(np.max(np.abs(action))) <= 1.0e-8
    )
    feasible = bool(
        translated.ik_converged
        and translated.guard_safe_candidate
        and (translated.application_scale > 0.0 or intentional_latched_hold)
    )
    return feasible, intentional_latched_hold


def _se_rl_replay_transition_v448(
    proposed_action: np.ndarray,
    selected_action: np.ndarray,
    applied_action: np.ndarray,
    filter_audit: dict[str, Any],
    *,
    translated_action_feasible: bool,
    intentional_latched_hold: bool,
) -> tuple[np.ndarray, bool, float, dict[str, Any]]:
    """Encode a filtered transition without hiding the actor's proposal.

    A safety filter is part of the environment in the SE-RL formulation.  The
    critic must therefore be conditioned on the *pre-filter* action that led
    to the filtered transition.  Storing only the selected safe action makes
    every rejected proposal alias to the same transition and removes the
    learning signal that should move the actor toward the safe action set.

    The three-second settle latch is a task state-machine action rather than a
    safety correction.  It remains an explicit zero-action behavior so a
    terminal success is not spuriously attributed to a nonzero proposal.
    """

    proposed = np.asarray(proposed_action, dtype=np.float32)
    selected = np.asarray(selected_action, dtype=np.float32)
    applied = np.asarray(applied_action, dtype=np.float32)
    for name, value in (
        ("proposed", proposed),
        ("selected", selected),
        ("applied", applied),
    ):
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError(f"V44.8 {name} replay action is invalid")
    filter_active = bool(filter_audit)
    selected_source = filter_audit.get("selected_source")
    strict_settle = bool(intentional_latched_hold and selected_source == "strict_settle_latched_hold")
    terminal_release = bool(selected_source == "strict_terminal_contact_release")
    task_state_machine_action = bool(strict_settle or terminal_release)
    prefilter_semantics = bool(filter_active and not task_state_machine_action)
    replay_action = proposed.copy() if prefilter_semantics else selected.copy()
    intervention_l2 = float(np.linalg.norm(proposed - selected))
    executed_projection_l2 = float(np.linalg.norm(replay_action - applied))
    filter_intervened = bool(filter_audit.get("intervened", not np.array_equal(proposed, selected)))
    clean_proposed_execution = bool(
        translated_action_feasible and (not prefilter_semantics or not filter_intervened)
    )
    audit = {
        "format": "edgearm-v44.8-se-rl-filtered-transition-v1",
        "replay_action_semantics": (
            "pre_filter_proposed_action"
            if prefilter_semantics
            else "executed_task_state_machine_action"
            if task_state_machine_action
            else "selected_action_without_policy_filter"
        ),
        "critic_observes_filtered_environment_transition": True,
        "filter_intervened": filter_intervened,
        "policy_filter_intervention_l2": intervention_l2,
        "replay_to_applied_projection_l2": executed_projection_l2,
        "clean_proposed_execution": clean_proposed_execution,
        "strict_settle_action_alias_suppressed": strict_settle,
        "terminal_release_action_alias_suppressed": terminal_release,
        "task_state_machine_action_alias_suppressed": (task_state_machine_action),
    }
    return (
        replay_action,
        clean_proposed_execution,
        executed_projection_l2,
        audit,
    )


def _run_episode_v43(
    env: RealisticEdgeArmEnvV10,
    adapter: StockGripperTaskFrameAdapterV22,
    bundle: GoalConditionedHerSACBundleV43,
    *,
    requested_seed: int,
    maximum_steps: int,
    deterministic: bool,
    random_action_probability: float,
    action_seed: int,
    collect_replay: bool,
    policy_action_callback: Callable[..., np.ndarray] | None = None,
    policy_action_filter: Callable[
        [np.ndarray, np.ndarray, np.ndarray],
        tuple[np.ndarray, dict[str, Any]],
    ]
    | None = None,
    policy_action_selector: Callable[
        [np.ndarray, np.ndarray, np.ndarray],
        tuple[np.ndarray, dict[str, Any]],
    ]
    | None = None,
    pre_action_observation_callback: Callable[[RealisticEdgeArmEnvV10, dict[str, Any]], None] | None = None,
    transition_audit_callback: Callable[[dict[str, Any]], None] | None = None,
    post_step_causal_stabilizer: Callable[
        [
            RealisticEdgeArmEnvV10,
            dict[str, Any],
            np.ndarray,
            bool,
            bool,
        ],
        tuple[bool, bool, dict[str, Any]],
    ]
    | None = None,
    episode_reset: Callable[..., dict[str, Any]] | None = None,
    colored_random_exploration: (AxisScaledColoredExplorationConfigV605 | None) = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    if maximum_steps < 1:
        raise ValueError("V43 episode maximum steps must be positive")
    renderer = _ResetOnlyRendererV43()
    reset_function = reset_stock_taskframe_episode_v22 if episode_reset is None else episode_reset
    reset_audit = reset_function(
        env,
        renderer,
        adapter,
        requested_seed=requested_seed,
        obstacle=False,
        stress=False,
    )
    selected_seed = int(reset_audit["selected_seed"])
    rng = np.random.default_rng(action_seed)
    random_explorer = (
        None
        if colored_random_exploration is None
        else AxisScaledColoredExplorerV605(
            colored_random_exploration,
            seed=action_seed ^ 0x605,
        )
    )
    kernel = StockGripperRolloutKernelV22()
    initial_distance = float(env.distance_to_target())
    initial_target_coverage = float(env.block_target_coverage())
    rows: dict[str, list[Any]] = {
        name: []
        for name in (
            "neutral_state",
            "next_neutral_state",
            "achieved_goal",
            "next_achieved_goal",
            "desired_goal",
            "action",
            "applied_action",
            "terminal",
            "failure_terminal",
            "strict_success",
            "action_feasible",
            "execution_action_feasible_v741",
            "safety_violation",
            "execution_failure_terminal_v741",
            "valid_contact",
            "invalid_contact",
            "raw_contact_evidence_v646",
            "step_block_displacement_m",
            "projection_l2",
            "safeguard_projected_action_v720",
            "safeguard_projection_valid_v720",
            "strict_target_coverage",
            "next_strict_target_coverage",
            "strict_hold_fraction",
            "next_strict_hold_fraction",
            "episode_step",
        )
    }
    valid_contact_steps = 0
    invalid_contact_steps = 0
    ik_failure_steps = 0
    safety_steps = 0
    strict_success = False
    failure_terminal = False
    terminal_reason = "external_time_limit"
    policy_filter_intervention_steps = 0
    task_state_machine_override_steps = 0
    policy_filter_risk_probabilities: list[float] = []
    policy_filter_selected_scales: list[float] = []
    intentional_latched_hold_steps = 0
    intentional_terminal_release_steps = 0
    replay_prefilter_action_steps = 0
    replay_clean_proposed_execution_steps = 0
    policy_filter_intervention_l2_values: list[float] = []
    safeguard_projection_l2_values_v720: list[float] = []
    actual_effect_projection_l2_values_v720: list[float] = []
    policy_selector_intervention_steps = 0
    policy_selector_q_improvements: list[float] = []
    minimum_tool_precontact_distance = float("inf")
    maximum_precontact_face_alignment = 0.0
    maximum_target_coverage = initial_target_coverage
    for episode_step in range(maximum_steps):
        privileged = build_privileged_effect_state_v1(env)
        desired = desired_goal_from_privileged_v43(privileged)
        achieved = achieved_goal_from_privileged_v43(privileged)
        precontact_distance, precontact_alignment = _precontact_geometry_audit_v604(privileged, desired)
        minimum_tool_precontact_distance = min(
            minimum_tool_precontact_distance,
            precontact_distance,
        )
        maximum_precontact_face_alignment = max(
            maximum_precontact_face_alignment,
            precontact_alignment,
        )
        action = (
            _policy_action(
                bundle,
                privileged,
                desired,
                deterministic=deterministic,
                rng=rng,
                random_action_probability=random_action_probability,
                random_explorer=random_explorer,
            )
            if policy_action_callback is None
            else policy_action_callback(
                bundle=bundle,
                privileged_state=privileged,
                desired_goal=desired,
                adapter=adapter,
                deterministic=deterministic,
                rng=rng,
                random_action_probability=random_action_probability,
                random_explorer=random_explorer,
            )
        )
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (3,) or not np.all(np.isfinite(action)):
            raise RuntimeError("V43 policy action callback returned an invalid action")
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        actor_action = action.copy()
        selection_audit: dict[str, Any] = {}
        if policy_action_selector is not None:
            selected_policy_action, selection_audit = policy_action_selector(
                action.copy(), privileged.copy(), desired.copy()
            )
            selected_policy_action = np.asarray(selected_policy_action, dtype=np.float32)
            if selected_policy_action.shape != (3,) or not np.all(np.isfinite(selected_policy_action)):
                raise RuntimeError("V43 policy action selector returned an invalid action")
            action = np.clip(selected_policy_action, -1.0, 1.0).astype(np.float32)
            policy_selector_intervention_steps += int(bool(selection_audit.get("intervened", False)))
            q_improvement = selection_audit.get("selected_conservative_q_improvement")
            if isinstance(q_improvement, (int, float)) and np.isfinite(float(q_improvement)):
                policy_selector_q_improvements.append(float(q_improvement))
        proposed_action = action.copy()
        filter_audit: dict[str, Any] = {}
        if policy_action_filter is not None:
            filtered_action, filter_audit = policy_action_filter(
                action.copy(), privileged.copy(), desired.copy()
            )
            filtered_action = np.asarray(filtered_action, dtype=np.float32)
            if filtered_action.shape != (3,) or not np.all(np.isfinite(filtered_action)):
                raise RuntimeError("V43 policy action filter returned an invalid action")
            action = np.clip(filtered_action, -1.0, 1.0).astype(np.float32)
            selected_source = filter_audit.get("selected_source")
            task_state_machine_override = bool(
                selected_source
                in {
                    "strict_settle_latched_hold",
                    "strict_terminal_contact_release",
                }
            )
            terminal_release_override = bool(selected_source == "strict_terminal_contact_release")
            policy_filter_intervention_steps += int(
                bool(filter_audit.get("intervened", False)) and not task_state_machine_override
            )
            task_state_machine_override_steps += int(task_state_machine_override)
            intentional_terminal_release_steps += int(terminal_release_override)
            risk_probability = filter_audit.get("selected_risk_probability")
            if (
                not task_state_machine_override
                and isinstance(risk_probability, (int, float))
                and np.isfinite(float(risk_probability))
            ):
                policy_filter_risk_probabilities.append(float(risk_probability))
            selected_scale = filter_audit.get("selected_task_action_scale")
            if (
                not task_state_machine_override
                and isinstance(selected_scale, (int, float))
                and np.isfinite(float(selected_scale))
            ):
                policy_filter_selected_scales.append(float(selected_scale))
        if pre_action_observation_callback is not None:
            pre_action_observation_callback(
                env,
                {
                    "format": "edgearm-v455-pre-action-observation-callback-v1",
                    "requested_seed": int(requested_seed),
                    "selected_seed": int(selected_seed),
                    "episode_step": int(episode_step),
                    "privileged_state": privileged.copy(),
                    "desired_goal": desired.copy(),
                    "achieved_goal": achieved.copy(),
                    "actor_action": actor_action.copy(),
                    "policy_selection_audit": dict(selection_audit),
                    "proposed_action": proposed_action.copy(),
                    "selected_action": action.copy(),
                    "filter_audit": dict(filter_audit),
                },
            )
        block_before = env.block_xy().copy()
        shield_deadlock = False
        causal_stabilizer_audit: dict[str, Any] = {}
        preserve_latched_target = bool(filter_audit.get("selected_source") == "strict_settle_latched_hold")
        try:
            translated = adapter.translate(
                action,
                preserve_latched_target=preserve_latched_target,
            )
        except StockTaskFrameNoSafeRecoveryV13 as shield_error:
            shield_deadlock = True
            shield_failure_audit = {
                "exception_type": type(shield_error).__name__,
                "exception_message": str(shield_error),
                "last_guard_report": dict(getattr(adapter, "last_guard_report", {})),
                "last_recovery_report": dict(getattr(adapter, "last_recovery_report", {})),
            }
            next_privileged = privileged.copy()
            applied_action = np.zeros(3, dtype=np.float32)
            submitted_joint_action = np.zeros(6, dtype=np.float32)
            action_feasible = False
            transition_terminal = True
            transition_failure = True
            transition_strict = False
            transition_safety = True
            valid_contact = False
            invalid_contact = False
            displacement = 0.0
            terminal_reason = "v22_action_shield_terminal"
            info = None
            telemetry = None
        else:
            shield_failure_audit = None
            _observation, _reward, terminated, truncated, info = env.step(translated.submitted_joint_action)
            if post_step_causal_stabilizer is not None:
                terminated, truncated, causal_stabilizer_audit = post_step_causal_stabilizer(
                    env,
                    info,
                    block_before.copy(),
                    bool(terminated),
                    bool(truncated),
                )
                if (
                    type(terminated) is not bool
                    or type(truncated) is not bool
                    or type(causal_stabilizer_audit) is not dict
                ):
                    raise RuntimeError("V43 post-step causal stabilizer returned invalid evidence")
            next_privileged = build_privileged_effect_state_v1(env)
            telemetry = kernel.transition_contact(
                info,
                block_before_xy_m=block_before,
                block_after_xy_m=env.block_xy(),
            )
            applied_action = np.asarray(translated.applied_task_action, dtype=np.float32)
            submitted_joint_action = np.asarray(translated.submitted_joint_action, dtype=np.float32)
            (
                action_feasible,
                intentional_latched_hold,
            ) = _translated_action_feasible_v43(
                translated,
                action,
                filter_audit,
            )
            transition_terminal = bool(terminated or truncated)
            transition_failure = bool(info.get("terminal_failure", False))
            transition_strict = bool(info.get("success", False))
            safety_value = info.get("safety_stop", False)
            transition_safety = bool(safety_value)
            valid_contact = bool(telemetry["valid_push_side_contact_any"])
            invalid_contact = bool(telemetry["invalid_tool_block_contact_any"])
            displacement = float(telemetry["step_block_displacement_m"])
            terminal_reason = str(info.get("terminal_reason", "nonterminal"))
            intentional_latched_hold_steps += int(intentional_latched_hold)
        selector_applied_action_observer = getattr(
            policy_action_selector,
            "observe_applied_action",
            None,
        )
        if callable(selector_applied_action_observer):
            selector_applied_action_observer(
                applied_action.copy(),
                action_feasible=bool(action_feasible),
            )
        if episode_step + 1 >= maximum_steps and not transition_terminal:
            transition_terminal = True
            terminal_reason = "external_time_limit"
        next_achieved = achieved_goal_from_privileged_v43(next_privileged)
        (
            replay_action,
            replay_action_feasible,
            replay_projection_l2,
            replay_transition_audit,
        ) = _se_rl_replay_transition_v448(
            proposed_action,
            action,
            applied_action,
            filter_audit,
            translated_action_feasible=bool(action_feasible),
            intentional_latched_hold=bool(False if shield_deadlock else intentional_latched_hold),
        )
        if shield_deadlock:
            safeguard_projected_action_v720 = np.zeros(
                3,
                dtype=np.float32,
            )
            safeguard_projection_valid_v720 = False
            safeguard_projection_audit_v720 = {
                "valid": False,
                "reason": "no_safe_translation_terminal",
            }
        else:
            safeguard_projection_v720 = safeguard_projected_task_action_v720(translated)
            safeguard_projected_action_v720 = safeguard_projection_v720.action.copy()
            safeguard_projection_valid_v720 = bool(safeguard_projection_v720.valid)
            safeguard_projection_audit_v720 = {
                "format": safeguard_projection_v720.format,
                "valid": safeguard_projection_valid_v720,
                "guard_selected_scale": (safeguard_projection_v720.guard_selected_scale),
                "adapter_requested_to_safe_l2": (safeguard_projection_v720.requested_to_safe_l2),
                "replay_proposal_to_safe_l2": float(
                    np.linalg.norm(replay_action - safeguard_projected_action_v720)
                ),
                "live_post_step_effect_excluded_from_penalty": True,
            }
        actual_effect_projection_l2_v720 = float(np.linalg.norm(replay_action - applied_action))
        if safeguard_projection_valid_v720:
            safeguard_projection_l2_values_v720.append(
                float(np.linalg.norm(replay_action - safeguard_projected_action_v720))
            )
        actual_effect_projection_l2_values_v720.append(actual_effect_projection_l2_v720)
        replay_prefilter_action_steps += int(
            replay_transition_audit["replay_action_semantics"] == "pre_filter_proposed_action"
        )
        prefilter_replay_step = bool(
            replay_transition_audit["replay_action_semantics"] == "pre_filter_proposed_action"
        )
        replay_clean_proposed_execution_steps += int(prefilter_replay_step and replay_action_feasible)
        if prefilter_replay_step:
            policy_filter_intervention_l2_values.append(
                float(replay_transition_audit["policy_filter_intervention_l2"])
            )
        if transition_audit_callback is not None:
            translation_audit = (
                {
                    "application_scale": float(translated.application_scale),
                    "ik_converged": bool(translated.ik_converged),
                    "guard_safe_candidate": bool(translated.guard_safe_candidate),
                    "guard_selected_scale": float(translated.guard_selected_scale),
                    "guard_minimum_one_step_clearance_m": float(
                        translated.guard_minimum_one_step_clearance_m
                    ),
                    "guard_minimum_braking_clearance_m": float(translated.guard_minimum_braking_clearance_m),
                    "face_label": str(translated.face_label),
                    "failure_reason": str(translated.failure_reason),
                    "latched_target_preserved": preserve_latched_target,
                }
                if not shield_deadlock
                else {
                    "application_scale": 0.0,
                    "ik_converged": False,
                    "guard_safe_candidate": False,
                    "guard_selected_scale": 0.0,
                    "face_label": "v22_action_shield_terminal",
                    "failure_reason": "no_guard_safe_candidate",
                    "latched_target_preserved": preserve_latched_target,
                }
            )
            if not shield_deadlock:
                dls_projection = getattr(
                    translated,
                    "dls_projection",
                    None,
                )
                translation_scale = np.asarray(
                    getattr(
                        translated,
                        "translation_scale_xyz_m",
                        np.ones(3, dtype=np.float64),
                    ),
                    dtype=np.float64,
                )
                if dls_projection is not None:
                    predicted_local = np.asarray(
                        dls_projection.predicted_local_delta_m,
                        dtype=np.float64,
                    )
                    if translation_scale.shape != (3,) or np.any(translation_scale <= 0.0):
                        raise RuntimeError("V43 DLS translation scale audit is invalid")
                    translation_audit["dls_projection_v688"] = {
                        "requested_task_action": np.asarray(
                            dls_projection.requested_task_action,
                            dtype=np.float64,
                        ).tolist(),
                        "requested_local_delta_m": np.asarray(
                            dls_projection.requested_local_delta_m,
                            dtype=np.float64,
                        ).tolist(),
                        "predicted_local_delta_m": predicted_local.tolist(),
                        "predicted_normalized_task_action": (predicted_local / translation_scale).tolist(),
                        "requested_to_predicted_task_l2": float(
                            dls_projection.requested_to_predicted_task_l2
                        ),
                        "joint_limit_scale": float(dls_projection.joint_limit_scale),
                        "position_singular_values": np.asarray(
                            dls_projection.position_singular_values,
                            dtype=np.float64,
                        ).tolist(),
                        "orientation_nullspace_singular_values": np.asarray(
                            dls_projection.orientation_nullspace_singular_values,
                            dtype=np.float64,
                        ).tolist(),
                        "orientation_residual_before": np.asarray(
                            dls_projection.orientation_residual_before,
                            dtype=np.float64,
                        ).tolist(),
                        "predicted_orientation_residual_after": np.asarray(
                            dls_projection.predicted_orientation_residual_after,
                            dtype=np.float64,
                        ).tolist(),
                    }
            transition_audit_callback(
                {
                    "format": "edgearm-v43-transition-audit-callback-v1",
                    "requested_seed": int(requested_seed),
                    "selected_seed": int(selected_seed),
                    "episode_step": int(episode_step),
                    "privileged_state": privileged.copy(),
                    "next_privileged_state": next_privileged.copy(),
                    "desired_goal": desired.copy(),
                    "achieved_goal": achieved.copy(),
                    "next_achieved_goal": next_achieved.copy(),
                    "actor_action": actor_action.copy(),
                    "policy_selection_audit": dict(selection_audit),
                    "proposed_action": proposed_action.copy(),
                    "selected_action": action.copy(),
                    "applied_action": applied_action.copy(),
                    "submitted_joint_action": submitted_joint_action.copy(),
                    "replay_action": replay_action.copy(),
                    "replay_transition_audit": dict(replay_transition_audit),
                    "safeguard_projection_audit_v720": dict(safeguard_projection_audit_v720),
                    "block_before_xy_m": block_before.copy(),
                    "block_after_xy_m": env.block_xy().copy(),
                    "filter_audit": dict(filter_audit),
                    "translation_audit": translation_audit,
                    "execution_guard_audit": dict(getattr(adapter, "last_guard_report", {})),
                    "shield_failure_audit": shield_failure_audit,
                    "post_step_causal_stabilizer_audit": dict(causal_stabilizer_audit),
                    "intentional_latched_hold": bool(False if shield_deadlock else intentional_latched_hold),
                    "action_feasible": bool(action_feasible),
                    "safety_violation": bool(transition_safety or invalid_contact),
                    "valid_contact": bool(valid_contact),
                    "invalid_contact": bool(invalid_contact),
                    "step_block_displacement_m": float(displacement),
                    "terminal": bool(transition_terminal),
                    "failure_terminal": bool(transition_failure),
                    "strict_success": bool(transition_strict),
                    "terminal_reason": str(terminal_reason),
                    "contact_telemetry": (None if telemetry is None else dict(telemetry)),
                    # Full 96-geometry physics arrays are retained only on an
                    # invalid-contact row.  This keeps ordinary diagnostics
                    # compact while preserving exact causal evidence.
                    "invalid_contact_physics_trace": (
                        None if not invalid_contact or info is None else info["physics_substep_contact_v1"]
                    ),
                }
            )
        rows["neutral_state"].append(goal_neutral_privileged_state_v43(privileged))
        rows["next_neutral_state"].append(goal_neutral_privileged_state_v43(next_privileged))
        rows["achieved_goal"].append(achieved)
        rows["next_achieved_goal"].append(next_achieved)
        rows["desired_goal"].append(desired)
        rows["action"].append(replay_action)
        rows["applied_action"].append(applied_action)
        rows["terminal"].append(transition_terminal)
        rows["failure_terminal"].append(transition_failure)
        rows["strict_success"].append(transition_strict)
        rows["action_feasible"].append(replay_action_feasible)
        rows["execution_action_feasible_v741"].append(action_feasible)
        rows["safety_violation"].append(transition_safety or invalid_contact)
        rows["execution_failure_terminal_v741"].append(transition_failure or shield_deadlock)
        rows["valid_contact"].append(valid_contact)
        rows["invalid_contact"].append(invalid_contact)
        rows["raw_contact_evidence_v646"].append(
            bool(
                causal_stabilizer_audit.get(
                    "raw_robot_block_contact_this_step",
                    False,
                )
            )
        )
        rows["step_block_displacement_m"].append(displacement)
        rows["projection_l2"].append(replay_projection_l2)
        rows["safeguard_projected_action_v720"].append(safeguard_projected_action_v720)
        rows["safeguard_projection_valid_v720"].append(safeguard_projection_valid_v720)
        rows["strict_target_coverage"].append(
            float(privileged[_PRIVILEGED_SLICES_V43["strict_target_coverage"]][0])
        )
        rows["next_strict_target_coverage"].append(
            float(next_privileged[_PRIVILEGED_SLICES_V43["strict_target_coverage"]][0])
        )
        maximum_target_coverage = max(
            maximum_target_coverage,
            float(rows["next_strict_target_coverage"][-1]),
        )
        rows["strict_hold_fraction"].append(
            float(privileged[_PRIVILEGED_SLICES_V43["strict_success_streak_over_hold_steps"]][0])
        )
        rows["next_strict_hold_fraction"].append(
            float(next_privileged[_PRIVILEGED_SLICES_V43["strict_success_streak_over_hold_steps"]][0])
        )
        rows["episode_step"].append(episode_step)
        valid_contact_steps += int(valid_contact)
        invalid_contact_steps += int(invalid_contact)
        ik_failure_steps += int(not action_feasible)
        safety_steps += int(transition_safety or invalid_contact)
        strict_success = strict_success or transition_strict
        failure_terminal = failure_terminal or transition_failure or shield_deadlock
        if transition_terminal:
            break

    final_distance = float(env.distance_to_target())
    final_target_coverage = float(env.block_target_coverage())
    final_strict_hold_fraction = float(
        np.clip(
            env._strict_success_streak / max(int(env.realism_config.strict_success_hold_steps), 1),
            0.0,
            1.0,
        )
    )
    realism_reset = dict(env.episode_domain.get("realism_v7", {}))
    taskframe_reset = dict(env.episode_domain.get("stock_gripper_taskframe_reset_v12", {}))
    initial_tip_gap = reset_audit.get(
        "initial_tool_block_tip_gap_m",
        taskframe_reset.get("selected_minimum_tip_block_signed_distance_m"),
    )
    episode_arrays = {name: np.asarray(value) for name, value in rows.items()}
    acquisition_motion_audit = acquisition_rollout_audit_v642(
        episode_arrays["neutral_state"],
        episode_arrays["next_neutral_state"],
        episode_arrays["desired_goal"],
        episode_arrays["valid_contact"],
        episode_arrays["action_feasible"],
        episode_arrays["action"],
        episode_arrays["applied_action"],
    )
    episode = episode_arrays if collect_replay else None
    record = {
        "format": GOAL_CONDITIONED_HER_EVALUATION_FORMAT_V43,
        "requested_seed": requested_seed,
        "selected_seed": selected_seed,
        "reset_attempt_index": int(reset_audit["selected_attempt_index"]),
        "task_aligned_privileged_reset": bool(
            reset_audit.get(
                "task_aligned_privileged_reset",
                realism_reset.get("task_aligned_privileged_reset", True),
            )
        ),
        "task_independent_final_home_reset": bool(
            reset_audit.get("task_independent_final_home_reset", False)
        ),
        "initial_tool_block_xy_distance_m": reset_audit.get("initial_tool_block_xy_distance_m"),
        "policy_must_learn_visual_approach": bool(
            reset_audit.get("policy_must_learn_visual_approach", False)
        ),
        "deployment_reset_equivalent": bool(
            reset_audit.get(
                "deployment_reset_equivalent",
                taskframe_reset.get("deployment_reset_equivalent", False),
            )
        ),
        "curriculum_reset_approach_actions": int(
            reset_audit.get(
                "curriculum_reset_approach_actions",
                taskframe_reset.get("curriculum_reset_approach_actions", 0),
            )
        ),
        "initial_tool_block_tip_gap_m": (None if initial_tip_gap is None else float(initial_tip_gap)),
        "rows": len(rows["terminal"]),
        "initial_block_target_distance_m": initial_distance,
        "initial_target_coverage": initial_target_coverage,
        "final_block_target_distance_m": final_distance,
        "final_target_coverage": final_target_coverage,
        "final_strict_hold_fraction": final_strict_hold_fraction,
        "minimum_tool_precontact_distance_m": (minimum_tool_precontact_distance),
        "maximum_precontact_face_alignment": (maximum_precontact_face_alignment),
        "maximum_target_coverage": maximum_target_coverage,
        "net_target_progress_m": initial_distance - final_distance,
        "obstacle_present": bool(env.obstacle_enabled),
        "stress_condition": False,
        "strict_success": strict_success,
        "failure_terminal": failure_terminal,
        "terminal_reason": terminal_reason,
        "valid_contact_steps": valid_contact_steps,
        "invalid_contact_steps": invalid_contact_steps,
        "ik_failure_steps": ik_failure_steps,
        "safety_steps": safety_steps,
        "deterministic_policy": deterministic,
        "random_action_probability": random_action_probability,
        "policy_action_filter_active": policy_action_filter is not None,
        "policy_action_selector_active": policy_action_selector is not None,
        "post_step_causal_stabilizer_active": (post_step_causal_stabilizer is not None),
        "policy_action_selector_intervention_steps": (policy_selector_intervention_steps),
        "policy_action_selector_mean_conservative_q_improvement": (
            None if not policy_selector_q_improvements else float(np.mean(policy_selector_q_improvements))
        ),
        "policy_action_filter_intervention_steps": policy_filter_intervention_steps,
        "task_state_machine_override_steps": task_state_machine_override_steps,
        "policy_action_filter_mean_selected_risk_probability": (
            None if not policy_filter_risk_probabilities else float(np.mean(policy_filter_risk_probabilities))
        ),
        "policy_action_filter_mean_selected_task_action_scale": (
            None if not policy_filter_selected_scales else float(np.mean(policy_filter_selected_scales))
        ),
        "intentional_latched_hold_steps": intentional_latched_hold_steps,
        "intentional_terminal_release_steps": (intentional_terminal_release_steps),
        "replay_action_semantics": (
            "pre_filter_proposed_action_in_filtered_mdp"
            if policy_action_filter is not None
            else "selected_action_without_policy_filter"
        ),
        "replay_prefilter_action_steps": replay_prefilter_action_steps,
        "replay_clean_proposed_execution_steps": (replay_clean_proposed_execution_steps),
        "policy_action_filter_mean_intervention_l2": (
            None
            if not policy_filter_intervention_l2_values
            else float(np.mean(policy_filter_intervention_l2_values))
        ),
        "safeguard_projection_valid_steps_v720": int(len(safeguard_projection_l2_values_v720)),
        "safeguard_projection_mean_intervention_l2_v720": (
            float(np.mean(safeguard_projection_l2_values_v720))
            if safeguard_projection_l2_values_v720
            else 0.0
        ),
        "live_effect_mean_mismatch_l2_v720": (
            float(np.mean(actual_effect_projection_l2_values_v720))
            if actual_effect_projection_l2_values_v720
            else 0.0
        ),
        "live_post_step_effect_excluded_from_v720_projection_penalty": True,
        "safety_filter_treated_as_environment": policy_action_filter is not None,
        "her_success_claimed": False,
        "expert_calls": 0,
        "behavior_cloning_steps": 0,
        "bulk_vla_data_use_allowed": False,
        "production_admission": False,
        "acquisition_motion_audit_v642": acquisition_motion_audit,
    }
    record["automatic_failure_audit_v604"] = _automatic_failure_audit_v604(record)
    return record, episode


def _evaluate_v43(
    environment_config: RealisticEnvV10Config,
    action_config: StockGripperTaskSpaceActionConfigV12,
    scene_path: Path,
    bundle: GoalConditionedHerSACBundleV43,
    *,
    seed_base: int,
    episodes: int,
    maximum_steps: int,
    policy_action_filter: Callable[
        [np.ndarray, np.ndarray, np.ndarray],
        tuple[np.ndarray, dict[str, Any]],
    ]
    | None = None,
    policy_action_selector: Callable[
        [np.ndarray, np.ndarray, np.ndarray],
        tuple[np.ndarray, dict[str, Any]],
    ]
    | None = None,
    episode_progress_callback: Callable[[int, dict[str, Any]], None] | None = None,
    task_independent_home_reset: bool = False,
) -> dict[str, Any]:
    env = RealisticEdgeArmEnvV10(
        environment_config,
        seed=seed_base,
        model_scene_path=scene_path,
    )
    if task_independent_home_reset:
        adapter = StockGripperHomeTaskFrameAdapterV597(env, action_config)
        episode_reset = reset_stock_home_taskframe_episode_v597
    else:
        adapter = StockGripperTaskFrameAdapterV22(env, action_config)
        episode_reset = reset_stock_taskframe_episode_v22
    records = []
    for index in range(episodes):
        record, _episode = _run_episode_v43(
            env,
            adapter,
            bundle,
            requested_seed=seed_base + index,
            maximum_steps=maximum_steps,
            deterministic=True,
            random_action_probability=0.0,
            action_seed=seed_base ^ (index + 0x43E),
            collect_replay=False,
            policy_action_filter=policy_action_filter,
            policy_action_selector=policy_action_selector,
            episode_reset=episode_reset,
        )
        records.append(record)
        if episode_progress_callback is not None:
            episode_progress_callback(index + 1, record)
    success_count = sum(int(row["strict_success"]) for row in records)
    return {
        "format": GOAL_CONDITIONED_HER_EVALUATION_FORMAT_V43,
        "created_at_utc": _utc_now(),
        "seed_base": seed_base,
        "episode_count": episodes,
        "strict_success_count": success_count,
        "strict_success_rate": success_count / episodes,
        "contact_episode_count": sum(int(row["valid_contact_steps"] > 0) for row in records),
        "failure_episode_count": sum(int(row["failure_terminal"]) for row in records),
        "mean_net_target_progress_m": float(np.mean([row["net_target_progress_m"] for row in records])),
        "mean_final_block_target_distance_m": float(
            np.mean([row["final_block_target_distance_m"] for row in records])
        ),
        "episodes": records,
        "exact_three_second_environment_success_only": True,
        "her_successes_in_numerator": 0,
        "task_independent_home_reset": task_independent_home_reset,
        "production_admission": False,
    }


def _checkpoint_payload(
    bundle: GoalConditionedHerSACBundleV43,
    config: GoalConditionedHerSACConfigV43,
    replay: GoalConditionedHerReplayV43,
    *,
    run_plan_sha256: str,
    phase: str,
    online_episode_index: int,
) -> dict[str, Any]:
    return {
        "format": GOAL_CONDITIONED_HER_CHECKPOINT_FORMAT_V43,
        "algorithm_format": GOAL_CONDITIONED_HER_SAC_FORMAT_V43,
        "created_at_utc": _utc_now(),
        "phase": phase,
        "online_episode_index": online_episode_index,
        "update_index": bundle.update_index,
        "config": asdict(config),
        "run_plan_sha256": run_plan_sha256,
        "actor_state_dict": bundle.actor.state_dict(),
        "critic_state_dict": bundle.critic.state_dict(),
        "target_critic_state_dict": bundle.target_critic.state_dict(),
        "feasibility_state_dict": bundle.feasibility.state_dict(),
        "actor_optimizer_state_dict": bundle.actor_optimizer.state_dict(),
        "critic_optimizer_state_dict": bundle.critic_optimizer.state_dict(),
        "feasibility_optimizer_state_dict": bundle.feasibility_optimizer.state_dict(),
        "replay_manifest": replay.manifest(),
        "simulator_privileged_actor": True,
        "deployable_visual_policy": False,
        "expert_calls": 0,
        "behavior_cloning_steps": 0,
        "successful_demonstration_admission": False,
        "production_admission": False,
    }


def _restore_checkpoint(
    bundle: GoalConditionedHerSACBundleV43,
    path: Path,
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if (
        type(payload) is not dict
        or payload.get("format") != GOAL_CONDITIONED_HER_CHECKPOINT_FORMAT_V43
        or payload.get("algorithm_format") != GOAL_CONDITIONED_HER_SAC_FORMAT_V43
    ):
        raise ValueError("V43 resume checkpoint identity changed")
    bundle.actor.load_state_dict(payload["actor_state_dict"], strict=True)
    bundle.critic.load_state_dict(payload["critic_state_dict"], strict=True)
    bundle.target_critic.load_state_dict(payload["target_critic_state_dict"], strict=True)
    bundle.feasibility.load_state_dict(payload["feasibility_state_dict"], strict=True)
    bundle.actor_optimizer.load_state_dict(payload["actor_optimizer_state_dict"])
    bundle.critic_optimizer.load_state_dict(payload["critic_optimizer_state_dict"])
    bundle.feasibility_optimizer.load_state_dict(payload["feasibility_optimizer_state_dict"])
    bundle.update_index = int(payload["update_index"])
    return payload


def run_goal_conditioned_her_sac_v43(
    *,
    output_dir: Path,
    collection_plan: Path,
    seed_replay_h5: Sequence[Path],
    initialization_seed: int,
    sampling_seed_base: int,
    online_seed_base: int,
    evaluation_seed_base: int,
    pretrain_updates: int,
    online_episodes: int,
    updates_per_episode: int,
    evaluation_every_episodes: int,
    evaluation_episodes: int,
    maximum_episode_steps: int,
    initial_random_action_probability: float,
    final_random_action_probability: float,
    device: str,
    config: GoalConditionedHerSACConfigV43 | None = None,
    resume_checkpoint: Path | None = None,
    resume_replay_npz: Path | None = None,
    task_independent_home_reset: bool = False,
    fresh_replay: bool = False,
    training_curriculum_stage: int | None = None,
    colored_random_exploration: (AxisScaledColoredExplorationConfigV605 | None) = None,
) -> dict[str, Any]:
    for name, value in (
        ("pretrain_updates", pretrain_updates),
        ("online_episodes", online_episodes),
        ("updates_per_episode", updates_per_episode),
        ("evaluation_every_episodes", evaluation_every_episodes),
        ("evaluation_episodes", evaluation_episodes),
        ("maximum_episode_steps", maximum_episode_steps),
    ):
        if (
            type(value) is not int
            or value < 0
            or (name not in {"pretrain_updates", "online_episodes", "updates_per_episode"} and value < 1)
        ):
            raise ValueError(f"V43 {name} is invalid")
    if pretrain_updates + online_episodes < 1:
        raise ValueError("V43 run must perform training or online collection")
    probabilities = (initial_random_action_probability, final_random_action_probability)
    if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
        raise ValueError("V43 exploration probability is invalid")
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"V43 output already exists: {destination}")
    selected_config = config or GoalConditionedHerSACConfigV43()
    selected_config.validate()
    if type(task_independent_home_reset) is not bool:
        raise TypeError("V43 Home-reset selector must be boolean")
    if type(fresh_replay) is not bool:
        raise TypeError("V43 fresh-replay selector must be boolean")
    if training_curriculum_stage not in (None, 31, 32):
        raise ValueError("V43 training curriculum stage must be 31 or 32")
    if task_independent_home_reset and training_curriculum_stage is not None:
        raise ValueError("V43 Home reset and privileged contact curriculum are exclusive")
    if colored_random_exploration is not None:
        if type(colored_random_exploration) is not (AxisScaledColoredExplorationConfigV605):
            raise TypeError("V605 colored exploration config type changed")
        colored_random_exploration.validate()
    if fresh_replay and resume_replay_npz is not None:
        raise ValueError("V43 fresh replay cannot resume a persisted replay")
    if fresh_replay and seed_replay_h5:
        raise ValueError("V43 fresh replay cannot ingest seed replay files")
    if fresh_replay and pretrain_updates:
        raise ValueError("V43 empty fresh replay cannot run pretraining")
    if task_independent_home_reset and not (selected_config.home_acquisition_reward_active):
        raise ValueError("V43 Home reset requires the V598 acquisition reward")
    resolved_device = _resolve_device(device)
    source_plan, environment_config, action_config, scene_path = _load_collection_contract(collection_plan)
    requested_curriculum_stage = (
        32
        if task_independent_home_reset
        else (30 if training_curriculum_stage is None else training_curriculum_stage)
    )
    if environment_config.reverse_curriculum_stage_v26 != requested_curriculum_stage:
        environment_config = replace(
            environment_config,
            reverse_curriculum_stage_v26=requested_curriculum_stage,
            strict_success_hold_steps=90,
        )
    if (
        task_independent_home_reset or training_curriculum_stage is not None
    ) and environment_config.max_steps < maximum_episode_steps:
        environment_config = replace(
            environment_config,
            max_steps=maximum_episode_steps,
        )
    if fresh_replay:
        replay = GoalConditionedHerReplayV43(selected_config)
    elif resume_replay_npz is None:
        replay = GoalConditionedHerReplayV43.from_h5(seed_replay_h5, selected_config)
    else:
        replay = GoalConditionedHerReplayV43.load_npz(resume_replay_npz, selected_config)
    bundle = initialize_goal_conditioned_her_sac_v43(
        initialization_seed,
        device=resolved_device,
        config=selected_config,
    )
    resume_payload = None
    if resume_checkpoint is not None:
        resume_payload = _restore_checkpoint(bundle, resume_checkpoint)

    source_file = Path(__file__).resolve()
    run_plan = {
        "format": GOAL_CONDITIONED_HER_TRAIN_RUN_FORMAT_V43,
        "algorithm_format": GOAL_CONDITIONED_HER_SAC_FORMAT_V43,
        "created_at_utc": _utc_now(),
        "output_dir": str(destination),
        "resolved_device": resolved_device,
        "configuration": asdict(selected_config),
        "pretrain_updates": pretrain_updates,
        "online_episodes": online_episodes,
        "updates_per_episode": updates_per_episode,
        "evaluation_every_episodes": evaluation_every_episodes,
        "evaluation_episodes": evaluation_episodes,
        "maximum_episode_steps": maximum_episode_steps,
        "initial_random_action_probability": initial_random_action_probability,
        "final_random_action_probability": final_random_action_probability,
        "initialization_seed": initialization_seed,
        "sampling_seed_base": sampling_seed_base,
        "online_seed_base": online_seed_base,
        "evaluation_seed_base": evaluation_seed_base,
        "source_collection_plan_path": str(Path(collection_plan).resolve()),
        "source_collection_plan_sha256": source_plan["run_plan_sha256"],
        "seed_replay_h5": [str(Path(path).resolve()) for path in seed_replay_h5],
        "seed_replay_sha256": [sha256_file_v1(path) for path in seed_replay_h5],
        "initial_replay_manifest": replay.manifest(),
        "scene_path": str(scene_path),
        "scene_sha256": sha256_file_v1(scene_path),
        "environment_config": asdict(environment_config),
        "action_adapter_config": asdict(action_config),
        "execution_kernel": StockGripperRolloutKernelV22.format,
        "source_type": SOURCE_TYPE,
        "simulator_privileged_actor": True,
        "visual_actor_training": False,
        "act_training": False,
        "future_her_learning_only": True,
        "her_success_admitted_as_demonstration": False,
        "strict_success_requires_environment_three_second_hold": True,
        "failed_episodes_persisted_and_reused": True,
        "resume_checkpoint": None if resume_checkpoint is None else str(Path(resume_checkpoint).resolve()),
        "resume_replay_npz": None if resume_replay_npz is None else str(Path(resume_replay_npz).resolve()),
        "resume_update_index": None if resume_payload is None else int(resume_payload["update_index"]),
        "task_independent_home_reset": task_independent_home_reset,
        "training_curriculum_stage": training_curriculum_stage,
        "training_curriculum_object_target_distance_is_full_task": bool(training_curriculum_stage == 32),
        "training_curriculum_object_target_distance_is_bridge_only": bool(training_curriculum_stage == 31),
        "training_curriculum_arm_reset_is_privileged": bool(training_curriculum_stage in (31, 32)),
        "training_curriculum_trajectories_admitted_as_final_home_data": False,
        "colored_random_exploration": (
            None if colored_random_exploration is None else asdict(colored_random_exploration)
        ),
        "colored_random_exploration_uses_task_rule_or_expert_path": False,
        "fresh_replay": fresh_replay,
        "home_acquisition_reward_active": (selected_config.home_acquisition_reward_active),
        "home_acquisition_her_rows_suppressed": True,
        "home_acquisition_auxiliary_goal_is_not_action_path": True,
        "source_hashes": {
            "edgearm/train_goal_conditioned_her_sac_v43.py": sha256_file_v1(source_file),
            "edgearm/goal_conditioned_her_sac_v43.py": sha256_file_v1(
                source_file.with_name("goal_conditioned_her_sac_v43.py")
            ),
            "edgearm/stock_gripper_taskframe_v22.py": sha256_file_v1(
                source_file.with_name("stock_gripper_taskframe_v22.py")
            ),
        },
        "expert_calls": 0,
        "behavior_cloning_steps": 0,
        "physical_samples": 0,
        "production_admission": False,
    }
    run_plan["run_plan_sha256"] = canonical_sha256_v1(run_plan)
    destination.mkdir(parents=True)
    (destination / "checkpoints").mkdir()
    (destination / "evaluations").mkdir()
    _atomic_json(destination / "run_plan.json", run_plan)
    _atomic_json(
        destination / "run_state.json",
        {
            "status": "running",
            "phase": "offline_pretraining",
            "update_index": bundle.update_index,
            "online_episode_index": 0,
            "replay_transition_count": replay.transition_count,
            "updated_at_utc": _utc_now(),
        },
    )

    latest_metrics: dict[str, Any] | None = None
    evaluations: list[dict[str, Any]] = []
    best_training_evaluation: dict[str, Any] | None = None
    best_training_checkpoint: Path | None = None
    best_training_score: tuple[int, float, int, int] | None = None
    try:
        for local_update in range(1, pretrain_updates + 1):
            batch = replay.sample(
                batch_size=selected_config.batch_size,
                seed=sampling_seed_base + bundle.update_index,
            )
            metrics = goal_conditioned_her_sac_update_v43(bundle, batch, selected_config)
            latest_metrics = asdict(metrics)
            latest_metrics["phase"] = "offline_pretraining"
            latest_metrics["local_pretrain_update"] = local_update
            _append_jsonl(destination / "metrics.jsonl", latest_metrics)
            if local_update % 50 == 0 or local_update == pretrain_updates:
                _atomic_json(
                    destination / "run_state.json",
                    {
                        "status": "running",
                        "phase": "offline_pretraining",
                        "update_index": bundle.update_index,
                        "local_pretrain_update": local_update,
                        "pretrain_updates": pretrain_updates,
                        "online_episode_index": 0,
                        "replay_transition_count": replay.transition_count,
                        "latest_metrics": latest_metrics,
                        "updated_at_utc": _utc_now(),
                    },
                )

        pretrain_checkpoint = destination / "checkpoints" / "after_pretrain.pt"
        _atomic_torch_save(
            pretrain_checkpoint,
            _checkpoint_payload(
                bundle,
                selected_config,
                replay,
                run_plan_sha256=run_plan["run_plan_sha256"],
                phase="after_pretrain",
                online_episode_index=0,
            ),
        )

        env = RealisticEdgeArmEnvV10(
            environment_config,
            seed=online_seed_base,
            model_scene_path=scene_path,
        )
        if task_independent_home_reset:
            adapter = StockGripperHomeTaskFrameAdapterV597(env, action_config)
            episode_reset = reset_stock_home_taskframe_episode_v597
        else:
            adapter = StockGripperTaskFrameAdapterV22(env, action_config)
            episode_reset = reset_stock_taskframe_episode_v22
        for online_index in range(1, online_episodes + 1):
            fraction = (online_index - 1) / max(online_episodes - 1, 1)
            random_probability = initial_random_action_probability + fraction * (
                final_random_action_probability - initial_random_action_probability
            )
            record, episode = _run_episode_v43(
                env,
                adapter,
                bundle,
                requested_seed=online_seed_base + online_index - 1,
                maximum_steps=maximum_episode_steps,
                deterministic=False,
                random_action_probability=float(random_probability),
                action_seed=online_seed_base ^ (online_index * 0x43A11),
                collect_replay=True,
                episode_reset=episode_reset,
                colored_random_exploration=colored_random_exploration,
            )
            if episode is None:
                raise RuntimeError("V43 online collection lost its replay episode")
            replay.add_episode(
                episode,
                source=f"online_v43_episode_{online_index:06d}",
            )
            record["online_episode_index"] = online_index
            record["replay_transition_count_after_append"] = replay.transition_count
            _append_jsonl(destination / "online_episodes.jsonl", record)

            for local_update in range(1, updates_per_episode + 1):
                batch = replay.sample(
                    batch_size=selected_config.batch_size,
                    seed=sampling_seed_base + bundle.update_index,
                )
                metrics = goal_conditioned_her_sac_update_v43(bundle, batch, selected_config)
                latest_metrics = asdict(metrics)
                latest_metrics.update(
                    {
                        "phase": "online_training",
                        "online_episode_index": online_index,
                        "local_update_after_episode": local_update,
                    }
                )
                _append_jsonl(destination / "metrics.jsonl", latest_metrics)

            replay.save_npz(destination / "replay_latest.npz")
            checkpoint = destination / "checkpoints" / f"episode_{online_index:06d}.pt"
            _atomic_torch_save(
                checkpoint,
                _checkpoint_payload(
                    bundle,
                    selected_config,
                    replay,
                    run_plan_sha256=run_plan["run_plan_sha256"],
                    phase="online_training",
                    online_episode_index=online_index,
                ),
            )

            if online_index % evaluation_every_episodes == 0 or online_index == online_episodes:
                evaluation = _evaluate_v43(
                    environment_config,
                    action_config,
                    scene_path,
                    bundle,
                    seed_base=evaluation_seed_base,
                    episodes=evaluation_episodes,
                    maximum_steps=maximum_episode_steps,
                    task_independent_home_reset=(task_independent_home_reset),
                )
                evaluation["online_episode_index"] = online_index
                evaluation["update_index"] = bundle.update_index
                evaluation_path = destination / "evaluations" / f"episode_{online_index:06d}.json"
                _atomic_json(evaluation_path, evaluation)
                evaluations.append(evaluation)
                selection_score = _training_selection_score_v610(evaluation)
                if best_training_score is None or selection_score > best_training_score:
                    best_training_score = selection_score
                    best_training_evaluation = evaluation
                    best_training_checkpoint = checkpoint
                    _atomic_json(
                        destination / "best_training_selection.json",
                        {
                            "format": ("edgearm-v610-curriculum-checkpoint-selection-v1"),
                            "selection_score": list(selection_score),
                            "checkpoint": str(checkpoint),
                            "checkpoint_sha256": sha256_file_v1(checkpoint),
                            "evaluation": evaluation,
                            "production_admission": False,
                        },
                    )

            _atomic_json(
                destination / "run_state.json",
                {
                    "status": "running",
                    "phase": "online_training",
                    "update_index": bundle.update_index,
                    "online_episode_index": online_index,
                    "online_episodes": online_episodes,
                    "latest_online_episode": record,
                    "latest_metrics": latest_metrics,
                    "latest_evaluation": evaluations[-1] if evaluations else None,
                    "replay_transition_count": replay.transition_count,
                    "replay_episode_count": replay.episode_count,
                    "updated_at_utc": _utc_now(),
                },
            )
    except Exception as error:
        _atomic_json(
            destination / "failure.json",
            {
                "format": GOAL_CONDITIONED_HER_TRAIN_RUN_FORMAT_V43,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "update_index": bundle.update_index,
                "replay_manifest": replay.manifest(),
                "failed_at_utc": _utc_now(),
                "production_admission": False,
            },
        )
        _atomic_json(
            destination / "run_state.json",
            {
                "status": "failed",
                "phase": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "update_index": bundle.update_index,
                "replay_transition_count": replay.transition_count,
                "updated_at_utc": _utc_now(),
            },
        )
        raise

    final_checkpoint = destination / "checkpoints" / "final.pt"
    _atomic_torch_save(
        final_checkpoint,
        _checkpoint_payload(
            bundle,
            selected_config,
            replay,
            run_plan_sha256=run_plan["run_plan_sha256"],
            phase="complete",
            online_episode_index=online_episodes,
        ),
    )
    replay.save_npz(destination / "replay_final.npz")
    final_evaluation = evaluations[-1] if evaluations else None
    home_strict_gate_passed = False
    home_wilson_lower_bound = 0.0
    if final_evaluation is not None:
        evaluation_count = int(final_evaluation["episode_count"])
        evaluation_successes = int(final_evaluation["strict_success_count"])
        home_wilson_lower_bound = wilson_lower_bound_v26(
            evaluation_successes,
            evaluation_count,
        )
        evaluation_records = list(final_evaluation["episodes"])
        home_strict_gate_passed = bool(
            task_independent_home_reset
            and evaluation_count >= 48
            and float(final_evaluation["strict_success_rate"]) >= 0.80
            and home_wilson_lower_bound >= 0.65
            and all(
                row["task_aligned_privileged_reset"] is False
                and row["task_independent_final_home_reset"] is True
                and float(row["initial_target_coverage"]) == 0.0
                and float(row["initial_block_target_distance_m"]) >= MINIMUM_FULL_TASK_INITIAL_DISTANCE_M_V607
                and int(row["invalid_contact_steps"]) == 0
                and int(row["safety_steps"]) == 0
                and (
                    not bool(row["strict_success"])
                    or float(row["net_target_progress_m"]) >= MINIMUM_FULL_TASK_NET_PROGRESS_M_V607
                )
                for row in evaluation_records
            )
        )
    summary = {
        "format": GOAL_CONDITIONED_HER_TRAIN_RUN_FORMAT_V43,
        "status": "complete",
        "completed_at_utc": _utc_now(),
        "update_index": bundle.update_index,
        "online_episode_count": online_episodes,
        "replay_manifest": replay.manifest(),
        "final_checkpoint": str(final_checkpoint),
        "final_checkpoint_sha256": sha256_file_v1(final_checkpoint),
        "best_training_checkpoint": (
            None if best_training_checkpoint is None else str(best_training_checkpoint)
        ),
        "best_training_checkpoint_sha256": (
            None if best_training_checkpoint is None else sha256_file_v1(best_training_checkpoint)
        ),
        "best_training_evaluation": best_training_evaluation,
        "best_training_selection_is_production_admission": False,
        "final_evaluation": final_evaluation,
        "strict_success_count": (
            0 if final_evaluation is None else int(final_evaluation["strict_success_count"])
        ),
        "strict_success_rate": (
            0.0 if final_evaluation is None else float(final_evaluation["strict_success_rate"])
        ),
        "her_successes_counted_as_strict_success": 0,
        "successful_rl_data_generation_ready": (
            home_strict_gate_passed
            if task_independent_home_reset
            else False
            if training_curriculum_stage is not None
            else bool(final_evaluation is not None and final_evaluation["strict_success_count"] > 0)
        ),
        "training_curriculum_only": bool(training_curriculum_stage is not None),
        "training_curriculum_stage": training_curriculum_stage,
        "training_curriculum_trajectories_admitted_as_final_data": False,
        "home_strict_gate_passed": home_strict_gate_passed,
        "home_strict_gate_minimum_evaluation_episodes": 48,
        "home_strict_gate_minimum_success_rate": 0.80,
        "home_strict_gate_minimum_initial_object_target_distance_m": (
            MINIMUM_FULL_TASK_INITIAL_DISTANCE_M_V607
        ),
        "home_strict_gate_minimum_successful_net_progress_m": (MINIMUM_FULL_TASK_NET_PROGRESS_M_V607),
        "home_strict_gate_minimum_wilson_lower_bound": 0.65,
        "home_strict_gate_wilson_lower_bound": home_wilson_lower_bound,
        "bulk_multimodal_generation_started": False,
        "remaining_gate": (
            "48-task Home-start strict gate then wrist-multimodal bulk collection"
            if task_independent_home_reset
            else "transfer learned contact/transport skill to Home-start training"
            if training_curriculum_stage is not None
            else "held-out strict success then wrist-multimodal bulk collection"
        ),
        "expert_calls": 0,
        "behavior_cloning_steps": 0,
        "physical_samples": 0,
        "production_admission": False,
    }
    _atomic_json(destination / "summary.json", summary)
    _atomic_json(
        destination / "run_state.json",
        {
            "status": "complete",
            "phase": "complete",
            "update_index": bundle.update_index,
            "online_episode_index": online_episodes,
            "strict_success_count": summary["strict_success_count"],
            "strict_success_rate": summary["strict_success_rate"],
            "replay_transition_count": replay.transition_count,
            "updated_at_utc": _utc_now(),
        },
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--collection-plan", type=Path, required=True)
    parser.add_argument("--seed-replay-h5", type=Path, action="append", default=[])
    parser.add_argument("--initialization-seed", type=int, default=43_100_000)
    parser.add_argument("--sampling-seed-base", type=int, default=43_200_000)
    parser.add_argument("--online-seed-base", type=int, default=43_300_000)
    parser.add_argument("--evaluation-seed-base", type=int, default=43_400_000)
    parser.add_argument("--pretrain-updates", type=int, default=2_000)
    parser.add_argument("--online-episodes", type=int, default=40)
    parser.add_argument("--updates-per-episode", type=int, default=256)
    parser.add_argument("--evaluation-every-episodes", type=int, default=5)
    parser.add_argument("--evaluation-episodes", type=int, default=4)
    parser.add_argument("--maximum-episode-steps", type=int, default=360)
    parser.add_argument("--initial-random-action-probability", type=float, default=0.20)
    parser.add_argument("--final-random-action-probability", type=float, default=0.02)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--actor-infeasibility-coefficient", type=float, default=0.30)
    parser.add_argument("--projection-penalty", type=float, default=0.10)
    parser.add_argument("--infeasible-action-penalty", type=float, default=0.60)
    parser.add_argument("--target-settle-action-penalty", type=float, default=0.0)
    parser.add_argument("--target-settle-priority", type=float, default=0.0)
    parser.add_argument("--target-coverage-gain-bonus", type=float, default=1.0)
    parser.add_argument("--strict-hold-progress-bonus", type=float, default=2.0)
    parser.add_argument("--target-entry-priority", type=float, default=6.0)
    parser.add_argument("--successful-contact-credit-priority", type=float, default=4.0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--resume-replay-npz", type=Path)
    parser.add_argument("--task-independent-home-reset", action="store_true")
    parser.add_argument(
        "--training-curriculum-stage",
        type=int,
        choices=(31, 32),
        help=(
            "Training-only privileged gripper reset with a 15-19 cm object "
            "task; never admitted as final Home-start data."
        ),
    )
    parser.add_argument("--fresh-replay", action="store_true")
    parser.add_argument(
        "--colored-random-exploration",
        action="store_true",
        help="Use zero-mean axis-scaled AR(1) exploration instead of white noise.",
    )
    parser.add_argument("--colored-exploration-rho", type=float, default=0.95)
    parser.add_argument(
        "--colored-exploration-standard-deviation",
        type=float,
        nargs=3,
        metavar=("FORWARD", "LATERAL", "VERTICAL"),
        default=(0.65, 0.22, 0.12),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = GoalConditionedHerSACConfigV43(
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        phase_progress_reward_active=(
            args.task_independent_home_reset or args.training_curriculum_stage is not None
        ),
        phase_retention_reward_active=(
            args.task_independent_home_reset or args.training_curriculum_stage is not None
        ),
        home_acquisition_reward_active=(args.task_independent_home_reset),
        actor_infeasibility_coefficient=(args.actor_infeasibility_coefficient),
        projection_penalty=args.projection_penalty,
        infeasible_action_penalty=args.infeasible_action_penalty,
        target_settle_action_penalty=(args.target_settle_action_penalty),
        target_settle_priority=args.target_settle_priority,
        target_coverage_gain_bonus=args.target_coverage_gain_bonus,
        strict_hold_progress_bonus=args.strict_hold_progress_bonus,
        target_entry_priority=args.target_entry_priority,
        successful_contact_credit_priority=(args.successful_contact_credit_priority),
    )
    colored_exploration = (
        AxisScaledColoredExplorationConfigV605(
            standard_deviation=tuple(args.colored_exploration_standard_deviation),
            autoregressive_rho=args.colored_exploration_rho,
        )
        if args.colored_random_exploration
        else None
    )
    summary = run_goal_conditioned_her_sac_v43(
        output_dir=args.output_dir,
        collection_plan=args.collection_plan,
        seed_replay_h5=args.seed_replay_h5,
        initialization_seed=args.initialization_seed,
        sampling_seed_base=args.sampling_seed_base,
        online_seed_base=args.online_seed_base,
        evaluation_seed_base=args.evaluation_seed_base,
        pretrain_updates=args.pretrain_updates,
        online_episodes=args.online_episodes,
        updates_per_episode=args.updates_per_episode,
        evaluation_every_episodes=args.evaluation_every_episodes,
        evaluation_episodes=args.evaluation_episodes,
        maximum_episode_steps=args.maximum_episode_steps,
        initial_random_action_probability=args.initial_random_action_probability,
        final_random_action_probability=args.final_random_action_probability,
        device=args.device,
        config=config,
        resume_checkpoint=args.resume_checkpoint,
        resume_replay_npz=args.resume_replay_npz,
        task_independent_home_reset=args.task_independent_home_reset,
        fresh_replay=args.fresh_replay,
        training_curriculum_stage=args.training_curriculum_stage,
        colored_random_exploration=colored_exploration,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GOAL_CONDITIONED_HER_EVALUATION_FORMAT_V43",
    "GOAL_CONDITIONED_HER_TRAIN_RUN_FORMAT_V43",
    "AxisScaledColoredExplorationConfigV605",
    "AxisScaledColoredExplorerV605",
    "run_goal_conditioned_her_sac_v43",
]
