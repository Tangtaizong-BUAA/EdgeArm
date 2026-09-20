"""Visual asymmetric PPO for the ``sim_rl_scratch`` EdgeArm source.

The actor receives only causal deployable observations: low-resolution RGB,
reported joint state, previously executed actions, view/history masks and a
task id.  The critic receives the simulator privileged effect state.  During
training only, a geometry head is supervised with simulator-relative geometry;
those labels are never actor inputs and are not required for deployment.  No
expert, behavior-cloning checkpoint or residual baseline is accepted.

This first implementation is deliberately a compact recurrent data-generator
baseline.  It is not the final sparse 4D VLA policy and every artifact it
writes remains diagnostic until the declared replay and blind gates pass.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

import h5py
import mujoco
import numpy as np
import torch
from torch import nn

from .keyboard_cartesian_runtime_v1 import (
    KeyboardCartesianConfig,
    NearestBlockFaceTracker,
    PositionFaceAlignedIK,
    slew_parallel_plane_normal,
    table_angle_face_normal,
)
from .joint_path_planner_v1 import JointPathPlannerConfig
from .physical_expert_v12 import (
    PHYSICAL_EXPERT_V12_CONTACT_PART_CENTRAL_SIDE_MARGIN_M,
)
from .ppo_utils_v1 import (
    compute_gae_termination_truncation_v1,
    finite_module_parameters_v1,
    sample_squashed_gaussian_v1,
    squashed_gaussian_log_prob_v1,
    state_dict_sha256_v1,
)
from .privileged_effect_state_v1 import (
    PRIVILEGED_EFFECT_STATE_DIM,
    PRIVILEGED_EFFECT_STATE_SCHEMA_SHA256,
    build_privileged_effect_state_v1,
    privileged_effect_state_slices_v1,
)
from .scratch_ppo_v6_candidate import (
    ScratchPotentialRewardV6Candidate,
    ScratchSafetyEvidenceV6Candidate,
)
from .side_contact_ik_v1 import SideContactIKConfig, SideContactIKPlannerV1
from .sim2real_env_v10 import (
    RealisticEdgeArmEnvV10,
    V10SafetyFilterInfeasible,
)
from .stock_gripper_action_guard_v2 import (
    STOCK_GRIPPER_ACTION_GUARD_FORMAT_V2,
    StockGripperActionGuardConfigV2,
    StockGripperActionGuardV2,
)
from .stock_gripper_action_guard_v3 import (
    STOCK_GRIPPER_ACTION_GUARD_FORMAT_V3,
    LatchedAbsoluteTargetInfeasibleV3,
    StockGripperActionGuardConfigV3,
    StockGripperActionGuardV3,
)


SOURCE_TYPE = "sim_rl_scratch"
POLICY_FORMAT = "edgearm-asymmetric-multiview-stock-taskframe-ar-geometry-aux-ppo-v13"
ROLLOUT_FORMAT = "edgearm-sim-rl-scratch-online-stock-taskframe-ar-rollout-v13"
CHECKPOINT_FORMAT = "edgearm-asymmetric-multiview-stock-taskframe-ar-ppo-checkpoint-v13"
H5_FORMAT = "edgearm-sim-rl-scratch-online-stock-taskframe-ar-trajectory-h5-v13"
EVALUATION_FORMAT = "edgearm-asymmetric-multiview-stock-taskframe-ar-heldout-evaluation-v13"
SHIELD_TERMINAL_REASON_V13 = "v13_action_shield_terminal:no_guard_verified_recovery"
CAUSAL_VISUAL_INFERENCE_CACHE_FORMAT_V24 = (
    "edgearm-causal-selected-view-overlap-cache-v24"
)
ACTOR_ARCHITECTURE = (
    "shared-spatial-cnn-selected-view-gating-masked-grucell-taskframe-flz-"
    "causal-ar1-latent-action-relative-geometry-aux-guarded-recovery-v13"
)
CRITIC_ARCHITECTURE = "privileged-effect-mlp-163-256-256-v1"
TASK_INSTRUCTION_EN = "push the blue block into the green target region"
TASK_INSTRUCTION_ZH = "将蓝色方块推到绿色目标区域"
JOINT_ACTION_DIM = 6
POLICY_ACTION_DIM = 3
VISUAL_GEOMETRY_TARGET_DIM = 5
VISUAL_GEOMETRY_TARGET_SCHEMA = (
    "block_minus_tool_xyz_over_0.25_0.25_0.15__target_minus_block_xy_over_0.25_0.25_clip_-2_2-v1"
)
VISUAL_GEOMETRY_NORMALIZATION = np.asarray(
    [0.25, 0.25, 0.15, 0.25, 0.25],
    dtype=np.float32,
)
# Backward-compatible name for deployable joint-action history consumers.
ACTION_DIM = JOINT_ACTION_DIM
JOINT_STATE_DIM = 12
VIEW_NAMES = ("wrist", "front", "angled", "overhead")
CAMERA_NAMES = (
    "edgearm_wrist",
    "edgearm_front",
    "edgearm_angled",
    "edgearm_overhead",
)
RESET_RETRY_STRIDE_V11 = 10_003
MAX_RESET_ATTEMPTS_V11 = 8
RESET_RETRY_STRIDE_V12 = 10_003
MAX_RESET_ATTEMPTS_V12 = 8
RESET_RETRY_STRIDE_V13 = 10_003
MAX_RESET_ATTEMPTS_V13 = 8
MAX_CURRICULUM_RESET_ATTEMPTS_V16 = 16


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256_v1(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def sha256_file_v1(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _module_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration as error:  # pragma: no cover - all formal modules have parameters
        raise ValueError("module has no parameters") from error


def _torch_generator(device: torch.device, seed: int) -> torch.Generator | None:
    if device.type == "mps":
        torch.manual_seed(seed)
        if hasattr(torch, "mps"):
            torch.mps.manual_seed(seed)
        return None
    generator = torch.Generator(device=device.type)
    generator.manual_seed(seed)
    return generator


def autoregressive_action_distribution_v1(
    base: torch.distributions.Normal,
    previous_pre_tanh: torch.Tensor,
    rho: float,
) -> torch.distributions.Normal:
    """Build a causal AR(1) latent-action policy with exact PPO likelihood.

    ``previous_pre_tanh`` is an explicit policy state stored in every rollout
    row.  The conditional mean ``(1-rho)*mu + rho*u[t-1]`` preserves the
    actor's stationary mean while the innovation scale preserves variance for
    a stationary zero-mean policy.  PPO recomputes this same conditional
    density instead of treating post-hoc action smoothing as independent.
    """

    if type(base) is not torch.distributions.Normal:
        raise TypeError("autoregressive action policy requires an exact Normal base")
    if previous_pre_tanh.shape != base.loc.shape:
        raise ValueError("previous_pre_tanh must match the base action shape")
    if not torch.isfinite(previous_pre_tanh).all():
        raise ValueError("previous_pre_tanh must be finite")
    if isinstance(rho, bool) or not isinstance(rho, (int, float)):
        raise TypeError("action autoregressive rho must be numeric")
    rho_value = float(rho)
    if not 0.0 <= rho_value < 0.99:
        raise ValueError("action autoregressive rho must be in [0,0.99)")
    mean = (1.0 - rho_value) * base.loc + rho_value * previous_pre_tanh
    innovation_scale = base.scale * float(np.sqrt(1.0 - rho_value * rho_value))
    return torch.distributions.Normal(mean, innovation_scale)


def autoregressive_applied_feedback_state_v21(
    applied_task_action: np.ndarray,
) -> np.ndarray:
    """Map the action that IK actually admitted back into the AR latent state."""

    applied = np.asarray(applied_task_action, dtype=np.float32)
    if (
        applied.shape != (POLICY_ACTION_DIM,)
        or not np.all(np.isfinite(applied))
        or np.any(np.abs(applied) > 1.0 + 1.0e-6)
    ):
        raise ValueError("execution-aware AR feedback action is invalid")
    bounded = np.clip(applied, -1.0 + 1.0e-6, 1.0 - 1.0e-6)
    result = np.arctanh(bounded).astype(np.float32)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("execution-aware AR feedback state is non-finite")
    return result


def autoregressive_ik_guarded_feedback_state_v21(
    proposed_pre_tanh: np.ndarray,
    *,
    ik_converged: bool,
) -> np.ndarray:
    """Keep an admitted latent action, but clear an action IK rejected."""

    proposed = np.asarray(proposed_pre_tanh, dtype=np.float32)
    if proposed.shape != (POLICY_ACTION_DIM,) or not np.all(np.isfinite(proposed)):
        raise ValueError("IK-guarded AR proposal is invalid")
    if type(ik_converged) is not bool:
        raise TypeError("IK-guarded AR convergence flag must be boolean")
    return proposed.copy() if ik_converged else np.zeros(POLICY_ACTION_DIM, dtype=np.float32)


def visual_geometry_target_from_privileged_v1(state: np.ndarray) -> np.ndarray:
    """Extract normalized training-only relative geometry without expert actions."""

    source = np.asarray(state, dtype=np.float32)
    if source.ndim < 1 or source.shape[-1] != PRIVILEGED_EFFECT_STATE_DIM:
        raise ValueError(f"visual geometry target requires [...,{PRIVILEGED_EFFECT_STATE_DIM}] state")
    if not np.all(np.isfinite(source)):
        raise ValueError("visual geometry source must be finite")
    slices = privileged_effect_state_slices_v1()
    block_xyz = source[..., slices["block_pose_xyz_quaternion_wxyz"]][..., :3]
    target_xy = source[..., slices["target_xy_m"]]
    tool_xyz = source[..., slices["tool_pose_position_rotation"]][..., :3]
    relative = np.concatenate(
        (block_xyz - tool_xyz, target_xy - block_xyz[..., :2]),
        axis=-1,
    )
    normalized = np.clip(relative / VISUAL_GEOMETRY_NORMALIZATION, -2.0, 2.0)
    result = normalized.astype(np.float32, copy=False)
    if result.shape[-1] != VISUAL_GEOMETRY_TARGET_DIM or not np.all(np.isfinite(result)):
        raise RuntimeError("visual geometry target construction failed")
    return result


@dataclass(frozen=True)
class AsymmetricMultiViewPPOConfigV1:
    history_steps: int = 4
    image_height: int = 64
    image_width: int = 64
    rollout_steps: int = 128
    update_epochs: int = 4
    batch_size: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.20
    value_clip_ratio: float = 0.20
    learning_rate: float = 2.0e-4
    entropy_coef: float = 0.004
    value_coef: float = 0.5
    visual_geometry_auxiliary_coef: float = 0.25
    action_autoregressive_rho: float = 0.80
    max_grad_norm: float = 0.5
    target_kl: float = 0.025
    obstacle_probability: float = 0.50
    stress_probability: float = 0.30
    auxiliary_view_dropout_probability: float = 0.25
    seed: int = 11_240_000

    def validate(self) -> None:
        integer_fields = (
            "history_steps",
            "image_height",
            "image_width",
            "rollout_steps",
            "update_epochs",
            "batch_size",
            "seed",
        )
        for name in integer_fields:
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.history_steps < 1:
            raise ValueError("history_steps must be positive")
        if self.image_height < 32 or self.image_width < 32:
            raise ValueError("online visual resolution must be at least 32x32")
        if self.rollout_steps < 1 or self.update_epochs < 1 or self.batch_size < 1:
            raise ValueError("rollout/update/batch sizes must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field.name} must be finite numeric")
            if not np.isfinite(value):
                raise ValueError(f"{field.name} must be finite")
        if not 0.0 < self.gamma <= 1.0 or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gamma or gae_lambda is invalid")
        if not 0.0 < self.clip_ratio < 1.0:
            raise ValueError("clip_ratio must be in (0,1)")
        if not 0.0 < self.value_clip_ratio < 1.0:
            raise ValueError("value_clip_ratio must be in (0,1)")
        if self.learning_rate <= 0.0 or self.max_grad_norm <= 0.0 or self.target_kl <= 0.0:
            raise ValueError("learning rate, gradient norm and target KL must be positive")
        if self.entropy_coef < 0.0 or self.value_coef < 0.0 or self.visual_geometry_auxiliary_coef <= 0.0:
            raise ValueError("loss coefficients must be non-negative")
        if not 0.0 <= self.action_autoregressive_rho < 0.99:
            raise ValueError("action_autoregressive_rho must be in [0,0.99)")
        for name in (
            "obstacle_probability",
            "stress_probability",
            "auxiliary_view_dropout_probability",
        ):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")


class SharedVisualEncoderV1(nn.Module):
    """One compact convolutional encoder shared by every view and time."""

    output_dim = 96

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            # Preserve a coarse spatial grid. A global average would make the
            # actor nearly translation invariant and discard the block/target
            # location information needed for closed-loop pushing.
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.SiLU(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("shared visual encoder requires [N,3,H,W]")
        normalized = images.to(dtype=torch.float32).div(255.0)
        return self.network(normalized)


class SelectedViewRecurrentActorV1(nn.Module):
    """Causal visual actor with selected-view gating and masked recurrence."""

    def __init__(self, *, task_count: int = 1) -> None:
        super().__init__()
        if type(task_count) is not int or task_count < 1:
            raise ValueError("task_count must be a positive integer")
        self.task_count = task_count
        self.visual = SharedVisualEncoderV1()
        self.view_embedding = nn.Embedding(len(VIEW_NAMES), self.visual.output_dim)
        self.view_score = nn.Linear(self.visual.output_dim, 1)
        self.proprio = nn.Sequential(
            nn.Linear(JOINT_STATE_DIM + ACTION_DIM, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
        )
        self.task_embedding = nn.Embedding(task_count, 16)
        self.recurrent = nn.GRUCell(self.visual.output_dim + 64 + 16, 128)
        self.trunk = nn.Sequential(nn.Linear(128, 128), nn.LayerNorm(128), nn.SiLU())
        self.mean_head = nn.Linear(128, POLICY_ACTION_DIM)
        self.visual_geometry_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, VISUAL_GEOMETRY_TARGET_DIM),
        )
        self.log_std = nn.Parameter(torch.full((POLICY_ACTION_DIM,), -0.70))
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)

    @staticmethod
    def _validate_shapes(
        rgb_history: torch.Tensor,
        joint_history: torch.Tensor,
        action_history: torch.Tensor,
        history_valid: torch.Tensor,
        view_valid: torch.Tensor,
        task_id: torch.Tensor,
    ) -> tuple[int, int, int]:
        if rgb_history.ndim != 6:
            raise ValueError("rgb_history must be [B,T,V,H,W,3]")
        batch, steps, views, _height, _width, channels = rgb_history.shape
        if views != len(VIEW_NAMES) or channels != 3:
            raise ValueError("rgb_history view/channel layout is invalid")
        if joint_history.shape != (batch, steps, JOINT_STATE_DIM):
            raise ValueError("joint_history shape is invalid")
        if action_history.shape != (batch, steps, ACTION_DIM):
            raise ValueError("action_history shape is invalid")
        if history_valid.shape != (batch, steps) or history_valid.dtype != torch.bool:
            raise ValueError("history_valid shape/dtype is invalid")
        if view_valid.shape != (batch, steps, views) or view_valid.dtype != torch.bool:
            raise ValueError("view_valid shape/dtype is invalid")
        if task_id.shape != (batch,) or task_id.dtype != torch.long:
            raise ValueError("task_id shape/dtype is invalid")
        if not bool(torch.all(history_valid.any(dim=1)).item()):
            raise ValueError("every policy sample requires at least one causal history row")
        if bool(torch.any(view_valid & ~history_valid.unsqueeze(-1)).item()):
            raise ValueError("invalid history slots cannot expose camera views")
        if not bool(torch.all((view_valid & history_valid.unsqueeze(-1)).any(dim=(1, 2))).item()):
            raise ValueError("every policy sample requires at least one visible camera")
        return batch, steps, views

    def temporal_features(
        self,
        rgb_history: torch.Tensor,
        joint_history: torch.Tensor,
        action_history: torch.Tensor,
        history_valid: torch.Tensor,
        view_valid: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        batch, steps, views = self._validate_shapes(
            rgb_history,
            joint_history,
            action_history,
            history_valid,
            view_valid,
            task_id,
        )
        image_batch = rgb_history.permute(0, 1, 2, 5, 3, 4).reshape(
            batch * steps * views,
            3,
            rgb_history.shape[3],
            rgb_history.shape[4],
        )
        visual = self.visual(image_batch).reshape(batch, steps, views, -1)
        view_ids = torch.arange(views, device=visual.device)
        visual = visual + self.view_embedding(view_ids).view(1, 1, views, -1)
        logits = self.view_score(torch.tanh(visual)).squeeze(-1)
        masked_logits = logits.masked_fill(~view_valid, -1.0e4)
        weights = torch.softmax(masked_logits, dim=-1) * view_valid.to(dtype=visual.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        fused_visual = (visual * weights.unsqueeze(-1)).sum(dim=2)

        proprio = self.proprio(torch.cat((joint_history, action_history), dim=-1))
        task = self.task_embedding(task_id).unsqueeze(1).expand(-1, steps, -1)
        return torch.cat((fused_visual, proprio, task), dim=-1)

    def encode_temporal_features(
        self,
        temporal_input: torch.Tensor,
        history_valid: torch.Tensor,
    ) -> torch.Tensor:
        if temporal_input.ndim != 3:
            raise ValueError("temporal actor features must be [B,T,D]")
        batch, steps, feature_dim = temporal_input.shape
        expected_feature_dim = self.visual.output_dim + 64 + 16
        if feature_dim != expected_feature_dim:
            raise ValueError("temporal actor feature width changed")
        if history_valid.shape != (batch, steps) or history_valid.dtype != torch.bool:
            raise ValueError("temporal actor history mask is invalid")
        if not bool(torch.all(history_valid.any(dim=1)).item()):
            raise ValueError("every temporal actor sample requires a valid step")
        hidden = torch.zeros(batch, 128, device=temporal_input.device, dtype=temporal_input.dtype)
        for step in range(steps):
            proposed = self.recurrent(temporal_input[:, step], hidden)
            valid = history_valid[:, step].unsqueeze(-1)
            hidden = torch.where(valid, proposed, hidden)
        return self.trunk(hidden)

    def encode(
        self,
        rgb_history: torch.Tensor,
        joint_history: torch.Tensor,
        action_history: torch.Tensor,
        history_valid: torch.Tensor,
        view_valid: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        temporal_input = self.temporal_features(
            rgb_history,
            joint_history,
            action_history,
            history_valid,
            view_valid,
            task_id,
        )
        return self.encode_temporal_features(temporal_input, history_valid)

    def forward(
        self,
        rgb_history: torch.Tensor,
        joint_history: torch.Tensor,
        action_history: torch.Tensor,
        history_valid: torch.Tensor,
        view_valid: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.encode(
            rgb_history,
            joint_history,
            action_history,
            history_valid,
            view_valid,
            task_id,
        )
        return self.mean_head(encoded)

    def distribution(
        self,
        rgb_history: torch.Tensor,
        joint_history: torch.Tensor,
        action_history: torch.Tensor,
        history_valid: torch.Tensor,
        view_valid: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.distributions.Normal:
        mean = self(
            rgb_history,
            joint_history,
            action_history,
            history_valid,
            view_valid,
            task_id,
        )
        standard_deviation = self.log_std.clamp(-5.0, 1.0).exp()
        return torch.distributions.Normal(mean, standard_deviation)

    def distribution_and_visual_geometry(
        self,
        rgb_history: torch.Tensor,
        joint_history: torch.Tensor,
        action_history: torch.Tensor,
        history_valid: torch.Tensor,
        view_valid: torch.Tensor,
        task_id: torch.Tensor,
    ) -> tuple[torch.distributions.Normal, torch.Tensor]:
        encoded = self.encode(
            rgb_history,
            joint_history,
            action_history,
            history_valid,
            view_valid,
            task_id,
        )
        mean = self.mean_head(encoded)
        standard_deviation = self.log_std.clamp(-5.0, 1.0).exp()
        return (
            torch.distributions.Normal(mean, standard_deviation),
            self.visual_geometry_head(encoded),
        )

    def predict_visual_geometry(
        self,
        rgb_history: torch.Tensor,
        joint_history: torch.Tensor,
        action_history: torch.Tensor,
        history_valid: torch.Tensor,
        view_valid: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.encode(
            rgb_history,
            joint_history,
            action_history,
            history_valid,
            view_valid,
            task_id,
        )
        return self.visual_geometry_head(encoded)

    def deterministic(
        self,
        rgb_history: torch.Tensor,
        joint_history: torch.Tensor,
        action_history: torch.Tensor,
        history_valid: torch.Tensor,
        view_valid: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        return torch.tanh(
            self(
                rgb_history,
                joint_history,
                action_history,
                history_valid,
                view_valid,
                task_id,
            )
        )


class CausalVisualInferenceCacheV24:
    """Cache overlapping visual features without changing actor semantics.

    The original actor evaluates a right-aligned window and re-encodes every
    RGB frame on every control step.  Consecutive windows overlap by all but
    one frame.  This cache evaluates the CNN/view gate only for the newly
    observed row, retains at most ``history_steps`` temporal features, and
    reruns the small GRU window from zero.  The policy distribution therefore
    remains equivalent to the persisted full-history replay while avoiding
    redundant visual work in collection and closed-loop evaluation.
    """

    format = CAUSAL_VISUAL_INFERENCE_CACHE_FORMAT_V24

    def __init__(
        self,
        actor: SelectedViewRecurrentActorV1,
        *,
        history_steps: int,
        task_id: int = 0,
    ) -> None:
        if type(actor) is not SelectedViewRecurrentActorV1:
            raise TypeError("V24 inference cache requires the exact visual actor")
        if type(history_steps) is not int or history_steps < 1:
            raise ValueError("V24 inference cache history must be positive")
        if type(task_id) is not int or not 0 <= task_id < actor.task_count:
            raise ValueError("V24 inference cache task id is invalid")
        self.actor = actor
        self.history_steps = history_steps
        self.task_id = task_id
        self._features: deque[torch.Tensor] = deque(maxlen=history_steps)

    def reset(self) -> None:
        self._features.clear()

    def distribution(
        self,
        rgb_frames: np.ndarray,
        joint_state: np.ndarray,
        previous_executed_action: np.ndarray,
        view_valid: np.ndarray,
    ) -> torch.distributions.Normal:
        device = _module_device(self.actor)
        rgb = np.asarray(rgb_frames)
        joints = np.asarray(joint_state)
        action = np.asarray(previous_executed_action)
        views = np.asarray(view_valid)
        if rgb.ndim != 4 or rgb.shape[0] != len(VIEW_NAMES) or rgb.shape[-1] != 3:
            raise ValueError("V24 inference cache RGB row is invalid")
        if rgb.dtype != np.uint8:
            raise ValueError("V24 inference cache RGB row must be uint8")
        if joints.shape != (JOINT_STATE_DIM,) or joints.dtype != np.float32:
            raise ValueError("V24 inference cache joint row is invalid")
        if action.shape != (ACTION_DIM,) or action.dtype != np.float32:
            raise ValueError("V24 inference cache action row is invalid")
        if views.shape != (len(VIEW_NAMES),) or views.dtype != np.bool_ or not np.any(views):
            raise ValueError("V24 inference cache view mask is invalid")
        valid = torch.ones((1, 1), dtype=torch.bool, device=device)
        task = torch.full((1,), self.task_id, dtype=torch.long, device=device)
        feature = self.actor.temporal_features(
            torch.from_numpy(rgb).to(device).unsqueeze(0).unsqueeze(0),
            torch.from_numpy(joints).to(device).unsqueeze(0).unsqueeze(0),
            torch.from_numpy(action).to(device).unsqueeze(0).unsqueeze(0),
            valid,
            torch.from_numpy(views).to(device).unsqueeze(0).unsqueeze(0),
            task,
        )
        self._features.append(feature[:, 0])
        temporal = torch.stack(tuple(self._features), dim=1)
        history_valid = torch.ones(
            (1, temporal.shape[1]),
            dtype=torch.bool,
            device=device,
        )
        encoded = self.actor.encode_temporal_features(temporal, history_valid)
        mean = self.actor.mean_head(encoded)
        standard_deviation = self.actor.log_std.clamp(-5.0, 1.0).exp()
        return torch.distributions.Normal(mean, standard_deviation)


class AsymmetricPrivilegedCriticV1(nn.Module):
    """Training-only value network; simulator state never enters the actor."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(PRIVILEGED_EFFECT_STATE_DIM, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.shape[-1] != PRIVILEGED_EFFECT_STATE_DIM:
            raise ValueError(f"critic requires {PRIVILEGED_EFFECT_STATE_DIM}-D state")
        return self.network(state).squeeze(-1)


@dataclass(frozen=True)
class AsymmetricMultiViewProvenanceV1:
    source_type: str
    policy_format: str
    random_initialization: bool
    initialization_seed: int
    expert_calls: int
    warm_start: bool
    behavior_cloning_steps: int
    zero_mean_action_head_initialization: bool
    actor_privileged_state_inputs: int
    actor_training_only_privileged_geometry_supervision: bool
    visual_geometry_target_dimension: int
    visual_geometry_target_schema: str
    critic_privileged_state_inputs: int
    policy_action_space: str
    low_level_action_adapter: str
    low_level_adapter_privileged_block_pose: bool
    low_level_adapter_privileged_reset_pose: bool
    curriculum_reset_deployment_equivalent: bool
    deployment_adapter_requires_visual_pose_estimator: bool
    actor_initial_state_sha256: str
    critic_initial_state_sha256: str
    privileged_state_schema_sha256: str
    source_hashes: dict[str, str]
    genesis_sha256: str

    def validate(self) -> None:
        if self.source_type != SOURCE_TYPE or self.policy_format != POLICY_FORMAT:
            raise ValueError("multiview provenance identity mismatch")
        if self.random_initialization is not True or self.initialization_seed < 0:
            raise ValueError("multiview PPO must start from a seeded random genesis")
        if self.expert_calls != 0 or self.warm_start is not False:
            raise ValueError("multiview PPO forbids expert calls and warm starts")
        if self.behavior_cloning_steps != 0 or self.actor_privileged_state_inputs != 0:
            raise ValueError("multiview actor provenance is not deployment-observable")
        if self.actor_training_only_privileged_geometry_supervision is not True:
            raise ValueError("multiview geometry supervision disclosure is missing")
        if self.visual_geometry_target_dimension != VISUAL_GEOMETRY_TARGET_DIM:
            raise ValueError("multiview geometry target dimension changed")
        if self.visual_geometry_target_schema != VISUAL_GEOMETRY_TARGET_SCHEMA:
            raise ValueError("multiview geometry target schema changed")
        if self.zero_mean_action_head_initialization is not True:
            raise ValueError("multiview scratch genesis must use unbiased action means")
        if self.critic_privileged_state_inputs != PRIVILEGED_EFFECT_STATE_DIM:
            raise ValueError("multiview critic privileged-state disclosure mismatch")
        if self.policy_action_space != ("bounded_taskframe_forward_lateral_vertical_causal_ar1_delta_v13"):
            raise ValueError("multiview policy action-space disclosure mismatch")
        if self.low_level_action_adapter != (
            "stock-gripper-taskframe-latched-absolute-target-float32-guarded-recovery-v13"
        ):
            raise ValueError("multiview low-level adapter disclosure mismatch")
        if self.low_level_adapter_privileged_block_pose is not True:
            raise ValueError("simulation IK adapter privilege must be disclosed")
        if self.low_level_adapter_privileged_reset_pose is not True:
            raise ValueError("simulation curriculum-reset privilege must be disclosed")
        if self.curriculum_reset_deployment_equivalent is not False:
            raise ValueError("simulation curriculum reset cannot claim deployment equivalence")
        if self.deployment_adapter_requires_visual_pose_estimator is not True:
            raise ValueError("deployment pose-estimator requirement must be disclosed")
        if self.privileged_state_schema_sha256 != PRIVILEGED_EFFECT_STATE_SCHEMA_SHA256:
            raise ValueError("privileged state schema hash mismatch")
        if not self.source_hashes:
            raise ValueError("multiview provenance has no source hashes")
        for value in (
            self.actor_initial_state_sha256,
            self.critic_initial_state_sha256,
            self.privileged_state_schema_sha256,
            self.genesis_sha256,
            *self.source_hashes.values(),
        ):
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError("multiview provenance contains a malformed hash")
        payload = asdict(self)
        payload.pop("genesis_sha256")
        if self.genesis_sha256 != canonical_sha256_v1(payload):
            raise ValueError("multiview genesis hash mismatch")


@dataclass
class AsymmetricMultiViewBundleV1:
    actor: SelectedViewRecurrentActorV1
    critic: AsymmetricPrivilegedCriticV1
    provenance: AsymmetricMultiViewProvenanceV1


def initialize_asymmetric_multiview_ppo_v1(
    seed: int,
    *,
    device: str | torch.device = "cpu",
    scene_path: Path | None = None,
) -> AsymmetricMultiViewBundleV1:
    if type(seed) is not int or seed < 0:
        raise ValueError("initialization seed must be a non-negative integer")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        actor = SelectedViewRecurrentActorV1(task_count=1)
        critic = AsymmetricPrivilegedCriticV1()
    actor_hash = state_dict_sha256_v1(actor.state_dict())
    critic_hash = state_dict_sha256_v1(critic.state_dict())
    source_hashes = {
        "edgearm/asymmetric_multiview_ppo_v1.py": sha256_file_v1(Path(__file__).resolve()),
        "edgearm/stock_gripper_action_guard_v3.py": sha256_file_v1(
            Path(__file__).with_name("stock_gripper_action_guard_v3.py")
        ),
        "edgearm/scratch_ppo_v6_candidate.py": sha256_file_v1(
            Path(__file__).with_name("scratch_ppo_v6_candidate.py")
        ),
    }
    if scene_path is not None:
        source_hashes["scene/edgearm_multiview_rl_scene_v1.xml"] = sha256_file_v1(scene_path)
    base = {
        "source_type": SOURCE_TYPE,
        "policy_format": POLICY_FORMAT,
        "random_initialization": True,
        "initialization_seed": seed,
        "expert_calls": 0,
        "warm_start": False,
        "behavior_cloning_steps": 0,
        "zero_mean_action_head_initialization": True,
        "actor_privileged_state_inputs": 0,
        "actor_training_only_privileged_geometry_supervision": True,
        "visual_geometry_target_dimension": VISUAL_GEOMETRY_TARGET_DIM,
        "visual_geometry_target_schema": VISUAL_GEOMETRY_TARGET_SCHEMA,
        "critic_privileged_state_inputs": PRIVILEGED_EFFECT_STATE_DIM,
        "policy_action_space": ("bounded_taskframe_forward_lateral_vertical_causal_ar1_delta_v13"),
        "low_level_action_adapter": (
            "stock-gripper-taskframe-latched-absolute-target-float32-guarded-recovery-v13"
        ),
        "low_level_adapter_privileged_block_pose": True,
        "low_level_adapter_privileged_reset_pose": True,
        "curriculum_reset_deployment_equivalent": False,
        "deployment_adapter_requires_visual_pose_estimator": True,
        "actor_initial_state_sha256": actor_hash,
        "critic_initial_state_sha256": critic_hash,
        "privileged_state_schema_sha256": PRIVILEGED_EFFECT_STATE_SCHEMA_SHA256,
        "source_hashes": source_hashes,
    }
    provenance = AsymmetricMultiViewProvenanceV1(
        **base,
        genesis_sha256=canonical_sha256_v1(base),
    )
    provenance.validate()
    return AsymmetricMultiViewBundleV1(
        actor=actor.to(device),
        critic=critic.to(device),
        provenance=provenance,
    )


@dataclass(frozen=True)
class TaskSpaceActionConfigV5:
    """Live-feedback XYZ action with a near-contact, collision-free reset."""

    translation_step_m: float = 0.006
    vertical_translation_step_m: float = 0.004
    minimum_tool_height_m: float = 0.055
    maximum_tool_height_m: float = 0.105
    curriculum_reset_tool_height_m: float = 0.075
    curriculum_reset_tool_standoff_m: float = 0.050
    curriculum_reset_minimum_block_clearance_m: float = 0.002
    curriculum_reset_block_settle_tolerance_m: float = 0.001
    curriculum_reset_settle_substeps: int = 10
    position_tolerance_m: float = 0.00075
    face_normal_tolerance_rad: float = 0.05
    gripper_table_angle_rad: float = float(np.pi / 2.0)
    face_normal_slew_rate_rad_s: float = float(np.deg2rad(75.0))
    maximum_ik_target_step_rad: float = 0.20
    backtracking_scales: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125)

    def validate(self) -> None:
        for name in (
            "translation_step_m",
            "vertical_translation_step_m",
            "curriculum_reset_minimum_block_clearance_m",
            "curriculum_reset_block_settle_tolerance_m",
            "position_tolerance_m",
            "face_normal_tolerance_rad",
            "gripper_table_angle_rad",
            "face_normal_slew_rate_rad_s",
            "maximum_ik_target_step_rad",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"task-space {name} must be numeric")
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"task-space {name} must be finite and positive")
        for name in (
            "minimum_tool_height_m",
            "maximum_tool_height_m",
            "curriculum_reset_tool_height_m",
            "curriculum_reset_tool_standoff_m",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"task-space {name} must be numeric")
            if not np.isfinite(value):
                raise ValueError(f"task-space {name} must be finite")
        if not self.minimum_tool_height_m < self.maximum_tool_height_m:
            raise ValueError("task-space tool-height bounds must be ordered")
        if (
            not self.minimum_tool_height_m
            <= self.curriculum_reset_tool_height_m
            <= self.maximum_tool_height_m
        ):
            raise ValueError("curriculum reset height must lie inside task-space bounds")
        if not 0.030 <= self.curriculum_reset_tool_standoff_m <= 0.070:
            raise ValueError("curriculum reset standoff must be in [0.03,0.07] m")
        if type(self.curriculum_reset_settle_substeps) is not int:
            raise ValueError("curriculum reset settle substeps must be an integer")
        if self.curriculum_reset_settle_substeps < 1:
            raise ValueError("curriculum reset settle substeps must be positive")
        if not 0.0 < self.gripper_table_angle_rad < np.pi:
            raise ValueError("task-space gripper table angle must be in (0,pi)")
        scales = tuple(float(value) for value in self.backtracking_scales)
        if not scales or any(not 0.0 < value <= 1.0 for value in scales):
            raise ValueError("task-space backtracking scales must be in (0,1]")
        if scales[0] != 1.0 or any(left <= right for left, right in zip(scales, scales[1:])):
            raise ValueError("task-space backtracking scales must start at one and decrease")

    @property
    def translation_scale_xyz_m(self) -> np.ndarray:
        return np.asarray(
            [
                self.translation_step_m,
                self.translation_step_m,
                self.vertical_translation_step_m,
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class TaskSpaceActionResultV5:
    requested_task_action: np.ndarray
    applied_task_action: np.ndarray
    submitted_joint_action: np.ndarray
    target_position_world_m: np.ndarray
    target_face_normal_world: np.ndarray
    application_scale: float
    ik_converged: bool
    face_label: str
    failure_reason: str
    guard_safe_candidate: bool = True
    guard_selected_scale: float = 1.0
    guard_minimum_one_step_clearance_m: float = 0.0
    guard_minimum_braking_clearance_m: float = 0.0
    guard_float32_execution_identity: bool = True


class PrivilegedFeedbackTaskSpaceIKAdapterV5:
    """Map visual-policy XYZ actions to live-feedback joint commands.

    The actor never sees simulator geometry. The adapter uses the simulated
    block pose only to preserve the stock-gripper face orientation requested by
    the user. A deployable counterpart must obtain that pose from wrist-camera
    perception/3D reconstruction; the provenance records this boundary. Every
    target is anchored to the current encoder/kinematic state, so actuator lag
    cannot strand a virtual Cartesian target ahead of the real arm.
    """

    def __init__(
        self,
        env: RealisticEdgeArmEnvV10,
        config: TaskSpaceActionConfigV5 | None = None,
    ) -> None:
        if type(env) is not RealisticEdgeArmEnvV10:
            raise TypeError("task-space adapter requires exact RealisticEdgeArmEnvV10")
        self.env = env
        self.config = config or TaskSpaceActionConfigV5()
        if type(self.config) is not TaskSpaceActionConfigV5:
            raise TypeError("task-space adapter requires exact TaskSpaceActionConfigV5")
        self.config.validate()
        ik_config = KeyboardCartesianConfig(
            translation_step_m=self.config.translation_step_m,
            position_tolerance_m=self.config.position_tolerance_m,
            face_normal_tolerance_rad=self.config.face_normal_tolerance_rad,
            face_normal_slew_rate_rad_s=self.config.face_normal_slew_rate_rad_s,
            maximum_ik_target_step_rad=self.config.maximum_ik_target_step_rad,
        )
        self.ik = PositionFaceAlignedIK(env, ik_config)
        self.tracker = NearestBlockFaceTracker(ik_config.face_switch_hysteresis_m)
        self._episode_active = False

    def begin_episode(self, seed: int) -> None:
        if type(seed) is not int or seed < 0:
            raise ValueError("task-space episode seed must be non-negative")
        block_xy = self.env.block_xy()
        push_direction = np.asarray(self.env.target_xy - block_xy, dtype=np.float64)
        push_direction /= max(float(np.linalg.norm(push_direction)), 1.0e-12)
        target_position = np.asarray(
            [
                *(block_xy - push_direction * self.config.curriculum_reset_tool_standoff_m),
                self.config.curriculum_reset_tool_height_m,
            ],
            dtype=np.float64,
        )
        self.tracker.reset()
        reset_face = self.tracker.select(self.env, target_position)
        reset_plane_normal = table_angle_face_normal(
            reset_face.tool_face_normal_world,
            self.config.gripper_table_angle_rad,
        )
        aligned_reset, accepted_reset_normal = self.ik.find_parallel_aligned_start(
            target_position,
            reset_plane_normal,
        )
        if not aligned_reset.converged:
            raise RuntimeError("task-space curriculum reset face-aligned IK did not converge")
        target_q = aligned_reset.target_joint_position_rad
        self.env._install_reset_joint_state(target_q)
        block_before_settle = self.env.block_xy().copy()
        for _ in range(self.config.curriculum_reset_settle_substeps):
            mujoco.mj_step(self.env.model, self.env.data)
        block_settle_displacement = float(np.linalg.norm(self.env.block_xy() - block_before_settle))
        self.env._install_reset_joint_state(
            np.asarray(self.env.data.qpos[:JOINT_ACTION_DIM], dtype=np.float64).copy()
        )
        self.env.data.time = 0.0
        mujoco.mj_forward(self.env.model, self.env.data)
        self.env.last_distance = self.env.distance_to_target()
        self.env._workspace_recovery_anchor_v10 = np.asarray(
            self.env.data.qpos[:JOINT_ACTION_DIM],
            dtype=np.float64,
        ).copy()
        audit = self.env._audit_reset_contacts(
            target_position,
            push_direction,
            block_settle_displacement_m=block_settle_displacement,
        )
        if not bool(audit["reset_valid"]):
            raise RuntimeError(f"task-space curriculum reset failed: {audit['reset_failure_reasons']}")
        if (
            float(audit["tool_block_signed_distance_m"])
            < self.config.curriculum_reset_minimum_block_clearance_m
            or int(audit["tool_block_contact_count"]) != 0
            or int(audit["forbidden_penetration_count"]) != 0
            or block_settle_displacement > self.config.curriculum_reset_block_settle_tolerance_m
        ):
            raise RuntimeError("task-space curriculum reset violated its clearance contract")
        self.env.episode_domain["taskspace_curriculum_reset_v5"] = {
            "format": "edgearm-taskspace-near-contact-reset-v5",
            "seed": seed,
            "source_type": SOURCE_TYPE,
            "expert_calls": 0,
            "expert_paths": 0,
            "policy_transitions_before_reset": 0,
            "privileged_reset_state_used": ["block_pose", "target_pose"],
            "deployment_reset_equivalent": False,
            "production_admission": False,
            "requested_tool_position_world_m": target_position.tolist(),
            "requested_tool_face_label": reset_face.label,
            "accepted_tool_face_normal_world": accepted_reset_normal.tolist(),
            "face_aligned_ik_position_error_m": aligned_reset.position_error_m,
            "face_aligned_ik_orientation_error_rad": aligned_reset.orientation_error_rad,
            "requested_tool_standoff_m": self.config.curriculum_reset_tool_standoff_m,
            "requested_tool_height_m": self.config.curriculum_reset_tool_height_m,
            "block_settle_displacement_m": block_settle_displacement,
            "reset_collision_audit": _json_safe(audit),
        }
        self.tracker.reset()
        current_q = np.asarray(
            self.env.observation()["joint_state"][:JOINT_ACTION_DIM],
            dtype=np.float64,
        )
        position, normal = self.ik.current_pose(current_q)
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(normal)):
            raise RuntimeError("task-space episode began with a non-finite tool pose")
        self._episode_active = True

    def translate(self, policy_action: np.ndarray) -> TaskSpaceActionResultV5:
        requested = np.asarray(policy_action, dtype=np.float64)
        if requested.shape != (POLICY_ACTION_DIM,) or not np.all(np.isfinite(requested)):
            raise ValueError("task-space policy action must be finite [3]")
        requested = np.clip(requested, -1.0, 1.0)
        if not self._episode_active:
            raise RuntimeError("task-space adapter must begin an episode before translating")
        current_reported_q = np.asarray(
            self.env.observation()["joint_state"][:JOINT_ACTION_DIM],
            dtype=np.float64,
        )
        current_position, current_normal = self.ik.current_pose(current_reported_q)
        current_normal /= max(float(np.linalg.norm(current_normal)), 1.0e-12)
        action_scale = self.config.translation_scale_xyz_m
        requested_delta = requested * action_scale
        raw_target = current_position + requested_delta
        minimum_z = max(self.env.config.workspace_z[0], self.config.minimum_tool_height_m)
        maximum_z = min(self.env.config.workspace_z[1], self.config.maximum_tool_height_m)
        lower_position_bound = np.asarray(
            [self.env.config.workspace_x[0], self.env.config.workspace_y[0], minimum_z],
            dtype=np.float64,
        )
        upper_position_bound = np.asarray(
            [self.env.config.workspace_x[1], self.env.config.workspace_y[1], maximum_z],
            dtype=np.float64,
        )
        clipped_target = np.clip(raw_target, lower_position_bound, upper_position_bound)
        # The live arm can lag or drift just outside the policy workspace.  A direct
        # projection back to the boundary may then be larger than one policy step,
        # which makes the recorded applied action escape [-1, 1].  Recover toward
        # the workspace one bounded Cartesian step at a time instead.
        bounded_delta = np.clip(
            clipped_target - current_position,
            -action_scale,
            action_scale,
        )
        original_selection = self.tracker.selection_state()
        last_failure = "ik_not_converged"
        for scale in self.config.backtracking_scales:
            self.tracker.restore_selection_state(original_selection)
            candidate_position = current_position + scale * bounded_delta
            try:
                face = self.tracker.select(self.env, candidate_position)
                requested_normal = table_angle_face_normal(
                    face.tool_face_normal_world,
                    self.config.gripper_table_angle_rad,
                )
                candidate_normal = slew_parallel_plane_normal(
                    current_normal,
                    requested_normal,
                    self.config.face_normal_slew_rate_rad_s * self.env.control_dt,
                )
                result, accepted_normal = self.ik.solve_parallel_face(
                    current_reported_q,
                    candidate_position,
                    candidate_normal,
                    preferred_normal_world=current_normal,
                )
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
                last_failure = f"{type(error).__name__}:{error}"
                continue
            if not result.converged:
                last_failure = "ik_not_converged"
                continue
            branch_jump = float(np.max(np.abs(result.target_joint_position_rad[:5] - current_reported_q[:5])))
            if branch_jump > self.config.maximum_ik_target_step_rad:
                last_failure = "ik_branch_discontinuity"
                continue
            submitted_joint_action = np.clip(
                (result.target_joint_position_rad - current_reported_q)
                / float(self.env.config.max_joint_delta),
                -1.0,
                1.0,
            )
            applied_task = np.clip(
                (candidate_position - current_position) / action_scale,
                -1.0,
                1.0,
            )
            return TaskSpaceActionResultV5(
                requested_task_action=requested.astype(np.float32),
                applied_task_action=applied_task.astype(np.float32),
                submitted_joint_action=submitted_joint_action.astype(np.float32),
                target_position_world_m=candidate_position.astype(np.float32),
                target_face_normal_world=accepted_normal.astype(np.float32),
                application_scale=float(scale),
                ik_converged=True,
                face_label=face.label,
                failure_reason="none",
            )
        self.tracker.restore_selection_state(original_selection)
        return TaskSpaceActionResultV5(
            requested_task_action=requested.astype(np.float32),
            applied_task_action=np.zeros(POLICY_ACTION_DIM, dtype=np.float32),
            submitted_joint_action=np.zeros(JOINT_ACTION_DIM, dtype=np.float32),
            target_position_world_m=current_position.astype(np.float32),
            target_face_normal_world=current_normal.astype(np.float32),
            application_scale=0.0,
            ik_converged=False,
            face_label="unchanged",
            failure_reason=last_failure,
        )


@dataclass(frozen=True)
class StockGripperTaskSpaceActionConfigV9:
    """Contact-reachable task-frame action geometry for the stock gripper.

    The 4 mm vertical authority is intentional rather than a larger generic
    exploration step.  With the exact force-limited V10 plant, the previous
    0.8 mm command could not overcome gravity sag and move around the limiting
    moving-jaw convex part during the final millimetres of contact acquisition.
    The guard still forecasts the exact float32 command plus an eight-decision
    braking tail, so the larger authority does not relax collision safety.
    """

    forward_translation_step_m: float = 0.0015
    lateral_translation_step_m: float = 0.0010
    vertical_translation_step_m: float = 0.0040
    minimum_tool_height_m: float = 0.0455
    maximum_tool_height_m: float = 0.0580
    reset_height_candidates_m: tuple[float, ...] = (0.047, 0.050, 0.052)
    reset_settle_substeps: int = 10
    reset_block_settle_tolerance_m: float = 0.001
    reset_minimum_safety_only_clearance_m: float = 0.00035
    ik_backtracking_scales: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125)
    guard: StockGripperActionGuardConfigV2 = StockGripperActionGuardConfigV2()

    def validate(self) -> None:
        for name in (
            "forward_translation_step_m",
            "lateral_translation_step_m",
            "vertical_translation_step_m",
            "minimum_tool_height_m",
            "maximum_tool_height_m",
            "reset_block_settle_tolerance_m",
            "reset_minimum_safety_only_clearance_m",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"stock task-frame {name} must be finite and positive")
        if not self.minimum_tool_height_m < self.maximum_tool_height_m:
            raise ValueError("stock task-frame height bounds must be ordered")
        heights = tuple(float(value) for value in self.reset_height_candidates_m)
        if not heights or any(
            not self.minimum_tool_height_m <= value <= self.maximum_tool_height_m for value in heights
        ):
            raise ValueError("stock reset heights must lie inside the task-frame bounds")
        if type(self.reset_settle_substeps) is not int or self.reset_settle_substeps < 1:
            raise ValueError("stock reset settle substeps must be a positive integer")
        scales = tuple(float(value) for value in self.ik_backtracking_scales)
        if (
            not scales
            or not np.isclose(scales[0], 1.0, rtol=0.0, atol=1.0e-12)
            or any(not 0.0 < value <= 1.0 for value in scales)
            or any(left <= right for left, right in zip(scales, scales[1:]))
        ):
            raise ValueError("stock IK backtracking scales must descend from one")
        if type(self.guard) is not StockGripperActionGuardConfigV2:
            raise TypeError("stock task-frame guard config must be exact V2")


class StockTaskFrameResetInfeasibleV11(RuntimeError):
    """Expected rejection of one randomized reset before policy collection."""


class StockGripperTaskFrameAdapterV9:
    """Map forward/lateral/vertical policy actions through stock geometry.

    No expert action or trajectory is queried.  Simulator block/target state is
    used only to define the task frame, curriculum reset, and collision shield;
    a real deployment must replace those privileges with calibrated perception.
    """

    def __init__(
        self,
        env: RealisticEdgeArmEnvV10,
        config: StockGripperTaskSpaceActionConfigV9 | None = None,
    ) -> None:
        if type(env) is not RealisticEdgeArmEnvV10:
            raise TypeError("stock task-frame adapter V9 requires exact V10")
        self.env = env
        self.config = config or StockGripperTaskSpaceActionConfigV9()
        if type(self.config) is not StockGripperTaskSpaceActionConfigV9:
            raise TypeError("stock task-frame adapter requires exact V9 config")
        self.config.validate()
        self._ik_config = SideContactIKConfig(
            minimum_tool_block_safety_clearance_m=(self.config.reset_minimum_safety_only_clearance_m),
            minimum_contact_part_central_side_margin_m=(
                PHYSICAL_EXPERT_V12_CONTACT_PART_CENTRAL_SIDE_MARGIN_M
            ),
            height_candidates_m=self.config.reset_height_candidates_m,
        )
        self.tracking = SideContactIKPlannerV1(env, self._ik_config)
        self.guard: StockGripperActionGuardV2 | None = None
        self._fk_scratch = mujoco.MjData(env.model)
        self._episode_active = False
        self.last_guard_report: dict[str, Any] = {}

    @staticmethod
    def _task_axes(env: RealisticEdgeArmEnvV10) -> tuple[np.ndarray, np.ndarray]:
        forward = np.asarray(env.target_xy - env.block_xy(), dtype=np.float64)
        forward /= max(float(np.linalg.norm(forward)), 1.0e-12)
        lateral = np.asarray([-forward[1], forward[0]], dtype=np.float64)
        return forward, lateral

    def begin_episode(self, seed: int) -> None:
        if type(seed) is not int or seed < 0:
            raise ValueError("stock task-frame episode seed must be non-negative")
        forward, _ = self._task_axes(self.env)
        self.tracking = SideContactIKPlannerV1(self.env, self._ik_config)
        reset_plan = self.tracking.solve(forward)
        if not bool(reset_plan["feasible"]):
            raise StockTaskFrameResetInfeasibleV11("stock task-frame reset IK found no feasible pose")
        best = dict(reset_plan["best"])
        target_q = np.asarray(best["joint_position_rad"], dtype=np.float64)
        self.env._install_reset_joint_state(target_q)
        block_before_settle = self.env.block_xy().copy()
        for _ in range(self.config.reset_settle_substeps):
            mujoco.mj_step(self.env.model, self.env.data)
        settled_q = np.asarray(
            self.env.data.qpos[:JOINT_ACTION_DIM],
            dtype=np.float64,
        ).copy()
        block_settle_displacement = float(np.linalg.norm(self.env.block_xy() - block_before_settle))
        self.env._install_reset_joint_state(settled_q)
        self.env.data.time = 0.0
        mujoco.mj_forward(self.env.model, self.env.data)
        self.env.last_distance = self.env.distance_to_target()
        self.env._workspace_recovery_anchor_v10 = settled_q.copy()

        safety_ids = tuple(int(value) for value in self.env._ids["tool_safety_geoms"])
        contact_ids = tuple(int(value) for value in self.env._ids["tool_contact_geoms"])
        contact_columns = {safety_ids.index(value) for value in contact_ids}
        safety_only_columns = [index for index in range(len(safety_ids)) if index not in contact_columns]
        distances = self.env._tool_safety_signed_distances_for_data(
            self.env._ids["block_geom"],
            self.env.data,
        )
        safety_only_minimum = float(np.min(distances[safety_only_columns]))
        contact_distances = self.env._tool_planning_signed_distances_for_data(
            self.env._ids["block_geom"],
            self.env.data,
        )
        desk_clearance = float(
            self.env._minimum_tool_safety_signed_distance_for_data(
                self.env._desk_geom,
                self.env.data,
            )
        )
        reset_failure_reasons: list[str] = []
        if distances.shape != (96,) or len(safety_only_columns) != 94:
            reset_failure_reasons.append("stock_geometry_identity")
        if safety_only_minimum < self.config.reset_minimum_safety_only_clearance_m:
            reset_failure_reasons.append("safety_only_clearance")
        if float(np.min(contact_distances)) < (JointPathPlannerConfig().minimum_tool_block_clearance_m):
            reset_failure_reasons.append("contact_part_clearance")
        if desk_clearance < self._ik_config.minimum_pusher_desk_clearance_m:
            reset_failure_reasons.append("tool_desk_clearance")
        if self.env._tool_block_contacts() != 0:
            reset_failure_reasons.append("initial_tool_block_contact")
        if block_settle_displacement > self.config.reset_block_settle_tolerance_m:
            reset_failure_reasons.append("block_settle_displacement")
        if reset_failure_reasons:
            raise StockTaskFrameResetInfeasibleV11(
                "stock task-frame reset failed: "
                + ",".join(reset_failure_reasons)
                + f"; safety94={safety_only_minimum:.12g}; "
                + f"tip_min={float(np.min(contact_distances)):.12g}; "
                + f"desk={desk_clearance:.12g}; "
                + f"block_settle={block_settle_displacement:.12g}"
            )

        self.tracking = SideContactIKPlannerV1(self.env, self._ik_config)
        self.guard = StockGripperActionGuardV2(self.env, self.config.guard)
        self.last_guard_report = {}
        self.env.episode_domain["stock_gripper_taskframe_reset_v11"] = {
            "format": "edgearm-stock-gripper-taskframe-reset-v11",
            "seed": seed,
            "source_type": SOURCE_TYPE,
            "stock_follower_unmodified": True,
            "added_contact_tool": False,
            "expert_calls": 0,
            "expert_paths": 0,
            "policy_transitions_before_reset": 0,
            "simulator_privileged": True,
            "privileged_reset_state_used": ["block_pose", "target_pose"],
            "deployment_reset_equivalent": False,
            "production_admission": False,
            "selected_height_m": float(best["height_m"]),
            "actual_tool_xyz_m": self.env.tool_xyz().tolist(),
            "block_settle_displacement_m": block_settle_displacement,
            "minimum_94_safety_only_block_clearance_m": safety_only_minimum,
            "reset_planner_minimum_full_tool_block_clearance_m": (
                self._ik_config.minimum_tool_block_safety_clearance_m
            ),
            "tip_block_signed_distance_m": contact_distances.tolist(),
            "minimum_tool_desk_clearance_m": desk_clearance,
            "safety_guard_format": STOCK_GRIPPER_ACTION_GUARD_FORMAT_V2,
            "safety_guard_config": asdict(self.config.guard),
            "vertical_authority_m": self.config.vertical_translation_step_m,
            "vertical_authority_rationale": (
                "measured V10 gravity-sag compensation and safe convex-corridor traversal"
            ),
        }
        self._episode_active = True

    def _failure_result(
        self,
        requested: np.ndarray,
        current_position: np.ndarray,
        current_normal: np.ndarray,
        failure_reason: str,
        *,
        guard_minimum_one_step_m: float = 0.0,
        guard_minimum_braking_m: float = 0.0,
    ) -> TaskSpaceActionResultV5:
        return TaskSpaceActionResultV5(
            requested_task_action=requested.astype(np.float32),
            applied_task_action=np.zeros(POLICY_ACTION_DIM, dtype=np.float32),
            submitted_joint_action=np.zeros(JOINT_ACTION_DIM, dtype=np.float32),
            target_position_world_m=current_position.astype(np.float32),
            target_face_normal_world=current_normal.astype(np.float32),
            application_scale=0.0,
            ik_converged=False,
            face_label="taskframe_hold",
            failure_reason=failure_reason,
            guard_safe_candidate=False,
            guard_selected_scale=0.0,
            guard_minimum_one_step_clearance_m=guard_minimum_one_step_m,
            guard_minimum_braking_clearance_m=guard_minimum_braking_m,
            guard_float32_execution_identity=True,
        )

    def translate(self, policy_action: np.ndarray) -> TaskSpaceActionResultV5:
        requested = np.asarray(policy_action, dtype=np.float64)
        if requested.shape != (POLICY_ACTION_DIM,) or not np.all(np.isfinite(requested)):
            raise ValueError("stock task-frame policy action must be finite [3]")
        requested = np.clip(requested, -1.0, 1.0)
        if not self._episode_active or self.guard is None:
            raise RuntimeError("stock task-frame adapter must begin an episode")
        current_q = np.asarray(self.env.data.qpos[:JOINT_ACTION_DIM], dtype=np.float64)
        current_position = self.env.tool_xyz().copy()
        current_normal = self.env.data.site_xmat[self.env._ids["tool_site"]].reshape(3, 3)[:, 1].copy()
        forward, lateral = self._task_axes(self.env)
        local_scale = np.asarray(
            [
                self.config.forward_translation_step_m,
                self.config.lateral_translation_step_m,
                self.config.vertical_translation_step_m,
            ],
            dtype=np.float64,
        )
        requested_local_delta = requested * local_scale
        requested_world_delta = np.asarray(
            [
                forward[0] * requested_local_delta[0] + lateral[0] * requested_local_delta[1],
                forward[1] * requested_local_delta[0] + lateral[1] * requested_local_delta[1],
                requested_local_delta[2],
            ],
            dtype=np.float64,
        )
        last_failure = "stock_side_contact_ik_not_feasible"
        guard_one_step_minimum = 0.0
        guard_braking_minimum = 0.0
        for ik_scale in self.config.ik_backtracking_scales:
            candidate_position = current_position + float(ik_scale) * requested_world_delta
            candidate_position[0] = np.clip(
                candidate_position[0],
                self.env.config.workspace_x[0],
                self.env.config.workspace_x[1],
            )
            candidate_position[1] = np.clip(
                candidate_position[1],
                self.env.config.workspace_y[0],
                self.env.config.workspace_y[1],
            )
            candidate_position[2] = np.clip(
                candidate_position[2],
                self.config.minimum_tool_height_m,
                self.config.maximum_tool_height_m,
            )
            tracked = self.tracking.track_target(
                candidate_position,
                forward,
                initial=current_q,
            )
            if not bool(tracked["feasible"]):
                continue
            qtarget = np.asarray(
                tracked["best"]["joint_position_rad"],
                dtype=np.float64,
            )
            reference = np.asarray(
                self.env._command_reference_reported_position(),
                dtype=np.float64,
            ) - np.asarray(self.env._zero_offset, dtype=np.float64)
            proposed = np.clip(
                (qtarget - reference) / float(self.env.config.max_joint_delta),
                -1.0,
                1.0,
            ).astype(np.float32)
            selected, guard_report = self.guard.select(
                proposed,
                require_hold_tail=True,
            )
            self.last_guard_report = guard_report
            attempts = list(guard_report["attempts"])
            one_step_values = [
                float(row["minimum_one_step_clearance_m"])
                for row in attempts
                if row["minimum_one_step_clearance_m"] is not None
            ]
            guard_one_step_minimum = min(one_step_values) if one_step_values else 0.0
            braking_values = [
                float(row["minimum_braking_clearance_m"])
                for row in attempts
                if row["minimum_braking_clearance_m"] is not None
            ]
            guard_braking_minimum = min(braking_values) if braking_values else guard_one_step_minimum
            if selected is None:
                last_failure = "stock_guard_no_safe_candidate"
                continue
            guard_one_step_minimum = float(
                guard_report["selected_forecast"]["minimum_forecast_94_safety_only_block_distance_m"]
            )
            guard_braking_minimum = float(
                guard_report["selected_hold_tail_forecast"][
                    "minimum_forecast_94_safety_only_block_distance_m"
                ]
            )
            guard_scale = float(guard_report["selected_scale"])
            if guard_scale <= 0.0 or float(np.max(np.abs(selected))) <= 1.0e-12:
                return self._failure_result(
                    requested,
                    current_position,
                    current_normal,
                    "stock_guard_selected_hold",
                    guard_minimum_one_step_m=guard_one_step_minimum,
                    guard_minimum_braking_m=guard_braking_minimum,
                )
            forecast_application = guard_report["selected_forecast"]["applications"][-1]
            submitted_target = np.asarray(
                forecast_application["applied_target_after_preflight_rad"],
                dtype=np.float64,
            )
            self._fk_scratch.qpos[:] = self.env.data.qpos
            self._fk_scratch.qvel[:] = self.env.data.qvel
            self._fk_scratch.qpos[:JOINT_ACTION_DIM] = submitted_target
            mujoco.mj_forward(self.env.model, self._fk_scratch)
            predicted_position = self._fk_scratch.site_xpos[self.env._ids["tool_site"]].copy()
            predicted_normal = (
                self._fk_scratch.site_xmat[self.env._ids["tool_site"]].reshape(3, 3)[:, 1].copy()
            )
            world_delta = predicted_position - current_position
            applied_task = np.asarray(
                [
                    np.dot(world_delta[:2], forward) / self.config.forward_translation_step_m,
                    np.dot(world_delta[:2], lateral) / self.config.lateral_translation_step_m,
                    world_delta[2] / self.config.vertical_translation_step_m,
                ],
                dtype=np.float64,
            )
            return TaskSpaceActionResultV5(
                requested_task_action=requested.astype(np.float32),
                applied_task_action=np.clip(applied_task, -1.0, 1.0).astype(np.float32),
                submitted_joint_action=selected,
                target_position_world_m=predicted_position.astype(np.float32),
                target_face_normal_world=predicted_normal.astype(np.float32),
                application_scale=float(ik_scale) * guard_scale,
                ik_converged=True,
                face_label="target_aligned_stock_gripper",
                failure_reason="none",
                guard_safe_candidate=True,
                guard_selected_scale=guard_scale,
                guard_minimum_one_step_clearance_m=guard_one_step_minimum,
                guard_minimum_braking_clearance_m=guard_braking_minimum,
                guard_float32_execution_identity=bool(guard_report["float32_forecast_execution_identity"]),
            )
        return self._failure_result(
            requested,
            current_position,
            current_normal,
            last_failure,
            guard_minimum_one_step_m=guard_one_step_minimum,
            guard_minimum_braking_m=guard_braking_minimum,
        )


@dataclass(frozen=True)
class StockGripperTaskSpaceActionConfigV12:
    """Persistent task target and latched joint-setpoint action contract."""

    forward_translation_step_m: float = 0.0015
    lateral_translation_step_m: float = 0.0010
    vertical_translation_step_m: float = 0.0040
    maximum_cartesian_goal_lead_m: float = 0.0040
    maximum_joint_target_delta_rad: float = 0.045
    minimum_tool_height_m: float = 0.0455
    maximum_tool_height_m: float = 0.0580
    reset_height_candidates_m: tuple[float, ...] = (0.047, 0.050, 0.052)
    reset_settle_substeps: int = 10
    reset_block_settle_tolerance_m: float = 0.001
    reset_minimum_safety_only_clearance_m: float = 0.00035
    # Optional simulator-privileged initial-state curriculum.  When enabled,
    # the ordinary collision-audited V12 reset is accepted only if the nearest
    # authoritative stock-gripper contact part lies inside this post-settle
    # gap band.  Rejected poses are retried with another randomized reset seed;
    # no approach action, expert path, or policy transition is generated.
    curriculum_reset_tip_gap_band_m: tuple[float, float] | None = None
    ik_backtracking_scales: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125)
    maximum_predicted_reverse_normalized_action: float = 0.05
    guard: StockGripperActionGuardConfigV3 = StockGripperActionGuardConfigV3()

    def validate(self) -> None:
        for name in (
            "forward_translation_step_m",
            "lateral_translation_step_m",
            "vertical_translation_step_m",
            "maximum_cartesian_goal_lead_m",
            "maximum_joint_target_delta_rad",
            "minimum_tool_height_m",
            "maximum_tool_height_m",
            "reset_block_settle_tolerance_m",
            "reset_minimum_safety_only_clearance_m",
            "maximum_predicted_reverse_normalized_action",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"stock task-frame V12 {name} must be positive")
        if not self.minimum_tool_height_m < self.maximum_tool_height_m:
            raise ValueError("stock task-frame V12 height bounds must be ordered")
        if self.maximum_joint_target_delta_rad > 0.050:
            raise ValueError("V12 joint target delta must leave encoder-noise authority")
        heights = tuple(float(value) for value in self.reset_height_candidates_m)
        if not heights or any(
            not self.minimum_tool_height_m <= value <= self.maximum_tool_height_m for value in heights
        ):
            raise ValueError("stock task-frame V12 reset heights are invalid")
        gap_band = self.curriculum_reset_tip_gap_band_m
        if gap_band is not None:
            if type(gap_band) is not tuple or len(gap_band) != 2:
                raise TypeError("stock task-frame curriculum reset gap band must be a two-tuple")
            lower_gap, upper_gap = (float(value) for value in gap_band)
            if (
                not np.isfinite(lower_gap)
                or not np.isfinite(upper_gap)
                or lower_gap < JointPathPlannerConfig().minimum_tool_block_clearance_m
                or lower_gap >= upper_gap
                or upper_gap > 0.020
            ):
                raise ValueError("stock task-frame curriculum reset gap band is invalid")
        if type(self.reset_settle_substeps) is not int or self.reset_settle_substeps < 1:
            raise ValueError("stock task-frame V12 reset settle must be positive")
        scales = tuple(float(value) for value in self.ik_backtracking_scales)
        if (
            not scales
            or not np.isclose(scales[0], 1.0, rtol=0.0, atol=1.0e-12)
            or any(not 0.0 < value <= 1.0 for value in scales)
            or any(left <= right for left, right in zip(scales, scales[1:]))
        ):
            raise ValueError("stock task-frame V12 IK scales must descend from one")
        if type(self.guard) is not StockGripperActionGuardConfigV3:
            raise TypeError("stock task-frame V12 guard config must be exact V3")


class StockTaskFrameResetInfeasibleV12(StockTaskFrameResetInfeasibleV11):
    """Expected rejection of one V12 randomized reset."""


class StockGripperTaskFrameAdapterV12:
    """Task-frame controller with persistent Cartesian and joint targets.

    The policy still emits forward/lateral/vertical increments.  Unlike V9,
    those increments update one bounded Cartesian goal and every accepted
    joint target remains latched until replaced.  Thus a policy hold does not
    turn encoder noise into a new servo setpoint on every frame.
    """

    def __init__(
        self,
        env: RealisticEdgeArmEnvV10,
        config: StockGripperTaskSpaceActionConfigV12 | None = None,
    ) -> None:
        if type(env) is not RealisticEdgeArmEnvV10:
            raise TypeError("stock task-frame adapter V12 requires exact V10")
        self.env = env
        self.config = config or StockGripperTaskSpaceActionConfigV12()
        if type(self.config) is not StockGripperTaskSpaceActionConfigV12:
            raise TypeError("stock task-frame adapter requires exact V12 config")
        self.config.validate()
        self._ik_config = SideContactIKConfig(
            minimum_tool_block_safety_clearance_m=(self.config.reset_minimum_safety_only_clearance_m),
            minimum_contact_part_central_side_margin_m=(
                PHYSICAL_EXPERT_V12_CONTACT_PART_CENTRAL_SIDE_MARGIN_M
            ),
            height_candidates_m=self.config.reset_height_candidates_m,
        )
        self.tracking = SideContactIKPlannerV1(env, self._ik_config)
        self.guard: StockGripperActionGuardV3 | None = None
        self._fk_scratch = mujoco.MjData(env.model)
        self._episode_active = False
        self._cartesian_goal: np.ndarray | None = None
        self._latched_joint_target: np.ndarray | None = None
        self.last_guard_report: dict[str, Any] = {}
        self.last_recovery_report: dict[str, Any] = {}
        self.last_policy_target_encoding_report: dict[str, Any] = {}
        self.recovery_count = 0

    @staticmethod
    def _task_axes(env: RealisticEdgeArmEnvV10) -> tuple[np.ndarray, np.ndarray]:
        return StockGripperTaskFrameAdapterV9._task_axes(env)

    def begin_episode(self, seed: int) -> None:
        if type(seed) is not int or seed < 0:
            raise ValueError("stock task-frame V12 seed must be non-negative")
        forward, _ = self._task_axes(self.env)
        self.tracking = SideContactIKPlannerV1(self.env, self._ik_config)
        reset_plan = self.tracking.solve(forward)
        if not bool(reset_plan["feasible"]):
            raise StockTaskFrameResetInfeasibleV12("stock task-frame V12 reset IK found no feasible pose")
        best = dict(reset_plan["best"])
        target_q = np.asarray(best["joint_position_rad"], dtype=np.float64)
        self.env._install_reset_joint_state(target_q)
        block_before_settle = self.env.block_xy().copy()
        for _ in range(self.config.reset_settle_substeps):
            mujoco.mj_step(self.env.model, self.env.data)
        settled_q = np.asarray(self.env.data.qpos[:JOINT_ACTION_DIM], dtype=np.float64).copy()
        block_settle_displacement = float(np.linalg.norm(self.env.block_xy() - block_before_settle))
        self.env._install_reset_joint_state(settled_q)
        self.env.data.time = 0.0
        mujoco.mj_forward(self.env.model, self.env.data)
        self.env.last_distance = self.env.distance_to_target()
        self.env._workspace_recovery_anchor_v10 = settled_q.copy()

        safety_ids = tuple(int(value) for value in self.env._ids["tool_safety_geoms"])
        contact_ids = tuple(int(value) for value in self.env._ids["tool_contact_geoms"])
        contact_columns = {safety_ids.index(value) for value in contact_ids}
        safety_only_columns = [index for index in range(len(safety_ids)) if index not in contact_columns]
        distances = self.env._tool_safety_signed_distances_for_data(
            self.env._ids["block_geom"], self.env.data
        )
        safety_only_minimum = float(np.min(distances[safety_only_columns]))
        contact_distances = self.env._tool_planning_signed_distances_for_data(
            self.env._ids["block_geom"], self.env.data
        )
        selected_tip_gap = float(np.min(contact_distances))
        desk_clearance = float(
            self.env._minimum_tool_safety_signed_distance_for_data(self.env._desk_geom, self.env.data)
        )
        reset_failure_reasons: list[str] = []
        if distances.shape != (96,) or len(safety_only_columns) != 94:
            reset_failure_reasons.append("stock_geometry_identity")
        if safety_only_minimum < self.config.reset_minimum_safety_only_clearance_m:
            reset_failure_reasons.append("safety_only_clearance")
        if selected_tip_gap < (JointPathPlannerConfig().minimum_tool_block_clearance_m):
            reset_failure_reasons.append("contact_part_clearance")
        curriculum_gap_band = self.config.curriculum_reset_tip_gap_band_m
        if curriculum_gap_band is not None and not (
            float(curriculum_gap_band[0]) <= selected_tip_gap <= float(curriculum_gap_band[1])
        ):
            reset_failure_reasons.append("curriculum_tip_gap_band")
        if desk_clearance < self._ik_config.minimum_pusher_desk_clearance_m:
            reset_failure_reasons.append("tool_desk_clearance")
        if self.env._tool_block_contacts() != 0:
            reset_failure_reasons.append("initial_tool_block_contact")
        if block_settle_displacement > self.config.reset_block_settle_tolerance_m:
            reset_failure_reasons.append("block_settle_displacement")
        if reset_failure_reasons:
            raise StockTaskFrameResetInfeasibleV12(
                "stock task-frame V12 reset failed: "
                + ",".join(reset_failure_reasons)
                + f"; selected_tip_gap={selected_tip_gap:.12g}"
            )

        self.tracking = SideContactIKPlannerV1(self.env, self._ik_config)
        self.guard = StockGripperActionGuardV3(self.env, self.config.guard)
        self._cartesian_goal = self.env.tool_xyz().copy()
        self._latched_joint_target = settled_q.copy()
        try:
            hold_action, hold_identity = self.guard.action_for_absolute_target(self._latched_joint_target)
            selected_hold, hold_report = self.guard.select(
                hold_action,
                baseline_action=hold_action,
                require_hold_tail=True,
            )
        except LatchedAbsoluteTargetInfeasibleV3 as error:
            raise StockTaskFrameResetInfeasibleV12(
                "stock task-frame V12 reset target cannot be held exactly"
            ) from error
        if selected_hold is None or not np.array_equal(selected_hold, hold_action):
            raise StockTaskFrameResetInfeasibleV12(
                "stock task-frame V12 reset failed dynamic latched-hold audit"
            )
        hold_tail = hold_report["selected_hold_tail_forecast"]
        if not bool(hold_tail["remaining_submissions_target_latched_holds"]):
            raise StockTaskFrameResetInfeasibleV12("stock task-frame V12 reset hold-tail identity failed")
        self.last_guard_report = hold_report
        self.last_recovery_report = {}
        self.last_policy_target_encoding_report = {}
        self.recovery_count = 0
        self.env.episode_domain["stock_gripper_taskframe_reset_v12"] = {
            "format": "edgearm-stock-gripper-taskframe-reset-v12",
            "seed": seed,
            "source_type": SOURCE_TYPE,
            "stock_follower_unmodified": True,
            "added_contact_tool": False,
            "expert_calls": 0,
            "expert_paths": 0,
            "policy_transitions_before_reset": 0,
            "simulator_privileged": True,
            "privileged_reset_state_used": ["block_pose", "target_pose"],
            "deployment_reset_equivalent": False,
            "production_admission": False,
            "selected_height_m": float(best["height_m"]),
            "selected_lateral_m": float(best["lateral_m"]),
            "selected_standoff_m": float(best["standoff_m"]),
            "actual_tool_xyz_m": self.env.tool_xyz().tolist(),
            "block_settle_displacement_m": block_settle_displacement,
            "minimum_94_safety_only_block_clearance_m": safety_only_minimum,
            "reset_planner_minimum_full_tool_block_clearance_m": (
                self._ik_config.minimum_tool_block_safety_clearance_m
            ),
            "tip_block_signed_distance_m": contact_distances.tolist(),
            "selected_minimum_tip_block_signed_distance_m": selected_tip_gap,
            "curriculum_reset_active": curriculum_gap_band is not None,
            "curriculum_reset_tip_gap_band_m": (
                None
                if curriculum_gap_band is None
                else [
                    float(curriculum_gap_band[0]),
                    float(curriculum_gap_band[1]),
                ]
            ),
            "curriculum_reset_selection_method": (
                "disabled" if curriculum_gap_band is None else "privileged_post_settle_rejection_sampling"
            ),
            "curriculum_reset_approach_actions": 0,
            "curriculum_reset_policy_transitions": 0,
            "curriculum_reset_expert_actions": 0,
            "minimum_tool_desk_clearance_m": desk_clearance,
            "safety_guard_format": STOCK_GRIPPER_ACTION_GUARD_FORMAT_V3,
            "safety_guard_config": asdict(self.config.guard),
            "literal_zero_action_hold_semantics": False,
            "latched_absolute_target_hold": True,
            "initial_latched_joint_target_rad": settled_q.tolist(),
            "initial_hold_target_identity": hold_identity,
            "initial_hold_tail_minimum_clearance_m": float(
                hold_tail["minimum_forecast_safety_only_block_distance_m"]
            ),
            "maximum_cartesian_goal_lead_m": (self.config.maximum_cartesian_goal_lead_m),
            "vertical_authority_m": self.config.vertical_translation_step_m,
        }
        self._episode_active = True

    def _pose_for_joint_position(
        self,
        joint_position: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._fk_scratch.qpos[:] = self.env.data.qpos
        self._fk_scratch.qvel[:] = self.env.data.qvel
        self._fk_scratch.qpos[:JOINT_ACTION_DIM] = np.asarray(joint_position, dtype=np.float64)
        mujoco.mj_forward(self.env.model, self._fk_scratch)
        position = self._fk_scratch.site_xpos[self.env._ids["tool_site"]].copy()
        normal = self._fk_scratch.site_xmat[self.env._ids["tool_site"]].reshape(3, 3)[:, 1].copy()
        return position, normal

    def _hold_result(
        self,
        requested: np.ndarray,
        current_position: np.ndarray,
        current_normal: np.ndarray,
        action: np.ndarray,
        reason: str,
        *,
        guard_safe: bool,
        one_step_minimum: float,
        braking_minimum: float,
    ) -> TaskSpaceActionResultV5:
        return TaskSpaceActionResultV5(
            requested_task_action=requested.astype(np.float32),
            applied_task_action=np.zeros(POLICY_ACTION_DIM, dtype=np.float32),
            submitted_joint_action=np.asarray(action, dtype=np.float32),
            target_position_world_m=current_position.astype(np.float32),
            target_face_normal_world=current_normal.astype(np.float32),
            application_scale=0.0,
            ik_converged=False,
            face_label="latched_absolute_target_hold",
            failure_reason=reason,
            guard_safe_candidate=guard_safe,
            guard_selected_scale=0.0,
            guard_minimum_one_step_clearance_m=one_step_minimum,
            guard_minimum_braking_clearance_m=braking_minimum,
            guard_float32_execution_identity=True,
        )

    def _representable_interior_recovery_hold(
        self,
        current_q: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find a fixed, command-representable target inside the workspace.

        Force-limited gravity dynamics may leave the live state a fraction of
        a millimetre outside the Cartesian workspace.  Such a state cannot be
        used as an exact hold because V10 must filter it.  Move a small,
        deterministic distance from V10's filtered boundary point toward the
        reset recovery anchor and require exact float32 reconstruction before
        returning it.
        """

        if self.guard is None:
            raise RuntimeError("stock task-frame V12 recovery requires its guard")
        live = np.asarray(current_q, dtype=np.float64)
        filtered, filter_reason = self.env._safety_filter(live)
        anchor = self.env._workspace_recovery_anchor_v10
        if anchor is None:
            raise LatchedAbsoluteTargetInfeasibleV3("V12 has no workspace recovery anchor")
        anchor = np.asarray(anchor, dtype=np.float64)
        failures: list[dict[str, Any]] = []
        for inward_scale in (0.02, 0.05, 0.10, 0.20, 0.40, 0.70, 1.0):
            candidate = filtered + float(inward_scale) * (anchor - filtered)
            candidate[5] = self.env.tool_gripper_joint_position_rad
            try:
                action, identity = self.guard.action_for_absolute_target(candidate)
            except LatchedAbsoluteTargetInfeasibleV3 as error:
                failures.append(
                    {
                        "inward_scale": float(inward_scale),
                        "error": str(error),
                    }
                )
                continue
            position, _normal = self._pose_for_joint_position(candidate)
            self.recovery_count += 1
            self.last_recovery_report = {
                "format": "edgearm-stock-taskframe-interior-recovery-v12",
                "recovery_index": self.recovery_count,
                "live_joint_position_rad": live.tolist(),
                "filtered_boundary_target_rad": np.asarray(filtered).tolist(),
                "filter_reason": str(filter_reason),
                "workspace_recovery_anchor_rad": anchor.tolist(),
                "selected_inward_scale": float(inward_scale),
                "selected_target_rad": candidate.tolist(),
                "selected_tool_position_world_m": position.tolist(),
                "target_identity": identity,
                "rejected_candidates": failures,
                "simulator_privileged": True,
                "production_admission": False,
            }
            return candidate, action
        raise LatchedAbsoluteTargetInfeasibleV3("V12 found no representable interior recovery hold target")

    def _encode_policy_target_action(
        self,
        proposed_target: np.ndarray,
        *,
        active_request: bool,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """V12 requires exact fixed-target reconstruction for every action."""

        if self.guard is None:  # pragma: no cover - guarded by translate
            raise RuntimeError("stock task-frame V12 target encoding needs guard")
        action, identity = self.guard.action_for_absolute_target(
            proposed_target
        )
        self.last_policy_target_encoding_report = {
            "format": "edgearm-stock-taskframe-policy-target-encoding-v1",
            "mode": "exact_absolute_target",
            "active_request": bool(active_request),
            "target_identity": identity,
        }
        return action, identity

    def _track_policy_target(
        self,
        candidate_goal: np.ndarray,
        forward: np.ndarray,
        *,
        initial: np.ndarray,
    ) -> dict[str, Any]:
        """Resolve one Cartesian target without changing V12 semantics.

        The method is intentionally a narrow extension point.  A controller
        that starts far from the contact manifold can provide a guarded local
        acquisition step, while the historical V12/V13/V22 adapters continue
        to use the exact side-contact tracker unchanged.
        """

        return self.tracking.track_target(
            candidate_goal,
            forward,
            initial=initial,
        )

    def _policy_tool_height_bounds(self) -> tuple[float, float]:
        """Return the ordinary contact-transport Cartesian height bounds."""

        return (
            float(self.config.minimum_tool_height_m),
            float(self.config.maximum_tool_height_m),
        )

    def translate(
        self,
        policy_action: np.ndarray,
        *,
        preserve_latched_target: bool = False,
    ) -> TaskSpaceActionResultV5:
        requested = np.asarray(policy_action, dtype=np.float64)
        if requested.shape != (POLICY_ACTION_DIM,) or not np.all(np.isfinite(requested)):
            raise ValueError("stock task-frame V12 action must be finite [3]")
        requested = np.clip(requested, -1.0, 1.0)
        if (
            not self._episode_active
            or self.guard is None
            or self._cartesian_goal is None
            or self._latched_joint_target is None
        ):
            raise RuntimeError("stock task-frame V12 must begin an episode")
        current_q = np.asarray(self.env.data.qpos[:JOINT_ACTION_DIM], dtype=np.float64).copy()
        current_position = self.env.tool_xyz().copy()
        current_normal = self.env.data.site_xmat[self.env._ids["tool_site"]].reshape(3, 3)[:, 1].copy()
        forward, lateral = self._task_axes(self.env)
        local_scale = np.asarray(
            [
                self.config.forward_translation_step_m,
                self.config.lateral_translation_step_m,
                self.config.vertical_translation_step_m,
            ],
            dtype=np.float64,
        )
        requested_local_delta = requested * local_scale
        requested_world_delta = np.asarray(
            [
                forward[0] * requested_local_delta[0] + lateral[0] * requested_local_delta[1],
                forward[1] * requested_local_delta[0] + lateral[1] * requested_local_delta[1],
                requested_local_delta[2],
            ],
            dtype=np.float64,
        )
        previous_goal = self._cartesian_goal.copy()
        tentative_goal = previous_goal + requested_world_delta
        tentative_goal[0] = np.clip(
            tentative_goal[0],
            self.env.config.workspace_x[0],
            self.env.config.workspace_x[1],
        )
        tentative_goal[1] = np.clip(
            tentative_goal[1],
            self.env.config.workspace_y[0],
            self.env.config.workspace_y[1],
        )
        minimum_tool_height, maximum_tool_height = (
            self._policy_tool_height_bounds()
        )
        tentative_goal[2] = np.clip(
            tentative_goal[2],
            minimum_tool_height,
            maximum_tool_height,
        )
        lead = tentative_goal - current_position
        lead_norm = float(np.linalg.norm(lead))
        if lead_norm > self.config.maximum_cartesian_goal_lead_m:
            tentative_goal = current_position + lead * (self.config.maximum_cartesian_goal_lead_m / lead_norm)

        try:
            baseline_action, _ = self.guard.action_for_absolute_target(self._latched_joint_target)
        except LatchedAbsoluteTargetInfeasibleV3:
            recovery_target, baseline_action = self._representable_interior_recovery_hold(current_q)
            recovery_position, _ = self._pose_for_joint_position(recovery_target)
            self._latched_joint_target = recovery_target.copy()
            self._cartesian_goal = recovery_position.copy()
            previous_goal = recovery_position.copy()
            tentative_goal = recovery_position.copy()

        active_request = bool(float(np.max(np.abs(requested))) > 1.0e-8)
        goal_scales = self.config.ik_backtracking_scales if active_request else (0.0,)
        last_reason = "stock_taskframe_v12_no_directionally_valid_candidate"
        last_one_step = 0.0
        last_braking = 0.0
        for goal_scale in goal_scales:
            if float(goal_scale) == 0.0:
                candidate_goal = previous_goal.copy()
                proposed_target = self._latched_joint_target.copy()
            else:
                candidate_goal = previous_goal + float(goal_scale) * (tentative_goal - previous_goal)
                tracked = self._track_policy_target(
                    candidate_goal,
                    forward,
                    initial=self._latched_joint_target,
                )
                if not bool(tracked["feasible"]):
                    last_reason = "stock_taskframe_v12_ik_not_feasible"
                    continue
                proposed_target = np.asarray(tracked["best"]["joint_position_rad"], dtype=np.float64)
                joint_delta = proposed_target - current_q
                joint_linf = float(np.max(np.abs(joint_delta)))
                if joint_linf > self.config.maximum_joint_target_delta_rad:
                    proposed_target = current_q + joint_delta * (
                        self.config.maximum_joint_target_delta_rad / joint_linf
                    )
                proposed_target[5] = self.env.tool_gripper_joint_position_rad
            try:
                proposed_action, _ = self._encode_policy_target_action(
                    proposed_target,
                    active_request=active_request,
                )
            except LatchedAbsoluteTargetInfeasibleV3:
                last_reason = "stock_taskframe_v12_target_not_representable"
                continue
            projected_executable_target = bool(
                active_request
                and self.last_policy_target_encoding_report.get("mode")
                == "filtered_float32_command_projection"
            )
            recovery_tail_anchor = (
                self.env._workspace_recovery_anchor_v10
                if projected_executable_target
                else None
            )
            selected, guard_report = self.guard.select(
                proposed_action,
                baseline_action=baseline_action,
                require_hold_tail=True,
                preserve_baseline_target=bool(
                    preserve_latched_target and not active_request
                ),
                recovery_tail_anchor_rad=recovery_tail_anchor,
            )
            self.last_guard_report = guard_report
            self.last_guard_report["policy_target_encoding"] = deepcopy(
                self.last_policy_target_encoding_report
            )
            self.last_guard_report["policy_continuation_strategy"] = (
                "receding_horizon_workspace_anchor_recovery"
                if projected_executable_target
                else "fixed_target_braking_hold"
            )
            attempts = list(guard_report["attempts"])
            one_values = [
                float(row["minimum_one_step_clearance_m"])
                for row in attempts
                if row["minimum_one_step_clearance_m"] is not None
            ]
            braking_values = [
                float(row["minimum_braking_clearance_m"])
                for row in attempts
                if row["minimum_braking_clearance_m"] is not None
            ]
            last_one_step = min(one_values) if one_values else 0.0
            last_braking = min(braking_values) if braking_values else last_one_step
            if selected is None:
                last_reason = "stock_taskframe_v12_guard_no_safe_candidate"
                continue
            guard_scale = float(guard_report["selected_scale"])
            selected_forecast = guard_report["selected_forecast"]
            candidate_rows = [
                row for row in selected_forecast["applications"] if bool(row["candidate_command"])
            ]
            if len(candidate_rows) != 1:
                raise RuntimeError("stock task-frame V12 lost candidate forecast identity")
            candidate_row = candidate_rows[0]
            physics_endpoint = np.asarray(candidate_row["physics_endpoint_rad"], dtype=np.float64)
            predicted_position, predicted_normal = self._pose_for_joint_position(physics_endpoint)
            world_delta = predicted_position - current_position
            applied_task = np.asarray(
                [
                    np.dot(world_delta[:2], forward) / self.config.forward_translation_step_m,
                    np.dot(world_delta[:2], lateral) / self.config.lateral_translation_step_m,
                    world_delta[2] / self.config.vertical_translation_step_m,
                ],
                dtype=np.float64,
            )
            dominant = int(np.argmax(np.abs(requested)))
            reversed_effect = bool(
                active_request
                and guard_scale > 0.0
                and requested[dominant] * applied_task[dominant]
                < -self.config.maximum_predicted_reverse_normalized_action
            )
            if reversed_effect:
                last_reason = "stock_taskframe_v12_predicted_direction_reversal"
                continue
            if active_request and guard_scale <= 0.0:
                last_reason = "stock_taskframe_v12_guard_selected_latched_hold"
                return self._hold_result(
                    requested,
                    current_position,
                    current_normal,
                    selected,
                    last_reason,
                    guard_safe=True,
                    one_step_minimum=float(
                        selected_forecast["minimum_forecast_safety_only_block_distance_m"]
                    ),
                    braking_minimum=float(
                        guard_report["selected_hold_tail_forecast"][
                            "minimum_forecast_safety_only_block_distance_m"
                        ]
                    ),
                )
            # The V3 braking proof repeatedly resubmits the forecast physics
            # endpoint.  Keep that exact endpoint as the next online baseline
            # rather than restoring the farther preflight servo command.
            selected_target = (
                self._latched_joint_target.copy()
                if preserve_latched_target and not active_request
                else physics_endpoint
            )
            self._latched_joint_target = selected_target.copy()
            if not (preserve_latched_target and not active_request):
                self._cartesian_goal = previous_goal + guard_scale * (
                    candidate_goal - previous_goal
                )
            one_step_minimum = float(selected_forecast["minimum_forecast_safety_only_block_distance_m"])
            braking_minimum = float(
                guard_report["selected_hold_tail_forecast"]["minimum_forecast_safety_only_block_distance_m"]
            )
            return TaskSpaceActionResultV5(
                requested_task_action=requested.astype(np.float32),
                applied_task_action=np.clip(applied_task, -1.0, 1.0).astype(np.float32),
                submitted_joint_action=selected,
                target_position_world_m=predicted_position.astype(np.float32),
                target_face_normal_world=predicted_normal.astype(np.float32),
                application_scale=float(goal_scale) * guard_scale,
                ik_converged=True,
                face_label="target_aligned_stock_gripper_latched_v12",
                failure_reason="none",
                guard_safe_candidate=True,
                guard_selected_scale=guard_scale,
                guard_minimum_one_step_clearance_m=one_step_minimum,
                guard_minimum_braking_clearance_m=braking_minimum,
                guard_float32_execution_identity=bool(guard_report["float32_forecast_execution_identity"]),
            )

        try:
            emergency_target, emergency_action = self._representable_interior_recovery_hold(current_q)
        except LatchedAbsoluteTargetInfeasibleV3:  # pragma: no cover
            raise RuntimeError("stock task-frame V12 has no safe representable recovery hold")
        self._latched_joint_target = emergency_target
        self._cartesian_goal = current_position.copy()
        return self._hold_result(
            requested,
            current_position,
            current_normal,
            emergency_action,
            last_reason,
            guard_safe=False,
            one_step_minimum=last_one_step,
            braking_minimum=last_braking,
        )


class StockTaskFrameNoSafeRecoveryV13(RuntimeError):
    """No recovery command passed the same online guard as policy actions."""


class StockGripperTaskFrameAdapterV13(StockGripperTaskFrameAdapterV12):
    """V12 task-frame control with fail-closed, guard-verified recovery.

    V12 verified ordinary policy candidates but, after every candidate failed,
    submitted the first merely representable inward recovery target directly.
    A deterministic Phase-A replay proved that path could cross the 0.25 mm
    hard clearance.  V13 keeps the V12 action space and latched-target hold,
    but requires every recovery target to pass the identical one-step and
    receding-horizon backup-policy forecast before it can be returned to the
    plant.
    """

    def begin_episode(self, seed: int) -> None:
        super().begin_episode(seed)
        self.env.episode_domain["stock_gripper_taskframe_runtime_v13"] = {
            "format": "edgearm-stock-gripper-taskframe-runtime-v13",
            "seed": int(seed),
            "source_type": SOURCE_TYPE,
            "base_reset_contract": "edgearm-stock-gripper-taskframe-reset-v12",
            "policy_action_space_changed_from_v12": False,
            "recovery_requires_same_guard_as_policy_action": True,
            "recovery_continuation_policy": (
                "receding_horizon_workspace_anchor_recovery"
            ),
            "recovery_continuation_exact_plant_forecast": True,
            "unguarded_recovery_submission": False,
            "no_safe_recovery_behavior": "raise_before_env_step",
            "simulator_privileged": True,
            "production_admission": False,
        }

    def _encode_policy_target_action(
        self,
        proposed_target: np.ndarray,
        *,
        active_request: bool,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Project active IK targets to the exact executable command.

        A V10 workspace filter can move a mathematically requested IK target
        by a small amount even when the corresponding float32 command is safe
        and task-aligned.  Active motion requests may use that executable
        command, which is subsequently checked by the unchanged exact guard
        and direction test.  Fixed-target holds retain V12's strict identity.
        """

        try:
            return super()._encode_policy_target_action(
                proposed_target,
                active_request=active_request,
            )
        except LatchedAbsoluteTargetInfeasibleV3 as error:
            if not active_request:
                raise
            if self.guard is None:  # pragma: no cover - guarded by translate
                raise RuntimeError(
                    "stock task-frame V13 target projection needs guard"
                ) from error
            target = np.asarray(proposed_target, dtype=np.float64)
            reference = np.asarray(
                self.env._command_reference_reported_position(),
                dtype=np.float64,
            ) - np.asarray(self.env._zero_offset, dtype=np.float64)
            raw_action = (target - reference) / float(
                self.env.config.max_joint_delta
            )
            action = np.clip(raw_action, -1.0, 1.0).astype(np.float32)
            reconstructed = self.guard.absolute_target_for_action(action)
            executable_target, executable_filter_reason = (
                self.env._safety_filter(reconstructed)
            )
            identity = {
                "format": (
                    "edgearm-stock-taskframe-executable-policy-target-"
                    "projection-v1"
                ),
                "mode": "filtered_float32_command_projection",
                "active_request": True,
                "requested_absolute_target_rad": target.tolist(),
                "observable_reference_rad": reference.tolist(),
                "raw_action_before_clip": raw_action.tolist(),
                "action_clipped": bool(np.any(np.abs(raw_action) > 1.0)),
                "float32_action_hex": action.tobytes().hex(),
                "reconstructed_absolute_target_rad": (
                    reconstructed.tolist()
                ),
                "executable_filtered_target_rad": np.asarray(
                    executable_target, dtype=np.float64
                ).tolist(),
                "execution_filter_reason": str(executable_filter_reason),
                "exact_absolute_target_encoding_error": str(error),
            }
            self.last_policy_target_encoding_report = identity
            return action, identity

    def _recovery_tail_anchor_rad(
        self,
        requested_recovery_target_rad: np.ndarray,
        workspace_recovery_anchor_rad: np.ndarray,
    ) -> np.ndarray:
        """V13 preserves its historical direct workspace-anchor tail."""

        del requested_recovery_target_rad
        return np.asarray(
            workspace_recovery_anchor_rad, dtype=np.float64
        ).copy()

    def _recovery_continuation_policy_name(self) -> str:
        return "receding_horizon_workspace_anchor_recovery"

    def _representable_interior_recovery_hold(
        self,
        current_q: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return only an inward recovery action with a complete guard proof."""

        if self.guard is None:
            raise RuntimeError("stock task-frame V13 recovery requires its guard")
        live = np.asarray(current_q, dtype=np.float64)
        if live.shape != (JOINT_ACTION_DIM,) or not np.all(np.isfinite(live)):
            raise ValueError("stock task-frame V13 recovery state is invalid")
        filtered, filter_reason = self.env._safety_filter(live)
        anchor_value = self.env._workspace_recovery_anchor_v10
        if anchor_value is None:
            raise StockTaskFrameNoSafeRecoveryV13("V13 has no workspace recovery anchor")
        anchor = np.asarray(anchor_value, dtype=np.float64)
        baseline_action: np.ndarray | None = None
        baseline_error = "none"
        if self._latched_joint_target is not None:
            try:
                baseline_action, _baseline_identity = self.guard.action_for_absolute_target(
                    self._latched_joint_target
                )
            except LatchedAbsoluteTargetInfeasibleV3 as error:
                baseline_error = str(error)

        failures: list[dict[str, Any]] = []
        encoding_adjustments: list[dict[str, Any]] = []
        for inward_scale in (0.02, 0.05, 0.10, 0.20, 0.40, 0.70, 1.0):
            candidate = filtered + float(inward_scale) * (anchor - filtered)
            candidate[5] = self.env.tool_gripper_joint_position_rad
            recovery_tail_anchor = self._recovery_tail_anchor_rad(
                candidate, anchor
            )
            if (
                recovery_tail_anchor.shape != (JOINT_ACTION_DIM,)
                or not np.all(np.isfinite(recovery_tail_anchor))
            ):
                raise RuntimeError(
                    "stock task-frame recovery tail anchor is invalid"
                )
            target_encoding_mode = "exact_absolute_target"
            exact_encoding_error = "none"
            try:
                candidate_action, identity = self.guard.action_for_absolute_target(candidate)
            except LatchedAbsoluteTargetInfeasibleV3 as error:
                # The requested mathematical target may change by a tiny
                # float32 reconstruction/safety-filter amount, or may be more
                # than one command step away.  Rejecting it here can create a
                # false safety deadlock even though the *executable* command
                # has a complete one-step and recursive-tail proof.  Project
                # the requested recovery into the physical command cube and
                # let the unchanged guard judge that exact float32 action.
                exact_encoding_error = str(error)
                reference = np.asarray(
                    self.env._command_reference_reported_position(),
                    dtype=np.float64,
                ) - np.asarray(self.env._zero_offset, dtype=np.float64)
                raw_action = (candidate - reference) / float(
                    self.env.config.max_joint_delta
                )
                candidate_action = np.clip(raw_action, -1.0, 1.0).astype(
                    np.float32
                )
                reconstructed = self.guard.absolute_target_for_action(
                    candidate_action
                )
                executable_target, executable_filter_reason = (
                    self.env._safety_filter(reconstructed)
                )
                target_encoding_mode = (
                    "filtered_float32_command_projection"
                )
                identity = {
                    "format": (
                        "edgearm-stock-taskframe-filtered-float32-"
                        "recovery-command-v1"
                    ),
                    "requested_absolute_target_rad": candidate.tolist(),
                    "observable_reference_rad": reference.tolist(),
                    "raw_action_before_clip": raw_action.tolist(),
                    "action_clipped": bool(np.any(np.abs(raw_action) > 1.0)),
                    "float32_action_hex": candidate_action.tobytes().hex(),
                    "reconstructed_absolute_target_rad": (
                        reconstructed.tolist()
                    ),
                    "executable_filtered_target_rad": np.asarray(
                        executable_target, dtype=np.float64
                    ).tolist(),
                    "execution_filter_reason": str(
                        executable_filter_reason
                    ),
                    "exact_absolute_target_encoding_error": (
                        exact_encoding_error
                    ),
                }
                encoding_adjustments.append(
                    {
                        "inward_scale": float(inward_scale),
                        "target_encoding_mode": target_encoding_mode,
                        "target_identity": identity,
                    }
                )
            recovery_baseline = candidate_action if baseline_action is None else baseline_action
            selected, guard_report = self.guard.select(
                candidate_action,
                baseline_action=recovery_baseline,
                require_hold_tail=True,
                recovery_tail_anchor_rad=recovery_tail_anchor,
            )
            if selected is None:
                attempts = list(guard_report["attempts"])
                failures.append(
                    {
                        "inward_scale": float(inward_scale),
                        "stage": "online_guard",
                        "target_encoding_mode": target_encoding_mode,
                        "exact_absolute_target_encoding_error": (
                            exact_encoding_error
                        ),
                        "candidate_identity": identity,
                        "attempts": attempts,
                    }
                )
                continue
            selected_forecast = dict(guard_report["selected_forecast"])
            candidate_rows = [
                row for row in selected_forecast["applications"] if bool(row["candidate_command"])
            ]
            if len(candidate_rows) != 1:
                raise RuntimeError("stock task-frame V13 lost recovery forecast identity")
            selected_target = np.asarray(
                candidate_rows[0]["physics_endpoint_rad"],
                dtype=np.float64,
            )
            position, _normal = self._pose_for_joint_position(selected_target)
            hold_tail = dict(guard_report["selected_hold_tail_forecast"])
            if not (
                bool(guard_report["safe_candidate_found"])
                and bool(hold_tail["hard_valid"])
                and bool(hold_tail["planning_margin_valid"])
                and bool(
                    hold_tail[
                        "remaining_submissions_follow_verified_backup_policy"
                    ]
                )
            ):
                raise RuntimeError("stock task-frame V13 accepted an incomplete guard proof")
            self.recovery_count += 1
            self.last_guard_report = guard_report
            self.last_recovery_report = {
                "format": "edgearm-stock-taskframe-guarded-interior-recovery-v13",
                "recovery_index": self.recovery_count,
                "live_joint_position_rad": live.tolist(),
                "filtered_boundary_target_rad": np.asarray(filtered).tolist(),
                "filter_reason": str(filter_reason),
                "workspace_recovery_anchor_rad": anchor.tolist(),
                "selected_inward_scale": float(inward_scale),
                "selected_guard_scale": float(guard_report["selected_scale"]),
                "selected_target_encoding_mode": target_encoding_mode,
                "selected_exact_absolute_target_encoding_error": (
                    exact_encoding_error
                ),
                "requested_recovery_target_rad": candidate.tolist(),
                "verified_recovery_tail_anchor_rad": (
                    recovery_tail_anchor.tolist()
                ),
                "selected_target_rad": selected_target.tolist(),
                "selected_tool_position_world_m": position.tolist(),
                "target_identity": identity,
                "minimum_one_step_clearance_m": float(
                    selected_forecast["minimum_forecast_safety_only_block_distance_m"]
                ),
                "minimum_braking_clearance_m": float(
                    hold_tail["minimum_forecast_safety_only_block_distance_m"]
                ),
                "rejected_candidates": failures,
                "target_encoding_adjustments": encoding_adjustments,
                "baseline_encoding_error": baseline_error,
                "same_guard_as_policy_action": True,
                "verified_continuation_policy": (
                    self._recovery_continuation_policy_name()
                ),
                "simulator_privileged": True,
                "production_admission": False,
            }
            return selected_target, np.asarray(selected, dtype=np.float32)

        self.last_recovery_report = {
            "format": "edgearm-stock-taskframe-guarded-interior-recovery-v13",
            "live_joint_position_rad": live.tolist(),
            "filtered_boundary_target_rad": np.asarray(filtered).tolist(),
            "filter_reason": str(filter_reason),
            "workspace_recovery_anchor_rad": anchor.tolist(),
            "rejected_candidates": failures,
            "target_encoding_adjustments": encoding_adjustments,
            "baseline_encoding_error": baseline_error,
            "same_guard_as_policy_action": True,
            "verified_continuation_policy": (
                self._recovery_continuation_policy_name()
            ),
            "safe_candidate_found": False,
            "simulator_privileged": True,
            "production_admission": False,
        }
        raise StockTaskFrameNoSafeRecoveryV13("V13 found no guard-verified interior recovery target")


class MultiViewRendererProtocolV1(Protocol):
    view_names: tuple[str, ...]

    def capture(self) -> tuple[np.ndarray, np.ndarray]: ...

    def begin_episode(self, seed: int) -> None: ...


class RolloutExecutionKernelProtocolV1(Protocol):
    """Versioned reset/contact/safety contract used by the shared collector."""

    format: str
    rollout_format: str
    evaluation_format: str
    safety_only_geom_count: int
    contact_candidate_geom_count: int
    contact_identity_format: str
    safety_guard_format: str

    def validate(
        self,
        env: RealisticEdgeArmEnvV10,
        action_adapter: StockGripperTaskFrameAdapterV13,
        reward: ScratchPotentialRewardV6Candidate,
    ) -> None: ...

    def reset_episode(
        self,
        env: RealisticEdgeArmEnvV10,
        renderer: MultiViewRendererProtocolV1,
        action_adapter: StockGripperTaskFrameAdapterV13,
        *,
        requested_seed: int,
        obstacle: bool,
        stress: bool,
    ) -> dict[str, Any]: ...

    def transition_contact(
        self,
        info: dict[str, Any],
        *,
        block_before_xy_m: np.ndarray,
        block_after_xy_m: np.ndarray,
    ) -> dict[str, Any]: ...

    def transition_safety(
        self,
        reward: ScratchPotentialRewardV6Candidate,
        env: RealisticEdgeArmEnvV10,
        info: dict[str, Any],
    ) -> ScratchSafetyEvidenceV6Candidate: ...

    def static_safety(
        self,
        reward: ScratchPotentialRewardV6Candidate,
        env: RealisticEdgeArmEnvV10,
    ) -> tuple[ScratchSafetyEvidenceV6Candidate, float]: ...

    def minimum_safety_only_clearance(
        self,
        telemetry: dict[str, Any],
    ) -> float: ...


def reset_stock_taskframe_episode_v11(
    env: RealisticEdgeArmEnvV10,
    renderer: MultiViewRendererProtocolV1,
    action_adapter: StockGripperTaskFrameAdapterV9,
    *,
    requested_seed: int,
    obstacle: bool,
    stress: bool,
) -> dict[str, Any]:
    """Install one valid randomized reset with bounded deterministic retries.

    Only the explicitly classified reset-infeasible exception is retryable.
    Programming errors and simulator failures remain fatal instead of being
    silently converted into a different data distribution.
    """

    if type(env) is not RealisticEdgeArmEnvV10:
        raise TypeError("stock reset retry requires exact RealisticEdgeArmEnvV10")
    if type(action_adapter) is not StockGripperTaskFrameAdapterV9:
        raise TypeError("stock reset retry requires exact task-frame adapter V9")
    if action_adapter.env is not env:
        raise ValueError("stock reset retry adapter belongs to another environment")
    if tuple(renderer.view_names) != VIEW_NAMES:
        raise ValueError("stock reset retry renderer view order changed")
    if type(requested_seed) is not int or requested_seed < 0:
        raise ValueError("stock reset retry seed must be a non-negative integer")
    if type(obstacle) is not bool or type(stress) is not bool:
        raise TypeError("stock reset retry conditions must be booleans")

    rejected: list[dict[str, Any]] = []
    for attempt_index in range(MAX_RESET_ATTEMPTS_V11):
        selected_seed = requested_seed + attempt_index * RESET_RETRY_STRIDE_V11
        env.reset(seed=selected_seed, obstacle=obstacle, stress=stress)
        try:
            action_adapter.begin_episode(selected_seed)
        except StockTaskFrameResetInfeasibleV11 as error:
            rejected.append(
                {
                    "attempt_index": attempt_index,
                    "seed": selected_seed,
                    "error_type": type(error).__name__,
                    "reason": str(error),
                }
            )
            continue

        renderer.begin_episode(selected_seed)
        audit = {
            "format": "edgearm-stock-taskframe-reset-retry-v11",
            "requested_seed": requested_seed,
            "selected_seed": selected_seed,
            "selected_attempt_index": attempt_index,
            "maximum_attempts": MAX_RESET_ATTEMPTS_V11,
            "retry_stride": RESET_RETRY_STRIDE_V11,
            "rejected_attempts": rejected,
            "obstacle": obstacle,
            "stress": stress,
            "source_type": SOURCE_TYPE,
            "expert_calls": 0,
            "expert_paths": 0,
            "deployment_reset_equivalent": False,
            "production_admission": False,
        }
        env.episode_domain["stock_taskframe_reset_retry_v11"] = _json_safe(audit)
        return audit

    reasons = "; ".join(f"seed={item['seed']}:{item['reason']}" for item in rejected)
    raise StockTaskFrameResetInfeasibleV11(
        f"stock task-frame reset exhausted {MAX_RESET_ATTEMPTS_V11} attempts: {reasons}"
    )


def reset_stock_taskframe_episode_v12(
    env: RealisticEdgeArmEnvV10,
    renderer: MultiViewRendererProtocolV1,
    action_adapter: StockGripperTaskFrameAdapterV12,
    *,
    requested_seed: int,
    obstacle: bool,
    stress: bool,
) -> dict[str, Any]:
    """Install one dynamically hold-safe V12 randomized reset."""

    if type(env) is not RealisticEdgeArmEnvV10:
        raise TypeError("stock V12 reset retry requires exact V10")
    if not isinstance(action_adapter, StockGripperTaskFrameAdapterV12):
        raise TypeError("stock V12 reset retry requires a V12-compatible adapter")
    if action_adapter.env is not env:
        raise ValueError("stock V12 reset adapter belongs to another environment")
    if tuple(renderer.view_names) != VIEW_NAMES:
        raise ValueError("stock V12 reset renderer view order changed")
    if type(requested_seed) is not int or requested_seed < 0:
        raise ValueError("stock V12 reset seed must be non-negative")
    if type(obstacle) is not bool or type(stress) is not bool:
        raise TypeError("stock V12 reset conditions must be booleans")

    maximum_attempts = (
        MAX_CURRICULUM_RESET_ATTEMPTS_V16
        if action_adapter.config.curriculum_reset_tip_gap_band_m is not None
        else MAX_RESET_ATTEMPTS_V12
    )
    rejected: list[dict[str, Any]] = []
    for attempt_index in range(maximum_attempts):
        selected_seed = requested_seed + attempt_index * RESET_RETRY_STRIDE_V12
        try:
            env.reset(seed=selected_seed, obstacle=obstacle, stress=stress)
            action_adapter.begin_episode(selected_seed)
        except (
            StockTaskFrameResetInfeasibleV12,
            V10SafetyFilterInfeasible,
        ) as error:
            rejected.append(
                {
                    "attempt_index": attempt_index,
                    "seed": selected_seed,
                    "error_type": type(error).__name__,
                    "reason": str(error),
                }
            )
            continue
        renderer.begin_episode(selected_seed)
        audit = {
            "format": "edgearm-stock-taskframe-reset-retry-v12",
            "requested_seed": requested_seed,
            "selected_seed": selected_seed,
            "selected_attempt_index": attempt_index,
            "maximum_attempts": maximum_attempts,
            "retry_stride": RESET_RETRY_STRIDE_V12,
            "rejected_attempts": rejected,
            "obstacle": obstacle,
            "stress": stress,
            "source_type": SOURCE_TYPE,
            "expert_calls": 0,
            "expert_paths": 0,
            "deployment_reset_equivalent": False,
            "production_admission": False,
        }
        env.episode_domain["stock_taskframe_reset_retry_v12"] = _json_safe(audit)
        return audit

    reasons = "; ".join(f"seed={item['seed']}:{item['reason']}" for item in rejected)
    raise StockTaskFrameResetInfeasibleV12(
        f"stock task-frame V12 reset exhausted {maximum_attempts} attempts: {reasons}"
    )


def reset_stock_taskframe_episode_v13(
    env: RealisticEdgeArmEnvV10,
    renderer: MultiViewRendererProtocolV1,
    action_adapter: StockGripperTaskFrameAdapterV13,
    *,
    requested_seed: int,
    obstacle: bool,
    stress: bool,
) -> dict[str, Any]:
    """Install the V12 reset plus the V13 fail-closed runtime contract."""

    if type(action_adapter) is not StockGripperTaskFrameAdapterV13:
        raise TypeError("stock V13 reset retry requires exact adapter V13")
    base = reset_stock_taskframe_episode_v12(
        env,
        renderer,
        action_adapter,
        requested_seed=requested_seed,
        obstacle=obstacle,
        stress=stress,
    )
    audit = dict(base)
    audit.update(
        {
            "format": "edgearm-stock-taskframe-reset-retry-v13",
            "maximum_attempts": int(base["maximum_attempts"]),
            "retry_stride": RESET_RETRY_STRIDE_V13,
            "base_reset_format": str(base["format"]),
            "recovery_requires_same_guard_as_policy_action": True,
            "unguarded_recovery_submission": False,
        }
    )
    env.episode_domain["stock_taskframe_reset_retry_v13"] = _json_safe(audit)
    return audit


class MultiViewRGBRendererV1:
    """One MuJoCo renderer reused across four named cameras."""

    view_names = VIEW_NAMES

    def __init__(
        self,
        env: RealisticEdgeArmEnvV10,
        *,
        height: int,
        width: int,
    ) -> None:
        if type(env) is not RealisticEdgeArmEnvV10:
            raise TypeError("multiview renderer requires exact RealisticEdgeArmEnvV10")
        if type(height) is not int or type(width) is not int or height < 32 or width < 32:
            raise ValueError("multiview renderer resolution is invalid")
        self.env = env
        self.height = height
        self.width = width
        self.camera_ids = tuple(
            int(mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, name)) for name in CAMERA_NAMES
        )
        missing = [name for name, camera_id in zip(CAMERA_NAMES, self.camera_ids) if camera_id < 0]
        if missing:
            raise ValueError(f"multiview scene is missing cameras: {missing}")
        self.renderer = mujoco.Renderer(env.model, height=height, width=width)
        self.episode_seed = env.seed

    def begin_episode(self, seed: int) -> None:
        self.episode_seed = int(seed)

    def capture(self) -> tuple[np.ndarray, np.ndarray]:
        mujoco.mj_forward(self.env.model, self.env.data)
        images: list[np.ndarray] = []
        poses: list[np.ndarray] = []
        for camera_name, camera_id in zip(CAMERA_NAMES, self.camera_ids):
            self.renderer.update_scene(self.env.data, camera=camera_name)
            image = np.asarray(self.renderer.render()).copy()
            if image.shape != (self.height, self.width, 3) or image.dtype != np.uint8:
                raise RuntimeError("MuJoCo multiview RGB shape/dtype changed")
            images.append(image)
            poses.append(
                np.concatenate(
                    (
                        self.env.data.cam_xpos[camera_id],
                        self.env.data.cam_xmat[camera_id],
                    )
                ).astype(np.float32)
            )
        return np.stack(images), np.stack(poses)

    def close(self) -> None:
        self.renderer.close()


@dataclass(frozen=True)
class AsymmetricMultiViewRolloutBatchV1:
    rgb_frames: np.ndarray
    policy_rgb_history: np.ndarray
    joint_state: np.ndarray
    policy_joint_history: np.ndarray
    previous_executed_action: np.ndarray
    policy_action_history: np.ndarray
    history_valid: np.ndarray
    view_valid: np.ndarray
    policy_view_history_valid: np.ndarray
    history_row_indices: np.ndarray
    camera_pose: np.ndarray
    privileged_state: np.ndarray
    next_privileged_state: np.ndarray
    visual_geometry_target: np.ndarray
    policy_action: np.ndarray
    previous_policy_pre_tanh: np.ndarray
    applied_task_action: np.ndarray
    executed_action: np.ndarray
    submitted_joint_action: np.ndarray
    execution_attempted: np.ndarray
    shield_rejected_before_step: np.ndarray
    ik_target_position_world_m: np.ndarray
    ik_target_face_normal_world: np.ndarray
    ik_application_scale: np.ndarray
    ik_converged: np.ndarray
    ik_face_label: np.ndarray
    ik_failure_reason: np.ndarray
    guard_safe_candidate: np.ndarray
    guard_selected_scale: np.ndarray
    guard_minimum_one_step_clearance_m: np.ndarray
    guard_minimum_braking_clearance_m: np.ndarray
    guard_float32_execution_identity: np.ndarray
    tool_block_contact_any: np.ndarray
    tool_block_contact_substep_count: np.ndarray
    valid_push_side_contact_any: np.ndarray
    valid_push_side_contact_substep_count: np.ndarray
    valid_push_side_contact_transient_only: np.ndarray
    invalid_tool_block_contact_any: np.ndarray
    valid_push_side_peak_normal_force_n: np.ndarray
    valid_push_side_normal_impulse_discrete_ns: np.ndarray
    minimum_executed_94_safety_only_block_clearance_m: np.ndarray
    step_block_displacement_m: np.ndarray
    pre_tanh: np.ndarray
    old_log_probs: np.ndarray
    raw_environment_reward: np.ndarray
    reward_before_potential: np.ndarray
    potential_before: np.ndarray
    potential_after: np.ndarray
    potential_next_for_shaping: np.ndarray
    shaped_rewards: np.ndarray
    values: np.ndarray
    next_values: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    strict_success: np.ndarray
    terminal_failure: np.ndarray
    safety_stop: np.ndarray
    terminal_reason: np.ndarray
    episode_ids: np.ndarray
    episode_step_ids: np.ndarray
    task_ids: np.ndarray
    requested_joint_target: np.ndarray
    queued_safe_joint_target: np.ndarray
    applied_joint_target: np.ndarray
    sim_qpos: np.ndarray
    sim_qvel: np.ndarray
    episode_records: tuple[dict[str, Any], ...]
    shaping_gamma: float
    potential_reward_config_sha256: str
    rollout_seed: int
    execution_kernel_format: str = "edgearm-stock-taskframe-rollout-kernel-v13"
    safety_only_geom_count: int = 94
    contact_candidate_geom_count: int = 2
    contact_identity_format: str = "edgearm-tool-contact-identity-v2"
    safety_guard_format: str = STOCK_GRIPPER_ACTION_GUARD_FORMAT_V3
    source_type: str = SOURCE_TYPE
    rollout_format: str = ROLLOUT_FORMAT

    @property
    def rewards(self) -> np.ndarray:
        return self.shaped_rewards

    @property
    def completed_episode_count(self) -> int:
        return int(np.count_nonzero(self.terminated | self.truncated))

    def validate(self) -> None:
        count, views, height, width, channels = np.asarray(self.rgb_frames).shape
        history = np.asarray(self.policy_rgb_history).shape[1]
        shapes = {
            "rgb_frames": (count, len(VIEW_NAMES), height, width, 3),
            "policy_rgb_history": (count, history, len(VIEW_NAMES), height, width, 3),
            "joint_state": (count, JOINT_STATE_DIM),
            "policy_joint_history": (count, history, JOINT_STATE_DIM),
            "previous_executed_action": (count, ACTION_DIM),
            "policy_action_history": (count, history, ACTION_DIM),
            "history_valid": (count, history),
            "view_valid": (count, len(VIEW_NAMES)),
            "policy_view_history_valid": (count, history, len(VIEW_NAMES)),
            "history_row_indices": (count, history),
            "camera_pose": (count, len(VIEW_NAMES), 12),
            "privileged_state": (count, PRIVILEGED_EFFECT_STATE_DIM),
            "next_privileged_state": (count, PRIVILEGED_EFFECT_STATE_DIM),
            "visual_geometry_target": (count, VISUAL_GEOMETRY_TARGET_DIM),
            "policy_action": (count, POLICY_ACTION_DIM),
            "previous_policy_pre_tanh": (count, POLICY_ACTION_DIM),
            "applied_task_action": (count, POLICY_ACTION_DIM),
            "executed_action": (count, JOINT_ACTION_DIM),
            "submitted_joint_action": (count, JOINT_ACTION_DIM),
            "execution_attempted": (count,),
            "shield_rejected_before_step": (count,),
            "ik_target_position_world_m": (count, 3),
            "ik_target_face_normal_world": (count, 3),
            "ik_application_scale": (count,),
            "ik_converged": (count,),
            "ik_face_label": (count,),
            "ik_failure_reason": (count,),
            "guard_safe_candidate": (count,),
            "guard_selected_scale": (count,),
            "guard_minimum_one_step_clearance_m": (count,),
            "guard_minimum_braking_clearance_m": (count,),
            "guard_float32_execution_identity": (count,),
            "tool_block_contact_any": (count,),
            "tool_block_contact_substep_count": (count,),
            "valid_push_side_contact_any": (count,),
            "valid_push_side_contact_substep_count": (count,),
            "valid_push_side_contact_transient_only": (count,),
            "invalid_tool_block_contact_any": (count,),
            "valid_push_side_peak_normal_force_n": (count,),
            "valid_push_side_normal_impulse_discrete_ns": (count,),
            "minimum_executed_94_safety_only_block_clearance_m": (count,),
            "step_block_displacement_m": (count,),
            "pre_tanh": (count, POLICY_ACTION_DIM),
            "old_log_probs": (count,),
            "raw_environment_reward": (count,),
            "reward_before_potential": (count,),
            "potential_before": (count,),
            "potential_after": (count,),
            "potential_next_for_shaping": (count,),
            "shaped_rewards": (count,),
            "values": (count,),
            "next_values": (count,),
            "terminated": (count,),
            "truncated": (count,),
            "strict_success": (count,),
            "terminal_failure": (count,),
            "safety_stop": (count,),
            "terminal_reason": (count,),
            "episode_ids": (count,),
            "episode_step_ids": (count,),
            "task_ids": (count,),
            "requested_joint_target": (count, ACTION_DIM),
            "queued_safe_joint_target": (count, ACTION_DIM),
            "applied_joint_target": (count, ACTION_DIM),
            "sim_qpos": (count, 13),
            "sim_qvel": (count, 12),
        }
        if count < 1 or views != len(VIEW_NAMES) or channels != 3 or history < 1:
            raise ValueError("multiview rollout leading dimensions are invalid")
        for name, shape in shapes.items():
            if np.asarray(getattr(self, name)).shape != shape:
                raise ValueError(f"multiview rollout {name} shape mismatch")
        if self.rgb_frames.dtype != np.uint8 or self.policy_rgb_history.dtype != np.uint8:
            raise ValueError("multiview rollout images must be uint8")
        for name in (
            "history_valid",
            "view_valid",
            "policy_view_history_valid",
            "terminated",
            "truncated",
            "strict_success",
            "terminal_failure",
            "safety_stop",
            "ik_converged",
            "guard_safe_candidate",
            "guard_float32_execution_identity",
            "execution_attempted",
            "shield_rejected_before_step",
            "tool_block_contact_any",
            "valid_push_side_contact_any",
            "valid_push_side_contact_transient_only",
            "invalid_tool_block_contact_any",
        ):
            if np.asarray(getattr(self, name)).dtype != np.dtype(bool):
                raise ValueError(f"multiview rollout {name} must be boolean")
        for name in (
            "joint_state",
            "policy_joint_history",
            "previous_executed_action",
            "policy_action_history",
            "camera_pose",
            "privileged_state",
            "next_privileged_state",
            "visual_geometry_target",
            "policy_action",
            "previous_policy_pre_tanh",
            "applied_task_action",
            "executed_action",
            "submitted_joint_action",
            "ik_target_position_world_m",
            "ik_target_face_normal_world",
            "ik_application_scale",
            "guard_selected_scale",
            "guard_minimum_one_step_clearance_m",
            "guard_minimum_braking_clearance_m",
            "valid_push_side_peak_normal_force_n",
            "valid_push_side_normal_impulse_discrete_ns",
            "minimum_executed_94_safety_only_block_clearance_m",
            "step_block_displacement_m",
            "pre_tanh",
            "old_log_probs",
            "raw_environment_reward",
            "reward_before_potential",
            "potential_before",
            "potential_after",
            "potential_next_for_shaping",
            "shaped_rewards",
            "values",
            "next_values",
            "requested_joint_target",
            "queued_safe_joint_target",
            "applied_joint_target",
            "sim_qpos",
            "sim_qvel",
        ):
            value = np.asarray(getattr(self, name))
            if value.dtype != np.float32 or not np.all(np.isfinite(value)):
                raise ValueError(f"multiview rollout {name} must be finite float32")
        if np.asarray(self.history_row_indices).dtype != np.dtype(np.int64):
            raise ValueError("history_row_indices must be int64")
        if np.asarray(self.episode_ids).dtype != np.dtype(np.int64):
            raise ValueError("episode_ids must be int64")
        if np.asarray(self.episode_step_ids).dtype != np.dtype(np.int64):
            raise ValueError("episode_step_ids must be int64")
        if np.asarray(self.task_ids).dtype != np.dtype(np.int64):
            raise ValueError("task_ids must be int64")
        for name in (
            "tool_block_contact_substep_count",
            "valid_push_side_contact_substep_count",
        ):
            value = np.asarray(getattr(self, name))
            if value.dtype != np.dtype(np.int64) or np.any(value < 0):
                raise ValueError(f"multiview rollout {name} must be non-negative int64")
        for name in ("terminal_reason", "ik_face_label", "ik_failure_reason"):
            if np.asarray(getattr(self, name)).dtype.kind not in {"U", "S"}:
                raise ValueError(f"{name} must be a string vector")
        if self.source_type != SOURCE_TYPE:
            raise ValueError("multiview rollout source identity mismatch")
        if not isinstance(self.rollout_format, str) or not self.rollout_format:
            raise ValueError("multiview rollout format is missing")
        if (
            self.execution_kernel_format == "edgearm-stock-taskframe-rollout-kernel-v13"
            and self.rollout_format != ROLLOUT_FORMAT
        ):
            raise ValueError("historical V13 rollout format changed")
        if type(self.rollout_seed) is not int or self.rollout_seed < 0:
            raise ValueError("multiview rollout seed is invalid")
        if not isinstance(self.execution_kernel_format, str) or not self.execution_kernel_format:
            raise ValueError("multiview execution kernel format is missing")
        if (
            type(self.safety_only_geom_count) is not int
            or type(self.contact_candidate_geom_count) is not int
            or self.safety_only_geom_count < 1
            or self.contact_candidate_geom_count < 2
            or self.safety_only_geom_count + self.contact_candidate_geom_count != 96
        ):
            raise ValueError("multiview stock safety/contact cardinality is invalid")
        if not isinstance(self.contact_identity_format, str) or not self.contact_identity_format:
            raise ValueError("multiview contact identity format is missing")
        if not isinstance(self.safety_guard_format, str) or not self.safety_guard_format:
            raise ValueError("multiview safety guard format is missing")
        if not np.isfinite(self.shaping_gamma) or not 0.0 < self.shaping_gamma <= 1.0:
            raise ValueError("multiview shaping gamma is invalid")
        if not isinstance(self.potential_reward_config_sha256, str):
            raise ValueError("multiview reward config hash is missing")
        if np.any(np.abs(self.policy_action) > 1.0 + 1.0e-6):
            raise ValueError("multiview policy action escaped tanh bounds")
        if np.any(np.abs(self.applied_task_action) > 1.0 + 1.0e-6):
            raise ValueError("multiview applied task action escaped bounds")
        if np.any(np.abs(self.submitted_joint_action) > 1.0 + 1.0e-6):
            raise ValueError("multiview submitted joint action escaped bounds")
        if np.any(self.execution_attempted & self.shield_rejected_before_step):
            raise ValueError("shield-rejected transitions cannot execute a command")
        if np.any(~self.execution_attempted & ~self.shield_rejected_before_step):
            raise ValueError("non-executed transitions require a shield-rejection reason")
        if np.any(
            self.shield_rejected_before_step
            & ~(self.terminated & self.terminal_failure & self.safety_stop & ~self.strict_success)
        ):
            raise ValueError("shield rejection must be a terminal safety failure")
        expected_geometry_target = visual_geometry_target_from_privileged_v1(self.privileged_state)
        if not np.array_equal(self.visual_geometry_target, expected_geometry_target):
            raise ValueError("visual geometry target differs from its privileged source")
        if np.any((self.ik_application_scale < 0.0) | (self.ik_application_scale > 1.0)):
            raise ValueError("multiview IK application scale escaped [0,1]")
        if np.any(self.ik_converged != (self.ik_application_scale > 0.0)):
            raise ValueError("multiview IK convergence and application scale disagree")
        if np.any((self.guard_selected_scale < 0.0) | (self.guard_selected_scale > 1.0)):
            raise ValueError("multiview guard selected scale escaped [0,1]")
        if np.any(self.execution_attempted & ~self.guard_float32_execution_identity):
            raise ValueError("executed command lost float32 forecast/execution identity")
        if np.any(
            self.shield_rejected_before_step
            & (
                self.guard_safe_candidate
                | self.guard_float32_execution_identity
                | (self.guard_selected_scale != 0.0)
            )
        ):
            raise ValueError("shield-rejected row incorrectly claims an accepted guard action")
        if np.any(
            self.shield_rejected_before_step
            & (
                np.any(self.applied_task_action != 0.0, axis=1)
                | np.any(self.executed_action != 0.0, axis=1)
                | np.any(self.submitted_joint_action != 0.0, axis=1)
            )
        ):
            raise ValueError("shield-rejected row must not contain an executed command")
        if np.any(self.valid_push_side_contact_any & ~self.tool_block_contact_any):
            raise ValueError("valid push-side contact requires raw tool/block contact")
        if np.any(self.valid_push_side_contact_transient_only & ~self.valid_push_side_contact_any):
            raise ValueError("transient contact marker requires valid push-side contact")
        if np.any(self.valid_push_side_contact_any != (self.valid_push_side_contact_substep_count > 0)):
            raise ValueError("valid contact summary disagrees with substep count")
        if np.any(self.tool_block_contact_any != (self.tool_block_contact_substep_count > 0)):
            raise ValueError("raw contact summary disagrees with substep count")
        if np.any(self.valid_push_side_peak_normal_force_n < 0.0):
            raise ValueError("valid contact peak force must be non-negative")
        if np.any(self.valid_push_side_normal_impulse_discrete_ns < 0.0):
            raise ValueError("valid contact impulse must be non-negative")
        if np.any(self.step_block_displacement_m < 0.0):
            raise ValueError("step block displacement must be non-negative")
        if not np.allclose(self.policy_action, np.tanh(self.pre_tanh), rtol=1.0e-6, atol=1.0e-6):
            raise ValueError("multiview policy action is not tanh(pre_tanh)")
        requested_ar_continuity = True
        applied_ar_continuity = True
        ik_guarded_ar_continuity = True
        for row in range(count):
            if self.episode_step_ids[row] == 0:
                if not np.array_equal(
                    self.previous_policy_pre_tanh[row],
                    np.zeros(POLICY_ACTION_DIM, dtype=np.float32),
                ):
                    raise ValueError("AR policy state did not reset at episode boundary")
            elif row == 0 or self.episode_ids[row - 1] != self.episode_ids[row]:
                raise ValueError("AR policy state has a discontinuous episode lineage")
            else:
                requested_ar_continuity &= np.array_equal(
                    self.previous_policy_pre_tanh[row],
                    self.pre_tanh[row - 1],
                )
                applied_ar_continuity &= np.array_equal(
                    self.previous_policy_pre_tanh[row],
                    autoregressive_applied_feedback_state_v21(self.applied_task_action[row - 1]),
                )
                ik_guarded_ar_continuity &= np.array_equal(
                    self.previous_policy_pre_tanh[row],
                    autoregressive_ik_guarded_feedback_state_v21(
                        self.pre_tanh[row - 1],
                        ik_converged=bool(self.ik_converged[row - 1]),
                    ),
                )
        if not any(
            (
                requested_ar_continuity,
                applied_ar_continuity,
                ik_guarded_ar_continuity,
            )
        ):
            raise ValueError("AR policy state has no supported requested, applied, or IK-guarded continuity")
        if np.any(self.terminated & self.truncated):
            raise ValueError("transition cannot terminate and truncate")
        if not bool((self.terminated | self.truncated)[-1]):
            raise ValueError("multiview rollout must end at an episode boundary")
        if np.any(self.strict_success & ~self.terminated):
            raise ValueError("strict success must be terminal")
        if np.any(~self.view_valid[:, 0]):
            raise ValueError("wrist view must remain available in the V1 actor")

        indices = np.asarray(self.history_row_indices)
        if not np.array_equal(self.history_valid, indices >= 0):
            raise ValueError("history validity and row indices disagree")
        for row in range(count):
            valid_indices = indices[row][indices[row] >= 0]
            if valid_indices.size < 1 or int(valid_indices[-1]) != row:
                raise ValueError("history must end at the current decision row")
            if np.any(valid_indices > row):
                raise ValueError("future row leaked into causal policy history")
            if np.any(self.episode_ids[valid_indices] != self.episode_ids[row]):
                raise ValueError("policy history crossed an episode boundary")
            offset = history - len(valid_indices)
            if not np.array_equal(
                self.policy_rgb_history[row, offset:],
                self.rgb_frames[valid_indices],
            ):
                raise ValueError("stored RGB history is not a causal row projection")
            if not np.array_equal(
                self.policy_joint_history[row, offset:],
                self.joint_state[valid_indices],
            ):
                raise ValueError("stored joint history is not a causal row projection")
            if not np.array_equal(
                self.policy_action_history[row, offset:],
                self.previous_executed_action[valid_indices],
            ):
                raise ValueError("stored action history is not previously executed history")
            if not np.array_equal(
                self.policy_view_history_valid[row, offset:],
                self.view_valid[valid_indices],
            ):
                raise ValueError("stored view history differs from per-row masks")


@dataclass(frozen=True)
class PPOUpdateMetricsV1:
    optimizer_steps: int
    epochs_completed: int
    target_kl_triggered: bool
    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    policy_clip_fraction: float
    value_clip_fraction: float
    visual_geometry_auxiliary_loss: float
    visual_geometry_auxiliary_rmse: float
    maximum_actor_preclip_gradient_norm: float
    maximum_critic_preclip_gradient_norm: float
    maximum_preclip_gradient_norm: float
    advantage_normalization: str = "global"
    advantage_group_count: int = 1


def _actor_tensors(
    batch: AsymmetricMultiViewRolloutBatchV1,
    device: torch.device,
    indices: np.ndarray | None = None,
) -> tuple[torch.Tensor, ...]:
    selected: Any = slice(None) if indices is None else indices
    return (
        torch.from_numpy(batch.policy_rgb_history[selected]).to(device),
        torch.from_numpy(batch.policy_joint_history[selected]).to(device),
        torch.from_numpy(batch.policy_action_history[selected]).to(device),
        torch.from_numpy(batch.history_valid[selected]).to(device),
        torch.from_numpy(batch.policy_view_history_valid[selected]).to(device),
        torch.from_numpy(batch.task_ids[selected]).to(device=device, dtype=torch.long),
    )


def _sample_view_valid_v1(rng: np.random.Generator, dropout_probability: float) -> np.ndarray:
    mask = np.ones(len(VIEW_NAMES), dtype=bool)
    mask[1:] = rng.random(len(VIEW_NAMES) - 1) >= dropout_probability
    mask[0] = True
    return mask


def transition_contact_telemetry_v10(
    info: dict[str, Any],
    *,
    block_before_xy_m: np.ndarray,
    block_after_xy_m: np.ndarray,
) -> dict[str, bool | float | int]:
    """Extract contact evidence over every physics substep of one decision.

    A contact can push the block and separate before the decision-rate frame is
    observed.  Frame-end contact counts therefore undercount valid pushing.
    V10 persists the authoritative V7 substep trace and explicitly marks the
    transient-only case; none of these privileged labels are actor inputs.
    """

    trace = info.get("physics_substep_contact_v1")
    if not isinstance(trace, dict):
        raise RuntimeError("V10 rollout lost physics-substep contact telemetry")
    physics_substeps = int(trace.get("physics_substeps", -1))
    safety_ids = tuple(int(value) for value in trace.get("tool_safety_geom_ids", ()))
    contact_ids = tuple(int(value) for value in trace.get("tool_contact_geom_ids", ()))
    if (
        physics_substeps < 1
        or len(safety_ids) != 96
        or len(contact_ids) != 2
        or len(set(contact_ids)) != 2
        or not set(contact_ids).issubset(safety_ids)
    ):
        raise RuntimeError("V10 rollout contact geometry identity changed")
    distances = np.asarray(
        trace.get("tool_safety_block_signed_distance_m", ()),
        dtype=np.float64,
    )
    raw_counts = np.asarray(trace.get("tool_block_contact_count", ()), dtype=np.int64)
    valid_counts = np.asarray(
        trace.get("valid_push_side_contact_count", ()),
        dtype=np.int64,
    )
    invalid_counts = np.asarray(
        trace.get("invalid_tool_block_contact_count", ()),
        dtype=np.int64,
    )
    expected_vector = (physics_substeps,)
    if (
        distances.shape != (physics_substeps, 96)
        or raw_counts.shape != expected_vector
        or valid_counts.shape != expected_vector
        or invalid_counts.shape != expected_vector
        or not np.all(np.isfinite(distances))
        or np.any(raw_counts < 0)
        or np.any(valid_counts < 0)
        or np.any(invalid_counts < 0)
    ):
        raise RuntimeError("V10 rollout contact substep arrays are invalid")
    contact_columns = {safety_ids.index(geom_id) for geom_id in contact_ids}
    safety_only_columns = [index for index in range(len(safety_ids)) if index not in contact_columns]
    if len(safety_only_columns) != 94:
        raise RuntimeError("V10 rollout requires 94 safety-only stock parts")
    before = np.asarray(block_before_xy_m, dtype=np.float64)
    after = np.asarray(block_after_xy_m, dtype=np.float64)
    if before.shape != (2,) or after.shape != (2,) or not np.all(np.isfinite(np.r_[before, after])):
        raise ValueError("V10 rollout block displacement endpoints are invalid")
    peak_force = float(trace.get("valid_push_side_peak_normal_force_n", 0.0))
    impulse = float(trace.get("valid_push_side_normal_impulse_discrete_ns", 0.0))
    if not np.isfinite(peak_force) or peak_force < 0.0:
        raise RuntimeError("V10 rollout valid-contact peak force is invalid")
    if not np.isfinite(impulse) or impulse < 0.0:
        raise RuntimeError("V10 rollout valid-contact impulse is invalid")
    valid_any = bool(np.any(valid_counts > 0))
    if valid_any != bool(trace.get("valid_push_side_contact_any", False)):
        raise RuntimeError("V10 rollout valid-contact summary disagrees with substeps")
    invalid_any = bool(np.any(invalid_counts > 0))
    if invalid_any != bool(trace.get("invalid_tool_block_contact_any", False)):
        raise RuntimeError("V10 rollout invalid-contact summary disagrees with substeps")
    return {
        "physics_substeps": physics_substeps,
        "tool_block_contact_any": bool(np.any(raw_counts > 0)),
        "tool_block_contact_substep_count": int(np.count_nonzero(raw_counts > 0)),
        "valid_push_side_contact_any": valid_any,
        "valid_push_side_contact_substep_count": int(np.count_nonzero(valid_counts > 0)),
        "valid_push_side_contact_transient_only": bool(valid_any and int(valid_counts[-1]) == 0),
        "invalid_tool_block_contact_any": invalid_any,
        "valid_push_side_peak_normal_force_n": peak_force,
        "valid_push_side_normal_impulse_discrete_ns": impulse,
        "minimum_executed_94_safety_only_block_clearance_m": float(np.min(distances[:, safety_only_columns])),
        "step_block_displacement_m": float(np.linalg.norm(after - before)),
    }


def _current_static_safety_evidence_v13(
    env: RealisticEdgeArmEnvV10,
    reward: ScratchPotentialRewardV6Candidate,
) -> tuple[ScratchSafetyEvidenceV6Candidate, float]:
    """Describe a safe state when the shield rejects before any physics step.

    This is deliberately not fabricated substep telemetry.  ``physics_substeps``
    is zero, all transition-contact counters are zero, and the current 96-part
    geometry is used only to prove that the pre-step state itself was safe.
    The terminal penalty comes from ``env_terminal_failure=True`` below, not
    from pretending that a collision occurred.
    """

    mujoco.mj_forward(env.model, env.data)
    safety_ids = tuple(int(value) for value in env._ids["tool_safety_geoms"])
    contact_ids = tuple(int(value) for value in env._ids["tool_contact_geoms"])
    if (
        len(safety_ids) != 96
        or len(contact_ids) != 2
        or len(set(contact_ids)) != 2
        or not set(contact_ids).issubset(safety_ids)
    ):
        raise RuntimeError("V13 shield terminal lost the stock 96-part geometry")
    contact_indices = tuple(safety_ids.index(value) for value in contact_ids)
    safety_only_indices = tuple(index for index in range(len(safety_ids)) if index not in contact_indices)
    if len(safety_only_indices) != 94:
        raise RuntimeError("V13 shield terminal did not resolve 94 safety-only parts")
    block_distances = np.asarray(
        env._tool_safety_signed_distances_for_data(env._ids["block_geom"], env.data),
        dtype=np.float64,
    )
    desk_distances = np.asarray(
        env._tool_safety_signed_distances_for_data(env._desk_geom, env.data),
        dtype=np.float64,
    )
    if (
        block_distances.shape != (96,)
        or desk_distances.shape != (96,)
        or not np.all(np.isfinite(block_distances))
        or not np.all(np.isfinite(desk_distances))
    ):
        raise RuntimeError("V13 shield terminal static geometry is invalid")
    safety_only = block_distances[np.asarray(safety_only_indices, dtype=np.int64)]
    limiting_local = int(np.argmin(safety_only))
    limiting_index = int(safety_only_indices[limiting_local])
    minimum_safety_only = float(safety_only[limiting_local])
    minimum_desk = float(np.min(desk_distances))
    if minimum_safety_only < reward.config.safety_only_block_clearance_m - 1.0e-12:
        raise RuntimeError(
            "V13 shield rejected after the live state had already crossed the 0.25 mm safety-only clearance"
        )
    if minimum_desk < -reward.config.penetration_tolerance_m - 1.0e-12:
        raise RuntimeError("V13 shield rejected after the live state had already penetrated the desk")
    evidence = ScratchSafetyEvidenceV6Candidate(
        physics_substeps=0,
        safety_geom_count=96,
        safety_only_geom_count=94,
        contact_safety_geom_indices=(
            int(contact_indices[0]),
            int(contact_indices[1]),
        ),
        minimum_safety_only_block_signed_distance_m=minimum_safety_only,
        limiting_safety_only_geom_index=limiting_index,
        minimum_contact_part_block_signed_distance_by_role_m=(
            float(block_distances[contact_indices[0]]),
            float(block_distances[contact_indices[1]]),
        ),
        minimum_full_safety_desk_signed_distance_m=minimum_desk,
        safety_only_clearance_violation=False,
        safety_only_penetration=False,
        unauthorized_contact_part_penetration=False,
        full_safety_desk_penetration=False,
        unauthorized_contact_part_penetration_count=0,
        authorized_contact_part_penetration_count=0,
        normalized_safety_only_block_cost=0.0,
        normalized_unauthorized_contact_cost=0.0,
        normalized_full_safety_desk_cost=0.0,
        normalized_safety_cost=0.0,
        safety_penalty=0.0,
        hard_safety_violation=False,
        hard_safety_reason="",
        privileged_reward_input=True,
        maximum_unauthorized_contact_penetration_depth_m=0.0,
    )
    return evidence, minimum_safety_only


def _rejected_guard_minimum_v13(
    report: object,
    key: str,
    *,
    fallback: float,
) -> float:
    values: list[float] = []

    def visit(value: object) -> None:
        if isinstance(value, dict):
            candidate = value.get(key)
            if candidate is not None and not isinstance(candidate, bool):
                try:
                    numeric = float(candidate)
                except (TypeError, ValueError):
                    numeric = float("nan")
                if np.isfinite(numeric):
                    values.append(numeric)
            for nested in value.values():
                visit(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                visit(nested)

    visit(report)
    return min(values) if values else float(fallback)


def _shield_rejected_taskspace_result_v13(
    env: RealisticEdgeArmEnvV10,
    action_adapter: StockGripperTaskFrameAdapterV13,
    policy_action: np.ndarray,
    *,
    static_minimum_clearance_m: float,
) -> TaskSpaceActionResultV5:
    current_q = np.asarray(env.data.qpos[:JOINT_ACTION_DIM], dtype=np.float64).copy()
    position, normal = action_adapter._pose_for_joint_position(current_q)
    report = action_adapter.last_recovery_report
    one_step = _rejected_guard_minimum_v13(
        report,
        "minimum_one_step_clearance_m",
        fallback=static_minimum_clearance_m,
    )
    braking = _rejected_guard_minimum_v13(
        report,
        "minimum_braking_clearance_m",
        fallback=one_step,
    )
    return TaskSpaceActionResultV5(
        requested_task_action=np.asarray(policy_action, dtype=np.float32).copy(),
        applied_task_action=np.zeros(POLICY_ACTION_DIM, dtype=np.float32),
        submitted_joint_action=np.zeros(JOINT_ACTION_DIM, dtype=np.float32),
        target_position_world_m=position.astype(np.float32),
        target_face_normal_world=normal.astype(np.float32),
        application_scale=0.0,
        ik_converged=False,
        face_label="shield_rejected_before_step_v13",
        failure_reason=SHIELD_TERMINAL_REASON_V13,
        guard_safe_candidate=False,
        guard_selected_scale=0.0,
        guard_minimum_one_step_clearance_m=one_step,
        guard_minimum_braking_clearance_m=braking,
        guard_float32_execution_identity=False,
    )


def _shield_rejection_transition_v13(
    env: RealisticEdgeArmEnvV10,
    reward: ScratchPotentialRewardV6Candidate,
    *,
    potential_before: float,
    gamma: float,
    episode_safety_violation_before: bool,
    execution_kernel: RolloutExecutionKernelProtocolV1 | None = None,
) -> tuple[Any, float]:
    if execution_kernel is None:
        safety, static_minimum = _current_static_safety_evidence_v13(env, reward)
    else:
        safety, static_minimum = execution_kernel.static_safety(reward, env)
    transition = reward.shape_transition(
        env_reward=0.0,
        potential_before=potential_before,
        potential_after=potential_before,
        gamma=gamma,
        env_terminated=True,
        env_truncated=False,
        v10_surface_success=False,
        env_terminal_failure=True,
        safety=safety,
        episode_safety_violation_before=episode_safety_violation_before,
    )
    transition = replace(transition, terminal_reason=SHIELD_TERMINAL_REASON_V13)
    if not (
        transition.credit_terminated
        and transition.safety_credit_terminal
        and not transition.v6_integrity_success
        and not transition.credit_truncated
    ):
        raise RuntimeError("V13 shield rejection did not become a terminal failure")
    return transition, static_minimum


def _right_aligned_history_v1(
    records: deque[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    *,
    history_steps: int,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rgb = np.zeros((history_steps, len(VIEW_NAMES), height, width, 3), dtype=np.uint8)
    joints = np.zeros((history_steps, JOINT_STATE_DIM), dtype=np.float32)
    actions = np.zeros((history_steps, ACTION_DIM), dtype=np.float32)
    history_valid = np.zeros(history_steps, dtype=bool)
    view_valid = np.zeros((history_steps, len(VIEW_NAMES)), dtype=bool)
    indices = np.full(history_steps, -1, dtype=np.int64)
    selected = list(records)[-history_steps:]
    offset = history_steps - len(selected)
    for index, (row, frame, joint, action, views) in enumerate(selected, start=offset):
        rgb[index] = frame
        joints[index] = joint
        actions[index] = action
        history_valid[index] = True
        view_valid[index] = views
        indices[index] = row
    return rgb, joints, actions, history_valid, view_valid, indices


class StockTaskFrameRolloutKernelV13:
    """Immutable historical V13 execution contract for the shared collector."""

    format = "edgearm-stock-taskframe-rollout-kernel-v13"
    rollout_format = ROLLOUT_FORMAT
    evaluation_format = EVALUATION_FORMAT
    safety_only_geom_count = 94
    contact_candidate_geom_count = 2
    contact_identity_format = "edgearm-tool-contact-identity-v2"
    safety_guard_format = STOCK_GRIPPER_ACTION_GUARD_FORMAT_V3

    def validate(
        self,
        env: RealisticEdgeArmEnvV10,
        action_adapter: StockGripperTaskFrameAdapterV13,
        reward: ScratchPotentialRewardV6Candidate,
    ) -> None:
        if type(action_adapter) is not StockGripperTaskFrameAdapterV13:
            raise TypeError("V13 rollout kernel requires exact stock task-frame adapter V13")
        if action_adapter.env is not env:
            raise ValueError("V13 rollout adapter belongs to another environment")
        if type(reward) is not ScratchPotentialRewardV6Candidate:
            raise TypeError("V13 rollout kernel requires exact V6 candidate reward")

    def reset_episode(
        self,
        env: RealisticEdgeArmEnvV10,
        renderer: MultiViewRendererProtocolV1,
        action_adapter: StockGripperTaskFrameAdapterV13,
        *,
        requested_seed: int,
        obstacle: bool,
        stress: bool,
    ) -> dict[str, Any]:
        return reset_stock_taskframe_episode_v13(
            env,
            renderer,
            action_adapter,
            requested_seed=requested_seed,
            obstacle=obstacle,
            stress=stress,
        )

    def transition_contact(
        self,
        info: dict[str, Any],
        *,
        block_before_xy_m: np.ndarray,
        block_after_xy_m: np.ndarray,
    ) -> dict[str, Any]:
        return transition_contact_telemetry_v10(
            info,
            block_before_xy_m=block_before_xy_m,
            block_after_xy_m=block_after_xy_m,
        )

    def transition_safety(
        self,
        reward: ScratchPotentialRewardV6Candidate,
        env: RealisticEdgeArmEnvV10,
        info: dict[str, Any],
    ) -> ScratchSafetyEvidenceV6Candidate:
        return reward.evaluate_transition_safety(env, info)

    def static_safety(
        self,
        reward: ScratchPotentialRewardV6Candidate,
        env: RealisticEdgeArmEnvV10,
    ) -> tuple[ScratchSafetyEvidenceV6Candidate, float]:
        return _current_static_safety_evidence_v13(env, reward)

    def minimum_safety_only_clearance(
        self,
        telemetry: dict[str, Any],
    ) -> float:
        return float(telemetry["minimum_executed_94_safety_only_block_clearance_m"])


DEFAULT_ROLLOUT_EXECUTION_KERNEL_V13 = StockTaskFrameRolloutKernelV13()


def collect_asymmetric_multiview_rollout_v1(
    env: RealisticEdgeArmEnvV10,
    renderer: MultiViewRendererProtocolV1,
    action_adapter: StockGripperTaskFrameAdapterV13,
    actor: SelectedViewRecurrentActorV1,
    critic: AsymmetricPrivilegedCriticV1,
    config: AsymmetricMultiViewPPOConfigV1,
    *,
    seed: int,
    potential_reward: ScratchPotentialRewardV6Candidate | None = None,
    autoregressive_applied_action_feedback: bool = False,
    autoregressive_zero_on_ik_failure: bool = False,
    execution_kernel: RolloutExecutionKernelProtocolV1 | None = None,
    episode_condition_schedule: tuple[tuple[bool, bool], ...] | None = None,
) -> AsymmetricMultiViewRolloutBatchV1:
    """Collect complete visual PPO episodes without expert-policy access."""

    config.validate()
    if type(env) is not RealisticEdgeArmEnvV10:
        raise TypeError("visual scratch rollout requires exact RealisticEdgeArmEnvV10")
    if tuple(renderer.view_names) != VIEW_NAMES:
        raise ValueError("visual scratch renderer view order changed")
    if type(seed) is not int or seed < 0:
        raise ValueError("visual scratch rollout seed must be non-negative")
    if type(autoregressive_applied_action_feedback) is not bool:
        raise TypeError("visual scratch AR feedback selector must be boolean")
    if type(autoregressive_zero_on_ik_failure) is not bool:
        raise TypeError("visual scratch IK-reset selector must be boolean")
    if autoregressive_applied_action_feedback and autoregressive_zero_on_ik_failure:
        raise ValueError("visual scratch AR feedback modes are mutually exclusive")
    if episode_condition_schedule is not None:
        if not isinstance(episode_condition_schedule, tuple) or not episode_condition_schedule:
            raise ValueError("visual scratch condition schedule must be a nonempty tuple")
        if any(
            not isinstance(item, tuple)
            or len(item) != 2
            or type(item[0]) is not bool
            or type(item[1]) is not bool
            for item in episode_condition_schedule
        ):
            raise TypeError("visual scratch condition schedule entries must be boolean pairs")
    reward = potential_reward or ScratchPotentialRewardV6Candidate()
    kernel = execution_kernel or DEFAULT_ROLLOUT_EXECUTION_KERNEL_V13
    kernel.validate(env, action_adapter, reward)
    device = _module_device(actor)
    if _module_device(critic) != device:
        raise ValueError("visual actor and critic must share one device")
    policy_generator = _torch_generator(device, seed ^ 0x4A77)
    episode_rng = np.random.default_rng(seed ^ 0x7A41C9)
    view_rng = np.random.default_rng(seed ^ 0x51E7)
    rows: dict[str, list[Any]] = {
        name: []
        for name in (
            "rgb_frames",
            "policy_rgb_history",
            "joint_state",
            "policy_joint_history",
            "previous_executed_action",
            "policy_action_history",
            "history_valid",
            "view_valid",
            "policy_view_history_valid",
            "history_row_indices",
            "camera_pose",
            "privileged_state",
            "next_privileged_state",
            "visual_geometry_target",
            "policy_action",
            "previous_policy_pre_tanh",
            "applied_task_action",
            "executed_action",
            "submitted_joint_action",
            "execution_attempted",
            "shield_rejected_before_step",
            "ik_target_position_world_m",
            "ik_target_face_normal_world",
            "ik_application_scale",
            "ik_converged",
            "ik_face_label",
            "ik_failure_reason",
            "guard_safe_candidate",
            "guard_selected_scale",
            "guard_minimum_one_step_clearance_m",
            "guard_minimum_braking_clearance_m",
            "guard_float32_execution_identity",
            "tool_block_contact_any",
            "tool_block_contact_substep_count",
            "valid_push_side_contact_any",
            "valid_push_side_contact_substep_count",
            "valid_push_side_contact_transient_only",
            "invalid_tool_block_contact_any",
            "valid_push_side_peak_normal_force_n",
            "valid_push_side_normal_impulse_discrete_ns",
            "minimum_executed_94_safety_only_block_clearance_m",
            "step_block_displacement_m",
            "pre_tanh",
            "old_log_probs",
            "raw_environment_reward",
            "reward_before_potential",
            "potential_before",
            "potential_after",
            "potential_next_for_shaping",
            "shaped_rewards",
            "values",
            "next_values",
            "terminated",
            "truncated",
            "strict_success",
            "terminal_failure",
            "safety_stop",
            "terminal_reason",
            "episode_ids",
            "episode_step_ids",
            "task_ids",
            "requested_joint_target",
            "queued_safe_joint_target",
            "applied_joint_target",
            "sim_qpos",
            "sim_qvel",
        )
    }
    episode_records: list[dict[str, Any]] = []
    history_records: deque[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = deque(
        maxlen=config.history_steps
    )
    episode_id = 0
    episode_step = 0
    previous_executed_action = np.zeros(ACTION_DIM, dtype=np.float32)
    previous_policy_pre_tanh = np.zeros(POLICY_ACTION_DIM, dtype=np.float32)
    episode_safety_violation = False
    actor.eval()
    inference_cache = CausalVisualInferenceCacheV24(
        actor,
        history_steps=config.history_steps,
    )

    def reset_episode(current_episode: int) -> None:
        nonlocal episode_step
        nonlocal previous_executed_action
        nonlocal previous_policy_pre_tanh
        nonlocal episode_safety_violation
        if episode_condition_schedule is None:
            obstacle = bool(episode_rng.random() < config.obstacle_probability)
            stress = bool(episode_rng.random() < config.stress_probability)
        else:
            obstacle, stress = episode_condition_schedule[
                current_episode % len(episode_condition_schedule)
            ]
        requested_reset_seed = seed + current_episode
        reset_audit = kernel.reset_episode(
            env,
            renderer,
            action_adapter,
            requested_seed=requested_reset_seed,
            obstacle=obstacle,
            stress=stress,
        )
        selected_reset_seed = int(reset_audit["selected_seed"])
        history_records.clear()
        inference_cache.reset()
        episode_step = 0
        previous_executed_action = np.zeros(ACTION_DIM, dtype=np.float32)
        previous_policy_pre_tanh = np.zeros(POLICY_ACTION_DIM, dtype=np.float32)
        episode_safety_violation = False
        domain = _json_safe(env.episode_domain)
        episode_records.append(
            {
                "episode_id": current_episode,
                "reset_seed": selected_reset_seed,
                "requested_reset_seed": requested_reset_seed,
                "selected_reset_seed": selected_reset_seed,
                "reset_attempt_index": int(reset_audit["selected_attempt_index"]),
                "rejected_reset_attempts": _json_safe(reset_audit["rejected_attempts"]),
                "obstacle_enabled": bool(env.obstacle_enabled),
                "stress_enabled": bool(env.current_stress),
                "episode_domain": domain,
                "episode_domain_sha256": canonical_sha256_v1(domain),
            }
        )

    reset_episode(episode_id)
    critic.eval()
    while len(rows["policy_action"]) < config.rollout_steps or not (
        rows["terminated"][-1] or rows["truncated"][-1]
    ):
        row_index = len(rows["policy_action"])
        observation = env.observation()
        joint_state = np.asarray(observation["joint_state"], dtype=np.float32).copy()
        if joint_state.shape != (JOINT_STATE_DIM,) or not np.all(np.isfinite(joint_state)):
            raise RuntimeError("visual scratch reported joint state is invalid")
        rgb_frames, camera_pose = renderer.capture()
        rgb_frames = np.asarray(rgb_frames, dtype=np.uint8)
        camera_pose = np.asarray(camera_pose, dtype=np.float32)
        expected_rgb = (
            len(VIEW_NAMES),
            config.image_height,
            config.image_width,
            3,
        )
        if rgb_frames.shape != expected_rgb or camera_pose.shape != (len(VIEW_NAMES), 12):
            raise RuntimeError("visual scratch renderer packet shape changed")
        view_valid = _sample_view_valid_v1(
            view_rng,
            config.auxiliary_view_dropout_probability,
        )
        history_records.append(
            (
                row_index,
                rgb_frames.copy(),
                joint_state.copy(),
                previous_executed_action.copy(),
                view_valid.copy(),
            )
        )
        (
            rgb_history,
            joint_history,
            action_history,
            history_valid,
            view_history_valid,
            history_indices,
        ) = _right_aligned_history_v1(
            history_records,
            history_steps=config.history_steps,
            height=config.image_height,
            width=config.image_width,
        )

        privileged = build_privileged_effect_state_v1(env).astype(np.float32, copy=False)
        visual_geometry_target = visual_geometry_target_from_privileged_v1(privileged)
        potential_before = reward.evaluate(env).potential
        with torch.no_grad():
            base_distribution = inference_cache.distribution(
                rgb_frames,
                joint_state,
                previous_executed_action,
                view_valid,
            )
            distribution = autoregressive_action_distribution_v1(
                base_distribution,
                torch.from_numpy(previous_policy_pre_tanh).to(device).unsqueeze(0),
                config.action_autoregressive_rho,
            )
            sample = sample_squashed_gaussian_v1(distribution, generator=policy_generator)
            value = float(critic(torch.from_numpy(privileged).to(device).unsqueeze(0)).item())
        policy_action = sample.action.squeeze(0).cpu().numpy().astype(np.float32)
        pre_tanh = sample.pre_tanh.squeeze(0).cpu().numpy().astype(np.float32)
        old_log_prob = float(sample.log_prob.item())
        sim_qpos = np.asarray(env.data.qpos, dtype=np.float32).copy()
        sim_qvel = np.asarray(env.data.qvel, dtype=np.float32).copy()

        execution_attempted = True
        shield_rejected_before_step = False
        try:
            taskspace = action_adapter.translate(policy_action)
        except StockTaskFrameNoSafeRecoveryV13:
            execution_attempted = False
            shield_rejected_before_step = True
            transition, static_minimum = _shield_rejection_transition_v13(
                env,
                reward,
                potential_before=potential_before,
                gamma=config.gamma,
                episode_safety_violation_before=episode_safety_violation,
                execution_kernel=kernel,
            )
            taskspace = _shield_rejected_taskspace_result_v13(
                env,
                action_adapter,
                policy_action,
                static_minimum_clearance_m=static_minimum,
            )
            env_reward = 0.0
            potential_after = potential_before
            contact_telemetry = {
                "tool_block_contact_any": False,
                "tool_block_contact_substep_count": 0,
                "valid_push_side_contact_any": False,
                "valid_push_side_contact_substep_count": 0,
                "valid_push_side_contact_transient_only": False,
                "invalid_tool_block_contact_any": False,
                "valid_push_side_peak_normal_force_n": 0.0,
                "valid_push_side_normal_impulse_discrete_ns": 0.0,
                "minimum_executed_94_safety_only_block_clearance_m": (static_minimum),
                "minimum_executed_safety_only_block_clearance_m": (static_minimum),
                "step_block_displacement_m": 0.0,
            }
            next_privileged = privileged.copy()
            next_value = value
            current_joint_target = np.asarray(env.data.qpos[:JOINT_ACTION_DIM], dtype=np.float32).copy()
            executed_action = np.zeros(JOINT_ACTION_DIM, dtype=np.float32)
            requested_joint_target = current_joint_target.copy()
            queued_safe_joint_target = current_joint_target.copy()
            applied_joint_target = current_joint_target.copy()
            safety_stop = True
            episode_records[-1]["v13_action_shield_terminal"] = {
                "decision_row": row_index,
                "episode_step": episode_step,
                "terminal_reason": SHIELD_TERMINAL_REASON_V13,
                "execution_attempted": False,
                "static_minimum_94_safety_only_block_clearance_m": (static_minimum),
                "static_minimum_safety_only_block_clearance_m": (static_minimum),
                "safety_only_geom_count": kernel.safety_only_geom_count,
                "recovery_report": _json_safe(action_adapter.last_recovery_report),
            }
        else:
            block_before_step_xy = env.block_xy().copy()
            _, env_reward, env_terminated, env_truncated, info = env.step(taskspace.submitted_joint_action)
            contact_telemetry = kernel.transition_contact(
                info,
                block_before_xy_m=block_before_step_xy,
                block_after_xy_m=env.block_xy(),
            )
            potential_after = reward.evaluate(env).potential
            safety = kernel.transition_safety(reward, env, info)
            surface_success = bool(info.get("success", False))
            env_failure = bool(
                info.get(
                    "terminal_failure",
                    env_terminated and not surface_success,
                )
            )
            transition = reward.shape_transition(
                env_reward=env_reward,
                potential_before=potential_before,
                potential_after=potential_after,
                gamma=config.gamma,
                env_terminated=bool(env_terminated),
                env_truncated=bool(env_truncated),
                v10_surface_success=surface_success,
                env_terminal_failure=env_failure,
                safety=safety,
                episode_safety_violation_before=episode_safety_violation,
            )
            next_privileged = build_privileged_effect_state_v1(env).astype(np.float32, copy=False)
            with torch.no_grad():
                next_value = float(critic(torch.from_numpy(next_privileged).to(device).unsqueeze(0)).item())
            transport = info.get("sim2real_v2")
            if not isinstance(transport, dict):
                raise RuntimeError("visual scratch step lost sim2real command lineage")
            executed_action = np.asarray(
                transport["actually_applied_delayed_action"],
                dtype=np.float32,
            ).copy()
            requested_joint_target = np.asarray(
                transport["requested_joint_target"],
                dtype=np.float32,
            ).copy()
            queued_safe_joint_target = np.asarray(
                transport["queued_safe_joint_target"],
                dtype=np.float32,
            ).copy()
            applied_joint_target = np.asarray(
                transport["delayed_joint_target"],
                dtype=np.float32,
            ).copy()
            safety_stop = bool(safety.hard_safety_violation or info.get("safety_stop"))
        episode_safety_violation = transition.episode_safety_violation

        appended = {
            "rgb_frames": rgb_frames,
            "policy_rgb_history": rgb_history,
            "joint_state": joint_state,
            "policy_joint_history": joint_history,
            "previous_executed_action": previous_executed_action.copy(),
            "policy_action_history": action_history,
            "history_valid": history_valid,
            "view_valid": view_valid,
            "policy_view_history_valid": view_history_valid,
            "history_row_indices": history_indices,
            "camera_pose": camera_pose,
            "privileged_state": privileged,
            "next_privileged_state": next_privileged,
            "visual_geometry_target": visual_geometry_target,
            "policy_action": policy_action,
            "previous_policy_pre_tanh": previous_policy_pre_tanh.copy(),
            "applied_task_action": taskspace.applied_task_action,
            "executed_action": executed_action,
            "submitted_joint_action": taskspace.submitted_joint_action,
            "execution_attempted": execution_attempted,
            "shield_rejected_before_step": shield_rejected_before_step,
            "ik_target_position_world_m": taskspace.target_position_world_m,
            "ik_target_face_normal_world": taskspace.target_face_normal_world,
            "ik_application_scale": taskspace.application_scale,
            "ik_converged": taskspace.ik_converged,
            "ik_face_label": taskspace.face_label,
            "ik_failure_reason": taskspace.failure_reason,
            "guard_safe_candidate": taskspace.guard_safe_candidate,
            "guard_selected_scale": taskspace.guard_selected_scale,
            "guard_minimum_one_step_clearance_m": (taskspace.guard_minimum_one_step_clearance_m),
            "guard_minimum_braking_clearance_m": (taskspace.guard_minimum_braking_clearance_m),
            "guard_float32_execution_identity": (taskspace.guard_float32_execution_identity),
            "tool_block_contact_any": contact_telemetry["tool_block_contact_any"],
            "tool_block_contact_substep_count": contact_telemetry["tool_block_contact_substep_count"],
            "valid_push_side_contact_any": contact_telemetry["valid_push_side_contact_any"],
            "valid_push_side_contact_substep_count": contact_telemetry[
                "valid_push_side_contact_substep_count"
            ],
            "valid_push_side_contact_transient_only": contact_telemetry[
                "valid_push_side_contact_transient_only"
            ],
            "invalid_tool_block_contact_any": contact_telemetry["invalid_tool_block_contact_any"],
            "valid_push_side_peak_normal_force_n": contact_telemetry["valid_push_side_peak_normal_force_n"],
            "valid_push_side_normal_impulse_discrete_ns": contact_telemetry[
                "valid_push_side_normal_impulse_discrete_ns"
            ],
            "minimum_executed_94_safety_only_block_clearance_m": (
                kernel.minimum_safety_only_clearance(contact_telemetry)
            ),
            "step_block_displacement_m": contact_telemetry["step_block_displacement_m"],
            "pre_tanh": pre_tanh,
            "old_log_probs": old_log_prob,
            "raw_environment_reward": transition.v10_env_reward_raw,
            "reward_before_potential": transition.reward_before_potential,
            "potential_before": transition.potential_before,
            "potential_after": transition.potential_after,
            "potential_next_for_shaping": transition.potential_next_for_shaping,
            "shaped_rewards": transition.shaped_reward,
            "values": value,
            "next_values": next_value,
            "terminated": transition.credit_terminated,
            "truncated": transition.credit_truncated,
            "strict_success": transition.v6_integrity_success,
            "terminal_failure": transition.safety_credit_terminal,
            "safety_stop": safety_stop,
            "terminal_reason": transition.terminal_reason,
            "episode_ids": episode_id,
            "episode_step_ids": episode_step,
            "task_ids": 0,
            "requested_joint_target": requested_joint_target,
            "queued_safe_joint_target": queued_safe_joint_target,
            "applied_joint_target": applied_joint_target,
            "sim_qpos": sim_qpos,
            "sim_qvel": sim_qvel,
        }
        for name, value_to_append in appended.items():
            rows[name].append(value_to_append)
        previous_executed_action = executed_action
        if autoregressive_applied_action_feedback:
            previous_policy_pre_tanh = autoregressive_applied_feedback_state_v21(
                taskspace.applied_task_action
            )
        elif autoregressive_zero_on_ik_failure:
            previous_policy_pre_tanh = autoregressive_ik_guarded_feedback_state_v21(
                pre_tanh,
                ik_converged=bool(taskspace.ik_converged),
            )
        else:
            previous_policy_pre_tanh = pre_tanh.copy()
        episode_step += 1
        if (transition.credit_terminated or transition.credit_truncated) and len(
            rows["policy_action"]
        ) < config.rollout_steps:
            episode_id += 1
            reset_episode(episode_id)

    float_names = {
        "joint_state",
        "policy_joint_history",
        "previous_executed_action",
        "policy_action_history",
        "camera_pose",
        "privileged_state",
        "next_privileged_state",
        "visual_geometry_target",
        "policy_action",
        "previous_policy_pre_tanh",
        "applied_task_action",
        "executed_action",
        "submitted_joint_action",
        "ik_target_position_world_m",
        "ik_target_face_normal_world",
        "ik_application_scale",
        "guard_selected_scale",
        "guard_minimum_one_step_clearance_m",
        "guard_minimum_braking_clearance_m",
        "valid_push_side_peak_normal_force_n",
        "valid_push_side_normal_impulse_discrete_ns",
        "minimum_executed_94_safety_only_block_clearance_m",
        "step_block_displacement_m",
        "pre_tanh",
        "old_log_probs",
        "raw_environment_reward",
        "reward_before_potential",
        "potential_before",
        "potential_after",
        "potential_next_for_shaping",
        "shaped_rewards",
        "values",
        "next_values",
        "requested_joint_target",
        "queued_safe_joint_target",
        "applied_joint_target",
        "sim_qpos",
        "sim_qvel",
    }
    bool_names = {
        "history_valid",
        "view_valid",
        "policy_view_history_valid",
        "terminated",
        "truncated",
        "strict_success",
        "terminal_failure",
        "safety_stop",
        "ik_converged",
        "guard_safe_candidate",
        "guard_float32_execution_identity",
        "execution_attempted",
        "shield_rejected_before_step",
        "tool_block_contact_any",
        "valid_push_side_contact_any",
        "valid_push_side_contact_transient_only",
        "invalid_tool_block_contact_any",
    }
    int_names = {
        "history_row_indices",
        "episode_ids",
        "episode_step_ids",
        "task_ids",
        "tool_block_contact_substep_count",
        "valid_push_side_contact_substep_count",
    }
    converted: dict[str, np.ndarray] = {}
    for name, values in rows.items():
        if name in {"rgb_frames", "policy_rgb_history"}:
            converted[name] = np.asarray(values, dtype=np.uint8)
        elif name in float_names:
            converted[name] = np.asarray(values, dtype=np.float32)
        elif name in bool_names:
            converted[name] = np.asarray(values, dtype=bool)
        elif name in int_names:
            converted[name] = np.asarray(values, dtype=np.int64)
        elif name in {"terminal_reason", "ik_face_label", "ik_failure_reason"}:
            converted[name] = np.asarray(values, dtype=str)
        else:  # pragma: no cover - all row names are classified above
            raise RuntimeError(f"unclassified multiview rollout field: {name}")
    batch = AsymmetricMultiViewRolloutBatchV1(
        **converted,
        episode_records=tuple(episode_records),
        shaping_gamma=float(np.float32(config.gamma)),
        potential_reward_config_sha256=reward.config_sha256,
        rollout_seed=seed,
        execution_kernel_format=kernel.format,
        safety_only_geom_count=kernel.safety_only_geom_count,
        contact_candidate_geom_count=kernel.contact_candidate_geom_count,
        contact_identity_format=kernel.contact_identity_format,
        safety_guard_format=kernel.safety_guard_format,
        rollout_format=kernel.rollout_format,
    )
    batch.validate()
    return batch


def evaluate_asymmetric_multiview_policy_v1(
    env: RealisticEdgeArmEnvV10,
    renderer: MultiViewRendererProtocolV1,
    action_adapter: StockGripperTaskFrameAdapterV13,
    actor: SelectedViewRecurrentActorV1,
    config: AsymmetricMultiViewPPOConfigV1,
    *,
    seed_base: int,
    episodes: int = 4,
    potential_reward: ScratchPotentialRewardV6Candidate | None = None,
    autoregressive_applied_action_feedback: bool = False,
    autoregressive_zero_on_ik_failure: bool = False,
    execution_kernel: RolloutExecutionKernelProtocolV1 | None = None,
    episode_condition_schedule: tuple[tuple[bool, bool], ...] | None = None,
) -> dict[str, Any]:
    """Run deterministic, held-out visual episodes without updating the policy."""

    config.validate()
    if type(env) is not RealisticEdgeArmEnvV10:
        raise TypeError("visual held-out evaluation requires exact RealisticEdgeArmEnvV10")
    if tuple(renderer.view_names) != VIEW_NAMES:
        raise ValueError("visual held-out renderer view order changed")
    if type(seed_base) is not int or seed_base < 0:
        raise ValueError("held-out seed base must be non-negative")
    if type(episodes) is not int or episodes < 1:
        raise ValueError("held-out episode count must be positive")
    if type(autoregressive_applied_action_feedback) is not bool:
        raise TypeError("held-out AR feedback selector must be boolean")
    if type(autoregressive_zero_on_ik_failure) is not bool:
        raise TypeError("held-out IK-reset selector must be boolean")
    if autoregressive_applied_action_feedback and autoregressive_zero_on_ik_failure:
        raise ValueError("held-out AR feedback modes are mutually exclusive")
    if episode_condition_schedule is not None:
        if (
            not isinstance(episode_condition_schedule, tuple)
            or len(episode_condition_schedule) != episodes
            or any(
                not isinstance(item, tuple)
                or len(item) != 2
                or type(item[0]) is not bool
                or type(item[1]) is not bool
                for item in episode_condition_schedule
            )
        ):
            raise ValueError(
                "held-out condition schedule must contain one boolean pair per episode"
            )
    reward = potential_reward or ScratchPotentialRewardV6Candidate()
    kernel = execution_kernel or DEFAULT_ROLLOUT_EXECUTION_KERNEL_V13
    kernel.validate(env, action_adapter, reward)
    device = _module_device(actor)
    conditions = (
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    )
    episode_results: list[dict[str, Any]] = []
    actor.eval()
    inference_cache = CausalVisualInferenceCacheV24(
        actor,
        history_steps=config.history_steps,
    )
    for episode_index in range(episodes):
        obstacle, stress = (
            episode_condition_schedule[episode_index]
            if episode_condition_schedule is not None
            else conditions[episode_index % len(conditions)]
        )
        requested_episode_seed = seed_base + episode_index
        reset_audit = kernel.reset_episode(
            env,
            renderer,
            action_adapter,
            requested_seed=requested_episode_seed,
            obstacle=obstacle,
            stress=stress,
        )
        episode_seed = int(reset_audit["selected_seed"])
        inference_cache.reset()
        previous_executed_action = np.zeros(ACTION_DIM, dtype=np.float32)
        previous_policy_pre_tanh = np.zeros(POLICY_ACTION_DIM, dtype=np.float32)
        episode_safety_violation = False
        initial_evaluation = reward.evaluate(env)
        initial_potential = initial_evaluation.potential
        maximum_potential = initial_potential
        total_raw_reward = 0.0
        total_shaped_reward = 0.0
        positive_potential_steps = 0
        strict_success = False
        terminal_reason = "missing_episode_boundary"
        completed_steps = 0
        contact_steps = 0
        invalid_contact_steps = 0
        contact_substeps = 0
        transient_contact_steps = 0
        valid_contact_impulse_ns = 0.0
        ik_failure_steps = 0
        action_shield_rejection_steps = 0
        applied_scale_sum = 0.0
        policy_action_sum = np.zeros(POLICY_ACTION_DIM, dtype=np.float64)
        applied_task_action_sum = np.zeros(POLICY_ACTION_DIM, dtype=np.float64)
        positive_policy_forward_steps = 0
        positive_applied_forward_steps = 0
        for episode_step in range(env.config.max_steps):
            observation = env.observation()
            joint_state = np.asarray(observation["joint_state"], dtype=np.float32).copy()
            rgb_frames, _camera_pose = renderer.capture()
            rgb_frames = np.asarray(rgb_frames, dtype=np.uint8)
            view_valid = np.ones(len(VIEW_NAMES), dtype=bool)
            with torch.no_grad():
                distribution = autoregressive_action_distribution_v1(
                    inference_cache.distribution(
                        rgb_frames,
                        joint_state,
                        previous_executed_action,
                        view_valid,
                    ),
                    torch.from_numpy(previous_policy_pre_tanh).to(device).unsqueeze(0),
                    config.action_autoregressive_rho,
                )
                deterministic_pre_tanh = distribution.loc.squeeze(0)
                policy_action = torch.tanh(deterministic_pre_tanh).cpu().numpy()
                proposed_policy_pre_tanh = deterministic_pre_tanh.cpu().numpy().astype(np.float32)
            potential_before = reward.evaluate(env).potential
            try:
                taskspace = action_adapter.translate(policy_action)
            except StockTaskFrameNoSafeRecoveryV13:
                transition, static_minimum = _shield_rejection_transition_v13(
                    env,
                    reward,
                    potential_before=potential_before,
                    gamma=config.gamma,
                    episode_safety_violation_before=episode_safety_violation,
                    execution_kernel=kernel,
                )
                taskspace = _shield_rejected_taskspace_result_v13(
                    env,
                    action_adapter,
                    policy_action,
                    static_minimum_clearance_m=static_minimum,
                )
                env_reward = 0.0
                potential_after = potential_before
                contact_telemetry = {
                    "valid_push_side_contact_any": False,
                    "valid_push_side_contact_substep_count": 0,
                    "valid_push_side_contact_transient_only": False,
                    "valid_push_side_normal_impulse_discrete_ns": 0.0,
                    "invalid_tool_block_contact_any": False,
                }
                previous_executed_action = np.zeros(ACTION_DIM, dtype=np.float32)
                action_shield_rejection_steps += 1
            else:
                block_before_step_xy = env.block_xy().copy()
                _, env_reward, env_terminated, env_truncated, info = env.step(
                    taskspace.submitted_joint_action
                )
                contact_telemetry = kernel.transition_contact(
                    info,
                    block_before_xy_m=block_before_step_xy,
                    block_after_xy_m=env.block_xy(),
                )
                potential_after = reward.evaluate(env).potential
                safety = kernel.transition_safety(reward, env, info)
                surface_success = bool(info.get("success", False))
                env_failure = bool(
                    info.get(
                        "terminal_failure",
                        env_terminated and not surface_success,
                    )
                )
                transition = reward.shape_transition(
                    env_reward=env_reward,
                    potential_before=potential_before,
                    potential_after=potential_after,
                    gamma=config.gamma,
                    env_terminated=bool(env_terminated),
                    env_truncated=bool(env_truncated),
                    v10_surface_success=surface_success,
                    env_terminal_failure=env_failure,
                    safety=safety,
                    episode_safety_violation_before=episode_safety_violation,
                )
                transport = info.get("sim2real_v2")
                if not isinstance(transport, dict):
                    raise RuntimeError("held-out evaluation lost sim2real command lineage")
                previous_executed_action = np.asarray(
                    transport["actually_applied_delayed_action"],
                    dtype=np.float32,
                ).copy()
            if autoregressive_applied_action_feedback:
                previous_policy_pre_tanh = autoregressive_applied_feedback_state_v21(
                    taskspace.applied_task_action
                )
            elif autoregressive_zero_on_ik_failure:
                previous_policy_pre_tanh = autoregressive_ik_guarded_feedback_state_v21(
                    proposed_policy_pre_tanh,
                    ik_converged=bool(taskspace.ik_converged),
                )
            else:
                previous_policy_pre_tanh = proposed_policy_pre_tanh
            episode_safety_violation = transition.episode_safety_violation
            policy_action_sum += np.asarray(policy_action, dtype=np.float64)
            applied_task_action_sum += np.asarray(taskspace.applied_task_action, dtype=np.float64)
            positive_policy_forward_steps += int(policy_action[0] > 0.0)
            positive_applied_forward_steps += int(taskspace.applied_task_action[0] > 0.0)
            maximum_potential = max(maximum_potential, potential_after)
            positive_potential_steps += int(potential_after > potential_before)
            contact_steps += int(contact_telemetry["valid_push_side_contact_any"])
            invalid_contact_steps += int(
                contact_telemetry["invalid_tool_block_contact_any"]
            )
            contact_substeps += int(contact_telemetry["valid_push_side_contact_substep_count"])
            transient_contact_steps += int(contact_telemetry["valid_push_side_contact_transient_only"])
            valid_contact_impulse_ns += float(contact_telemetry["valid_push_side_normal_impulse_discrete_ns"])
            ik_failure_steps += int(not taskspace.ik_converged)
            applied_scale_sum += taskspace.application_scale
            total_raw_reward += float(env_reward)
            total_shaped_reward += float(transition.shaped_reward)
            completed_steps = episode_step + 1
            strict_success = bool(transition.v6_integrity_success)
            terminal_reason = transition.terminal_reason
            if transition.credit_terminated or transition.credit_truncated:
                break
        else:  # pragma: no cover - V10 must emit the configured time limit
            raise RuntimeError("held-out evaluation did not reach an episode boundary")
        final_evaluation = reward.evaluate(env)
        episode_results.append(
            {
                "episode_index": episode_index,
                "seed": episode_seed,
                "requested_seed": requested_episode_seed,
                "selected_seed": episode_seed,
                "reset_attempt_index": int(reset_audit["selected_attempt_index"]),
                "rejected_reset_attempts": _json_safe(reset_audit["rejected_attempts"]),
                "obstacle": obstacle,
                "stress": stress,
                "steps": completed_steps,
                "strict_success": strict_success,
                "terminal_reason": terminal_reason,
                "episode_safety_violation": episode_safety_violation,
                "initial_potential": float(initial_potential),
                "final_potential": float(final_evaluation.potential),
                "maximum_potential": float(maximum_potential),
                "initial_coverage_progress": float(initial_evaluation.coverage),
                "final_coverage_progress": float(final_evaluation.coverage),
                "initial_block_progress": float(initial_evaluation.normalized_block_progress),
                "final_block_progress": float(final_evaluation.normalized_block_progress),
                "initial_contact_approach_progress": float(initial_evaluation.contact_approach_progress),
                "final_contact_approach_progress": float(final_evaluation.contact_approach_progress),
                "positive_potential_step_fraction": positive_potential_steps / max(completed_steps, 1),
                "valid_push_side_contact_transition_count": contact_steps,
                "invalid_tool_block_contact_transition_count": invalid_contact_steps,
                "valid_push_side_contact_substep_count": contact_substeps,
                "transient_valid_push_side_contact_transition_count": (transient_contact_steps),
                "valid_push_side_normal_impulse_discrete_ns": (valid_contact_impulse_ns),
                "ik_failure_step_count": ik_failure_steps,
                "action_shield_rejection_step_count": (action_shield_rejection_steps),
                "mean_ik_application_scale": applied_scale_sum / max(completed_steps, 1),
                "mean_policy_task_action": (policy_action_sum / max(completed_steps, 1)).tolist(),
                "mean_applied_task_action": (applied_task_action_sum / max(completed_steps, 1)).tolist(),
                "positive_policy_forward_step_fraction": (
                    positive_policy_forward_steps / max(completed_steps, 1)
                ),
                "positive_applied_forward_step_fraction": (
                    positive_applied_forward_steps / max(completed_steps, 1)
                ),
                "final_block_target_distance_m": float(final_evaluation.block_target_distance_m),
                "final_target_coverage": float(final_evaluation.coverage),
                "total_raw_environment_reward": total_raw_reward,
                "total_shaped_reward": total_shaped_reward,
            }
        )
    success_count = sum(int(item["strict_success"]) for item in episode_results)
    return {
        "format": kernel.evaluation_format,
        "source_type": SOURCE_TYPE,
        "policy_format": POLICY_FORMAT,
        "execution_kernel_format": kernel.format,
        "contact_identity_format": kernel.contact_identity_format,
        "contact_candidate_geom_count": kernel.contact_candidate_geom_count,
        "safety_only_geom_count": kernel.safety_only_geom_count,
        "safety_guard_format": kernel.safety_guard_format,
        "deterministic_policy": True,
        "actor_inference_cache_format": CAUSAL_VISUAL_INFERENCE_CACHE_FORMAT_V24,
        "overlapping_visual_history_reencoded": False,
        "action_autoregressive_rho": config.action_autoregressive_rho,
        "autoregressive_policy_state": "previous_policy_pre_tanh",
        "autoregressive_feedback_mode": (
            "applied_task_action_inverse_tanh"
            if autoregressive_applied_action_feedback
            else (
                "requested_pre_tanh_zeroed_on_ik_failure"
                if autoregressive_zero_on_ik_failure
                else "requested_policy_pre_tanh"
            )
        ),
        "expert_calls": 0,
        "behavior_cloning_steps": 0,
        "physical_samples": 0,
        "production_admission": False,
        "seed_base": seed_base,
        "episode_count": episodes,
        "strict_success_count": success_count,
        "strict_success_rate": success_count / episodes,
        "strict_success_hold_steps": int(env.realism_config.strict_success_hold_steps),
        "control_fps": int(env.config.fps),
        "strict_success_realized_hold_seconds": (
            env.realism_config.strict_success_hold_steps / env.config.fps
        ),
        "episode_condition_schedule": [
            [bool(item[0]), bool(item[1])]
            for item in (
                episode_condition_schedule
                if episode_condition_schedule is not None
                else tuple(conditions[index % len(conditions)] for index in range(episodes))
            )
        ],
        "action_shield_rejection_step_count": int(
            sum(int(item["action_shield_rejection_step_count"]) for item in episode_results)
        ),
        "valid_push_side_contact_transition_count": int(
            sum(int(item["valid_push_side_contact_transition_count"]) for item in episode_results)
        ),
        "invalid_tool_block_contact_transition_count": int(
            sum(int(item["invalid_tool_block_contact_transition_count"]) for item in episode_results)
        ),
        "contact_episode_count": int(
            sum(int(item["valid_push_side_contact_transition_count"] > 0) for item in episode_results)
        ),
        "shield_terminal_episode_count": int(
            sum(int(item["action_shield_rejection_step_count"] > 0) for item in episode_results)
        ),
        "safety_violation_episode_count": int(
            sum(int(item["episode_safety_violation"]) for item in episode_results)
        ),
        "mean_final_target_coverage": float(
            np.mean([item["final_target_coverage"] for item in episode_results])
        ),
        "mean_final_block_target_distance_m": float(
            np.mean([item["final_block_target_distance_m"] for item in episode_results])
        ),
        "episodes": episode_results,
    }


def normalize_advantages_by_group_v24(
    advantages: np.ndarray,
    group_ids: np.ndarray,
) -> np.ndarray:
    """Standardize each complete on-policy episode independently."""

    values = np.asarray(advantages)
    groups = np.asarray(group_ids)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.floating):
        raise ValueError("V24 advantages must be a floating vector")
    if groups.shape != values.shape or groups.dtype != np.int64:
        raise ValueError("V24 advantage group ids must be int64 and match advantages")
    if not np.all(np.isfinite(values)):
        raise ValueError("V24 advantages must be finite")
    unique_groups = np.unique(groups)
    if (
        len(unique_groups) < 1
        or unique_groups[0] != 0
        or not np.array_equal(unique_groups, np.arange(len(unique_groups)))
    ):
        raise ValueError("V24 advantage group ids must be contiguous from zero")
    normalized = np.zeros_like(values)
    for group_id in unique_groups:
        selected_group = groups == group_id
        group_advantages = values[selected_group]
        normalized[selected_group] = (
            group_advantages - group_advantages.mean()
        ) / (group_advantages.std() + 1.0e-8)
    if not np.all(np.isfinite(normalized)):
        raise RuntimeError("V24 grouped advantage normalization became non-finite")
    return normalized


def ppo_update_asymmetric_multiview_v1(
    actor: SelectedViewRecurrentActorV1,
    critic: AsymmetricPrivilegedCriticV1,
    batch: AsymmetricMultiViewRolloutBatchV1,
    config: AsymmetricMultiViewPPOConfigV1,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    advantage_group_ids: np.ndarray | None = None,
) -> tuple[PPOUpdateMetricsV1, torch.optim.Optimizer]:
    config.validate()
    batch.validate()
    if np.float32(batch.shaping_gamma) != np.float32(config.gamma):
        raise ValueError("rollout shaping gamma differs from PPO gamma")
    device = _module_device(actor)
    if _module_device(critic) != device:
        raise ValueError("visual actor and critic must share one device")
    if optimizer is None:
        optimizer = torch.optim.Adam(
            [*actor.parameters(), *critic.parameters()],
            lr=config.learning_rate,
        )
    advantages, returns = compute_gae_termination_truncation_v1(
        batch.rewards,
        batch.values,
        batch.next_values,
        batch.terminated,
        batch.truncated,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )
    if advantage_group_ids is None:
        normalized_advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1.0e-8
        )
        advantage_normalization = "global"
        advantage_group_count = 1
    else:
        groups = np.asarray(advantage_group_ids)
        unique_groups = np.unique(groups)
        normalized_advantages = normalize_advantages_by_group_v24(advantages, groups)
        advantage_normalization = "per_complete_episode_v24"
        advantage_group_count = len(unique_groups)
    pre_tanh = torch.from_numpy(batch.pre_tanh).to(device)
    previous_pre_tanh = torch.from_numpy(batch.previous_policy_pre_tanh).to(device)
    old_log_probs = torch.from_numpy(batch.old_log_probs).to(device)
    old_values = torch.from_numpy(batch.values).to(device)
    return_tensor = torch.from_numpy(returns).to(device)
    advantage_tensor = torch.from_numpy(normalized_advantages).to(device)
    privileged = torch.from_numpy(batch.privileged_state).to(device)
    visual_geometry_target = torch.from_numpy(batch.visual_geometry_target).to(device)
    rng = np.random.default_rng(config.seed)
    entropy_generator = _torch_generator(device, config.seed ^ 0x71B3)
    actor_parameters = list(actor.parameters())
    critic_parameters = list(critic.parameters())

    actor.eval()
    with torch.no_grad():
        expected_old_distribution = autoregressive_action_distribution_v1(
            actor.distribution(*_actor_tensors(batch, device)),
            previous_pre_tanh,
            config.action_autoregressive_rho,
        )
        expected_old = squashed_gaussian_log_prob_v1(
            expected_old_distribution,
            pre_tanh,
        )
    if not torch.allclose(expected_old, old_log_probs, rtol=1.0e-5, atol=1.0e-5):
        raise ValueError("rollout log probabilities do not belong to the current visual actor")

    indices = np.arange(len(batch.rewards))
    optimizer_steps = 0
    target_kl_triggered = False
    epochs_completed = 0
    metric_rows: list[tuple[float, float, float, float, float, float, float, float, float, float]] = []
    actor.train()
    critic.train()
    for epoch in range(config.update_epochs):
        rng.shuffle(indices)
        for start in range(0, len(indices), config.batch_size):
            chosen_np = indices[start : start + config.batch_size]
            chosen = torch.from_numpy(chosen_np).to(device=device, dtype=torch.long)
            base_distribution, visual_geometry_prediction = actor.distribution_and_visual_geometry(
                *_actor_tensors(batch, device, chosen_np)
            )
            distribution = autoregressive_action_distribution_v1(
                base_distribution,
                previous_pre_tanh[chosen],
                config.action_autoregressive_rho,
            )
            new_log_prob = squashed_gaussian_log_prob_v1(distribution, pre_tanh[chosen])
            log_ratio = new_log_prob - old_log_probs[chosen]
            ratio = log_ratio.exp()
            unclipped = ratio * advantage_tensor[chosen]
            clipped = ratio.clamp(1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * advantage_tensor[chosen]
            policy_loss = -torch.minimum(unclipped, clipped).mean()

            value = critic(privileged[chosen])
            clipped_value = old_values[chosen] + (value - old_values[chosen]).clamp(
                -config.value_clip_ratio,
                config.value_clip_ratio,
            )
            value_loss = (
                0.5
                * torch.maximum(
                    (value - return_tensor[chosen]).square(),
                    (clipped_value - return_tensor[chosen]).square(),
                ).mean()
            )
            entropy_sample = sample_squashed_gaussian_v1(
                distribution,
                generator=entropy_generator,
            )
            entropy = -entropy_sample.log_prob.mean()
            visual_geometry_loss = nn.functional.smooth_l1_loss(
                visual_geometry_prediction,
                visual_geometry_target[chosen],
            )
            total_loss = (
                policy_loss
                + config.value_coef * value_loss
                - config.entropy_coef * entropy
                + config.visual_geometry_auxiliary_coef * visual_geometry_loss
            )
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            # Actor and privileged critic have disjoint inputs and losses.  Clip
            # them independently so a critic return spike cannot suppress the
            # visual policy/geometry gradients through one shared global norm.
            actor_gradient_norm = nn.utils.clip_grad_norm_(
                actor_parameters,
                config.max_grad_norm,
            )
            critic_gradient_norm = nn.utils.clip_grad_norm_(
                critic_parameters,
                config.max_grad_norm,
            )
            optimizer.step()
            optimizer_steps += 1

            with torch.no_grad():
                post_distribution = autoregressive_action_distribution_v1(
                    actor.distribution(*_actor_tensors(batch, device, chosen_np)),
                    previous_pre_tanh[chosen],
                    config.action_autoregressive_rho,
                )
                post_log_prob = squashed_gaussian_log_prob_v1(
                    post_distribution,
                    pre_tanh[chosen],
                )
                post_log_ratio = post_log_prob - old_log_probs[chosen]
                post_ratio = post_log_ratio.exp()
                approximate_kl = ((post_ratio - 1.0) - post_log_ratio).mean()
                policy_clip_fraction = ((post_ratio - 1.0).abs() > config.clip_ratio).float().mean()
                post_value = critic(privileged[chosen])
                value_clip_fraction = (
                    ((post_value - old_values[chosen]).abs() > config.value_clip_ratio).float().mean()
                )
                post_geometry = actor.predict_visual_geometry(*_actor_tensors(batch, device, chosen_np))
                visual_geometry_rmse = (post_geometry - visual_geometry_target[chosen]).square().mean().sqrt()
            row = (
                float(policy_loss.detach().item()),
                float(value_loss.detach().item()),
                float(entropy.detach().item()),
                float(approximate_kl.item()),
                float(policy_clip_fraction.item()),
                float(value_clip_fraction.item()),
                float(visual_geometry_loss.detach().item()),
                float(visual_geometry_rmse.item()),
                float(actor_gradient_norm.item()),
                float(critic_gradient_norm.item()),
            )
            if not np.all(np.isfinite(row)):
                raise RuntimeError("visual PPO update produced non-finite metrics")
            metric_rows.append(row)
            if row[3] > config.target_kl:
                target_kl_triggered = True
                break
        epochs_completed = epoch + 1
        if target_kl_triggered:
            break
    if (
        optimizer_steps < 1
        or not finite_module_parameters_v1(actor)
        or not finite_module_parameters_v1(critic)
    ):
        raise RuntimeError("visual PPO update did not produce finite parameters")
    metrics = np.asarray(metric_rows, dtype=np.float64)
    means = metrics.mean(axis=0)
    return (
        PPOUpdateMetricsV1(
            optimizer_steps=optimizer_steps,
            epochs_completed=epochs_completed,
            target_kl_triggered=target_kl_triggered,
            policy_loss=float(means[0]),
            value_loss=float(means[1]),
            entropy=float(means[2]),
            approximate_kl=float(means[3]),
            policy_clip_fraction=float(means[4]),
            value_clip_fraction=float(means[5]),
            visual_geometry_auxiliary_loss=float(means[6]),
            visual_geometry_auxiliary_rmse=float(means[7]),
            maximum_actor_preclip_gradient_norm=float(metrics[:, 8].max()),
            maximum_critic_preclip_gradient_norm=float(metrics[:, 9].max()),
            maximum_preclip_gradient_norm=float(metrics[:, 8:10].max()),
            advantage_normalization=advantage_normalization,
            advantage_group_count=advantage_group_count,
        ),
        optimizer,
    )


def _write_dataset(
    group: h5py.Group,
    name: str,
    value: np.ndarray,
    *,
    policy_input_eligible: bool,
) -> None:
    array = np.asarray(value)
    kwargs: dict[str, Any] = {}
    if array.nbytes >= 1024 and array.dtype.kind not in {"U", "O"}:
        kwargs.update(compression="lzf", shuffle=True)
    if array.dtype.kind in {"U", "O"}:
        dataset = group.create_dataset(
            name,
            data=np.asarray(array, dtype=h5py.string_dtype("utf-8")),
        )
    else:
        dataset = group.create_dataset(name, data=array, **kwargs)
    dataset.attrs["policy_input_eligible"] = bool(policy_input_eligible)


def write_online_rollout_h5_v1(
    path: Path,
    batch: AsymmetricMultiViewRolloutBatchV1,
    *,
    update_index: int,
    provenance: AsymmetricMultiViewProvenanceV1,
    config: AsymmetricMultiViewPPOConfigV1,
    scene_sha256: str,
    resume_lineage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist one exact online rollout; high-resolution replay is separate."""

    batch.validate()
    provenance.validate()
    config.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    if path.exists() or partial.exists():
        raise FileExistsError(f"rollout output already exists: {path}")
    with h5py.File(partial, "w") as stream:
        stream.attrs.update(
            {
                "format": H5_FORMAT,
                "source_type": SOURCE_TYPE,
                "policy_format": POLICY_FORMAT,
                "update_index": int(update_index),
                "production_admission": False,
                "diagnostic_only": True,
                "synthetic_only": True,
                "physical_samples": 0,
                "expert_calls": 0,
                "warm_start": False,
                "scratch_checkpoint_resume": resume_lineage is not None,
                "expert_warm_start": False,
                "external_pretraining": False,
                "behavior_cloning_steps": 0,
                "zero_mean_action_head_initialization": (provenance.zero_mean_action_head_initialization),
                "actor_privileged_state_inputs": 0,
                "actor_training_only_privileged_geometry_supervision": (
                    provenance.actor_training_only_privileged_geometry_supervision
                ),
                "visual_geometry_target_dimension": provenance.visual_geometry_target_dimension,
                "visual_geometry_target_schema": provenance.visual_geometry_target_schema,
                "critic_privileged_state_inputs": PRIVILEGED_EFFECT_STATE_DIM,
                "policy_action_space": provenance.policy_action_space,
                "policy_action_dimension": POLICY_ACTION_DIM,
                "action_autoregressive_rho": config.action_autoregressive_rho,
                "autoregressive_policy_state": "previous_policy_pre_tanh",
                "autoregressive_log_likelihood_recomputed_by_ppo": True,
                "joint_command_dimension": JOINT_ACTION_DIM,
                "low_level_action_adapter": provenance.low_level_action_adapter,
                "low_level_adapter_privileged_block_pose": (
                    provenance.low_level_adapter_privileged_block_pose
                ),
                "low_level_adapter_privileged_reset_pose": (
                    provenance.low_level_adapter_privileged_reset_pose
                ),
                "curriculum_reset_deployment_equivalent": (provenance.curriculum_reset_deployment_equivalent),
                "deployment_adapter_requires_visual_pose_estimator": (
                    provenance.deployment_adapter_requires_visual_pose_estimator
                ),
                "stock_follower_unmodified": True,
                "added_contact_tool": False,
                "execution_kernel_format": batch.execution_kernel_format,
                "rollout_contact_identity_format": (batch.contact_identity_format),
                "contact_candidate_geom_count": (batch.contact_candidate_geom_count),
                "safety_only_geom_count": batch.safety_only_geom_count,
                "safety_guard_format": batch.safety_guard_format,
                "safety_guard_float32_forecast_execution_identity": True,
                "safety_guard_fixed_braking_tail": True,
                "safety_guard_latched_absolute_target_hold": True,
                "safety_guard_recovery_uses_identical_guard": True,
                "safety_guard_no_safe_recovery_behavior": ("terminal_failure_before_env_step"),
                "shield_rejection_is_physics_transition": False,
                "literal_zero_action_hold_semantics": False,
                "potential_reward_config_sha256": (batch.potential_reward_config_sha256),
                "shaping_gamma": batch.shaping_gamma,
                "rollout_format": batch.rollout_format,
                "contact_metric_source": "all_physics_substeps_not_frame_end",
                "transient_contact_persisted": True,
                "contact_telemetry_actor_input": False,
                "camera_views_json": json.dumps(VIEW_NAMES),
                "camera_names_json": json.dumps(CAMERA_NAMES),
                "online_rgb_resolution_json": json.dumps([config.image_height, config.image_width]),
                "online_depth_present": False,
                "online_segmentation_present": False,
                "high_resolution_replay_pending": True,
                "task_instruction_en": TASK_INSTRUCTION_EN,
                "task_instruction_zh": TASK_INSTRUCTION_ZH,
                "scene_sha256": scene_sha256,
                "genesis_sha256": provenance.genesis_sha256,
                "rollout_seed": batch.rollout_seed,
                "rollout_rows": len(batch.rewards),
                "completed_episodes": batch.completed_episode_count,
                "strict_success_transitions": int(np.count_nonzero(batch.strict_success)),
                "execution_attempted_transitions": int(np.count_nonzero(batch.execution_attempted)),
                "shield_rejected_before_step_transitions": int(
                    np.count_nonzero(batch.shield_rejected_before_step)
                ),
                "valid_push_side_contact_transitions": int(
                    np.count_nonzero(batch.valid_push_side_contact_any)
                ),
                "transient_valid_push_side_contact_transitions": int(
                    np.count_nonzero(batch.valid_push_side_contact_transient_only)
                ),
                "resume_lineage_json": json.dumps(
                    _json_safe(resume_lineage or {}),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )
        policy = stream.create_group("policy_observation")
        for name in (
            "rgb_frames",
            "policy_rgb_history",
            "joint_state",
            "policy_joint_history",
            "previous_executed_action",
            "policy_action_history",
            "history_valid",
            "view_valid",
            "policy_view_history_valid",
            "history_row_indices",
            "camera_pose",
            "task_ids",
            "previous_policy_pre_tanh",
        ):
            _write_dataset(
                policy,
                name,
                np.asarray(getattr(batch, name)),
                policy_input_eligible=True,
            )
        execution = stream.create_group("execution")
        for name in (
            "policy_action",
            "applied_task_action",
            "executed_action",
            "submitted_joint_action",
            "execution_attempted",
            "shield_rejected_before_step",
            "ik_target_position_world_m",
            "ik_target_face_normal_world",
            "ik_application_scale",
            "ik_converged",
            "ik_face_label",
            "ik_failure_reason",
            "guard_safe_candidate",
            "guard_selected_scale",
            "guard_minimum_one_step_clearance_m",
            "guard_minimum_braking_clearance_m",
            "guard_float32_execution_identity",
            "tool_block_contact_any",
            "tool_block_contact_substep_count",
            "valid_push_side_contact_any",
            "valid_push_side_contact_substep_count",
            "valid_push_side_contact_transient_only",
            "invalid_tool_block_contact_any",
            "valid_push_side_peak_normal_force_n",
            "valid_push_side_normal_impulse_discrete_ns",
            "minimum_executed_94_safety_only_block_clearance_m",
            "step_block_displacement_m",
            "requested_joint_target",
            "queued_safe_joint_target",
            "applied_joint_target",
            "episode_ids",
            "episode_step_ids",
        ):
            _write_dataset(
                execution,
                name,
                np.asarray(getattr(batch, name)),
                policy_input_eligible=False,
            )
        _write_dataset(
            execution,
            "minimum_executed_safety_only_block_clearance_m",
            np.asarray(batch.minimum_executed_94_safety_only_block_clearance_m),
            policy_input_eligible=False,
        )
        execution["minimum_executed_94_safety_only_block_clearance_m"].attrs[
            "compatibility_alias_actual_safety_only_geom_count"
        ] = batch.safety_only_geom_count
        evidence = stream.create_group("reward_and_outcome")
        for name in (
            "raw_environment_reward",
            "reward_before_potential",
            "potential_before",
            "potential_after",
            "potential_next_for_shaping",
            "shaped_rewards",
            "terminated",
            "truncated",
            "strict_success",
            "terminal_failure",
            "safety_stop",
            "terminal_reason",
        ):
            _write_dataset(
                evidence,
                name,
                np.asarray(getattr(batch, name)),
                policy_input_eligible=False,
            )
        training_only = stream.create_group("training_only")
        for name in (
            "privileged_state",
            "next_privileged_state",
            "visual_geometry_target",
            "values",
            "next_values",
            "old_log_probs",
            "pre_tanh",
            "sim_qpos",
            "sim_qvel",
        ):
            _write_dataset(
                training_only,
                name,
                np.asarray(getattr(batch, name)),
                policy_input_eligible=False,
            )
        episode_json = np.asarray(
            [
                json.dumps(_json_safe(record), ensure_ascii=False, sort_keys=True)
                for record in batch.episode_records
            ],
            dtype=h5py.string_dtype("utf-8"),
        )
        stream.create_dataset("episode_randomization_json", data=episode_json)
        stream.flush()
    partial.replace(path)
    return {
        "path": str(path),
        "sha256": sha256_file_v1(path),
        "byte_count": path.stat().st_size,
        "rows": len(batch.rewards),
        "episodes": batch.completed_episode_count,
        "strict_success_transitions": int(np.count_nonzero(batch.strict_success)),
        "execution_attempted_transitions": int(np.count_nonzero(batch.execution_attempted)),
        "shield_rejected_before_step_transitions": int(np.count_nonzero(batch.shield_rejected_before_step)),
        "valid_push_side_contact_transitions": int(np.count_nonzero(batch.valid_push_side_contact_any)),
        "transient_valid_push_side_contact_transitions": int(
            np.count_nonzero(batch.valid_push_side_contact_transient_only)
        ),
        "production_admission": False,
    }


def parameter_counts_v1(bundle: AsymmetricMultiViewBundleV1) -> dict[str, int]:
    return {
        "actor": sum(parameter.numel() for parameter in bundle.actor.parameters()),
        "critic": sum(parameter.numel() for parameter in bundle.critic.parameters()),
        "total": sum(parameter.numel() for parameter in bundle.actor.parameters())
        + sum(parameter.numel() for parameter in bundle.critic.parameters()),
    }


__all__ = [
    "ACTOR_ARCHITECTURE",
    "CAUSAL_VISUAL_INFERENCE_CACHE_FORMAT_V24",
    "CAMERA_NAMES",
    "CHECKPOINT_FORMAT",
    "CRITIC_ARCHITECTURE",
    "EVALUATION_FORMAT",
    "H5_FORMAT",
    "JOINT_ACTION_DIM",
    "MAX_CURRICULUM_RESET_ATTEMPTS_V16",
    "POLICY_FORMAT",
    "POLICY_ACTION_DIM",
    "RESET_RETRY_STRIDE_V11",
    "RESET_RETRY_STRIDE_V12",
    "RESET_RETRY_STRIDE_V13",
    "ROLLOUT_FORMAT",
    "SOURCE_TYPE",
    "TASK_INSTRUCTION_EN",
    "TASK_INSTRUCTION_ZH",
    "VISUAL_GEOMETRY_TARGET_DIM",
    "VISUAL_GEOMETRY_TARGET_SCHEMA",
    "VIEW_NAMES",
    "AsymmetricMultiViewBundleV1",
    "AsymmetricMultiViewPPOConfigV1",
    "AsymmetricMultiViewProvenanceV1",
    "AsymmetricMultiViewRolloutBatchV1",
    "AsymmetricPrivilegedCriticV1",
    "CausalVisualInferenceCacheV24",
    "MultiViewRGBRendererV1",
    "PPOUpdateMetricsV1",
    "PrivilegedFeedbackTaskSpaceIKAdapterV5",
    "SelectedViewRecurrentActorV1",
    "StockGripperTaskFrameAdapterV9",
    "StockGripperTaskFrameAdapterV12",
    "StockGripperTaskFrameAdapterV13",
    "StockGripperTaskSpaceActionConfigV9",
    "StockGripperTaskSpaceActionConfigV12",
    "StockTaskFrameResetInfeasibleV11",
    "StockTaskFrameResetInfeasibleV12",
    "StockTaskFrameNoSafeRecoveryV13",
    "TaskSpaceActionConfigV5",
    "TaskSpaceActionResultV5",
    "autoregressive_applied_feedback_state_v21",
    "autoregressive_action_distribution_v1",
    "autoregressive_ik_guarded_feedback_state_v21",
    "canonical_sha256_v1",
    "collect_asymmetric_multiview_rollout_v1",
    "evaluate_asymmetric_multiview_policy_v1",
    "initialize_asymmetric_multiview_ppo_v1",
    "normalize_advantages_by_group_v24",
    "parameter_counts_v1",
    "ppo_update_asymmetric_multiview_v1",
    "reset_stock_taskframe_episode_v11",
    "reset_stock_taskframe_episode_v12",
    "reset_stock_taskframe_episode_v13",
    "sha256_file_v1",
    "transition_contact_telemetry_v10",
    "visual_geometry_target_from_privileged_v1",
    "write_online_rollout_h5_v1",
]
