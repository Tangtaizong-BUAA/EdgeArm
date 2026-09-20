"""Cloneable simulation transaction for causal, same-state interventions.

No robot I/O. This uses the same wrist camera, buffer, command contract and
task_goal_v1 safety profile as evaluate_multimodal_act_v5. Environment truth
is exposed only through reward_state/diagnostics, not observe().
"""

from copy import deepcopy
import hashlib
import random

import mujoco
import numpy as np
from PIL import Image
import torch

from .candidate_command_contract_v2 import execution_command
from .contact_audit_run33 import ContactAudit, profile_failure
from .materialize_temporal_packets_v3 import kinematics
from .staged_push_rl import StagedPushEpisode, StagedPushStage, V10SafetyFilterInfeasible, transition_contact_telemetry_v22
from .temporal_online_v3 import TemporalOnlineBuffer


def snapshot_episode(episode):
    """Copy full plant/controller metadata, RNGs, scratch data and scene state.

    Preserve the live model identity (renderer owns it) and bound callbacks.
    The model arrays mutated by step(), unlike immutable meshes, are copied.
    mj_copyData includes warmstarts/contacts/actuator state, not just qpos.
    """
    env = episode.env
    data = {}
    for name, value in vars(env).items():
        if isinstance(value, mujoco.MjData):
            data[name] = mujoco.MjData(env.model)
            mujoco.mj_copyData(data[name], env.model, value)
    metadata = {
        k: v for k, v in vars(env).items()
        if not isinstance(v, (mujoco.MjModel, mujoco.MjData)) and not callable(v)
    }
    return dict(
        data=data, metadata=deepcopy(metadata),
        model={k: getattr(env.model, k).copy() for k in ("actuator_forcerange", "cam_pos", "cam_quat")},
        scene=deepcopy({k: v for k, v in vars(episode.multichoice).items() if k != "env"}),
        episode=deepcopy({k: v for k, v in vars(episode).items() if k not in ("env", "multichoice")}),
        torch=torch.get_rng_state().clone(),
        cuda=[v.clone() for v in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else None,
        numpy=deepcopy(np.random.get_state()), python=random.getstate(),
    )


def restore_episode(episode, snapshot):
    env = episode.env
    for k, value in snapshot["model"].items():
        getattr(env.model, k)[:] = value
    for k, value in snapshot["data"].items():
        mujoco.mj_copyData(getattr(env, k), env.model, value)
    # Remove dynamic metadata created by the counterfactual branch.
    for k, value in list(vars(env).items()):
        if k not in snapshot["metadata"] and not isinstance(value, (mujoco.MjModel, mujoco.MjData)) and not callable(value):
            delattr(env, k)
    vars(env).update(deepcopy(snapshot["metadata"]))
    for k in list(vars(episode.multichoice)):
        if k != "env":
            delattr(episode.multichoice, k)
    vars(episode.multichoice).update(deepcopy(snapshot["scene"]))
    vars(episode).update(deepcopy(snapshot["episode"]))
    torch.set_rng_state(snapshot["torch"])
    if snapshot["cuda"] is not None:
        torch.cuda.set_rng_state_all(snapshot["cuda"])
    np.random.set_state(snapshot["numpy"])
    random.setstate(snapshot["python"])


def causal_input_digest(inputs):
    h = hashlib.sha256()
    for key in sorted(inputs):
        x = inputs[key].detach().cpu().contiguous().numpy()
        h.update(key.encode())
        h.update(str((x.shape, x.dtype)).encode())
        h.update(x.tobytes())
    return h.hexdigest()


def render_noise_metrics(reference, candidate):
    delta = abs(np.asarray(reference, np.int16) - np.asarray(candidate, np.int16))
    return dict(max_lsb=int(delta.max()), changed_values=int((delta != 0).sum()),
                changed_fraction=float((delta != 0).mean()), mean_lsb=float(delta.mean()))


def within_render_noise(metrics):
    # Measured no-action EGL repeats already differ by 1 LSB at ~6/57600
    # channels. This is a renderer-only tolerance; physics/history stay exact.
    return metrics["max_lsb"] <= 1 and metrics["changed_fraction"] <= 0.001


class BranchSession:
    def __init__(self, config, action_contract, seed, device):
        self.episode = StagedPushEpisode(seed=seed, scene_mode="multichoice_v1")
        self.episode.reset(seed=seed, stage=StagedPushStage.CONTACT_TRANSPORT_HOLD)
        self.env = self.episode.env
        self.device, self.action_contract = device, action_contract
        f = 60 / np.tan(np.deg2rad(self.env.model.cam_fovy[self.env._ids["cameras"]["wrist"]]) / 2)
        intrinsics = np.array([[f, 0, 79.5], [0, f, 59.5], [0, 0, 1]], np.float32)
        self.buffer = TemporalOnlineBuffer(config, self.episode.multichoice.contract["instruction"], intrinsics)
        self.renderer = mujoco.Renderer(self.env.model, width=640, height=480)
        self.audit = ContactAudit("task_goal_v1")
        self.audit.__enter__()
        self.inputs = None
        self.reported = None
        self.end_kind = "sampler_cut"
        self.reason = "nonterminal"

    def close(self):
        self.audit.__exit__(None, None, None)
        self.renderer.close()

    def reward_state(self):
        return np.array([self.env.distance_to_target(), self.env.block_target_coverage(),
                         self.env._strict_success_streak * self.env.control_dt], np.float64)

    def observe(self):
        if self.inputs is not None:
            return self.inputs
        self.renderer.update_scene(self.env.data, camera="edgearm_wrist")
        rgb = np.asarray(Image.fromarray(self.renderer.render()).resize((160, 120)))
        self.reported = np.asarray(self.env.observation()["joint_state"], np.float32)
        tool, pose = kinematics(self.env, self.reported[:6], self.reported[6:])
        self.buffer.observe(rgb=rgb, joint=self.reported, tool=tool, camera_pose=pose,
                            time_s=float(self.env.data.time), geometry_valid=True)
        self.inputs = {k: v.to(self.device) for k, v in self.buffer.tensors().items()}
        return self.inputs

    def advance(self, normalized_command):
        if self.inputs is None or not self.buffer.pending or self.end_kind != "sampler_cut":
            raise ValueError("advance requires a live observed transaction")
        command = execution_command(normalized_command, self.action_contract, self.env.config.max_joint_delta)
        self.audit.clear()
        before_block = self.env.block_xy().copy()
        try:
            _, _, terminated, truncated, info = self.env.step(command)
        except V10SafetyFilterInfeasible:
            # A rejected command is an attempted terminal transition with
            # unchanged physical state, not an unpenalized omitted decision.
            self.end_kind, self.reason = "hard_failure", "safety_filter_infeasible"
            return dict(kind=self.end_kind, reason=self.reason, rejected=True)
        after = np.asarray(self.env.observation()["joint_state"], np.float32)
        transfer = info["sim2real_v2"]
        self.buffer.complete_action(
            submitted_command=command, applied_target=transfer["applied_queued_safe_joint_target"],
            reported_next_q=after[:6], feedback_valid=not transfer["submitted_command_ingress_lost"],
        )
        self.inputs = None
        contact = transition_contact_telemetry_v22(info, block_before_xy_m=before_block,
                                                 block_after_xy_m=self.env.block_xy())
        physical_failure = profile_failure("task_goal_v1", info["physics_substep_contact_v1"],
                                           self.episode.multichoice.failed, self.audit.physical_geometry_invalid)
        self.reason = str(info.get("terminal_reason", "nonterminal"))
        if physical_failure:
            self.end_kind, self.reason = "hard_failure", "unsafe_contact"
        elif info.get("success", False):
            self.end_kind = "success"
        elif terminated:
            self.end_kind = "hard_failure" if self.reason.startswith("safety_stop") else "task_failure"
        elif truncated:
            self.end_kind = "finite_timeout"
        return dict(kind=self.end_kind, reason=self.reason, rejected=False, command=command,
                    trace=np.r_[self.reported, command, after[:6]],
                    valid_contact=bool(contact["valid_push_side_contact_any"]),
                    invalid_contact=bool(contact["invalid_tool_block_contact_any"]),
                    effectful_contact=bool(contact["valid_push_side_contact_any"] and
                                           np.linalg.norm(self.env.block_xy() - before_block) > 1e-5),
                    safety_rewrite=bool(self.env.last_command_feedback_v1.get("applied_action_was_safety_modified", False)))

    def snapshot(self):
        if self.inputs is None or not self.buffer.pending:
            raise ValueError("snapshot only at an observed pre-action boundary")
        return dict(episode=snapshot_episode(self.episode), buffer=deepcopy(self.buffer),
                    inputs={k: v.clone() for k, v in self.inputs.items()}, reported=self.reported.copy(),
                    end_kind=self.end_kind, reason=self.reason, input_sha256=causal_input_digest(self.inputs))

    def restore(self, snapshot):
        restore_episode(self.episode, snapshot["episode"])
        self.buffer = deepcopy(snapshot["buffer"])
        self.inputs = {k: v.clone() for k, v in snapshot["inputs"].items()}
        self.reported = snapshot["reported"].copy()
        self.end_kind, self.reason = snapshot["end_kind"], snapshot["reason"]
        self.audit.clear()
        reference = snapshot["episode"]["data"]["data"]
        for name in ("qpos", "qvel", "qacc_warmstart", "act", "ctrl", "qfrc_applied", "xfrc_applied",
                     "mocap_pos", "mocap_quat", "userdata"):
            if not np.array_equal(getattr(self.env.data, name), getattr(reference, name)):
                raise ValueError("restored physical integration state differs: " + name)
        if self.env.data.time != reference.time:
            raise ValueError("restored physical time differs")
        # Rebuild from restored history, rather than verifying a cached copy.
        rebuilt = {k: v.to(self.device) for k, v in self.buffer.tensors().items()}
        if causal_input_digest(rebuilt) != snapshot["input_sha256"]:
            raise ValueError("restored causal history differs")
