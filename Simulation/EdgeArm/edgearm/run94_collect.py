"""DAgger collection using frozen spatial perception and current learned actions.

Dedicated training seeds only. Teacher labels are queried after the student
prediction and never enter the actor. Mixed execution is explicitly recorded.
"""

from pathlib import Path
import time

import numpy as np
import torch

from .candidate_command_contract_v2 import ACTION_CONTRACT
from .run34_repeat_eval import deterministic_runtime
from .run53_recovery import RecoveryTeacher
from .run42.domain import sample_domain
from .run42.session import DomainSession
from .run61_active_view import SurveyAndReturn
from .run63_control_probe import TinyTarget
from .run67_visual_state import VisualState, actor_input, language_indices
from .run68_visual_control import estimated_inputs
from .run74_observation_probe import observation_history_indices
from .run78_completion_probe import CompletionLatch, TerminalServo
from .run82_spatial_model import SparseSpatialPolicy
from .run88_keypoint_model import WristKeypoints
from .run91_static_memory import StaticSurveyMemory
from .sparse_4d_vla_act_v26 import Sparse4DVLAConfigV26
from .train_staged_hybrid_contact_sac import _atomic_json


MODES = ("memory_geometric", "memory_fast_keypoints")


def keypoint_due(mode, step, map_due):
    if mode not in MODES or step < 0:
        raise ValueError("declared observation schedule required")
    return map_due or (mode == "memory_fast_keypoints" and step > 220)


class FreshCompletionLatch(CompletionLatch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_observation_step = -1

    def observe(self, selected_xy, step, observation_step):
        if observation_step > step or observation_step < self.last_observation_step:
            raise ValueError("causal observation identity required")
        if observation_step == self.last_observation_step:
            return self.latched
        self.last_observation_step = int(observation_step)
        return self.update(selected_xy, step)


def factor_command(mode, proprio, selected, world, memory, anchor_control, learned):
    """Only proprioception, image-derived map and frozen modules are accepted."""
    if mode not in MODES:
        raise ValueError("undeclared contrast")
    if mode in ("memory_full", "memory_fresh_latch"):
        return learned.command(proprio, selected, memory)
    control = (
        anchor_control
        if mode
        in ("anchor_dense", "anchor_sparse", "memory_oldcontrol", "memory_geometric", "memory_fast_keypoints")
        else learned.control
    )
    return control(estimated_inputs(proprio, world, selected))


def image_indices(step, available, sparse):
    requested = observation_history_indices(step, step >= 220)
    if sparse:
        positions = np.maximum(0, np.searchsorted(available, requested, side="right") - 1)
        requested = np.asarray(available)[positions]
    if np.any(requested > step):
        raise ValueError("future image")
    return requested


def validate_collection(seed, beta, mode):
    if (
        not (100500000 <= seed // 9 < 100500012 or seed // 9 == 100500020)
        or mode != "memory_fast_keypoints"
        or beta not in (0.0, 0.25, 0.5)
    ):
        raise ValueError("dedicated bounded Run94 collection only")


def episode(job):
    seed, checkpoint, vision_path, control_path, keypoint_path, output, mode, beta = job
    validate_collection(seed, beta, mode)
    deterministic_runtime()
    torch.set_num_threads(1)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("policy_kind") != SparseSpatialPolicy.kind:
        raise ValueError("declared learned spatial policy required")
    learned = SparseSpatialPolicy().cuda().eval()
    learned.load_state_dict(saved["model"])
    vs = torch.load(vision_path, map_location="cpu", weights_only=True)
    visual = VisualState(recent_block_seconds=vs.get("recent_block_seconds")).cuda().eval()
    visual.load_state_dict(vs["model"])
    anchor = TinyTarget(118, 512).cuda().eval()
    anchor.load_state_dict(torch.load(control_path, map_location="cpu", weights_only=True)["model"])
    keypoint_saved = torch.load(keypoint_path, map_location="cpu", weights_only=True)
    if keypoint_saved.get("policy_kind") != WristKeypoints.kind:
        raise ValueError("declared keypoint checkpoint required")
    keypoint = WristKeypoints().cuda().eval()
    keypoint.load_state_dict(keypoint_saved["model"])
    correction = StaticSurveyMemory()
    keypoint_audit = []
    uses_memory = mode.startswith("memory_")
    sparse = uses_memory or mode == "anchor_sparse"
    session = DomainSession(
        Sparse4DVLAConfigV26(language_max_tokens=128, visual_memory_mode="episode_anchors_v54"),
        ACTION_CONTRACT,
        seed,
        sample_domain(seed + 6001, 0),
    )
    teacher = RecoveryTeacher(session, seed % 9, goal_retreat_coverage=0.995)
    if seed % 9 == 6:
        teacher.p["speed"] = 0.6
    rng = np.random.default_rng(seed + 9401)
    use_teacher = False
    examples, labels, phases, teacher_choices, raw_predictions = [], [], [], [], []
    survey = SurveyAndReturn(
        session.env.model, session.env._ids["tool_site"], session.env._ids["cameras"]["wrist"]
    )
    servo = TerminalServo(session.env.model, session.env._ids["tool_site"], "lift")
    latch = FreshCompletionLatch() if mode == "memory_fresh_latch" else CompletionLatch()
    folder = Path(output) / mode / f"episode_{seed}"
    folder.mkdir(parents=True, exist_ok=False)
    selection = language_indices(session.buffer.instruction)
    selected = torch.tensor([selection], device="cuda")
    q_history, commands, frames, trace, available = [], [], [], [], []
    memory = None
    last_observation = 0
    last_keypoint_observation = 0
    world = None
    trigger_audit = None
    contacts = effectful = rewrites = 0
    maximum_coverage = maximum_hold = 0.0
    started = time.time()
    try:
        session.observe()
        initial = session.reported[:6].copy()
        if session.reward_state()[1] != 0.0:
            raise ValueError("nonzero initial coverage")
        initial_block = session.env.block_xy().copy()  # only used by audit
        survey_displacement = 0.0
        return_error = None
        for step in range(900):
            session.observe()
            rows = session.buffer.rows
            previous = rows[-2]["applied_target"] if step else initial
            dummy = actor_input(
                session.reported,
                rows[-1]["tool"],
                q_history,
                commands,
                previous,
                initial,
                np.zeros((7, 2), np.float32),
                selection,
            )
            proprio = torch.from_numpy(np.r_[dummy[:108], dummy[112:118]])[None].cuda()
            with torch.inference_mode():
                map_due = not sparse or step % 8 == 0 or step in (90, 110, 220)
                if map_due:
                    available.append(step)
                    ids = image_indices(step, available, sparse)
                    rgb = torch.from_numpy(np.stack([rows[int(i)]["rgb"] for i in ids]))[None].cuda()
                    pose = torch.from_numpy(np.stack([rows[int(i)]["camera_pose"] for i in ids]))[None].cuda()
                    age = torch.tensor(
                        [[rows[int(i)]["time"] - rows[-1]["time"] for i in ids]], device="cuda"
                    )
                    K = torch.from_numpy(session.buffer.K)[None].cuda()
                    if uses_memory:
                        dt = torch.tensor([(step - last_observation) / 30], device="cuda")
                        _, memory = learned.step(rgb, pose, K, age, proprio, selected, dt, memory)
                    else:
                        world = visual(rgb, pose, K, age)
                    last_observation = step
                if keypoint_due(mode, step, map_due):
                    current_rgb = torch.from_numpy(rows[-1]["rgb"])[None].cuda()
                    current_pose = torch.from_numpy(rows[-1]["camera_pose"])[None].cuda()
                    current_K = torch.from_numpy(session.buffer.K)[None].cuda()
                    measurement = keypoint(current_rgb, current_pose, current_K)
                    memory, accepted = correction.update(memory, measurement, step)
                    world = memory["xyz"][..., :2]
                    last_keypoint_observation = step
                if step < 220:
                    action = survey.command(session.reported)
                else:
                    action = (
                        factor_command(mode, proprio, selected, world, memory, anchor, learned)[0]
                        .cpu()
                        .numpy()
                        .clip(-1, 1)
                    )
                    xy = world[0, list(selection)].cpu().numpy()
                    if correction.completion_observed(selection, step):
                        near = latch.update(xy, step)
                    else:
                        latch.history.clear()
                        near = latch.latched
                    if near:
                        action = servo.command(session.reported, previous)
            # Student prediction and stored input are fixed before privileged label queries.
            if step >= 220:
                with torch.inference_mode():
                    x = estimated_inputs(proprio, world, selected)[0].cpu().numpy()
                prediction = action.copy()
                label = teacher.command()
                if step == 220 or (step - 220) % 15 == 0:
                    use_teacher = rng.random() < beta
                examples.append(x.copy())
                labels.append(label.copy())
                phases.append(teacher.phase)
                teacher_choices.append(use_teacher)
                raw_predictions.append(prediction)
                action = label if use_teacher else prediction
            # Remaining truth is audit-only, never actor input.
            truth = np.stack((session.env.block_xy(), session.env.target_xy))
            estimate = world[0, list(selection)].cpu().numpy()
            keypoint_audit.append(
                np.r_[
                    step,
                    measurement["xyz"][0, list(selection), :2].cpu().numpy().ravel(),
                    measurement["confidence"][0, list(selection)].cpu().numpy(),
                    correction.last_accepted[0, list(selection)].cpu().numpy(),
                ]
            )
            metrics = session.reward_state()
            if latch.latched and trigger_audit is None:
                trigger_audit = dict(
                    step=step,
                    observation_step=last_observation,
                    true_distance_m=float(metrics[0]),
                    true_coverage=float(metrics[1]),
                    estimated_distance_m=float(np.linalg.norm(estimate[0] - estimate[1])),
                    audit_only=True,
                )
            if step == 220:
                return_error = float(np.max(np.abs(session.reported[:6] - initial)))
            if step < 220:
                survey_displacement = max(
                    survey_displacement, float(np.linalg.norm(truth[0] - initial_block))
                )
            trace.append(
                np.r_[
                    step,
                    estimate.ravel(),
                    truth.ravel(),
                    metrics,
                    np.linalg.norm(estimate - truth, axis=1),
                    latch.latched,
                    last_observation,
                    last_keypoint_observation,
                ]
            )
            q_history.append(session.reported.copy())
            commands.append(action.copy())
            if step % 8 == 0:
                frames.append(rows[-1]["rgb"].copy())
            result = session.advance(action)
            metrics = session.reward_state()
            contacts += int(result.get("valid_contact", False))
            effectful += int(result.get("effectful_contact", False))
            rewrites += int(result.get("safety_rewrite", False))
            maximum_coverage = max(maximum_coverage, metrics[1])
            maximum_hold = max(maximum_hold, metrics[2])
            if result["kind"] != "sampler_cut":
                break
        kind = session.end_kind if session.end_kind != "sampler_cut" else "finite_timeout"
        result = dict(
            seed=seed,
            mode=mode,
            safe_success=kind == "success",
            end_kind=kind,
            terminal_reason=session.reason,
            steps=len(commands),
            seconds=time.time() - started,
            maximum_coverage=float(maximum_coverage),
            maximum_hold_s=float(maximum_hold),
            valid_contact_steps=contacts,
            effectful_contact_steps=effectful,
            safety_rewrite_steps=rewrites,
            trigger_audit=trigger_audit,
            actor_uses_simulator_state=False,
            teacher_assisted=beta > 0,
            teacher_beta=beta,
            collection=True,
            teacher_queries=len(labels),
            teacher_executions=sum(teacher_choices),
            training_labels_include_failed_episodes=True,
            fixed_wrist_survey=True,
            fixed_survey_steps=220,
            scripted_completion=True,
            fresh_observation_completion=mode == "memory_fresh_latch",
            max_steps=900,
            survey_return_error_rad=return_error,
            survey_maximum_block_displacement_m=survey_displacement,
            model_updated=False,
            learned_image_keypoints=True,
            keypoint_update_stride_after_survey=1 if mode == "memory_fast_keypoints" else 8,
            latent_memory_update_stride=8,
            confidence_gated_completion=True,
            nominal_height_prior=True,
            independent_acceptance=False,
            production_admission=False,
            export_admission=False,
            final_vla_acceptance=False,
        )
        np.savez_compressed(
            folder / "trace.npz",
            audit=np.asarray(trace),
            reported=np.asarray(q_history),
            command=np.asarray(commands),
            wrist_rgb=np.asarray(frames),
            keypoint_audit=np.asarray(keypoint_audit),
        )
        np.savez_compressed(
            folder / "action_labels.npz",
            x=np.asarray(examples, np.float32),
            command=np.asarray(labels, np.float32),
            phase=np.asarray(phases),
            teacher_execution=np.asarray(teacher_choices, bool),
            student_prediction=np.asarray(raw_predictions, np.float32),
            time_step=np.arange(220, 220 + len(examples), dtype=np.int64),
        )
        _atomic_json(folder / "result.json", result)
        return result
    finally:
        session.close()
