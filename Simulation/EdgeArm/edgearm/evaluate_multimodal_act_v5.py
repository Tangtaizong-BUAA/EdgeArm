"""Matched-start, prior-only simulation probe for ACT V5 and its V3 baseline.

Privileged geometry is available only in post-action evaluation telemetry.
This probe deliberately makes no exact-Home, real-robot or VLA acceptance claim.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .candidate_command_contract_v2 import ACTION_CONTRACT, execution_command
from .materialize_temporal_packets_v3 import kinematics
from .multimodal_act_v5 import MultimodalACTV5
from .sparse_4d_vla_act_v26 import Sparse4DVLAConfigV26
from .temporal_input_contract_v3 import INPUT_KEYS
from .temporal_online_v3 import TemporalOnlineBuffer
from .temporal_spatial_policy_v3 import TemporalSpatialPolicyV3
from .train_multimodal_act_v5 import FORMAT, sha256
from .train_staged_hybrid_contact_sac import _atomic_json


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("action_contract") != ACTION_CONTRACT
        or frozenset(checkpoint["input_keys"]) != INPUT_KEYS
    ):
        raise ValueError("action or causal input contract mismatch")
    cfg = Sparse4DVLAConfigV26(**checkpoint["config"])
    if checkpoint["format"] == FORMAT:
        model = MultimodalACTV5(cfg)
    elif checkpoint["format"] == "temporal_spatial_v3_integration":
        model = TemporalSpatialPolicyV3(cfg)
    else:
        raise ValueError("unreviewed checkpoint architecture")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def rollout(model, checkpoint, seed, output, max_steps, contact_profile="strict", *, record_video=True):
    from .contact_audit_run33 import ContactAudit

    with ContactAudit(contact_profile) as audit:
        return _rollout(model, checkpoint, seed, output, max_steps, audit, record_video=record_video)


def _rollout(model, checkpoint, seed, output, max_steps, contact_audit, *, record_video=True):
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image
    from .staged_push_rl import (
        StagedPushEpisode,
        StagedPushStage,
        transition_contact_telemetry_v22,
        V10SafetyFilterInfeasible,
    )

    episode = StagedPushEpisode(seed=seed, scene_mode="multichoice_v1")
    episode.reset(seed=seed, stage=StagedPushStage.CONTACT_TRANSPORT_HOLD)
    env = episode.env
    f = 60 / np.tan(np.deg2rad(env.model.cam_fovy[env._ids["cameras"]["wrist"]]) / 2)
    intrinsics = np.array([[f, 0, 79.5], [0, f, 59.5], [0, 0, 1]], np.float32)
    buffer = TemporalOnlineBuffer(model.config, episode.multichoice.contract["instruction"], intrinsics)
    device = next(model.parameters()).device
    renderer = mujoco.Renderer(env.model, width=640, height=480)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0.22, 0, 0.08]
    camera.distance, camera.azimuth, camera.elevation = 0.78, 137, -38
    writer = (
        imageio.get_writer(
            output / f"{seed}_overview.mp4", fps=10, codec="libx264", quality=7, macro_block_size=None
        )
        if record_video
        else None
    )
    initial_distance = float(env.distance_to_target())
    initial_joint = np.asarray(env.observation()["joint_state"]).copy()
    initial_block = env.block_xy().copy()
    initial_coverage = float(env.block_target_coverage())
    invalid = contact = effectful = safety_rewrites = 0
    maximum_coverage = initial_coverage
    maximum_hold = 0.0
    minimum_distance = initial_distance
    maximum_displacement = 0.0
    success = False
    reason = "probe_step_limit"
    trace, telemetry_rows, latency, audits = [], [], [], []
    timings = dict(
        render=0.0,
        observation_and_inputs=0.0,
        inference=0.0,
        command_transfer=0.0,
        physics_and_protection=0.0,
    )
    began = time.time()
    try:
        for step in range(max_steps):
            tick = time.perf_counter()
            renderer.update_scene(env.data, camera="edgearm_wrist")
            rgb = np.asarray(Image.fromarray(renderer.render()).resize((160, 120)))
            if writer is not None and step % 3 == 0:
                renderer.update_scene(env.data, camera=camera)
                writer.append_data(renderer.render())
            timings["render"] += time.perf_counter() - tick
            tick = time.perf_counter()
            reported = np.asarray(env.observation()["joint_state"], np.float32)
            tool, pose = kinematics(env, reported[:6], reported[6:])
            buffer.observe(
                rgb=rgb,
                joint=reported,
                tool=tool,
                camera_pose=pose,
                time_s=float(env.data.time),
                geometry_valid=True,
            )
            inputs = {k: v.to(device) for k, v in buffer.tensors().items()}
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize(device)
            begin = time.perf_counter()
            timings["observation_and_inputs"] += begin - tick
            with torch.inference_mode():
                prediction = model(inputs, return_aux=True)
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize(device)
            latency.append(time.perf_counter() - begin)
            timings["inference"] += latency[-1]
            tick = time.perf_counter()
            if not torch.isfinite(prediction["action"]).all():
                raise ValueError("nonfinite action; no command submitted")
            command = execution_command(
                prediction["action"][0, 0].cpu().numpy(),
                checkpoint["action_contract"],
                env.config.max_joint_delta,
            )
            if step in (0, 31, 92):
                audits.append(
                    dict(
                        step=step,
                        visual_frames=int(inputs["visual_history_mask"].sum()),
                        proprio_frames=int(inputs["proprio_history_mask"].sum()),
                        completed_feedback_frames=int(inputs["command_feedback_mask"].sum()),
                        external_depth_pixels=int(inputs["depth_valid_mask_window"].sum()),
                        reference_points=int(inputs["reference_path_mask"].sum()),
                        future_teacher_actions_supplied=False,
                    )
                )
            # Object/target quantities are evaluation-only, after actor inference.
            before = env.block_xy().copy()
            contact_audit.clear()
            timings["command_transfer"] += time.perf_counter() - tick
            tick = time.perf_counter()
            try:
                _, _, terminated, truncated, info = env.step(command)
            except V10SafetyFilterInfeasible:
                timings["physics_and_protection"] += time.perf_counter() - tick
                reason = "safety_filter_infeasible"
                break
            timings["physics_and_protection"] += time.perf_counter() - tick
            transfer = info["sim2real_v2"]
            after = np.asarray(env.observation()["joint_state"], np.float32)
            buffer.complete_action(
                submitted_command=command,
                applied_target=transfer["applied_queued_safe_joint_target"],
                reported_next_q=after[:6],
                feedback_valid=not transfer["submitted_command_ingress_lost"],
            )
            trace.append(np.r_[reported, command, after[:6]])
            telemetry = transition_contact_telemetry_v22(
                info, block_before_xy_m=before, block_after_xy_m=env.block_xy()
            )
            invalid += int(telemetry["invalid_tool_block_contact_any"])
            contact += int(telemetry["valid_push_side_contact_any"])
            moved = float(np.linalg.norm(env.block_xy() - before))
            effectful += int(telemetry["valid_push_side_contact_any"] and moved > 1e-5)
            safety_rewrites += int(
                env.last_command_feedback_v1.get("applied_action_was_safety_modified", False)
            )
            coverage = float(env.block_target_coverage())
            hold = float(env._strict_success_streak) * env.control_dt
            distance = float(env.distance_to_target())
            displacement = float(np.linalg.norm(env.block_xy() - initial_block))
            maximum_coverage = max(maximum_coverage, coverage)
            maximum_hold = max(maximum_hold, hold)
            minimum_distance = min(minimum_distance, distance)
            maximum_displacement = max(maximum_displacement, displacement)
            telemetry_rows.append(
                [
                    float(env.data.time),
                    distance,
                    coverage,
                    hold,
                    displacement,
                    int(telemetry["valid_push_side_contact_any"]),
                    int(telemetry["invalid_tool_block_contact_any"]),
                ]
            )
            from .contact_audit_run33 import profile_failure

            physical_failure = profile_failure(
                contact_audit.profile,
                info["physics_substep_contact_v1"],
                episode.multichoice.failed,
                contact_audit.physical_geometry_invalid,
            )
            if physical_failure or (contact_audit.profile == "strict" and invalid):
                reason = "unsafe_contact"
                break
            success = bool(info.get("success", False))
            if terminated or truncated:
                reason = str(info.get("terminal_reason", "terminal"))
                break
            if step % 100 == 99:
                _atomic_json(
                    output / "episode_progress.json",
                    dict(
                        seed=seed,
                        steps=step + 1,
                        maximum_coverage=maximum_coverage,
                        maximum_hold_s=maximum_hold,
                        contact_steps=contact,
                        elapsed_seconds=time.time() - began,
                        production_admission=False,
                    ),
                )
        renderer.update_scene(env.data, camera=camera)
        final_rgb = renderer.render()
        if writer is not None:
            writer.append_data(final_rgb)
        Image.fromarray(final_rgb).save(output / f"{seed}_final.png")
    finally:
        if writer is not None:
            writer.close()
        renderer.close()
    np.savez_compressed(
        output / f"{seed}_trace.npz",
        reported_command_next_reported=np.asarray(trace),
        evaluation_telemetry=np.asarray(telemetry_rows),
        initial_joint=initial_joint,
    )
    return dict(
        seed=seed,
        steps=len(trace),
        strict_success=success,
        safe_success=success
        and reason != "unsafe_contact"
        and (contact_audit.profile == "task_goal_v1" or not episode.multichoice.failed),
        task_goal_success=success and reason != "unsafe_contact",
        strict_contact_success=success and invalid == 0 and not episode.multichoice.failed,
        contact_profile=contact_audit.profile,
        reason=reason,
        initial_distance_m=initial_distance,
        initial_coverage=initial_coverage,
        final_distance_m=float(env.distance_to_target()),
        maximum_distance_progress_m=initial_distance - minimum_distance,
        maximum_block_displacement_m=maximum_displacement,
        maximum_coverage=maximum_coverage,
        maximum_hold_s=maximum_hold,
        valid_contact_steps=contact,
        invalid_contact_steps=invalid,
        effectful_contact_steps=effectful,
        safety_rewrite_steps=safety_rewrites,
        input_audits=audits,
        inference_median_ms=float(np.median(latency) * 1000),
        inference_p95_ms=float(np.quantile(latency, 0.95) * 1000),
        multichoice_scene=episode.multichoice.audit(),
        elapsed_seconds=time.time() - began,
        timing_seconds=timings,
        start_stage="CONTACT_TRANSPORT_HOLD",
        exact_home_evaluated=False,
        production_admission=False,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--max-steps", type=int, default=900)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    p.add_argument(
        "--contact-profile", choices=("strict", "direction_tolerant_v1", "task_goal_v1"), default="strict"
    )
    args = p.parse_args()
    if args.max_steps < 1 or args.threads < 1 or len(set(args.seeds)) != len(args.seeds):
        p.error("positive limits and distinct seeds required")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    model, checkpoint = load_model(args.checkpoint)
    model.to(args.device)
    rows = []
    state = dict(
        status="running",
        device=args.device,
        seeds=args.seeds,
        checkpoint_sha256=sha256(args.checkpoint),
        checkpoint_epoch=checkpoint["epoch"],
        max_steps=args.max_steps,
        selection_role="development_probe_not_blind_acceptance",
        prior_only=True,
        safety_stop=(
            "desk penetration and existing environment/device guards; object contacts are diagnostic only"
            if args.contact_profile == "task_goal_v1"
            else "physical geometry/desk/unselected contact; direction stop depends on named profile"
        ),
        contact_profile=args.contact_profile,
        production_admission=False,
    )
    _atomic_json(output / "summary.json", state)
    try:
        for seed in args.seeds:
            row = rollout(model, checkpoint, seed, output, args.max_steps, args.contact_profile)
            _atomic_json(output / f"{seed}.json", row)
            rows.append(row)
            state.update(
                status="complete" if len(rows) == len(args.seeds) else "running",
                episodes=rows,
                completed=len(rows),
                safe_successes=sum(r["safe_success"] for r in rows),
                exact_home_evaluated=False,
                final_vla_acceptance=False,
            )
            _atomic_json(output / "summary.json", state)
            print(json.dumps(row), flush=True)
    except BaseException as exc:
        state.update(status="failed", exception=repr(exc))
        _atomic_json(output / "summary.json", state)
        raise


if __name__ == "__main__":
    main()
