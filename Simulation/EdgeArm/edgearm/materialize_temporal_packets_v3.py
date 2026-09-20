"""Replay-derived temporal packets. Simulator depth is AUXILIARY ONLY."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image

from .legacy_keyboard_sparse_4d_dataset_v40 import _step_info
from .staged_push_rl import StagedPushEpisode, StagedPushStage
from .train_staged_hybrid_contact_sac import _atomic_json


def kinematics(env, q, v, camera_local=None):
    """Forward kinematics from reported joints; never reads object state."""
    d = mujoco.MjData(env.model)
    d.qpos[:6] = q
    d.qvel[:6] = v
    mujoco.mj_forward(env.model, d)
    site = env._ids["tool_site"]
    jp = np.zeros((3, env.model.nv))
    jr = jp.copy()
    mujoco.mj_jacSite(env.model, d, jp, jr, site)
    tool = np.r_[d.site_xpos[site], d.site_xmat[site], jp[:, :6] @ v, jr[:, :6] @ v]
    cam = env._ids["cameras"]["wrist"]
    return tool.astype("float32"), np.r_[d.cam_xpos[cam], d.cam_xmat[cam]].astype("float32")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-run", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--per-pair", type=int, default=2)
    p.add_argument("--human-count", type=int, default=2)
    p.add_argument("--splits", nargs="+", choices=("train", "validation"), default=["train", "validation"])
    p.add_argument(
        "--exclude-packet-manifest",
        help="Exclude previously used parent trajectories, without changing original partitions",
    )
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument(
        "--skip-auxiliary-depth",
        action="store_true",
        help="Frozen-depth action training needs no new simulator-depth labels",
    )
    a = p.parse_args()
    if a.shard_count < 1 or not 0 <= a.shard_index < a.shard_count:
        p.error("invalid shard bounds")
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    records = json.loads((Path(a.base_run) / "dataset_manifest.json").read_text())["records"]
    if a.exclude_packet_manifest:
        excluded = json.loads(Path(a.exclude_packet_manifest).read_text())["records"]
        used = {r["parent_source_path"] for r in excluded}
        groups = {r["group"] for r in excluded}
        records = [r for r in records if r["source_path"] not in used and r["group"] not in groups]
    selected = []
    for pair in sorted({r["pair"] for r in records if r["source"] == "rl"}):
        # Keep both existing partitions represented; never re-split derivative frames.
        for split in a.splits:
            selected.extend(
                [r for r in records if r.get("pair") == pair and r["split"] == split][: a.per_pair]
            )
    for split in a.splits:
        selected.extend(
            [r for r in records if r["source"] == "human" and r["split"] == split][: a.human_count]
        )
    selected = selected[a.shard_index :: a.shard_count]
    admitted = []
    _atomic_json(
        out / "run_state.json",
        dict(
            status="materializing",
            completed=0,
            total=len(selected),
            pid=os.getpid(),
            production_admission=False,
        ),
    )
    for index, r in enumerate(selected):
        n = r["length"]
        source = r["source"]
        rgb = []
        tool = []
        pose = []
        states = []
        applied = []
        errors = []
        feedback = []
        depth = np.zeros((n, 120, 160), np.uint16)
        depth_mask = np.zeros(n, bool)
        with np.load(r["command_cache"]) as z:
            commands = z["command"].copy()
            valid = z["valid"].copy()
        ep = StagedPushEpisode(seed=r.get("seed", 11000020), scene_mode="multichoice_v1")
        ep.reset(seed=r.get("seed", 11000020), stage=StagedPushStage.CONTACT_TRANSPORT_HOLD)
        env = ep.env
        if source == "rl":
            with np.load(r["trajectory"]) as z:
                physical_q = z["qpos"].copy()
                times = z["time"][:-1].copy()
                replay_commands = z["submitted_joint_command"][1:].copy()
            reader = imageio.get_reader(Path(r["source_path"]).parent / "wrist_rgb.mp4")
            renderer = None if a.skip_auxiliary_depth else mujoco.Renderer(env.model, width=160, height=120)
            try:
                for t in range(n):
                    np.testing.assert_allclose(env.data.qpos, physical_q[t], rtol=0, atol=1e-8)
                    state = np.asarray(env.observation()["joint_state"], dtype=np.float32)
                    states.append(state)
                    fk, cam = kinematics(env, state[:6], state[6:])
                    tool.append(fk)
                    pose.append(cam)
                    rgb.append(np.asarray(Image.fromarray(reader.get_data(t)).resize((160, 120))))
                    if renderer is not None and t % 8 == 0:
                        renderer.enable_depth_rendering()
                        renderer.update_scene(env.data, camera="edgearm_wrist")
                        d = renderer.render()
                        renderer.disable_depth_rendering()
                        depth[t] = np.where((d > 0.01) & (d < 2), np.round(d * 1000), 0).astype(np.uint16)
                        depth_mask[t] = True
                    _, _, _, _, info = env.step(replay_commands[t])
                    tr = info["sim2real_v2"]
                    target = np.asarray(tr["applied_queued_safe_joint_target"], dtype=np.float32)
                    applied.append(target)
                    errors.append(target - np.asarray(env.observation()["joint_state"][:6]))
                    feedback.append(not tr["submitted_command_ingress_lost"])
                np.testing.assert_allclose(env.data.qpos, physical_q[-1], rtol=0, atol=1e-8)
            finally:
                reader.close()
                if renderer is not None:
                    renderer.close()
            # Intrinsics at the actual encoder image size; camera pose is an FK estimate.
            f = 60 / np.tan(np.deg2rad(env.model.cam_fovy[env._ids["cameras"]["wrist"]]) / 2)
            K = np.array([[f, 0, 79.5], [0, f, 59.5], [0, 0, 1]], np.float32)
            geometry = np.ones(n, bool)
            provenance = "legacy_staged_rl_not_certified_scratch"
            depth_source = (
                "not_generated_frozen_depth_action_training"
                if a.skip_auxiliary_depth
                else "deterministic_replay_simulator_depth_auxiliary_only"
            )
        else:
            with h5py.File(r["source_path"]) as h:
                s = h["steps"]
                q = s["q_before_rad"][:]
                v = s["dq_before_rad_s"][:]
                times = s["capture_monotonic_ns"][:].astype(float) / 1e9
                times -= times[0]
                cm = json.loads(h.attrs["camera_metadata_json"])["wrist"]
                K = np.asarray(cm["intrinsics"], np.float32)
                K[0] *= 160 / cm["width"]
                K[1] *= 120 / cm["height"]
                for t in range(n):
                    states.append(np.r_[q[t], v[t]])
                    fk, _ = kinematics(env, q[t], v[t])
                    tool.append(fk)
                    # Per-frame original camera extrinsics are not recoverable from this old capture.
                    pose.append(np.r_[np.zeros(3), np.eye(3).ravel()])
                    rgb.append(np.asarray(Image.fromarray(s["wrist_rgb"][t]).resize((160, 120))))
                    if not a.skip_auxiliary_depth:
                        d = np.asarray(
                            Image.fromarray(s["depth_wrist_mm"][t]).resize(
                                (160, 120), Image.Resampling.NEAREST
                            )
                        )
                        depth[t] = d
                        depth_mask[t] = True
                    tr = _step_info(s["step_info_json_zlib"], t)["sim2real_v2"]
                    target = np.asarray(tr["applied_queued_safe_joint_target"])
                    applied.append(target)
                    errors.append(target - s["q_after_rad"][t])
                    feedback.append(not tr["submitted_command_ingress_lost"])
            geometry = np.zeros(n, bool)
            provenance = "sim_human_legacy"
            depth_source = (
                "not_generated_frozen_depth_action_training"
                if a.skip_auxiliary_depth
                else "legacy_simulated_rgbd_auxiliary_only"
            )
        states = np.asarray(states, dtype=np.float32)
        motion = np.zeros((n, 6), np.float32)
        motion[1:] = (states[1:, :6] - states[:-1, :6]) / 0.1
        path = out / f"episode_{index:03d}.npz"
        np.savez_compressed(
            path,
            rgb=np.asarray(rgb),
            joint=states,
            previous_motion=motion,
            tool=np.asarray(tool),
            camera_pose=np.asarray(pose),
            time=times,
            K=K,
            geometry_valid=geometry,
            command=commands,
            command_valid=valid,
            applied_target=np.asarray(applied, dtype=np.float32),
            tracking_error=np.asarray(errors, dtype=np.float32),
            feedback_valid=np.asarray(feedback, bool),
            auxiliary_depth_mm=depth,
            auxiliary_depth_frame_mask=depth_mask,
        )
        record = dict(
            packet=str(path),
            packet_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            source=source,
            source_lineage=provenance,
            parent_source_path=r["source_path"],
            split=r["split"],
            group=r["group"],
            pair=r.get("pair"),
            length=n,
            instruction=r["instruction"],
            geometry_available_frames=int(geometry.sum()),
            depth_role=depth_source,
            production_admission=False,
        )
        admitted.append(record)
        _atomic_json(
            out / "manifest.json",
            dict(
                records=admitted,
                stage="integration_pilot_not_formal_training",
                real_human_count=0,
                sim_rl_scratch_lineage_verified=False,
                multiview_available=False,
                reference_paths_available=False,
                geometry_origin="RGB student depth plus joint-FK camera pose; auxiliary simulator depth is not actor input",
                production_admission=False,
            ),
        )
        _atomic_json(
            out / "run_state.json",
            dict(
                status="complete" if len(admitted) == len(selected) else "materializing",
                completed=len(admitted),
                total=len(selected),
                pid=os.getpid(),
                production_admission=False,
            ),
        )
        print(
            json.dumps(
                dict(
                    completed=len(admitted),
                    total=len(selected),
                    source=source,
                    frames=n,
                    geometry_frames=int(geometry.sum()),
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
