"""Recover perception labels from already collected Run94 training commands.

No teacher or learned policy runs here. Replayed reports, selected-object XY,
and every sampled RGB frame must match the original before labels are admitted.
Geometry and segmentation are labels only, in a file separate from inputs.
"""
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from .candidate_command_contract_v2 import ACTION_CONTRACT, execution_command
from .materialize_temporal_packets_v3 import kinematics
from .run42.domain import sample_domain
from .run42.session import DomainSession
from .run82_prepare_sequences import geometry_ids, sparse_geometry
from .sparse_4d_vla_act_v26 import Sparse4DVLAConfigV26
from .train_staged_hybrid_contact_sac import _atomic_json


def training_split(seed):
    group = int(seed) // 9
    if not 100500000 <= group < 100500012:
        raise ValueError("only original Run94 collection, never development/independent scenes")
    # These groups trained the action model already: perception validation only.
    return "train" if group < 100500010 else "validation"


def check_reproduction(report_error, object_error, rgb_max, rgb_mean):
    if report_error > 1e-6 or object_error > 1e-8 or rgb_max > 2 or rgb_mean > .01:
        raise ValueError("replay does not reproduce source state and RGB")


def prepare(job):
    source, output = map(Path, job)
    result = json.loads((source / "result.json").read_text())
    seed = result["seed"]
    split = training_split(seed)
    trace_path = source / "trace.npz"
    with np.load(trace_path, allow_pickle=False) as z:
        reports, commands = z["reported"].copy(), z["command"].copy()
        frames, audit = z["wrist_rgb"].copy(), z["audit"].copy()
    n = len(commands)
    if reports.shape != (n, 12) or audit.shape != (n, 17) or len(frames) != (n + 7) // 8:
        raise ValueError("complete source command/report/image timeline required")
    if not np.array_equal(audit[:, 0], np.arange(n)):
        raise ValueError("contiguous original control steps required")
    session = DomainSession(
        Sparse4DVLAConfigV26(language_max_tokens=128, visual_memory_mode="episode_anchors_v54"),
        ACTION_CONTRACT, seed, sample_domain(seed + 6001, 0),
    )
    geometries = geometry_ids(session)
    observed = {k: [] for k in ("rgb", "pose", "time_step")}
    labels = {k: [] for k in ("xyz", "present", "visible")}
    report_max = object_max = rgb_mean_max = 0.0
    rgb_max = 0
    try:
        for step, action in enumerate(commands):
            report = np.asarray(session.env.observation()["joint_state"], np.float32)
            report_max = max(report_max, float(np.max(np.abs(report - reports[step]))))
            object_max = max(object_max, float(np.max(np.abs(session.env.block_xy() - audit[step, 5:7]))))
            check_reproduction(report_max, object_max, rgb_max, rgb_mean_max)
            if step % 8 == 0:
                session.renderer.disable_segmentation_rendering()
                session.renderer.update_scene(session.env.data, camera="edgearm_wrist")
                rgb = np.asarray(Image.fromarray(session.renderer.render()).resize((160, 120)))
                difference = np.abs(rgb.astype(np.int16) - frames[step // 8].astype(np.int16))
                rgb_max = max(rgb_max, int(difference.max()))
                rgb_mean_max = max(rgb_mean_max, float(difference.mean()))
                check_reproduction(report_max, object_max, rgb_max, rgb_mean_max)
                _, pose = kinematics(session.env, report[:6], report[6:])
                observed["rgb"].append(frames[step // 8])
                observed["pose"].append(pose)
                observed["time_step"].append(step)
                centers, _, exists = sparse_geometry(session.env.model, session.env.data, geometries)
                session.renderer.enable_segmentation_rendering()
                session.renderer.update_scene(session.env.data, camera="edgearm_wrist")
                segmentation = session.renderer.render()
                pixels = np.asarray(Image.fromarray(segmentation[..., 0].astype(np.int32)).resize(
                    (160, 120), Image.Resampling.NEAREST))
                kinds = np.asarray(Image.fromarray(segmentation[..., 1].astype(np.int32)).resize(
                    (160, 120), Image.Resampling.NEAREST))
                counts = np.zeros(7, np.int64)
                for geom, color in geometries:
                    counts[color] = np.count_nonzero((pixels == geom) & (kinds == mujoco.mjtObj.mjOBJ_GEOM))
                labels["xyz"].append(centers)
                labels["present"].append(exists)
                labels["visible"].append(counts >= 8)
            # The recorded command, not a label or a new teacher, advances physics.
            session.env.step(execution_command(action, ACTION_CONTRACT, session.env.config.max_joint_delta))
        destination = output / f"episode_{seed}"
        destination.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(destination / "inputs.npz", **{k: np.asarray(v) for k, v in observed.items()},
                            K=session.buffer.K)
        np.savez_compressed(destination / "labels.npz", **{k: np.asarray(v) for k, v in labels.items()})
        row = dict(seed=seed, route=seed % 9, split=split, folder=str(destination), source=str(source),
                   source_trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                   frames=len(observed["rgb"]), steps=n, report_max_error=report_max,
                   object_max_error_m=object_max, rgb_max_error=rgb_max, rgb_mean_max_error=rgb_mean_max,
                   original_safe_success=result["safe_success"], replay_not_new_collection=True,
                   failed_episodes_valid_for_perception=True, actor_received_labels=False,
                   validation_is_not_independent=True)
        _atomic_json(destination / "replay_audit.json", row)
        return row
    finally:
        session.close()
