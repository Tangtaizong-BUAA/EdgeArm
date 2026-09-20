"""Collect language-conditioned keyboard demonstrations in the V12 simulator.

The operator controls only XYZ from a localhost browser page.  The page owns
the keyboard focus, so WASD never reaches MuJoCo's viewer shortcuts.  Motion is
captured at control rate and successful traces are deterministically replayed
after collection to produce causal wrist RGB-D/segmentation HDF5 episodes.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

import mujoco
import numpy as np
from PIL import Image

from .keyboard_cartesian_runtime_v1 import (
    KEYBOARD_CARTESIAN_RUNTIME_VERSION,
    KeyboardCartesianConfig,
    NearestBlockFaceTracker,
    PositionFaceAlignedIK,
    slew_parallel_plane_normal,
    table_angle_face_normal,
)
from .keyboard_web_control_v1 import (
    KEYBOARD_WEB_CONTROL_VERSION,
    KeyboardWebControlServer,
)
from .phone_deferred_capture_v1 import (
    phone_step_strict_success,
    read_phone_motion_trace,
    replay_phone_motion_trace,
    write_phone_motion_trace,
)
from .phone_task_language_v1 import PhoneTaskMetadata
from .phone_teleop_runtime_v1 import (
    CartesianTarget,
    IKResult,
    MujocoPhoneBackend,
    MujocoSO101CartesianIK,
    PhoneControlCycle,
    PhoneSafetyDecision,
    StopReason,
)
from .sim2real_env_v12 import RealisticEdgeArmEnvV12, RealisticEnvV12Config


KEYBOARD_VLA_BATCH_VERSION = "edgearm-keyboard-vla-batch-v4-planner-seed-hold-sync"
KEYBOARD_CONTROL_SOURCE = "keyboard"
KEYBOARD_IK_RECOVERY_VERSION = (
    "edgearm-keyboard-ik-recovery-v3-plan-seeded-hold-synchronized"
)
KEYBOARD_IDLE_GUARD_VERSION = "edgearm-keyboard-idle-guard-v1-freeze-outside-target"
KEYBOARD_IDLE_SETTLE_GRACE_SECONDS = 0.75
_IK_BACKTRACK_SCALES = (1.0, 0.5, 0.25, 0.125)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Separate localhost WASD control + V12 motion capture + deferred wrist RGB-D replay"
        )
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--successes", type=int, default=50)
    parser.add_argument("--max-attempts", type=int, default=150)
    parser.add_argument("--steps-per-attempt", type=int, default=2_400)
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--xy-speed-m-s", type=float, default=0.036)
    parser.add_argument("--z-speed-m-s", type=float, default=0.024)
    parser.add_argument("--table-angle-speed-deg-s", type=float, default=30.0)
    parser.add_argument("--seed-start", type=int, default=10_000)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--view-width", type=int, default=640)
    parser.add_argument("--view-height", type=int, default=480)
    parser.add_argument("--view-fps", type=float, default=15.0)
    parser.add_argument("--rgbd-width", type=int, default=224)
    parser.add_argument("--rgbd-height", type=int, default=168)
    parser.add_argument("--minimum-free-gib", type=float, default=12.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument("--no-open", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("successes", "max_attempts", "steps_per_attempt"):
        value = getattr(args, name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.max_attempts < args.successes:
        raise ValueError("--max-attempts cannot be less than --successes")
    for name, high in (
        ("hz", 120.0),
        ("xy_speed_m_s", 0.15),
        ("z_speed_m_s", 0.10),
        ("view_fps", 30.0),
        ("table_angle_speed_deg_s", 90.0),
    ):
        value = float(getattr(args, name))
        if not np.isfinite(value) or value <= 0 or value > high:
            raise ValueError(f"--{name.replace('_', '-')} must be in (0,{high}]")
    for name in ("view_width", "view_height", "rgbd_width", "rgbd_height"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not 0 <= args.port <= 65535:
        raise ValueError("--port must be in [0,65535]")
    if not np.isfinite(args.minimum_free_gib) or args.minimum_free_gib < 1.0:
        raise ValueError("--minimum-free-gib must be finite and at least 1")


def _json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_json_text(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(_json_text(value) + "\n")
        stream.flush()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row in {path}")
            records.append(value)
    return records


def _batch_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": KEYBOARD_VLA_BATCH_VERSION,
        "control_source": KEYBOARD_CONTROL_SOURCE,
        "keyboard_runtime_version": KEYBOARD_CARTESIAN_RUNTIME_VERSION,
        "web_control_version": KEYBOARD_WEB_CONTROL_VERSION,
        "successes": args.successes,
        "max_attempts": args.max_attempts,
        "steps_per_attempt": args.steps_per_attempt,
        "hz": args.hz,
        "xy_speed_m_s": args.xy_speed_m_s,
        "z_speed_m_s": args.z_speed_m_s,
        "table_angle_speed_deg_s": args.table_angle_speed_deg_s,
        "idle_guard_version": KEYBOARD_IDLE_GUARD_VERSION,
        "idle_settle_grace_seconds": KEYBOARD_IDLE_SETTLE_GRACE_SECONDS,
        "face_normal_slew_rate_deg_s": float(
            np.degrees(KeyboardCartesianConfig().face_normal_slew_rate_rad_s)
        ),
        "maximum_ik_target_step_rad": KeyboardCartesianConfig().maximum_ik_target_step_rad,
        "minimum_table_angle_degrees": 65.0,
        "maximum_table_angle_degrees": 90.0,
        "seed_start": args.seed_start,
        "rgbd_width": args.rgbd_width,
        "rgbd_height": args.rgbd_height,
        "minimum_free_gib": args.minimum_free_gib,
        "task_scope": "single_movable_block_single_target_keyboard_vla_collection",
        "language_grounding_claimed": False,
        "physical_samples": 0,
    }


def _prepare_output(
    args: argparse.Namespace,
) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]]]:
    root = args.output_root.resolve()
    config = _batch_config(args)
    config_path = root / "batch_config.json"
    if root.exists():
        if not args.resume:
            raise FileExistsError(f"output root exists; use --resume: {root}")
        if not config_path.exists():
            raise ValueError("existing output root has no batch_config.json")
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("resume arguments differ from immutable batch config")
    else:
        root.mkdir(parents=True)
        (root / "motion").mkdir()
        (root / "rgbd").mkdir()
        _atomic_json(config_path, config)
    return root, _read_jsonl(root / "attempts.jsonl"), _read_jsonl(root / "replays.jsonl")


def _aligned_reset(
    env: RealisticEdgeArmEnvV12,
    ik: PositionFaceAlignedIK,
    tracker: NearestBlockFaceTracker,
    config: KeyboardCartesianConfig,
    seed: int,
) -> tuple[np.ndarray, str, np.ndarray]:
    env.reset(seed=seed, obstacle=False)
    tracker.reset()
    position, _normal = ik.current_pose()
    target = position.copy()
    target[2] = float(np.clip(0.080, config.minimum_z_m, config.maximum_z_m))
    face = tracker.select(env, target)
    result, selected_normal = ik.find_parallel_aligned_start(
        target,
        face.tool_face_normal_world,
    )
    if not result.converged:
        raise RuntimeError(
            "face-aligned reset failed: "
            f"position={result.position_error_m * 1000:.2f}mm "
            f"face={np.degrees(result.orientation_error_rad):.2f}deg "
            f"safety={result.safety_reason!r}"
        )
    env.data.qpos[:6] = result.target_joint_position_rad
    env.data.qvel[:6] = 0.0
    env.data.ctrl[:6] = result.target_joint_position_rad
    mujoco.mj_forward(env.model, env.data)
    return target, face.label, selected_normal


def _held_velocity(
    held: frozenset[str],
    *,
    xy_speed_m_s: float,
    z_speed_m_s: float,
) -> np.ndarray:
    planar = np.asarray(
        [
            float("w" in held) - float("s" in held),
            float("a" in held) - float("d" in held),
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(planar))
    if norm > 1.0:
        planar /= norm
    return np.asarray(
        [
            planar[0] * xy_speed_m_s,
            planar[1] * xy_speed_m_s,
            (float("ArrowUp" in held) - float("ArrowDown" in held))
            * z_speed_m_s,
        ],
        dtype=np.float64,
    )


def _held_table_angle_velocity_deg_s(
    held: frozenset[str],
    *,
    speed_deg_s: float,
) -> float:
    return (float("e" in held) - float("r" in held)) * float(speed_deg_s)


def _jpeg(renderer: mujoco.Renderer, env: RealisticEdgeArmEnvV12) -> bytes:
    renderer.update_scene(env.data, camera="edgearm_angled")
    image = renderer.render()
    output = BytesIO()
    Image.fromarray(image).save(output, format="JPEG", quality=84, optimize=False)
    return output.getvalue()


def _status(
    server: KeyboardWebControlServer,
    *,
    state: str,
    task: PhoneTaskMetadata,
    successes: int,
    requested: int,
    rows: int,
    coverage: float,
    message: str,
    table_angle_degrees: float = 90.0,
    rejected_keys: frozenset[str] = frozenset(),
) -> None:
    server.set_status(
        {
            "state": state,
            "task_zh": task.task_text_zh,
            "task_en": task.task_text_en,
            "progress": f"{successes}/{requested}",
            "rows": rows,
            "coverage": float(np.clip(coverage, 0.0, 1.0)),
            "gripper_table_angle_degrees": float(table_angle_degrees),
            "rejected_keys": sorted(rejected_keys),
            "message": message,
        }
    )


def _cycle(
    env: RealisticEdgeArmEnvV12,
    ik: PositionFaceAlignedIK,
    audit_ik: MujocoSO101CartesianIK,
    backend: MujocoPhoneBackend,
    tracker: NearestBlockFaceTracker,
    target_position: np.ndarray,
    face_label: str,
    active_normal: np.ndarray,
    table_angle_rad: float,
    requested_table_angle_rad: float,
    delta: np.ndarray,
    config: KeyboardCartesianConfig,
    *,
    control_period_s: float = 1.0 / 30.0,
    previous_target_q_rad: np.ndarray | None = None,
    diagnostics_out: dict[str, Any] | None = None,
) -> tuple[PhoneControlCycle | None, np.ndarray, str, np.ndarray, float]:
    if not np.isfinite(control_period_s) or control_period_s <= 0.0:
        raise ValueError("control_period_s must be finite and positive")
    raw_requested = target_position + delta
    clipped_requested = raw_requested.copy()
    clipped_requested[0] = np.clip(clipped_requested[0], *env.config.workspace_x)
    clipped_requested[1] = np.clip(clipped_requested[1], *env.config.workspace_y)
    clipped_requested[2] = np.clip(
        clipped_requested[2], config.minimum_z_m, config.maximum_z_m
    )
    requested_displacement = clipped_requested - target_position
    requested_angle_delta = requested_table_angle_rad - table_angle_rad
    current_q = np.asarray(env.observation()["joint_state"][:6], dtype=np.float64)
    previous_target_q = (
        None
        if previous_target_q_rad is None
        else np.asarray(previous_target_q_rad, dtype=np.float64)
    )
    if previous_target_q is not None and (
        previous_target_q.shape != (6,) or not np.isfinite(previous_target_q).all()
    ):
        raise ValueError("previous_target_q_rad must be finite [6]")
    # The force-limited plant intentionally lags behind its commanded target.
    # Starting IK from that lagged observation can cross to another valid IK
    # branch even though the operator requested only a millimetre-scale move.
    # Continue from the last accepted planner branch instead.  A released key
    # explicitly synchronizes this state to the stopped plant below.
    ik_seed_q = current_q if previous_target_q is None else previous_target_q.copy()
    diagnostics: dict[str, Any] = {
        "planner_seed": "observed_joint_state" if previous_target_q is None else "accepted_target",
        "attempted_scales": [],
        "nonconverged_trials": 0,
        "discontinuous_trials": 0,
        "maximum_branch_jump_rad": 0.0,
        "accepted_scale": None,
        "rejection_reason": "none",
    }
    original_selection = tracker.selection_state()
    no_requested_change = bool(
        np.allclose(requested_displacement, 0.0, atol=1.0e-12, rtol=0.0)
        and np.isclose(requested_angle_delta, 0.0, atol=1.0e-12, rtol=0.0)
    )
    scales = (1.0,) if no_requested_change else _IK_BACKTRACK_SCALES
    for scale in scales:
        # Face tracking is transactional: a failed candidate must not silently
        # change which block face the next operator request targets.
        tracker.restore_selection_state(original_selection)
        candidate_position = target_position + scale * requested_displacement
        candidate_angle = table_angle_rad + scale * requested_angle_delta
        try:
            face = tracker.select(env, candidate_position)
            requested_face_normal = table_angle_face_normal(
                face.tool_face_normal_world,
                candidate_angle,
            )
            slewed_face_normal = slew_parallel_plane_normal(
                active_normal,
                requested_face_normal,
                config.face_normal_slew_rate_rad_s * control_period_s,
            )
        except (RuntimeError, ValueError) as error:
            diagnostics["attempted_scales"].append(float(scale))
            diagnostics["rejection_reason"] = f"face_target:{type(error).__name__}"
            continue
        result, candidate_normal = ik.solve_parallel_face(
            ik_seed_q,
            candidate_position,
            slewed_face_normal,
            preferred_normal_world=active_normal,
        )
        diagnostics["attempted_scales"].append(float(scale))
        if not result.converged:
            diagnostics["nonconverged_trials"] += 1
            diagnostics["rejection_reason"] = "ik_not_converged"
            continue
        branch_jump = (
            0.0
            if previous_target_q is None
            else float(
                np.max(
                    np.abs(
                        result.target_joint_position_rad[:5]
                        - previous_target_q[:5]
                    )
                )
            )
        )
        diagnostics["maximum_branch_jump_rad"] = max(
            float(diagnostics["maximum_branch_jump_rad"]), branch_jump
        )
        if previous_target_q is not None and branch_jump > config.maximum_ik_target_step_rad:
            # A converged IK branch can still be discontinuous with the prior
            # command branch.  Never enqueue such a target; the next smaller
            # Cartesian trial or a released/reversed key can recover safely.
            diagnostics["discontinuous_trials"] += 1
            diagnostics["rejection_reason"] = "ik_branch_discontinuity"
            continue
        if scale < 1.0:
            result = replace(result, application_scale=scale)
        step = backend.step(result.target_joint_position_rad)
        transform = audit_ik.current_ee_transform(result.target_joint_position_rad)
        cartesian = CartesianTarget(
            transform_world_from_ee=transform,
            gripper_position_rad=float(env.tool_gripper_joint_position_rad),
            phone_translation_clipped=False,
            phone_rotation_clipped=False,
            workspace_clipped=not np.allclose(raw_requested, clipped_requested),
            rate_limited=True,
        )
        decision = PhoneSafetyDecision(
            active=True,
            stop_reason=StopReason.NONE,
            sample_age_ns=0,
            new_sample=True,
        )
        diagnostics["accepted_scale"] = float(scale)
        diagnostics["rejection_reason"] = "none"
        if diagnostics_out is not None:
            diagnostics_out.clear()
            diagnostics_out.update(diagnostics)
        return (
            PhoneControlCycle(None, decision, cartesian, result, step),
            candidate_position,
            face.label,
            candidate_normal,
            float(candidate_angle),
        )
    # Do not advance physics for an unaccepted operator request.  Restore all
    # accepted control state, create no row, and let the caller release only
    # the offending key(s) so a different next command can run immediately.
    tracker.restore_selection_state(original_selection)
    if diagnostics_out is not None:
        diagnostics_out.clear()
        diagnostics_out.update(diagnostics)
    return None, target_position, face_label, active_normal, table_angle_rad


def _literal_hold_cycle(
    env: RealisticEdgeArmEnvV12,
    audit_ik: MujocoSO101CartesianIK,
    backend: MujocoPhoneBackend,
) -> PhoneControlCycle:
    """Advance one explicit zero-action settle row when no key is held.

    This is not presented as a face-alignment IK success.  It is a separate
    joint-hold controller contract whose Cartesian error is zero by definition.
    Recording it is essential: the environment must advance for strict settle,
    and replay must see the exact same zero-action transition.
    """

    plant_q = np.asarray(env.data.qpos[:6], dtype=np.float64).copy()
    transform = audit_ik.current_ee_transform(plant_q)
    step = backend.step(None)
    hold = IKResult(
        target_joint_position_rad=step.target_q_rad,
        position_error_m=0.0,
        orientation_error_rad=0.0,
        converged=True,
        joint_or_workspace_clipped=False,
        safety_reason="literal_zero_no_key_settle",
    )
    cartesian = CartesianTarget(
        transform_world_from_ee=transform,
        gripper_position_rad=float(plant_q[5]),
        phone_translation_clipped=False,
        phone_rotation_clipped=False,
        workspace_clipped=False,
        rate_limited=False,
    )
    decision = PhoneSafetyDecision(
        active=True,
        stop_reason=StopReason.NONE,
        sample_age_ns=0,
        new_sample=True,
    )
    return PhoneControlCycle(None, decision, cartesian, hold, step)


def _synchronize_controller_after_hold(
    env: RealisticEdgeArmEnvV12,
    ik: PositionFaceAlignedIK,
    tracker: NearestBlockFaceTracker,
    cycle: PhoneControlCycle,
) -> tuple[np.ndarray, str, np.ndarray, np.ndarray]:
    """Commit the stopped plant as the next Cartesian planning origin.

    A literal zero-action row intentionally abandons the previous commanded
    target and holds the arm where it actually is.  All Cartesian and joint
    controller state must therefore move to ``q_after`` transactionally;
    otherwise the next key press compares a fresh IK result to a stale branch
    and can reject every subsequent direction.
    """

    q_after = np.asarray(cycle.step_result.q_after_rad, dtype=np.float64).copy()
    if q_after.shape != (6,) or not np.isfinite(q_after).all():
        raise ValueError("hold q_after_rad must be finite [6]")
    position, normal = ik.current_pose(q_after)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < 1.0e-8 or not np.isfinite(normal_norm):
        raise RuntimeError("hold synchronization produced an invalid tool normal")
    normal = normal / normal_norm
    face = tracker.select(env, position)
    return position, face.label, normal, q_after


def _replay_pending(
    args: argparse.Namespace,
    root: Path,
    attempts: list[dict[str, Any]],
    existing: list[dict[str, Any]],
    server: KeyboardWebControlServer,
    task: PhoneTaskMetadata,
) -> list[dict[str, Any]]:
    records = list(existing)
    done = {int(record["attempt_index"]) for record in records}
    eligible = [record for record in attempts if bool(record["strict_success"])]
    for index, attempt in enumerate(eligible, 1):
        attempt_index = int(attempt["attempt_index"])
        if attempt_index in done:
            continue
        free = shutil.disk_usage(root).free
        if free < int(args.minimum_free_gib * 1024**3):
            break
        _status(
            server,
            state="RGBD_REPLAY",
            task=task,
            successes=len(eligible),
            requested=args.successes,
            rows=0,
            coverage=1.0,
            message=f"正在生成腕部 RGB-D：{index}/{len(eligible)}，此阶段无需操作",
        )
        trace = read_phone_motion_trace(Path(attempt["motion_trace"]))
        output = root / "rgbd" / f"attempt_{attempt_index:04d}_seed_{attempt['seed']}.h5"
        result = replay_phone_motion_trace(
            trace,
            output,
            width=args.rgbd_width,
            height=args.rgbd_height,
        )
        record = {
            "schema_version": KEYBOARD_VLA_BATCH_VERSION,
            "attempt_index": attempt_index,
            "seed": int(attempt["seed"]),
            "rgbd_path": str(output.resolve()),
            "result": result,
            "completed_unix_ns": time.time_ns(),
        }
        _append_jsonl(root / "replays.jsonl", record)
        records.append(record)
        done.add(attempt_index)
    return records


def main() -> None:
    args = _args()
    _validate_args(args)
    root, attempts, replays = _prepare_output(args)
    successes = sum(bool(record["strict_success"]) for record in attempts)
    attempt_index = max((int(record["attempt_index"]) for record in attempts), default=0)
    server = KeyboardWebControlServer(port=args.port)
    server.start()
    print(f"KEYBOARD_CONTROL_URL={server.url}", flush=True)
    if not args.no_open:
        subprocess.Popen(
            ["open", server.url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    env = RealisticEdgeArmEnvV12(
        RealisticEnvV12Config(max_steps=args.steps_per_attempt),
        seed=args.seed_start,
    )
    env.reset(seed=args.seed_start, obstacle=False)
    config = KeyboardCartesianConfig()
    ik = PositionFaceAlignedIK(env, config)
    audit_ik = MujocoSO101CartesianIK(env)
    tracker = NearestBlockFaceTracker(config.face_switch_hysteresis_m)
    backend = MujocoPhoneBackend(env)
    renderer = mujoco.Renderer(
        env.model,
        height=args.view_height,
        width=args.view_width,
    )
    period_s = 1.0 / args.hz
    idle_settle_steps = max(
        1, int(np.ceil(KEYBOARD_IDLE_SETTLE_GRACE_SECONDS * args.hz))
    )
    render_period_s = 1.0 / args.view_fps
    last_render = 0.0
    stop_requested = False
    task = PhoneTaskMetadata.from_environment(env)

    try:
        while (
            successes < args.successes
            and attempt_index < args.max_attempts
            and not stop_requested
        ):
            attempt_index += 1
            seed = args.seed_start + attempt_index - 1
            _status(
                server,
                state="INITIALIZING",
                task=task,
                successes=successes,
                requested=args.successes,
                rows=0,
                coverage=0.0,
                message="正在计算自动腕部对齐起始姿态……",
            )
            target, face_label, active_normal = _aligned_reset(
                env,
                ik,
                tracker,
                config,
                seed,
            )
            table_angle_rad = np.pi / 2.0
            initial_arm_qpos = np.asarray(env.data.qpos[:6], dtype=np.float64).copy()
            initial_arm_qvel = np.asarray(env.data.qvel[:6], dtype=np.float64).copy()
            initial_arm_ctrl = np.asarray(env.data.ctrl[:6], dtype=np.float64).copy()
            previous_target_q = initial_arm_qpos.copy()
            task = PhoneTaskMetadata.from_environment(env)
            cycles: list[PhoneControlCycle] = []
            idle_cycles = 0
            strict_success = False
            termination_reason = "step_limit"
            coverage = 0.0
            print(
                f"EPISODE_READY=A{attempt_index:03d} seed={seed}\n"
                f"TASK_ZH={task.task_text_zh}\nTASK_EN={task.task_text_en}",
                flush=True,
            )

            while len(cycles) < args.steps_per_attempt:
                started = time.monotonic()
                snapshot = server.snapshot()
                if "quit" in snapshot.events:
                    termination_reason = "operator_quit"
                    stop_requested = True
                    break
                if "abort" in snapshot.events:
                    termination_reason = "operator_abort"
                    break
                if started - last_render >= render_period_s:
                    server.set_frame(_jpeg(renderer, env))
                    last_render = started
                if not snapshot.connected:
                    _status(
                        server,
                        state="PAUSED_DISCONNECTED",
                        task=task,
                        successes=successes,
                        requested=args.successes,
                        rows=len(cycles),
                        coverage=coverage,
                        message="控制页面未聚焦或心跳中断：仿真已冻结，不记录空帧",
                        table_angle_degrees=np.degrees(table_angle_rad),
                    )
                    time.sleep(min(period_s, 0.03))
                    continue
                if not cycles and not snapshot.held_keys:
                    _status(
                        server,
                        state="READY_WAIT_INPUT",
                        task=task,
                        successes=successes,
                        requested=args.successes,
                        rows=0,
                        coverage=coverage,
                        message="已连接；第一次按下移动键后才开始计时和记录",
                        table_angle_degrees=np.degrees(table_angle_rad),
                    )
                    time.sleep(min(period_s, 0.03))
                    continue
                if not snapshot.held_keys:
                    idle_cycles += 1
                    if coverage < 0.95 and idle_cycles > idle_settle_steps:
                        _status(
                            server,
                            state="PAUSED_IDLE",
                            task=task,
                            successes=successes,
                            requested=args.successes,
                            rows=len(cycles),
                            coverage=coverage,
                            message="已暂停且不写帧；按任意移动键继续，X 可重来本条",
                            table_angle_degrees=np.degrees(table_angle_rad),
                        )
                        time.sleep(min(period_s, 0.03))
                        continue
                    cycle = _literal_hold_cycle(env, audit_ik, backend)
                    target, face_label, active_normal, previous_target_q = (
                        _synchronize_controller_after_hold(env, ik, tracker, cycle)
                    )
                    status_state = "SETTLING_ZERO_HOLD"
                    message = "已松键：执行并记录零动作稳定帧"
                else:
                    idle_cycles = 0
                    velocity = _held_velocity(
                        snapshot.held_keys,
                        xy_speed_m_s=args.xy_speed_m_s,
                        z_speed_m_s=args.z_speed_m_s,
                    )
                    requested_table_angle_rad = float(
                        np.clip(
                            table_angle_rad
                            + np.deg2rad(
                                _held_table_angle_velocity_deg_s(
                                    snapshot.held_keys,
                                    speed_deg_s=args.table_angle_speed_deg_s,
                                )
                            )
                            * period_s,
                            config.minimum_table_angle_rad,
                            config.maximum_table_angle_rad,
                        )
                    )
                    diagnostics: dict[str, Any] = {}
                    cycle, target, face_label, active_normal, accepted_table_angle_rad = _cycle(
                        env,
                        ik,
                        audit_ik,
                        backend,
                        tracker,
                        target,
                        face_label,
                        active_normal,
                        table_angle_rad,
                        requested_table_angle_rad,
                        velocity * period_s,
                        config,
                        control_period_s=period_s,
                        previous_target_q_rad=previous_target_q,
                        diagnostics_out=diagnostics,
                    )
                    if cycle is None:
                        server.reject_keys_until_release(snapshot.held_keys)
                        _status(
                            server,
                            state="IK_BLOCKED",
                            task=task,
                            successes=successes,
                            requested=args.successes,
                            rows=len(cycles),
                            coverage=coverage,
                            message=(
                                "本次指令已安全取消且未写入轨迹："
                                f"{diagnostics.get('rejection_reason', 'unknown')}；"
                                "请松开该键后重按"
                            ),
                            table_angle_degrees=np.degrees(table_angle_rad),
                            rejected_keys=snapshot.held_keys,
                        )
                        remaining = period_s - (time.monotonic() - started)
                        if remaining > 0:
                            time.sleep(remaining)
                        continue
                    assert cycle.ik_result is not None
                    previous_target_q = cycle.ik_result.target_joint_position_rad.copy()
                    table_angle_rad = accepted_table_angle_rad
                    status_state = "RECORDING"
                    scale = cycle.ik_result.application_scale
                    message = (
                        f"边界已自动缩步至 {scale * 100:.1f}%，可继续操作"
                        if scale < 1.0
                        else "WASD/方向键移动；E/R 调整夹爪与桌面角度"
                    )
                cycle.step_result.info["keyboard_control_v4"] = {
                    "held_keys": sorted(snapshot.held_keys),
                    "face_label": face_label,
                    "control_state": status_state,
                    "idle_cycle_count": idle_cycles,
                }
                cycles.append(cycle)
                realism = cycle.step_result.info.get("realism_v6", {})
                coverage = float(realism.get("strict_target_coverage", 0.0))
                if coverage >= 0.95:
                    idle_cycles = 0
                strict_success = phone_step_strict_success(dict(cycle.step_result.info))
                _status(
                    server,
                    state=status_state,
                    task=task,
                    successes=successes,
                    requested=args.successes,
                    rows=len(cycles),
                    coverage=coverage,
                    message=message,
                    table_angle_degrees=np.degrees(table_angle_rad),
                )
                if strict_success:
                    termination_reason = "strict_success"
                    break
                if cycle.step_result.terminated:
                    termination_reason = "environment_terminated"
                    break
                if cycle.step_result.truncated:
                    termination_reason = "environment_truncated"
                    break
                remaining = period_s - (time.monotonic() - started)
                if remaining > 0:
                    time.sleep(remaining)

            if cycles:
                motion_path = (
                    root
                    / "motion"
                    / f"attempt_{attempt_index:04d}_seed_{seed}.motion.h5"
                )
                written = write_phone_motion_trace(
                    motion_path,
                    cycles,
                    episode_seed=seed,
                    environment_profile=env.profile_version,
                    episode_success=strict_success,
                    termination_reason=termination_reason,
                    task_metadata=task,
                    control_source=KEYBOARD_CONTROL_SOURCE,
                    control_runtime_version=KEYBOARD_CARTESIAN_RUNTIME_VERSION,
                    initial_arm_pose_overridden=True,
                    initial_arm_qpos_rad=initial_arm_qpos,
                    initial_arm_qvel_rad_s=initial_arm_qvel,
                    initial_arm_ctrl_rad=initial_arm_ctrl,
                )
                record = {
                    "schema_version": KEYBOARD_VLA_BATCH_VERSION,
                    "attempt_index": attempt_index,
                    "seed": seed,
                    "strict_success": strict_success,
                    "termination_reason": termination_reason,
                    "rows": len(cycles),
                    "maximum_strict_target_coverage": max(
                        (
                            float(
                                cycle.step_result.info.get("realism_v6", {}).get(
                                    "strict_target_coverage", 0.0
                                )
                            )
                            for cycle in cycles
                        ),
                        default=0.0,
                    ),
                    "task": task.contract(),
                    "episode_domain": env.episode_domain,
                    "operator_input_source": KEYBOARD_CONTROL_SOURCE,
                    "ik_recovery_version": KEYBOARD_IK_RECOVERY_VERSION,
                    "motion_trace": str(written),
                    "created_unix_ns": time.time_ns(),
                }
                _append_jsonl(root / "attempts.jsonl", record)
                attempts.append(record)
                if strict_success:
                    successes += 1
                print(
                    f"ATTEMPT_DONE={attempt_index} strict_success={strict_success} "
                    f"progress={successes}/{args.successes} rows={len(cycles)}",
                    flush=True,
                )
            elif not stop_requested:
                attempt_index -= 1

        renderer.close()
        if not args.capture_only:
            replays = _replay_pending(args, root, attempts, replays, server, task)
        successful_attempts = {
            int(record["attempt_index"])
            for record in attempts
            if bool(record["strict_success"])
        }
        replayed_attempts = {int(record["attempt_index"]) for record in replays}
        summary = {
            "schema_version": KEYBOARD_VLA_BATCH_VERSION,
            "attempts": len(attempts),
            "strict_successes": successes,
            "requested_strict_successes": args.successes,
            "capture_complete": successes >= args.successes,
            "rgbd_replays_complete": successful_attempts <= replayed_attempts,
            "rgbd_replays": len(replayed_attempts),
            "stop_requested": stop_requested,
            "free_gib": shutil.disk_usage(root).free / 1024**3,
            "updated_unix_ns": time.time_ns(),
        }
        _atomic_json(root / "batch_summary.json", summary)
        _status(
            server,
            state="COMPLETE",
            task=task,
            successes=successes,
            requested=args.successes,
            rows=0,
            coverage=1.0 if successes else 0.0,
            message=(
                "采集与腕部 RGB-D 回放已完成，可以关闭页面"
                if summary["rgbd_replays_complete"]
                else "动作采集已停止；部分 RGB-D 因磁盘门限尚未生成"
            ),
        )
        print(_json_text(summary), flush=True)
        time.sleep(2.0)
    finally:
        try:
            renderer.close()
        except Exception:
            pass
        server.stop()


if __name__ == "__main__":
    main()


__all__ = ["KEYBOARD_VLA_BATCH_VERSION", "main"]
