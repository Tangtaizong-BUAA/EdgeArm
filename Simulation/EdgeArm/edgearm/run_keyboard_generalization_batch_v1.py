"""Large, resumable V13 keyboard collection with deferred wrist RGB-D replay."""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import mujoco
import numpy as np
from PIL import Image

from .external_episode_store_v1 import ExternalEpisodeStoreV1
from .generalization_task_language_v1 import GeneralizationTaskMetadataV1
from .keyboard_cartesian_runtime_v1 import (
    KEYBOARD_CARTESIAN_RUNTIME_VERSION,
    KeyboardCartesianConfig,
    NearestBlockFaceTracker,
    PositionFaceAlignedIK,
)
from .keyboard_web_control_v1 import KEYBOARD_WEB_CONTROL_VERSION, KeyboardWebControlServer
from .phone_deferred_capture_v1 import (
    phone_step_strict_success,
    read_phone_motion_trace,
    replay_phone_motion_trace,
    write_phone_motion_trace,
)
from .phone_teleop_runtime_v1 import MujocoPhoneBackend, MujocoSO101CartesianIK
from .run_keyboard_vla_batch_v1 import (
    KEYBOARD_CONTROL_SOURCE,
    KEYBOARD_IDLE_GUARD_VERSION,
    KEYBOARD_IDLE_SETTLE_GRACE_SECONDS,
    KEYBOARD_IK_RECOVERY_VERSION,
    _cycle,
    _held_table_angle_velocity_deg_s,
    _held_velocity,
    _literal_hold_cycle,
    _synchronize_controller_after_hold,
)
from .sim2real_env_v13 import (
    GENERALIZATION_DYNAMICS_PROFILE_V13,
    GENERALIZATION_KEYBOARD_TELEOP_TRANSPORT_V13,
    GENERALIZATION_REQUIRED_STABLE_SECONDS_V13,
    RealisticEdgeArmEnvV13,
    keyboard_teleop_env_config_v13,
)


KEYBOARD_GENERALIZATION_BATCH_VERSION = (
    "edgearm-keyboard-generalization-batch-v3-stable-human-teleop"
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--successes", type=int, default=500)
    parser.add_argument("--max-attempts", type=int, default=1_500)
    parser.add_argument("--steps-per-attempt", type=int, default=3_000)
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--xy-speed-m-s", type=float, default=0.036)
    parser.add_argument("--z-speed-m-s", type=float, default=0.024)
    parser.add_argument("--table-angle-speed-deg-s", type=float, default=30.0)
    parser.add_argument("--seed-start", type=int, default=30_000)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--view-width", type=int, default=640)
    parser.add_argument("--view-height", type=int, default=480)
    parser.add_argument("--view-fps", type=float, default=15.0)
    parser.add_argument("--rgbd-width", type=int, default=224)
    parser.add_argument("--rgbd-height", type=int, default=168)
    parser.add_argument("--minimum-free-gib", type=float, default=200.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument("--replay-only", action="store_true")
    parser.add_argument("--verify-existing-hashes", action="store_true")
    parser.add_argument(
        "--replay-state-mode",
        choices=("strict_exact", "command_reexecution"),
        default="strict_exact",
    )
    parser.add_argument("--no-open", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("successes", "max_attempts", "steps_per_attempt"):
        if type(getattr(args, name)) is not int or getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be a positive integer")
    if args.max_attempts < args.successes:
        raise ValueError("--max-attempts cannot be less than --successes")
    if args.capture_only and args.replay_only:
        raise ValueError("--capture-only and --replay-only are mutually exclusive")
    for name, upper in (
        ("hz", 120.0),
        ("xy_speed_m_s", 0.15),
        ("z_speed_m_s", 0.10),
        ("table_angle_speed_deg_s", 90.0),
        ("view_fps", 30.0),
    ):
        value = float(getattr(args, name))
        if not np.isfinite(value) or not 0.0 < value <= upper:
            raise ValueError(f"--{name.replace('_', '-')} must be in (0,{upper}]")
    if not 0 <= args.port <= 65535:
        raise ValueError("--port must be in [0,65535]")
    if args.minimum_free_gib < 1.0:
        raise ValueError("--minimum-free-gib must be at least 1")


def _json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(_json_text(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(_json_text(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quarantine_failed_replay(
    store: ExternalEpisodeStoreV1,
    staging: Path,
    final: Path,
    *,
    attempt_index: int,
    seed: int,
    result: dict[str, Any],
) -> Path:
    if not staging.is_file():
        raise FileNotFoundError(staging)
    mode = str(result.get("sensor_materialization_mode", "unknown"))
    safe_mode = "".join(character if character.isalnum() else "_" for character in mode)
    quarantine = final.with_name(
        f"{final.stem}.failed_{safe_mode}_not_admitted.quarantine{final.suffix}"
    )
    if quarantine.exists():
        raise FileExistsError(quarantine)
    size = staging.stat().st_size
    digest = _sha256_file(staging)
    os.replace(staging, quarantine)
    _append_jsonl(
        store.root / "replay_failures.jsonl",
        {
            "schema_version": KEYBOARD_GENERALIZATION_BATCH_VERSION,
            "attempt_index": attempt_index,
            "seed": seed,
            "quarantine_path": str(quarantine),
            "bytes": size,
            "sha256": digest,
            "result": result,
            "failure": "rgbd_replay_not_admitted",
            "completed_unix_ns": time.time_ns(),
        },
    )
    return quarantine


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"non-object JSONL row at {path}:{line_number}")
        rows.append(value)
    return rows


def _immutable_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": KEYBOARD_GENERALIZATION_BATCH_VERSION,
        "environment_profile": GENERALIZATION_DYNAMICS_PROFILE_V13,
        "human_teleop_transport_profile": GENERALIZATION_KEYBOARD_TELEOP_TRANSPORT_V13,
        "control_source": KEYBOARD_CONTROL_SOURCE,
        "keyboard_runtime_version": KEYBOARD_CARTESIAN_RUNTIME_VERSION,
        "web_control_version": KEYBOARD_WEB_CONTROL_VERSION,
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
        "seed_start": args.seed_start,
        "rgbd_width": args.rgbd_width,
        "rgbd_height": args.rgbd_height,
        "minimum_free_gib": args.minimum_free_gib,
        "required_continuous_stable_seconds": GENERALIZATION_REQUIRED_STABLE_SECONDS_V13,
        "task_scope": "stratified_shape_size_friction_lighting_multi_obstacle_keyboard_vla",
        "physical_samples": 0,
        "physically_calibrated": False,
    }


def _prepare(
    args: argparse.Namespace,
    store: ExternalEpisodeStoreV1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    store.initialize(resume=args.resume)
    commits = store.verify_commits(verify_hashes=args.verify_existing_hashes)
    config = _immutable_config(args)
    config_path = store.root / "dataset_contract.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("resume arguments differ from the immutable V13 dataset contract")
    else:
        _atomic_json(config_path, config)
    attempts = _read_jsonl(store.root / "attempts.jsonl")
    replays = _read_jsonl(store.root / "replays.jsonl")
    motion_commits = {item.attempt_index for item in commits if item.kind == "keyboard_motion_trace"}
    rgbd_commits = {item.attempt_index for item in commits if item.kind == "wrist_rgbd_episode"}
    attempt_indices = {int(item["attempt_index"]) for item in attempts}
    replay_indices = {int(item["attempt_index"]) for item in replays}
    if motion_commits != attempt_indices:
        raise RuntimeError(
            "motion commit ledger and attempts manifest differ; inspect before resume: "
            f"committed_only={sorted(motion_commits - attempt_indices)} "
            f"manifest_only={sorted(attempt_indices - motion_commits)}"
        )
    if rgbd_commits != replay_indices:
        raise RuntimeError(
            "RGB-D commit ledger and replay manifest differ; inspect before resume: "
            f"committed_only={sorted(rgbd_commits - replay_indices)} "
            f"manifest_only={sorted(replay_indices - rgbd_commits)}"
        )
    successes = sum(bool(item.get("strict_success_3s", False)) for item in attempts)
    if args.successes < successes:
        raise ValueError("requested success goal cannot be below already committed successes")
    goal = {
        "requested_successes": args.successes,
        "maximum_attempts": args.max_attempts,
        "updated_unix_ns": time.time_ns(),
    }
    _atomic_json(store.root / "collection_goal.json", goal)
    _append_jsonl(store.root / "goal_history.jsonl", goal)
    return attempts, replays


def _jpeg(renderer: mujoco.Renderer, env: RealisticEdgeArmEnvV13) -> bytes:
    renderer.update_scene(env.data, camera="edgearm_angled")
    image = renderer.render()
    output = BytesIO()
    Image.fromarray(image).save(output, format="JPEG", quality=84, optimize=False)
    return output.getvalue()


def _status(
    server: KeyboardWebControlServer,
    *,
    state: str,
    task: GeneralizationTaskMetadataV1,
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


def _aligned_reset(
    env: RealisticEdgeArmEnvV13,
    ik: PositionFaceAlignedIK,
    tracker: NearestBlockFaceTracker,
    config: KeyboardCartesianConfig,
    *,
    seed: int,
    schedule_index: int,
) -> tuple[np.ndarray, str, np.ndarray]:
    env.reset(seed=seed, schedule_index=schedule_index)
    tracker.reset()
    position, _normal = ik.current_pose()
    target = position.copy()
    target[2] = float(np.clip(0.080, config.minimum_z_m, config.maximum_z_m))
    face = tracker.select(env, target)
    result, selected_normal = ik.find_parallel_aligned_start(target, face.tool_face_normal_world)
    if not result.converged:
        raise RuntimeError(
            "V13 face-aligned reset failed: "
            f"position={result.position_error_m * 1000:.2f}mm "
            f"face={np.degrees(result.orientation_error_rad):.2f}deg "
            f"safety={result.safety_reason!r}"
        )
    env.data.qpos[:6] = result.target_joint_position_rad
    env.data.qvel[:6] = 0.0
    env.data.ctrl[:6] = result.target_joint_position_rad
    mujoco.mj_forward(env.model, env.data)
    return target, face.label, selected_normal


def _replay_pending(
    args: argparse.Namespace,
    store: ExternalEpisodeStoreV1,
    attempts: list[dict[str, Any]],
    existing: list[dict[str, Any]],
    server: KeyboardWebControlServer,
    task: GeneralizationTaskMetadataV1,
) -> list[dict[str, Any]]:
    records = list(existing)
    done = {int(record["attempt_index"]) for record in records}
    eligible = [record for record in attempts if bool(record.get("strict_success_3s", False))]
    for ordinal, attempt in enumerate(eligible, 1):
        attempt_index = int(attempt["attempt_index"])
        if attempt_index in done:
            continue
        store.assert_capacity()
        _status(
            server,
            state="RGBD_REPLAY",
            task=task,
            successes=len(eligible),
            requested=args.successes,
            rows=0,
            coverage=1.0,
            message=f"正在生成腕部 RGB-D：{ordinal}/{len(eligible)}；此阶段无需操作",
        )
        trace = read_phone_motion_trace(Path(attempt["motion_trace"]))
        final = store.root / "rgbd" / f"attempt_{attempt_index:05d}_seed_{attempt['seed']}.h5"
        staging = store.stage_path(final)
        result = replay_phone_motion_trace(
            trace,
            staging,
            width=args.rgbd_width,
            height=args.rgbd_height,
            state_match_mode=args.replay_state_mode,
        )
        if result["vla_eligible"] is not True:
            quarantine = _quarantine_failed_replay(
                store,
                staging,
                final,
                attempt_index=attempt_index,
                seed=int(attempt["seed"]),
                result=result,
            )
            raise RuntimeError(
                "RGB-D replay did not reproduce an admitted success: "
                f"attempt={attempt_index}, quarantined={quarantine}"
            )
        commit = store.commit(
            staging,
            final,
            kind="wrist_rgbd_episode",
            attempt_index=attempt_index,
            episode_seed=int(attempt["seed"]),
        )
        result["output"] = str(final)
        record = {
            "schema_version": KEYBOARD_GENERALIZATION_BATCH_VERSION,
            "attempt_index": attempt_index,
            "seed": int(attempt["seed"]),
            "rgbd_path": str(final),
            "file_commit": commit.contract(),
            "result": result,
            "completed_unix_ns": time.time_ns(),
        }
        _append_jsonl(store.root / "replays.jsonl", record)
        records.append(record)
        done.add(attempt_index)
    return records


def main() -> None:
    args = _args()
    _validate_args(args)
    store = ExternalEpisodeStoreV1(
        args.output_root,
        minimum_free_gib=args.minimum_free_gib,
    )
    attempts, replays = _prepare(args, store)
    successes = sum(bool(record.get("strict_success_3s", False)) for record in attempts)
    attempt_index = max((int(record["attempt_index"]) for record in attempts), default=0)

    server = KeyboardWebControlServer(port=args.port)
    server.start()
    print(f"KEYBOARD_CONTROL_URL={server.url}", flush=True)
    if not args.no_open:
        subprocess.Popen(["open", server.url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    env = RealisticEdgeArmEnvV13(
        keyboard_teleop_env_config_v13(max_steps=args.steps_per_attempt),
        seed=args.seed_start,
    )
    env.reset(seed=args.seed_start, schedule_index=0)
    config = KeyboardCartesianConfig()
    ik = PositionFaceAlignedIK(env, config)
    audit_ik = MujocoSO101CartesianIK(env)
    tracker = NearestBlockFaceTracker(config.face_switch_hysteresis_m)
    backend = MujocoPhoneBackend(env)
    renderer = mujoco.Renderer(env.model, height=args.view_height, width=args.view_width)
    task = GeneralizationTaskMetadataV1.from_scenario(env.current_scenario)
    period_s = 1.0 / args.hz
    idle_settle_steps = max(
        1, int(np.ceil(KEYBOARD_IDLE_SETTLE_GRACE_SECONDS * args.hz))
    )
    render_period_s = 1.0 / args.view_fps
    last_render = 0.0
    stop_requested = False

    try:
        if not args.replay_only:
            while successes < args.successes and attempt_index < args.max_attempts and not stop_requested:
                attempt_index += 1
                seed = args.seed_start + attempt_index - 1
                store.assert_capacity()
                _status(
                    server,
                    state="INITIALIZING",
                    task=task,
                    successes=successes,
                    requested=args.successes,
                    rows=0,
                    coverage=0.0,
                    message="正在构建随机场景并计算腕部对齐……",
                )
                target, face_label, active_normal = _aligned_reset(
                    env,
                    ik,
                    tracker,
                    config,
                    seed=seed,
                    schedule_index=attempt_index - 1,
                )
                scenario = env.current_scenario
                assert scenario is not None
                task = GeneralizationTaskMetadataV1.from_scenario(scenario)
                table_angle_rad = np.pi / 2.0
                initial_arm_qpos = np.asarray(env.data.qpos[:6], dtype=np.float64).copy()
                initial_arm_qvel = np.asarray(env.data.qvel[:6], dtype=np.float64).copy()
                initial_arm_ctrl = np.asarray(env.data.ctrl[:6], dtype=np.float64).copy()
                previous_target_q = initial_arm_qpos.copy()
                cycles = []
                idle_cycles = 0
                strict_success = False
                termination_reason = "step_limit"
                coverage = 0.0
                maximum_stable_seconds = 0.0
                print(
                    f"EPISODE_READY=A{attempt_index:05d} seed={seed} "
                    f"shape={scenario.object_shape} size={scenario.object_size_tier} "
                    f"friction={scenario.friction_tier} light={scenario.lighting_tier} "
                    f"layout={scenario.obstacle_layout}\nTASK_ZH={task.task_text_zh}\n"
                    f"TASK_EN={task.task_text_en}",
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
                            message="控制页面未连接：仿真冻结且不写空帧",
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
                            message=(
                                f"已连接；{scenario.object_shape}/{scenario.object_size_tier}，"
                                f"{scenario.obstacle_layout}；首次按键后开始记录"
                            ),
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
                        control_state = "SETTLING_ZERO_HOLD"
                    else:
                        idle_cycles = 0
                        velocity = _held_velocity(
                            snapshot.held_keys,
                            xy_speed_m_s=args.xy_speed_m_s,
                            z_speed_m_s=args.z_speed_m_s,
                        )
                        requested_angle = float(
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
                        cycle, target, face_label, active_normal, accepted_angle = _cycle(
                            env,
                            ik,
                            audit_ik,
                            backend,
                            tracker,
                            target,
                            face_label,
                            active_normal,
                            table_angle_rad,
                            requested_angle,
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
                                    "本次指令已安全取消且不写帧："
                                    f"{diagnostics.get('rejection_reason', 'unknown')}；"
                                    "请松开后重按"
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
                        table_angle_rad = accepted_angle
                        control_state = "RECORDING"
                    cycle.step_result.info["keyboard_control_v4"] = {
                        "held_keys": sorted(snapshot.held_keys),
                        "face_label": face_label,
                        "control_state": control_state,
                        "idle_cycle_count": idle_cycles,
                    }
                    cycles.append(cycle)
                    realism_v6 = cycle.step_result.info.get("realism_v6", {})
                    realism_v13 = cycle.step_result.info.get("realism_v13", {})
                    coverage = float(realism_v6.get("strict_target_coverage", 0.0))
                    if coverage >= 0.95:
                        idle_cycles = 0
                    stable_seconds = float(realism_v13.get("continuous_stable_seconds", 0.0))
                    maximum_stable_seconds = max(maximum_stable_seconds, stable_seconds)
                    strict_success = phone_step_strict_success(dict(cycle.step_result.info))
                    if not bool(realism_v13.get("strict_no_obstacle_contact", True)):
                        termination_reason = "forbidden_obstacle_contact"
                        _status(
                            server,
                            state="ATTEMPT_FAILED_CONTACT",
                            task=task,
                            successes=successes,
                            requested=args.successes,
                            rows=len(cycles),
                            coverage=coverage,
                            message="本条已碰到障碍物，严格门判定失败，正在保存失败轨迹",
                            table_angle_degrees=np.degrees(table_angle_rad),
                        )
                        break
                    if stable_seconds > 0.0:
                        state = "STABLE_HOLD_3S"
                        message = (
                            f"保持松键稳定：{stable_seconds:.2f}/"
                            f"{GENERALIZATION_REQUIRED_STABLE_SECONDS_V13:.2f} 秒"
                        )
                    else:
                        state = "RECORDING" if snapshot.held_keys else "SETTLING_ZERO_HOLD"
                        message = "WASD/方向键移动；E/R 调角；进入目标后松键保持 3 秒"
                    _status(
                        server,
                        state=state,
                        task=task,
                        successes=successes,
                        requested=args.successes,
                        rows=len(cycles),
                        coverage=coverage,
                        message=message,
                        table_angle_degrees=np.degrees(table_angle_rad),
                    )
                    if strict_success:
                        termination_reason = "strict_success_stable_3s"
                        break
                    if cycle.step_result.truncated:
                        termination_reason = "environment_truncated"
                        break
                    remaining = period_s - (time.monotonic() - started)
                    if remaining > 0:
                        time.sleep(remaining)

                if cycles:
                    final_motion = (
                        store.root / "motion" / f"attempt_{attempt_index:05d}_seed_{seed}.motion.h5"
                    )
                    staging_motion = store.stage_path(final_motion)
                    write_phone_motion_trace(
                        staging_motion,
                        cycles,
                        episode_seed=seed,
                        environment_profile=env.profile_version,
                        episode_success=strict_success,
                        termination_reason=termination_reason,
                        task_metadata=task,
                        scene_contract=scenario.contract(),
                        control_source=KEYBOARD_CONTROL_SOURCE,
                        control_runtime_version=KEYBOARD_CARTESIAN_RUNTIME_VERSION,
                        initial_arm_pose_overridden=True,
                        initial_arm_qpos_rad=initial_arm_qpos,
                        initial_arm_qvel_rad_s=initial_arm_qvel,
                        initial_arm_ctrl_rad=initial_arm_ctrl,
                    )
                    commit = store.commit(
                        staging_motion,
                        final_motion,
                        kind="keyboard_motion_trace",
                        attempt_index=attempt_index,
                        episode_seed=seed,
                    )
                    record = {
                        "schema_version": KEYBOARD_GENERALIZATION_BATCH_VERSION,
                        "attempt_index": attempt_index,
                        "seed": seed,
                        "strict_success_3s": strict_success,
                        "termination_reason": termination_reason,
                        "rows": len(cycles),
                        "maximum_strict_target_coverage": max(
                            float(
                                item.step_result.info.get("realism_v6", {}).get("strict_target_coverage", 0.0)
                            )
                            for item in cycles
                        ),
                        "maximum_continuous_stable_seconds": maximum_stable_seconds,
                        "task": task.contract(),
                        "scene_contract": scenario.contract(),
                        "scene_contract_sha256": scenario.contract_sha256,
                        "episode_domain": env.episode_domain,
                        "operator_input_source": KEYBOARD_CONTROL_SOURCE,
                        "ik_recovery_version": KEYBOARD_IK_RECOVERY_VERSION,
                        "motion_trace": str(final_motion),
                        "file_commit": commit.contract(),
                        "created_unix_ns": time.time_ns(),
                    }
                    _append_jsonl(store.root / "attempts.jsonl", record)
                    attempts.append(record)
                    if strict_success:
                        successes += 1
                    print(
                        f"ATTEMPT_DONE={attempt_index} strict_success_3s={strict_success} "
                        f"progress={successes}/{args.successes} rows={len(cycles)}",
                        flush=True,
                    )
                elif not stop_requested:
                    attempt_index -= 1

        if not args.capture_only:
            replays = _replay_pending(args, store, attempts, replays, server, task)
        successful_attempts = {
            int(record["attempt_index"])
            for record in attempts
            if bool(record.get("strict_success_3s", False))
        }
        replayed_attempts = {int(record["attempt_index"]) for record in replays}
        summary = {
            "schema_version": KEYBOARD_GENERALIZATION_BATCH_VERSION,
            "attempts": len(attempts),
            "strict_successes_3s": successes,
            "requested_strict_successes": args.successes,
            "capture_complete": successes >= args.successes,
            "rgbd_replays_complete": successful_attempts <= replayed_attempts,
            "rgbd_replays": len(replayed_attempts),
            "stop_requested": stop_requested,
            "free_gib": store.assert_capacity() / 1024**3,
            "updated_unix_ns": time.time_ns(),
        }
        _atomic_json(store.root / "batch_summary.json", summary)
        _status(
            server,
            state="COMPLETE",
            task=task,
            successes=successes,
            requested=args.successes,
            rows=0,
            coverage=1.0 if successes else 0.0,
            message=(
                "V13 动作与腕部 RGB-D 已完成"
                if summary["rgbd_replays_complete"]
                else "动作采集已停止；仍有 RGB-D 待回放"
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


__all__ = ["KEYBOARD_GENERALIZATION_BATCH_VERSION", "main"]
