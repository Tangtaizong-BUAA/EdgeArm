"""Fail-closed iPhone/Android 6-DoF teleoperation runtime for EdgeArm.

The phone transport is intentionally separated from robot execution.  The
official LeRobot ``Phone`` teleoperator (HEBI Mobile I/O on iOS, WebXR on
Android) is read on a daemon thread because its current iOS feedback call can
block on Wi-Fi.  The control thread consumes only validated, host-timestamped
snapshots and stops producing motion targets when the stream is stale, the
deadman is released, tracking is invalid, or the reader fails.

This module contains no implicit hardware discovery and connecting a phone can
never move a robot.  A backend must explicitly consume the generated joint
target.  The MuJoCo backend included here is safe for local validation.  The
physical backend lives in :mod:`edgearm.physical_phone_teleop_v1` and requires
an explicit, expiring motion approval before it can send anything.
"""

from __future__ import annotations

import copy
import json
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Protocol

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


PHONE_TELEOP_RUNTIME_VERSION = "edgearm-phone-teleop-runtime-v4"
PHONE_CARTESIAN_MAPPING_VERSION = "edgearm-phone-tool-relative-position-locked-orientation-v1"
JOINT_COUNT = 6


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_vector(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    result = result.copy()
    result.setflags(write=False)
    return result


def _json_safe_mapping(value: Mapping[str, Any]) -> dict[str, float | int | bool | str]:
    result: dict[str, float | int | bool | str] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            continue
        if isinstance(item, (bool, str)):
            result[key] = item
        elif isinstance(item, (int, np.integer)):
            result[key] = int(item)
        elif isinstance(item, (float, np.floating)) and np.isfinite(item):
            result[key] = float(item)
    return result


class PhoneOS(str, Enum):
    IOS = "ios"


class PhoneConnectionStatus(str, Enum):
    NOT_STARTED = "not_started"
    SEARCHING = "searching"
    FOUND_WAITING_FOR_B1 = "found_waiting_for_b1"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    ANDROID = "android"


class TrackingState(str, Enum):
    """Tracking quality carried by a phone sample.

    LeRobot's current HEBI adapter does not expose HEBI's AR quality field, so
    official-adapter samples are truthfully marked ``unknown`` rather than
    promoted to ``normal``.
    """

    NORMAL = "normal"
    UNKNOWN = "unknown"
    LIMITED = "limited"
    UNAVAILABLE = "unavailable"


class StopReason(str, Enum):
    NONE = "none"
    NO_SAMPLE = "no_sample"
    DISABLED = "deadman_released"
    STALE = "phone_stream_stale"
    TRACKING = "tracking_not_accepted"
    READER_ERROR = "phone_reader_error"
    DISCONNECTED = "phone_disconnected"
    INVALID_TIME = "invalid_sample_time"
    KINEMATIC_JUMP = "phone_kinematic_jump"
    ESTOP = "estop"


@dataclass(frozen=True)
class PhonePoseSample:
    sequence: int
    receive_monotonic_ns: int
    position_m: np.ndarray
    orientation_xyzw: np.ndarray
    enabled: bool
    raw_inputs: Mapping[str, Any]
    tracking_state: TrackingState = TrackingState.UNKNOWN
    device_timestamp_ns: int | None = None
    source: str = "lerobot_phone"

    def __post_init__(self) -> None:
        object.__setattr__(self, "sequence", _positive_int(self.sequence, "sequence"))
        object.__setattr__(
            self,
            "receive_monotonic_ns",
            _positive_int(self.receive_monotonic_ns, "receive_monotonic_ns"),
        )
        object.__setattr__(self, "position_m", _finite_vector(self.position_m, (3,), "position_m"))
        quaternion = _finite_vector(
            self.orientation_xyzw, (4,), "orientation_xyzw"
        ).copy()
        norm = float(np.linalg.norm(quaternion))
        if norm < 1.0e-8:
            raise ValueError("orientation_xyzw must be a nonzero quaternion")
        quaternion /= norm
        quaternion.setflags(write=False)
        object.__setattr__(self, "orientation_xyzw", quaternion)
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be bool")
        object.__setattr__(self, "tracking_state", TrackingState(self.tracking_state))
        if self.device_timestamp_ns is not None:
            object.__setattr__(
                self,
                "device_timestamp_ns",
                _positive_int(self.device_timestamp_ns, "device_timestamp_ns"),
            )
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a non-empty string")
        object.__setattr__(self, "source", self.source.strip())
        object.__setattr__(self, "raw_inputs", _json_safe_mapping(self.raw_inputs))
        for control_name in ("a3", "reservedButtonA", "reservedButtonB"):
            if control_name not in self.raw_inputs:
                continue
            control_value = self.raw_inputs[control_name]
            if (
                isinstance(control_value, bool)
                or not isinstance(control_value, (int, float))
                or not np.isfinite(control_value)
            ):
                raise ValueError(f"raw_inputs[{control_name!r}] must be finite numeric")

    @property
    def gripper_velocity(self) -> float:
        if self.source.endswith("android"):
            open_button = float(self.raw_inputs.get("reservedButtonA", 0.0))
            close_button = float(self.raw_inputs.get("reservedButtonB", 0.0))
            return float(np.clip(open_button - close_button, -1.0, 1.0))
        return float(np.clip(float(self.raw_inputs.get("a3", 0.0)), -1.0, 1.0))

    def audit_json(self) -> str:
        return json.dumps(
            {
                "sequence": self.sequence,
                "receive_monotonic_ns": self.receive_monotonic_ns,
                "position_m": self.position_m.tolist(),
                "orientation_xyzw": self.orientation_xyzw.tolist(),
                "enabled": self.enabled,
                "raw_inputs": dict(self.raw_inputs),
                "tracking_state": self.tracking_state.value,
                "device_timestamp_ns": self.device_timestamp_ns,
                "source": self.source,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


class BlockingPhoneSource(Protocol):
    """Narrow interface implemented by official and test phone sources."""

    def connect(self) -> None: ...

    def read(self) -> PhonePoseSample | None: ...

    def disconnect(self) -> None: ...


class LeRobotPhoneSource:
    """Adapter around the official LeRobot Phone teleoperator.

    Imports are lazy so unit tests and simulation-only tools do not require the
    optional ``lerobot[phone]`` dependencies.  This adapter reports tracking as
    unknown because the upstream action contract currently omits AR quality and
    device timestamps.
    """

    def __init__(
        self,
        phone_os: PhoneOS = PhoneOS.IOS,
        *,
        clock: Callable[[], int] = time.monotonic_ns,
        connect_timeout_s: float = 180.0,
        discovery_retry_s: float = 1.0,
    ) -> None:
        if not np.isfinite(connect_timeout_s) or connect_timeout_s <= 0:
            raise ValueError("connect_timeout_s must be finite and positive")
        if not np.isfinite(discovery_retry_s) or discovery_retry_s <= 0:
            raise ValueError("discovery_retry_s must be finite and positive")
        self.phone_os = PhoneOS(phone_os)
        self._clock = clock
        self.connect_timeout_s = float(connect_timeout_s)
        self.discovery_retry_s = float(discovery_retry_s)
        self._phone: Any | None = None
        self._sequence = 0
        self._status_lock = threading.Lock()
        self._connection_status = PhoneConnectionStatus.NOT_STARTED

    @property
    def connection_status(self) -> PhoneConnectionStatus:
        with self._status_lock:
            return self._connection_status

    def _set_connection_status(self, status: PhoneConnectionStatus) -> None:
        with self._status_lock:
            self._connection_status = status

    def connect(self) -> None:
        if self._phone is not None:
            raise RuntimeError("LeRobot phone source is already connected")
        self._set_connection_status(PhoneConnectionStatus.SEARCHING)
        try:
            from lerobot.teleoperators.phone import Phone, PhoneConfig
            from lerobot.teleoperators.phone.config_phone import PhoneOS as LeRobotPhoneOS
        except ImportError as error:  # pragma: no cover - depends on optional install
            raise ImportError(
                "LeRobot phone dependencies are missing; install the project's phone extra"
            ) from error
        selected = (
            LeRobotPhoneOS.IOS
            if self.phone_os is PhoneOS.IOS
            else LeRobotPhoneOS.ANDROID
        )
        phone = Phone(PhoneConfig(phone_os=selected))
        deadline = time.monotonic() + self.connect_timeout_s
        if self.phone_os is PhoneOS.IOS:
            import hebi

            lookup = hebi.Lookup()
            while True:
                group = lookup.get_group_from_names(["HEBI"], ["mobileIO"])
                if group is not None:
                    self._set_connection_status(
                        PhoneConnectionStatus.FOUND_WAITING_FOR_B1
                    )
                    break
                if time.monotonic() >= deadline:
                    self._set_connection_status(PhoneConnectionStatus.ERROR)
                    raise RuntimeError(
                        "Mobile I/O not found; expected Family=HEBI and Name=mobileIO"
                    )
                time.sleep(self.discovery_retry_s)
        while True:
            try:
                phone.connect()
                break
            except RuntimeError as error:
                retryable = (
                    self.phone_os is PhoneOS.IOS
                    and "Mobile I/O not found" in str(error)
                    and time.monotonic() < deadline
                )
                if not retryable:
                    self._set_connection_status(PhoneConnectionStatus.ERROR)
                    raise
                time.sleep(self.discovery_retry_s)
        self._phone = phone
        self._set_connection_status(PhoneConnectionStatus.CONNECTED)

    def read(self) -> PhonePoseSample | None:
        if self._phone is None:
            raise RuntimeError("LeRobot phone source is not connected")
        action = self._phone.get_action()
        if not action:
            return None
        required = {"phone.pos", "phone.rot", "phone.raw_inputs", "phone.enabled"}
        if set(action) != required:
            raise RuntimeError(
                "LeRobot phone action contract changed: " + repr(sorted(action))
            )
        rotation = action["phone.rot"]
        if not hasattr(rotation, "as_quat"):
            raise RuntimeError("LeRobot phone rotation no longer exposes as_quat()")
        self._sequence += 1
        return PhonePoseSample(
            sequence=self._sequence,
            receive_monotonic_ns=_positive_int(int(self._clock()), "clock"),
            position_m=np.asarray(action["phone.pos"], dtype=np.float64),
            orientation_xyzw=np.asarray(rotation.as_quat(), dtype=np.float64),
            enabled=bool(action["phone.enabled"]),
            raw_inputs=action["phone.raw_inputs"],
            tracking_state=TrackingState.UNKNOWN,
            device_timestamp_ns=None,
            source=f"lerobot_phone_{self.phone_os.value}",
        )

    def disconnect(self) -> None:
        phone, self._phone = self._phone, None
        if phone is not None:
            phone.disconnect()
        if self.connection_status is not PhoneConnectionStatus.ERROR:
            self._set_connection_status(PhoneConnectionStatus.DISCONNECTED)


def set_phone_viewer_status(
    viewer: Any | None,
    status: PhoneConnectionStatus | str,
    detail: str = "",
) -> None:
    """Show connection state in the MuJoCo UI without entering sensor frames."""

    if viewer is None:
        return
    value = status.value if isinstance(status, PhoneConnectionStatus) else str(status)
    labels = {
        PhoneConnectionStatus.NOT_STARTED.value: "PHONE: NOT STARTED",
        PhoneConnectionStatus.SEARCHING.value: "PHONE: SEARCHING...",
        PhoneConnectionStatus.FOUND_WAITING_FOR_B1.value: "PHONE: FOUND - HOLD B1",
        PhoneConnectionStatus.CONNECTED.value: "PHONE: CONNECTED",
        PhoneConnectionStatus.DISCONNECTED.value: "PHONE: DISCONNECTED",
        PhoneConnectionStatus.ERROR.value: "PHONE: ERROR",
        "active": "PHONE: ACTIVE",
    }
    viewer.set_texts(
        (
            None,
            None,
            labels.get(value, f"PHONE: {value.upper()}"),
            detail,
        )
    )


@dataclass(frozen=True)
class PhoneReaderSnapshot:
    sample: PhonePoseSample | None
    connected: bool
    reader_error: str


class AsyncLatestPhoneReader:
    """Isolate a potentially blocking phone transport from the control loop."""

    def __init__(self, source: BlockingPhoneSource) -> None:
        self.source = source
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._connected_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: PhonePoseSample | None = None
        self._connected = False
        self._reader_error = ""

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            raise RuntimeError("phone reader is already running")
        self._stop.clear()
        self._connected_event.clear()
        with self._lock:
            self._latest = None
            self._connected = False
            self._reader_error = ""
        self._thread = threading.Thread(
            target=self._run,
            name="edgearm-phone-reader-v1",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        try:
            self.source.connect()
            with self._lock:
                self._connected = True
            self._connected_event.set()
            while not self._stop.is_set():
                sample = self.source.read()
                if sample is None:
                    # Some sources poll instead of blocking.  Avoid turning an
                    # empty source into a full-CPU busy loop.
                    time.sleep(0.001)
                    continue
                with self._lock:
                    if self._latest is not None and sample.sequence <= self._latest.sequence:
                        raise RuntimeError("phone source sequence did not increase")
                    self._latest = sample
        except Exception as error:  # transport failures are reported to the safety gate
            with self._lock:
                self._reader_error = f"{type(error).__name__}: {error}"
            self._connected_event.set()
        finally:
            with self._lock:
                self._connected = False
            try:
                self.source.disconnect()
            except Exception as error:
                with self._lock:
                    if not self._reader_error:
                        self._reader_error = f"disconnect {type(error).__name__}: {error}"

    def wait_until_ready(self, timeout_s: float) -> bool:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self._connected_event.wait(timeout_s)
        return self.snapshot().connected

    def snapshot(self) -> PhoneReaderSnapshot:
        with self._lock:
            return PhoneReaderSnapshot(
                sample=self._latest,
                connected=self._connected,
                reader_error=self._reader_error,
            )

    def stop(self, *, join_timeout_s: float = 1.0) -> None:
        if join_timeout_s < 0:
            raise ValueError("join_timeout_s must be non-negative")
        self._stop.set()
        # Revoke the control-plane connection synchronously.  The underlying
        # transport may remain blocked, but no consumer can treat its cached
        # sample as live after stop() returns.
        with self._lock:
            self._connected = False
        if self._thread is not None:
            self._thread.join(join_timeout_s)
            if self._thread.is_alive():
                # The upstream iOS feedback call can block indefinitely.  We
                # deliberately do not call disconnect concurrently with it;
                # the daemon thread cannot hold process shutdown open, and a
                # visible error keeps every consumer fail-closed.
                with self._lock:
                    if not self._reader_error:
                        self._reader_error = "phone reader did not stop before timeout"


@dataclass(frozen=True)
class PhoneSafetyConfig:
    stale_after_ns: int = 150_000_000
    future_tolerance_ns: int = 5_000_000
    allow_unknown_tracking: bool = False
    max_phone_linear_speed_m_s: float = 4.0
    max_phone_angular_speed_rad_s: float = 10.0

    def __post_init__(self) -> None:
        _positive_int(self.stale_after_ns, "stale_after_ns")
        _positive_int(self.future_tolerance_ns, "future_tolerance_ns")
        for name in (
            "max_phone_linear_speed_m_s",
            "max_phone_angular_speed_rad_s",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class PhoneSafetyDecision:
    active: bool
    stop_reason: StopReason
    sample_age_ns: int
    new_sample: bool

    def __post_init__(self) -> None:
        if not isinstance(self.active, bool) or not isinstance(self.new_sample, bool):
            raise ValueError("active and new_sample must be bool")
        object.__setattr__(self, "stop_reason", StopReason(self.stop_reason))
        if not isinstance(self.sample_age_ns, int) or isinstance(self.sample_age_ns, bool):
            raise ValueError("sample_age_ns must be an integer")
        if self.active and self.stop_reason is not StopReason.NONE:
            raise ValueError("an active decision cannot carry a stop reason")
        if not self.active and self.stop_reason is StopReason.NONE:
            raise ValueError("an inactive decision must carry a stop reason")


class PhoneSafetyGate:
    def __init__(self, config: PhoneSafetyConfig | None = None) -> None:
        self.config = config or PhoneSafetyConfig()
        self._last_sequence = 0
        self._last_pose_sample: PhonePoseSample | None = None

    def evaluate(
        self,
        snapshot: PhoneReaderSnapshot,
        *,
        now_ns: int,
        estop: bool = False,
    ) -> PhoneSafetyDecision:
        now_ns = _positive_int(now_ns, "now_ns")
        sample = snapshot.sample
        if estop:
            return PhoneSafetyDecision(False, StopReason.ESTOP, 0, False)
        if snapshot.reader_error:
            return PhoneSafetyDecision(False, StopReason.READER_ERROR, 0, False)
        if not snapshot.connected:
            return PhoneSafetyDecision(False, StopReason.DISCONNECTED, 0, False)
        if sample is None:
            return PhoneSafetyDecision(False, StopReason.NO_SAMPLE, 0, False)
        age_ns = now_ns - sample.receive_monotonic_ns
        if age_ns < -self.config.future_tolerance_ns:
            return PhoneSafetyDecision(False, StopReason.INVALID_TIME, age_ns, False)
        new_sample = sample.sequence > self._last_sequence
        if sample.sequence < self._last_sequence:
            return PhoneSafetyDecision(False, StopReason.INVALID_TIME, age_ns, False)
        if new_sample:
            self._last_sequence = sample.sequence
        if age_ns > self.config.stale_after_ns:
            return PhoneSafetyDecision(False, StopReason.STALE, age_ns, new_sample)
        tracking_ok = sample.tracking_state is TrackingState.NORMAL or (
            sample.tracking_state is TrackingState.UNKNOWN
            and self.config.allow_unknown_tracking
        )
        if not tracking_ok:
            return PhoneSafetyDecision(False, StopReason.TRACKING, age_ns, new_sample)
        if not sample.enabled:
            self._last_pose_sample = sample
            return PhoneSafetyDecision(False, StopReason.DISABLED, age_ns, new_sample)
        previous = self._last_pose_sample
        self._last_pose_sample = sample
        if previous is not None and previous.enabled and new_sample:
            delta_time_s = (
                sample.receive_monotonic_ns - previous.receive_monotonic_ns
            ) / 1.0e9
            if delta_time_s <= 0:
                return PhoneSafetyDecision(False, StopReason.INVALID_TIME, age_ns, new_sample)
            linear_speed = float(
                np.linalg.norm(sample.position_m - previous.position_m) / delta_time_s
            )
            angular_delta = Rotation.from_quat(previous.orientation_xyzw).inv() * Rotation.from_quat(
                sample.orientation_xyzw
            )
            angular_speed = float(np.linalg.norm(angular_delta.as_rotvec()) / delta_time_s)
            if (
                linear_speed > self.config.max_phone_linear_speed_m_s
                or angular_speed > self.config.max_phone_angular_speed_rad_s
            ):
                return PhoneSafetyDecision(
                    False,
                    StopReason.KINEMATIC_JUMP,
                    age_ns,
                    new_sample,
                )
        return PhoneSafetyDecision(True, StopReason.NONE, age_ns, new_sample)

    def reset(self) -> None:
        self._last_sequence = 0
        self._last_pose_sample = None


@dataclass(frozen=True)
class RetargetConfig:
    translation_scale: float = 0.50
    max_phone_translation_m: float = 0.40
    max_phone_rotation_rad: float = 1.40
    max_ee_step_m: float = 0.008
    max_ee_rotation_step_rad: float = 0.08
    gripper_step_rad: float = 0.020
    orientation_weight: float = 0.05
    ik_damping: float = 2.5e-3
    ik_iterations: int = 60
    ik_step_limit_rad: float = 0.06
    ik_position_tolerance_m: float = 0.004
    ik_orientation_tolerance_rad: float = 0.20
    translation_smoothing_alpha: float = 1.0
    rotation_smoothing_alpha: float = 1.0
    ik_backtracking_scales: tuple[float, ...] = ()
    orientation_control_enabled: bool = True

    def __post_init__(self) -> None:
        for name in (
            "translation_scale",
            "max_phone_translation_m",
            "max_phone_rotation_rad",
            "max_ee_step_m",
            "max_ee_rotation_step_rad",
            "gripper_step_rad",
            "orientation_weight",
            "ik_damping",
            "ik_step_limit_rad",
            "ik_position_tolerance_m",
            "ik_orientation_tolerance_rad",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "translation_smoothing_alpha",
            "rotation_smoothing_alpha",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0 or value > 1.0:
                raise ValueError(f"{name} must be finite in (0,1]")
        scales = tuple(float(value) for value in self.ik_backtracking_scales)
        if any(not np.isfinite(value) or value <= 0.0 or value >= 1.0 for value in scales):
            raise ValueError("ik_backtracking_scales must be finite in (0,1)")
        if any(left <= right for left, right in zip(scales, scales[1:])):
            raise ValueError("ik_backtracking_scales must be strictly decreasing")
        object.__setattr__(self, "ik_backtracking_scales", scales)
        _positive_int(self.ik_iterations, "ik_iterations")
        if not isinstance(self.orientation_control_enabled, bool):
            raise ValueError("orientation_control_enabled must be bool")


SMOOTH_PHONE_CONTROL_PROFILE_V2 = "edgearm-phone-smooth-control-v2"
SMOOTH_PHONE_RETARGET_CONFIG_V2 = RetargetConfig(
    max_ee_step_m=0.006,
    max_ee_rotation_step_rad=0.05,
    ik_position_tolerance_m=0.0015,
    ik_orientation_tolerance_rad=0.12,
    translation_smoothing_alpha=0.55,
    rotation_smoothing_alpha=0.45,
    ik_backtracking_scales=(0.5, 0.25, 0.125),
)
INTUITIVE_CARTESIAN_RETARGET_CONFIG_V3 = replace(
    SMOOTH_PHONE_RETARGET_CONFIG_V2,
    orientation_control_enabled=False,
)


@dataclass(frozen=True)
class CartesianTarget:
    transform_world_from_ee: np.ndarray
    gripper_position_rad: float
    phone_translation_clipped: bool
    phone_rotation_clipped: bool
    workspace_clipped: bool
    rate_limited: bool

    def __post_init__(self) -> None:
        transform = _finite_vector(
            self.transform_world_from_ee, (4, 4), "transform_world_from_ee"
        )
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0]):
            raise ValueError("transform_world_from_ee must be homogeneous")
        object.__setattr__(self, "transform_world_from_ee", transform)
        if not np.isfinite(self.gripper_position_rad):
            raise ValueError("gripper_position_rad must be finite")


class PhoneCartesianRetargeter:
    """Clutched phone translation in the tool-forward Cartesian frame."""
    # Matches LeRobot's current rotvec mapping [ry, rx, -rz].
    _ORIENTATION_MAP = np.asarray(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
        dtype=np.float64,
    )

    def __init__(self, config: RetargetConfig | None = None) -> None:
        self.config = config or RetargetConfig()
        self._engaged = False
        self._phone_position_reference: np.ndarray | None = None
        self._phone_rotation_reference: Rotation | None = None
        self._ee_reference: np.ndarray | None = None
        self._last_target: np.ndarray | None = None
        self._last_gripper_rad: float | None = None

    def reset(self) -> None:
        self._engaged = False
        self._phone_position_reference = None
        self._phone_rotation_reference = None
        self._ee_reference = None
        self._last_target = None
        self._last_gripper_rad = None

    def commit_applied_target(self, target: CartesianTarget) -> None:
        """Synchronize the filter state with the target actually sent to MuJoCo.

        ``target()`` predicts a filtered command before IK and plant safety are
        known.  The control loop calls this method after IK selection so an
        unreachable request cannot advance the filter and cause a later jump.
        """

        if not self._engaged:
            return
        self._last_target = target.transform_world_from_ee.copy()
        self._last_gripper_rad = float(target.gripper_position_rad)

    @staticmethod
    def _position_map_world_from_phone(ee_reference: np.ndarray) -> np.ndarray:
        """Map phone right/top/up to tool right/forward/world-up.

        The gripper's authored wide pushing face uses site-local +Y as its
        forward normal.  Phone +X is screen-right, +Y points toward the
        earpiece/top edge, and +Z is vertical after the iOS clutch calibration.
        """

        forward = np.asarray(ee_reference[:3, 1], dtype=np.float64).copy()
        forward[2] = 0.0
        norm = float(np.linalg.norm(forward))
        if norm < 1.0e-8:
            forward = np.asarray([1.0, 0.0, 0.0])
        else:
            forward /= norm
        left = np.cross(np.asarray([0.0, 0.0, 1.0]), forward)
        right = -left
        up = np.asarray([0.0, 0.0, 1.0])
        return np.column_stack([right, forward, up])

    def target(
        self,
        sample: PhonePoseSample,
        decision: PhoneSafetyDecision,
        *,
        current_ee_transform: np.ndarray,
        current_gripper_rad: float,
        workspace_min_m: np.ndarray,
        workspace_max_m: np.ndarray,
        gripper_min_rad: float,
        gripper_max_rad: float,
    ) -> CartesianTarget | None:
        current = np.asarray(current_ee_transform, dtype=np.float64)
        if current.shape != (4, 4) or not np.isfinite(current).all():
            raise ValueError("current_ee_transform must be finite [4,4]")
        workspace_min = _finite_vector(workspace_min_m, (3,), "workspace_min_m")
        workspace_max = _finite_vector(workspace_max_m, (3,), "workspace_max_m")
        if np.any(workspace_max <= workspace_min):
            raise ValueError("workspace bounds must be ordered")
        if not decision.active:
            self._engaged = False
            return None

        phone_rotation = Rotation.from_quat(sample.orientation_xyzw)
        if not self._engaged:
            self._engaged = True
            self._phone_position_reference = sample.position_m.copy()
            self._phone_rotation_reference = phone_rotation
            self._ee_reference = current.copy()
            self._last_target = current.copy()
            self._last_gripper_rad = float(current_gripper_rad)

        if (
            self._phone_position_reference is None
            or self._phone_rotation_reference is None
            or self._ee_reference is None
            or self._last_target is None
            or self._last_gripper_rad is None
        ):
            raise RuntimeError("retargeter engage state is incomplete")

        phone_delta = np.asarray(sample.position_m) - self._phone_position_reference
        phone_translation_clipped = False
        translation_norm = float(np.linalg.norm(phone_delta))
        if translation_norm > self.config.max_phone_translation_m:
            phone_delta *= self.config.max_phone_translation_m / translation_norm
            phone_translation_clipped = True
        position_map = self._position_map_world_from_phone(self._ee_reference)
        mapped_delta = position_map @ phone_delta
        desired_position = (
            self._ee_reference[:3, 3]
            + self.config.translation_scale * mapped_delta
        )
        clipped_position = np.clip(desired_position, workspace_min, workspace_max)
        workspace_clipped = not np.allclose(clipped_position, desired_position)

        phone_rotation_clipped = False
        if self.config.orientation_control_enabled:
            phone_relative = self._phone_rotation_reference.inv() * phone_rotation
            phone_rotvec = phone_relative.as_rotvec()
            angle = float(np.linalg.norm(phone_rotvec))
            if angle > self.config.max_phone_rotation_rad:
                phone_rotvec *= self.config.max_phone_rotation_rad / angle
                phone_rotation_clipped = True
            mapped_rotvec = self._ORIENTATION_MAP @ phone_rotvec
            desired_rotation = self._ee_reference[:3, :3] @ Rotation.from_rotvec(
                mapped_rotvec
            ).as_matrix()
        else:
            desired_rotation = self._ee_reference[:3, :3].copy()

        rate_limited = False
        previous_position = self._last_target[:3, 3]
        position_step = clipped_position - previous_position
        position_step *= self.config.translation_smoothing_alpha
        clipped_position = previous_position + position_step
        step_norm = float(np.linalg.norm(position_step))
        if step_norm > self.config.max_ee_step_m:
            clipped_position = (
                previous_position
                + position_step * (self.config.max_ee_step_m / step_norm)
            )
            rate_limited = True

        previous_rotation = Rotation.from_matrix(self._last_target[:3, :3])
        rotation_step = Rotation.from_matrix(desired_rotation) * previous_rotation.inv()
        rotation_step_vector = rotation_step.as_rotvec()
        rotation_step_vector *= self.config.rotation_smoothing_alpha
        desired_rotation = (
            Rotation.from_rotvec(rotation_step_vector) * previous_rotation
        ).as_matrix()
        rotation_step_norm = float(np.linalg.norm(rotation_step_vector))
        if rotation_step_norm > self.config.max_ee_rotation_step_rad:
            rotation_step_vector *= (
                self.config.max_ee_rotation_step_rad / rotation_step_norm
            )
            desired_rotation = (
                Rotation.from_rotvec(rotation_step_vector) * previous_rotation
            ).as_matrix()
            rate_limited = True

        gripper = float(
            np.clip(
                self._last_gripper_rad
                + sample.gripper_velocity * self.config.gripper_step_rad,
                gripper_min_rad,
                gripper_max_rad,
            )
        )
        target = np.eye(4, dtype=np.float64)
        target[:3, :3] = desired_rotation
        target[:3, 3] = clipped_position
        self._last_target = target.copy()
        self._last_gripper_rad = gripper
        return CartesianTarget(
            transform_world_from_ee=target,
            gripper_position_rad=gripper,
            phone_translation_clipped=phone_translation_clipped,
            phone_rotation_clipped=phone_rotation_clipped,
            workspace_clipped=workspace_clipped,
            rate_limited=rate_limited,
        )


@dataclass(frozen=True)
class IKResult:
    target_joint_position_rad: np.ndarray
    position_error_m: float
    orientation_error_rad: float
    converged: bool
    joint_or_workspace_clipped: bool
    safety_reason: str
    application_scale: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_joint_position_rad",
            _finite_vector(
                self.target_joint_position_rad,
                (JOINT_COUNT,),
                "target_joint_position_rad",
            ),
        )
        scale = float(self.application_scale)
        if not np.isfinite(scale) or scale < 0.0 or scale > 1.0:
            raise ValueError("application_scale must be finite in [0,1]")
        object.__setattr__(self, "application_scale", scale)


def _interpolate_cartesian_target(
    current_transform: np.ndarray,
    current_gripper_rad: float,
    requested: CartesianTarget,
    scale: float,
) -> CartesianTarget:
    """Return a pose on the shortest SE(3) path to a requested target."""

    current = _finite_vector(current_transform, (4, 4), "current_transform")
    if not np.isfinite(scale) or scale <= 0.0 or scale >= 1.0:
        raise ValueError("scale must be finite in (0,1)")
    partial = np.eye(4, dtype=np.float64)
    partial[:3, 3] = current[:3, 3] + scale * (
        requested.transform_world_from_ee[:3, 3] - current[:3, 3]
    )
    current_rotation = Rotation.from_matrix(current[:3, :3])
    relative_rotation = (
        Rotation.from_matrix(requested.transform_world_from_ee[:3, :3])
        * current_rotation.inv()
    )
    partial[:3, :3] = (
        Rotation.from_rotvec(scale * relative_rotation.as_rotvec())
        * current_rotation
    ).as_matrix()
    gripper = float(
        current_gripper_rad
        + scale * (requested.gripper_position_rad - current_gripper_rad)
    )
    return CartesianTarget(
        transform_world_from_ee=partial,
        gripper_position_rad=gripper,
        phone_translation_clipped=requested.phone_translation_clipped,
        phone_rotation_clipped=requested.phone_rotation_clipped,
        workspace_clipped=requested.workspace_clipped,
        rate_limited=True,
    )


class MujocoSO101CartesianIK:
    """Damped closed-loop IK against the exact active EdgeArm MuJoCo model."""

    def __init__(self, env: Any, config: RetargetConfig | None = None) -> None:
        self.env = env
        self.config = config or RetargetConfig()
        self._scratch = mujoco.MjData(env.model)

    def current_ee_transform(self, joint_position_rad: np.ndarray | None = None) -> np.ndarray:
        data = self.env.data
        if joint_position_rad is not None:
            q = _finite_vector(joint_position_rad, (JOINT_COUNT,), "joint_position_rad")
            self._scratch.qpos[:] = self.env.data.qpos
            self._scratch.qvel[:] = self.env.data.qvel
            self._scratch.qpos[:JOINT_COUNT] = q
            mujoco.mj_forward(self.env.model, self._scratch)
            data = self._scratch
        site = int(self.env._ids["tool_site"])
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.asarray(data.site_xmat[site]).reshape(3, 3)
        transform[:3, 3] = np.asarray(data.site_xpos[site])
        return transform

    def solve(
        self,
        current_joint_position_rad: np.ndarray,
        target: CartesianTarget,
    ) -> IKResult:
        q_current = _finite_vector(
            current_joint_position_rad,
            (JOINT_COUNT,),
            "current_joint_position_rad",
        )
        self._scratch.qpos[:] = self.env.data.qpos
        self._scratch.qvel[:] = self.env.data.qvel
        self._scratch.qpos[:JOINT_COUNT] = q_current
        desired = target.transform_world_from_ee
        site = int(self.env._ids["tool_site"])
        joint_ranges = np.asarray(self.env.joint_ranges, dtype=np.float64)
        position_error_norm = float("inf")
        for iteration in range(self.config.ik_iterations):
            mujoco.mj_forward(self.env.model, self._scratch)
            current_position = np.asarray(self._scratch.site_xpos[site])
            current_rotation = np.asarray(self._scratch.site_xmat[site]).reshape(3, 3)
            position_error = desired[:3, 3] - current_position
            orientation_error = Rotation.from_matrix(
                desired[:3, :3] @ current_rotation.T
            ).as_rotvec()
            position_error_norm = float(np.linalg.norm(position_error))
            # Always execute at least one damped update.  Previously, a target
            # inside the acceptance tolerance returned the current joints,
            # turning a stream of small phone motions into periodic jumps.
            if iteration > 0 and (
                position_error_norm <= self.config.ik_position_tolerance_m
                and float(np.linalg.norm(orientation_error))
                <= self.config.ik_orientation_tolerance_rad
            ):
                break
            jac_pos = np.zeros((3, self.env.model.nv), dtype=np.float64)
            jac_rot = np.zeros((3, self.env.model.nv), dtype=np.float64)
            mujoco.mj_jacSite(
                self.env.model, self._scratch, jac_pos, jac_rot, site
            )
            weight = self.config.orientation_weight
            jacobian = np.vstack(
                [jac_pos[:, :5], weight * jac_rot[:, :5]]
            )
            error = np.concatenate([position_error, weight * orientation_error])
            normal = jacobian @ jacobian.T
            delta = jacobian.T @ np.linalg.solve(
                normal + self.config.ik_damping * np.eye(normal.shape[0]), error
            )
            delta = np.clip(
                delta,
                -self.config.ik_step_limit_rad,
                self.config.ik_step_limit_rad,
            )
            self._scratch.qpos[:5] = np.clip(
                self._scratch.qpos[:5] + delta,
                joint_ranges[:5, 0],
                joint_ranges[:5, 1],
            )
        candidate = np.asarray(self._scratch.qpos[:JOINT_COUNT]).copy()
        candidate[5] = np.clip(
            target.gripper_position_rad,
            joint_ranges[5, 0],
            joint_ranges[5, 1],
        )
        filtered, safety_reason = self.env._safety_filter(candidate)
        filtered = np.asarray(filtered, dtype=np.float64)
        clipped = not np.allclose(filtered, candidate, atol=1.0e-12, rtol=0.0)
        final_pose = self.current_ee_transform(filtered)
        final_position_error = float(
            np.linalg.norm(desired[:3, 3] - final_pose[:3, 3])
        )
        final_orientation_error = float(
            np.linalg.norm(
                Rotation.from_matrix(
                    desired[:3, :3] @ final_pose[:3, :3].T
                ).as_rotvec()
            )
        )
        return IKResult(
            target_joint_position_rad=filtered,
            position_error_m=final_position_error,
            orientation_error_rad=final_orientation_error,
            converged=(
                final_position_error <= self.config.ik_position_tolerance_m
                and final_orientation_error <= self.config.ik_orientation_tolerance_rad
            ),
            joint_or_workspace_clipped=clipped or bool(safety_reason),
            safety_reason=str(safety_reason),
        )

    def solve_with_backtracking(
        self,
        current_joint_position_rad: np.ndarray,
        target: CartesianTarget,
        *,
        current_ee_transform: np.ndarray | None = None,
    ) -> tuple[CartesianTarget, IKResult]:
        """Select the largest safe converged step, or explicitly hold.

        Backtracking is attempted only after the requested target fails, so the
        common path has no extra solve.  Every candidate still passes through
        the environment's exact joint, workspace, desk, and collision filter.
        """

        current = _finite_vector(
            current_joint_position_rad,
            (JOINT_COUNT,),
            "current_joint_position_rad",
        )
        result = self.solve(current, target)
        if result.converged:
            return target, result
        pose = (
            self.current_ee_transform(current)
            if current_ee_transform is None
            else _finite_vector(
                current_ee_transform,
                (4, 4),
                "current_ee_transform",
            )
        )
        for scale in self.config.ik_backtracking_scales:
            partial = _interpolate_cartesian_target(
                pose,
                float(current[5]),
                target,
                scale,
            )
            partial_result = self.solve(current, partial)
            if partial_result.converged:
                return partial, replace(
                    partial_result,
                    application_scale=scale,
                )
        return target, replace(result, application_scale=0.0)


@dataclass(frozen=True)
class SimStepResult:
    q_before_rad: np.ndarray
    dq_before_rad_s: np.ndarray
    target_q_rad: np.ndarray
    queued_target_q_rad: np.ndarray
    submitted_normalized_action: np.ndarray
    q_after_rad: np.ndarray
    dq_after_rad_s: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]
    command_queued_monotonic_ns: int
    completed_monotonic_ns: int

    def __post_init__(self) -> None:
        for name in (
            "q_before_rad",
            "dq_before_rad_s",
            "target_q_rad",
            "queued_target_q_rad",
            "submitted_normalized_action",
            "q_after_rad",
            "dq_after_rad_s",
        ):
            object.__setattr__(
                self,
                name,
                _finite_vector(getattr(self, name), (JOINT_COUNT,), name),
            )
        if not np.isfinite(self.reward):
            raise ValueError("reward must be finite")
        for name in ("command_queued_monotonic_ns", "completed_monotonic_ns"):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        if self.completed_monotonic_ns < self.command_queued_monotonic_ns:
            raise ValueError("completed timestamp cannot precede queued timestamp")


class MujocoPhoneBackend:
    """Execute a joint target only inside an EdgeArm MuJoCo environment."""

    physical_hardware = False

    def __init__(
        self,
        env: Any,
        *,
        clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.env = env
        self._clock = clock

    def step(self, target_q_rad: np.ndarray | None) -> SimStepResult:
        before = self.env.observation()["joint_state"]
        q_before = np.asarray(before[:JOINT_COUNT], dtype=np.float64)
        dq_before = np.asarray(before[JOINT_COUNT:], dtype=np.float64)
        if target_q_rad is None:
            target = q_before.copy()
            action = np.zeros(JOINT_COUNT, dtype=np.float64)
        else:
            target = _finite_vector(target_q_rad, (JOINT_COUNT,), "target_q_rad")
            action = np.clip(
                (target - q_before) / float(self.env.config.max_joint_delta),
                -1.0,
                1.0,
            )
        queued_ns = _positive_int(int(self._clock()), "clock")
        queued_target = q_before + action * float(self.env.config.max_joint_delta)
        observation, reward, terminated, truncated, info = self.env.step(action)
        completed_ns = _positive_int(int(self._clock()), "clock")
        after = np.asarray(observation["joint_state"], dtype=np.float64)
        return SimStepResult(
            q_before_rad=q_before,
            dq_before_rad_s=dq_before,
            target_q_rad=target,
            queued_target_q_rad=queued_target,
            submitted_normalized_action=action,
            q_after_rad=after[:JOINT_COUNT],
            dq_after_rad_s=after[JOINT_COUNT:],
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=copy.deepcopy(info),
            command_queued_monotonic_ns=queued_ns,
            completed_monotonic_ns=completed_ns,
        )


@dataclass(frozen=True)
class PhoneControlCycle:
    sample: PhonePoseSample | None
    decision: PhoneSafetyDecision
    cartesian_target: CartesianTarget | None
    ik_result: IKResult | None
    step_result: SimStepResult


class PhoneMujocoControlLoop:
    """One deterministic control-cycle composition used by CLI and tests."""

    def __init__(
        self,
        env: Any,
        reader: AsyncLatestPhoneReader,
        *,
        safety_gate: PhoneSafetyGate | None = None,
        retargeter: PhoneCartesianRetargeter | None = None,
        ik: MujocoSO101CartesianIK | None = None,
        backend: MujocoPhoneBackend | None = None,
        clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.env = env
        self.reader = reader
        self.safety_gate = safety_gate or PhoneSafetyGate()
        self.retargeter = retargeter or PhoneCartesianRetargeter()
        self.ik = ik or MujocoSO101CartesianIK(env, self.retargeter.config)
        self.backend = backend or MujocoPhoneBackend(env, clock=clock)
        if not isinstance(self.backend, MujocoPhoneBackend):
            raise TypeError("PhoneMujocoControlLoop accepts only MujocoPhoneBackend")
        if self.backend.physical_hardware is not False or self.backend.env is not env:
            raise ValueError("MuJoCo backend must be non-physical and bound to the same env")
        self._clock = clock

    def cycle(self, *, estop: bool = False) -> PhoneControlCycle:
        snapshot = self.reader.snapshot()
        now_ns = _positive_int(int(self._clock()), "clock")
        decision = self.safety_gate.evaluate(snapshot, now_ns=now_ns, estop=estop)
        sample = snapshot.sample
        current = self.env.observation()["joint_state"][:JOINT_COUNT]
        current_ee = self.ik.current_ee_transform(current)
        cartesian_target: CartesianTarget | None = None
        ik_result: IKResult | None = None
        target_q: np.ndarray | None = None
        # A repeated latest-value snapshot is a hold, not a new controller
        # command.  In particular this prevents repeated gripper integration
        # when the phone update rate is lower than the control-loop rate.
        if sample is not None and (decision.new_sample or not decision.active):
            cartesian_target = self.retargeter.target(
                sample,
                decision,
                current_ee_transform=current_ee,
                current_gripper_rad=float(current[5]),
                workspace_min_m=np.asarray(
                    [bound[0] for bound in (
                        self.env.config.workspace_x,
                        self.env.config.workspace_y,
                        self.env.config.workspace_z,
                    )]
                ),
                workspace_max_m=np.asarray(
                    [bound[1] for bound in (
                        self.env.config.workspace_x,
                        self.env.config.workspace_y,
                        self.env.config.workspace_z,
                    )]
                ),
                gripper_min_rad=float(self.env.joint_ranges[5, 0]),
                gripper_max_rad=float(self.env.joint_ranges[5, 1]),
            )
        if cartesian_target is not None and decision.new_sample:
            applied_target, ik_result = self.ik.solve_with_backtracking(
                current,
                cartesian_target,
                current_ee_transform=current_ee,
            )
            if ik_result.converged:
                cartesian_target = applied_target
                target_q = ik_result.target_joint_position_rad
                self.retargeter.commit_applied_target(applied_target)
            else:
                hold_target = CartesianTarget(
                    transform_world_from_ee=current_ee,
                    gripper_position_rad=float(current[5]),
                    phone_translation_clipped=cartesian_target.phone_translation_clipped,
                    phone_rotation_clipped=cartesian_target.phone_rotation_clipped,
                    workspace_clipped=cartesian_target.workspace_clipped,
                    rate_limited=True,
                )
                self.retargeter.commit_applied_target(hold_target)
        step_result = self.backend.step(target_q)
        return PhoneControlCycle(
            sample=sample,
            decision=decision,
            cartesian_target=cartesian_target,
            ik_result=ik_result,
            step_result=step_result,
        )


@dataclass
class ScriptedPhoneSource:
    """Deterministic offline source for smoke tests and fault injection."""

    samples: list[PhonePoseSample]
    _connected: bool = field(default=False, init=False)
    _index: int = field(default=0, init=False)

    def connect(self) -> None:
        self._connected = True

    def read(self) -> PhonePoseSample | None:
        if not self._connected:
            raise RuntimeError("scripted phone source is not connected")
        if self._index >= len(self.samples):
            time.sleep(0.001)
            return None
        sample = self.samples[self._index]
        self._index += 1
        return sample

    def disconnect(self) -> None:
        self._connected = False


__all__ = [
    "PHONE_TELEOP_RUNTIME_VERSION",
    "PHONE_CARTESIAN_MAPPING_VERSION",
    "AsyncLatestPhoneReader",
    "BlockingPhoneSource",
    "CartesianTarget",
    "IKResult",
    "INTUITIVE_CARTESIAN_RETARGET_CONFIG_V3",
    "LeRobotPhoneSource",
    "MujocoPhoneBackend",
    "MujocoSO101CartesianIK",
    "PhoneCartesianRetargeter",
    "PhoneControlCycle",
    "PhoneConnectionStatus",
    "PhoneMujocoControlLoop",
    "PhoneOS",
    "PhonePoseSample",
    "PhoneReaderSnapshot",
    "PhoneSafetyConfig",
    "PhoneSafetyDecision",
    "PhoneSafetyGate",
    "RetargetConfig",
    "SMOOTH_PHONE_CONTROL_PROFILE_V2",
    "SMOOTH_PHONE_RETARGET_CONFIG_V2",
    "ScriptedPhoneSource",
    "SimStepResult",
    "StopReason",
    "TrackingState",
    "set_phone_viewer_status",
]
