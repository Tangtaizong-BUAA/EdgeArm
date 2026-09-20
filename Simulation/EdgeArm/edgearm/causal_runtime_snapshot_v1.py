"""Source-neutral immutable MJCF runtime capture for causal collection.

The runtime bundle contains only the exact XML/STL bytes consumed by MuJoCo.
It deliberately carries no expert, PPO, controller, or promotion identity, so
both V11 and scratch policies can compile the same physical scene without
claiming that the scene is part of either policy artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path, PurePosixPath
import stat

from .production_env import ProductionMjcfBundleV1


CAUSAL_RUNTIME_SNAPSHOT_FORMAT = "edgearm-causal-runtime-mjcf-snapshot-v1"
CAUSAL_RUNTIME_MAIN_LOGICAL_PATH = "SO101/edgearm_m2_m4_scene.xml"
CAUSAL_RUNTIME_ASSET_PATHS = (
    CAUSAL_RUNTIME_MAIN_LOGICAL_PATH,
    "SO101/scene.xml",
    "SO101/so101_new_calib.xml",
    "SO101/assets/waveshare_mounting_plate_so101_v2.stl",
    "SO101/assets/sts3215_03a_v1.stl",
    "SO101/assets/motor_holder_so101_base_v1.stl",
    "SO101/assets/wrist_roll_follower_so101_v1.stl",
    "SO101/assets/moving_jaw_so101_v1.stl",
    "SO101/assets/base_motor_holder_so101_v1.stl",
    "SO101/assets/upper_arm_so101_v1.stl",
    "SO101/assets/wrist_roll_pitch_so101_v2.stl",
    "SO101/assets/under_arm_so101_v1.stl",
    "SO101/assets/rotation_pitch_so101_v1.stl",
    "SO101/assets/motor_holder_so101_wrist_v1.stl",
    "SO101/assets/sts3215_03a_no_horn_v1.stl",
    "SO101/assets/base_so101_v2.stl",
)
_MAX_RUNTIME_FILE_BYTES = 64 * 1024 * 1024


def _canonical_logical_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("runtime logical path must be canonical relative POSIX")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("runtime logical path must be canonical relative POSIX")
    return path


def _directory_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise RuntimeError("runtime capture requires O_NOFOLLOW and O_DIRECTORY")
    return os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY


def _read_regular_file_below_root_v1(root_descriptor: int, logical_path: str) -> bytes:
    path = _canonical_logical_path(logical_path)
    current = os.dup(root_descriptor)
    try:
        for part in path.parts[:-1]:
            child = os.open(part, _directory_flags(), dir_fd=current)
            os.close(current)
            current = child
        descriptor = os.open(
            path.parts[-1],
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=current,
        )
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"runtime asset is not a regular file: {logical_path}")
            if before.st_size < 1 or before.st_size > _MAX_RUNTIME_FILE_BYTES:
                raise ValueError(f"runtime asset has an invalid size: {logical_path}")
            chunks: list[bytes] = []
            remaining = int(before.st_size)
            while remaining:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    raise ValueError(f"runtime asset changed while read: {logical_path}")
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                raise ValueError(f"runtime asset grew while read: {logical_path}")
            after = os.fstat(descriptor)
            identity_before = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            identity_after = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            if identity_after != identity_before:
                raise ValueError(f"runtime asset changed while read: {logical_path}")
            return b"".join(chunks)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise ValueError(
            f"runtime asset is missing, unreadable, or traverses a symlink: {logical_path}"
        ) from error
    finally:
        os.close(current)


def runtime_bundle_sha256_v1(files: tuple[tuple[str, bytes], ...]) -> str:
    digest = hashlib.sha256()
    namespace = b"runtime_asset"
    for logical_path, payload in files:
        canonical = _canonical_logical_path(logical_path).as_posix().encode("utf-8")
        if not isinstance(payload, bytes) or not payload:
            raise ValueError("runtime bundle payloads must be non-empty bytes")
        digest.update(len(namespace).to_bytes(4, "big"))
        digest.update(namespace)
        digest.update(len(canonical).to_bytes(4, "big"))
        digest.update(canonical)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CausalRuntimeSnapshotV1:
    runtime_root: Path
    files: tuple[tuple[str, bytes], ...]
    runtime_bundle_sha256: str
    format: str = CAUSAL_RUNTIME_SNAPSHOT_FORMAT

    def __post_init__(self) -> None:
        if self.format != CAUSAL_RUNTIME_SNAPSHOT_FORMAT:
            raise ValueError("causal runtime snapshot format mismatch")
        if tuple(path for path, _payload in self.files) != CAUSAL_RUNTIME_ASSET_PATHS:
            raise ValueError("causal runtime snapshot inventory differs from the exact closure")
        if runtime_bundle_sha256_v1(self.files) != self.runtime_bundle_sha256:
            raise ValueError("causal runtime snapshot hash differs from its bytes")

    def production_bundle_v1(self) -> ProductionMjcfBundleV1:
        return ProductionMjcfBundleV1(
            main_logical_path=CAUSAL_RUNTIME_MAIN_LOGICAL_PATH,
            files=self.files,
        )


def capture_causal_runtime_snapshot_v1(runtime_root: Path) -> CausalRuntimeSnapshotV1:
    root = Path(runtime_root).expanduser()
    try:
        root_mode = root.lstat().st_mode
    except OSError as error:
        raise ValueError("causal runtime root is missing or unreadable") from error
    if stat.S_ISLNK(root_mode) or not stat.S_ISDIR(root_mode):
        raise ValueError("causal runtime root must be a non-symlink directory")
    root = root.resolve(strict=True)
    descriptor = os.open(root, _directory_flags())
    try:
        first = tuple(
            (path, _read_regular_file_below_root_v1(descriptor, path)) for path in CAUSAL_RUNTIME_ASSET_PATHS
        )
        second = tuple(
            (path, _read_regular_file_below_root_v1(descriptor, path)) for path in CAUSAL_RUNTIME_ASSET_PATHS
        )
    finally:
        os.close(descriptor)
    if first != second:
        raise ValueError("causal runtime closure changed across its capture boundary")
    return CausalRuntimeSnapshotV1(
        runtime_root=root,
        files=first,
        runtime_bundle_sha256=runtime_bundle_sha256_v1(first),
    )


def revalidate_causal_runtime_snapshot_v1(snapshot: CausalRuntimeSnapshotV1) -> None:
    if not isinstance(snapshot, CausalRuntimeSnapshotV1):
        raise TypeError("runtime revalidation requires CausalRuntimeSnapshotV1")
    current = capture_causal_runtime_snapshot_v1(snapshot.runtime_root)
    if current.files != snapshot.files or current.runtime_bundle_sha256 != (snapshot.runtime_bundle_sha256):
        raise ValueError("causal runtime closure changed after snapshot capture")


__all__ = [
    "CAUSAL_RUNTIME_ASSET_PATHS",
    "CAUSAL_RUNTIME_MAIN_LOGICAL_PATH",
    "CAUSAL_RUNTIME_SNAPSHOT_FORMAT",
    "CausalRuntimeSnapshotV1",
    "capture_causal_runtime_snapshot_v1",
    "revalidate_causal_runtime_snapshot_v1",
    "runtime_bundle_sha256_v1",
]
