"""Crash-safe file commits for large EdgeArm datasets on external APFS disks."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Iterator


EXTERNAL_EPISODE_STORE_VERSION = "edgearm-external-episode-store-v1"


def _json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class CommittedFileV1:
    relative_path: str
    kind: str
    bytes: int
    sha256: str
    attempt_index: int
    episode_seed: int
    committed_unix_ns: int
    store_version: str = EXTERNAL_EPISODE_STORE_VERSION

    def contract(self) -> dict[str, Any]:
        return {
            "store_version": self.store_version,
            "relative_path": self.relative_path,
            "kind": self.kind,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "attempt_index": self.attempt_index,
            "episode_seed": self.episode_seed,
            "committed_unix_ns": self.committed_unix_ns,
        }


class ExternalEpisodeStoreV1:
    """Append-only manifest plus atomic per-file commit protocol."""

    def __init__(self, root: Path, *, minimum_free_gib: float = 200.0) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_absolute():
            raise ValueError("external episode root must be absolute")
        if not isinstance(minimum_free_gib, (int, float)) or minimum_free_gib < 1.0:
            raise ValueError("minimum_free_gib must be at least 1")
        self.minimum_free_bytes = int(float(minimum_free_gib) * 1024**3)
        self.ledger_path = self.root / "file_commits.jsonl"

    def initialize(self, *, resume: bool) -> None:
        if self.root.exists():
            if not self.root.is_dir() or self.root.is_symlink():
                raise ValueError("external episode root must be a real directory")
            if not resume and any(self.root.iterdir()):
                raise FileExistsError(f"output root is not empty; use --resume: {self.root}")
        else:
            self.root.mkdir(parents=True)
        for name in ("motion", "rgbd", "manifests"):
            (self.root / name).mkdir(exist_ok=True)
        stale_partials = sorted(self.root.rglob("*.partial"))
        if stale_partials:
            listing = ", ".join(str(path) for path in stale_partials[:5])
            raise RuntimeError(f"incomplete .partial files require inspection before resume: {listing}")
        self.assert_capacity()
        _fsync_directory(self.root)

    def assert_capacity(self) -> int:
        free = shutil.disk_usage(self.root).free
        if free < self.minimum_free_bytes:
            raise RuntimeError(
                f"external dataset disk floor reached: {free / 1024**3:.1f} GiB free, "
                f"requires {self.minimum_free_bytes / 1024**3:.1f} GiB"
            )
        return free

    def _inside_root(self, path: Path) -> Path:
        resolved = Path(path).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as error:
            raise ValueError("episode path escapes the configured output root") from error
        return resolved

    def stage_path(self, final_path: Path) -> Path:
        final = self._inside_root(final_path)
        if final.exists():
            raise FileExistsError(final)
        final.parent.mkdir(parents=True, exist_ok=True)
        staging = final.with_name(final.name + ".partial")
        if staging.exists():
            raise FileExistsError(f"stale partial file requires explicit inspection before retry: {staging}")
        return staging

    def commit(
        self,
        staging_path: Path,
        final_path: Path,
        *,
        kind: str,
        attempt_index: int,
        episode_seed: int,
    ) -> CommittedFileV1:
        self.assert_capacity()
        staging = self._inside_root(staging_path)
        final = self._inside_root(final_path)
        if staging != final.with_name(final.name + ".partial"):
            raise ValueError("staging path does not match the final .partial contract")
        if not staging.is_file() or staging.is_symlink():
            raise FileNotFoundError(staging)
        if final.exists():
            raise FileExistsError(final)
        with staging.open("rb") as stream:
            os.fsync(stream.fileno())
        size = staging.stat().st_size
        digest = _sha256_file(staging)
        os.replace(staging, final)
        _fsync_directory(final.parent)
        record = CommittedFileV1(
            relative_path=final.relative_to(self.root).as_posix(),
            kind=str(kind),
            bytes=size,
            sha256=digest,
            attempt_index=int(attempt_index),
            episode_seed=int(episode_seed),
            committed_unix_ns=time.time_ns(),
        )
        with self.ledger_path.open("a", encoding="utf-8") as stream:
            stream.write(_json_text(record.contract()) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(self.root)
        return record

    def records(self) -> Iterator[CommittedFileV1]:
        if not self.ledger_path.exists():
            return
        with self.ledger_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"non-object commit row at line {line_number}")
                yield CommittedFileV1(**payload)

    def verify_commits(self, *, verify_hashes: bool) -> list[CommittedFileV1]:
        records = list(self.records())
        seen: set[str] = set()
        for record in records:
            if record.relative_path in seen:
                raise ValueError(f"duplicate committed path: {record.relative_path}")
            seen.add(record.relative_path)
            path = self._inside_root(self.root / record.relative_path)
            if not path.is_file() or path.is_symlink():
                raise FileNotFoundError(path)
            if path.stat().st_size != record.bytes:
                raise ValueError(f"committed file size mismatch: {path}")
            if verify_hashes and _sha256_file(path) != record.sha256:
                raise ValueError(f"committed file hash mismatch: {path}")
        return records


__all__ = [
    "CommittedFileV1",
    "EXTERNAL_EPISODE_STORE_VERSION",
    "ExternalEpisodeStoreV1",
]
