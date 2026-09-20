"""Transitive Python-source provenance for V26 scratch-RL collection.

The historical run plan hashed a short hand-maintained list.  V26 walks every
relative ``edgearm`` import reachable from the collection entrypoint and the
behavior-critical roots.  Training recomputes the graph and requires exact
path and byte-hash identity before consuming a rollout.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any


BEHAVIOR_SOURCE_CLOSURE_FORMAT_V26 = "edgearm-v26-transitive-python-source-closure-v1"
BEHAVIOR_SOURCE_ROOTS_V26 = (
    "collect_feasible_multiview_rollout_v22.py",
    "asymmetric_multiview_ppo_v1.py",
    "stock_gripper_taskframe_v22.py",
    "stock_gripper_rollout_kernel_v22.py",
    "stock_gripper_reward_v22.py",
    "sim2real_env_v10.py",
    "v23_checkpoint_lineage.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_candidates(path: Path, package_root: Path) -> set[Path]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    module_parts = list(path.relative_to(package_root).with_suffix("").parts)
    candidates: set[Path] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level < 1:
            continue
        ascend = node.level - 1
        if ascend > len(module_parts) - 1:
            raise ValueError(f"V26 relative import escapes edgearm package: {path}")
        base = module_parts[: len(module_parts) - 1 - ascend]
        imported = node.module.split(".") if node.module else []
        target_parts = [*base, *imported]
        if target_parts:
            module_path = package_root.joinpath(*target_parts).with_suffix(".py")
            package_path = package_root.joinpath(*target_parts, "__init__.py")
            if module_path.is_file():
                candidates.add(module_path.resolve())
            elif package_path.is_file():
                candidates.add(package_path.resolve())
        if node.module is None:
            for alias in node.names:
                alias_path = package_root.joinpath(*base, alias.name).with_suffix(".py")
                if alias_path.is_file():
                    candidates.add(alias_path.resolve())
    return candidates


def build_behavior_source_closure_v26(package_root: Path) -> dict[str, Any]:
    root = Path(package_root).expanduser().resolve()
    if root.name != "edgearm" or not root.is_dir():
        raise ValueError("V26 source closure requires the edgearm package directory")
    pending = [root / name for name in BEHAVIOR_SOURCE_ROOTS_V26]
    own_path = Path(__file__).resolve()
    pending.append(own_path)
    visited: set[Path] = set()
    while pending:
        current = pending.pop().resolve()
        if current in visited:
            continue
        if not current.is_file() or current.suffix != ".py":
            raise FileNotFoundError(f"V26 source closure root/import is missing: {current}")
        if root not in current.parents:
            raise ValueError(f"V26 source closure escaped package root: {current}")
        visited.add(current)
        pending.extend(_relative_candidates(current, root) - visited)
    files = {
        "edgearm/" + str(path.relative_to(root)): _sha256(path)
        for path in sorted(visited)
    }
    return {
        "format": BEHAVIOR_SOURCE_CLOSURE_FORMAT_V26,
        "root_files": ["edgearm/" + name for name in BEHAVIOR_SOURCE_ROOTS_V26],
        "file_count": len(files),
        "files": files,
    }


def verify_behavior_source_closure_v26(
    recorded: dict[str, Any],
    package_root: Path,
) -> dict[str, str]:
    if type(recorded) is not dict:
        raise TypeError("V26 recorded source closure must be a dictionary")
    if recorded.get("format") != BEHAVIOR_SOURCE_CLOSURE_FORMAT_V26:
        raise ValueError("V26 source closure format changed")
    current = build_behavior_source_closure_v26(package_root)
    if recorded.get("root_files") != current["root_files"]:
        raise ValueError("V26 source closure roots changed")
    if recorded.get("file_count") != current["file_count"]:
        raise ValueError("V26 source closure file count changed")
    recorded_files = recorded.get("files")
    if recorded_files != current["files"]:
        raise ValueError("V26 behavior source graph or bytes changed after collection")
    return dict(current["files"])


__all__ = [
    "BEHAVIOR_SOURCE_CLOSURE_FORMAT_V26",
    "BEHAVIOR_SOURCE_ROOTS_V26",
    "build_behavior_source_closure_v26",
    "verify_behavior_source_closure_v26",
]
