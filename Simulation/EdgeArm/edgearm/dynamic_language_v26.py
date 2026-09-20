"""Semantically validated per-episode and per-row language for V26 VLA data."""

from __future__ import annotations

import json
from typing import Any

import h5py
import numpy as np

from .production_env import BLOCK_COLORS, TARGET_COLORS


VLA_LANGUAGE_DATASET_FORMAT_V26 = "edgearm-v26-dynamic-per-episode-language-v1"


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def canonical_task_language_v26(record: dict[str, Any]) -> tuple[str, str]:
    domain = record.get("episode_domain")
    if type(domain) is not dict:
        raise TypeError("V26 episode record lost domain randomization")
    block = domain.get("block_color")
    target = domain.get("target_color")
    obstacle = domain.get("obstacle")
    if block not in BLOCK_COLORS or target not in TARGET_COLORS or type(obstacle) is not bool:
        raise ValueError("V26 episode language cannot resolve color/obstacle semantics")
    block_zh = BLOCK_COLORS[block][1]
    target_zh = TARGET_COLORS[target][1]
    if obstacle:
        return (
            f"Push the {block} block into the {target} target zone while avoiding the red obstacle.",
            f"避开红色障碍物，把{block_zh}方块推入{target_zh}目标区。",
        )
    return (
        f"Push the {block} block into the {target} target zone.",
        f"把{block_zh}方块推入{target_zh}目标区。",
    )


def _expected_language(stream: h5py.File) -> tuple[list[str], list[str], np.ndarray]:
    if "episode_randomization_json" not in stream:
        raise RuntimeError("V26 language persistence lost episode randomization")
    execution = stream.get("execution")
    if not isinstance(execution, h5py.Group) or "episode_ids" not in execution:
        raise RuntimeError("V26 language persistence lost row episode identity")
    records = [
        json.loads(_decode(value))
        for value in stream["episode_randomization_json"][...]
    ]
    if any(type(record) is not dict for record in records):
        raise TypeError("V26 episode randomization row is not a dictionary")
    by_episode: dict[int, tuple[str, str]] = {}
    for record in records:
        episode_id = record.get("episode_id")
        if type(episode_id) is not int or episode_id in by_episode:
            raise ValueError("V26 episode language identity is invalid or duplicated")
        by_episode[episode_id] = canonical_task_language_v26(record)
    if set(by_episode) != set(range(len(records))):
        raise ValueError("V26 episode language identity is not contiguous from zero")
    row_episode_ids = np.asarray(execution["episode_ids"][...], dtype=np.int64)
    if row_episode_ids.ndim != 1 or any(int(value) not in by_episode for value in row_episode_ids):
        raise ValueError("V26 row language references an unknown episode")
    episode_en = [by_episode[index][0] for index in range(len(records))]
    episode_zh = [by_episode[index][1] for index in range(len(records))]
    return episode_en, episode_zh, row_episode_ids


def persist_dynamic_language_v26(stream: h5py.File) -> dict[str, Any]:
    episode_en, episode_zh, row_episode_ids = _expected_language(stream)
    if "language_v26" in stream:
        raise FileExistsError("V26 rollout already contains a language group")
    row_en = [episode_en[int(index)] for index in row_episode_ids]
    row_zh = [episode_zh[int(index)] for index in row_episode_ids]
    language = stream.create_group("language_v26")
    for name, values in (
        ("episode_instruction_en", episode_en),
        ("episode_instruction_zh", episode_zh),
        ("row_instruction_en", row_en),
        ("row_instruction_zh", row_zh),
    ):
        dataset = language.create_dataset(
            name,
            data=np.asarray(values, dtype=h5py.string_dtype("utf-8")),
        )
        dataset.attrs["policy_input_eligible"] = False
        dataset.attrs["future_vla_input_eligible"] = True
    language.create_dataset("row_episode_ids", data=row_episode_ids)
    language.attrs.update(
        {
            "format": VLA_LANGUAGE_DATASET_FORMAT_V26,
            "semantic_source": "episode_domain_block_target_color_and_obstacle",
            "current_scratch_rl_actor_consumes_language": False,
            "future_vla_policy_consumes_language": True,
            "semantics_validated": True,
        }
    )
    for stale_name in ("task_instruction_en", "task_instruction_zh"):
        if stale_name in stream.attrs:
            del stream.attrs[stale_name]
    stream.attrs.update(
        {
            "task_instruction_mode": "dynamic_per_episode_and_row",
            "task_instruction_group": "/language_v26",
            "task_instruction_format": VLA_LANGUAGE_DATASET_FORMAT_V26,
            "task_instruction_semantics_validated": True,
        }
    )
    return verify_dynamic_language_v26(stream)


def verify_dynamic_language_v26(stream: h5py.File) -> dict[str, Any]:
    episode_en, episode_zh, row_episode_ids = _expected_language(stream)
    language = stream.get("language_v26")
    if not isinstance(language, h5py.Group):
        raise RuntimeError("V26 rollout lost its dynamic language group")
    if _decode(language.attrs.get("format")) != VLA_LANGUAGE_DATASET_FORMAT_V26:
        raise ValueError("V26 language format changed")
    if not bool(language.attrs.get("semantics_validated", False)):
        raise ValueError("V26 language does not claim validated semantics")
    expected = {
        "episode_instruction_en": episode_en,
        "episode_instruction_zh": episode_zh,
        "row_instruction_en": [episode_en[int(index)] for index in row_episode_ids],
        "row_instruction_zh": [episode_zh[int(index)] for index in row_episode_ids],
    }
    for name, values in expected.items():
        if name not in language:
            raise RuntimeError(f"V26 rollout lost language dataset: {name}")
        actual = [_decode(value) for value in language[name][...]]
        if actual != values:
            raise ValueError(f"V26 language semantics changed: {name}")
        if bool(language[name].attrs.get("policy_input_eligible", True)):
            raise ValueError("V26 scratch RL must not consume language as an actor input")
        if not bool(language[name].attrs.get("future_vla_input_eligible", False)):
            raise ValueError("V26 language is not marked for future VLA input")
    if not np.array_equal(language["row_episode_ids"][...], row_episode_ids):
        raise ValueError("V26 language row/episode identity changed")
    if any(name in stream.attrs for name in ("task_instruction_en", "task_instruction_zh")):
        raise ValueError("V26 rollout retained a stale global task instruction")
    return {
        "format": VLA_LANGUAGE_DATASET_FORMAT_V26,
        "episode_count": len(episode_en),
        "row_count": len(row_episode_ids),
        "unique_english_instruction_count": len(set(episode_en)),
        "unique_chinese_instruction_count": len(set(episode_zh)),
        "semantics_validated": True,
        "current_scratch_rl_actor_consumes_language": False,
        "future_vla_input_eligible": True,
    }


__all__ = [
    "VLA_LANGUAGE_DATASET_FORMAT_V26",
    "canonical_task_language_v26",
    "persist_dynamic_language_v26",
    "verify_dynamic_language_v26",
]
