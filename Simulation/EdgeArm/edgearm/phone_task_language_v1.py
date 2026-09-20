"""Fail-closed language/task contract for phone-collected VLA episodes."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from .production_env import BLOCK_COLORS, TARGET_COLORS
from .vla_data import encode_language

if TYPE_CHECKING:
    from .generalization_task_language_v1 import GeneralizationTaskMetadataV1


PHONE_TASK_LANGUAGE_SCHEMA_VERSION = "edgearm-phone-task-language-v1"
PHONE_LANGUAGE_TOKENIZER_VERSION = "edgearm-utf8-byte-bos-eos-v1"
PHONE_LANGUAGE_MAX_TOKENS = 128


def _decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def make_phone_task_id(
    block_color: str,
    target_color: str,
    obstacle_enabled: bool,
) -> str:
    suffix = "avoid-red-obstacle" if obstacle_enabled else "clear-path"
    return f"push-{block_color}-block-to-{target_color}-target--{suffix}"


@dataclass(frozen=True)
class PhoneTaskMetadata:
    """One immutable instruction bound to one complete trajectory."""

    task_id: str
    task_text_en: str
    task_text_zh: str
    block_color: str
    target_color: str
    obstacle_enabled: bool
    schema_version: str = PHONE_TASK_LANGUAGE_SCHEMA_VERSION
    tokenizer_version: str = PHONE_LANGUAGE_TOKENIZER_VERSION
    language_max_tokens: int = PHONE_LANGUAGE_MAX_TOKENS

    def __post_init__(self) -> None:
        if self.schema_version != PHONE_TASK_LANGUAGE_SCHEMA_VERSION:
            raise ValueError("phone task language schema version mismatch")
        if self.tokenizer_version != PHONE_LANGUAGE_TOKENIZER_VERSION:
            raise ValueError("phone task tokenizer version mismatch")
        if self.language_max_tokens != PHONE_LANGUAGE_MAX_TOKENS:
            raise ValueError("phone task language length contract mismatch")
        if self.block_color not in BLOCK_COLORS:
            raise ValueError(f"unknown block color: {self.block_color!r}")
        if self.target_color not in TARGET_COLORS:
            raise ValueError(f"unknown target color: {self.target_color!r}")
        if not isinstance(self.obstacle_enabled, bool):
            raise ValueError("obstacle_enabled must be bool")
        expected_id = make_phone_task_id(
            self.block_color,
            self.target_color,
            self.obstacle_enabled,
        )
        if self.task_id != expected_id or not re.fullmatch(r"[a-z0-9-]+", self.task_id):
            raise ValueError("task_id does not match the scene contract")
        english = self.task_text_en.strip()
        chinese = self.task_text_zh.strip()
        if not english or not chinese:
            raise ValueError("both English and Chinese task text are required")
        object.__setattr__(self, "task_text_en", english)
        object.__setattr__(self, "task_text_zh", chinese)
        if self.block_color not in english.casefold():
            raise ValueError("English task text does not name the actual block color")
        if self.target_color not in english.casefold():
            raise ValueError("English task text does not name the actual target color")
        block_zh = BLOCK_COLORS[self.block_color][1]
        target_zh = TARGET_COLORS[self.target_color][1]
        if block_zh not in chinese:
            raise ValueError("Chinese task text does not name the actual block color")
        if target_zh not in chinese:
            raise ValueError("Chinese task text does not name the actual target color")
        english_mentions_obstacle = any(
            word in english.casefold() for word in ("obstacle", "barrier")
        )
        chinese_mentions_obstacle = any(word in chinese for word in ("障碍", "挡板"))
        if self.obstacle_enabled and not (
            english_mentions_obstacle and chinese_mentions_obstacle
        ):
            raise ValueError("obstacle task text must describe the obstacle in both languages")
        if not self.obstacle_enabled and (
            english_mentions_obstacle or chinese_mentions_obstacle
        ):
            raise ValueError("clear-path task text must not claim an obstacle")
        for label, text in (("English", english), ("Chinese", chinese)):
            encoded_length = len(text.encode("utf-8")) + 2
            if encoded_length > self.language_max_tokens:
                raise ValueError(
                    f"{label} task text requires {encoded_length} tokens, "
                    f"exceeding {self.language_max_tokens}"
                )

    @classmethod
    def from_environment(cls, env: Any) -> PhoneTaskMetadata:
        obstacle_enabled = bool(env.obstacle_enabled)
        block_color = str(env.color_name)
        target_color = str(env.target_color_name)
        return cls(
            task_id=make_phone_task_id(block_color, target_color, obstacle_enabled),
            task_text_en=str(env.task_text),
            task_text_zh=str(env.task_text_zh),
            block_color=block_color,
            target_color=target_color,
            obstacle_enabled=obstacle_enabled,
        )

    def contract(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def contract_json(self) -> str:
        return _canonical_json(self.contract())

    @property
    def contract_sha256(self) -> str:
        return hashlib.sha256(self.contract_json.encode("utf-8")).hexdigest()

    def encoded(self, language: str) -> tuple[np.ndarray, np.ndarray]:
        if language == "en":
            text = self.task_text_en
        elif language == "zh":
            text = self.task_text_zh
        else:
            raise ValueError("language must be 'en' or 'zh'")
        return encode_language(text, max_length=self.language_max_tokens)


def write_phone_task_metadata(
    stream: h5py.File,
    task: "PhoneTaskMetadata | GeneralizationTaskMetadataV1",
) -> None:
    """Write raw text and exact tokens, or validate an existing contract."""

    if "task_language" in stream:
        stored = read_phone_task_metadata(stream, required=True)
        if stored != task:
            raise ValueError("existing HDF5 task language contract differs")
        return
    stream.attrs.update(
        {
            "task_language_schema_version": task.schema_version,
            "task_id": task.task_id,
            "task": task.task_text_en,
            "task_zh": task.task_text_zh,
            "block_color": task.block_color,
            "target_color": task.target_color,
            "obstacle": task.obstacle_enabled,
            "language_tokenizer_version": task.tokenizer_version,
            "language_max_tokens": task.language_max_tokens,
            "task_contract_json": task.contract_json,
            "task_contract_sha256": task.contract_sha256,
            "language_condition_present": True,
        }
    )
    group = stream.create_group("task_language")
    for language in ("en", "zh"):
        tokens, mask = task.encoded(language)
        group.create_dataset(f"tokens_{language}", data=tokens, dtype=np.int64)
        group.create_dataset(f"mask_{language}", data=mask, dtype=np.bool_)


def read_phone_task_metadata(
    stream: h5py.File,
    *,
    required: bool,
) -> "PhoneTaskMetadata | GeneralizationTaskMetadataV1 | None":
    schema = _decode(stream.attrs.get("task_language_schema_version", ""))
    if not schema:
        if required:
            raise ValueError("phone episode has no language task contract")
        return None
    payload_text = _decode(stream.attrs.get("task_contract_json", ""))
    if not payload_text:
        raise ValueError("phone episode task contract JSON is missing")
    expected_hash = _decode(stream.attrs.get("task_contract_sha256", ""))
    observed_hash = hashlib.sha256(payload_text.encode("utf-8")).hexdigest()
    if expected_hash != observed_hash:
        raise ValueError("phone episode task contract hash mismatch")
    payload = json.loads(payload_text)
    if schema == PHONE_TASK_LANGUAGE_SCHEMA_VERSION:
        task: PhoneTaskMetadata | GeneralizationTaskMetadataV1 = PhoneTaskMetadata(
            **payload
        )
    else:
        from .generalization_task_language_v1 import (
            GENERALIZATION_TASK_LANGUAGE_SCHEMA_VERSION,
            GeneralizationTaskMetadataV1,
        )

        if schema != GENERALIZATION_TASK_LANGUAGE_SCHEMA_VERSION:
            raise ValueError(f"unsupported task language schema: {schema}")
        payload["obstacle_shapes"] = tuple(payload.get("obstacle_shapes", ()))
        task = GeneralizationTaskMetadataV1(**payload)
    if task.schema_version != schema:
        raise ValueError("phone episode task schema attribute differs from contract")
    scalar_checks = {
        "task_id": task.task_id,
        "task": task.task_text_en,
        "task_zh": task.task_text_zh,
        "block_color": task.block_color,
        "target_color": task.target_color,
        "language_tokenizer_version": task.tokenizer_version,
    }
    for name, expected in scalar_checks.items():
        if _decode(stream.attrs.get(name, "")) != expected:
            raise ValueError(f"phone episode {name} attribute differs from task contract")
    if bool(stream.attrs.get("obstacle", not task.obstacle_enabled)) != task.obstacle_enabled:
        raise ValueError("phone episode obstacle attribute differs from task contract")
    if int(stream.attrs.get("language_max_tokens", -1)) != task.language_max_tokens:
        raise ValueError("phone episode language length differs from task contract")
    if not bool(stream.attrs.get("language_condition_present", False)):
        raise ValueError("phone episode denies its stored language condition")
    if "task_language" not in stream:
        raise ValueError("phone episode token group is missing")
    group = stream["task_language"]
    for language in ("en", "zh"):
        expected_tokens, expected_mask = task.encoded(language)
        token_name = f"tokens_{language}"
        mask_name = f"mask_{language}"
        if token_name not in group or mask_name not in group:
            raise ValueError(f"phone episode {language} token payload is missing")
        np.testing.assert_array_equal(group[token_name][:], expected_tokens)
        np.testing.assert_array_equal(group[mask_name][:], expected_mask)
    return task


__all__ = [
    "PHONE_LANGUAGE_MAX_TOKENS",
    "PHONE_LANGUAGE_TOKENIZER_VERSION",
    "PHONE_TASK_LANGUAGE_SCHEMA_VERSION",
    "PhoneTaskMetadata",
    "make_phone_task_id",
    "read_phone_task_metadata",
    "write_phone_task_metadata",
]
