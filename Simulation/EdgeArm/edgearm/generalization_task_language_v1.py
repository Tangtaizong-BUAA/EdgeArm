"""Language contract bound to one exact V13 generalization scene."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Any

import numpy as np

from .generalization_scene_v1 import GeneralizationScenarioV1, OBSTACLE_SHAPES, OBJECT_SHAPES
from .production_env import BLOCK_COLORS, TARGET_COLORS
from .vla_data import encode_language


GENERALIZATION_TASK_LANGUAGE_SCHEMA_VERSION = "edgearm-generalization-task-language-v1"
GENERALIZATION_LANGUAGE_TOKENIZER_VERSION = "edgearm-utf8-byte-bos-eos-v1"
GENERALIZATION_LANGUAGE_MAX_TOKENS = 192

_SHAPE_EN = {"box": "box", "cylinder": "cylinder", "ellipsoid": "ellipsoid"}
_SHAPE_ZH = {"box": "长方体", "cylinder": "圆柱体", "ellipsoid": "椭球体"}
_LAYOUT_EN = {
    "direct_path": "on the direct path",
    "gate": "as a gate",
    "slalom": "in a slalom",
    "target_edge": "at the target edge",
    "target_clutter": "inside the target area",
}
_LAYOUT_ZH = {
    "direct_path": "位于直线路径上的",
    "gate": "构成门形通道的",
    "slalom": "交错排列的",
    "target_edge": "位于目标区边缘的",
    "target_clutter": "位于目标区内的",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class GeneralizationTaskMetadataV1:
    task_id: str
    task_text_en: str
    task_text_zh: str
    block_color: str
    target_color: str
    object_shape: str
    obstacle_layout: str
    obstacle_shapes: tuple[str, ...]
    scene_contract_sha256: str
    schema_version: str = GENERALIZATION_TASK_LANGUAGE_SCHEMA_VERSION
    tokenizer_version: str = GENERALIZATION_LANGUAGE_TOKENIZER_VERSION
    language_max_tokens: int = GENERALIZATION_LANGUAGE_MAX_TOKENS

    def __post_init__(self) -> None:
        if self.schema_version != GENERALIZATION_TASK_LANGUAGE_SCHEMA_VERSION:
            raise ValueError("generalization task language schema mismatch")
        if self.tokenizer_version != GENERALIZATION_LANGUAGE_TOKENIZER_VERSION:
            raise ValueError("generalization task tokenizer mismatch")
        if self.language_max_tokens != GENERALIZATION_LANGUAGE_MAX_TOKENS:
            raise ValueError("generalization task token length mismatch")
        if self.block_color not in BLOCK_COLORS or self.target_color not in TARGET_COLORS:
            raise ValueError("generalization task contains an unknown color")
        if self.object_shape not in OBJECT_SHAPES:
            raise ValueError("generalization task contains an unknown object shape")
        obstacle_shapes = tuple(str(item) for item in self.obstacle_shapes)
        if any(item not in OBSTACLE_SHAPES for item in obstacle_shapes):
            raise ValueError("generalization task contains an unknown obstacle shape")
        object.__setattr__(self, "obstacle_shapes", obstacle_shapes)
        if self.obstacle_layout == "clear" and obstacle_shapes:
            raise ValueError("clear task cannot contain obstacle shapes")
        if self.obstacle_layout != "clear" and not obstacle_shapes:
            raise ValueError("obstacle task requires obstacle shapes")
        if not re.fullmatch(r"[0-9a-f]{64}", self.scene_contract_sha256):
            raise ValueError("scene contract SHA-256 is invalid")
        if not re.fullmatch(r"[a-z0-9-]+", self.task_id):
            raise ValueError("generalization task_id is not canonical")
        english = self.task_text_en.strip()
        chinese = self.task_text_zh.strip()
        if not english or not chinese:
            raise ValueError("both task languages are required")
        object.__setattr__(self, "task_text_en", english)
        object.__setattr__(self, "task_text_zh", chinese)
        if self.block_color not in english.casefold() or self.target_color not in english.casefold():
            raise ValueError("English task text does not name the realized colors")
        if _SHAPE_EN[self.object_shape] not in english.casefold():
            raise ValueError("English task text does not name the manipulated-object shape")
        if (
            BLOCK_COLORS[self.block_color][1] not in chinese
            or TARGET_COLORS[self.target_color][1] not in chinese
        ):
            raise ValueError("Chinese task text does not name the realized colors")
        if _SHAPE_ZH[self.object_shape] not in chinese:
            raise ValueError("Chinese task text does not name the manipulated-object shape")
        mentions_en = "obstacle" in english.casefold()
        mentions_zh = "障碍" in chinese
        if self.obstacle_enabled != (mentions_en and mentions_zh):
            raise ValueError("task text obstacle claim differs from the realized scene")
        for label, text in (("English", english), ("Chinese", chinese)):
            length = len(text.encode("utf-8")) + 2
            if length > self.language_max_tokens:
                raise ValueError(f"{label} task requires {length} tokens")

    @property
    def obstacle_enabled(self) -> bool:
        return bool(self.obstacle_shapes)

    @property
    def obstacle_count(self) -> int:
        return len(self.obstacle_shapes)

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

    @classmethod
    def from_scenario(cls, scenario: GeneralizationScenarioV1) -> "GeneralizationTaskMetadataV1":
        shape_en = _SHAPE_EN[scenario.object_shape]
        shape_zh = _SHAPE_ZH[scenario.object_shape]
        block_zh = BLOCK_COLORS[scenario.block_color][1]
        target_zh = TARGET_COLORS[scenario.target_color][1]
        if scenario.obstacle_count:
            english = (
                f"Push the {scenario.block_color} {shape_en} into the "
                f"{scenario.target_color} target while avoiding {scenario.obstacle_count} "
                f"obstacles {_LAYOUT_EN[scenario.obstacle_layout]}."
            )
            chinese = (
                f"避开{_LAYOUT_ZH[scenario.obstacle_layout]}{scenario.obstacle_count}个障碍物，"
                f"把{block_zh}{shape_zh}推入{target_zh}目标区。"
            )
            suffix = f"{scenario.obstacle_layout.replace('_', '-')}-{scenario.obstacle_count}-obstacles"
        else:
            english = (
                f"Push the {scenario.block_color} {shape_en} into the {scenario.target_color} target zone."
            )
            chinese = f"把{block_zh}{shape_zh}推入{target_zh}目标区。"
            suffix = "clear-path"
        return cls(
            task_id=(
                f"push-{scenario.block_color}-{scenario.object_shape}-to-"
                f"{scenario.target_color}-target--{suffix}"
            ),
            task_text_en=english,
            task_text_zh=chinese,
            block_color=scenario.block_color,
            target_color=scenario.target_color,
            object_shape=scenario.object_shape,
            obstacle_layout=scenario.obstacle_layout,
            obstacle_shapes=tuple(item.shape for item in scenario.obstacles),
            scene_contract_sha256=scenario.contract_sha256,
        )


__all__ = [
    "GENERALIZATION_LANGUAGE_MAX_TOKENS",
    "GENERALIZATION_LANGUAGE_TOKENIZER_VERSION",
    "GENERALIZATION_TASK_LANGUAGE_SCHEMA_VERSION",
    "GeneralizationTaskMetadataV1",
]
