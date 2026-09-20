"""Pure source-neutral constants shared by V11 and scratch schema profiles.

This module contains no producer, checkpoint, replay, policy-provider, trainer,
or collector import.  Keeping the shared schema payload here prevents the V11
collector's frozen import closure from acquiring scratch implementation code.
"""

from __future__ import annotations

from .causal_4d_act_v1 import CAUSAL_4D_ACT_V1_INPUT_KEYS
from .causal_collection_execution_core_v1 import SOURCE_SCHEMA_PROVENANCE_FIELDS
from .causal_v2_data import POLICY_INPUT_DATASETS
from .trajectory_contract_v1 import (
    CAUSAL_V2_EPISODE_ATTRS,
    CAUSAL_V2_EXECUTION_DATASETS,
    CAUSAL_V2_ROOT_ATTRS,
)


COMMON_POLICY_INPUT_KEYS = tuple(sorted(CAUSAL_4D_ACT_V1_INPUT_KEYS))
COMMON_POLICY_INPUT_DATASETS = tuple(sorted(POLICY_INPUT_DATASETS))
COMMON_EXECUTION_DATASETS = tuple(CAUSAL_V2_EXECUTION_DATASETS)
COMMON_ROOT_ATTRIBUTES = tuple(CAUSAL_V2_ROOT_ATTRS)
COMMON_EPISODE_ATTRIBUTES = tuple(CAUSAL_V2_EPISODE_ATTRS)
COMMON_SOURCE_SCHEMA_PROVENANCE_FIELDS = SOURCE_SCHEMA_PROVENANCE_FIELDS

COMMON_MANIFEST_FIELDS = (
    "episode_index",
    "shard",
    "group",
    "frames",
    "seed",
    "strict_success",
    "action_label_eligible",
    "episode_integrity_pass",
    "action_admission_pass",
    "action_admission_gate_format",
    "failure_code",
    "unique_live_episode_key",
    "dataset_format",
    "task_scope",
    "trajectory_origin",
    "collection_profile_hash",
    "source_type",
    "domain",
)


def unique_names_v1(*groups: tuple[str, ...]) -> tuple[str, ...]:
    """Return first-occurrence-ordered unique schema names."""

    return tuple(dict.fromkeys(value for group in groups for value in group))


__all__ = [
    "COMMON_EPISODE_ATTRIBUTES",
    "COMMON_EXECUTION_DATASETS",
    "COMMON_MANIFEST_FIELDS",
    "COMMON_POLICY_INPUT_DATASETS",
    "COMMON_POLICY_INPUT_KEYS",
    "COMMON_ROOT_ATTRIBUTES",
    "COMMON_SOURCE_SCHEMA_PROVENANCE_FIELDS",
    "unique_names_v1",
]
