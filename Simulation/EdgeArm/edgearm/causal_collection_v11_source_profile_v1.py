"""V11-only causal collection schema and immutable producer preparation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import PurePosixPath
import re
from typing import Any

from .action_admission_v1 import (
    GENERIC_ACTION_LABEL_DATASET,
    V11_SOURCE_TYPE,
)
from .causal_collection_execution_core_v1 import (
    H5RowDatasetContractV1,
    SourceCollectionPreparationV1,
    SourceSchemaProfileV1,
)
from .causal_collection_source_profile_common_v1 import (
    COMMON_EPISODE_ATTRIBUTES,
    COMMON_EXECUTION_DATASETS,
    COMMON_MANIFEST_FIELDS,
    COMMON_POLICY_INPUT_DATASETS,
    COMMON_POLICY_INPUT_KEYS,
    COMMON_ROOT_ATTRIBUTES,
    COMMON_SOURCE_SCHEMA_PROVENANCE_FIELDS,
    unique_names_v1,
)
from .trajectory_contract_v1 import (
    CAUSAL_V2_V11_EPISODE_ATTRS,
    CAUSAL_V2_V11_ROOT_ATTRS,
)


V11_POLICY_SOURCE_ID = 1
V11_RAW_POLICY_ACTION_DATASET = "teacher_reference_action"
V11_POLICY_SOURCE_ID_DATASET = "teacher_source_id"
V11_PRODUCER_BINDING_FIELDS = (
    "producer_artifact_kind",
    "producer_artifact_manifest_path",
    "producer_artifact_manifest_sha256",
    "runtime_asset_sha256",
    "producer_runtime_closure_sha256",
    "policy_artifact_sha256",
)
V11_COLLECTION_EXECUTION_BINDING_FIELDS = ("collection_execution_evidence_level",)
V11_EPISODE_REQUEST_EVIDENCE_FIELDS = ("frozen_worker_request_sha256",)
V11_SOURCE_BINDING_FIELDS = (
    *V11_PRODUCER_BINDING_FIELDS,
    *V11_COLLECTION_EXECUTION_BINDING_FIELDS,
)
V11_FROZEN_WORKER_SYNTHETIC_EVIDENCE_LEVEL = "frozen_worker_synthetic"
V11_IN_PROCESS_SYNTHETIC_EVIDENCE_LEVEL = "in_process_smoke_synthetic"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_V11_DECISION_CONTRACTS = (
    H5RowDatasetContractV1(V11_RAW_POLICY_ACTION_DATASET, "float32", (6,)),
    H5RowDatasetContractV1("teacher_safe_action", "float32", (6,)),
    H5RowDatasetContractV1("teacher_requested_joint_target", "float32", (6,)),
    H5RowDatasetContractV1("teacher_queued_safe_joint_target", "float32", (6,)),
    H5RowDatasetContractV1("teacher_safety_clipped", "uint8", ()),
    H5RowDatasetContractV1(
        "diagnostic_teacher_global_joint_target",
        "float32",
        (6,),
    ),
    H5RowDatasetContractV1(
        "diagnostic_teacher_global_joint_target_valid",
        "uint8",
        (),
    ),
    H5RowDatasetContractV1("diagnostic_teacher_planned_phase_id", "uint8", ()),
    H5RowDatasetContractV1(
        "diagnostic_teacher_precontact_plan_feasible",
        "uint8",
        (),
    ),
    H5RowDatasetContractV1(
        V11_POLICY_SOURCE_ID_DATASET,
        "uint8",
        (),
        constant_scalar=V11_POLICY_SOURCE_ID,
    ),
)
_V11_EFFECT_CONTRACTS = (
    H5RowDatasetContractV1(
        "diagnostic_teacher_precontact_completion_event",
        "uint8",
        (),
    ),
    H5RowDatasetContractV1(
        "diagnostic_teacher_precontact_completed_once",
        "uint8",
        (),
    ),
    H5RowDatasetContractV1(
        "effect_applied_parent_precontact_completed_once",
        "uint8",
        (),
    ),
    H5RowDatasetContractV1("effect_causal_valid_push_transition", "uint8", ()),
)


V11_SOURCE_SCHEMA_PROFILE_V1 = SourceSchemaProfileV1(
    source_type=V11_SOURCE_TYPE,
    source_numeric_id=V11_POLICY_SOURCE_ID,
    controller_identity="physical_expert_v11_via_sim2real_v2_queue",
    controller_version="edgearm-privileged-physical-expert-v11",
    primary_action_label_dataset=GENERIC_ACTION_LABEL_DATASET,
    raw_policy_action_dataset=V11_RAW_POLICY_ACTION_DATASET,
    policy_source_id_dataset=V11_POLICY_SOURCE_ID_DATASET,
    policy_input_keys=COMMON_POLICY_INPUT_KEYS,
    policy_input_datasets=COMMON_POLICY_INPUT_DATASETS,
    required_common_datasets=COMMON_EXECUTION_DATASETS,
    decision_dataset_contracts=_V11_DECISION_CONTRACTS,
    effect_dataset_contracts=_V11_EFFECT_CONTRACTS,
    required_root_attributes=unique_names_v1(
        COMMON_ROOT_ATTRIBUTES,
        tuple(CAUSAL_V2_V11_ROOT_ATTRS),
        ("source_type",),
        COMMON_SOURCE_SCHEMA_PROVENANCE_FIELDS,
        V11_SOURCE_BINDING_FIELDS,
    ),
    required_episode_attributes=unique_names_v1(
        COMMON_EPISODE_ATTRIBUTES,
        tuple(CAUSAL_V2_V11_EPISODE_ATTRS),
        ("source_type",),
        COMMON_SOURCE_SCHEMA_PROVENANCE_FIELDS,
        V11_SOURCE_BINDING_FIELDS,
        V11_EPISODE_REQUEST_EVIDENCE_FIELDS,
    ),
    required_manifest_fields=unique_names_v1(
        COMMON_MANIFEST_FIELDS,
        COMMON_SOURCE_SCHEMA_PROVENANCE_FIELDS,
        V11_SOURCE_BINDING_FIELDS,
        V11_EPISODE_REQUEST_EVIDENCE_FIELDS,
    ),
    root_binding_fields=V11_SOURCE_BINDING_FIELDS,
    episode_binding_fields=V11_SOURCE_BINDING_FIELDS,
    manifest_binding_fields=V11_SOURCE_BINDING_FIELDS,
    exact_root_attribute_values=(
        ("source_type", V11_SOURCE_TYPE),
        ("act_primary_label_dataset", GENERIC_ACTION_LABEL_DATASET),
    ),
    exact_episode_attribute_values=(
        ("source_type", V11_SOURCE_TYPE),
        ("act_primary_label_dataset", GENERIC_ACTION_LABEL_DATASET),
    ),
    exact_manifest_field_values=(("source_type", V11_SOURCE_TYPE),),
    safe_action_diagnostic_alias_dataset="teacher_safe_action",
)
V11_SOURCE_SCHEMA_PROFILE_SHA256_V1 = V11_SOURCE_SCHEMA_PROFILE_V1.sha256_v1()


def build_v11_source_collection_preparation_v1(
    producer_binding: Mapping[str, Any],
) -> SourceCollectionPreparationV1:
    """Bind the exact V11 producer artifact to all collection output levels."""

    if not isinstance(producer_binding, Mapping):
        raise ValueError("V11 source binding must be a mapping")
    supplied_fields = set(producer_binding)
    if supplied_fields == set(V11_PRODUCER_BINDING_FIELDS):
        binding = {
            **dict(producer_binding),
            "collection_execution_evidence_level": (V11_IN_PROCESS_SYNTHETIC_EVIDENCE_LEVEL),
        }
    elif supplied_fields == set(V11_SOURCE_BINDING_FIELDS):
        binding = dict(producer_binding)
    else:
        raise ValueError("V11 producer binding fields differ from the source profile")
    if binding["producer_artifact_kind"] != "v11_controller_bundle_manifest":
        raise ValueError("V11 producer artifact kind differs from the source profile")
    path = binding["producer_artifact_manifest_path"]
    pure_path = PurePosixPath(path) if isinstance(path, str) and path else None
    if (
        pure_path is None
        or pure_path.is_absolute()
        or "\\" in path
        or pure_path.as_posix() != path
        or any(part in {"", ".", ".."} for part in pure_path.parts)
    ):
        raise ValueError("V11 producer artifact manifest path is not canonical relative POSIX")
    for name in (
        "producer_artifact_manifest_sha256",
        "runtime_asset_sha256",
        "producer_runtime_closure_sha256",
        "policy_artifact_sha256",
    ):
        if not isinstance(binding[name], str) or _SHA256_RE.fullmatch(binding[name]) is None:
            raise ValueError(f"V11 producer {name} must be a lowercase SHA-256")
    evidence_level = binding["collection_execution_evidence_level"]
    if evidence_level not in {
        V11_FROZEN_WORKER_SYNTHETIC_EVIDENCE_LEVEL,
        V11_IN_PROCESS_SYNTHETIC_EVIDENCE_LEVEL,
    }:
        raise ValueError("V11 collection execution evidence level is unsupported")
    profile = V11_SOURCE_SCHEMA_PROFILE_V1
    preparation = SourceCollectionPreparationV1(
        source_type=profile.source_type,
        collection_profile_components={
            "source_schema_profile_format": profile.format,
            "source_schema_profile_sha256": profile.sha256_v1(),
            "source_type": profile.source_type,
            "controller_identity": profile.controller_identity,
            "controller_version": profile.controller_version,
        },
        root_attributes=binding,
        manifest_fields=binding.copy(),
    )
    preparation.validate_for_profile_v1(profile)
    return preparation


__all__ = [
    "V11_POLICY_SOURCE_ID",
    "V11_POLICY_SOURCE_ID_DATASET",
    "V11_COLLECTION_EXECUTION_BINDING_FIELDS",
    "V11_FROZEN_WORKER_SYNTHETIC_EVIDENCE_LEVEL",
    "V11_IN_PROCESS_SYNTHETIC_EVIDENCE_LEVEL",
    "V11_EPISODE_REQUEST_EVIDENCE_FIELDS",
    "V11_PRODUCER_BINDING_FIELDS",
    "V11_RAW_POLICY_ACTION_DATASET",
    "V11_SOURCE_BINDING_FIELDS",
    "V11_SOURCE_SCHEMA_PROFILE_SHA256_V1",
    "V11_SOURCE_SCHEMA_PROFILE_V1",
    "build_v11_source_collection_preparation_v1",
]
