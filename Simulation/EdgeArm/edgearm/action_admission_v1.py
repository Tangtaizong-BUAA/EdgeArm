"""Source-neutral action admission for Causal V2 trajectories.

Episode integrity and action-label admission are deliberately separate.  A
complete, causally aligned failed rollout is still useful execution evidence,
but none of its action rows may enter ACT training unless the source-specific
admission rule passes.  V11 admission is an exact wrapper around its existing
promotion result.  Scratch-PPO admission instead proves expert-free,
full-action genesis and never fabricates V11 promotion evidence.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib
import json
import re
from typing import Any, Mapping, Sequence

EPISODE_INTEGRITY_GATE_FORMAT = "edgearm-causal-v2-episode-integrity-gate-v1"
ACTION_ADMISSION_GATE_FORMAT = "edgearm-causal-v2-action-admission-gate-v1"
ACTION_SOURCE_REGISTRY_FORMAT = "edgearm-causal-v2-action-source-registry-v1"

V11_SOURCE_TYPE = "synthetic_v11_privileged_expert"
SCRATCH_SOURCE_TYPE = "synthetic_ppo_from_scratch"
SUPPORTED_SOURCE_TYPES = (V11_SOURCE_TYPE, SCRATCH_SOURCE_TYPE)

GENERIC_ACTION_LABEL_DATASET = "decision_safe_action"
V11_TEACHER_DIAGNOSTIC_ALIAS = "teacher_safe_action"
SCRATCH_ACTION_DIM = 6
SCRATCH_CHECKPOINT_FORMAT_V1 = "edgearm-realism-v7-full-action-ppo-from-scratch-v1"
# Historical public alias.  It must always retain the V1 literal.
SCRATCH_CHECKPOINT_FORMAT = SCRATCH_CHECKPOINT_FORMAT_V1
SCRATCH_CHECKPOINT_FORMAT_V2 = "edgearm-realism-v9-full-action-ppo-from-scratch-v2"
SCRATCH_POLICY_PARAMETERIZATION = "six_dimensional_normal_latent_then_single_tanh_no_baseline_v1"
SCRATCH_POTENTIAL_REWARD_V1_VERSION = "edgearm-v7-source-neutral-potential-reward-v1"
SCRATCH_POTENTIAL_REWARD_V1_FORMULA = (
    "8*coverage-4*block_target_distance-1.5*clip(tool_precontact_xy_distance)"
    "-0.5*clip(abs(tool_z-operational_z_ref))"
)
SCRATCH_POTENTIAL_REWARD_V1_CONFIG_DEFAULTS: dict[str, float] = {
    "coverage_coefficient": 8.0,
    "block_target_distance_coefficient": 4.0,
    "tool_precontact_xy_coefficient": 1.5,
    "operational_z_error_coefficient": 0.5,
    "precontact_gap_m": 0.010,
    "tool_precontact_xy_clip_m": 0.40,
    "operational_z_error_clip_m": 0.25,
}
# Historical public aliases.  V2 must never be exposed through these names.
SCRATCH_POTENTIAL_REWARD_VERSION = SCRATCH_POTENTIAL_REWARD_V1_VERSION
SCRATCH_POTENTIAL_REWARD_FORMULA = SCRATCH_POTENTIAL_REWARD_V1_FORMULA
SCRATCH_POTENTIAL_REWARD_CONFIG_DEFAULTS = (
    SCRATCH_POTENTIAL_REWARD_V1_CONFIG_DEFAULTS
)
SCRATCH_POTENTIAL_REWARD_V2_VERSION = (
    "edgearm-v9-per-jaw-safety-union-potential-reward-v2"
)
SCRATCH_POTENTIAL_REWARD_V2_FORMULA = (
    "8*coverage-4*block_target_distance-1.5*clip(max_per_tip_precontact_xy_error)"
    "-0.5*clip(abs(min_full_safety_desk_distance-runtime_desk_clearance))"
)
SCRATCH_POTENTIAL_REWARD_V2_CONFIG_DEFAULTS: dict[str, float] = {
    "coverage_coefficient": 8.0,
    "block_target_distance_coefficient": 4.0,
    "worst_tip_precontact_xy_coefficient": 1.5,
    "safety_desk_clearance_error_coefficient": 0.5,
    "precontact_gap_m": 0.010,
    "worst_tip_precontact_xy_clip_m": 0.40,
    "safety_desk_clearance_error_clip_m": 0.25,
}
SCRATCH_V1_ENVIRONMENT_PROFILES = (
    "stock_gripper_v8",
    "legacy_push_plate_v7",
)
SCRATCH_V2_ENVIRONMENT_PROFILE = "stock_gripper_v9"
SCRATCH_CHECKPOINT_BINDING_SCOPE = "collection_time_hash_declaration_only"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _scratch_contract_module() -> Any:
    """Load scratch implementation only when a scratch-specific API is used."""

    module = importlib.import_module(".scratch_ppo_v1", __package__)
    exact = {
        "ACTION_DIM": SCRATCH_ACTION_DIM,
        "CHECKPOINT_FORMAT": SCRATCH_CHECKPOINT_FORMAT,
        "CHECKPOINT_FORMAT_V1": SCRATCH_CHECKPOINT_FORMAT_V1,
        "CHECKPOINT_FORMAT_V2": SCRATCH_CHECKPOINT_FORMAT_V2,
        "POLICY_PARAMETERIZATION": SCRATCH_POLICY_PARAMETERIZATION,
        "POTENTIAL_REWARD_FORMULA": SCRATCH_POTENTIAL_REWARD_FORMULA,
        "POTENTIAL_REWARD_VERSION": SCRATCH_POTENTIAL_REWARD_VERSION,
        "POTENTIAL_REWARD_V1_FORMULA": SCRATCH_POTENTIAL_REWARD_V1_FORMULA,
        "POTENTIAL_REWARD_V1_VERSION": SCRATCH_POTENTIAL_REWARD_V1_VERSION,
        "POTENTIAL_REWARD_V2_FORMULA": SCRATCH_POTENTIAL_REWARD_V2_FORMULA,
        "POTENTIAL_REWARD_V2_VERSION": SCRATCH_POTENTIAL_REWARD_V2_VERSION,
    }
    for name, expected in exact.items():
        if getattr(module, name, None) != expected:
            raise RuntimeError(f"scratch implementation contract drifted: {name}")
    if asdict(module.ScratchPotentialRewardV1Config()) != (
        SCRATCH_POTENTIAL_REWARD_V1_CONFIG_DEFAULTS
    ):
        raise RuntimeError("scratch V1 potential-reward defaults drifted")
    if asdict(module.ScratchPotentialRewardV2Config()) != (
        SCRATCH_POTENTIAL_REWARD_V2_CONFIG_DEFAULTS
    ):
        raise RuntimeError("scratch V2 potential-reward defaults drifted")
    return module


def _scratch_checkpoint_contract_v1(
    potential_reward_version: Any,
) -> dict[str, Any] | None:
    if potential_reward_version == SCRATCH_POTENTIAL_REWARD_V1_VERSION:
        return {
            "checkpoint_format": SCRATCH_CHECKPOINT_FORMAT_V1,
            "potential_reward_version": SCRATCH_POTENTIAL_REWARD_V1_VERSION,
            "potential_reward_formula": SCRATCH_POTENTIAL_REWARD_V1_FORMULA,
            "environment_profiles": SCRATCH_V1_ENVIRONMENT_PROFILES,
            "environment_profile_required": False,
        }
    if potential_reward_version == SCRATCH_POTENTIAL_REWARD_V2_VERSION:
        return {
            "checkpoint_format": SCRATCH_CHECKPOINT_FORMAT_V2,
            "potential_reward_version": SCRATCH_POTENTIAL_REWARD_V2_VERSION,
            "potential_reward_formula": SCRATCH_POTENTIAL_REWARD_V2_FORMULA,
            "environment_profiles": (SCRATCH_V2_ENVIRONMENT_PROFILE,),
            "environment_profile_required": True,
        }
    return None


_SCRATCH_SOURCE_EVIDENCE_FIELDS_V1 = frozenset(
    {
        "source_type",
        "checkpoint_format",
        "policy_parameterization",
        "action_dim",
        "full_action",
        "residual_policy",
        "checkpoint_sha256",
        "actor_state_sha256",
        "critic_state_sha256",
        "potential_reward_version",
        "potential_reward_formula",
        "potential_reward_config",
        "potential_reward_config_sha256",
        "checkpoint_binding_scope",
        "checkpoint_artifact_coaudit_pass",
        "provenance",
    }
)
_SCRATCH_SOURCE_EVIDENCE_FIELDS_V2 = frozenset(
    {*_SCRATCH_SOURCE_EVIDENCE_FIELDS_V1, "environment_profile"}
)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def action_source_registry() -> dict[str, Any]:
    """Return the immutable source/admission registry stored at shard root."""

    return {
        "format": ACTION_SOURCE_REGISTRY_FORMAT,
        "generic_action_label_dataset": GENERIC_ACTION_LABEL_DATASET,
        "sources": {
            V11_SOURCE_TYPE: {
                "admission_rule": "exact_v11_promotion_wrapper",
                "teacher_diagnostic_alias": V11_TEACHER_DIAGNOSTIC_ALIAS,
            },
            SCRATCH_SOURCE_TYPE: {
                "admission_rule": "strict_success_and_expert_free_full_action_checkpoint",
                "checkpoint_format": SCRATCH_CHECKPOINT_FORMAT,
                "versioned_checkpoint_contracts": {
                    SCRATCH_POTENTIAL_REWARD_V1_VERSION: {
                        "checkpoint_format": SCRATCH_CHECKPOINT_FORMAT_V1,
                        "potential_reward_formula": SCRATCH_POTENTIAL_REWARD_V1_FORMULA,
                        "environment_profiles": list(
                            SCRATCH_V1_ENVIRONMENT_PROFILES
                        ),
                        "environment_profile_required_in_source_evidence": False,
                    },
                    SCRATCH_POTENTIAL_REWARD_V2_VERSION: {
                        "checkpoint_format": SCRATCH_CHECKPOINT_FORMAT_V2,
                        "potential_reward_formula": SCRATCH_POTENTIAL_REWARD_V2_FORMULA,
                        "environment_profiles": [
                            SCRATCH_V2_ENVIRONMENT_PROFILE
                        ],
                        "environment_profile_required_in_source_evidence": True,
                    },
                },
                "checkpoint_binding_scope": SCRATCH_CHECKPOINT_BINDING_SCOPE,
                "formal_coverage_requires_checkpoint_artifact_coaudit": True,
                "teacher_diagnostic_alias": None,
            },
        },
    }


def action_source_registry_json() -> str:
    return json.dumps(action_source_registry(), sort_keys=True, separators=(",", ":"))


def _strict_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def scratch_potential_reward_config_sha256(
    config: Mapping[str, Any],
    *,
    potential_reward_version: str = SCRATCH_POTENTIAL_REWARD_V1_VERSION,
) -> str:
    """Recompute one explicitly versioned scratch reward configuration hash.

    The default is intentionally the historical V1 contract.  Callers handling
    V2 must name V2; field-shape guessing is forbidden because both hashes bind
    different geometry semantics.
    """

    scratch_contract = _scratch_contract_module()
    contract = _scratch_checkpoint_contract_v1(potential_reward_version)
    if contract is None:
        raise ValueError("scratch potential reward version is unsupported")
    try:
        reward_config = scratch_contract.potential_reward_config_from_dict_v1(
            potential_reward_version,
            dict(config) if isinstance(config, Mapping) else config,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("scratch potential reward config is invalid") from error
    if not isinstance(config, Mapping) or asdict(reward_config) != dict(config):
        raise ValueError("scratch potential reward config is non-canonical")
    return reward_config.sha256()


def _scratch_provenance_reason_codes(evidence: Mapping[str, Any]) -> list[str]:
    scratch_contract = _scratch_contract_module()
    reasons: list[str] = []
    reward_version = evidence.get("potential_reward_version")
    contract = _scratch_checkpoint_contract_v1(reward_version)
    if contract is None:
        reasons.append("scratch_potential_reward_version_unsupported")
        expected_fields = None
    elif contract["environment_profile_required"]:
        expected_fields = _SCRATCH_SOURCE_EVIDENCE_FIELDS_V2
    else:
        expected_fields = _SCRATCH_SOURCE_EVIDENCE_FIELDS_V1
    if expected_fields is not None and set(evidence) != expected_fields:
        reasons.append("scratch_source_evidence_fields_mismatch")

    exact = {
        "source_type": SCRATCH_SOURCE_TYPE,
        "policy_parameterization": SCRATCH_POLICY_PARAMETERIZATION,
        "action_dim": SCRATCH_ACTION_DIM,
        "full_action": True,
        "residual_policy": False,
        "checkpoint_binding_scope": SCRATCH_CHECKPOINT_BINDING_SCOPE,
        "checkpoint_artifact_coaudit_pass": False,
    }
    if contract is not None:
        exact.update(
            {
                "checkpoint_format": contract["checkpoint_format"],
                "potential_reward_version": contract[
                    "potential_reward_version"
                ],
                "potential_reward_formula": contract["potential_reward_formula"],
            }
        )
        if contract["environment_profile_required"]:
            exact["environment_profile"] = SCRATCH_V2_ENVIRONMENT_PROFILE
    for name, expected in exact.items():
        if evidence.get(name) != expected:
            reasons.append(f"scratch_{name}_mismatch")

    for name in ("checkpoint_sha256", "actor_state_sha256", "critic_state_sha256"):
        if not _valid_sha256(evidence.get(name)):
            reasons.append(f"scratch_{name}_invalid")
    reward_config = evidence.get("potential_reward_config")
    try:
        if contract is None:
            raise ValueError("unsupported potential reward version")
        expected_reward_hash = scratch_potential_reward_config_sha256(
            reward_config,
            potential_reward_version=contract["potential_reward_version"],
        )
    except (TypeError, ValueError):
        expected_reward_hash = None
        reasons.append("scratch_potential_reward_config_invalid")
    stored_reward_hash = evidence.get("potential_reward_config_sha256")
    if not _valid_sha256(stored_reward_hash) or stored_reward_hash != expected_reward_hash:
        reasons.append("scratch_potential_reward_config_sha256_mismatch")

    provenance = evidence.get("provenance")
    if not isinstance(provenance, Mapping):
        return [*reasons, "scratch_provenance_missing_or_invalid"]
    provenance_exact = {
        "source_type": SCRATCH_SOURCE_TYPE,
        "random_initialization": True,
        "expert_calls": 0,
        "warm_start": False,
        "behavior_cloning_steps": 0,
    }
    if contract is not None:
        provenance_exact.update(
            {
                "checkpoint_format": contract["checkpoint_format"],
                "potential_reward_version": contract[
                    "potential_reward_version"
                ],
            }
        )
    for name, expected in provenance_exact.items():
        if provenance.get(name) != expected:
            reasons.append(f"scratch_provenance_{name}_mismatch")
    for name in (
        "actor_initial_state_sha256",
        "critic_initial_state_sha256",
        "privileged_state_schema_sha256",
        "genesis_sha256",
        "potential_reward_config_sha256",
    ):
        if not _valid_sha256(provenance.get(name)):
            reasons.append(f"scratch_provenance_{name}_invalid")
    if provenance.get("potential_reward_config_sha256") != stored_reward_hash:
        reasons.append("scratch_provenance_potential_reward_config_sha256_mismatch")
    trainer_hashes = provenance.get("trainer_source_hashes")
    expected_sources = {
        "ppo_utils_v1.py",
        "privileged_effect_state_v1.py",
        "scratch_ppo_v1.py",
    }
    if (
        not isinstance(trainer_hashes, Mapping)
        or set(trainer_hashes) != expected_sources
        or any(not _valid_sha256(value) for value in trainer_hashes.values())
    ):
        reasons.append("scratch_provenance_trainer_source_hashes_invalid")
    try:
        scratch_contract.ScratchPPOProvenanceV1.from_dict(dict(provenance))
    except (TypeError, ValueError):
        reasons.append("scratch_provenance_authoritative_validation_failed")
    return reasons


def validate_scratch_source_evidence(evidence: Mapping[str, Any]) -> tuple[str, ...]:
    """Return deterministic fail-closed reason codes for scratch provenance."""

    if not isinstance(evidence, Mapping):
        return ("scratch_evidence_missing_or_invalid",)
    return tuple(sorted(set(_scratch_provenance_reason_codes(evidence))))


def scratch_source_evidence_from_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    environment_profile: str | None = None,
) -> dict[str, Any]:
    """Extract the admission boundary from an already validated checkpoint.

    The caller declares the hash of the checkpoint bytes after separately
    validating the checkpoint payload.  This function cannot reopen those
    bytes, so the evidence is explicitly collection-time-only and records
    ``checkpoint_artifact_coaudit_pass=false``.  HDF5-only formal coverage
    remains closed until a separate artifact co-audit is implemented.
    """

    if not isinstance(checkpoint, Mapping):
        raise ValueError("scratch checkpoint evidence source must be an object")
    if not _valid_sha256(checkpoint_sha256):
        raise ValueError("scratch checkpoint file hash is invalid")
    reward_version = checkpoint.get("potential_reward_version")
    contract = _scratch_checkpoint_contract_v1(reward_version)
    if contract is None:
        raise ValueError("scratch checkpoint potential reward version is unsupported")
    if contract["environment_profile_required"]:
        if environment_profile != SCRATCH_V2_ENVIRONMENT_PROFILE:
            raise ValueError(
                "scratch V2 checkpoint requires environment_profile=stock_gripper_v9"
            )
    elif environment_profile is not None and environment_profile not in (
        SCRATCH_V1_ENVIRONMENT_PROFILES
    ):
        raise ValueError("scratch V1 checkpoint environment profile is incompatible")
    source = {
        "source_type": checkpoint.get("source_type"),
        "checkpoint_format": checkpoint.get("format"),
        "policy_parameterization": checkpoint.get("policy_parameterization"),
        "action_dim": checkpoint.get("action_dim"),
        "full_action": True,
        "residual_policy": False,
        "checkpoint_sha256": checkpoint_sha256,
        "actor_state_sha256": checkpoint.get("actor_state_sha256"),
        "critic_state_sha256": checkpoint.get("critic_state_sha256"),
        "potential_reward_version": checkpoint.get("potential_reward_version"),
        "potential_reward_formula": checkpoint.get("potential_reward_formula"),
        "potential_reward_config": checkpoint.get("potential_reward_config"),
        "potential_reward_config_sha256": checkpoint.get("potential_reward_config_sha256"),
        "checkpoint_binding_scope": SCRATCH_CHECKPOINT_BINDING_SCOPE,
        "checkpoint_artifact_coaudit_pass": False,
        "provenance": checkpoint.get("provenance"),
    }
    if contract["environment_profile_required"]:
        source["environment_profile"] = environment_profile
    failures = validate_scratch_source_evidence(source)
    if failures:
        raise ValueError(f"scratch checkpoint admission evidence invalid: {list(failures)}")
    return source


def build_v11_action_admission(
    promotion: Mapping[str, Any],
    *,
    episode_integrity_pass: bool,
    strict_success: bool,
) -> dict[str, Any]:
    """Wrap, but never reinterpret, the authoritative V11 promotion object."""

    if not isinstance(promotion, Mapping):
        raise ValueError("V11 promotion evidence must be an object")
    promotion_copy = dict(promotion)
    promotion_passed = _strict_bool(promotion_copy.get("passed"))
    promotion_eligible = _strict_bool(promotion_copy.get("action_label_eligible"))
    if promotion_passed is None or promotion_eligible is not promotion_passed:
        raise ValueError("V11 promotion pass/eligibility is invalid or inconsistent")
    reasons: list[str] = []
    if not episode_integrity_pass:
        reasons.append("episode_integrity_failed")
    if not strict_success:
        reasons.append("strict_success_required")
    if not promotion_passed:
        failed_checks = promotion_copy.get("failed_checks")
        if isinstance(failed_checks, Sequence) and not isinstance(failed_checks, (str, bytes)):
            reasons.extend(f"v11_promotion_failed:{name}" for name in failed_checks)
        else:
            reasons.append("v11_promotion_failed")
    passed = bool(episode_integrity_pass and strict_success and promotion_passed)
    evidence = {
        "format": ACTION_ADMISSION_GATE_FORMAT,
        "source_type": V11_SOURCE_TYPE,
        "passed": passed,
        "episode_integrity_pass": bool(episode_integrity_pass),
        "strict_success": bool(strict_success),
        "source_evidence_format": promotion_copy.get("format"),
        "source_evidence_sha256": canonical_sha256(promotion_copy),
        "source_evidence": promotion_copy,
    }
    return {
        "format": ACTION_ADMISSION_GATE_FORMAT,
        "passed": passed,
        "evidence": evidence,
        "reason_codes": sorted(set(reasons)),
    }


def build_scratch_action_admission(
    source_evidence: Mapping[str, Any],
    *,
    episode_integrity_pass: bool,
    strict_success: bool,
) -> dict[str, Any]:
    """Admit successful scratch rollouts only with expert-free provenance."""

    source_copy = dict(source_evidence) if isinstance(source_evidence, Mapping) else {}
    reasons = list(validate_scratch_source_evidence(source_copy))
    if not episode_integrity_pass:
        reasons.append("episode_integrity_failed")
    if not strict_success:
        reasons.append("strict_success_required")
    passed = not reasons
    evidence = {
        "format": ACTION_ADMISSION_GATE_FORMAT,
        "source_type": SCRATCH_SOURCE_TYPE,
        "passed": passed,
        "episode_integrity_pass": bool(episode_integrity_pass),
        "strict_success": bool(strict_success),
        "source_evidence_format": source_copy.get("checkpoint_format"),
        "source_evidence_sha256": canonical_sha256(source_copy),
        "source_evidence": source_copy,
    }
    return {
        "format": ACTION_ADMISSION_GATE_FORMAT,
        "passed": passed,
        "evidence": evidence,
        "reason_codes": sorted(set(reasons)),
    }


def validate_action_admission_record(
    record: Mapping[str, Any],
    *,
    source_type: str,
    episode_integrity_pass: bool,
    strict_success: bool,
    v11_promotion: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Independently validate one stored source-specific admission record."""

    failures: list[str] = []
    if not isinstance(record, Mapping):
        return ("action_admission_record_missing_or_invalid",)
    if record.get("format") != ACTION_ADMISSION_GATE_FORMAT:
        failures.append("action_admission_format_mismatch")
    stored_passed = _strict_bool(record.get("passed"))
    if stored_passed is None:
        failures.append("action_admission_pass_not_boolean")
    reason_codes = record.get("reason_codes")
    if (
        not isinstance(reason_codes, list)
        or any(not isinstance(value, str) or not value for value in reason_codes)
        or reason_codes != sorted(set(reason_codes))
    ):
        failures.append("action_admission_reason_codes_invalid")
    evidence = record.get("evidence")
    if not isinstance(evidence, Mapping):
        return tuple(sorted(set([*failures, "action_admission_evidence_missing_or_invalid"])))
    if evidence.get("format") != ACTION_ADMISSION_GATE_FORMAT:
        failures.append("action_admission_evidence_format_mismatch")
    if evidence.get("source_type") != source_type:
        failures.append("action_admission_source_type_mismatch")
    if _strict_bool(evidence.get("episode_integrity_pass")) is not episode_integrity_pass:
        failures.append("action_admission_integrity_mismatch")
    if _strict_bool(evidence.get("strict_success")) is not strict_success:
        failures.append("action_admission_strict_success_mismatch")
    source_evidence = evidence.get("source_evidence")
    if not isinstance(source_evidence, Mapping):
        failures.append("action_admission_source_evidence_missing_or_invalid")
        source_evidence = {}
    if evidence.get("source_evidence_sha256") != canonical_sha256(dict(source_evidence)):
        failures.append("action_admission_source_evidence_hash_mismatch")

    if source_type == V11_SOURCE_TYPE:
        if v11_promotion is None or dict(source_evidence) != dict(v11_promotion):
            failures.append("action_admission_v11_promotion_wrapper_mismatch")
            expected = None
        else:
            try:
                expected = build_v11_action_admission(
                    v11_promotion,
                    episode_integrity_pass=episode_integrity_pass,
                    strict_success=strict_success,
                )
            except ValueError:
                expected = None
                failures.append("action_admission_v11_promotion_invalid")
    elif source_type == SCRATCH_SOURCE_TYPE:
        expected = build_scratch_action_admission(
            source_evidence,
            episode_integrity_pass=episode_integrity_pass,
            strict_success=strict_success,
        )
    else:
        expected = None
        failures.append("action_admission_source_type_unsupported")

    if expected is not None:
        if stored_passed is not expected["passed"]:
            failures.append("action_admission_stored_pass_mismatch")
        if evidence.get("passed") is not expected["passed"]:
            failures.append("action_admission_evidence_pass_mismatch")
        if reason_codes != expected["reason_codes"]:
            failures.append("action_admission_reason_codes_mismatch")
    return tuple(sorted(set(failures)))


__all__ = [
    "ACTION_ADMISSION_GATE_FORMAT",
    "ACTION_SOURCE_REGISTRY_FORMAT",
    "EPISODE_INTEGRITY_GATE_FORMAT",
    "GENERIC_ACTION_LABEL_DATASET",
    "SCRATCH_CHECKPOINT_FORMAT",
    "SCRATCH_CHECKPOINT_FORMAT_V1",
    "SCRATCH_CHECKPOINT_FORMAT_V2",
    "SCRATCH_POTENTIAL_REWARD_CONFIG_DEFAULTS",
    "SCRATCH_POTENTIAL_REWARD_FORMULA",
    "SCRATCH_POTENTIAL_REWARD_VERSION",
    "SCRATCH_POTENTIAL_REWARD_V1_CONFIG_DEFAULTS",
    "SCRATCH_POTENTIAL_REWARD_V1_FORMULA",
    "SCRATCH_POTENTIAL_REWARD_V1_VERSION",
    "SCRATCH_POTENTIAL_REWARD_V2_CONFIG_DEFAULTS",
    "SCRATCH_POTENTIAL_REWARD_V2_FORMULA",
    "SCRATCH_POTENTIAL_REWARD_V2_VERSION",
    "SCRATCH_POLICY_PARAMETERIZATION",
    "SCRATCH_SOURCE_TYPE",
    "SCRATCH_V1_ENVIRONMENT_PROFILES",
    "SCRATCH_V2_ENVIRONMENT_PROFILE",
    "SUPPORTED_SOURCE_TYPES",
    "V11_SOURCE_TYPE",
    "V11_TEACHER_DIAGNOSTIC_ALIAS",
    "action_source_registry",
    "action_source_registry_json",
    "build_scratch_action_admission",
    "build_v11_action_admission",
    "canonical_sha256",
    "scratch_potential_reward_config_sha256",
    "scratch_source_evidence_from_checkpoint",
    "validate_action_admission_record",
    "validate_scratch_source_evidence",
]
