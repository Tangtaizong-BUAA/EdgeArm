"""Fail-closed provenance and action-admission contract for V26 tri-source data.

The three sources share policy input and target semantics, but they do not
share provenance rules.  In particular, scratch RL must remain independent of
experts and behavior cloning, while human trajectories may supervise actions
only after a strict success or an explicitly admitted correction.  Source
identity is audit metadata and is never a deployable actor input.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping


TRISOURCE_CONTRACT_FORMAT_V26 = "edgearm-v26-trisource-causal-admission-v1"
REAL_HUMAN_SOURCE_V26 = "real_human"
SIM_HUMAN_SOURCE_V26 = "sim_human"
SIM_RL_SCRATCH_SOURCE_V26 = "sim_rl_scratch"
TRISOURCE_SOURCE_TYPES_V26 = (
    REAL_HUMAN_SOURCE_V26,
    SIM_HUMAN_SOURCE_V26,
    SIM_RL_SCRATCH_SOURCE_V26,
)


@dataclass(frozen=True, slots=True)
class TrisourceEpisodeEvidenceV26:
    """Minimal immutable evidence needed before one episode can enter VLA data."""

    source_type: str
    episode_uid: str
    parent_episode_uid: str | None
    live_capture: bool
    derived_replay: bool
    causal_audit_passed: bool
    future_leakage_detected: bool
    action_row_count: int
    admitted_action_row_count: int
    strict_success: bool
    correction_admitted: bool
    actor_policy_input_keys: tuple[str, ...]
    expert_calls: int
    behavior_cloning_steps: int
    physical_sample_count: int
    simulator_privileged_actor_input: bool

    def validate(self) -> None:
        canonical_trisource_source_v26(self.source_type)
        if not isinstance(self.episode_uid, str) or not self.episode_uid.strip():
            raise ValueError("V26 tri-source episode_uid must be non-empty")
        if self.parent_episode_uid is not None and (
            not isinstance(self.parent_episode_uid, str)
            or not self.parent_episode_uid.strip()
            or self.parent_episode_uid == self.episode_uid
        ):
            raise ValueError("V26 tri-source parent episode identity is invalid")
        booleans = (
            self.live_capture,
            self.derived_replay,
            self.causal_audit_passed,
            self.future_leakage_detected,
            self.strict_success,
            self.correction_admitted,
            self.simulator_privileged_actor_input,
        )
        if any(type(value) is not bool for value in booleans):
            raise TypeError("V26 tri-source evidence flags must be exact booleans")
        counts = (
            self.action_row_count,
            self.admitted_action_row_count,
            self.expert_calls,
            self.behavior_cloning_steps,
            self.physical_sample_count,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("V26 tri-source evidence counts must be non-negative integers")
        if self.admitted_action_row_count > self.action_row_count:
            raise ValueError("V26 admitted action rows exceed episode action rows")
        if not isinstance(self.actor_policy_input_keys, tuple) or any(
            not isinstance(name, str) or not name
            for name in self.actor_policy_input_keys
        ):
            raise TypeError("V26 actor policy input keys must be a tuple of names")
        if len(set(self.actor_policy_input_keys)) != len(self.actor_policy_input_keys):
            raise ValueError("V26 actor policy input keys contain duplicates")
        if self.live_capture == self.derived_replay:
            raise ValueError("V26 episode must be exactly one of live capture or derived replay")
        if self.derived_replay and self.parent_episode_uid is None:
            raise ValueError("V26 derived replay must bind its live parent episode")
        if self.live_capture and self.parent_episode_uid is not None:
            raise ValueError("V26 live capture cannot claim a parent trajectory")


def canonical_trisource_source_v26(source_type: str) -> str:
    """Return one exact canonical source and reject legacy aliases."""

    if type(source_type) is not str or source_type not in TRISOURCE_SOURCE_TYPES_V26:
        raise ValueError(
            "V26 source_type must be exactly real_human, sim_human, or sim_rl_scratch"
        )
    return source_type


def episode_action_admission_v26(
    evidence: TrisourceEpisodeEvidenceV26,
) -> dict[str, Any]:
    """Audit one source-specific episode without changing shared target semantics."""

    evidence.validate()
    source = evidence.source_type
    actor_keys = set(evidence.actor_policy_input_keys)
    source_shortcut_absent = not actor_keys.intersection(
        {"source", "source_id", "source_type", "provenance_source"}
    )
    simulator = source in {SIM_HUMAN_SOURCE_V26, SIM_RL_SCRATCH_SOURCE_V26}
    outcome_or_correction = evidence.strict_success or (
        source in {REAL_HUMAN_SOURCE_V26, SIM_HUMAN_SOURCE_V26}
        and evidence.correction_admitted
    )
    checks = {
        "causal_audit_passed": evidence.causal_audit_passed,
        "future_leakage_absent": not evidence.future_leakage_detected,
        "source_shortcut_absent_from_actor": source_shortcut_absent,
        "simulator_privilege_absent_from_actor": (
            not evidence.simulator_privileged_actor_input if simulator else True
        ),
        "admitted_actions_require_success_or_correction": (
            evidence.admitted_action_row_count == 0 or outcome_or_correction
        ),
        "scratch_has_no_expert_calls": (
            evidence.expert_calls == 0
            if source == SIM_RL_SCRATCH_SOURCE_V26
            else True
        ),
        "scratch_has_no_behavior_cloning": (
            evidence.behavior_cloning_steps == 0
            if source == SIM_RL_SCRATCH_SOURCE_V26
            else True
        ),
        "physical_samples_match_source": (
            evidence.physical_sample_count > 0
            if source == REAL_HUMAN_SOURCE_V26
            else evidence.physical_sample_count == 0
        ),
        "derived_replay_has_parent": (
            evidence.parent_episode_uid is not None
            if evidence.derived_replay
            else True
        ),
    }
    return {
        "format": TRISOURCE_CONTRACT_FORMAT_V26,
        "source_type": source,
        "episode_uid": evidence.episode_uid,
        "checks": checks,
        "action_supervision_eligible": all(checks.values()),
        "unique_live_episode_increment": int(evidence.live_capture),
        "derived_replay_increment": int(evidence.derived_replay),
        "admitted_action_row_count": evidence.admitted_action_row_count,
        "source_is_actor_input": False,
        "production_admission": False,
    }


def audit_trisource_episode_set_v26(
    episodes: Iterable[TrisourceEpisodeEvidenceV26],
    *,
    required_sources: Iterable[str] = (),
) -> dict[str, Any]:
    """Audit identities, per-source counts, and all per-episode action gates."""

    evidence = tuple(episodes)
    if not evidence:
        raise ValueError("V26 tri-source audit requires at least one episode")
    required = tuple(canonical_trisource_source_v26(value) for value in required_sources)
    if len(set(required)) != len(required):
        raise ValueError("V26 required source list contains duplicates")
    seen_uids: set[str] = set()
    uid_to_source: dict[str, str] = {}
    reports: list[dict[str, Any]] = []
    source_counts = {
        source: {
            "episode_count": 0,
            "unique_live_episode_count": 0,
            "derived_replay_count": 0,
            "admitted_action_row_count": 0,
        }
        for source in TRISOURCE_SOURCE_TYPES_V26
    }
    for item in evidence:
        item.validate()
        if item.episode_uid in seen_uids:
            raise ValueError("V26 tri-source episode_uid is duplicated")
        seen_uids.add(item.episode_uid)
        uid_to_source[item.episode_uid] = item.source_type
        report = episode_action_admission_v26(item)
        reports.append(report)
        counts = source_counts[item.source_type]
        counts["episode_count"] += 1
        counts["unique_live_episode_count"] += report["unique_live_episode_increment"]
        counts["derived_replay_count"] += report["derived_replay_increment"]
        counts["admitted_action_row_count"] += item.admitted_action_row_count
    present = {source for source, counts in source_counts.items() if counts["episode_count"]}
    derived_parent_links_valid = all(
        not item.derived_replay
        or uid_to_source.get(str(item.parent_episode_uid)) == item.source_type
        for item in evidence
    )
    checks = {
        "required_sources_present": set(required).issubset(present),
        "required_sources_have_admitted_actions": all(
            source_counts[source]["admitted_action_row_count"] > 0
            for source in required
        ),
        "all_episode_action_gates_pass": all(
            report["action_supervision_eligible"] for report in reports
        ),
        "episode_identities_unique": True,
        "derived_replay_parents_present_and_same_source": derived_parent_links_valid,
        "derived_replays_not_counted_as_unique_live": all(
            report["unique_live_episode_increment"] == 0
            for report in reports
            if report["derived_replay_increment"] == 1
        ),
    }
    return {
        "format": TRISOURCE_CONTRACT_FORMAT_V26,
        "required_sources": list(required),
        "present_sources": sorted(present),
        "source_counts": source_counts,
        "episode_reports": reports,
        "checks": checks,
        "training_mix_eligible": all(checks.values()),
        "source_type_is_policy_input": False,
        "production_admission": False,
    }


def trisource_contract_payload_v26() -> Mapping[str, Any]:
    """Return a stable model-facing declaration with no mutable registry state."""

    return {
        "format": TRISOURCE_CONTRACT_FORMAT_V26,
        "canonical_source_types": list(TRISOURCE_SOURCE_TYPES_V26),
        "source_type_is_policy_input": False,
        "derived_replay_counts_as_unique_live_episode": False,
        "shared_action_target": "safe applied/executed action with feedback evidence",
        "episode_evidence_fields": list(asdict(TrisourceEpisodeEvidenceV26(
            source_type=SIM_RL_SCRATCH_SOURCE_V26,
            episode_uid="schema-only",
            parent_episode_uid=None,
            live_capture=True,
            derived_replay=False,
            causal_audit_passed=True,
            future_leakage_detected=False,
            action_row_count=0,
            admitted_action_row_count=0,
            strict_success=False,
            correction_admitted=False,
            actor_policy_input_keys=(),
            expert_calls=0,
            behavior_cloning_steps=0,
            physical_sample_count=0,
            simulator_privileged_actor_input=False,
        )).keys()),
        "production_admission": False,
    }


__all__ = [
    "REAL_HUMAN_SOURCE_V26",
    "SIM_HUMAN_SOURCE_V26",
    "SIM_RL_SCRATCH_SOURCE_V26",
    "TRISOURCE_CONTRACT_FORMAT_V26",
    "TRISOURCE_SOURCE_TYPES_V26",
    "TrisourceEpisodeEvidenceV26",
    "audit_trisource_episode_set_v26",
    "canonical_trisource_source_v26",
    "episode_action_admission_v26",
    "trisource_contract_payload_v26",
]
