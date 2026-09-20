"""Fail-closed provenance primitives for future scripted recovery injection.

This module does not inject, drop, hold, retract, or otherwise modify a robot
command or simulator state.  It only defines the immutable evidence envelope
that a later, separately reviewed injector must produce before recovery rows
can be considered formal evidence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping


RECOVERY_INJECTOR_FORMAT = "edgearm-recovery-injector-v1"
RECOVERY_EVENT_LEDGER_FORMAT = "edgearm-recovery-event-ledger-v1"
RECOVERY_EVENT_RECORD_FORMAT = "edgearm-recovery-event-record-v1"
TRAJECTORY_ORIGIN_LIVE_PRIMARY_EXECUTION = "live_primary_execution"
DISABLED_RECOVERY_INJECTOR_PROFILE = {
    "enabled": False,
    "format": RECOVERY_INJECTOR_FORMAT,
    "implementation_status": "evidence_schema_only_no_state_or_command_mutation",
    "supported_intervention_kinds": [],
    "trajectory_origin": TRAJECTORY_ORIGIN_LIVE_PRIMARY_EXECUTION,
}

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_INT64_MAX = (1 << 63) - 1
_UINT64_MAX = (1 << 64) - 1
_EVENT_KEYS = frozenset(
    {
        "format",
        "event_uid",
        "kind",
        "origin_code",
        "trajectory_origin",
        "request",
        "effect",
        "outcome",
        "replay_verified",
    }
)
_REQUEST_KEYS = frozenset(
    {
        "row_index",
        "control_step",
        "command_ids",
        "hold_substeps",
        "precondition_sha256",
    }
)
_EFFECT_KEYS = frozenset(
    {
        "row_index",
        "control_step",
        "command_ids",
        "hold_substeps",
        "effect_sha256",
    }
)
_INTERVENTION_KINDS = frozenset(
    {
        "scripted_retract_contact_loss",
        "scripted_command_drop_burst",
        "scripted_runtime_clearance_hold",
    }
)
RECOVERY_INJECTION_KIND_TO_ORIGIN_CODE = MappingProxyType(
    {kind: kind for kind in sorted(_INTERVENTION_KINDS)}
)
_OUTCOMES = frozenset(
    {
        "injected_effect_observed",
        "requested_no_effect",
        "aborted_precondition",
        "failed_closed",
    }
)


def canonical_json(value: Any) -> str:
    """Return the one accepted UTF-8 JSON representation."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


DISABLED_RECOVERY_INJECTOR_PROFILE_SHA256 = canonical_json_sha256(DISABLED_RECOVERY_INJECTOR_PROFILE)
EMPTY_RECOVERY_EVENT_LEDGER_SHA256 = canonical_json_sha256([])


def recovery_injector_source_sha256() -> str:
    """Hash the exact module bytes used to declare the disabled injector."""

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _HASH_RE.fullmatch(value) is not None


def _strict_int(value: Any, *, name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 0 or value > maximum:
        raise ValueError(f"{name} is outside its declared integer range")
    return value


def _validate_stage(
    value: Any,
    *,
    stage: str,
    frame_count: int,
    maximum_control_step: int | None,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{stage} must be an object")
    expected = _REQUEST_KEYS if stage == "request" else _EFFECT_KEYS
    if frozenset(value) != expected:
        raise ValueError(f"{stage} has unknown or missing fields")
    row = _strict_int(value["row_index"], name=f"{stage}.row_index", maximum=_INT64_MAX)
    if row >= frame_count:
        raise ValueError(f"{stage}.row_index is outside the trajectory")
    control = _strict_int(value["control_step"], name=f"{stage}.control_step", maximum=_INT64_MAX)
    if maximum_control_step is not None and control > maximum_control_step:
        raise ValueError(f"{stage}.control_step exceeds the trajectory")
    command_ids = value["command_ids"]
    if not isinstance(command_ids, list) or not command_ids:
        raise ValueError(f"{stage}.command_ids must be a non-empty list")
    if len(command_ids) != len(set(command_ids)):
        raise ValueError(f"{stage}.command_ids must be unique")
    for index, command_id in enumerate(command_ids):
        _strict_int(
            command_id,
            name=f"{stage}.command_ids[{index}]",
            maximum=_INT64_MAX,
        )
    hold_substeps = value["hold_substeps"]
    if not isinstance(hold_substeps, list):
        raise ValueError(f"{stage}.hold_substeps must be a list")
    if len(hold_substeps) != len(set(hold_substeps)):
        raise ValueError(f"{stage}.hold_substeps must be unique")
    for index, substep in enumerate(hold_substeps):
        _strict_int(
            substep,
            name=f"{stage}.hold_substeps[{index}]",
            maximum=_UINT64_MAX,
        )
    hash_name = "precondition_sha256" if stage == "request" else "effect_sha256"
    if not is_sha256(value[hash_name]):
        raise ValueError(f"{stage}.{hash_name} must be a lowercase 64-hex SHA-256")
    return value


def event_uid(record_without_uid: Mapping[str, Any]) -> str:
    return f"edgearm-recovery-event-v1:{canonical_json_sha256(record_without_uid)}"


def validate_event_record(
    value: str | Mapping[str, Any],
    *,
    frame_count: int,
    maximum_control_step: int | None = None,
    require_replay_verified: bool = False,
) -> dict[str, Any]:
    """Validate one canonical, immutable recovery-event record.

    A JSON string must already be canonical.  Mapping inputs are accepted for
    in-memory construction, but their UID still binds every other field.
    """

    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    if isinstance(value, str):
        try:
            record = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("event record is not valid JSON") from exc
        if canonical_json(record) != value:
            raise ValueError("event record JSON is not canonical")
    elif isinstance(value, Mapping):
        record = dict(value)
    else:
        raise ValueError("event record must be a canonical JSON string or object")
    if frozenset(record) != _EVENT_KEYS:
        raise ValueError("event record has unknown or missing fields")
    if record["format"] != RECOVERY_EVENT_RECORD_FORMAT:
        raise ValueError("event record format mismatch")
    if record["kind"] not in _INTERVENTION_KINDS:
        raise ValueError("unknown recovery intervention kind")
    if record["origin_code"] != RECOVERY_INJECTION_KIND_TO_ORIGIN_CODE[record["kind"]]:
        raise ValueError("recovery intervention kind and origin code do not match")
    if record["trajectory_origin"] != TRAJECTORY_ORIGIN_LIVE_PRIMARY_EXECUTION:
        raise ValueError("recovery event is not bound to live primary execution")
    request = _validate_stage(
        record["request"],
        stage="request",
        frame_count=frame_count,
        maximum_control_step=maximum_control_step,
    )
    effect = _validate_stage(
        record["effect"],
        stage="effect",
        frame_count=frame_count,
        maximum_control_step=maximum_control_step,
    )
    if int(effect["row_index"]) < int(request["row_index"]):
        raise ValueError("effect row precedes request row")
    if int(effect["control_step"]) < int(request["control_step"]):
        raise ValueError("effect control step precedes request control step")
    if record["outcome"] not in _OUTCOMES:
        raise ValueError("unknown recovery event outcome")
    if type(record["replay_verified"]) is not bool:
        raise ValueError("replay_verified must be a strict boolean")
    if require_replay_verified and record["replay_verified"] is not True:
        raise ValueError("recovery event is not replay verified")
    core = {name: record[name] for name in sorted(_EVENT_KEYS - {"event_uid"})}
    if record["event_uid"] != event_uid(core):
        raise ValueError("event_uid does not bind the canonical record")
    return record


def validate_event_ledger(
    value: str | list[Any],
    *,
    frame_count: int,
    maximum_control_step: int | None = None,
    require_replay_verified: bool = False,
) -> list[dict[str, Any]]:
    """Validate a canonical ordered ledger and reject duplicate event UIDs."""

    if isinstance(value, str):
        try:
            ledger = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("event ledger is not valid JSON") from exc
        if canonical_json(ledger) != value:
            raise ValueError("event ledger JSON is not canonical")
    elif isinstance(value, list):
        ledger = value
    else:
        raise ValueError("event ledger must be a canonical JSON string or list")
    validated = [
        validate_event_record(
            record,
            frame_count=frame_count,
            maximum_control_step=maximum_control_step,
            require_replay_verified=require_replay_verified,
        )
        for record in ledger
    ]
    uids = [record["event_uid"] for record in validated]
    if len(uids) != len(set(uids)):
        raise ValueError("event ledger contains duplicate event UIDs")
    if [int(record["request"]["row_index"]) for record in validated] != sorted(
        int(record["request"]["row_index"]) for record in validated
    ):
        raise ValueError("event ledger is not ordered by request row")
    return validated
