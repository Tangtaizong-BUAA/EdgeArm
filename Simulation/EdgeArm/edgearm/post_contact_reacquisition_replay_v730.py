"""History-aligned replay labels for learned post-contact reacquisition.

V729 selects the learned acquisition option after a confirmed contact loss.
The legacy V626 geometric gate can nevertheless remain exactly zero near the
precontact pose, excluding those same states from acquisition replay and
zeroing the acquisition actor's contribution during optimization.  V730
derives the V729 mode from already recorded episode history and exposes a
learning-only gate override for those rows.

No action, route, waypoint, future simulator state, or expert label is
constructed.  The only historical input is contact evidence available before
the recorded action (instantaneous contact or the preceding transition's
valid-contact result).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


POST_CONTACT_REACQUISITION_REPLAY_FORMAT_V730 = (
    "edgearm-v730-post-contact-reacquisition-replay-v1"
)


@dataclass(frozen=True)
class PostContactReacquisitionReplayConfigV730:
    no_contact_grace_steps: int = 3
    effective_acquisition_gate: float = 1.0
    reacquisition_sampling_fraction: float = 0.35
    replay_priority_increment: float = 4.0

    def validate(self) -> None:
        values = np.asarray(
            (
                self.effective_acquisition_gate,
                self.reacquisition_sampling_fraction,
                self.replay_priority_increment,
            ),
            dtype=np.float64,
        )
        if (
            type(self.no_contact_grace_steps) is not int
            or not 0 <= self.no_contact_grace_steps <= 30
            or not np.all(np.isfinite(values))
            or not 0.0 < self.effective_acquisition_gate <= 1.0
            or not 0.0 < self.reacquisition_sampling_fraction <= 0.50
            or self.replay_priority_increment < 0.0
        ):
            raise ValueError("V730 replay configuration is invalid")


def derive_post_contact_reacquisition_labels_v730(
    *,
    episode_index: np.ndarray,
    episode_step: np.ndarray,
    instantaneous_contact: np.ndarray,
    valid_contact: np.ndarray,
    config: PostContactReacquisitionReplayConfigV730 | None = None,
) -> dict[str, Any]:
    """Derive current/next V729 modes using only causally prior evidence."""

    selected = config or PostContactReacquisitionReplayConfigV730()
    if type(selected) is not PostContactReacquisitionReplayConfigV730:
        raise TypeError("V730 requires its exact replay configuration")
    selected.validate()
    episodes = np.asarray(episode_index)
    steps = np.asarray(episode_step)
    contact = np.asarray(instantaneous_contact)
    valid = np.asarray(valid_contact)
    count = len(episodes)
    if (
        episodes.shape != (count,)
        or steps.shape != (count,)
        or contact.shape != (count,)
        or valid.shape != (count,)
        or episodes.dtype.kind not in "iu"
        or steps.dtype.kind not in "iu"
        or contact.dtype != np.dtype(bool)
        or valid.dtype != np.dtype(bool)
        or count < 1
    ):
        raise ValueError("V730 replay arrays are invalid")

    current_mode = np.zeros(count, dtype=bool)
    next_mode = np.zeros(count, dtype=bool)
    mode_entry = np.zeros(count, dtype=bool)
    mode_exit_after_transition = np.zeros(count, dtype=bool)
    effective_contact = np.zeros(count, dtype=bool)
    no_contact_steps = np.zeros(count, dtype=np.int32)
    episode_with_entry_count = 0
    episode_with_recontact_count = 0

    for episode in np.unique(episodes):
        rows = np.flatnonzero(episodes == episode)
        if (
            not len(rows)
            or np.any(np.diff(rows) != 1)
            or np.any(np.diff(steps[rows]) != 1)
        ):
            raise ValueError("V730 replay episode ordering is invalid")
        seen_contact = False
        gap = 0
        previous_valid_contact = False
        previous_mode = False
        for row in rows:
            contact_now = bool(contact[row] or previous_valid_contact)
            effective_contact[row] = contact_now
            if contact_now:
                seen_contact = True
                gap = 0
            elif seen_contact:
                gap += 1
            no_contact_steps[row] = gap
            active = bool(
                seen_contact
                and gap > selected.no_contact_grace_steps
            )
            current_mode[row] = active
            mode_entry[row] = bool(active and not previous_mode)
            previous_mode = active
            previous_valid_contact = bool(valid[row])

        if np.any(mode_entry[rows]):
            episode_with_entry_count += 1
        for local_index, row in enumerate(rows[:-1]):
            following = rows[local_index + 1]
            next_mode[row] = current_mode[following]
            mode_exit_after_transition[row] = bool(
                current_mode[row] and not current_mode[following]
            )
        # The final next state is outside replay.  Its target is always masked
        # by the terminal flag, so fail closed instead of inventing history.
        next_mode[rows[-1]] = False
        mode_exit_after_transition[rows[-1]] = False
        if np.any(mode_exit_after_transition[rows]):
            episode_with_recontact_count += 1

    audit = {
        "format": POST_CONTACT_REACQUISITION_REPLAY_FORMAT_V730,
        "configuration": asdict(selected),
        "transition_count": count,
        "effective_contact_transition_count": int(
            np.count_nonzero(effective_contact)
        ),
        "confirmed_contact_loss_transition_count": int(
            np.count_nonzero(current_mode)
        ),
        "mode_entry_count": int(np.count_nonzero(mode_entry)),
        "mode_exit_after_transition_count": int(
            np.count_nonzero(mode_exit_after_transition)
        ),
        "episode_with_mode_entry_count": episode_with_entry_count,
        "episode_with_recontact_count": episode_with_recontact_count,
        "reacquisition_contact_transition_count": int(
            np.count_nonzero(current_mode & valid)
        ),
        "current_label_uses_future_transition_information": False,
        "current_label_uses_instantaneous_or_preceding_contact": True,
        "next_label_copied_from_next_recorded_state": True,
        "action_or_waypoint_label_created": False,
        "expert_action_used": False,
        "bulk_vla_data_use_allowed": False,
        "production_admission": False,
    }
    return {
        "current_mode": current_mode,
        "next_mode": next_mode,
        "mode_entry": mode_entry,
        "mode_exit_after_transition": mode_exit_after_transition,
        "effective_contact": effective_contact,
        "no_contact_steps": no_contact_steps,
        "audit": audit,
    }


__all__ = [
    "POST_CONTACT_REACQUISITION_REPLAY_FORMAT_V730",
    "PostContactReacquisitionReplayConfigV730",
    "derive_post_contact_reacquisition_labels_v730",
]
