"""Audit goal direction and condition balance in an admitted V22/V25 rollout."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .asymmetric_multiview_ppo_v1 import (
    sha256_file_v1,
    visual_geometry_target_from_privileged_v1,
)
from .goal_directed_feasible_on_policy_ppo_v23 import (
    TARGET_OFFSET_NORMALIZATION_M_V23,
    target_distance_progress_from_batch_v23,
)
from .feasible_on_policy_ppo_v21 import RolloutEligibilityConfigV21
from .v22_rollout_h5 import load_feasible_multiview_rollout_v22


GOAL_DIRECTED_ROLLOUT_AUDIT_FORMAT_V25 = (
    "edgearm-v25-goal-directed-rollout-condition-audit-v1"
)
BLOCK_MOTION_THRESHOLD_M_V25 = (
    RolloutEligibilityConfigV21().block_motion_threshold_m
)


@dataclass(frozen=True)
class FrontierConditionAdmissionConfigV25:
    minimum_complete_episodes_per_condition: int = 2
    minimum_active_episode_fraction: float = 0.50
    minimum_contact_transitions_per_condition: int = 2
    minimum_net_target_progress_m_per_condition: float = 0.001
    maximum_invalid_contact_fraction_per_condition: float = 0.02

    def validate(self) -> None:
        if self.minimum_complete_episodes_per_condition < 1:
            raise ValueError("V25 condition episode minimum must be positive")
        if self.minimum_contact_transitions_per_condition < 1:
            raise ValueError("V25 condition contact minimum must be positive")
        if not 0.0 < self.minimum_active_episode_fraction <= 1.0:
            raise ValueError("V25 active episode fraction must lie in (0,1]")
        if self.minimum_net_target_progress_m_per_condition <= 0.0:
            raise ValueError("V25 condition progress minimum must be positive")
        if not 0.0 <= self.maximum_invalid_contact_fraction_per_condition < 1.0:
            raise ValueError("V25 invalid contact fraction must lie in [0,1)")


def _distance_m(offset: np.ndarray) -> np.ndarray:
    scaled = np.multiply(
        np.asarray(offset, dtype=np.float32),
        TARGET_OFFSET_NORMALIZATION_M_V23,
        dtype=np.float32,
    )
    return np.linalg.norm(scaled, axis=1).astype(np.float32)


def audit_goal_directed_rollout_v25(rollout_path: Path) -> dict[str, Any]:
    path = Path(rollout_path).expanduser().resolve()
    batch = load_feasible_multiview_rollout_v22(path)
    progress = target_distance_progress_from_batch_v23(batch)
    before_distance = _distance_m(batch.visual_geometry_target[:, -2:])
    after_geometry = visual_geometry_target_from_privileged_v1(
        batch.next_privileged_state
    )
    after_distance = _distance_m(after_geometry[:, -2:])
    episodes: list[dict[str, Any]] = []
    for episode_id, record in enumerate(batch.episode_records):
        selected = batch.episode_ids == episode_id
        indices = np.flatnonzero(selected)
        if not len(indices):
            raise RuntimeError(f"V25 rollout episode {episode_id} has no rows")
        first = int(indices[0])
        last = int(indices[-1])
        episode_progress = progress[selected]
        contact = batch.valid_push_side_contact_any[selected]
        episodes.append(
            {
                "episode_id": episode_id,
                "selected_seed": int(record["selected_reset_seed"]),
                "obstacle": bool(record["obstacle_enabled"]),
                "stress": bool(record["stress_enabled"]),
                "rows": int(np.count_nonzero(selected)),
                "valid_contact_transitions": int(np.count_nonzero(contact)),
                "block_motion_transitions": int(
                    np.count_nonzero(
                        batch.step_block_displacement_m[selected]
                        > BLOCK_MOTION_THRESHOLD_M_V25
                    )
                ),
                "invalid_contact_transitions": int(
                    np.count_nonzero(batch.invalid_tool_block_contact_any[selected])
                ),
                "action_shield_rejection_transitions": int(
                    np.count_nonzero(batch.shield_rejected_before_step[selected])
                ),
                "safety_stop_transitions": int(
                    np.count_nonzero(batch.safety_stop[selected])
                ),
                "terminal_failure_transitions": int(
                    np.count_nonzero(batch.terminal_failure[selected])
                ),
                "ik_failure_transitions": int(
                    np.count_nonzero(~batch.ik_converged[selected])
                ),
                "strict_success": bool(np.any(batch.strict_success[selected])),
                "initial_target_distance_m": float(before_distance[first]),
                "final_target_distance_m": float(after_distance[last]),
                "net_target_progress_m": float(episode_progress.sum()),
                "positive_target_progress_transitions": int(
                    np.count_nonzero(episode_progress > 0.0)
                ),
                "target_regression_transitions": int(
                    np.count_nonzero(episode_progress < 0.0)
                ),
                "contact_target_progress_mean_m": (
                    float(episode_progress[contact].mean())
                    if np.any(contact)
                    else 0.0
                ),
                "minimum_safety_only_clearance_m": float(
                    np.min(
                        batch.minimum_executed_94_safety_only_block_clearance_m[
                            selected
                        ]
                    )
                ),
            }
        )
    aggregate = {
        "rows": len(batch.rewards),
        "complete_episodes": batch.completed_episode_count,
        "valid_contact_transitions": int(
            np.count_nonzero(batch.valid_push_side_contact_any)
        ),
        "block_motion_transitions": int(
            np.count_nonzero(
                batch.step_block_displacement_m > BLOCK_MOTION_THRESHOLD_M_V25
            )
        ),
        "block_motion_threshold_m": BLOCK_MOTION_THRESHOLD_M_V25,
        "invalid_contact_transitions": int(
            np.count_nonzero(batch.invalid_tool_block_contact_any)
        ),
        "action_shield_rejection_transitions": int(
            np.count_nonzero(batch.shield_rejected_before_step)
        ),
        "safety_stop_transitions": int(np.count_nonzero(batch.safety_stop)),
        "terminal_failure_transitions": int(
            np.count_nonzero(batch.terminal_failure)
        ),
        "ik_failure_transitions": int(np.count_nonzero(~batch.ik_converged)),
        "strict_success_episodes": int(
            sum(int(item["strict_success"]) for item in episodes)
        ),
        "net_target_progress_m": float(progress.sum()),
        "positive_target_progress_transitions": int(
            np.count_nonzero(progress > 0.0)
        ),
        "target_regression_transitions": int(np.count_nonzero(progress < 0.0)),
        "minimum_safety_only_clearance_m": float(
            np.min(batch.minimum_executed_94_safety_only_block_clearance_m)
        ),
    }
    return {
        "format": GOAL_DIRECTED_ROLLOUT_AUDIT_FORMAT_V25,
        "rollout_path": str(path),
        "rollout_sha256": sha256_file_v1(path),
        "aggregate": aggregate,
        "episodes": episodes,
        "actor_policy_inputs_include_privileged_geometry": False,
        "audit_uses_privileged_training_labels": True,
        "expert_calls": 0,
        "behavior_cloning_steps": 0,
        "physical_samples": 0,
        "production_admission": False,
    }


def frontier_condition_admission_v25(
    audit: dict[str, Any],
    config: FrontierConditionAdmissionConfigV25 | None = None,
) -> dict[str, Any]:
    """Require task-active evidence independently in every sampled condition."""

    selected_config = config or FrontierConditionAdmissionConfigV25()
    selected_config.validate()
    episodes = audit.get("episodes")
    aggregate = audit.get("aggregate")
    if not isinstance(episodes, list) or not episodes or not isinstance(aggregate, dict):
        raise TypeError("V25 condition admission requires a complete rollout audit")
    grouped: dict[tuple[bool, bool], list[dict[str, Any]]] = {}
    for episode in episodes:
        if not isinstance(episode, dict):
            raise TypeError("V25 condition admission episode is not a record")
        key = (bool(episode["obstacle"]), bool(episode["stress"]))
        grouped.setdefault(key, []).append(episode)
    condition_results: dict[str, Any] = {}
    all_checks: list[bool] = []
    for (obstacle, stress), rows in sorted(grouped.items()):
        episode_count = len(rows)
        active_episode_count = sum(
            int(
                int(row["valid_contact_transitions"]) > 0
                and float(row["net_target_progress_m"]) > 0.0
            )
            for row in rows
        )
        contact_count = sum(int(row["valid_contact_transitions"]) for row in rows)
        progress_m = sum(float(row["net_target_progress_m"]) for row in rows)
        invalid_count = sum(int(row["invalid_contact_transitions"]) for row in rows)
        transition_count = sum(int(row["rows"]) for row in rows)
        active_fraction = active_episode_count / episode_count
        invalid_fraction = invalid_count / transition_count
        checks = {
            "enough_complete_episodes": (
                episode_count
                >= selected_config.minimum_complete_episodes_per_condition
            ),
            "enough_task_active_episodes": (
                active_fraction >= selected_config.minimum_active_episode_fraction
            ),
            "enough_valid_contact": (
                contact_count
                >= selected_config.minimum_contact_transitions_per_condition
            ),
            "enough_net_target_progress": (
                progress_m
                >= selected_config.minimum_net_target_progress_m_per_condition
            ),
            "invalid_contact_fraction_within_bound": (
                invalid_fraction
                <= selected_config.maximum_invalid_contact_fraction_per_condition
            ),
            "no_action_shield_rejection": all(
                int(row["action_shield_rejection_transitions"]) == 0 for row in rows
            ),
            "no_safety_stop": all(
                int(row["safety_stop_transitions"]) == 0 for row in rows
            ),
            "no_terminal_failure": all(
                int(row["terminal_failure_transitions"]) == 0 for row in rows
            ),
        }
        all_checks.extend(checks.values())
        condition_results[
            f"obstacle={int(obstacle)},stress={int(stress)}"
        ] = {
            "episode_count": episode_count,
            "active_episode_count": active_episode_count,
            "active_episode_fraction": active_fraction,
            "valid_contact_transitions": contact_count,
            "net_target_progress_m": progress_m,
            "invalid_contact_transitions": invalid_count,
            "invalid_contact_fraction": invalid_fraction,
            "checks": checks,
        }
    return {
        "format": "edgearm-v25-frontier-condition-rollout-admission-v1",
        "config": asdict(selected_config),
        "condition_count": len(condition_results),
        "conditions": condition_results,
        "all_conditions_pass": all(all_checks),
        "production_admission": False,
    }
def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = audit_goal_directed_rollout_v25(args.rollout)
    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = Path(args.output).expanduser().resolve()
        if output.exists():
            raise FileExistsError(f"V25 rollout audit output exists: {output}")
        partial = output.with_suffix(output.suffix + ".partial")
        partial.write_text(encoded, encoding="utf-8")
        partial.replace(output)
    print(encoded, end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "BLOCK_MOTION_THRESHOLD_M_V25",
    "FrontierConditionAdmissionConfigV25",
    "GOAL_DIRECTED_ROLLOUT_AUDIT_FORMAT_V25",
    "audit_goal_directed_rollout_v25",
    "frontier_condition_admission_v25",
]
