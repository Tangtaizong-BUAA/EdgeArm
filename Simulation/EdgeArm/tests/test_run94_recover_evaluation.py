import json

import pytest

from edgearm.run94_recover_evaluation import MODE, SEEDS, completed_results, select_candidate


def test_preserves_only_complete_expected_episodes(tmp_path):
    directory = tmp_path / MODE / f"episode_{SEEDS[0]}"
    directory.mkdir(parents=True)
    row = dict(
        seed=SEEDS[0], mode=MODE, actor_uses_simulator_state=False, teacher_assisted=False, max_steps=900
    )
    (directory / "result.json").write_text(json.dumps(row))
    with pytest.raises(ValueError):
        completed_results(tmp_path)
    (directory / "trace.npz").write_bytes(b"fixture")
    assert completed_results(tmp_path) == [row]
    assert len(SEEDS) == len(set(SEEDS)) == 36
    assert all(97100000 <= seed // 9 < 97100004 for seed in SEEDS)


def test_gate_is_not_relaxed_for_more_unsafe_successes():
    def row(success, hard, path):
        return dict(
            episodes=36,
            successes=success,
            hard_failures=hard,
            block_out_of_bounds=0,
            mean_coverage=0.9,
            checkpoint=path,
        )

    best, _, passed = select_candidate([row(21, 1, "first"), row(26, 3, "second")])
    assert best == "first" and not passed
    best, _, passed = select_candidate([row(26, 3, "second"), row(26, 1, "third")])
    assert best == "third" and passed
    incomplete = row(12, 0, "partial") | {"episodes": 14}
    with pytest.raises(ValueError):
        select_candidate([incomplete])
