"""Current-policy corrections with route-balanced, explicit old-data rehearsal."""
from collections import defaultdict

import numpy as np
from torch.utils.data import Sampler


class OnPolicyRecoveryBatches(Sampler):
    """10% human; equal routes with 55% fresh, 25% past recovery, 20% old RL.

Half of fresh examples come from the first 32 teacher-controlled transitions,
where the state still reflects the current ACT's actual error. No failed ACT
prefix action is sampled as a positive label.
"""
    def __init__(self, records, batch_size, batches, seed):
        self.records, self.batch_size, self.batches, self.seed = records, batch_size, batches, seed
        self.pools = defaultdict(list)
        for i, r in enumerate(records):
            if r['split'] != 'train' or not r['valid_times']:
                raise ValueError('train trajectories with valid labels required')
            kind = 'fresh' if r.get('run59_fresh') else r['run54_pool']
            route = 'human' if kind == 'human' else r['pair']
            if kind == 'fresh' and (r['teacher_start_step'] <= 0 or
                    min(r['valid_times']) < r['teacher_start_step']):
                raise ValueError('fresh labels must exclude actual ACT roll-in')
            self.pools[(route, kind)].append(i)
        if not self.pools[('human', 'human')]:
            raise ValueError('human rehearsal required')
        self.missing_fresh_routes = []
        for r in range(9):
            for kind in ('recovery', 'old'):
                if not self.pools[(f'{r//3},{r%3}', kind)]:
                    raise ValueError(f'missing route {r} source {kind}; do not hide coverage gaps')
            if not self.pools[(f'{r//3},{r%3}', 'fresh')]:
                self.missing_fresh_routes.append(r)
        if len(self.missing_fresh_routes) > 3:
            raise ValueError('fresh corrections must cover at least six routes')

    def __len__(self):
        return self.batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        cycle, cursor, decks = [], 0, {}
        for _ in range(self.batches):
            batch = []
            for _ in range(self.batch_size):
                if cursor == len(cycle):
                    cycle, cursor = rng.permutation(np.repeat(np.arange(10), 10)).tolist(), 0
                route, cursor = cycle[cursor], cursor+1
                if route == 9:
                    key = ('human', 'human')
                else:
                    u = rng.random()
                    kind = 'fresh' if u < .55 else 'recovery' if u < .8 else 'old'
                    if kind == 'fresh' and route in self.missing_fresh_routes:
                        kind = 'recovery'  # Reported fallback, not a fabricated fresh example.
                    key = (f'{route//3},{route%3}', kind)
                if not decks.get(key):
                    decks[key] = rng.permutation(self.pools[key]).tolist()
                ri = decks[key].pop()
                row = self.records[ri]
                times = row['valid_times']
                if row.get('run59_fresh') and rng.random() < .5:
                    times = times[:32]
                batch.append((ri, int(rng.choice(times))))
            yield batch


class NominalRecoveryBatches(Sampler):
    """Run60 data-only ablation: do not mix legacy plant/feedback domains.

Keep the Run59 fresh-to-previous recovery ratio (0.495:0.225), its boundary
oversampling, all nine routes, and the original submitted-command labels.
Legacy data is not deleted or silently relabeled; it is outside this experiment.
"""
    fresh_probability = .495/(.495+.225)
    allow_missing_fresh = False
    def __init__(self, records, batch_size, batches, seed):
        self.records, self.batch_size, self.batches, self.seed = records, batch_size, batches, seed
        self.pools = defaultdict(list)
        self.missing_fresh_routes = []
        for i, r in enumerate(records):
            if r['split'] != 'train' or r['run54_pool'] != 'recovery' or not r['valid_times']:
                raise ValueError('only verified nominal training recoveries in this ablation')
            if min(r['valid_times']) < r['teacher_start_step']:
                raise ValueError('failed roll-in cannot be an action label')
            self.pools[(r['pair'], bool(r.get('run59_fresh')))].append(i)
        for route in range(9):
            for fresh in (False, True):
                if not self.pools[(f'{route//3},{route%3}', fresh)]:
                    if fresh and self.allow_missing_fresh:
                        self.missing_fresh_routes.append(route)
                        continue
                    raise ValueError('preserve both recovery cohorts on all nine routes')
        if len(self.missing_fresh_routes) > 3:
            raise ValueError('fresh corrections must cover at least six routes')

    def __len__(self):
        return self.batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        cycle, cursor, decks = [], 0, {}
        for _ in range(self.batches):
            batch = []
            for _ in range(self.batch_size):
                if cursor == len(cycle):
                    cycle, cursor = rng.permutation(np.repeat(np.arange(9), 10)).tolist(), 0
                route, cursor = cycle[cursor], cursor+1
                fresh = rng.random() < self.fresh_probability
                if route in self.missing_fresh_routes:
                    fresh = False
                key = (f'{route//3},{route%3}', fresh)
                if not decks.get(key):
                    decks[key] = rng.permutation(self.pools[key]).tolist()
                ri = decks[key].pop()
                times = self.records[ri]['valid_times']
                if fresh and rng.random() < .5:
                    times = times[:32]
                batch.append((ri, int(rng.choice(times))))
            yield batch


class SurveyRecoveryBatches(NominalRecoveryBatches):
    """50% current survey recoveries / 50% previous nominal recoveries.

    A missing new route explicitly falls back to its previous successful data;
    it never disappears from the nine-route denominator or evaluation.
    """
    fresh_probability = .5
    allow_missing_fresh = True


def survey_training_rows(rows):
    """Use the shared sampler flag without changing source metadata on disk."""
    out = []
    for row in rows:
        fresh = bool(row.get('run62_fresh'))
        if fresh and (not row.get('active_wrist_survey') or row.get('teacher_start_step') not in (220, 250)
                      or min(row['valid_times'], default=-1) < row['teacher_start_step']):
            raise ValueError('survey recovery needs matching real history and suffix-only labels')
        out.append(row | dict(run59_fresh=fresh))
    return out
