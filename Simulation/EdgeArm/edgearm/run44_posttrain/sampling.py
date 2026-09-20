"""Same global source schedule on both ranks, then disjoint rank slicing."""
from collections import defaultdict
import numpy as np
from torch.utils.data import Sampler
from .data import COHORTS


def pair_probabilities(counts, uniform_mix, max_oversampling):
    counts = np.asarray(counts, dtype=float)
    if (counts.ndim != 1 or not len(counts) or np.any(counts <= 0)
            or not 0 <= uniform_mix <= 1 or max_oversampling < 1):
        raise ValueError('invalid bounded pair sampling configuration')
    empirical = counts / counts.sum()
    cap = max_oversampling * empirical
    result = np.minimum((1-uniform_mix)*empirical+uniform_mix/len(counts), cap)
    # Redistribute only into remaining headroom, never back into capped pairs.
    headroom = cap-result
    remaining = 1-result.sum()
    if remaining > 1e-14:
        result += remaining*headroom/headroom.sum()
    return result


class BalancedBatches(Sampler):
    def __init__(self, records, mass, *, batch_size, batches, rank=0, world_size=2,
                 seed=4401, stride=2, uniform_mix=.3, max_oversampling=3.):
        if set(mass)!=set(COHORTS) or not np.isclose(sum(mass.values()),1) or min(mass.values())<=0:
            raise ValueError('explicit five-way rehearsal mix required')
        self.records, self.mass = records, mass
        self.batch_size, self.batches, self.rank, self.world_size = batch_size, batches, rank, world_size
        self.seed, self.epoch, self.stride = seed, 0, stride
        self.uniform_mix, self.max_oversampling = uniform_mix, max_oversampling
        self.pools = defaultdict(lambda: defaultdict(list))
        for i,r in enumerate(records):
            if r['split']!='train':
                raise ValueError('validation trajectory in optimizer sampler')
            if not r['valid_times']:
                raise ValueError('empty action trajectory')
            self.pools[r['cohort']][r.get('stratum') or r.get('pair') or 'human'].append(i)
        missing = set(COHORTS)-set(self.pools)
        if missing:
            raise ValueError('missing rehearsal cohorts; do not redistribute silently: '+str(sorted(missing)))
        self.source_cycle = [c for c in COHORTS for _ in range(round(100*mass[c]))]
        if len(self.source_cycle)!=100:
            raise ValueError('source mass must be representable in percentage points')

    def __len__(self):
        return self.batches

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed+100003*self.epoch)
        source_cycle, cursor, decks = [], 0, {}
        pairs, probabilities = {}, {}
        for c, pool in self.pools.items():
            pairs[c] = sorted(pool)
            probabilities[c] = pair_probabilities([len(pool[k]) for k in pairs[c]],
                self.uniform_mix, self.max_oversampling)
        for _ in range(self.batches):
            global_batch = []
            for _ in range(self.batch_size*self.world_size):
                if cursor>=len(source_cycle):
                    source_cycle = rng.permutation(self.source_cycle).tolist()
                    cursor=0
                cohort = source_cycle[cursor]
                cursor += 1
                pair = pairs[cohort][int(rng.choice(len(pairs[cohort]),p=probabilities[cohort]))]
                key = cohort,pair
                if not decks.get(key):
                    decks[key] = rng.permutation(self.pools[cohort][pair]).tolist()
                ri = decks[key].pop()
                times = self.records[ri]['valid_times'][::self.stride]
                # Uniform progress thirds prevents long middle sections from
                # removing approach/hold supervision. It does not invent recovery.
                chunks = [x for x in np.array_split(times,3) if len(x)]
                ts = chunks[int(rng.integers(len(chunks)))]
                t = int(ts[int(rng.integers(len(ts)))])
                global_batch.append((ri,t))
            yield global_batch[self.rank::self.world_size]


class ValidationBatches(Sampler):
    def __init__(self, records, batch_size, rank=0, world_size=1, windows_per_episode=16):
        self.indices = []
        self.batch_size = batch_size
        for ri,r in enumerate(records):
            if r['split']!='validation':
                raise ValueError('training episode in validation loader')
            if ri % world_size != rank:
                continue
            ts = r['valid_times']
            ids = np.unique(np.linspace(0,len(ts)-1,min(windows_per_episode,len(ts)),dtype=int))
            self.indices.extend((ri,ts[i]) for i in ids)

    def __len__(self):
        return (len(self.indices)+self.batch_size-1)//self.batch_size

    def __iter__(self):
        for i in range(0,len(self.indices),self.batch_size):
            yield self.indices[i:i+self.batch_size]
