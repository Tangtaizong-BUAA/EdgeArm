"""Explicit success quotas for ACT data; not an RL training profile."""
from collections import Counter


def strata(total=1000):
    if total < 100 or total % 100:
        raise ValueError('target must be a positive multiple of 100')
    unit = total // 100
    groups = [dict(name='familiar_center', pair=4, jitter_m=0., target=40 * unit)]
    groups += [dict(name=f'pair_{p // 3}_{p % 3}', pair=p, jitter_m=0., target=6 * unit)
               for p in range(9) if p != 4]
    groups += [dict(name=f'center_jitter_{mm}mm', pair=4, jitter_m=mm / 1000., target=4 * unit)
               for mm in (5, 10, 15)]
    return groups


def admission(result):
    """Do not confuse a valid terminal episode with a successful demonstration."""
    return bool(result.get('safe_success') and result.get('end_kind') == 'success'
        and result.get('maximum_hold_s', 0) >= 3. - 1e-6
        and result.get('initial_coverage', 1) == 0
        and result.get('training_rollin_steps', -1) == 0
        and result.get('rejected_commands', 1) == 0
        and result.get('causal_packet_audit', {}).get('passed') is True)


def counts(groups, records):
    out = {g['name']: dict(target=g['target'], attempts=0, accepted=0,
                          train=0, validation=0, failures={}) for g in groups}
    for r in records:
        c = out[r['stratum']]
        c['attempts'] += 1
        if r['accepted']:
            c['accepted'] += 1
            c[r['split']] += 1
        else:
            reason = r.get('reason', r['end_kind'])
            c['failures'][reason] = c['failures'].get(reason, 0) + 1
    return out


def next_wave(groups, records, capacity, *, phase, pilot_attempts=4,
              eligible=None, max_attempts_factor=6):
    c = counts(groups, records)
    pending = Counter()
    wave = []
    for _ in range(capacity):
        available = []
        for i, g in enumerate(groups):
            name, v = g['name'], c[g['name']]
            if phase == 'pilot':
                remaining = pilot_attempts - v['attempts'] - pending[name]
                priority = (v['attempts'] + pending[name], i)
            else:
                if name not in (eligible or set()):
                    continue
                remaining = min(v['target'] - v['accepted'] - pending[name],
                    max_attempts_factor * v['target'] - v['attempts'] - pending[name])
                priority = ((v['accepted'] + pending[name]) / v['target'], i)
            if remaining > 0:
                available.append((priority, g))
        if not available:
            break
        _, group = min(available, key=lambda x: x[0])
        name = group['name']
        wave.append(dict(group, attempt=c[name]['attempts'] + pending[name]))
        pending[name] += 1
    return wave


def replay_proposal():
    return dict(status='proposal_not_training_started', sampling_unit='episode_then_time',
                old_rl=.40, old_human_sim=.10, new_success=.50,
                failed_actions_are_bc_targets=False,
                old_validation_is_training_data=False,
                requires_original_and_edge_closed_loop_regression=True)


def teacher_wave(groups, records, capacity, eligible, in_flight=()):
    """Round-robin routes; low-yield routes get one diagnostic slot, not all CPUs."""
    summary=counts(groups,records)
    pending=Counter(j['stratum'] for j in in_flight)
    wave=[]
    for _ in range(capacity):
        candidates=[]
        for i,g in enumerate(groups):
            name=g['name']; c=summary[name]
            if name not in eligible or c['accepted']+pending[name]>=g['target']:
                continue
            if c['attempts']+pending[name]>=6*g['target']:
                continue
            poor=c['attempts']>=20 and c['accepted']/c['attempts']<.10
            if poor and pending[name]>=1:
                continue
            candidates.append(((pending[name],-c['accepted']/max(c['attempts'],1),i),g))
        if not candidates:
            break
        _,g=min(candidates)
        name=g['name']
        wave.append(dict(g,attempt=summary[name]['attempts']+pending[name]))
        pending[name]+=1
    return wave
