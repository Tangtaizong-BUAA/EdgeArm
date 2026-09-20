"""Account for invalid initial IK states without fabricating transitions."""
import ast
from pathlib import Path

from ..train_staged_hybrid_contact_sac import _atomic_json

PREFIX = 'V7 reset exhausted deterministic clean-state resampling: '


def ik_reset_audit(error):
    if not isinstance(error, RuntimeError) or not str(error).startswith(PREFIX):
        return None
    try:
        audit = ast.literal_eval(str(error)[len(PREFIX):])
    except (ValueError, SyntaxError):
        return None
    if (not isinstance(audit, dict)
            or audit.get('reset_failure_reasons') != ['task_aligned_reset_ik_not_converged']
            or audit.get('forbidden_penetration_count') != 0
            or audit.get('forbidden_contacts') != []):
        return None
    return audit


def record_invalid_reset(job, audit):
    path = Path(job['output'])
    path.mkdir(parents=True, exist_ok=False)
    result = dict(safe_success=False, end_kind='invalid_reset', steps=0,
        reason='task_aligned_reset_ik_not_converged', invalid_reset=True,
        training_rollin_steps=0, initial_coverage=0., maximum_coverage=0.,
        maximum_hold_s=0., valid_contact_steps=0, safety_rewrite_steps=0,
        final_reward_state=[0., 0., 0.], domain_stage=job['parameters']['stage'],
        domain_seed=job['parameters']['seed'], seed=job['seed'],
        reset_audit=audit, excluded_from_replay=True, counts_as_evaluation_failure=True)
    _atomic_json(path/'scenario.json', dict(parameters=job['parameters'], seed=job['seed'],
        sampling_seed=job['sampling_seed'], source=job['source'],
        policy_version=job['policy_version'], reset_audit=audit))
    _atomic_json(path/'result.json', result)
    _atomic_json(path/'episode_progress.json', result)
    return dict(result=result, batch=None, invalid_reset=True, episode_id=job['output'],
                policy_version=job['policy_version'])
