"""Causal command labels, deliberately distinct from realized-motion history."""
import numpy as np

ACTION_CONTRACT = {
    'version': 'edgearm-submitted-joint-command-v2',
    'output': 'normalized_submitted_relative_joint_target_command',
    'joint_count': 6,
    'max_joint_delta_rad': 0.055,
    'reference': 'env._command_reference_reported_position',
    'bounds': [-1.0, 1.0],
    'rl_alignment': 'image_state[t] -> submitted_joint_command[t+1]',
    'human_alignment': 'image_q_before[t] -> submitted_normalized_action[t]',
    'history': 'past_completed_realized_joint_delta_div_0.1',
    'controller': 'env.step(predicted_command); no realized-delta conversion',
}


def aligned_commands(commands, source, length, source_scale):
    if not np.isclose(source_scale, ACTION_CONTRACT['max_joint_delta_rad'], rtol=0, atol=1e-12):
        raise ValueError('source command units differ; explicit conversion/audit required')
    commands = np.asarray(commands, dtype=np.float64)
    if source == 'rl':
        if commands.shape != (length + 1, 6):
            raise ValueError('RL command rows must include the initial sentinel')
        if not np.all(commands[0] == 0):
            raise ValueError('RL initial command sentinel must be zero')
        commands = commands[1:]
    elif source != 'human':
        raise ValueError('unknown source')
    if commands.shape != (length, 6) or not np.isfinite(commands).all():
        raise ValueError('malformed command labels')
    if np.max(np.abs(commands), initial=0) > 1 + 1e-6:
        raise ValueError('command outside environment action bounds')
    return commands.astype(np.float32)


def execution_command(prediction, contract, env_scale):
    if contract != ACTION_CONTRACT:
        raise ValueError('checkpoint action contract missing or incompatible')
    if not np.isclose(env_scale, contract['max_joint_delta_rad'], rtol=0, atol=1e-12):
        raise ValueError('deployment command units differ')
    x = np.asarray(prediction, dtype=np.float64)
    if x.shape != (6,) or not np.isfinite(x).all():
        raise ValueError('nonfinite/malformed model command')
    return np.clip(x, -1, 1)
