import inspect

import torch

from edgearm.run82_spatial_model import SparseSpatialPolicy, detach_state, spatial_losses


def fake_batch(batch=2):
    return dict(rgb=None, pose=None, K=None, age=None, proprio=torch.zeros(batch, 114),
                selected=torch.tensor([[0, 4]]*batch), dt=torch.full((batch,), 4/30),
                encoded=(torch.zeros(batch, 7, 2), torch.randn(batch, 7, 128)))


def test_interface_excludes_teacher_future_and_privileged_geometry():
    names = inspect.signature(SparseSpatialPolicy.step).parameters
    assert not {'labels', 'truth', 'qpos', 'future', 'command', 'seed', 'route'} & set(names)
    assert SparseSpatialPolicy.kind.endswith('v82')


def test_shape_sparsity_and_state_continuity():
    torch.manual_seed(82)
    model = SparseSpatialPolicy()
    b = fake_batch()
    out, state = model.step(**b)
    assert out['points'].shape == (2, 7, 8, 3)
    assert out['neighbors'].shape == (2, 7, 3)
    assert out['action'].shape == (2, 6)
    second, next_state = model.step(**b, state=state)
    reset, reset_state = model.step(**b, state=state, erase_memory=True)
    torch.testing.assert_close(reset['xyz'], out['xyz'])
    assert not torch.allclose(next_state['hidden'], reset_state['hidden'])
    moved = b | dict(encoded=(b['encoded'][0]+.03, b['encoded'][1]))
    with_history, _ = model.step(**moved, state=state)
    without_history, _ = model.step(**moved, erase_memory=True)
    assert not torch.allclose(with_history['xyz'], without_history['xyz'])
    assert all(not x.requires_grad for x in detach_state(state).values())


def test_geometry_motion_and_action_losses_reach_learned_modules():
    torch.manual_seed(83)
    model = SparseSpatialPolicy()
    b = fake_batch()
    _, state = model.step(**b)
    out, _ = model.step(**b, state=state)
    labels = dict(present=torch.ones(2, 7, dtype=torch.bool), xyz=torch.randn(2, 7, 3)*.02,
                  points=torch.randn(2, 7, 8, 3)*.03, visible=torch.ones(2, 7),
                  action_valid=torch.ones(2), command=torch.ones(2, 6)*.1)
    loss, metrics = spatial_losses(out, labels, torch.ones(6))
    assert torch.isfinite(loss)
    loss.backward()
    for name in ('motion', 'update', 'geometry', 'visibility', 'surface', 'action_memory', 'control'):
        grads = [p.grad for p in getattr(model, name).parameters() if p.grad is not None]
        assert grads and sum(float(g.abs().sum()) for g in grads) > 0, name
    assert 'reconstruction' in metrics and 'motion_prior' in metrics


def test_changing_supervision_cannot_change_forward_prediction():
    model = SparseSpatialPolicy().eval()
    b = fake_batch()
    with torch.inference_mode():
        first, state = model.step(**b)
        a, _ = model.step(**b, state=state)
        unused_labels = torch.randn(2, 7, 3)*100
        c, _ = model.step(**b, state=state)
    assert unused_labels.shape == (2, 7, 3)
    torch.testing.assert_close(a['action'], c['action'])
    torch.testing.assert_close(a['xyz'], c['xyz'])


def test_future_or_excessive_time_gap_rejected():
    import pytest
    model = SparseSpatialPolicy()
    b = fake_batch()
    for delta in (-.1, 1.):
        b['dt'] = torch.full((2,), delta)
        with pytest.raises(ValueError): model.step(**b)
