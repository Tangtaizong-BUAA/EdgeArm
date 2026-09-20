import inspect

import torch

from edgearm.run88_keypoint_model import WristKeypoints, project_labels, backproject, keypoint_loss


def camera():
    rotation = torch.eye(3)  # OpenGL camera looks down its negative Z axis.
    pose = torch.cat((torch.tensor([0.3, 0.0, 0.4]), rotation.flatten()))[None]
    K = torch.tensor([[[130.0, 0.0, 79.5], [0.0, 130.0, 59.5], [0.0, 0.0, 1.0]]])
    return pose, K


def test_known_plane_projection_roundtrip():
    pose, K = camera()
    xyz = torch.tensor([[[0.3, 0.0, 0.051], [0.34, 0.02, 0.026]]])
    uv, visible = project_labels(xyz, pose, K)
    restored, valid = backproject(uv, pose, K, xyz[0, :, 2])
    assert visible.all() and valid.all()
    torch.testing.assert_close(restored, xyz, atol=1e-6, rtol=1e-6)


def test_backward_and_deployable_signature():
    torch.manual_seed(88)
    model = WristKeypoints()
    pose, K = camera()
    rgb = torch.randint(0, 256, (1, 120, 160, 3), dtype=torch.uint8)
    output = model(rgb, pose, K)
    xyz = torch.tensor([0.3, 0.0, 0.051])[None, None].expand(1, 7, 3)
    mask = torch.tensor([[1, 1, 0, 0, 1, 1, 1]], dtype=torch.bool)
    loss, _ = keypoint_loss(output, xyz, mask, mask, pose, K)
    loss.backward()
    assert torch.isfinite(loss) and any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
    )
    assert output["uv"].shape == (1, 7, 2)
    assert (output["confidence"] >= 0).all() and (output["confidence"] <= 1).all()
    assert set(inspect.signature(model.forward).parameters) == {"rgb", "pose", "K"}


def test_behind_camera_labels_are_absent():
    pose, K = camera()
    _, valid = project_labels(torch.tensor([[[0.3, 0.0, 0.5]]]), pose, K)
    assert not valid.any()
