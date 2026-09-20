# EdgeArm Run102 model card

License: Apache-2.0. Model repository: `YuxuanGong/EdgeArm-Run102`.
Four exact checkpoint files are released with SHA-256 checksums:

| File | Role |
|---|---|
| `checkpoint.pt` | Run82 update-500 learned spatial memory, including its observation backbone |
| `control.pt` | Run100 round-1 action MLP |
| `keypoint.pt` | Run101 update-1200 current-frame keypoints |
| `vision.pt` | Historical compatible vision object loaded by the frozen interface; not an extra active ensemble |

These are the exact frozen research checkpoints; inspect them using PyTorch's
`weights_only=True`, never load arbitrary untrusted pickle files. Source hashes and
model hashes are separate: portable release wrappers are new code, not part of the
original recorded 72 episodes.

## Observations and actions

Wrist RGB + reported joints/velocities/FK + causal action/joint history + camera
calibration + explicit seven-color instruction parsing. The controller receives
114 proprioceptive/history values and four *estimated* block/target XY values.
Its six normalized joint command outputs pass through the existing command
contract and joint/workspace/camera projection. No ground-truth object coordinates,
teacher actions, route IDs, or future frames are deployment inputs.

## Evaluation

Nominal MuJoCo simulation, task-conditioned `CONTACT_TRANSPORT_HOLD` initial pose,
nine equally weighted routes, eight held-out scene groups, 72 complete episodes.
51 succeeded, 20 timed out, one went out of bounds, zero hard-contact failures.
All successes met the original coverage, speed, and consecutive three-second hold
criterion. There is a fixed 220-step observation program inside the 900-step total.

Wilson approximate 95% interval: 59.49–80.06%; scene-group bootstrap: 63.89–77.78%.
No real-robot, exact-Home, broad language, or fully randomized-environment result is
claimed. The weights are a research release, not hardware safety certification.
Training data remains private by owner choice. Production admission remains false.

