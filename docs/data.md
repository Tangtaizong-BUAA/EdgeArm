# Data contract

## Scope and access

Training data remains private. The public repository contains no captured robot
trajectories or recorded RGB, including failed trajectories. Dataset schemas and
the code to create your own data are public. Model weights are released separately.

Do not sum physical episodes, recolored sequences, sampled frames, teacher queries,
and optimizer draws. They measure different things.

## Sources

| Source | Entry point | Use |
|---|---|---|
| Human simulation teleoperation | `edgearm.run_keyboard_generalization_batch_v1` | Human commands and causal state records |
| Deferred RGB materialization | `edgearm.materialize_keyboard_strict_rgbd_batch_v40` | Render/verify recorded simulation data |
| Policy/teacher simulation collection | `edgearm.run43_data_collection`, `edgearm.run46_hard_collect` | Historical bulk and difficult-path collection |
| Color-consistent recovery data | `edgearm.run76_color_dagger` | Physical trajectories and recolored observations |
| Student-state teacher queries | `edgearm.run94_control_fit` / `run94_collect` | DAgger correction labels, including unsuccessful source episodes |
| Replay and geometric labeling | `edgearm.run82_prepare_sequences`, `run101_replay_keypoint_labels` | Separate causal inputs from training-only targets |

Historical collection modules may require their matching checkpoints and manifests.
Their presence is not evidence that every historical dataset is available.
Real-robot data requires separate hardware calibration and validation; the released
benchmark is simulation-only.

## Spatial sequence format

`manifest.json` contains `records`: each record has `seed`, `route`, `variant`,
`split`, and `folder`. `folder` points to the following pair:

| File | Keys | Meaning |
|---|---|---|
| `inputs.npz` | `rgb`, `pose`, `K`, `proprio`, `time_step`, `selected` | Only deployable observations |
| `labels.npz` | `xyz`, `points`, `present`, `visible`, `command`, `action_valid` | Training-only supervision |

RGB is a time sequence of wrist images; `pose` is estimated from reported joints
and calibration; `K` is the camera intrinsic matrix. `proprio` has 114 causal
joint/FK/history values, `selected` has two color indices. Time indices increase
strictly. Seven semantic slots contain XYZ and eight surface points each. Simulator
centers and visibility labels must never be inserted into `inputs.npz`.

The historical spatial split is scene groups 100200000–100200009 for training and
100200010–100200011 for perception validation. Color variants stay with their
physical source episode. Recoloring is not another independent trajectory.

## Action correction format

`action_labels.npz` stores `x[N,118]`, `command[N,6]`, and `time_step[N]`.
The first 114 input values are causal proprioception/history, the last four are
estimated selected-block/target XY. Store actor inputs before querying the teacher.
Commands use the original versioned command contract; do not reinterpret them as
raw radians without that transform. Only recorded training groups
100500000–100500011 are accepted by the historical label-alignment recipe.

Teacher labels on failed source trajectories are usable local corrections, not
certified full successful demonstrations. Failed student actions are not positive
imitation labels. Functional replay uses the frozen old policy's predictions,
not fabricated teacher success labels.

## Visual recovery format

Run101 replays executed commands, checks reported joints, block position and RGB
reproduction, and samples every eighth frame. Deployment inputs and simulator
training labels are written separately. Groups 100500010–100500011 are held out
from this perception fit, but were previously used for action learning: they are
**not** independently held-out policy tests.

## Public evaluation split

The recorded result used scene groups 98000100–98000107, nine routes per group.
They are now public benchmark seeds; reuse measures benchmark reproduction, not
fresh independent generalization. Do not fit or select checkpoints on them and
call the result independent. New research must pre-register separate held-out
groups and freeze a candidate before evaluation.

## Interoperability

The native NPZ records preserve causal inputs, labels, and command provenance.
`edgearm.export_lerobot` is a historical exporter, not a promise that every native
sequence already matches the latest LeRobotDataset schema. Validate units, action
semantics, timestamps, and episode splits before exporting to another framework.

