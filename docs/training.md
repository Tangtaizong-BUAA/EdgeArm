# Training mainline

This release exposes the actual staged method. It does not rename supervised
DAgger as reinforcement learning. Entry modules are unchanged from the local
research implementation and their hashes are retained.

## 1. Prepare sequences and learn spatial memory

Use `edgearm.run82_prepare_sequences` to replay the color-consistent collection.
It writes causal observations and privileged labels separately. Use
`edgearm.run82_train_spatial` with continuous chunks of length 12, carry memory
between chunks, detach at chunk boundaries, and sample one sequence per route.
The network has seven 128-dimensional slots, top-three sparse neighbors, GRU
updates, action-conditioned motion, visibility gating, and sparse point supervision.

The loss is
`Lpos + 0.25 Lmotion + 0.15 Lpoints + 0.15 Lvis + 0.01 Lunc + lambda Laction`.
Position/point scales are 20 mm. AdamW learning rates are 1e-5 for vision,
3e-5 for the base action MLP, and 2e-4 for new spatial modules. The final candidate
retains spatial update 500. The auxiliary memory action head is not its final actor.

## 2. On-policy correction / DAgger

`edgearm.run94_control_fit` freezes perception and fits the 118→512→512→512→6
SiLU action network. It collects three sets of 36 physical attempts, with teacher
execution probabilities 0.5/0.25/0 in 15-step segments. Zero teacher execution
still permits training-only label queries. Fixed observation/completion remain.

Each optimizer draw has 576 new states (64 per route) and 288 old states
(32 per route), with 2 mm noise on the new estimated positions.
`Lteacher + 0.2 Lreplay + 0.02 Lanchor` uses normalized Smooth-L1 losses.
AdamW 2e-5, gradient clipping 1, 3×1200 updates. The actual subsequent initializer
was round 2, not automatically the final round or the nominal best field.

## 3. Make teacher labels execution-consistent

`edgearm.run100_feasible_labels` projects existing teacher queries using the same
joint camera/workspace constraint as deployment. SLSQP receives reported joints,
velocity, and static geometry; it does not read object truth. Original image-derived
118-dimensional inputs stay unchanged. The recorded run accepted 46,532 queries,
discarded 15, and changed 947 by more than 1 mrad.

Perception is frozen. AdamW 1e-5, the same replay/anchor loss, two rounds of 600
updates; round 1 was retained. Projection feasibility is not a physical success
guarantee. Never loosen the simulator's collision or three-second criterion to
make an optimization run appear successful.

## 4. Adapt current-frame perception

`edgearm.run101_keypoint_recovery` replays the same 108 training attempts and
adapts the keypoint network. New RGB has 7,153 training / 1,683 validation frames;
old RGB has 28,840 / 1,586. Draw 72 frames from each source, eight per route.
Optimize visible/absent heatmap cross-entropy plus 0.5 normalized pixel Smooth-L1.
AdamW 1e-4, weight decay 1e-4, gradient clipping 5; evaluate every 600 updates.
The recorded run stopped at 1200, not its planned 1800 maximum.

The final actor uses current keypoints each control step and learned spatial
memory every eighth step. Calibrated rays, nominal object height, confidence
fusion, static-goal locking, joint projection, and observation/completion programs
are explicit engineering components, not learned depth or a full 4D reconstruction.

## 5. Model selection and evaluation

Development gate: 36 complete episodes, at least 26 successes, at most one hard
failure, and zero out-of-bounds episodes. Run101 achieved 28/36 before the final
72-episode independent evaluation. Repeated development rounds do not increase
the independent sample size. Use `edgearm.run102_policy_acceptance` for new
research with its source/hash/development binding; use the public `edgearm evaluate`
command for portable benchmark reproduction.

## Execution recipes

`recipes/*.json` are print-first CLI recipes. Relative paths resolve against the
repository root. Paths under `data/` and `checkpoints/` must be populated locally.
The historical fitting modules retain bounded wall times and fixed seed cohorts.
They are not a generic arbitrary-dataset trainer. To change cohorts or action
semantics, change the contract and its tests together and give the experiment a
new name. Exact retraining requires the private corpus and original initializers;
using the released final weights as an initializer is a new fine-tuning experiment.

For historical ACT use `edgearm.train_sparse_4d_vla_act_v26 --help`.
For the real online residual RL experiment use `edgearm.run80_group_residual_rl --help`.
These are research baselines, not required extra stages after the released final model.

