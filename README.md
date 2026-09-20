# EdgeArm

**Single-wrist visual block pushing through spatial memory, on-policy correction,
execution-consistent labels, and perception adaptation.**

[中文说明](README.zh-CN.md) · [Training pipeline](docs/training.md) ·
[Data contract](docs/data.md) · [Model card](docs/model-card.md) ·
[Checkpoints](https://huggingface.co/YuxuanGong/EdgeArm-Run102)

EdgeArm studies a concrete task: select the block named by a color instruction,
push it to the named target region, and maintain success for **3 seconds**.
Three blocks and three non-overlapping target regions form nine equally weighted
start–target routes. The deployed policy uses wrist RGB, reported joints,
action history, and calibration—not simulator object coordinates.

## Final method

1. **Learn spatial memory:** continuous sequences supervise motion-conditioned
   seven-slot memory, visibility, sparse geometry, and auxiliary actions.
2. **Collect visited-state corrections:** query a recovery teacher on states
   reached by the student; anneal teacher execution while retaining old-policy replay.
3. **Align labels with execution:** project teacher targets jointly into the
   declared camera-clearance, workspace, and joint constraints; fit the action MLP.
4. **Adapt perception:** replay training commands, verify RGB/state reproduction,
   label keypoints separately, and balance new/old visual sources.
5. **Freeze and evaluate:** select on complete development episodes, then run a
   frozen candidate on reserved scene groups.

The final system is a **modular visual policy**, not an end-to-end ACT checkpoint.
Historical ACT and online residual-RL code is included for the data pipeline and
research lineage; the final fitting stages are supervised learning / DAgger.

## Recorded result

| Evaluation | Successes | Rate |
|---|---:|---:|
| Frozen Run102, 8 held-out scene groups × 9 routes | 51 / 72 | 70.83% |
| Per-route successes, each out of 8 | 6, 8, 2, 6, 8, 8, 1, 7, 5 | Equal route weights |

There were 20 timeouts, one block out of bounds, and zero hard-contact failures.
The approximate episode-level Wilson 95% interval is 59.49–80.06%; this is not an
80% success claim. This benchmark starts at a task-conditioned
`CONTACT_TRANSPORT_HOLD` pose, includes a fixed 220-step observation program and
scripted completion inside a 900-step budget, and is simulation-only.

## Install from a checkout

Linux + NVIDIA GPU is the training/rendering target. Install a CUDA-compatible
PyTorch wheel for your machine first, then:

```bash
git clone https://github.com/Tangtaizong-BUAA/EdgeArm.git
cd EdgeArm
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,hub]'
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
edgearm doctor
pytest -q
```

Keep the checkout: simulation asset paths are relative to `Simulation/SO101`.
Editable installation is supported; a standalone asset-containing wheel is not
currently promised. On macOS, CPU unit tests do not require EGL; rendering needs
an appropriate macOS OpenGL context. Do not use the Linux EGL settings there.

## Download and evaluate weights

The same four Apache-2.0 checkpoints are also available directly in
[GitHub Release v0.1.0](https://github.com/Tangtaizong-BUAA/EdgeArm/releases/tag/v0.1.0),
alongside `checksums.json`. Download all four `.pt` files into `checkpoints/run102`.
These are identical to the Hugging Face weights; training data is not included.

```bash
edgearm download --output checkpoints/run102
edgearm evaluate --weights checkpoints/run102 --output outputs/benchmark-replay \
  --workers 4 --groups 8 --group-start 98000100
```

This reproduces **published benchmark seeds**, not a fresh independent test.
`edgearm evaluate` checks checkpoint hashes, saves a source fingerprint, and
explicitly labels results as benchmark replay. Exact scores may vary with the
physics/numerical/runtime stack. The original guarded independent acceptance
entry remains available separately.

## Data and training

**Training data is not publicly distributed.** The source includes collection,
rendering, labeling, split guards, training, evaluation, and export components.
To train, supply locally collected data in the documented format; check commands
before running them:

```bash
edgearm stages
edgearm run recipes/spatial_memory.json
# Print-only by default. Pass --execute to launch after reviewing paths.
```

The four released weights reproduce the final candidate. Exact historical
retraining additionally requires private training data and historical initializer
weights; availability of code is not a claim that those files are public.
No recorded training samples, private run logs, SSH settings, or credentials are
included. The tiny test fixture is synthetic and is never counted as robot data.

## Repository map

```text
Simulation/EdgeArm/edgearm/       dependency-closed research implementation
Simulation/EdgeArm/edgearm_release/  public CLI and validation utilities
Simulation/SO101/                robot MJCF and mesh assets
recipes/                        explicit, reviewable stage commands
docs/                           data, method, model, and reproduction contracts
tests/                          synthetic release-level tests
provenance/                     source/checkpoint checksums and public metrics
```

Research `runNN` module names remain stable to preserve checkpoint and experiment
traceability. Public stage names hide them only at the CLI boundary; we have not
renamed the internals and silently claimed numerical equivalence.

## License and attribution

Code and the released model weights are Apache-2.0. Upstream SO-ARM100 assets
retain their license and attribution; see [NOTICE](NOTICE) and
[third-party notes](docs/third-party.md). No license to the non-public training
dataset is granted by this repository.

Release organization follows the code/data/model separation used by
[LeRobot](https://github.com/huggingface/lerobot) and the explicit training/evaluation
entry points of [ACT](https://github.com/tonyzhaozh/act). This is not an official
release of either project. Please cite [CITATION.cff](CITATION.cff).
