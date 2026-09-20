# Reproducibility and release status

## Included

- Dependency-closed implementation of the final spatial-memory, DAgger,
  feasible-label and keypoint-adaptation chain, plus relevant historical data/ACT/RL modules.
- SO101 simulation assets with upstream attribution.
- Print-first stage recipes, a portable CLI, guarded historical independent
  evaluation, and a separately labeled public benchmark-replay entry.
- Exact four frozen Run102 checkpoints in the model repository, SHA-256 verified.
- Public aggregate benchmark statistics; no raw evaluation trajectory or training sample.

## Checks performed for v0.1.0

- 32 local tests passed, including causal input/label separation, split and
  candidate gates, action fitting, keypoint losses, geometry gradients and
  preservation of the active simulator state.
- Editable installation in a separate temporary environment succeeded using
  already installed dependencies; installed `edgearm doctor` and recipe printing worked.
- All four released checkpoints loaded into their corresponding architectures on CPU.
- Core training/acceptance `--help` commands imported successfully.
- Narrow release hygiene scan found no credential or private-machine-path
  signatures and no included training arrays/videos/checkpoints in the code tree.
  This is not a comprehensive independent security audit.

The local test runtime was Python 3.13.9, PyTorch 2.11.0, MuJoCo 3.10.0, SciPy
1.17.1, and NumPy 2.2.6; the temporary installed check also passed with NumPy 2.4.4.
CUDA was not available on the packaging host. These are release-test versions,
not a reconstructed lockfile for the original GPU experiment.

## Not newly re-run for packaging

No fresh GPU training or complete 72-episode CUDA replay was performed while
packaging. The 51/72 result is the archived frozen Run102 experiment, not a new
score from the public wrapper. CPU tests are not a substitute for task evaluation.

Exact historical retraining requires data and initializer checkpoints not included
in this release. The owner explicitly retained training-data privacy. A user can
run inference with released weights and generate their own data with the source;
do not claim that a code-and-weights release is an open dataset or a turnkey
from-scratch reproduction of private training history.

## Asset and provenance preservation

`provenance/source-files.json` binds extracted research files to their local
source hashes. New wrapper/docs files are tracked by the public Git commit.
`provenance/checkpoints.json` pins four file hashes and the Hub revision.
Record your environment, source revision, models, seeds, options, and all terminal
outcomes when evaluating. Keep public benchmark replay distinct from a newly
pre-registered independent seed set.

