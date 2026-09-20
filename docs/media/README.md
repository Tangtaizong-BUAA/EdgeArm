# Report figures and evaluation media

These assets illustrate the public technical report. **They are not an open
training dataset.** No training RGB, raw state/action packages, credentials or
private logs are distributed here.

## Videos

| Case | Original frozen result | Frames | Playback | Download |
|:--|:--|--:|:--|:--|
| Run102 / 882000924 / route 6 | Success, 670 steps; 3 s hold | 84 | 3.75 fps; 22.40 s | [MP4](https://github.com/Tangtaizong-BUAA/EdgeArm/releases/download/v0.1.0/run102-success-882000924.mp4) |
| Run102 / 882000938 / route 2 | Success, 671 steps; 3 s hold | 84 | 3.75 fps; 22.40 s | [MP4](https://github.com/Tangtaizong-BUAA/EdgeArm/releases/download/v0.1.0/run102-success-882000938.mp4) |

Selection is a post-hoc difficult-route success showcase. Route 6 (1/8 successes)
and route 2 (2/8) were the two lowest-success routes in the frozen Run102 test.
We use the only route-6 success and the lowest-seed route-2 success. Both reached
maximum coverage 1.0 and a 3-second hold without teacher execution. Difficulty
here is empirical route performance, not a claim of obstacles or a separately
defined hard benchmark. These selected successes do not replace the complete
51/72 aggregate. Native wrist frames are 160×120 RGB, stored
every eighth control step at 30 Hz. We encode every stored frame in order,
without interpolation, retiming, image enhancement or scene generation. The
last displayed frame need not be the terminal success-transition frame; encoding
a whole final sample interval can extend playback slightly beyond the last
recorded control instant. MP4 uses nearest-neighbor enlargement to 640×480 and H.264 compression;
GIF previews use 320×240 and palette quantization. No policy was rerun.

Each accompanying `run102-*.json` records selection, trace checksum, frame count,
transforms and output checksums. Raw traces remain private. The public
[`render_evaluation_media.py`](../../tools/render_evaluation_media.py) documents
the export operation for an owner-supplied archived trace.

## Figures

| Stem | Claim and source | Interpretation |
|:--|:--|:--|
| `scene_overview` | Frozen scene constructor and selected episode's original RGB | Left: local initial-scene reconstruction; right: archived observation. Overview is not policy input. |
| `wrist_sequence` | Four original RGB frames from successful episode 882000900 | One illustrative time sequence, not four independent observations. |
| `method` | Final inference implementation and dependencies | Learned memory/control plus explicit geometric and program components. |
| `training_data` | Run100 route label counts and Run101 source frame counts | States and frames, not independent physical episodes. |
| `training_updates` | Actual Run94/100/101 update counts and selected checkpoints | Different modules' updates are not equal compute units. |
| `route_results` | All 72 frozen Run102 outcomes, grouped by route and scene | Per-route Wilson intervals and full success/timeout/out-of-bounds matrix. |

PNG files are the reviewed raster presentation; SVG files retain editable
scientific marks and text. Scene figures contain raster layers, not vector
photographs. FigureRequest, FigureIR and DataProfile accompany each figure.
Their data-source paths identify historical evidence (not all source archives
are public); hashes preserve the evidence binding. Local personal font paths
were replaced by font names for publication, and no font binaries are bundled.
The original deterministic figure sources are retained in the project research
archive; the published SVG and numerical values support inspection and editing.

Fonts: Arial and Microsoft YaHei; restrained blue/orange/neutral palette. On
systems lacking those fonts, view the PNG for the reviewed typography. Aggregate
data are in `training_data.values.json`, `training_updates.values.json` and
[`benchmark.json`](../../provenance/benchmark.json). Statistical and scientific
interpretation is documented next to every figure in the main report.

The media are project-authored simulation figures and evaluation illustrations
under the repository's Apache-2.0 terms, with upstream robot-asset attribution
retained in [NOTICE](../../NOTICE). This does not grant access to the private
training corpus.
