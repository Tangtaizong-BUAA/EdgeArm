# Report media review

- Six existing evidence figures were reused without changing quantitative marks.
  All six FigureRequest/FigureIR contracts passed again; all six SVG structural
  checks passed. Mixed raster layers in scene, timeline and outcome matrix are
  retained and disclosed, not described as all-vector figures.
- The actual PNGs were visually inspected for labels, units, clipping and panel
  order. Quantitative figures retain their original scales, denominators and
  legend distinctions. SVG retains text; PNG is the font-stable GitHub display.
- Two evaluation-only videos were encoded from original ordered wrist frames.
  FFprobe confirmed 65 / 113 frames, H.264, 640×480 and 15/4 fps. Full decoding
  of both files completed without errors. This is media verification, not a new
  policy or physics evaluation. Representative frames were also viewed.
- GitHub Release SHA-256 digests matched the local video metadata. Training
  trajectories and raw trace packages were not uploaded.
- `tools/check_report.py` checks local Markdown links, aggregate count sums,
  figure XML structure, and preview checksums. `tools/audit_release.py` includes
  SVG and metadata scanning. These automated checks do not replace semantic
  review; no independent scientific peer review is claimed.
- Original figure renderers and full experimental source receipts remain in the
  private research archive. Public figures include editable SVG, contracts,
  aggregate values and source hashes. Only local personal font paths were
  removed from exported metadata; fonts themselves are not distributed.
