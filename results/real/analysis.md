# Analysis: real-photo experiment

Companion to [report.md](report.md) (auto-generated numbers for the final version).
The first version's report is in [v1/](v1/).

## Setup

- 40 photos of a cardboard board with 4 screws, fixed 1920x1080 webcam looking straight down,
  taped outline, room light. Small placement differences between photos.
- Split by capture order, every third photo to training: **14 training, 26 test**
  (good 4/8, missing screw 4/8, crack 2/4, scratch 2/4, dark 1/1, blurry 1/1).
- Defects are drawn: cracks = dark pen zig-zag, scratches = thin pencil/pen line.
- Everything was calibrated or chosen on the 14 training photos only: screw presence check,
  image-quality limits, trained baseline, and the cardboard surface prompt.
  The test set was run once for v1 and once for v2.

## Final results (26 unseen test photos)

| Method | Accuracy | Escaped defects ↓ | Good boards falsely rejected ↓ | Sent to human | Time / board (CPU) |
|---|---|---|---|---|---|
| Baseline 1: trained classifier | 77% | 6% (1/16) | 50% (4/8) | 4% | <0.1 s |
| Baseline 2: naive Gemma | 46% | **75% (12/16)**: passed 8/8 missing screws and 4/4 scratches at "0.95 confidence" | 0% | 0% | 18 s |
| Ours v1 (metal-plate surface prompt) | 65% | 0% (0/16) | 100% (8/8) | 12% | 49 s |
| **Ours v2 (cardboard surface prompt)** | **85% (22/26)** | **0% (0/16)** | 38% (3/8) | 12% | 73 s |

## v1 → v2: what changed and why

v1 reused the surface prompt written for the synthetic metal plates. On a good board Gemma said:
*"The metal plate surface exhibits a textured appearance with numerous small scratches, marks, and
variations in tone, indicating wear and use"*, and it rejected all 8 good boards.

v2 gives the cardboard its own prompt (part of the part spec, like the screw positions): normal
cardboard texture, folds, ridges, shading and specks are not damage; damage is a distinct, sharp,
dark line. Chosen on the training boards (7/8 correct, each training defect flagged in the tile
where it is), then the test set was run once.

## Component by component (v2)

| Component | Type | Result |
|---|---|---|
| Alignment | Deterministic | All 160 screw crops centred on the screw or hole ([screw_crops_aligned.jpg](screw_crops_aligned.jpg)) |
| **Screw presence check** | Deterministic, calibrated on 14 photos | **8/8 empty holes found at the correct corner, in 0.1 s, no AI needed.** 0/96 real screws called empty; 4 UNSURE (go to human) |
| Image-quality gate | Deterministic, calibrated on 1 dark + 1 blurry photo | Caught the dark and the blurry test photo; wrongly flagged 1 sharp scratch photo as blurry |
| **Gemma screw second opinion** | AI | Agreed with the presence check on every screw it was asked about |
| **Gemma surface check** | AI | Rejected 4/4 cracks and 3/3 remaining scratches; passed 5/8 good boards |

## Is it right for the right reason?

We drew the tile Gemma flagged on every surface rejection ([flagged_regions.jpg](flagged_regions.jpg)):

| Flagged tile | Count |
|---|---|
| Contains the drawn crack/scratch | **6 / 7** (4/4 cracks, 2/3 scratches) |
| Wrong place: scratch_05 flagged faint printed marks at the top, not the pencil lines | 1 / 7 |
| Good boards: all 3 flagged "surface left", which contains a real fold crease in the cardboard | 3 false rejects |

The fold crease is a genuinely sharp dark line, so this is a reasonable mistake; a real part spec
would mark that crease as a known feature or move the tile.

## What this shows

1. **The naive use of a small VLM is dangerous**: confident and wrong on 12/16 defects.
2. **Deterministic checks transfer to real photos with a tiny calibration set**: alignment, presence
   check and image gate caught every missing screw instantly from 14 calibration photos.
3. **The AI is useful for the open-ended part** (is this a crack?) once its question is written
   for the actual material, and it explains its decision in words the operator reads.
4. **The safety design held in both versions**: 0/16 defects escaped even when the AI component
   was failing; errors were over-rejections (cost: a re-check), never shipped defects.
5. **Prompts are part of the specification** and must be validated per material, like any other
   inspection parameter.

## Limitations

- Very small dataset (26 test photos; 1 dark, 1 blurry); one error moves a rate by 4-12 points.
- Defects are drawn marks, not real cracks or scratches.
- Quality-gate limits come from one example each; one false "blurry".
- The cardboard prompt was selected on 8 training boards; not validated on other materials.
- ~2.5 min per good board on a laptop CPU (9 Gemma calls); missing screws take 0.1 s.
- Missing-screw training labels assume the empty position is the one least like a screw.
