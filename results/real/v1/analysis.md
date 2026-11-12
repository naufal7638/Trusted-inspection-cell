# Analysis: real-photo experiment

Companion to [report.md](report.md) (auto-generated numbers).

## Setup

- 40 photos of a cardboard board with 4 screws, fixed 1920x1080 webcam looking straight down,
  taped outline, room light. Taken by the team; small placement differences between photos.
- Split by capture order, every third photo to training: **14 training, 26 test**
  (good 4/8, missing screw 4/8, crack 2/4, scratch 2/4, dark 1/1, blurry 1/1).
- Calibrated on the training photos only: screw presence check, image-quality limits,
  trained baseline. Screw positions are found per photo by a deterministic alignment step.
- Prompts were NOT re-tuned for cardboard (see below).

## Results (26 unseen test photos)

| Method | Accuracy | Escaped defects | Good boards falsely rejected | Time / board |
|---|---|---|---|---|
| Baseline 1: trained classifier | 77% | 6% (1/16) | 50% (4/8) | <0.1 s |
| Baseline 2: naive Gemma | 46% | **75% (12/16)**: passed 8/8 missing screws and 4/4 scratches at "0.95 confidence" | 0% | 18 s |
| Ours: inspection cell | 65% | **0% (0/16)** | **100% (8/8)** | 49 s |

## Component by component (ours)

| Component | Result | Verdict |
|---|---|---|
| Alignment (deterministic) | All 160 screw crops centred on the screw or hole | Works |
| **Screw presence check** (deterministic, calibrated on 14 photos) | **8/8 empty holes found, each at the correct corner, in 0.1 s without AI.** 0 of 96 real screws called empty. 4 screws on scratch boards UNSURE (go to human) | **Works** |
| Image-quality gate (calibrated on 1 dark + 1 blurry training photo) | Caught the blurry and the dark test photo; also flagged 1 sharp scratch photo as blurry (sharpness 177 < 230) | Works, but the limit is a guess from one example |
| **Gemma surface check** | Answered DAMAGED for "surface top" on **every** board, including all 8 good ones | **Fails** |

## Why the surface check failed

The surface prompt was written for the synthetic metal plates and still says "metal plate surface".
On a good cardboard board Gemma answered: *"The metal plate surface exhibits a textured appearance
with numerous small scratches, marks, and variations in tone, indicating wear and use."*
Normal cardboard (corrugation, fibres, printed marks) is read as damage. This is a deployment
lesson: an inspection prompt is part of the part spec and must be written and validated for each
material, the same way the screw positions and quality limits were.

Because every board was rejected at the first surface tile, the cell's 4/4 crack and 3/4 scratch
rejections are **not evidence** that the surface check detects drawn defects on cardboard.

## What this shows

- The **deterministic parts transferred to real photos**: alignment, presence check and image gate
  needed only a 14-photo calibration and caught every missing screw instantly.
- The **safety design held**: even with the AI component failing, no defect was passed (0/16 escaped).
  The failure mode is over-rejection, which costs re-checks, not shipped defects.
- The **naive AI baseline is dangerous**: confident and wrong on 12 of 16 defects.
- The **AI judgement component needs per-material prompt validation** before deployment.

## Limitations

- Very small dataset (26 test photos; 1 dark, 1 blurry).
- Defects are pen/pencil marks, not real cracks or scratches.
- The quality-gate limits come from one example each.
- Missing-screw training labels assume the empty position is the one least like a screw.
