# Analysis: synthetic experiment

Companion to [report.md](report.md) (auto-generated numbers). This file explains them.

## Final results (40 unseen test images, 10 per label)

| Method | Accuracy | Escaped defects | Good parts falsely rejected | Time / part (CPU) |
|---|---|---|---|---|
| Baseline 1: trained classifier (logistic regression, 64x64 pixels) | 65% | 33% (10/30) | 40% (4/10) | <0.1 s |
| Baseline 2: naive Gemma 3 4B (whole image, trusts own confidence) | 25% | **100% (30/30)** | 0% | 18 s |
| **Ours: inspection cell** | **95%** | **0% (0/30)** | 20% (2/10) | 87 s (0 s for missing screws, ~145 s for good parts) |

The trained classifier on its full 100-image test set: 65% accuracy, 27% escaped, 60% false rejects.

## How we got here (development history)

| Version | What happened | Lesson |
|---|---|---|
| Naive Gemma | Answered "none, confidence 0.95" for every single test image, including missing screws and cracks | A small VLM's self-reported confidence is meaningless; it cannot be the decision-maker |
| Cell v1 (Gemma checks each screw close-up) | Screws in the test set are rotated, so the slot looks like "X" instead of "+". Gemma called them empty and rejected **4/4 good parts**; 1 missing screw still escaped (first 16 images before a crash) | Prompt was tuned on unrotated samples; distribution shift breaks it |
| Prompt v2 (tuned on training crops only) | Read 6/6 screws correctly but caught only 2/6 empty holes, and invented "a dark round-headed screw with a cross-shaped slot" for an empty hole | Rewording moves the bias, it does not fix perception |
| **Cell v2 (final)** | Screw presence moved to a deterministic check calibrated on the training set; Gemma kept as second opinion on screws and as the only judge of surface damage | Measurable questions go to deterministic code; open-ended ones to the model |

Two runs were also lost to Ollama being OOM-killed (model memory grew from 3.7 to 8.4 GB). During
both crashes the cell sent every part to human review and the circuit breaker stopped calling the
dead model: no part was passed without a working check. Fixed with `num_ctx=2048` and unloading
the model every 20 calls.

## Is ours right for the right reason?

**Missing screws (10/10 rejected):** every one was caught by the deterministic presence check, which
found the correct empty position each time. Gemma was not needed for these parts.

**Cracks and scratches (20/20 rejected):** these depend on Gemma's surface check. We drew the region
Gemma flagged on every image ([flagged_regions.png](flagged_regions.png)):

| Flagged region vs actual defect | Count |
|---|---|
| Defect clearly inside the flagged region | 13 / 20 |
| Defect only touches the edge of the flagged region | 6 / 20 |
| Defect is elsewhere (crack_004: flagged "surface top", crack is in the centre) | 1 / 20 |

Combined with the 2/10 good parts where Gemma flagged clean surface, the surface check leans towards
calling surfaces damaged. Our rules make that the safe direction (a false reject costs a re-check;
a false pass ships a defect), but some of the 20/20 is luck, not perception. The plain-English reason
shown to the operator names the region, so an operator can see when it points at the wrong place.

## Limitations

- **Synthetic data is easy for the presence check.** Clean drawn screws on a flat plate separate
  perfectly (training accuracy 100%). Real photos with shadows, reflections and dark screws will be
  harder; the real-photo experiment is the true test.
- **Gemma subset:** Gemma methods ran on 40 of the 100 test images (CPU time); the trained baseline ran on all 100.
- **Speed:** ~2.5 min per good part on a laptop CPU (9 Gemma calls). Fine for a sampled audit station,
  far too slow for an inline 1-part-per-second line. A GPU / edge accelerator, fewer surface tiles, or
  batching the crops would be needed.
- **Misaligned parts** are not checked.
- **Fixed jig assumption:** regions are fixed positions; a part placed elsewhere in the frame is not handled.
- **Small test set:** 10 images per label; one error moves a rate by 10 points.
