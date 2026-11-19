# Holdout plan (written and committed before the holdout is touched)

The holdout people are 3, 9, 16 and 17 (`splits.json`, drawn from a fixed seed in Phase 2). No model,
baseline, threshold or setting has been scored on them so far. They are scored once, by
`scripts/holdout.py`, with everything below fixed in advance. Whatever comes out is reported, including
results that contradict the development folds.

## Final models

Every model is trained on all 16 development people. There is no early stopping on holdout data. Each
method trains for a fixed number of epochs: the median, over its 12 development runs (4 folds x 3 seeds),
of the epoch early stopping chose (`results/gans/histories.csv`, `results/simple_models/histories.csv`),
plus one, since epochs count from 0. Seeds 0, 1, 2.

| method | as in |
|---|---|
| unet | Phase 3 |
| pix2pix_adv0.1 | Phase 4 |
| cycle_paired | Phase 4 |
| cycle_unpaired | Phase 4 |
| cgan_2g6d | Phase 4 (the 2-generator, 6-discriminator cGAN) |
| avg_beat | Phase 2 baseline, averages and detector from all 16 development people |

## Settings

1. **No calibration** (the plan's original question): every method above, on each holdout person's
   'late' part.
2. **Calibration** (the practical finding): the U-Net with 10, 30 and 60 s from the start of the
   person's music record:
   - shift only;
   - fine-tuned for 100 steps (fixed in advance: the most common good choice in development, and every
     option from 50 to 400 scored within about 0.01);
   - the person's own average beat from the same chunk.

## Scores

- Pearson r, RMSE, beat sensitivity / PPV, R timing error, average-beat correlation, R amplitude ratio.
- Clinical intervals for avg_beat, unet, unet fine-tuned on 30 s and the person's own average beat
  from 30 s: QRS, PR and QT per-beat error, and P-wave agreement.
- Person-by-person comparisons are reported, but with 4 people they are descriptive only; no p-values.

## Questions, with what development predicts

| question | development answer |
|---|---|
| U-Net vs average-beat baseline, no calibration | U-Net better (0.62 vs 0.37) |
| Any GAN vs U-Net | no better (within seed noise) |
| Unpaired CycleGAN | far below both (0.12) |
| U-Net fine-tuned on 10 s | ~0.96 |
| fine-tuned vs the person's own average beat | fine-tuned better |
