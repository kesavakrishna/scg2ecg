# What the original work did (Phase 1 note)

Sources: the 2023 ICPEEV paper (`DL_for_Signals_Research_Paper_Published.pdf`), the eight notebooks in
`3-2Project-main/` (near-copies of one file), and the CEBS files on PhysioNet. The notebook is the code
Neeraj used for the IEEE Signal Processing Letters 2022 paper ([1] in the 2023 paper); the 2023 paper
adds an identification module on top.

## Two papers, one model

- **SPL 2022 (Neeraj, Satija, Mathew, Behera):** the attentive CycleGAN, SCG to ECG on CEBS. Reports
  average Pearson r of 0.890 / 0.913 / 0.876 for the rest / music / after-music records of all 20 people.
  This is the "prior work" the plan asked about: it is the model we are rebuilding.
- **ICPEEV 2023 (this paper):** an "identification module" (Pan-Tompkins) to tell ECG from SCG in a mixed
  signal before handing it to the SPL model. Its own results are QRS-detection accuracies of 93-97%. The
  abstract names the MIT-BIH arrhythmia database, but the CSVs in the repo (`DBS_1.csv`, `DBS_2.csv`) are
  the Kaggle PTB heartbeat files (125 Hz single beats); the saved notebook cells that call the detector
  fail with a `NameError`, and `Code.py` assumes 250 Hz on 1 kHz data. The 93-97% cannot be traced to code.

## The checklist, answered from the code

| Question | Answer |
|---|---|
| Data | CEBS, one rest record (`bNNN.edf`) per run; ECG lead I (channel 1); the SCG channel |
| Split | Same person, same 5 min record: first 75% of non-overlapping 1 s windows train, last 25% test, no gap |
| Paired? | Yes, and directly supervised (below) |
| Preprocessing | Each signal z-scored over the whole record (test part included), FFT-resampled 5 kHz to 1 kHz, no filter |
| Model | Generator: five same-length Conv1D layers (k = 30) and an attention gate clipped to [0.8, 1]; no down- or upsampling. Discriminator: four Conv1D layers |
| Training | 1 epoch, batch size 1 (225 steps), Adam 1e-3, one seed, windows in time order |
| Checkpoint | The weights after the single epoch; no selection step in the code |
| Metrics | Pooled over test windows: r, RMSE, MAE, PRD, plus a thresholded "accuracy". PRD has the x100 inside the square root, so it reads 10x too low (4.8 printed = 48%) |
| Per-person spread | Not in the code; one record per run, results appended to a CSV by hand |
| Leftovers | Code and the 20 rest EDFs only. No weights, no processed data, no music / after-music data |

## What the training loop really optimises

The generator's loss has three parts; only one of them does anything:

1. **Supervised (weight 4):** Huber loss between G(SCG) and the ECG recorded at the same time. This is
   the whole learning signal. So the model is a paired, supervised CNN, not an unpaired CycleGAN.
2. **"Cycle" (weight 0.04):** the target is the model's own reconstruction computed just before the step,
   so the loss is ~1e-5 (dropout noise only).
3. **Adversarial (weight 1):** the discriminator's last layer is an instance normalisation over its
   1-channel output, so its output is always `gain x (zero-mean, unit-SD pattern) + bias`. Its MSE loss
   against a constant target is `gain^2 + (bias - target)^2` whatever the input, so it cannot tell real
   from fake (Keras accuracy stuck at 50%) and passes ~zero gradient back to the generator.

The generator's last layer is the same instance normalisation, so every 1 s output window has the same
mean and SD: the model cannot draw a flat line, whatever the input.

## The data problem nobody noticed

For subjects 2-20 the rest record is a byte-identical copy of the after-music record (`b002.dat ==
p002.dat`, ...; 19 of 20). Only subject 1 has a distinct rest record. So in SPL 2022 the "basal" and
"post-music" columns are, for 19 of 20 people, two runs on the same data (0.890 vs 0.876: the difference
is training noise). The bug is still in CEBS 1.0.0 as distributed today; the rebuild uses each signal once.

## What this means for the rebuild

- The original number is a same-person, same-record score; it says nothing about new people.
- "Does the CycleGAN beat a supervised model?" has a twist: the original already was a supervised model.
  The rebuild therefore has three questions: reproduce the number (faithful port, `scg2ecg/legacy.py`);
  test it against the average-heartbeat baseline and on unseen people; and only then ask whether a
  correctly wired GAN or CycleGAN adds anything.

## Replication result (results/replication/summary.md)

PyTorch port, original protocol, 16 development people, 3 seeds each:

- The port reproduces the saved numbers: subject 1 r = 0.882 (notebook 0.877); subject 10, 2023 variant,
  r = 0.781 (notebook 0.790 / 0.771).
- Mean over people: 2021 model r = 0.832; the same model with the adversarial and cycle terms deleted
  r = 0.840 (GAN better for 3 of 16 people, p = 0.06): the GAN parts add nothing.
- The average-heartbeat baseline, fitted on the same first 75% of the record, gets r = 0.943 and beats the
  model for 15 of 16 people (p = 0.0006). The exception is subject 18, whose irregular rhythm trips the
  baseline's SCG beat detector.
- Fed near-zero noise, the model still outputs a full-amplitude waveform (window SD 0.84 vs 0.85 on real
  SCG): it cannot draw a flat line.
