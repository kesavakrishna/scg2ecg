# scg2ecg

Rebuilding a 2021/2023 SCG-to-ECG model in PyTorch and asking one question honestly: can a model draw a
usable ECG from chest vibrations (seismocardiogram, SCG) for a person it has never seen, and does a GAN beat
a plain supervised model at it?

Data: the [CEBS database](https://physionet.org/content/cebsdb/1.0.0/) on PhysioNet (20 people; ECG leads
I and II, respiration and SCG at 5 kHz; ODC-By 1.0).
External test data: [SensSmartTech](https://physionet.org/content/senssmarttech/1.0.0/) on PhysioNet
(Lazović et al.; 32 people; ECG and a chest accelerometer; CC BY 4.0).

## Setup

```
python -m venv --system-site-packages .venv      # reuses a system torch with CUDA, if present
.venv\Scripts\pip install -r requirements.txt
curl -L -o data/cebsdb-1.0.0.zip https://physionet.org/content/cebsdb/get-zip/1.0.0/
# unzip so that the records sit in data/files/ (b001.dat, b001.hea, b001.atr, ...)
# for the external test only (CC BY 4.0, 265 MB):
curl -L -o data/external/senssmarttech/senssmarttech.zip https://physionet.org/content/senssmarttech/get-zip/1.0.0/
# unzip and rename the extracted folder to data/external/senssmarttech/sst
```

## Reproduce

Run from the repo root, in this order (later steps load earlier steps' models). Times are approximate on a
laptop RTX 4060 (8 GB) and 16 CPU cores, about one GPU day in total, mostly the GANs.

| step | command | writes | time |
|---|---|---|---|
| data check | `.venv\Scripts\python -m scripts.data_check` | results/data_check | minutes |
| baselines | `.venv\Scripts\python -m scripts.baselines` | results/baselines | ~15 min |
| original model, ported | `.venv\Scripts\python -m scripts.replicate_original` | results/replication | ~2 h |
| ridge, U-Net | `.venv\Scripts\python -m scripts.simple_models` | results/simple_models | ~1.5 h |
| GANs | `.venv\Scripts\python -m scripts.gans` | results/gans | ~12 h |
| calibration | `.venv\Scripts\python -m scripts.calibration` (then `--part transfer`) | results/calibration | ~1.5 h |
| clinical scoring | `.venv\Scripts\python -m scripts.clinical` | results/clinical | ~1.5 h (CPU) |
| holdout, once | `.venv\Scripts\python -m scripts.holdout` | results/holdout | ~3 h |
| heartbeat check | `.venv\Scripts\python -m scripts.gate` | results/gate | ~20 min |
| external test | `.venv\Scripts\python -m scripts.external` | results/external | ~30 min |
| cheaper calibration | `.venv\Scripts\python -m scripts.cheap_calibration` | results/cheap_calibration | ~1 h |
| leave one out | `.venv\Scripts\python -m scripts.loso` | results/loso | ~1.5 h |

Long steps are resumable: re-running one skips the work already saved.

Set `SCG_DEVICE=cpu` to keep a step off a GPU that other programs are using.

## Results in brief

Mean Pearson r over the 16 development people. The first two rows use the original protocol (same
person, same 5 min recording); the others score each person's 'late' data.

| method | r |
|---|---|
| the original model and protocol (same person, same 5 min recording) | 0.83 |
| average-heartbeat baseline on that protocol | 0.94 |
| average-heartbeat baseline, unseen people | 0.37 |
| 1D U-Net, unseen people | 0.62 |
| paired GANs (pix2pix, CycleGAN, 2-generator / 6-discriminator cGAN) | 0.59-0.62 |
| CycleGAN, unpaired | 0.12 |
| U-Net fine-tuned on 10 s of the new person's data | 0.96 |
| U-Net, same people it was trained on | 0.99 |

Holdout (4 people, scored once): the U-Net fine-tuned on 10 s reaches 0.94, the best method for every
holdout person. Without calibration, the average-heartbeat baseline (0.69) beat the U-Net (0.60) on these
four people, reversing the development result. See `docs/09_holdout.md`.

Follow-ups, after the holdout:
- **External test** on SensSmartTech (32 new people, a different sensor and lab): CEBS models score r ≈ 0.2
  without calibration and 0.86 after fine-tuning on 10 s from another recording; about 0.5 after exercise
  (`docs/11_external.md`).
- **Cheaper calibration:** 3 s of ECG gives 0.94; beat times alone give 0.84 (`docs/12_cheap_calibration.md`).
- **Heartbeat check:** SCG periodicity passes 92% of real windows and about 0% of noise (`docs/10_gate.md`).
- **Leave one out:** trained on 15 people, the U-Net beats the baseline for 14 of 16 (`docs/13_loso.md`).

Notes per phase in `docs/`: what the original did (01), GANs (04), calibration (05), holdout plan (06),
clinical scoring (07), a write-up draft (08), holdout results (09), and follow-ups (10-13).

## Evaluation rules

- **Split by person.** 4 people (3, 9, 16, 17) are a locked holdout, drawn from a fixed seed
  (`splits.json`), scored once at the end. The other 16 form 4 folds: test on 4, tune / early-stop on 2,
  train on 10.
- **Same-person setting** for comparison: train on the first 70% of each person's music record, test on
  the rest plus the after-music record (10 s gap). Both settings score the same test data.
- **Every model is compared with an average-heartbeat baseline:** the training people's average ECG beat
  pasted at beats detected in the SCG. It needs no learning beyond two averages.
- Scores: Pearson r, RMSE, PRD, plus R-peaks found by neurokit2 in the output, compared with the
  database's lead I beat annotations (sensitivity, PPV, timing error, R-R error).

## A data note

In CEBS 1.0.0 the rest record of subjects 2-20 is a byte-identical copy of the after-music record
(`b002.dat == p002.dat`, ...). Only subject 1 has a distinct rest record. The annotation files were not
copied: `b002.atr`-`b020.atr` still mark the beats of the genuine (now missing) rest recordings, so they
do not line up with the `bNNN.dat` signals. This code uses each signal once, with its matching
annotations (`scg2ecg/data.py: usable_records`).

## Layout

```
scg2ecg/   data loading, splits, metrics, baselines, models, windowing
docs/      notes on each phase
scripts/   one script per step; each writes results/<step>/summary.md
results/   generated tables and figures
```
