# scg2ecg

Rebuilding a 2021/2023 SCG-to-ECG model in PyTorch and asking one question honestly: can a model draw a
usable ECG from chest vibrations (seismocardiogram, SCG) for a person it has never seen, and does a GAN beat
a plain supervised model at it?

Data: the [CEBS database](https://physionet.org/content/cebsdb/1.0.0/) on PhysioNet (20 people; ECG leads
I and II, respiration and SCG at 5 kHz; ODC-By 1.0).

## Setup

```
python -m venv --system-site-packages .venv      # reuses a system torch with CUDA, if present
.venv\Scripts\pip install -r requirements.txt
curl -L -o data/cebsdb-1.0.0.zip https://physionet.org/content/cebsdb/get-zip/1.0.0/
# unzip so that the records sit in data/files/ (b001.dat, b001.hea, b001.atr, ...)
```

Run the steps from the repo root, in order:

```
.venv\Scripts\python -m scripts.data_check        # results/data_check
.venv\Scripts\python -m scripts.baselines         # results/baselines
.venv\Scripts\python -m scripts.replicate_original  # results/replication (the original model, ported)
.venv\Scripts\python -m scripts.simple_models     # results/simple_models (ridge, U-Net)
.venv\Scripts\python -m scripts.gans              # results/gans (pix2pix, CycleGAN paired / unpaired)
.venv\Scripts\python -m scripts.calibration       # results/calibration (10-60 s of a new person's data)
.venv\Scripts\python -m scripts.calibration --part transfer   # calibration from a separate recording
.venv\Scripts\python -m scripts.clinical        # results/clinical (intervals, irregular beats, noise input)
.venv\Scripts\python -m scripts.holdout         # results/holdout (scored once; plan in docs/06)
```

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

Notes per phase in `docs/`: what the original did (01), GANs (04), calibration (05), holdout plan (06),
clinical scoring (07), a write-up draft (08), and the holdout results (09).

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
