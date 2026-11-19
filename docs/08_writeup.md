# SCG to ECG, rebuilt (draft write-up)

*Draft for part 2 of the "revisiting my papers" series. Every number links to a generated file in this
repo. To edit before posting: the voice, the intro, and anything about the coauthors. Give them a heads-up
before it goes out.*

## The question

Can a model draw a usable ECG from chest vibrations (a seismocardiogram, SCG) for a person it has never
seen? And does a CycleGAN beat a plain supervised model at it?

In 2022 a paper reported SCG-to-ECG with an attentive CycleGAN: Pearson r of 0.89 / 0.91 / 0.88 on the
CEBS database. I coauthored a 2023 follow-up that built on that model. This is a rebuild in PyTorch, on
the same public data, with the evaluation done the way I would do it now.

## What the original number measured

Reading the original code (`docs/01_original.md`):

- **It was a same-person, same-recording score.** One model per person, trained on the first 75% of a
  5-minute recording and tested on the last 25%.
- **The "CycleGAN" trained as a supervised model.**
  - The supervised Huber loss carried all the learning.
  - The "cycle" loss compared the model's output with its own earlier output, so it was about zero.
  - The discriminator ended in an instance normalisation over its single output channel, which makes its
    loss the same whatever the input. It could not tell real from fake.
- **For 19 of 20 people, two of the three result columns came from the same data.** In CEBS 1.0.0 the
  rest recordings of subjects 2-20 are byte-identical copies of the after-music recordings, so
  "basal" and "post-music" were two runs on one dataset. This is still in the database as distributed.
- **A faithful PyTorch port reproduces the saved numbers** (subject 1: 0.88 against 0.877 saved). Deleting
  the GAN parts entirely changes nothing (0.84 against 0.83 over 16 people).
- **Pasting the person's average heartbeat at beats found in the SCG scores higher**, 0.94, beating the
  model for 15 of 16 people. Within one short recording, the ECG is close to one repeated beat.

## Evaluating on people the model has not seen

- 4 people are locked away as a holdout. The other 16 are split into 4 folds.
- In each fold the model trains on 10 people, early-stops on 2 and is tested on 4.
- Every number below is on the last 30% of each person's music recording plus their after-music
  recording, with a 10 s gap.
- Every model is compared, person by person, with the average-heartbeat baseline.

| model (unseen people) | mean r | median r | worst person |
|---|---|---|---|
| average-heartbeat baseline | 0.37 | 0.42 | 0.01 |
| ridge regression (a linear filter) | -0.01 | 0.00 | -0.60 |
| 1D U-Net, supervised | **0.62** | 0.71 | 0.03 |
| same U-Net, tested on the same people it trained on | 0.99 | 0.99 | 0.98 |

The U-Net beats the baseline for 14 of 16 development people, though not on the holdout (below). The
step from "same person" (0.99) to "new person" (0.62) is the real story. The spread is wide: 0.90 for some people, near zero for others.

## Does a GAN help?

Five GAN set-ups, all on the same U-Net generator, data, folds and seeds (`docs/04_gans.md`):

| model | mean r | better than U-Net for |
|---|---|---|
| pix2pix (adversarial weight 0.1) | 0.62 | 7 / 16 |
| pix2pix (adversarial weight 1) | 0.59 | 4 / 16 |
| CycleGAN, paired | 0.61 | 4 / 16 |
| cGAN with 2 generators and 6 discriminators | 0.61 | 6 / 16 |
| CycleGAN, unpaired | 0.12 | 1 / 16 |

**No.** With paired data, the GANs tie the supervised model at best. Without paired data, the CycleGAN
falls below even the average-heartbeat baseline.

The reason shows when the discriminator is trained against the finished U-Net. On the people it trains
on, it can barely tell the U-Net's output from real ECG (50-58% accuracy), because the generator is
already close to perfect there. The errors are on new people, whom the discriminator never sees. An
adversarial loss sharpens what the model does on its training data. It does not fix generalisation.

## Where the error comes from: timing

If each output is moved by the single best time shift, U-Net r goes from 0.62 to 0.75. Most people are
within 10 ms, and the near-zero scores come from a few large shifts:

- subject 6: 38 ms early, r 0.03 → 0.84 after the shift;
- subject 7: 18 ms late, 0.46 → 0.91.

The delay between the heart's electrical R-wave and the main chest vibration is personal (22-122 ms
across people in CEBS). A narrow QRS shifted by its own width correlates at about zero, even if its shape
is perfect.

The unpaired CycleGAN is consistently 47 ms late. Nothing in unpaired training says when the output
beat should happen.

## Ten seconds of calibration

If each new person contributes a short paired recording, the 10-60 s at the start of their session
(`docs/05_calibration.md`):

| method | 10 s | 30 s | 60 s |
|---|---|---|---|
| shift the U-Net output by the delay measured on the clip | 0.73 | 0.72 | 0.72 |
| the person's own average beat from the clip | 0.82 | 0.91 | 0.91 |
| **fine-tune the U-Net on the clip** | **0.96** | **0.97** | **0.98** |

- **Fine-tuning on 10 s lifts every one of the 16 people.** Subject 6 goes from 0.03 to 0.98, and the
  lowest score is now 0.90 (subject 2).
- **It also holds across recordings:** calibrating on the separate after-music recording works as well.
- **The caveat:** CEBS has one session per person, so a calibration surviving a new day or a re-attached
  sensor is untested.
- **What it means in practice:** record an ECG once, then monitor with the SCG sensor alone. It does not
  mean no ECG at all.

## Would a clinician accept it?

From `docs/07_clinical.md`:

- **Intervals.** Without calibration, every model's QRS, PR and QT errors are 34-52 ms per beat; the QRS
  error is half a QRS. After 30 s of calibration they are 8-15 ms.
- **Irregular beats.** The person's own average beat pastes the same shape everywhere, so it fails at the
  beats that differ: per-beat r 0.73. The fine-tuned U-Net follows the actual vibration and keeps their
  shape: 0.97. That is the argument for a learned model over a template, and it only appears on the
  beats that are not average.
- **Noise.** Every model draws a full-size, convincing ECG from white noise or scrambled SCG. A detached
  sensor would produce a believable fake ECG. Any real use needs a check that there is a heartbeat in the
  SCG before showing the output.

## The holdout

The four holdout people were scored once, by a plan committed beforehand (`docs/06_holdout_plan.md`,
results in `docs/09_holdout.md`).

| method | development (16) | holdout (4) |
|---|---|---|
| average-heartbeat baseline | 0.37 | **0.69** |
| U-Net | 0.62 | 0.60 |
| best GAN (paired CycleGAN) | 0.61 | 0.69 |
| U-Net fine-tuned on 10 s | 0.96 | **0.94** |

- **The calibration result held.** Fine-tuning on 10 s was the best method for all four holdout people
  (0.89-0.98).
- **The unseen-person ranking did not.** On these four people the plain average-beat baseline beat the
  U-Net, for all four. Its SCG beat detector happened to work well on them (8 ms timing error, against
  24 ms in development). In development, the same baseline ranged from 0.01 to 0.68 across people.
- **The GANs did not reliably win either.** The paired CycleGAN came out ahead of the U-Net on the holdout
  (4 of 4) after tying it in development (4 of 16). One seed of the six-discriminator cGAN failed to draw
  QRS complexes at all at its pre-set epoch count.

Small test sets swing. The pre-registered plan is what makes this result reportable rather than
something to quietly re-run.

## After the holdout: four follow-ups

These came after the holdout, so they are development or external evidence, not holdout results.

- **A new sensor, a new lab, 32 new people** (`docs/11_external.md`, SensSmartTech on PhysioNet).
  - Without calibration, every CEBS-trained model scores r ≈ 0.1-0.2. A different accelerometer and chest
    position are enough to break them.
  - Fine-tuning on 10 s from a different recording of the same person reaches 0.86, or 0.90 with 26 s,
    better than no calibration for every person.
  - After exercise, at higher heart rates, it falls to about 0.5: a calibration at rest does not cover a
    faster heart.
- **How little calibration?** (`docs/12_cheap_calibration.md`)
  - 3 s of ECG, about three beats, already gives 0.94 on CEBS.
  - Beat times alone, without the ECG's shape, give 0.84, and only if the times are accurate to a few ms.
  - The person-specific part is mostly their beat shape.
- **A heartbeat check** (`docs/10_gate.md`).
  - The periodicity of the SCG's energy envelope passes 92% of real 8 s windows and essentially none of the
    noise or scrambled SCG. That stops the models from drawing an ECG out of nothing.
  - It is weakest on the person with the most irregular rhythm.
- **Was the holdout reversal about training-set size?** No (`docs/13_loso.md`). Trained on 15 people, the
  U-Net still beats the baseline for 14 of 16 development people (0.65 against 0.41). The four holdout
  people just suited the baseline.

## What I would tell my 2022 self

- **Split by person, and say which question the number answers.** Same-person r of 0.9 and new-person r
  of 0.6 are different results.
- **Compare against the dumbest baseline that could work.** Here that is the person's average beat, and
  it beat the published model.
- **Check that every loss term moves.** A discriminator whose loss cannot depend on its input is easy to
  miss if you only look at the final r.
- **Look at the data files.** Two of three result columns were the same recording.
- **Lock a holdout and write the plan down before you look.** Here it overturned a 14-of-16 result.
- **The interesting result is not the architecture.** It is that the person-specific part is small: a few
  seconds of their own ECG. It survives a new sensor and lab (0.86-0.90), but not a big change in heart rate.
- **Test what the model does when there is nothing to see.** Every model here drew a convincing ECG from
  noise. A ten-line periodicity check stops that.

## Reproduce

Everything runs from the repo root (`README.md`). It takes about one GPU day on a laptop RTX 4060, most
of it on the GANs.
