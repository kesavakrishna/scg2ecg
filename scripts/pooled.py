"""B4: does training on more than one sensor help on a sensor never seen? (training part; tested in scripts/rhc.py)

Three U-Nets, same recipe (early stopping on the same mixed tune set, up to 15 epochs, seeds 0-2):
  cebs     14 CEBS development people
  sst      28 SensSmartTech people
  pooled   both
Tune set (all three): CEBS development people 12 and 13 ('late' parts) and the first 4 SensSmartTech people.
Scored on SCG-RHC (heart-failure patients, a third sensor) by scripts/rhc.py, without and with calibration.

Writes results/pooled/{models *.pt (not in git), histories.csv}.
Run from the repo root:  .venv\\Scripts\\python -m scripts.pooled
"""
from __future__ import annotations

import time

import pandas as pd
import torch

from scg2ecg.paths import RESULTS
from scg2ecg.splits import get_splits

from .external import load_rec, recordings
from .simple_models import SEEDS, segs, train_unet

OUT = RESULTS / "pooled"
CEBS_TUNE = [12, 13]
SST_TUNE = 4  # first N SensSmartTech people


def sst_segments(people) -> list[dict]:
    recs = recordings()
    out = []
    for r in recs[recs.subject.isin(people)].itertuples():
        d = load_rec(r._asdict())
        out.append({"x": d["x"], "y": d["y"], "r": d["r"]})
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sp = get_splits()
    sst_people = sorted(recordings().subject.unique())
    cebs_train = [s for s in sp["dev"] if s not in CEBS_TUNE]
    sst_tune, sst_train = sst_people[:SST_TUNE], sst_people[SST_TUNE:]
    tune = segs(CEBS_TUNE, "late") + sst_segments(sst_tune)
    sets = {"cebs": segs(cebs_train, "all"), "sst": sst_segments(sst_train)}
    sets["pooled"] = sets["cebs"] + sets["sst"]
    hpath = OUT / "histories.csv"
    hist_rows = pd.read_csv(hpath).to_dict("records") if hpath.exists() else []
    for name, train in sets.items():
        for seed in SEEDS:
            path = OUT / f"{name}_seed{seed}.pt"
            if path.exists():
                continue
            t0 = time.time()
            model, r_tune, hist = train_unet(train, tune, seed, max_epochs=15, patience=3)
            torch.save(model.state_dict(), path)
            hist_rows += [{"train_set": name, "seed": seed, **h} for h in hist]
            print(f"{name} seed {seed}: tune r {r_tune:.3f}, {len(hist)} epochs, {(time.time() - t0) / 60:.1f} min",
                  flush=True)
            pd.DataFrame(hist_rows).to_csv(hpath, index=False)


if __name__ == "__main__":
    main()
