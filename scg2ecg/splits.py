"""Person-level splits.

* A 4-person holdout, drawn once from a fixed seed and not touched until the end. Subjects 1 and 10 are
  kept out of the draw because the replication of the 2021/2023 notebooks needs them (the notebooks'
  saved results are for those two people).
* The other 16 people form 4 folds of 4. In fold k: test on group k, tune / early-stop on 2 people from
  group k+1, train on the remaining 10. Everyone outside the holdout is tested exactly once.
* A same-person time split: 'early' (rest record if distinct + first 70% of the music record) versus
  'late' (last 30% of the music record after a 10 s gap + the after-music record). Both settings are
  scored on 'late', so the same-person versus unseen-person gap compares identical test data.
"""
from __future__ import annotations

import json

import numpy as np

from .data import SUBJECTS, load, record_name, usable_records
from .paths import ROOT

SEED = 20261112
SPLITS_FILE = ROOT / "splits.json"
NOT_IN_HOLDOUT = (1, 10)
TRAIN_FRAC = 0.70
GAP_S = 10.0

Segment = tuple[str, int, int]  # (record name, start, stop) in samples at the working rate


def make_splits() -> dict:
    rng = np.random.default_rng(SEED)
    pool = [s for s in SUBJECTS if s not in NOT_IN_HOLDOUT]
    holdout = sorted(int(s) for s in rng.choice(pool, size=4, replace=False))
    dev = [s for s in SUBJECTS if s not in holdout]
    order = [int(s) for s in rng.permutation(dev)]
    groups = [sorted(order[4 * k:4 * k + 4]) for k in range(4)]
    folds = []
    for k in range(4):
        test = groups[k]
        tune = groups[(k + 1) % 4][:2]
        train = sorted(s for s in dev if s not in test and s not in tune)
        folds.append({"test": test, "tune": tune, "train": train})
    return {"seed": SEED, "holdout": holdout, "dev": dev, "folds": folds}


def get_splits() -> dict:
    """The splits, written to splits.json on first use and checked against it afterwards."""
    s = make_splits()
    if SPLITS_FILE.exists():
        if json.loads(SPLITS_FILE.read_text()) != s:
            raise RuntimeError("splits.json differs from the seeded draw; refusing to continue")
    else:
        SPLITS_FILE.write_text(json.dumps(s, indent=2) + "\n")
    return s


def segments(subject: int, fs: int = 500) -> dict[str, list[Segment]]:
    recs = usable_records(subject)
    n = {r: load(r, fs).n for r in recs}
    m, p = record_name("m", subject), record_name("p", subject)
    cut = int(TRAIN_FRAC * n[m])
    early = [(r, 0, n[r]) for r in recs if r.startswith("b")] + [(m, 0, cut)]
    late = [(m, cut + int(GAP_S * fs), n[m]), (p, 0, n[p])]
    return {"all": [(r, 0, n[r]) for r in recs], "early": early, "late": late}
