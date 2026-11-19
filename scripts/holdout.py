"""Phase 6: the one-time holdout scoring, exactly as fixed in docs/06_holdout_plan.md.

Holdout people 3, 9, 16, 17. Final models are trained on all 16 development people for a fixed number of
epochs (median early-stopping epoch over the 12 development runs of each method, + 1), seeds 0-2.

Writes results/holdout/{epochs.csv, per_person.csv, clinical.csv, summary.md} and the model weights (not in git).
Run from the repo root:  .venv\\Scripts\\python -m scripts.holdout [--part train score summary]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch

from scg2ecg.baselines import AverageBeat
from scg2ecg.gans import train_gan
from scg2ecg.metrics import morphology, score
from scg2ecg.models import UNet1d
from scg2ecg.paths import RESULTS
from scg2ecg.splits import get_splits, segments
from scg2ecg.windows import load_segments, stitch_predict

from .calibration import apply_shift, avg_beat_pieces, best_shift, calib_segment, finetune, raw_piece
from .clinical import delineate
from .gans import CONFIGS
from .simple_models import DEV, FS, LEN, SEEDS, segs, train_unet

OUT = RESULTS / "holdout"
METHODS = ["unet", "pix2pix_adv0.1", "cycle_paired", "cycle_unpaired", "cgan_2g6d"]
CALIB_S = (10, 30, 60)
FT_STEPS = 100
CLIN_CALIB = 30
METRICS = ["r", "rmse", "beat_se", "beat_ppv", "r_timing_abs_ms", "rr_mae_ms", "template_r", "beat_r", "r_amp_ratio"]


def fixed_epochs() -> dict[str, int]:
    """Median over the 12 development runs of the epoch early stopping chose, + 1 (epochs count from 0)."""
    out = {}
    h = pd.read_csv(RESULTS / "simple_models" / "histories.csv").query("setting == 'unseen'")
    best = h.loc[h.groupby(["fold", "seed"])["val_r"].idxmax(), "epoch"]
    out["unet"] = int(np.median(best)) + 1
    g = pd.read_csv(RESULTS / "gans" / "histories.csv")
    for m in METHODS[1:]:
        gm = g[g.method == m]
        best = gm.loc[gm.groupby(["fold", "seed"])["val_r"].idxmax(), "epoch"]
        assert len(best) == 12, f"{m}: {len(best)} development runs, expected 12"
        out[m] = int(np.median(best)) + 1
    return out


def wpath(m, seed):
    return OUT / f"{m}_seed{seed}.pt"


def train_all(sp):
    ep = fixed_epochs()
    pd.Series(ep, name="epochs").to_csv(OUT / "epochs.csv")
    print("fixed epochs:", ep, flush=True)
    train = segs(sp["dev"], "all")
    for m in METHODS:
        for seed in SEEDS:
            if wpath(m, seed).exists():
                continue
            t0 = time.time()
            if m == "unet":
                model, _, _ = train_unet(train, None, seed, max_epochs=ep[m], patience=10 ** 6)
            else:
                cfg = CONFIGS[m]
                cfg.max_epochs, cfg.patience = ep[m], 10 ** 6
                model, _, _ = train_gan(cfg, train, None, seed, LEN, DEV)
            torch.save(model.state_dict(), wpath(m, seed))
            print(f"{m} seed {seed}: {ep[m]} epochs, {(time.time() - t0) / 60:.1f} min", flush=True)


def load(m, seed):
    g = UNet1d().to(DEV)
    g.load_state_dict(torch.load(wpath(m, seed), map_location=DEV))
    return g.eval()


def scored(base, test, preds):
    args = ([t["y"] for t in test], preds, [t["r"] for t in test], FS)
    return {**base, **score(*args), **morphology(*args)}


def clinical_rows(s, name, seed, preds, test):
    """Interval errors against the real ECG, beat by beat (same delineation as scripts/clinical.py)."""
    per = []
    for i, t in enumerate(test):
        _, b = delineate((None, preds[i], t["r"]))
        _, rb = delineate((None, t["y"], t["r"]))
        per.append(b.merge(rb, on="r", suffixes=("", "_real")))
    d = pd.concat(per, ignore_index=True)
    row = {"subject": s, "method": name, "seed": seed}
    for c in ("qrs_ms", "pr_ms", "qt_ms"):
        row[f"{c}_beat_mae"] = float(np.nanmedian(np.abs(d[c] - d[f"{c}_real"])))
        row[f"{c}_median_err"] = float(np.nanmedian(d[c]) - np.nanmedian(d[f"{c}_real"]))
    row["p_agree"] = float((d.p_found == d.p_found_real).mean())
    return row


def score_all(sp):
    rows, clin = [], []
    ab = AverageBeat(FS).fit(avg_beat_pieces(sp["dev"], "all"))
    ab.tune(avg_beat_pieces(sp["dev"], "all"))
    for s in sp["holdout"]:
        t0 = time.time()
        test = load_segments(segments(s, FS)["late"], FS)
        raw = [raw_piece(sg) for sg in segments(s, FS)["late"]]
        base = {"subject": s}
        p = [ab.predict(x)[0] for x, _, _ in raw]
        rows.append(scored({**base, "method": "avg_beat", "calib_s": 0, "seed": None}, test, p))
        clin.append(clinical_rows(s, "avg_beat", None, p, test))
        for c in CALIB_S:
            cx, cy, cr = raw_piece(calib_segment(s, c))
            own = AverageBeat(FS).fit([(cx, cy, cr)])
            own.tune([(cx, cy, cr)])
            p = [own.predict(x)[0] for x, _, _ in raw]
            rows.append(scored({**base, "method": "avg_beat_own", "calib_s": c, "seed": None}, test, p))
            if c == CLIN_CALIB:
                clin.append(clinical_rows(s, f"avg_beat_own_{c}s", None, p, test))
        for m in METHODS:
            for seed in SEEDS:
                g = load(m, seed)
                p = [stitch_predict(g, t["x"], LEN, DEV) for t in test]
                rows.append(scored({**base, "method": m, "calib_s": 0, "seed": seed}, test, p))
                if m != "unet":
                    continue
                clin.append(clinical_rows(s, "unet", seed, p, test))
                for c in CALIB_S:
                    chunk = load_segments([calib_segment(s, c)], FS)[0]
                    d = best_shift(chunk["y"], stitch_predict(g, chunk["x"], LEN, DEV))
                    rows.append(scored({**base, "method": "unet_shift", "calib_s": c, "seed": seed,
                                        "shift_ms": d * 1000 / FS}, test, [apply_shift(q, d) for q in p]))
                    ft = finetune(g, chunk, [FT_STEPS], seed)[FT_STEPS]
                    pf = [stitch_predict(ft, t["x"], LEN, DEV) for t in test]
                    rows.append(scored({**base, "method": "unet_finetune", "calib_s": c, "seed": seed}, test, pf))
                    if c == CLIN_CALIB:
                        clin.append(clinical_rows(s, f"unet_finetune_{c}s", seed, pf, test))
        print(f"holdout subject {s} scored ({(time.time() - t0) / 60:.1f} min)", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "per_person.csv", index=False)
    pd.DataFrame(clin).to_csv(OUT / "clinical.csv", index=False)


def summarize():
    df = pd.read_csv(OUT / "per_person.csv")
    df["label"] = df.method + df.calib_s.map(lambda c: "" if c == 0 else f" {int(c)}s")
    pp = df.groupby(["label", "subject"])[METRICS].mean().reset_index()
    order = ["avg_beat", *METHODS] + [f"{m} {c}s" for m in ("avg_beat_own", "unet_shift", "unet_finetune")
                                       for c in CALIB_S]
    table = pp.groupby("label")[METRICS].mean().reindex(order).round(3)
    piv = pp.pivot(index="subject", columns="label", values="r").reindex(columns=order).round(3)
    seed_sd = df[df.seed.notna()].groupby(["label", "subject"])["r"].std().groupby("label").median().round(3)
    dev = {}
    for name, path, col in (("simple_models", RESULTS / "simple_models" / "per_person.csv", None),
                            ("gans", RESULTS / "gans" / "per_person.csv", None),
                            ("calibration", RESULTS / "calibration" / "per_person.csv", None)):
        if path.exists():
            d = pd.read_csv(path)
            if "setting" in d:
                d = d[d.setting == "unseen"]
            if "calib_s" in d:
                d["label"] = d.method + d.calib_s.map(lambda c: "" if c == 0 else f" {int(c)}s")
            else:
                d["label"] = d.method
            for lab, g in d.groupby("label"):
                dev.setdefault(lab, g.groupby("subject")["r"].mean().mean())
    cmp = pd.DataFrame({"development (16 people)": pd.Series(dev), "holdout (4 people)": table["r"]}).reindex(order)
    clin = pd.read_csv(OUT / "clinical.csv")
    cl = clin.groupby(["method", "subject"]).mean(numeric_only=True).groupby("method").mean()
    for c in [c for c in cl if c.endswith("_median_err")]:
        cl[c] = clin.groupby(["method", "subject"])[c].mean().abs().groupby("method").mean()
    cl = cl.drop(columns=["seed"], errors="ignore").round(2)
    ep = pd.read_csv(OUT / "epochs.csv", index_col=0)
    lines = ["# Holdout results (generated by scripts/holdout.py, plan: docs/06_holdout_plan.md)", "",
             "Scored once. Holdout people 3, 9, 16, 17; test data = each person's 'late' part. Final models trained on "
             "all 16 development people for a fixed number of epochs (below). Seeded rows averaged over 3 seeds per "
             "person. With 4 people, comparisons are descriptive only.", "",
             "## Means over the 4 holdout people", "", table.to_markdown(), "",
             "## Holdout vs development, mean r", "", cmp.round(3).to_markdown(), "",
             "## Pearson r per person", "", piv.T.to_markdown(), "",
             f"Median seed SD of r per person: {seed_sd.to_dict()}", "",
             "## Clinical intervals (ms; *_beat_mae = median per-beat error, *_median_err = |error of the person's "
             "median interval|)", "", cl.to_markdown(), "",
             "## Fixed training epochs", "", ep.to_markdown(), ""]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", nargs="+", default=["train", "score", "summary"], choices=["train", "score", "summary"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sp = get_splits()
    assert sp["holdout"] == [3, 9, 16, 17]
    if "train" in args.part:
        train_all(sp)
    if "score" in args.part:
        score_all(sp)
    if "summary" in args.part:
        summarize()


if __name__ == "__main__":
    main()
