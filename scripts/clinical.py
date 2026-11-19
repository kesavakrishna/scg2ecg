"""Phase 5: would a clinician accept the reconstructed ECG? Unseen-person setting, 16 development people.

Signals per person (test data = the 'late' part, as everywhere):
  real_ecg, avg_beat (population), unet, pix2pix_adv0.1, cycle_paired                           no calibration
  avg_beat_own_30s, unet_shift_30s, unet_finetune_30s                                             30 s calibration
Seeded models: all 3 seeds.

1. Intervals. Every signal is delineated by the same method (neurokit2 'dwt') at the database's R-peak marks,
   so the beats compared are the same. Per beat: QRS duration, PR interval, QT interval, P wave found or not.
   Scores per person: median per-beat absolute error against the real ECG's delineation, and the error of the
   person's median interval (the number a clinician would read off).
2. Irregular beats: beats whose R-R interval differs from the local median by >20% (as in the data check).
   Does the output still have a beat there, and how well is it shaped, compared with the person's regular beats?
3. Flat-line test: inputs with no heartbeat (zeros, white noise, SCG cut into shuffled 50 ms blocks). An honest
   model should not draw a regular ECG from them. Score: beats per minute found in the output, output SD.

Writes results/clinical/{per_person.csv, irregular.csv, flatline.csv, summary.md}.
Run from the repo root:  .venv\\Scripts\\python -m scripts.clinical
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import median_filter
from scipy.stats import wilcoxon

from scg2ecg.baselines import AverageBeat
from scg2ecg.metrics import detect_r_peaks, match_beats, pearson
from scg2ecg.models import UNet1d
from scg2ecg.paths import RESULTS
from scg2ecg.splits import get_splits, segments
from scg2ecg.windows import load_segments, normalize_scg, stitch_predict

from .calibration import apply_shift, avg_beat_pieces, best_shift, calib_segment, finetune, load_unet, raw_piece
from .simple_models import DEV, FS, LEN, SEEDS

OUT = RESULTS / "clinical"
CALIB = 30
GANS = ["pix2pix_adv0.1", "cycle_paired", "cgan_2g6d"]  # flat-line test
BEAT_GANS = ["pix2pix_adv0.1", "cycle_paired"]  # interval scoring (cgan_2g6d is scored in results/gans)
WORKERS = 4
PRE, POST = int(0.25 * FS), int(0.45 * FS)


# ---------------------------------------------------------------- per-beat measurements (runs in workers)

def delineate(job):
    """job = (key, signal, r_marks). Returns key and a per-beat DataFrame of intervals (ms) and the beat itself."""
    import neurokit2 as nk
    key, x, r = job
    r = r[(r > PRE) & (r < len(x) - POST)]
    out = pd.DataFrame({"r": r})
    names = {"P_Onsets": "p_on", "P_Peaks": "p_pk", "R_Onsets": "qrs_on", "R_Offsets": "qrs_off", "T_Offsets": "t_off"}
    try:
        _, w = nk.ecg_delineate(np.asarray(x, float), rpeaks=r, sampling_rate=FS, method="dwt")
        for k, v in names.items():
            out[v] = np.asarray(w[f"ECG_{k}"], float)[:len(r)]
    except Exception:
        for v in names.values():
            out[v] = np.nan
    ms = 1000.0 / FS
    out["qrs_ms"] = (out.qrs_off - out.qrs_on) * ms
    out["pr_ms"] = (out.qrs_on - out.p_on) * ms
    out["qt_ms"] = (out.t_off - out.qrs_on) * ms
    out["p_found"] = out.p_pk.notna()
    # implausible delineations count as not found
    for c, lo, hi in (("qrs_ms", 40, 200), ("pr_ms", 60, 320), ("qt_ms", 200, 600)):
        out.loc[(out[c] < lo) | (out[c] > hi), c] = np.nan
    return key, out[["r", "qrs_ms", "pr_ms", "qt_ms", "p_found"]]


# ---------------------------------------------------------------- signals for one person

def gan_model(name, k, seed):
    path = RESULTS / "gans" / f"{name}_fold{k}_seed{seed}.pt"
    if not path.exists():
        return None
    m = UNet1d().to(DEV)
    m.load_state_dict(torch.load(path, map_location=DEV))
    return m.eval()


def person_signals(s, k, ab, best_steps) -> dict:
    """{(method, seed): [one array per test segment]} for one person."""
    test = load_segments(segments(s, FS)["late"], FS)
    raw = [raw_piece(sg) for sg in segments(s, FS)["late"]]
    sig = {("real_ecg", -1): [t["y"] for t in test],
           ("avg_beat", -1): [ab.predict(x)[0] for x, _, _ in raw]}
    cx, cy, cr = raw_piece(calib_segment(s, CALIB))
    own = AverageBeat(FS).fit([(cx, cy, cr)])
    own.tune([(cx, cy, cr)])
    sig[(f"avg_beat_own_{CALIB}s", -1)] = [own.predict(x)[0] for x, _, _ in raw]
    chunk = load_segments([calib_segment(s, CALIB)], FS)[0]
    for seed in SEEDS:
        m = load_unet(k, seed)
        p = [stitch_predict(m, t["x"], LEN, DEV) for t in test]
        sig[("unet", seed)] = p
        d = best_shift(chunk["y"], stitch_predict(m, chunk["x"], LEN, DEV))
        sig[(f"unet_shift_{CALIB}s", seed)] = [apply_shift(q, d) for q in p]
        ft = finetune(m, chunk, [best_steps], seed)[best_steps]
        sig[(f"unet_finetune_{CALIB}s", seed)] = [stitch_predict(ft, t["x"], LEN, DEV) for t in test]
        for g in BEAT_GANS:
            gm = gan_model(g, k, seed)
            if gm is not None:
                sig[(g, seed)] = [stitch_predict(gm, t["x"], LEN, DEV) for t in test]
    return sig, [t["r"] for t in test]


def irregular_flags(r: np.ndarray) -> np.ndarray:
    """Per beat: True if the R-R interval before it differs from the local median by >20%."""
    rr = np.diff(r).astype(float)
    flag = np.zeros(len(r), bool)
    if len(rr) > 11:
        local = median_filter(rr, size=11, mode="nearest")
        flag[1:] = np.abs(rr - local) > 0.2 * local
    return flag


def beat_corr(y, p, r):
    return np.array([pearson(y[i - PRE:i + POST], p[i - PRE:i + POST]) for i in r])


# ---------------------------------------------------------------- flat-line test

def flatline(sp) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(0)
    n = 60 * FS
    for k, fold in enumerate(sp["folds"]):
        s = fold["test"][0]
        x = load_segments(segments(s, FS)["late"], FS)[0]["x"][:n]
        b = int(0.05 * FS)
        shuffled = x[:n // b * b].reshape(-1, b)[rng.permutation(n // b)].ravel()
        inputs = {"zeros": np.zeros(n, np.float32),
                  "white_noise": rng.standard_normal(n).astype(np.float32),
                  "shuffled_scg_50ms": normalize_scg(shuffled, FS),
                  "real_scg": x}
        for seed in SEEDS:
            models = {"unet": load_unet(k, seed), **{g: gan_model(g, k, seed) for g in GANS}}
            for name, m in models.items():
                if m is None:
                    continue
                for inp, xi in inputs.items():
                    p = stitch_predict(m, xi, LEN, DEV)
                    det = detect_r_peaks(p, FS)
                    rows.append({"method": name, "fold": k, "seed": seed, "input": inp, "out_sd": float(p.std()),
                                 "beats_per_min": len(det) / (n / FS / 60)})
    return pd.DataFrame(rows)


def flatline_figure(sp):
    """What the fold-0 seed-0 U-Net draws from each kind of input."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(1)
    n, W = 30 * FS, 5 * FS
    s = sp["folds"][0]["test"][0]
    seg = load_segments(segments(s, FS)["late"], FS)[0]
    x = seg["x"][:n]
    b = int(0.05 * FS)
    inputs = {"real SCG": x, "SCG cut into 50 ms blocks, shuffled": normalize_scg(x.reshape(-1, b)[rng.permutation(n // b)].ravel(), FS),
              "white noise": rng.standard_normal(n).astype(np.float32), "zeros": np.zeros(n, np.float32)}
    m = load_unet(0, 0)
    fig, axes = plt.subplots(len(inputs), 1, figsize=(12, 7), sharex=True)
    t = np.arange(W) / FS
    a = n // 2
    for ax, (name, xi) in zip(axes, inputs.items()):
        p = stitch_predict(m, xi, LEN, DEV)
        ax.plot(t, xi[a:a + W] / 4 + 3, color="#999999", lw=0.6, label="input (scaled)")
        ax.plot(t, p[a:a + W], color="#c0392b", lw=0.9, label="U-Net output")
        if name == "real SCG":
            ax.plot(t, seg["y"][a:a + W] - 3, "k", lw=0.8, label="real ECG (offset)")
        ax.set_title(f"input: {name}", fontsize=8, loc="left")
        ax.set_yticks([])
    axes[0].legend(fontsize=7, ncol=3, loc="upper right")
    axes[-1].set_xlabel(f"seconds (subject {s}'s unseen-person U-Net, fold 0 seed 0)")
    fig.tight_layout()
    fig.savefig(OUT / "flatline.png", dpi=90)
    plt.close(fig)


# ---------------------------------------------------------------- main

def run(sp):
    choice = pd.read_csv(RESULTS / "calibration" / "finetune_choice.csv")
    path, ipath = OUT / "per_person.csv", OUT / "irregular.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []  # resumable: finished people are kept
    irr_rows = pd.read_csv(ipath).to_dict("records") if ipath.exists() else []
    done = {r["subject"] for r in rows}
    pool = ProcessPoolExecutor(WORKERS)
    for k, fold in enumerate(sp["folds"]):
        ch = choice[(choice.fold == k) & (choice.calib_s == CALIB)]
        best_steps = int(ch.groupby("steps")["r"].mean().idxmax())
        ab = AverageBeat(FS).fit(avg_beat_pieces(fold["train"], "all"))
        ab.tune(avg_beat_pieces(fold["tune"], "all"))
        for s in fold["test"]:
            if s in done:
                continue
            t0 = time.time()
            sig, refs = person_signals(s, k, ab, best_steps)
            jobs = [((m, seed, i), x, refs[i]) for (m, seed), xs in sig.items() for i, x in enumerate(xs)]
            beats = dict(pool.map(delineate, jobs))
            real = {i: beats[("real_ecg", -1, i)] for i in range(len(refs))}
            for (m, seed), xs in sig.items():
                per = []
                for i, x in enumerate(xs):
                    b, rb = beats[(m, seed, i)], real[i]
                    d = b.merge(rb, on="r", suffixes=("", "_real"))
                    d["irregular"] = irregular_flags(rb.r.to_numpy())[np.searchsorted(rb.r, d.r)]
                    d["beat_r"] = beat_corr(sig[("real_ecg", -1)][i], x, d.r.to_numpy())
                    det = detect_r_peaks(x, FS)
                    ri, _ = match_beats(d.r.to_numpy(), det, FS)
                    d["present"] = False
                    d.loc[d.index[ri], "present"] = True
                    per.append(d)
                d = pd.concat(per, ignore_index=True)
                row = {"subject": s, "fold": k, "method": m, "seed": seed, "beats": len(d)}
                for c in ("qrs_ms", "pr_ms", "qt_ms"):
                    row[f"{c}_beat_mae"] = float(np.nanmedian(np.abs(d[c] - d[f"{c}_real"])))
                    row[f"{c}_median_err"] = float(np.nanmedian(d[c]) - np.nanmedian(d[f"{c}_real"]))
                    row[f"{c}_found"] = float(d[c].notna().mean())
                row["p_found"] = float(d.p_found.mean())
                row["p_agree"] = float((d.p_found == d.p_found_real).mean())
                rows.append(row)
                for irr in (False, True):
                    sub = d[d.irregular == irr]
                    if len(sub):
                        irr_rows.append({"subject": s, "method": m, "seed": seed, "irregular": irr, "beats": len(sub),
                                         "present": float(sub.present.mean()), "beat_r": float(sub.beat_r.median())})
            print(f"fold {k} subject {s}: {len(sig)} signals, {(time.time() - t0) / 60:.1f} min", flush=True)
            pd.DataFrame(rows).to_csv(path, index=False)
            pd.DataFrame(irr_rows).to_csv(ipath, index=False)
    pool.shutdown()


def summarize():
    df = pd.read_csv(OUT / "per_person.csv")
    order = ["real_ecg", "avg_beat", "unet", *BEAT_GANS, f"avg_beat_own_{CALIB}s", f"unet_shift_{CALIB}s",
             f"unet_finetune_{CALIB}s"]
    order = [m for m in order if m in set(df.method)]
    cols = [c for c in df.columns if c.endswith(("_beat_mae", "_median_err", "_found")) or c in ("p_found", "p_agree")]
    pp = df.groupby(["method", "subject"])[cols].mean().reset_index()
    pp_abs = pp.copy()
    for c in [c for c in cols if c.endswith("_median_err")]:
        pp_abs[c] = pp_abs[c].abs()
    table = pp_abs.groupby("method")[cols].mean().reindex(order).round(2)
    bias = pp.groupby("method")[[c for c in cols if c.endswith("_median_err")]].mean().reindex(order).round(1)
    tests = []
    for c in ("qrs_ms_beat_mae", "pr_ms_beat_mae", "qt_ms_beat_mae"):
        piv = pp.pivot(index="subject", columns="method", values=c)
        for a, b in (("unet", "avg_beat"), (f"unet_finetune_{CALIB}s", "unet"),
                     (f"unet_finetune_{CALIB}s", f"avg_beat_own_{CALIB}s"), *[(g, "unet") for g in BEAT_GANS]):
            if a in piv and b in piv:
                d = (piv[a] - piv[b]).dropna()
                tests.append(f"- {c}: {a} lower than {b} for {(d < 0).sum()} of {len(d)} people; "
                             f"mean difference {d.mean():+.1f} ms; Wilcoxon p = {wilcoxon(d).pvalue:.3g}")
    irr = pd.read_csv(OUT / "irregular.csv")
    irr_tab = irr.groupby(["method", "irregular"]).apply(
        lambda g: pd.Series({"beats": g.beats.sum() / g.seed.nunique(),
                             "present": np.average(g.present, weights=g.beats),
                             "beat_r_median": g.beat_r.median()})).unstack().reindex(order).round(3)
    by_person = irr[irr.irregular].groupby(["subject", "method"]).beat_r.mean().unstack().reindex(columns=order)
    big = irr[irr.irregular].groupby("subject").beats.max()
    by_person = by_person.loc[big[big >= 10].index].round(3)
    lines = ["# Clinical scoring, unseen people (generated by scripts/clinical.py)", "",
             "Every signal is delineated with neurokit2 'dwt' at the database's R-peak marks. Intervals are compared "
             "beat by beat with the real ECG's delineation; real_ecg's own row shows the delineator against itself "
             "(zero by construction). Seeded rows averaged over 3 seeds per person, then over 16 people.", "",
             "## Interval errors (ms) and how often the delineator finds each wave", "",
             "*_beat_mae: median per-beat |output - real|. *_median_err: |the person's median interval in the "
             "output - in the real ECG|. *_found: fraction of beats with a plausible interval. p_agree: fraction of "
             "beats where output and real agree on whether there is a P wave.", "",
             table.to_markdown(), "",
             "## Bias: mean over people of (output median interval - real median interval), ms", "",
             bias.to_markdown(), "",
             "## Person-by-person", "", *tests, "",
             "## Irregular beats (R-R differs >20% from the local median)", "",
             "present: fraction of beats with an R-peak detected in the output within 150 ms. beat_r_median: median "
             "per-beat correlation with the real beat (-250 to +450 ms).", "",
             irr_tab.to_markdown(), "",
             "Per person, median beat_r on irregular beats (people with at least 10 irregular test beats):", "",
             by_person.to_markdown(), ""]
    fl = OUT / "flatline.csv"
    if fl.exists():
        f = pd.read_csv(fl)
        ft = f.groupby(["method", "input"]).agg(beats_per_min=("beats_per_min", "mean"),
                                                out_sd=("out_sd", "mean")).unstack().round(2)
        lines += ["## Flat-line test: what each model draws from inputs with no heartbeat", "",
                  "One minute of input per fold and seed (zeros; white noise; the real SCG cut into 50 ms blocks "
                  "and shuffled; the real SCG for reference). beats_per_min: R-peaks neurokit finds in the output. "
                  "out_sd: output SD (the real ECG target has SD 1).", "", ft.to_markdown(), ""]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", nargs="+", default=["beats", "flatline", "summary"],
                    choices=["beats", "flatline", "figure", "summary"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sp = get_splits()
    if "beats" in args.part:
        run(sp)
    if "flatline" in args.part:
        flatline(sp).to_csv(OUT / "flatline.csv", index=False)
        flatline_figure(sp)
    if "figure" in args.part:
        flatline_figure(sp)
    if "summary" in args.part:
        summarize()


if __name__ == "__main__":
    main()
