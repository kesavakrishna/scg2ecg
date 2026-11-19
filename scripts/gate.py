"""Can a simple check stop the models from drawing an ECG out of noise? (scg2ecg/gate.py)

Per fold, the two scores are computed on 8 s windows. Thresholds come from the fold's 2 tune people only (the
models never trained on them; using training people would make the seeds agree too readily): each is set so that
95% of their real SCG windows pass. The gate is then tested on the fold's 4 unseen people:
  real        their 'late' SCG (should pass)
  negatives   built from their own normalised SCG or matched in scale (should fail): white, pink and brown noise;
              the SCG cut into 50 ms or 200 ms blocks and shuffled; and the real SCG with its sign flipped
              (still a real heartbeat, reported separately: a polarity check, not a negative)
A window passes the gate when both scores clear their thresholds. Also checked: on real windows, are the ones
the gate rejects the ones where the U-Net's output is poor?

Writes results/gate/{windows.csv, summary.md, scores.png}.
Run from the repo root:  .venv\\Scripts\\python -m scripts.gate
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scg2ecg.gate import periodicity, seed_agreement, windows
from scg2ecg.metrics import pearson
from scg2ecg.paths import RESULTS
from scg2ecg.splits import get_splits, segments
from scg2ecg.windows import load_segments, normalize_scg, stitch_predict

from .calibration import load_unet
from .simple_models import DEV, FS, LEN, SEEDS

OUT = RESULTS / "gate"
PASS_FRAC = 0.95
NEG_SECONDS = 120


def colored_noise(n, rng, alpha):
    """Noise with power ~ 1/f^alpha (0 white, 1 pink, 2 brown), scaled to unit SD."""
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    spec = (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))) / f ** (alpha / 2)
    x = np.fft.irfft(spec, n)
    return ((x - x.mean()) / x.std()).astype(np.float32)


def shuffled(x, block, rng):
    b = int(block * FS)
    m = len(x) // b
    return normalize_scg(x[:m * b].reshape(m, b)[rng.permutation(m)].ravel(), FS)


def score_signal(x, models, y=None) -> list[dict]:
    preds = [stitch_predict(m, x, LEN, DEV) for m in models]
    out = []
    for a, b in windows(len(x), FS):
        row = {"periodicity": periodicity(x[a:b], FS), "seed_agreement": seed_agreement([p[a:b] for p in preds])}
        if y is not None:
            row["unet_r"] = pearson(y[a:b], np.mean([p[a:b] for p in preds], 0))
        out.append(row)
    return out


def run(sp) -> pd.DataFrame:
    rows = []
    for k, fold in enumerate(sp["folds"]):
        models = [load_unet(k, s) for s in SEEDS]
        for s in fold["tune"]:  # threshold data: the 2 tune people (not trained on, not tested on)
            for t in load_segments(segments(s, FS)["late"], FS):
                rows += [{"fold": k, "subject": s, "role": "threshold", "input": "real", **r}
                         for r in score_signal(t["x"], models, t["y"])]
        for s in fold["test"]:
            rng = np.random.default_rng(1000 * k + s)
            test = load_segments(segments(s, FS)["late"], FS)
            for t in test:
                rows += [{"fold": k, "subject": s, "role": "test", "input": "real", **r}
                         for r in score_signal(t["x"], models, t["y"])]
            x = test[0]["x"][:NEG_SECONDS * FS]
            n = len(x)
            negs = {"white_noise": colored_noise(n, rng, 0), "pink_noise": colored_noise(n, rng, 1),
                    "brown_noise": colored_noise(n, rng, 2), "shuffled_50ms": shuffled(x, 0.05, rng),
                    "shuffled_200ms": shuffled(x, 0.2, rng), "sign_flipped_real": -x}
            for name, xi in negs.items():
                rows += [{"fold": k, "subject": s, "role": "test", "input": name, **r} for r in score_signal(xi, models)]
            print(f"fold {k} subject {s} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "windows.csv", index=False)
    return df


def summarize(df: pd.DataFrame):
    out = []
    for k, g in df.groupby("fold"):
        tr = g[(g.role == "threshold") & (g.input == "real")]
        th = {c: tr[c].quantile(1 - PASS_FRAC) for c in ("periodicity", "seed_agreement")}
        te = g[g.role == "test"].copy()
        te["pass_periodicity"] = te.periodicity >= th["periodicity"]
        te["pass_agreement"] = te.seed_agreement >= th["seed_agreement"]
        te["pass_both"] = te.pass_periodicity & te.pass_agreement
        te["th_periodicity"], te["th_agreement"] = th["periodicity"], th["seed_agreement"]
        out.append(te)
    te = pd.concat(out)
    # per person first, then mean over people
    rates = te.groupby(["input", "subject"])[["pass_periodicity", "pass_agreement", "pass_both"]].mean() \
        .groupby("input").mean().round(3)
    order = ["real", "sign_flipped_real", "white_noise", "pink_noise", "brown_noise", "shuffled_50ms", "shuffled_200ms"]
    rates = rates.reindex(order)
    real = te[te.input == "real"]
    per = real.groupby("subject")[["pass_both"]].mean().round(3)
    q = real.assign(kept=real.pass_both)
    kept = q.groupby("kept")["unet_r"].agg(["count", "median", "mean"]).round(3)
    th = te.groupby("fold")[["th_periodicity", "th_agreement"]].first().round(3)
    lines = ["# Heartbeat check (generated by scripts/gate.py)", "",
             f"8 s windows. Thresholds per fold from the 2 tune people's real windows ({PASS_FRAC:.0%} of them "
             "pass); tested on the fold's 4 unseen people. Pass rates are per person, then averaged over the 16 "
             "development people. Real = the person's 'late' SCG; negatives use 2 min per person.", "",
             "## Fraction of windows that pass", "", rates.to_markdown(), "",
             "real should be near 0.95; sign_flipped_real is a real heartbeat with the sensor's sign reversed "
             "(a polarity check); every other row should be near 0.", "",
             "## Thresholds per fold", "", th.to_markdown(), "",
             "## Real windows: is the U-Net output worse where the gate rejects?", "",
             "unet_r = correlation of the 3-seed mean output with the real ECG in that window.", "",
             kept.to_markdown(), "",
             "## Pass rate on real windows, per person", "", per.T.to_markdown(), ""]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"real": "#1a5276", "sign_flipped_real": "#5dade2", "white_noise": "#c0392b", "pink_noise": "#e67e22",
              "brown_noise": "#a04000", "shuffled_50ms": "#7d3c98", "shuffled_200ms": "#c39bd3"}
    for name in order:
        g = te[te.input == name]
        g = g.sample(min(len(g), 800), random_state=0)
        ax.scatter(g.periodicity, g.seed_agreement, s=5, alpha=0.5, color=colors[name], label=name)
    ax.axvline(te.th_periodicity.median(), color="k", lw=0.8, ls="--")
    ax.axhline(te.th_agreement.median(), color="k", lw=0.8, ls="--")
    ax.set_xlabel("SCG envelope periodicity (autocorrelation peak at 40-150 bpm)")
    ax.set_ylabel("agreement of three U-Net seeds (mean pairwise r)")
    ax.legend(fontsize=7, markerscale=3)
    ax.set_title("8 s windows of unseen people; dashed = median fold threshold", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "scores.png", dpi=100)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = run(get_splits())
    summarize(df)


if __name__ == "__main__":
    main()
