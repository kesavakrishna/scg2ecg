"""Replication rung: the 2021 notebook's protocol and model, ported faithfully, on each development person.

Per person (rest EDF; first 75% of 1 s windows train, last 25% test; same person, same record):
  gan2021   the notebook's model and training loop                        (3 seeds)
  sup2021   ablation: same generator, only the direct Huber loss           (3 seeds)
  gan2023   the 2023 re-run variant (LayerNorm over time), subjects 1, 10  (3 seeds)
  avg_beat         the average-heartbeat baseline fitted on the same 75%  (no learning, no seed)
  avg_beat_true_r  the same average beat pasted at the true R-peaks

Scored like the notebook (Pearson r pooled over test windows, on the z-scored unfiltered 1 kHz lead I),
plus the beat scores. Saved notebook results to compare with: subject 1 (2021 run) r = 0.877;
subject 10 (2023 re-runs) r = 0.790 and 0.771.

Writes results/replication/{per_run.csv, summary.md, examples.png}.
Run from the repo root:  .venv\\Scripts\\python -m scripts.replicate_original
"""
from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wfdb
from scipy.stats import wilcoxon

from scg2ecg.baselines import AverageBeat
from scg2ecg.legacy import legacy_data, predict, train_gan, train_supervised
from scg2ecg.metrics import beat_counts, detect_r_peaks, pearson, prd, prd_legacy, rmse, summarize_beats
from scg2ecg.paths import RAW, RESULTS
from scg2ecg.splits import get_splits

FS = 1000
SEEDS = range(3)
OUT = RESULTS / "replication"
SAVED = {(1, "gan2021"): [0.877], (10, "gan2023"): [0.790, 0.771]}


def r_marks(subject: int) -> np.ndarray:
    ann = wfdb.rdann(str(RAW / f"b{subject:03d}"), "atr")
    return np.round(ann.sample[ann.chan == 0] / 5).astype(np.int64)


def scores(target, pred, ref, oracle) -> dict:
    det = detect_r_peaks(pred, FS)
    return {"r": pearson(target, pred), "rmse": rmse(target, pred), "mae": float(np.mean(np.abs(target - pred))),
            "prd": prd(target, pred), "prd_notebook": prd_legacy(target, pred),
            "r_vs_avg_beat_true_r": pearson(pred, oracle),
            **summarize_beats([beat_counts(ref, det, len(pred), FS)])}


def run_subject(s: int) -> tuple[list[dict], dict]:
    """Every method and seed for one person (runs in a worker process)."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows, keep = [], {}
    d = legacy_data(s)
    a, b = d["n_train"] * 1000, d["n_windows"] * 1000
    target = d["ecg"][a:b]
    marks = r_marks(s)
    ref = marks[(marks >= a) & (marks < b)] - a
    tr = marks[marks < a]

    ab = AverageBeat(FS).fit([(d["scg"][:a], d["ecg"][:a], tr)])
    ab.tune([(d["scg"][:a], d["ecg"][:a], tr)])
    oracle = ab.paste(ref, b - a)
    pred_ab = ab.predict(d["scg"][a:b])[0]
    base = {"subject": s, "n_train_windows": d["n_train"], "n_test_windows": d["n_windows"] - d["n_train"]}
    rows.append({**base, "method": "avg_beat", "seed": None, **scores(target, pred_ab, ref, oracle)})
    rows.append({**base, "method": "avg_beat_true_r", "seed": None, **scores(target, oracle, ref, oracle)})
    keep[(s, "avg_beat")] = pred_ab
    keep[(s, "target")] = target

    noise = (0.01 * np.random.default_rng(0).standard_normal((16, 1, 1000))).astype(np.float32)
    runs = [("gan2021", "2021"), ("sup2021", "2021")] + ([("gan2023", "2023")] if s in (1, 10) else [])
    for method, variant in runs:
        for seed in SEEDS:
            t0 = time.time()
            extra = {}
            if method.startswith("gan"):
                g, log = train_gan(d["X_train"], d["Y_train"], seed, variant, device)
                tail = slice(-50, None)
                extra = {"d_loss_last50": float(np.mean(log["d_loss"][tail])),
                         "d_out_std_last50": float(np.mean(log["d_out_std"][tail])),
                         "g_adv_last50": float(np.mean(log["g_adv"][tail])),
                         "g_sup_last50": float(np.mean(log["g_sup"][tail])),
                         "g_cyc_last50": float(np.mean(log["g_cyc"][tail]))}
            else:
                g = train_supervised(d["X_train"], d["Y_train"], seed, variant, device)
            pred = predict(g, d["X_test"], device).reshape(-1)
            out_noise = predict(g, noise, device)
            rows.append({**base, "method": method, "seed": seed, **scores(target, pred, ref, oracle),
                         "out_std_real": float(pred.reshape(-1, 1000).std(1).mean()),
                         "out_std_noise_input": float(out_noise.std(axis=(1, 2)).mean()),
                         "target_std": float(target.reshape(-1, 1000).std(1).mean()),
                         "seconds": time.time() - t0, **extra})
            print(f"  subject {s} {method} seed {seed}: r={rows[-1]['r']:.3f} ({time.time() - t0:.0f}s)", flush=True)
            if seed == 0:
                keep[(s, method)] = pred
    return rows, keep


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sp = get_splits()
    rows, keep = [], {}
    t0 = time.time()
    # one process: three workers sharing an 8 GB laptop GPU ran slower than one
    for s_rows, s_keep in map(run_subject, sp["dev"]):
        rows += s_rows
        keep.update(s_keep)
        sub = pd.DataFrame(s_rows)
        print(f"[{(time.time() - t0) / 60:5.1f} min] subject {sub.subject.iloc[0]:2d}: " +
              "  ".join(f"{m} r={g['r'].mean():.3f}" for m, g in sub.groupby("method", sort=False)), flush=True)
        pd.DataFrame(rows).to_csv(OUT / "per_run.csv", index=False)


    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_run.csv", index=False)

    # per person: mean over seeds
    pp = df.groupby(["subject", "method"])[["r", "rmse", "prd", "prd_notebook", "beat_se", "beat_ppv",
                                             "r_timing_abs_ms", "rr_mae_ms", "r_vs_avg_beat_true_r"]].mean()
    pr = pp["r"].unstack()
    seed_sd = df[df.seed.notna()].groupby(["subject", "method"])["r"].std().unstack()
    means = pp.groupby("method").mean().round(3)
    print(means.to_string())

    tests = []
    for a_, b_ in (("gan2021", "sup2021"), ("gan2021", "avg_beat"), ("sup2021", "avg_beat")):
        both = pr[[a_, b_]].dropna()
        w = wilcoxon(both[a_], both[b_])
        tests.append(f"- {a_} vs {b_}: {a_} better for {(both[a_] > both[b_]).sum()} of {len(both)} people; "
                     f"mean difference {(both[a_] - both[b_]).mean():+.3f}; Wilcoxon p = {w.pvalue:.3g}")

    saved = []
    for (s, m), vals in SAVED.items():
        ours = df[(df.subject == s) & (df.method == m)]["r"]
        saved.append(f"- subject {s}, {m}: saved notebook r = {vals}; port, 3 seeds: "
                     f"{np.round(ours.values, 3).tolist()} (mean {ours.mean():.3f})")

    gan = df[df.method == "gan2021"]
    lines = [
        "# Replication of the 2021 notebook (generated by scripts/replicate_original.py)", "",
        "Protocol: per person, rest EDF, first 75% of 1 s windows train / last 25% test (same person, same "
        "5 min record), z-scored unfiltered lead I at 1 kHz, 1 epoch, batch size 1. 16 development people.", "",
        "## Against the notebook's saved numbers", "", *saved, "",
        "## Mean over people (seed-averaged)", "", means.to_markdown(), "",
        "## Pearson r per person (mean over 3 seeds where seeded)", "", pr.round(3).to_markdown(), "",
        "## Seed spread: SD of r across 3 seeds", "", seed_sd.round(3).to_markdown(), "",
        "## Person-by-person comparisons", "", *tests, "",
        "## Is the discriminator doing anything? (gan2021, last 50 training steps)", "",
        f"- D loss: mean {gan['d_loss_last50'].mean():.4f}, SD across runs {gan['d_loss_last50'].std():.4f}",
        f"- SD of D's output on a real window: mean {gan['d_out_std_last50'].mean():.4f} "
        "(instance norm on the 1-channel output fixes it to |gain| for every input)",
        f"- generator adversarial loss {gan['g_adv_last50'].mean():.4f}, supervised {gan['g_sup_last50'].mean():.4f}, "
        f"'cycle' {gan['g_cyc_last50'].mean():.2e}", "",
        "## Can it draw a flat line?", "",
        f"- output SD per 1 s window on real SCG: {gan['out_std_real'].mean():.3f}; on near-zero noise input: "
        f"{gan['out_std_noise_input'].mean():.3f}; real ECG target: {gan['target_std'].mean():.3f}",
        "",
        "r_vs_avg_beat_true_r: correlation between a method's output and the average beat pasted at the true "
        "R-peaks. prd_notebook is the notebook's formula (x100 inside the square root).",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[2:8] + tests))

    fig, axes = plt.subplots(2, 1, figsize=(13, 5.5), sharex=True)
    t = np.arange(5000) / FS
    for ax, s in zip(axes, (1, 10)):
        ax.plot(t, keep[(s, "target")][:5000], "k", lw=1.0, label="real ECG (z-scored, unfiltered)")
        ax.plot(t, keep[(s, "gan2021")][:5000], color="#c0392b", lw=0.9, label="2021 model (port, seed 0)")
        ax.plot(t, keep[(s, "avg_beat")][:5000] - 4, color="#2471a3", lw=0.9, label="average beat (offset)")
        ax.set_title(f"subject {s}: first 5 s of the test part", fontsize=9, loc="left")
        ax.set_yticks([])
    axes[0].legend(fontsize=7, ncol=3, loc="upper right")
    axes[-1].set_xlabel("seconds")
    fig.tight_layout()
    fig.savefig(OUT / "examples.png", dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    main()
