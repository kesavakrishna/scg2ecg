"""Phase 4b: can a short paired recording of a new person fix most of the unseen-person error?

Phase 4 found that much of the U-Net's error on a new person is a constant time shift (the person's own
delay between R-wave and chest vibration). Here each development person, as an unseen person, gives a
calibration chunk: the first C seconds (C = 10, 30, 60) of their music record, SCG and ECG together,
recorded ~35 min before the test data (the 'late' part, as everywhere else). Methods:

  unet                  the Phase 3 unseen-person U-Net, no calibration
  unet_shift            its output moved by the shift that maximises r on the calibration chunk
  unet_finetune         the U-Net fine-tuned on the calibration chunk. The number of steps is chosen per fold
                        and C on the fold's 2 tune people (their own chunks and late parts), never on test people
  avg_beat              the population average-heartbeat baseline, no calibration
  avg_beat_shift        its output moved by the shift fitted on the calibration chunk
  avg_beat_own          the average-heartbeat baseline built from the person's own calibration beats only

Writes results/calibration/{per_person.csv, finetune_choice.csv, summary.md, examples.png}.
Run from the repo root:  .venv\\Scripts\\python -m scripts.calibration
"""
from __future__ import annotations

import copy
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon

from scg2ecg.baselines import AverageBeat
from scg2ecg.data import load, prepared, record_name
from scg2ecg.metrics import morphology, pearson, score
from scg2ecg.models import UNet1d
from scg2ecg.paths import RESULTS
from scg2ecg.splits import get_splits, segments
from scg2ecg.windows import load_segments, stitch_predict

from .simple_models import DEV, FS, LEN, SEEDS

OUT = RESULTS / "calibration"
CALIB_S = (10, 30, 60)
MAX_LAG_MS = 200
STEP_GRID = (25, 50, 100, 200, 400)
FT_LR = 1e-4
FT_BATCH = 16
METRICS = ["r", "rmse", "beat_se", "beat_ppv", "r_timing_abs_ms", "rr_mae_ms", "template_r", "beat_r", "r_amp_ratio"]


def calib_segment(s: int, c: int) -> tuple:
    return (record_name("m", s), 0, c * FS)


def test_segments(s: int) -> list[dict]:
    return load_segments(segments(s, FS)["late"], FS)


def best_shift(y: np.ndarray, p: np.ndarray) -> int:
    """Shift d (samples, positive = output late) maximising r(y[i], p[i + d]) over the overlap."""
    L = int(MAX_LAG_MS * FS / 1000)
    n = len(y)
    rs = [pearson(y[L:n - L], p[L + d:n - L + d]) for d in range(-L, L + 1)]
    return int(np.argmax(rs)) - L


def apply_shift(p: np.ndarray, d: int) -> np.ndarray:
    """Move the output earlier by d samples (later if d < 0), repeating the edge value."""
    if d == 0:
        return p
    return np.concatenate([p[d:], np.full(d, p[-1])]) if d > 0 else np.concatenate([np.full(-d, p[0]), p[:d]])


def finetune(model: UNet1d, chunk: dict, steps: list[int], seed: int, eval_fn=None):
    """Fine-tune a copy on random crops of one calibration chunk. Returns {step: model copy or eval score}."""
    m = copy.deepcopy(model).train()
    opt = torch.optim.AdamW(m.parameters(), lr=FT_LR, weight_decay=0.0)
    rng = np.random.default_rng(seed)
    x, y = chunk["x"], chunk["y"]
    out = {}
    for step in range(1, max(steps) + 1):
        a = rng.integers(0, len(x) - LEN + 1, FT_BATCH)
        xb = torch.from_numpy(np.stack([x[i:i + LEN] for i in a])[:, None]).to(DEV)
        yb = torch.from_numpy(np.stack([y[i:i + LEN] for i in a])[:, None]).to(DEV)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
            loss = F.l1_loss(m(xb).float(), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step in steps:
            m.eval()
            out[step] = eval_fn(m) if eval_fn else copy.deepcopy(m)
            m.train()
    return out


def load_unet(k: int, seed: int) -> UNet1d:
    m = UNet1d().to(DEV)
    m.load_state_dict(torch.load(RESULTS / "simple_models" / f"unet_unseen_fold{k}_seed{seed}.pt", map_location=DEV))
    return m.eval()


def scored(base: dict, test: list[dict], preds: list[np.ndarray]) -> dict:
    args = ([t["y"] for t in test], preds, [t["r"] for t in test], FS)
    return {**base, **score(*args), **morphology(*args)}


def choose_steps(fold, k: int) -> dict[int, int]:
    """Per calibration length: the fine-tuning step count with the best mean r on the fold's tune people."""
    rows = []
    for s in fold["tune"]:
        test = test_segments(s)
        for c in CALIB_S:
            chunk = load_segments([calib_segment(s, c)], FS)[0]
            for seed in SEEDS:
                r_at = finetune(load_unet(k, seed), chunk, list(STEP_GRID), seed,
                                lambda m: float(np.mean([pearson(t["y"], stitch_predict(m, t["x"], LEN, DEV))
                                                         for t in test])))
                rows += [{"fold": k, "tune_subject": s, "calib_s": c, "seed": seed, "steps": st, "r": r}
                         for st, r in r_at.items()]
    df = pd.DataFrame(rows)
    best = df.groupby(["calib_s", "steps"])["r"].mean().unstack().idxmax(axis=1).to_dict()
    return best, df


def avg_beat_pieces(people, part):
    out = []
    for s in people:
        for name, a, b in segments(s, FS)[part]:
            scg, ecg, r = prepared(name, FS)
            out.append((scg[a:b], ecg[a:b], r[(r >= a) & (r < b)] - a))
    return out


def raw_piece(seg):
    name, a, b = seg
    scg, ecg, r = prepared(name, FS)
    return scg[a:b], ecg[a:b], r[(r >= a) & (r < b)] - a


def run(sp):
    """Resumable: people already in per_person.csv are skipped, and a fold's step choice is reused from
    finetune_choice.csv (both are written as the run goes)."""
    path, cpath = OUT / "per_person.csv", OUT / "finetune_choice.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {r["subject"] for r in rows if r["method"] == "unet_finetune" and r["calib_s"] == CALIB_S[-1]
            and r["seed"] == SEEDS[-1]}
    for k, fold in enumerate(sp["folds"]):
        t0 = time.time()
        old = pd.read_csv(cpath) if cpath.exists() else pd.DataFrame(columns=["fold"])
        if all(s in done for s in fold["test"]):
            print(f"fold {k}: all test people done, skipping", flush=True)
            if not (old.fold == k).any():  # the step-choice table was lost in an interrupted run: redo it
                pd.concat([old, choose_steps(fold, k)[1]], ignore_index=True).to_csv(cpath, index=False)
            continue
        if (old.fold == k).any():
            ch = old[old.fold == k]
        else:
            _, ch = choose_steps(fold, k)
            pd.concat([old, ch], ignore_index=True).to_csv(cpath, index=False)
        best_steps = {int(c): int(st) for c, st in  # ints: values read back from the CSV are floats
                      ch.groupby(["calib_s", "steps"])["r"].mean().unstack().idxmax(axis=1).items()}
        print(f"fold {k}: fine-tune steps chosen on tune people {best_steps} ({time.time() - t0:.0f}s)", flush=True)

        ab = AverageBeat(FS).fit(avg_beat_pieces(fold["train"], "all"))
        ab.tune(avg_beat_pieces(fold["tune"], "all"))
        for s in fold["test"]:
            if s in done:
                continue
            rows = [r for r in rows if r["subject"] != s]  # drop a partly scored person
            test = test_segments(s)
            raw_test = [raw_piece(sg) for sg in segments(s, FS)["late"]]
            base = {"fold": k, "subject": s}
            # average-beat baseline
            p_ab = [ab.predict(x)[0] for x, _, _ in raw_test]
            rows.append(scored({**base, "method": "avg_beat", "calib_s": 0, "seed": None}, test, p_ab))
            for c in CALIB_S:
                cx, cy, cr = raw_piece(calib_segment(s, c))
                d = best_shift(cy, ab.predict(cx)[0])
                rows.append(scored({**base, "method": "avg_beat_shift", "calib_s": c, "seed": None, "shift_ms": d * 1000 / FS},
                                   test, [apply_shift(p, d) for p in p_ab]))
                own = AverageBeat(FS).fit([(cx, cy, cr)])
                own.tune([(cx, cy, cr)])
                rows.append(scored({**base, "method": "avg_beat_own", "calib_s": c, "seed": None}, test,
                                   [own.predict(x)[0] for x, _, _ in raw_test]))
            # U-Net
            for seed in SEEDS:
                m = load_unet(k, seed)
                p_u = [stitch_predict(m, t["x"], LEN, DEV) for t in test]
                rows.append(scored({**base, "method": "unet", "calib_s": 0, "seed": seed}, test, p_u))
                for c in CALIB_S:
                    chunk = load_segments([calib_segment(s, c)], FS)[0]
                    d = best_shift(chunk["y"], stitch_predict(m, chunk["x"], LEN, DEV))
                    rows.append(scored({**base, "method": "unet_shift", "calib_s": c, "seed": seed, "shift_ms": d * 1000 / FS},
                                       test, [apply_shift(p, d) for p in p_u]))
                    steps = best_steps[c]
                    ft = finetune(m, chunk, [steps], seed)[steps]
                    rows.append(scored({**base, "method": "unet_finetune", "calib_s": c, "seed": seed, "steps": steps},
                                       test, [stitch_predict(ft, t["x"], LEN, DEV) for t in test]))
                    del ft
            print(f"  fold {k} subject {s} done ({(time.time() - t0) / 60:.1f} min)", flush=True)
            pd.DataFrame(rows).to_csv(path, index=False)


def summarize(sp):
    df = pd.read_csv(OUT / "per_person.csv")
    df["label"] = df.method + df.calib_s.map(lambda c: "" if c == 0 else f" {c}s")
    pp = df.groupby(["label", "method", "calib_s", "subject"])[METRICS].mean().reset_index()
    order = (pp[["label", "method", "calib_s"]].drop_duplicates()
             .assign(o=lambda d: d.method.map({"avg_beat": 0, "avg_beat_shift": 1, "avg_beat_own": 2, "unet": 3,
                                                "unet_shift": 4, "unet_finetune": 5}))
             .sort_values(["o", "calib_s"]).label.tolist())
    table = pp.groupby("label")[METRICS].mean().reindex(order).round(3)
    spread = pp.groupby("label")["r"].agg(["mean", "std", "min", "median", "max"]).reindex(order).round(3)
    piv = pp.pivot(index="subject", columns="label", values="r")
    tests = []
    for a, b in ([(f"unet_shift {c}s", "unet") for c in CALIB_S] + [(f"unet_finetune {c}s", "unet") for c in CALIB_S] +
                 [(f"unet_finetune {c}s", f"unet_shift {c}s") for c in CALIB_S] +
                 [(f"unet_finetune {c}s", f"avg_beat_own {c}s") for c in CALIB_S] +
                 [(f"unet_shift {c}s", f"avg_beat_shift {c}s") for c in CALIB_S]):
        d = (piv[a] - piv[b]).dropna()
        tests.append(f"- {a} vs {b}: better for {(d > 0).sum()} of {len(d)} people; mean r difference {d.mean():+.3f}; "
                     f"Wilcoxon p = {wilcoxon(d).pvalue:.3g}")
    shifts = df[df.method == "unet_shift"].groupby(["subject", "calib_s"])["shift_ms"].mean().unstack().round(0)
    ch = pd.read_csv(OUT / "finetune_choice.csv")
    chosen = ch.groupby(["fold", "calib_s", "steps"])["r"].mean().unstack().round(3)
    lags = RESULTS / "gans" / "lags.csv"
    oracle = pd.read_csv(lags).query("method == 'unet'").groupby("subject")[["best_lag_ms", "r_best_lag"]].mean() \
        if lags.exists() else None
    same = pd.read_csv(RESULTS / "simple_models" / "per_person.csv").query("setting == 'same' and method == 'unet'")["r"]

    lines = ["# Calibrating to a new person (generated by scripts/calibration.py)", "",
             "Every person is an unseen person (fold models from Phase 3). Calibration chunk: the first 10 / 30 / 60 s "
             "of the person's music record, ~35 min before the test data. Test data: the 'late' part, as in every "
             "other table. Seeded rows are averaged over the 3 seeds per person first.", "",
             "## Means over the 16 development people", "", table.to_markdown(), "",
             f"For scale: the same-person U-Net (trained on ~35 min of each person's own data) has mean r "
             f"{same.mean():.3f}"
             + (f"; the U-Net with the best shift chosen on the test data itself (Phase 4 upper bound) "
                f"{oracle['r_best_lag'].mean():.3f}." if oracle is not None else "."), "",
             "## Spread of r across people", "", spread.to_markdown(), "",
             "## Person-by-person", "", *tests, "",
             "## Shift fitted on the calibration chunk, U-Net (ms, mean over seeds; positive = output was late)", "",
             (shifts.join(oracle["best_lag_ms"].round(0).rename("best on test data")) if oracle is not None else shifts)
             .to_markdown(), "",
             "## Fine-tuning steps: mean r on the tune people (choice = best per fold and length)", "",
             chosen.to_markdown(), "",
             "## Pearson r per person", "", piv.reindex(columns=order).round(3).to_markdown(), ""]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def plot_examples(sp):
    df = pd.read_csv(OUT / "per_person.csv")
    u = df[(df.method == "unet")].groupby("subject")["r"].mean()
    show = [int(s) for s in u.nsmallest(3).index]
    W = 4 * FS
    t = np.arange(W) / FS
    fig, axes = plt.subplots(len(show), 1, figsize=(13, 2.8 * len(show)), sharex=True)
    for ax, s in zip(axes, show):
        k = next(i for i, f in enumerate(sp["folds"]) if s in f["test"])
        m = load_unet(k, 0)
        test = test_segments(s)[0]
        chunk = load_segments([calib_segment(s, 30)], FS)[0]
        d = best_shift(chunk["y"], stitch_predict(m, chunk["x"], LEN, DEV))
        row = df[(df.subject == s) & (df.method == "unet_finetune") & (df.calib_s == 30) & (df.seed == 0)]
        ft = finetune(m, chunk, [int(row.steps.iloc[0])], 0)[int(row.steps.iloc[0])]
        a = len(test["x"]) // 2
        x = test["x"][a - LEN:a + W + LEN]
        y = test["y"][a:a + W]
        p = stitch_predict(m, x, LEN, DEV)
        curves = [("U-Net, no calibration", p[LEN:LEN + W], "#2471a3"),
                  (f"shifted ({d * 1000 / FS:+.0f} ms, from 30 s)", apply_shift(p, d)[LEN:LEN + W], "#d35400"),
                  ("fine-tuned on 30 s", stitch_predict(ft, x, LEN, DEV)[LEN:LEN + W], "#16a085")]
        for j, (lab, c, col) in enumerate(curves):
            ax.plot(t, y - 4 * j, "k", lw=0.7, alpha=0.6)
            ax.plot(t, c - 4 * j, color=col, lw=0.9, label=lab)
        ax.set_title(f"subject {s}; black = real ECG", fontsize=8, loc="left")
        ax.set_yticks([])
        ax.legend(fontsize=7, ncol=3, loc="upper right")
    axes[-1].set_xlabel("seconds (4 s from the middle of the test part, seed 0)")
    fig.tight_layout()
    fig.savefig(OUT / "examples.png", dpi=90)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sp = get_splits()
    run(sp)
    summarize(sp)
    plot_examples(sp)


if __name__ == "__main__":
    main()
