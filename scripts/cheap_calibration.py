"""Follow-up 2: can calibration be cheaper than 10 s of ECG? Development people as unseen people (fold models).

Calibration clip: the start of the person's music record, as in scripts/calibration.py. Test data: the 'late' part.
Everything is fixed in advance: 100 fine-tuning steps (the holdout plan's choice), 3 seeds.

  ecg_finetune Ns       fine-tune on N s of real ECG, N = 3, 5, 10 (10 s repeats Phase 4b with the fixed step count)
  beats_shift Ns        only beat times (R-peaks) from the clip, as a smartwatch ECG could give: shift the U-Net output
                        by the median offset between its own detected R-peaks and the given ones; N = 10, 30
  beats_pseudo Ns       only beat times: fine-tune on a pseudo-target, the fold's population average beat pasted at the
                        given beat times. Teaches the person's timing, not their beat shape; N = 10, 30
  *_jitter10ms          the beat-time methods with Gaussian timing noise (SD 10 ms) added to every given beat

Writes results/cheap_calibration/{per_person.csv, summary.md}.
Run from the repo root:  .venv\\Scripts\\python -m scripts.cheap_calibration
"""
from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon

from scg2ecg.baselines import AverageBeat
from scg2ecg.metrics import detect_r_peaks, match_beats, morphology, score
from scg2ecg.paths import RESULTS
from scg2ecg.splits import get_splits, segments
from scg2ecg.windows import load_segments, stitch_predict

from .calibration import apply_shift, avg_beat_pieces, calib_segment, load_unet
from .simple_models import DEV, FS, LEN, SEEDS

OUT = RESULTS / "cheap_calibration"
STEPS = 100
ECG_S = (3, 5, 10)
BEAT_S = (10, 30)
JITTER_MS = 10.0
METRICS = ["r", "beat_se", "beat_ppv", "r_timing_abs_ms", "template_r", "beat_r", "r_amp_ratio"]


def finetune_on(model, x, y, seed, steps=STEPS, lr=1e-4, batch=16):
    """Fine-tune a copy on random crops of (x, y). Crops are as long as the clip allows (a multiple of 16)."""
    crop = min(LEN, len(x) // 16 * 16)
    m = copy.deepcopy(model).train()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.0)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        a = rng.integers(0, len(x) - crop + 1, batch)
        xb = torch.from_numpy(np.stack([x[i:i + crop] for i in a])[:, None]).to(DEV)
        yb = torch.from_numpy(np.stack([y[i:i + crop] for i in a])[:, None]).to(DEV)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
            loss = F.l1_loss(m(xb).float(), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return m.eval()


def beat_offset(pred, beats) -> int:
    """Median (output R - given R) in samples over matched beats; positive = output late."""
    det = detect_r_peaks(pred, FS)
    ri, dj = match_beats(beats, det, FS, 150.0)
    return int(np.median(det[dj] - beats[ri])) if len(ri) else 0


def run(sp) -> pd.DataFrame:
    path = OUT / "per_person.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {r["subject"] for r in rows}
    for k, fold in enumerate(sp["folds"]):
        ab = AverageBeat(FS).fit(avg_beat_pieces(fold["train"], "all"))
        for s in fold["test"]:
            if s in done:
                continue
            t0 = time.time()
            test = load_segments(segments(s, FS)["late"], FS)
            longest = load_segments([calib_segment(s, max(BEAT_S))], FS)[0]
            rng = np.random.default_rng(s)

            def add(base, preds):
                args = ([t["y"] for t in test], preds, [t["r"] for t in test], FS)
                rows.append({"fold": k, "subject": s, **base, **score(*args), **morphology(*args)})

            for seed in SEEDS:
                m = load_unet(k, seed)
                p_test = [stitch_predict(m, t["x"], LEN, DEV) for t in test]
                add({"method": "unet", "calib_s": 0, "seed": seed}, p_test)
                for c in ECG_S:
                    n = c * FS
                    ft = finetune_on(m, longest["x"][:n], longest["y"][:n], seed)
                    add({"method": "ecg_finetune", "calib_s": c, "seed": seed},
                        [stitch_predict(ft, t["x"], LEN, DEV) for t in test])
                for c in BEAT_S:
                    n = c * FS
                    x = longest["x"][:n]
                    r_true = longest["r"][longest["r"] < n]
                    for jitter in (0.0, JITTER_MS):
                        r = r_true + (np.round(rng.normal(0, jitter * FS / 1000, len(r_true))).astype(int) if jitter else 0)
                        r = np.clip(np.sort(r), 0, n - 1)
                        tag = "_jitter10ms" if jitter else ""
                        d = beat_offset(stitch_predict(m, x, LEN, DEV), r)
                        add({"method": f"beats_shift{tag}", "calib_s": c, "seed": seed, "shift_ms": d * 1000 / FS},
                            [apply_shift(p, d) for p in p_test])
                        pseudo = ab.paste(r, n).astype(np.float32)
                        ft = finetune_on(m, x, pseudo, seed)
                        add({"method": f"beats_pseudo{tag}", "calib_s": c, "seed": seed},
                            [stitch_predict(ft, t["x"], LEN, DEV) for t in test])
            print(f"fold {k} subject {s} done ({(time.time() - t0) / 60:.1f} min)", flush=True)
            pd.DataFrame(rows).to_csv(path, index=False)
    return pd.DataFrame(rows)


def summarize():
    df = pd.read_csv(OUT / "per_person.csv")
    df["label"] = df.method + df.calib_s.map(lambda c: "" if c == 0 else f" {int(c)}s")
    pp = df.groupby(["label", "subject"])[METRICS].mean().reset_index()
    order = ["unet"] + [f"ecg_finetune {c}s" for c in ECG_S] + \
            [f"{m}{j} {c}s" for m in ("beats_shift", "beats_pseudo") for j in ("", "_jitter10ms") for c in BEAT_S]
    table = pp.groupby("label")[METRICS].mean().reindex(order).round(3)
    piv = pp.pivot(index="subject", columns="label", values="r")
    tests = []
    for a, b in [(f"ecg_finetune {c}s", "unet") for c in ECG_S] + \
                [(f"beats_shift {c}s", "unet") for c in BEAT_S] + [(f"beats_pseudo {c}s", "unet") for c in BEAT_S] + \
                [(f"beats_pseudo {c}s", f"beats_shift {c}s") for c in BEAT_S] + \
                [("ecg_finetune 3s", "beats_pseudo 30s"), ("ecg_finetune 3s", "ecg_finetune 10s")]:
        d = (piv[a] - piv[b]).dropna()
        tests.append(f"- {a} vs {b}: better for {(d > 0).sum()} of {len(d)} people; mean r difference {d.mean():+.3f}; "
                     f"Wilcoxon p = {wilcoxon(d).pvalue:.3g}")
    ref = RESULTS / "calibration" / "summary.md"
    lines = ["# Cheaper calibration (generated by scripts/cheap_calibration.py)", "",
             "Development people as unseen people (Phase 3 fold models), 3 seeds, 100 fine-tuning steps throughout. "
             "Calibration clip from the start of the music record; test data = the 'late' part. Per person first.", "",
             "## Means over the 16 development people", "", table.to_markdown(), "",
             "For reference (results/calibration): U-Net shifted using 10 s of ECG waveform 0.725; fine-tuned on 10 s "
             "of ECG with steps chosen on tune people 0.964.", "",
             "## Person by person", "", *tests, "",
             "## Pearson r per person", "", piv.reindex(columns=order).round(3).to_markdown(), ""]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    run(get_splits())
    summarize()


if __name__ == "__main__":
    main()
