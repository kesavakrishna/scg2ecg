"""A1: does a calibration at rest survive a faster heart? SensSmartTech (corrected loader, scripts/external.py).

Calibrated at rest, the U-Net reaches r 0.86 at rest but ~0.5 after exercise (docs/11_external.md). Per person, from
the CEBS final U-Net (3 seeds, 100 fine-tuning steps), four calibrations:
  rest_10s     10 s of the first rest recording (the external test's method)
  rest_20s     20 s of the first rest recording (control: more data, same heart rate)
  rest_fast    10 s of the first rest recording + 10 s of the first after-exercise recording (highest heart rate)
  rest_warp    10 s of rest + the same 10 s time-warped to 1.2x, 1.4x, 1.6x the heart rate (no extra recording).
               Per beat: the first 100 ms after R kept, the systolic part up to an estimated QT end
               (0.42 s x sqrt(RR)) shortened by sqrt(f), the rest compressed to fill the new R-R interval.
               The same warp is applied to SCG and ECG.
Tested on the same recordings for every method: the person's other rest recordings and their after-exercise
recordings except the first one (used by rest_fast). Writes results/heart_rate/{per_recording.csv, summary.md}.
Run from the repo root:  .venv\\Scripts\\python -m scripts.heart_rate
"""
from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon

from scg2ecg.metrics import morphology, score
from scg2ecg.paths import RESULTS
from scg2ecg.windows import stitch_predict

from .external import load_models, load_rec, recordings
from .simple_models import DEV, FS, LEN, SEEDS

OUT = RESULTS / "heart_rate"
STEPS = 100
CLIP_S = 10
WARPS = (1.2, 1.4, 1.6)
METRICS = ["r", "beat_se", "beat_ppv", "r_timing_abs_ms", "template_r", "beat_r"]


def warp_beats(x: np.ndarray, y: np.ndarray, r: np.ndarray, f: float):
    """Speed the heart up by factor f, beat by beat (see module docstring). Returns warped (x, y)."""
    xs, ys = [], []
    keep = int(0.1 * FS)
    for a, b in zip(r[:-1], r[1:]):
        L = b - a
        qt = min(int(0.42 * np.sqrt(L / FS) * FS), L - 1)
        k1 = min(keep, qt)
        new_L = max(int(round(L / f)), k1 + 2)
        new_qt = min(k1 + int(round((qt - k1) / np.sqrt(f))), new_L - 1)
        # new sample positions -> old positions, piecewise linear through (0,0), (k1,k1), (new_qt,qt), (new_L,L)
        t = np.arange(new_L)
        old = np.interp(t, [0, k1, new_qt, new_L], [0, k1, qt, L])
        idx = np.arange(a, b + 1)
        xs.append(np.interp(a + old, idx, x[a:b + 1]))
        ys.append(np.interp(a + old, idx, y[a:b + 1]))
    return np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.float32)


def finetune_clips(model, clips, seed, steps=STEPS, lr=1e-4, batch=16):
    """Fine-tune a copy on random crops drawn evenly from several (x, y) clips."""
    crop = min(LEN, min(len(x) for x, _ in clips) // 16 * 16)
    m = copy.deepcopy(model).train()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.0)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        xb, yb = [], []
        for _ in range(batch):
            x, y = clips[rng.integers(len(clips))]
            a = rng.integers(0, len(x) - crop + 1)
            xb.append(x[a:a + crop])
            yb.append(y[a:a + crop])
        xb = torch.from_numpy(np.stack(xb)[:, None]).to(DEV)
        yb = torch.from_numpy(np.stack(yb)[:, None]).to(DEV)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
            loss = F.l1_loss(m(xb).float(), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return m.eval()


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "per_recording.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {r["subject"] for r in rows}
    recs = recordings()
    unets = load_models("unet")
    for s, g in recs.groupby("subject"):
        rest, fast = g[g.state == "B"].file.tolist(), g[g.state == "A"].file.tolist()
        if s in done or len(rest) < 2 or len(fast) < 2:
            continue
        t0 = time.time()
        data = {int(r.file): load_rec(r._asdict()) for r in g.itertuples()}
        n = CLIP_S * FS
        rest_rec, fast_rec = data[rest[0]], data[fast[0]]
        rc = (rest_rec["x"][:n], rest_rec["y"][:n])
        rr = rest_rec["r"][rest_rec["r"] < n]
        calibs = {"rest_10s": [rc],
                  "rest_20s": [(rest_rec["x"][:2 * n], rest_rec["y"][:2 * n])],
                  "rest_fast": [rc, (fast_rec["x"][:n], fast_rec["y"][:n])],
                  "rest_warp": [rc] + [warp_beats(rc[0], rc[1], rr, f) for f in WARPS]}
        tests = [f for f in rest[1:] + fast[1:]]
        for seed, u in enumerate(unets):
            for name, clips in calibs.items():
                ft = finetune_clips(u, clips, seed)
                for f in tests:
                    d = data[f]
                    st = g[g.file == f].iloc[0]
                    args = ([d["y"]], [stitch_predict(ft, d["x"], LEN, DEV)], [d["r"]], FS)
                    rows.append({"subject": s, "file": f, "state": st.state, "hr": st.hr, "method": name, "seed": seed,
                                 "calib_hr_rest": g[g.file == rest[0]].hr.iloc[0],
                                 "calib_hr_fast": g[g.file == fast[0]].hr.iloc[0], **score(*args), **morphology(*args)})
        print(f"subject {s}: {len(tests)} test recordings ({time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(path, index=False)


def summarize():
    df = pd.read_csv(OUT / "per_recording.csv")
    df["hr_gain"] = df.hr / df.calib_hr_rest
    order = ["rest_10s", "rest_20s", "rest_fast", "rest_warp"]
    lines = ["# Calibrating across heart rates (generated by scripts/heart_rate.py)", "",
             "SensSmartTech, CEBS final U-Net, 3 seeds, 100 fine-tuning steps. Same test recordings for every method. "
             "Per person first, then averaged.", ""]
    for state, name in (("B", "rest recordings"), ("A", "after-exercise recordings")):
        sub = df[df.state == state]
        pp = sub.groupby(["method", "subject"])[METRICS].mean().reset_index()
        t = pp.groupby("method")[METRICS].mean().reindex(order).round(3)
        t.insert(0, "people", pp.groupby("method").subject.nunique().reindex(order))
        lines += [f"## {name}", "", t.to_markdown(), ""]
        piv = pp.pivot(index="subject", columns="method", values="r")
        for a in order[1:]:
            d = (piv[a] - piv["rest_10s"]).dropna()
            lines.append(f"- {a} vs rest_10s: better for {(d > 0).sum()} of {len(d)} people; mean r difference "
                         f"{d.mean():+.3f}; Wilcoxon p = {wilcoxon(d).pvalue:.3g}")
        lines.append("")
    a = df[df.state == "A"].copy()
    a["band"] = pd.cut(a.hr_gain, [0, 1.15, 1.3, 1.5, 3], labels=["<1.15x", "1.15-1.3x", "1.3-1.5x", ">1.5x"])
    band = a.groupby(["band", "method"], observed=True).r.mean().unstack().reindex(columns=order).round(3)
    band.insert(0, "recordings", a.groupby("band", observed=True).file.nunique())
    lines += ["## After exercise, by heart rate relative to the rest calibration", "",
              "r averaged over recording-seeds in each band (heart rate of the test recording / of the rest recording "
              "used for calibration).", "", band.to_markdown(), ""]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    run()
    summarize()


if __name__ == "__main__":
    main()
