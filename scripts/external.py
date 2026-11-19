"""External test on a second public dataset: SensSmartTech (PhysioNet, CC BY 4.0).

32 healthy volunteers, 30 s recordings: ECG lead I (and II, V3, V4) and one accelerometer axis perpendicular to the
chest, placed between the V3 and V4 electrodes (CEBS: a different sensor, placement and lab). Each person has 3+ rest
recordings ('B', before activity) and 6-12 recordings after activity ('A', higher heart rate). No beat annotations:
R-peaks come from neurokit2 on the real lead I.

Plan, fixed before running (committed with this file):
  models       the final CEBS models of the holdout phase (trained on the 16 CEBS development people, seeds 0-2):
               unet, cycle_paired; and the average-heartbeat baseline fitted on the same 16 people.
  no calibration    every recording of every person.
  calibration       from the person's first rest recording (first 10 s, or all 30 s):
                    unet_shift, unet_finetune (100 steps, as in the holdout plan), avg_beat_own.
                    Tested on the person's OTHER rest recordings ('rest') and on all after-activity recordings
                    ('after_activity'), never on the calibration recording.
  scores       r, beat sensitivity / PPV, R timing error, average-beat correlation, per person then averaged.
  diagnostic   unet on the sign-flipped accelerometer (the sensors' polarity may differ); labelled as such, not used
               to choose anything.

Writes results/external/{per_recording.csv, summary.md, examples.png}.
Run from the repo root:  .venv\\Scripts\\python -m scripts.external
"""
from __future__ import annotations

import csv
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wfdb
from scipy.signal import resample_poly

from scg2ecg.baselines import AverageBeat
from scg2ecg.data import ECG_BAND, SCG_BAND, bandpass, zscore
from scg2ecg.metrics import detect_r_peaks, morphology, score
from scg2ecg.models import UNet1d
from scg2ecg.paths import RESULTS, ROOT
from scg2ecg.splits import get_splits
from scg2ecg.windows import normalize_scg, stitch_predict

from .calibration import apply_shift, avg_beat_pieces, best_shift, finetune
from .simple_models import DEV, FS, LEN, SEEDS

SST = ROOT / "data" / "external" / "senssmarttech" / "sst"
OUT = RESULTS / "external"
FT_STEPS = 100
TRIM_S = 2.0  # 0 reproduces the committed plan exactly; see docs/11_external.md
CALIB_S = (10, 26) if TRIM_S else (10, 30)  # 26 s = the whole calibration recording after trimming
METRICS = ["r", "beat_se", "beat_ppv", "r_timing_abs_ms", "template_r", "beat_r", "r_amp_ratio"]


def recordings() -> pd.DataFrame:
    rows = list(csv.reader(open(SST / "Demographics.csv", encoding="utf-8-sig")))[2:]
    out = [{"file": int(r[0]), "subject": int(r[1]), "time": r[2], "ecg": r[8], "acc": r[11], "state": r[12].strip(),
            "hr": float(r[13])} for r in rows if r and r[0].strip()]
    return pd.DataFrame(out).sort_values(["subject", "time"]).reset_index(drop=True)


def load_rec(row) -> dict:
    """At 500 Hz: band-passed accelerometer (raw and normalised), z-scored band-passed lead I, R-peaks."""
    e = wfdb.rdrecord(str(SST / "WFDB" / row["ecg"]))
    a = wfdb.rdrecord(str(SST / "WFDB" / row["acc"]))
    assert e.fs == 1000 and a.fs == 1000 and e.sig_name[0] == "I"
    ecg = resample_poly(e.p_signal[:, 0].astype(np.float64), 1, 2)
    acc = resample_poly(a.p_signal[:, 0].astype(np.float64), 1, 2)
    n = min(len(ecg), len(acc))
    if not TRIM_S:  # as planned (results/external/*_as_planned.*): the filter's edge transient stays in
        y = zscore(bandpass(ecg[:n], *ECG_BAND, FS))
        scg = bandpass(acc[:n], *SCG_BAND, FS)
        return {"x": normalize_scg(scg, FS), "scg": scg, "y": y, "r": detect_r_peaks(y, FS)}
    # corrected: the raw lead I sits on a large DC offset, which zero-phase filtering of a 30 s clip turns into a
    # transient of up to 14 SD at both ends; remove the median first and drop TRIM_S seconds at each end
    k = int(TRIM_S * FS)
    y = bandpass(ecg[:n] - np.median(ecg[:n]), *ECG_BAND, FS)[k:n - k]
    scg = bandpass(acc[:n] - np.median(acc[:n]), *SCG_BAND, FS)
    x = normalize_scg(scg, FS)[k:n - k]
    y = zscore(y)
    return {"x": x, "scg": scg[k:n - k], "y": y, "r": detect_r_peaks(y, FS)}


def load_models(m):
    out = []
    for seed in SEEDS:
        g = UNet1d().to(DEV)
        g.load_state_dict(torch.load(RESULTS / "holdout" / f"{m}_seed{seed}.pt", map_location=DEV))
        out.append(g.eval())
    return out


def scored(base, rec, pred):
    args = ([rec["y"]], [pred], [rec["r"]], FS)
    return {**base, **score(*args), **morphology(*args)}


def run() -> pd.DataFrame:
    recs = recordings()
    sp = get_splits()
    ab = AverageBeat(FS).fit(avg_beat_pieces(sp["dev"], "all"))
    ab.tune(avg_beat_pieces(sp["dev"], "all"))
    unets, cycles = load_models("unet"), load_models("cycle_paired")
    rows = []
    for s, g in recs.groupby("subject"):
        t0 = time.time()
        data = {int(r.file): load_rec(r._asdict()) for r in g.itertuples()}
        rest = g[g.state == "B"].file.tolist()
        calib_file, others = rest[0], rest[1:]
        for r in g.itertuples():
            d = data[r.file]
            part = "calibration_recording" if r.file == calib_file else ("rest" if r.state == "B" else "after_activity")
            base = {"subject": s, "file": r.file, "state": r.state, "part": part, "hr": r.hr}
            rows.append(scored({**base, "method": "avg_beat", "calib_s": 0, "seed": None}, d, ab.predict(d["scg"])[0]))
            for seed, (u, c) in enumerate(zip(unets, cycles)):
                rows.append(scored({**base, "method": "unet", "calib_s": 0, "seed": seed}, d,
                                   stitch_predict(u, d["x"], LEN, DEV)))
                rows.append(scored({**base, "method": "cycle_paired", "calib_s": 0, "seed": seed}, d,
                                   stitch_predict(c, d["x"], LEN, DEV)))
                rows.append(scored({**base, "method": "unet_sign_flipped_input", "calib_s": 0, "seed": seed}, d,
                                   stitch_predict(u, -d["x"], LEN, DEV)))
        cal = data[calib_file]
        tests = [f for f in g.file if f != calib_file]
        for c in CALIB_S:
            n = c * FS
            chunk = {"x": cal["x"][:n], "y": cal["y"][:n]}
            cr = cal["r"][cal["r"] < n]
            own = AverageBeat(FS).fit([(cal["scg"][:n], cal["y"][:n], cr)])
            own.tune([(cal["scg"][:n], cal["y"][:n], cr)])
            for seed, u in enumerate(unets):
                d_shift = best_shift(chunk["y"], stitch_predict(u, chunk["x"], LEN, DEV))
                ft = finetune(u, chunk, [FT_STEPS], seed)[FT_STEPS]
                for f in tests:
                    d = data[f]
                    st = g[g.file == f].iloc[0]
                    base = {"subject": s, "file": f, "state": st.state, "part": "rest" if st.state == "B" else
                            "after_activity", "hr": st.hr, "calib_s": c, "seed": seed}
                    p = stitch_predict(u, d["x"], LEN, DEV)
                    rows.append(scored({**base, "method": "unet_shift", "shift_ms": d_shift * 1000 / FS}, d,
                                       apply_shift(p, d_shift)))
                    rows.append(scored({**base, "method": "unet_finetune"}, d, stitch_predict(ft, d["x"], LEN, DEV)))
                    if seed == 0:
                        rows.append(scored({**base, "method": "avg_beat_own", "seed": None}, d, own.predict(d["scg"])[0]))
        print(f"subject {s}: {len(g)} recordings ({time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "per_recording.csv", index=False)
    return pd.DataFrame(rows)


def summarize():
    df = pd.read_csv(OUT / "per_recording.csv")
    df["label"] = df.method + df.calib_s.map(lambda c: "" if c == 0 else f" {int(c)}s")
    order = ["avg_beat", "unet", "cycle_paired", "unet_sign_flipped_input"] + \
            [f"{m} {c}s" for m in ("unet_shift", "avg_beat_own", "unet_finetune") for c in CALIB_S]
    tabs = []
    for part in ("rest", "after_activity"):
        sub = df[df.part == part]
        pp = sub.groupby(["label", "subject"])[METRICS].mean().reset_index()
        t = pp.groupby("label")[METRICS].mean().reindex(order).round(3)
        t.insert(0, "people", pp.groupby("label").subject.nunique().reindex(order))
        tabs += [f"### {part.replace('_', ' ')} recordings", "", t.to_markdown(), ""]
    pp = df[df.part == "rest"].groupby(["label", "subject"]).r.mean().unstack(0)
    cmp = []
    for a, b in (("unet", "avg_beat"), ("cycle_paired", "unet"), ("unet_finetune 10s", "unet"),
                 ("unet_finetune 10s", "avg_beat_own 10s"),
                 (f"unet_finetune {CALIB_S[-1]}s", f"avg_beat_own {CALIB_S[-1]}s")):
        d = (pp[a] - pp[b]).dropna()
        cmp.append(f"- {a} vs {b}: better for {(d > 0).sum()} of {len(d)} people; mean r difference {d.mean():+.3f}")
    hr = df[(df.part == "after_activity") & (df.label == "unet_finetune 10s")]
    hr_corr = np.corrcoef(hr.hr, hr.r)[0, 1]
    lines = ["# External test: SensSmartTech (generated by scripts/external.py)", "",
             "Models trained on CEBS only (the holdout phase's final models). 32 new people, a different sensor and "
             "placement. Calibration uses the person's first rest recording; scores are on their other rest "
             "recordings and on their after-activity recordings. Per person first, then averaged.", "",
             "## Means over people", "", *tabs,
             "## Person by person (rest recordings)", "", *cmp, "",
             f"After activity, fine-tuned on 10 s: correlation between the recording's heart rate and its r = "
             f"{hr_corr:.2f} ({len(hr)} recording-seeds).", "",
             "unet_sign_flipped_input is a diagnostic (the accelerometer's sign may differ between the two sensors), "
             "not a method; nothing was chosen with it.", ""]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def plot_examples():
    recs = recordings()
    unet = load_models("unet")[0]
    df = pd.read_csv(OUT / "per_recording.csv")
    u = df[(df.method == "unet_finetune") & (df.calib_s == 10) & (df.part == "rest")].groupby("subject").r.mean()
    show = [int(u.idxmax()), int((u - u.median()).abs().idxmin()), int(u.idxmin())]
    fig, axes = plt.subplots(len(show), 1, figsize=(13, 2.8 * len(show)), sharex=True)
    W = 6 * FS
    t = np.arange(W) / FS
    for ax, s in zip(axes, show):
        g = recs[recs.subject == s]
        rest = g[g.state == "B"].file.tolist()
        cal, test = load_rec(g[g.file == rest[0]].iloc[0]), load_rec(g[g.file == rest[1]].iloc[0])
        ft = finetune(unet, {"x": cal["x"][:10 * FS], "y": cal["y"][:10 * FS]}, [FT_STEPS], 0)[FT_STEPS]
        a = 10 * FS
        curves = [("U-Net trained on CEBS, no calibration", stitch_predict(unet, test["x"], LEN, DEV), "#2471a3"),
                  ("fine-tuned on 10 s of another recording", stitch_predict(ft, test["x"], LEN, DEV), "#16a085")]
        for j, (lab, p, col) in enumerate(curves):
            ax.plot(t, test["y"][a:a + W] - 5 * j, "k", lw=0.7, alpha=0.6)
            ax.plot(t, p[a:a + W] - 5 * j, color=col, lw=0.9, label=lab)
        ax.set_title(f"SensSmartTech subject {s}; black = real lead I", fontsize=8, loc="left")
        ax.set_yticks([])
        ax.legend(fontsize=7, ncol=2, loc="upper right")
    axes[-1].set_xlabel("seconds (from the person's second rest recording; seed 0)")
    fig.tight_layout()
    fig.savefig(OUT / "examples.png", dpi=90)
    plt.close(fig)


def main():
    import sys
    OUT.mkdir(parents=True, exist_ok=True)
    if "--summary-only" not in sys.argv:
        run()
    summarize()
    plot_examples()


if __name__ == "__main__":
    main()
