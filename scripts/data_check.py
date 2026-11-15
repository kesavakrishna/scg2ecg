"""Phase 2, the data check (no models).

Writes to results/data_check/:
  records.csv    every record: length, checksum status, duplicate status, whether it is used
  subjects.csv   development people only: minutes, beats, heart rate, irregular beats, R-to-SCG gap,
                 and how similar each person's average beats are to everyone else's
  plots/         per development person: 10 random 4 s windows and the 3 worst-looking ones
  summary.md     the numbers above in prose-ready form

The 4 holdout people get only the file-integrity checks: no beats, no heart rate, no plots.
Run from the repo root:  .venv\\Scripts\\python -m scripts.data_check
"""
from __future__ import annotations

import hashlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb
from scipy.ndimage import median_filter

from scg2ecg.baselines import r_locked
from scg2ecg.data import (CHANNELS, FS_RAW, SUBJECTS, bandpass, duplicate_of, listed_hashes, load,
                          prepared, record_name, usable_records)
from scg2ecg.metrics import pearson
from scg2ecg.paths import RAW, RESULTS
from scg2ecg.splits import get_splits

FS = 500
OUT = RESULTS / "data_check"
WIN_S = 4.0
SCG_MS = (-100, 400)
ECG_MS = (-250, 450)


def ms(x):
    return int(round(x * FS / 1000))


def check_files(holdout: set[int]) -> pd.DataFrame:
    listed = listed_hashes()
    rows = []
    for s in SUBJECTS:
        for kind in "bmp":
            name = record_name(kind, s)
            hdr = wfdb.rdheader(str(RAW / name))
            ok = all(hashlib.sha256((RAW / f"{name}.{ext}").read_bytes()).hexdigest() == listed[f"{name}.{ext}"]
                     for ext in ("dat", "hea", "atr"))
            rows.append({
                "record": name, "subject": s, "kind": kind,
                "minutes": round(hdr.sig_len / hdr.fs / 60, 2),
                "fs": hdr.fs, "channels": "/".join(hdr.sig_name),
                "format_ok": hdr.fs == FS_RAW and list(hdr.sig_name) == list(CHANNELS),
                "checksum_ok": ok,
                "duplicate_of": duplicate_of(name) or "",
                "used": name in usable_records(s),
                "holdout": s in holdout,
            })
    return pd.DataFrame(rows)


def person_pieces(s: int):
    return [prepared(r, FS) for r in usable_records(s)]


def paste(beat: np.ndarray, r: np.ndarray, pre: int, n: int) -> np.ndarray:
    out = np.zeros(n + len(beat), np.float32)
    for k in r:
        if 0 <= k - pre and k - pre + len(beat) <= len(out):
            out[k - pre:k - pre + len(beat)] += beat
    return out[:n]


def describe_person(s: int) -> tuple[dict, dict]:
    pieces = person_pieces(s)
    minutes = sum(len(e) for _, e, _ in pieces) / FS / 60
    beats = sum(len(r) for _, _, r in pieces)

    rr = np.concatenate([np.diff(r) / FS for _, _, r in pieces])
    rr_ok = rr[(rr > 0.3) & (rr < 2.0)]
    hr = 60.0 / rr_ok
    local = median_filter(rr_ok, size=11, mode="nearest")
    irregular = np.abs(rr_ok - local) > 0.2 * local

    # R-locked averages (SCG z-scored per record, so people with a stronger signal don't dominate)
    s_pre, s_post = ms(-SCG_MS[0]), ms(SCG_MS[1])
    e_pre, e_post = ms(-ECG_MS[0]), ms(ECG_MS[1])
    scg_beats = np.concatenate([r_locked((x - x.mean()) / x.std(), r, s_pre, s_post) for x, _, r in pieces])
    ecg_beats = np.concatenate([r_locked(e, r, e_pre, e_post) for _, e, r in pieces])
    scg_avg, ecg_avg = scg_beats.mean(0), ecg_beats.mean(0)

    # main SCG peak of the average beat within 0-250 ms after R, and per-beat timing around it
    lo, hi = s_pre, s_pre + ms(250)
    k = lo + int(np.argmax(np.abs(scg_avg[lo:hi])))
    sign = float(np.sign(scg_avg[k]))
    w = ms(40)
    per_beat = (np.argmax(sign * scg_beats[:, k - w:k + w + 1], axis=1) + k - w - s_pre) * 1000.0 / FS
    gap = (k - s_pre) * 1000.0 / FS

    row = {
        "subject": s, "records": "+".join(usable_records(s)), "minutes": round(minutes, 1), "beats": beats,
        "hr_median": round(float(np.median(hr)), 1), "hr_p5": round(float(np.percentile(hr, 5)), 1),
        "hr_p95": round(float(np.percentile(hr, 95)), 1),
        "irregular_pct": round(100 * float(irregular.mean()), 2),
        "gap_ms": gap, "gap_beat_median_ms": float(np.median(per_beat)),
        "gap_beat_iqr_ms": float(np.percentile(per_beat, 75) - np.percentile(per_beat, 25)),
        "scg_peak_sign": int(sign), "scg_peak_amp": float(scg_avg[k]),
    }
    return row, {"scg_avg": scg_avg, "ecg_avg": ecg_avg}


def plot_person(s: int, avgs: dict, rng: np.random.Generator) -> list[dict]:
    """10 random 4 s windows + the 3 windows the person's own average beat explains worst."""
    s_pre, e_pre = ms(-SCG_MS[0]), ms(-ECG_MS[0])
    W = int(WIN_S * FS)
    windows = []
    for name in usable_records(s):
        scg, ecg, r = prepared(name, FS)
        scg_z = (scg - scg.mean()) / scg.std()
        scg_hat = paste(avgs["scg_avg"].astype(np.float32), r, s_pre, len(scg))
        ecg_hat = paste(avgs["ecg_avg"].astype(np.float32), r, e_pre, len(ecg))
        for a in range(0, len(ecg) - W + 1, W):
            b = a + W
            rs, re = pearson(scg_z[a:b], scg_hat[a:b]), pearson(ecg[a:b], ecg_hat[a:b])
            windows.append({"record": name, "start": a, "r_scg": rs, "r_ecg": re,
                            "ugly": 1 - min(np.nan_to_num(rs), np.nan_to_num(re))})
    df = pd.DataFrame(windows)
    pick = list(rng.choice(len(df), size=10, replace=False)) + list(df["ugly"].nlargest(3).index)
    fig, axes = plt.subplots(len(pick), 1, figsize=(13, 2.0 * len(pick)), sharex=True)
    t = np.arange(W) / FS
    for ax, (j, i) in zip(axes, enumerate(pick)):
        w = df.loc[i]
        scg, ecg, r = prepared(w["record"], FS)
        a = int(w["start"])
        e = ecg[a:a + W]
        g = scg[a:a + W]
        g = (g - g.mean()) / (g.std() + 1e-9)
        ax.plot(t, e + 3, lw=0.8, color="#1f4e79", label="ECG lead I")
        ax.plot(t, g, lw=0.8, color="#b5541c", label="SCG")
        rr = r[(r >= a) & (r < a + W)] - a
        ax.plot(rr / FS, e[rr] + 3, "kv", ms=4)
        tag = "random" if j < 10 else "WORST"
        ax.set_title(f"{tag}  {w['record']}  t={a / FS / 60:.2f} min   r(SCG, own avg beat)={w['r_scg']:.2f}  "
                     f"r(ECG, own avg beat)={w['r_ecg']:.2f}", fontsize=8, loc="left")
        ax.set_yticks([])
    axes[0].legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("seconds")
    fig.suptitle(f"Subject {s}: ECG (top, R-marks from .atr) and SCG (bottom), z-scored", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "plots" / f"subject_{s:02d}.png", dpi=80)
    plt.close(fig)
    return [{"subject": s, **df.loc[i].to_dict()} for i in df["ugly"].nlargest(3).index]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plots").mkdir(exist_ok=True)
    sp = get_splits()
    holdout = set(sp["holdout"])

    rec = check_files(holdout)
    rec.to_csv(OUT / "records.csv", index=False)
    print(rec.groupby("kind")[["minutes"]].describe().round(2))
    print("checksums ok:", rec["checksum_ok"].all(), " format ok:", rec["format_ok"].all(),
          " duplicates:", (rec["duplicate_of"] != "").sum())

    rng = np.random.default_rng(0)
    rows, avgs, worst = [], {}, []
    for s in sp["dev"]:
        row, a = describe_person(s)
        rows.append(row)
        avgs[s] = a
        worst += plot_person(s, a, rng)
        print(f"subject {s:2d}: {row['minutes']:.1f} min, {row['beats']} beats, HR {row['hr_median']} "
              f"({row['hr_p5']}-{row['hr_p95']}), gap {row['gap_ms']:.0f} ms, irregular {row['irregular_pct']}%")
    sub = pd.DataFrame(rows)

    # how person-specific are the average beats? correlate each person's with the mean of the others
    for key, col in (("scg_avg", "scg_beat_r_vs_others"), ("ecg_avg", "ecg_beat_r_vs_others")):
        vals = []
        for s in sp["dev"]:
            others = np.mean([avgs[o][key] for o in sp["dev"] if o != s], axis=0)
            vals.append(pearson(avgs[s][key], others))
        sub[col] = np.round(vals, 3)
    sub.to_csv(OUT / "subjects.csv", index=False)
    pd.DataFrame(worst).to_csv(OUT / "worst_windows.csv", index=False)

    # average-beat figure: every development person's R-locked ECG and SCG
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    te = (np.arange(len(avgs[sp["dev"][0]]["ecg_avg"])) - ms(-ECG_MS[0])) * 1000 / FS
    ts = (np.arange(len(avgs[sp["dev"][0]]["scg_avg"])) - ms(-SCG_MS[0])) * 1000 / FS
    for s in sp["dev"]:
        a1.plot(te, avgs[s]["ecg_avg"], lw=0.8, label=str(s))
        a2.plot(ts, avgs[s]["scg_avg"], lw=0.8)
    a1.set_title("Average lead I beat per person (z-scored)")
    a2.set_title("Average SCG beat per person (z-scored)")
    for a in (a1, a2):
        a.axvline(0, color="k", lw=0.5)
        a.set_xlabel("ms from R-peak")
    a1.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "average_beats.png", dpi=100)
    plt.close(fig)

    g = sub["gap_ms"]
    lines = [
        "# Data check (generated by scripts/data_check.py)", "",
        f"Holdout (locked, integrity checks only): subjects {sp['holdout']}. Development: {sp['dev']}.", "",
        "## Files", "",
        f"- {len(rec)} records, checksums all match SHA256SUMS.txt: {bool(rec['checksum_ok'].all())}; "
        f"all 4 channels at 5 kHz: {bool(rec['format_ok'].all())}.",
        f"- Rest records that are byte-identical copies of the after-music record: "
        f"{', '.join(rec.loc[rec['duplicate_of'] != '', 'record'])} "
        f"({(rec['duplicate_of'] != '').sum()} of 20). Only subject 1 has a distinct rest record.",
        f"- Used: {int(rec['used'].sum())} records "
        f"({rec.loc[rec['used'], 'minutes'].sum() / 60:.1f} h); "
        f"development people: {sub['minutes'].sum() / 60:.1f} h, {int(sub['beats'].sum())} beats.", "",
        "## People (development only)", "",
        sub.to_markdown(index=False), "",
        "## R-peak to main SCG peak", "",
        f"- Per-person gap: median {g.median():.0f} ms, range {g.min():.0f}-{g.max():.0f} ms, SD {g.std():.0f} ms.",
        f"- Within a person, beat-to-beat IQR: median {sub['gap_beat_iqr_ms'].median():.0f} ms "
        f"(range {sub['gap_beat_iqr_ms'].min():.0f}-{sub['gap_beat_iqr_ms'].max():.0f}).",
        f"- SCG main-peak sign: {dict(sub['scg_peak_sign'].value_counts())}.",
        f"- Each person's average SCG beat vs the others' mean: r median {sub['scg_beat_r_vs_others'].median():.2f} "
        f"(min {sub['scg_beat_r_vs_others'].min():.2f}); average ECG beat: median "
        f"{sub['ecg_beat_r_vs_others'].median():.2f} (min {sub['ecg_beat_r_vs_others'].min():.2f}).", "",
        "Plots: plots/subject_XX.png (10 random 4 s windows + 3 worst), average_beats.png.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-12:]))


if __name__ == "__main__":
    main()
