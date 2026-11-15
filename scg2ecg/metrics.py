"""Scores: waveform closeness, and beat-level checks on R-peaks found in the output."""
from __future__ import annotations

import numpy as np
import neurokit2 as nk


def pearson(t: np.ndarray, p: np.ndarray) -> float:
    t = np.asarray(t, np.float64) - np.mean(t)
    p = np.asarray(p, np.float64) - np.mean(p)
    d = np.sqrt((t * t).sum() * (p * p).sum())
    return float((t * p).sum() / d) if d > 0 else float("nan")


def rmse(t: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(t, np.float64) - p) ** 2)))


def prd(t: np.ndarray, p: np.ndarray) -> float:
    """Percentage root-mean-square difference (%). Use on a zero-mean target."""
    t = np.asarray(t, np.float64)
    return float(100.0 * np.sqrt(np.sum((t - p) ** 2) / np.sum(t ** 2)))


def prd_legacy(t: np.ndarray, p: np.ndarray) -> float:
    """The 2021 notebook's PRD: the x100 sits inside the square root, so it reads 10x too low."""
    t = np.asarray(t, np.float64)
    return float(np.sqrt(np.sum((t - p) ** 2) / np.sum(t ** 2) * 100))


def detect_r_peaks(ecg: np.ndarray, fs: int) -> np.ndarray:
    """A standard detector (neurokit2), applied identically to real and reconstructed ECG."""
    x = np.asarray(ecg, np.float64)
    if not np.isfinite(x).all() or x.std() < 1e-9:
        return np.zeros(0, np.int64)
    clean = nk.ecg_clean(x, sampling_rate=fs, method="neurokit")
    try:
        _, info = nk.ecg_peaks(clean, sampling_rate=fs, method="neurokit")
    except Exception:  # neurokit raises on signals with no usable peaks
        return np.zeros(0, np.int64)
    return np.asarray(info["ECG_R_Peaks"], dtype=np.int64)


def match_beats(ref: np.ndarray, det: np.ndarray, fs: int, tol_ms: float = 150.0):
    """Greedy one-to-one matching, nearest pairs first, within +-tol. Returns matched (ref_idx, det_idx)."""
    ref = np.asarray(ref, np.int64)
    det = np.asarray(det, np.int64)
    if len(ref) == 0 or len(det) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    tol = tol_ms * fs / 1000.0
    pos = np.searchsorted(ref, det)
    cand = []
    for j, k in enumerate(pos):
        for i in (k - 1, k):
            if 0 <= i < len(ref):
                d = abs(int(det[j]) - int(ref[i]))
                if d <= tol:
                    cand.append((d, i, j))
    cand.sort()
    used_r, used_d, ri, dj = set(), set(), [], []
    for _, i, j in cand:
        if i not in used_r and j not in used_d:
            used_r.add(i)
            used_d.add(j)
            ri.append(i)
            dj.append(j)
    order = np.argsort(ri)
    return np.asarray(ri, np.int64)[order], np.asarray(dj, np.int64)[order]


def beat_counts(ref: np.ndarray, det: np.ndarray, n: int, fs: int, edge_s: float = 1.0, tol_ms: float = 150.0) -> dict:
    """Matched-beat counts and errors for one segment. Beats within `edge_s` of either end are ignored."""
    e = int(edge_s * fs)
    ref = np.asarray(ref)[(np.asarray(ref) >= e) & (np.asarray(ref) < n - e)]
    det = np.asarray(det)[(np.asarray(det) >= e) & (np.asarray(det) < n - e)]
    ri, dj = match_beats(ref, det, fs, tol_ms)
    timing = (det[dj] - ref[ri]) * 1000.0 / fs
    consec = np.diff(ri) == 1
    rr_err = (np.diff(det[dj]) - np.diff(ref[ri]))[consec] * 1000.0 / fs
    return {"tp": len(ri), "fn": len(ref) - len(ri), "fp": len(det) - len(ri), "timing_ms": timing, "rr_err_ms": rr_err}


def summarize_beats(parts: list[dict]) -> dict:
    tp = sum(p["tp"] for p in parts)
    fn = sum(p["fn"] for p in parts)
    fp = sum(p["fp"] for p in parts)
    timing = np.concatenate([p["timing_ms"] for p in parts]) if parts else np.zeros(0)
    rr = np.concatenate([p["rr_err_ms"] for p in parts]) if parts else np.zeros(0)
    return {
        "beat_se": tp / (tp + fn) if tp + fn else float("nan"),
        "beat_ppv": tp / (tp + fp) if tp + fp else float("nan"),
        "r_timing_abs_ms": float(np.median(np.abs(timing))) if len(timing) else float("nan"),
        "rr_mae_ms": float(np.mean(np.abs(rr))) if len(rr) else float("nan"),
    }


def score(targets: list[np.ndarray], preds: list[np.ndarray], refs: list[np.ndarray], fs: int, detect: bool = True) -> dict:
    """Score one person over one or more segments: waveform metrics pooled, beat metrics summed."""
    t = np.concatenate(targets)
    p = np.concatenate(preds)
    out = {"r": pearson(t, p), "rmse": rmse(t, p), "prd": prd(t, p)}
    if detect:
        parts = [beat_counts(r, detect_r_peaks(pr, fs), len(pr), fs) for pr, r in zip(preds, refs)]
        out.update(summarize_beats(parts))
    return out


def morphology(targets: list[np.ndarray], preds: list[np.ndarray], refs: list[np.ndarray], fs: int,
               pre_ms: float = 250.0, post_ms: float = 450.0) -> dict:
    """Beat shape, cut at the database's R-peak marks (so no detector enters).

    template_r      correlation of the output's average beat with the real average beat
    beat_r          median per-beat correlation (each output beat with the real beat at the same time)
    r_amp_ratio     median R amplitude of the output / of the real ECG (R = max within +-30 ms of the mark,
                    minus the beat's median). Below 1: flattened beats, the usual sign of regression to the mean
    """
    a, b = int(pre_ms * fs / 1000), int(post_ms * fs / 1000)
    w = int(30 * fs / 1000)
    T, P = [], []
    for t, p, r in zip(targets, preds, refs):
        r = np.asarray(r)
        r = r[(r >= a) & (r < len(t) - b)]
        T += [t[i - a:i + b] for i in r]
        P += [p[i - a:i + b] for i in r]
    if len(T) < 2:
        return {k: float("nan") for k in ("template_r", "beat_r", "r_amp_ratio")}
    T, P = np.asarray(T, np.float64), np.asarray(P, np.float64)

    def amp(X):
        return np.median(X[:, a - w:a + w + 1].max(1) - np.median(X, 1))

    return {"template_r": pearson(T.mean(0), P.mean(0)),
            "beat_r": float(np.median([pearson(x, y) for x, y in zip(T, P)])),
            "r_amp_ratio": float(amp(P) / amp(T))}
