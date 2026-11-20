"""A heartbeat check for the SCG input: withhold the reconstructed ECG when the SCG has no clear heartbeat.

Every model here draws a full-size ECG from noise (results/clinical). Two scores per window:

  periodicity     the strongest autocorrelation peak of the SCG's 5-30 Hz energy envelope at a heart-rate lag
                  (0.4-1.5 s, i.e. 40-150 bpm). A heartbeat repeats every R-R interval; noise and scrambled SCG
                  do not. Uses only the SCG.
  seed_agreement  mean pairwise correlation of three U-Nets trained with different seeds. On real SCG they agree;
                  fed noise, each invents its own beats.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from .data import bandpass

WIN_S = 8.0
LAG_S = (0.4, 1.5)


def envelope(x: np.ndarray, fs: int) -> np.ndarray:
    e = bandpass(x, 5.0, 30.0, fs) ** 2
    return uniform_filter1d(e.astype(np.float64), int(0.05 * fs))


def periodicity(x: np.ndarray, fs: int) -> float:
    """Highest normalised autocorrelation of the envelope at lags of 0.4-1.5 s, for one window."""
    e = envelope(x, fs)
    e = e - e.mean()
    if e.std() < 1e-12:
        return 0.0
    n = len(e)
    f = np.fft.rfft(e, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac = ac / ac[0]
    ac = ac * n / (n - np.arange(n))  # unbiased: shorter overlaps are not penalised
    lo, hi = int(LAG_S[0] * fs), int(LAG_S[1] * fs)
    return float(ac[lo:hi].max())


def windows(n: int, fs: int, win_s: float = WIN_S) -> list[tuple[int, int]]:
    w = int(win_s * fs)
    return [(a, a + w) for a in range(0, n - w + 1, w)]


def seed_agreement(preds: list[np.ndarray]) -> float:
    rs = []
    for i in range(len(preds)):
        for j in range(i + 1, len(preds)):
            a, b = preds[i] - preds[i].mean(), preds[j] - preds[j].mean()
            d = np.sqrt((a * a).sum() * (b * b).sum())
            rs.append(float((a * b).sum() / d) if d > 0 else 0.0)
    return float(np.mean(rs))


def beat_regularity(x: np.ndarray, fs: int) -> float:
    """Share of intervals between detected vibration beats within +-20% of the window's median interval. A real
    rhythm with a few early beats keeps most intervals near the median; fragments of scrambled SCG land at random
    gaps. 0 if fewer than 3 beats."""
    e = envelope(x, fs)
    pk, _ = find_peaks(e, distance=int(0.33 * fs), prominence=0.25 * np.percentile(e, 95))
    if len(pk) < 3:
        return 0.0
    d = np.diff(pk).astype(float)
    return float(np.mean(np.abs(d - np.median(d)) <= 0.2 * np.median(d)))


def beat_consistency(x: np.ndarray, fs: int) -> float:
    """Do the vibration beats in this window look alike, whenever they come? Beats are found as peaks of the
    5-30 Hz envelope (at most 180 bpm); the score is the median correlation of each beat (-100..+400 ms) with the
    window's median beat. Unlike periodicity, it does not penalise uneven beat timing. 0 if fewer than 3 beats."""
    e = envelope(x, fs)
    pk, _ = find_peaks(e, distance=int(0.33 * fs), prominence=0.25 * np.percentile(e, 95))
    a, b = int(0.1 * fs), int(0.4 * fs)
    pk = pk[(pk >= a) & (pk < len(x) - b)]
    if len(pk) < 3:
        return 0.0
    beats = np.stack([x[p - a:p + b] for p in pk]).astype(np.float64)
    beats -= beats.mean(1, keepdims=True)
    med = np.median(beats, 0)
    med -= med.mean()
    num = beats @ med
    den = np.sqrt((beats ** 2).sum(1) * (med ** 2).sum()) + 1e-12
    return float(np.median(num / den))
