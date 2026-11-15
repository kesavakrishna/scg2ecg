"""The average-heartbeat baseline: no learning beyond two averages.

1. Average the SCG and the ECG around every annotated R-peak in the training data, giving one
   R-locked SCG beat and one R-locked ECG beat.
2. Find beats in new SCG with one of two detectors, chosen (with its threshold) on tune people:
   * 'ncc': slide the average SCG beat along the signal (normalised cross-correlation, a matched
     filter). The template is R-locked, so each peak position is directly the estimated R time.
   * 'envelope': peaks of the 5-30 Hz SCG energy envelope, shifted back by the training people's
     median envelope-peak-to-R delay. Blind to waveform shape and polarity.
3. Paste the average ECG beat at each detected beat.

If a model cannot beat this, it has learned when the heart beats, not what the ECG looks like.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import correlate, find_peaks
from scipy.signal.windows import tukey

from .data import bandpass
from .metrics import match_beats

GRIDS = {"ncc": np.arange(0.05, 0.85, 0.05), "envelope": np.arange(0.05, 1.05, 0.05)}


def r_locked(x: np.ndarray, r: np.ndarray, pre: int, post: int) -> np.ndarray:
    r = r[(r - pre >= 0) & (r + post <= len(x))]
    idx = r[:, None] + np.arange(-pre, post)[None, :]
    return x[idx]


def ncc(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Pearson correlation of template t with every window of x ('valid' positions)."""
    x = np.asarray(x, np.float64)
    t = np.asarray(t, np.float64)
    L = len(t)
    t0 = t - t.mean()
    t0 /= np.linalg.norm(t0)
    num = correlate(x, t0, mode="valid", method="fft")
    c1 = np.concatenate([[0.0], np.cumsum(x)])
    c2 = np.concatenate([[0.0], np.cumsum(x * x)])
    s1 = c1[L:] - c1[:-L]
    s2 = c2[L:] - c2[:-L]
    ss = np.maximum(s2 - s1 * s1 / L, 1e-12)
    return num / np.sqrt(ss)


class AverageBeat:
    def __init__(self, fs: int, ecg_ms=(-250, 450), scg_ms=(-100, 400), min_rr_s: float = 0.4):
        self.fs = fs
        self.e_pre, self.e_post = int(-ecg_ms[0] * fs / 1000), int(ecg_ms[1] * fs / 1000)
        self.s_pre, self.s_post = int(-scg_ms[0] * fs / 1000), int(scg_ms[1] * fs / 1000)
        self.min_rr = int(min_rr_s * fs)
        self.mode = "ncc"
        self.threshold = 0.3

    def envelope(self, scg: np.ndarray) -> np.ndarray:
        x = bandpass(scg, 5.0, 30.0, self.fs).astype(np.float64)
        L = int(0.1 * self.fs)
        e = np.convolve(x * x, np.ones(L) / L, mode="same")
        return e / (np.percentile(e, 95) + 1e-12)

    def fit(self, pieces: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> "AverageBeat":
        """pieces: (scg, ecg_target, r_peaks) per training segment."""
        e = np.concatenate([r_locked(ecg, r, self.e_pre, self.e_post) for _, ecg, r in pieces])
        s = np.concatenate([r_locked((scg - scg.mean()) / (scg.std() + 1e-12), r, self.s_pre, self.s_post)
                            for scg, _, r in pieces])
        self.ecg_beat = (e.mean(0) * tukey(e.shape[1], 0.2)).astype(np.float32)
        self.scg_beat = s.mean(0).astype(np.float32)
        self.n_beats = len(e)
        w = int(0.3 * self.fs)
        offs = np.concatenate([np.argmax(r_locked(self.envelope(scg), r, 0, w), axis=1) for scg, _, r in pieces])
        self.env_offset = int(np.median(offs))
        return self

    def trace(self, scg: np.ndarray, mode: str) -> tuple[np.ndarray, int]:
        """Detection trace and the shift from a trace peak to the estimated R position."""
        if mode == "ncc":
            return ncc(scg, self.scg_beat), self.s_pre
        return self.envelope(scg), -self.env_offset

    def detect(self, scg: np.ndarray, mode: str | None = None, threshold: float | None = None) -> np.ndarray:
        c, shift = self.trace(scg, mode or self.mode)
        th = self.threshold if threshold is None else threshold
        k, _ = find_peaks(c, height=th, distance=self.min_rr)
        r = k + shift
        return r[(r >= 0) & (r < len(scg))].astype(np.int64)

    def paste(self, r: np.ndarray, n: int) -> np.ndarray:
        out = np.zeros(n + self.e_pre + self.e_post, np.float32)
        for k in r:
            out[k:k + self.e_pre + self.e_post] += self.ecg_beat
        return out[self.e_pre:self.e_pre + n]

    def predict(self, scg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = self.detect(scg)
        return self.paste(r, len(scg)), r

    def tune(self, pieces, modes=("ncc", "envelope"), tol_ms: float = 100.0) -> float:
        """Pick the detector and threshold that maximise beat F1 on the given (non-test) segments."""
        best = (-1.0, self.mode, self.threshold)
        for mode in modes:
            traces = [self.trace(scg, mode) for scg, _, _ in pieces]
            for th in GRIDS[mode]:
                tp = fn = fp = 0
                for (c, shift), (_, _, r) in zip(traces, pieces):
                    k, _ = find_peaks(c, height=th, distance=self.min_rr)
                    det = k + shift
                    ri, _ = match_beats(r, det, self.fs, tol_ms)
                    tp += len(ri)
                    fn += len(r) - len(ri)
                    fp += len(det) - len(ri)
                f1 = 2 * tp / (2 * tp + fn + fp) if tp else 0.0
                if f1 > best[0]:
                    best = (f1, mode, float(th))
        _, self.mode, self.threshold = best
        return best[0]
