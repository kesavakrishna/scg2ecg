"""Model inputs, a per-person window Dataset, and stitched inference over long segments.

Input normalisation: the band-passed SCG is divided by its own moving SD over 4 s (after removing the
moving mean). It uses only the SCG itself, so no statistics cross people, and it does not create the
window-edge jumps that per-window z-scoring would.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import uniform_filter1d
from torch.utils.data import Dataset

from .data import prepared
from .splits import Segment

NORM_S = 4.0


def normalize_scg(x: np.ndarray, fs: int) -> np.ndarray:
    x = np.asarray(x, np.float64)
    L = int(NORM_S * fs)
    m = uniform_filter1d(x, L, mode="reflect")
    v = uniform_filter1d(x * x, L, mode="reflect") - m * m
    z = (x - m) / np.sqrt(np.maximum(v, 1e-12))
    return np.clip(z, -10, 10).astype(np.float32)


def load_segments(segs: list[Segment], fs: int) -> list[dict]:
    """[{'x': normalised SCG, 'y': ECG target, 'r': R-peaks, 'name', 'start'}] for each segment."""
    out = []
    for name, a, b in segs:
        scg, ecg, r = prepared(name, fs)
        x = normalize_scg(scg, fs)  # normalise the whole record, then cut: no edge effect at the cut
        out.append({"name": name, "start": a, "x": x[a:b], "y": ecg[a:b], "r": r[(r >= a) & (r < b)] - a})
    return out


class WindowDataset(Dataset):
    """Fixed-length windows cut from a list of segments. Each window lies inside one segment, so it comes from
    one person and one split. A random offset per epoch (set_epoch) varies the window grid."""

    def __init__(self, segments: list[dict], length: int, stride: int):
        self.segs, self.length, self.stride = segments, length, stride
        self.set_epoch(0)

    def set_epoch(self, epoch: int):
        rng = np.random.default_rng(epoch)
        self.index = []
        for i, s in enumerate(self.segs):
            n = len(s["x"])
            if n < self.length:
                continue
            off = int(rng.integers(0, self.stride))
            self.index += [(i, a) for a in range(off, n - self.length + 1, self.stride)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i, a = self.index[k]
        s = self.segs[i]
        return (torch.from_numpy(s["x"][a:a + self.length])[None],
                torch.from_numpy(s["y"][a:a + self.length])[None])


@torch.no_grad()
def stitch_predict(model: torch.nn.Module, x: np.ndarray, length: int, device: str, batch: int = 64,
                   hop: int | None = None) -> np.ndarray:
    """Predict a long signal with overlapping windows (50% by default) blended with Hann weights, so the
    weak window edges get little weight."""
    hop = hop or length // 2
    n = len(x)
    if n < length:
        xp = np.pad(x, (0, length - n))
        return stitch_predict(model, xp, length, device, batch, hop)[:n]
    starts = list(range(0, n - length + 1, hop))
    if starts[-1] != n - length:
        starts.append(n - length)
    w = np.hanning(length).astype(np.float32) + 1e-3
    out = np.zeros(n, np.float32)
    wsum = np.zeros(n, np.float32)
    model.eval()
    for i in range(0, len(starts), batch):
        ss = starts[i:i + batch]
        xb = torch.from_numpy(np.stack([x[a:a + length] for a in ss])[:, None]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            yb = model(xb)
        yb = yb.float().cpu().numpy()[:, 0]
        for a, y in zip(ss, yb):
            out[a:a + length] += w * y
            wsum[a:a + length] += w
    return out / wsum
