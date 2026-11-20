"""Rungs 2 and 3 of the model ladder: ridge regression (as an FIR filter) and a supervised 1D U-Net."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge


# ---------------------------------------------------------------- ridge regression

class RidgeFIR:
    """ECG[t] = w . SCG[t-half : t+half] + b, fitted with sklearn Ridge on a subsample of time points.
    A linear, time-invariant filter: exactly what one nn.Conv1d(1, 1, K) computes."""

    def __init__(self, fs: int, half_ms: int = 500, alpha: float = 1.0):
        self.half = int(half_ms * fs / 1000)
        self.K = 2 * self.half + 1
        self.alpha = alpha

    def design(self, segs: list[dict], every: int):
        X, y = [], []
        for s in segs:
            x = np.pad(s["x"], self.half)
            idx = np.arange(0, len(s["x"]), every)
            X.append(np.lib.stride_tricks.sliding_window_view(x, self.K)[idx])
            y.append(s["y"][idx])
        return np.concatenate(X).astype(np.float32), np.concatenate(y).astype(np.float32)

    def fit(self, segs: list[dict], every: int = 50) -> "RidgeFIR":
        X, y = self.design(segs, every)
        self.n_fit = len(y)
        m = Ridge(alpha=self.alpha).fit(X, y)
        self.w, self.b = m.coef_.astype(np.float32), float(m.intercept_)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        # correlate with w == convolve with reversed w; 'same' keeps alignment for odd K
        return (np.correlate(np.pad(x, self.half), self.w, mode="valid") + self.b).astype(np.float32)


class ConvFIR(nn.Module):
    """The same model as RidgeFIR, as a single PyTorch layer."""

    def __init__(self, K: int):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, K, padding=K // 2)

    def forward(self, x):
        return self.conv(x)


# ---------------------------------------------------------------- U-Net

class Block(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, k, padding=k // 2), nn.GroupNorm(8, cout), nn.GELU(),
            nn.Conv1d(cout, cout, k, padding=k // 2), nn.GroupNorm(8, cout), nn.GELU())

    def forward(self, x):
        return self.net(x)


class UNet1d(nn.Module):
    """Encoder-decoder with skip connections. Input length must be divisible by 2**len(ch)-1.
    Receptive field at the bottleneck covers about 2 s at 500 Hz, enough to see neighbouring beats."""

    def __init__(self, ch=(16, 32, 64, 128, 256), k: int = 9, cin: int = 1):
        super().__init__()
        self.inc = Block(cin, ch[0], k)
        self.down = nn.ModuleList(nn.Conv1d(ch[i], ch[i], 2, stride=2) for i in range(len(ch) - 1))
        self.enc = nn.ModuleList(Block(ch[i], ch[i + 1], k) for i in range(len(ch) - 1))
        self.up = nn.ModuleList(nn.ConvTranspose1d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(len(ch) - 1)))
        self.dec = nn.ModuleList(Block(2 * ch[i], ch[i], k) for i in reversed(range(len(ch) - 1)))
        self.out = nn.Conv1d(ch[0], 1, 1)

    def forward(self, x):
        skips = []
        h = self.inc(x)
        for down, enc in zip(self.down, self.enc):
            skips.append(h)
            h = enc(down(h))
        for up, dec in zip(self.up, self.dec):
            h = up(h)
            h = dec(torch.cat([h, skips.pop()], dim=1))
        return self.out(h)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
