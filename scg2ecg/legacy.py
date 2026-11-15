"""A faithful PyTorch port of the 2021 Keras notebook (b_to_b_SCG_to_ECG_signal_1_without_bandpass_filter_DEC21).

Ported as written, quirks included, so the original number can be reproduced:
  * data: one person's rest EDF; SCG (channel 4) and lead I; each z-scored over the whole record, then
    FFT-resampled 5 kHz -> 1 kHz with no filter; non-overlapping 1 s windows; first 75% train, last 25% test
  * generator: Conv1D(30, k=30) -> attention gate (Keras MultiHeadAttention, 2 heads, key_dim=1000, applied
    as LayerNorm(x * attn), then clipped to [0.8, 1] and multiplied in) -> Conv1D 17/13/13 (k=30) -> Conv1D(1, k=1);
    every conv is followed by LeakyReLU(0.2) and keras_contrib InstanceNormalization(axis=None), which
    normalises over time and channels with a scalar gain and bias (including after the output conv)
  * discriminator: four Conv1D(k=13) layers, the same activation + instance norm, including on its 1-channel output
  * losses (Keras 'huber_loss', delta 1): adversarial x1 for each generator; direct supervised terms
    G_AB(scg) vs ecg and G_BA(ecg) vs scg x4; "cycle" terms x0.04 whose targets are the model's own
    reconstructions computed just before the step; the identity outputs are built but unused
  * one Adam (lr 1e-3, Keras update rule, eps 1e-7) shared by the two discriminators and the combined model,
    so its step counter advances 5 times per batch; 1 epoch, batch size 1, windows in time order
  * variant '2021' has an untrained keras Normalization layer after the attention (an identity); the 2023
    re-run replaced it with LayerNormalization(axis=1), i.e. normalising over time with per-sample gains
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample

from .paths import LEGACY_EDF


# ---------------------------------------------------------------- data

def read_edf(path) -> np.ndarray:
    """Minimal EDF reader (16-bit, equal-rate channels) returning physical values, shape (n, channels)."""
    with open(path, "rb") as f:
        h = f.read(256)
        n_rec, ns = int(h[236:244]), int(h[252:256])
        sh = f.read(256 * ns)

        def field(off, width):
            return [sh[off + i * width: off + (i + 1) * width].decode().strip() for i in range(ns)]

        o = 16 * ns + 80 * ns + 8 * ns
        pmin, pmax = map(float, field(o, 8)), map(float, field(o + 8 * ns, 8))
        dmin, dmax = map(float, field(o + 16 * ns, 8)), map(float, field(o + 24 * ns, 8))
        pmin, pmax, dmin, dmax = list(pmin), list(pmax), list(dmin), list(dmax)
        nsamp = list(map(int, field(o + 32 * ns + 80 * ns, 8)))
        raw = np.frombuffer(f.read(), dtype="<i2")
    rec_len = sum(nsamp)
    raw = raw[: n_rec * rec_len].reshape(n_rec, rec_len)
    out, start = [], 0
    for i in range(ns):
        d = raw[:, start:start + nsamp[i]].reshape(-1).astype(np.float64)
        start += nsamp[i]
        out.append((d - dmin[i]) * (pmax[i] - pmin[i]) / (dmax[i] - dmin[i]) + pmin[i])
    return np.stack(out, 1)


def legacy_data(subject: int):
    """Windows exactly as the notebook built them, plus the continuous 1 kHz signals.

    Returns dict with X_train, Y_train, X_test, Y_test as (N, 1, 1000) float32, and scg/ecg (1 kHz),
    n_train (windows), n_windows.
    """
    x = read_edf(LEGACY_EDF / f"b{subject:03d}.edf")
    scg, ecg = x[:, 3], x[:, 0]
    scg = (scg - scg.mean()) / scg.std()   # sklearn.preprocessing.scale(axis=1)
    ecg = (ecg - ecg.mean()) / ecg.std()
    n = int(len(ecg) / 5)
    ecg = resample(ecg, n)
    scg = resample(scg, n)
    starts = list(range(0, n - 1000, 1000))
    Xw = np.stack([scg[i:i + 1000] for i in starts])[:, None, :].astype(np.float32)
    Yw = np.stack([ecg[i:i + 1000] for i in starts])[:, None, :].astype(np.float32)
    n_train = int(math.floor(0.75 * len(starts)))  # train_test_split(train_size=0.75, shuffle=False)
    return {"X_train": Xw[:n_train], "Y_train": Yw[:n_train], "X_test": Xw[n_train:], "Y_test": Yw[n_train:],
            "scg": scg.astype(np.float32), "ecg": ecg.astype(np.float32),
            "n_train": n_train, "n_windows": len(starts)}


# ---------------------------------------------------------------- layers

def glorot_(t: torch.Tensor, fan_in: int, fan_out: int):
    lim = math.sqrt(6.0 / (fan_in + fan_out))
    with torch.no_grad():
        t.uniform_(-lim, lim)


class InstNormAll(nn.Module):
    """keras_contrib InstanceNormalization(axis=None): per sample over (channels, time); scalar gain and bias."""

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.eps = eps

    def forward(self, x):
        m = x.mean(dim=(1, 2), keepdim=True)
        s = x.std(dim=(1, 2), keepdim=True, unbiased=False) + self.eps
        return (x - m) / s * self.gamma + self.beta


class ConvIN(nn.Module):
    """Conv1D(padding='same', LeakyReLU 0.2) followed by InstNormAll, as conv1DWithSINE / d_layer."""

    def __init__(self, cin: int, cout: int, k: int):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, k, padding="same")
        glorot_(self.conv.weight, cin * k, cout * k)
        nn.init.zeros_(self.conv.bias)
        self.norm = InstNormAll()

    def forward(self, x):
        return self.norm(F.leaky_relu(self.conv(x), 0.2))


class KerasMHA(nn.Module):
    """tf.keras MultiHeadAttention(num_heads, key_dim) for self-attention, with Keras's initialisation."""

    def __init__(self, d: int, heads: int, key_dim: int):
        super().__init__()
        self.h, self.dk = heads, key_dim
        self.q, self.k, self.v = (nn.Linear(d, heads * key_dim) for _ in range(3))
        self.o = nn.Linear(heads * key_dim, d)
        for lin in (self.q, self.k, self.v):  # kernel (d, heads, key_dim)
            glorot_(lin.weight, heads * d, key_dim * d)
            nn.init.zeros_(lin.bias)
        glorot_(self.o.weight, key_dim * heads, d * heads)  # kernel (heads, key_dim, d)
        nn.init.zeros_(self.o.bias)

    def forward(self, x):  # (B, T, d)
        B, T, _ = x.shape
        q, k, v = (f(x).view(B, T, self.h, self.dk).transpose(1, 2) for f in (self.q, self.k, self.v))
        # explicit softmax(q k^T / sqrt(dk)) v: ~3x faster than the fused kernel for a 1000-wide head
        a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.dk), dim=-1) @ v
        return self.o(a.transpose(1, 2).reshape(B, T, self.h * self.dk))


class TransformerBlock(nn.Module):
    def __init__(self, d: int, heads: int, key_dim: int, rate: float = 0.1):
        super().__init__()
        self.att = KerasMHA(d, heads, key_dim)
        self.drop = nn.Dropout(rate)
        self.ln = nn.LayerNorm(d, eps=1e-4)

    def forward(self, x):  # (B, T, d)
        return self.ln(x * self.drop(self.att(x)))  # note: multiply, not add


class LayerNormTime(nn.Module):
    """Keras LayerNormalization(axis=1) on (B, T, C): normalise over time, gain/bias per time step."""

    def __init__(self, T: int, eps: float = 1e-3):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(T))
        self.beta = nn.Parameter(torch.zeros(T))
        self.eps = eps

    def forward(self, x):
        m = x.mean(dim=1, keepdim=True)
        v = x.var(dim=1, keepdim=True, unbiased=False)
        return (x - m) / torch.sqrt(v + self.eps) * self.gamma[None, :, None] + self.beta[None, :, None]


class Generator(nn.Module):
    def __init__(self, T: int = 1000, variant: str = "2021"):
        super().__init__()
        self.c0 = ConvIN(1, 30, 30)
        self.tb = TransformerBlock(30, 2, T)
        self.post = nn.Identity() if variant == "2021" else LayerNormTime(T)
        self.body = nn.Sequential(ConvIN(30, 17, 30), ConvIN(17, 13, 30), ConvIN(13, 13, 30), ConvIN(13, 1, 1))

    def forward(self, x):  # (B, 1, T)
        v = self.c0(x)
        a = self.post(self.tb(v.transpose(1, 2))).transpose(1, 2)
        return self.body(v * a.clamp(0.8, 1.0))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(ConvIN(1, 13, 13), ConvIN(13, 13, 13), ConvIN(13, 13, 13), ConvIN(13, 1, 13))

    def forward(self, x):
        return self.net(x)


class KerasAdam:
    """tf.keras Adam update (lr_t = lr*sqrt(1-b2^t)/(1-b1^t); var -= lr_t*m/(sqrt(v)+eps)) with one step
    counter shared by every model that uses this instance, as when one Keras optimizer is compiled into several."""

    def __init__(self, lr=1e-3, b1=0.9, b2=0.999, eps=1e-7):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.t = 0
        self.state: dict[nn.Parameter, tuple[torch.Tensor, torch.Tensor]] = {}

    @torch.no_grad()
    def step(self, params):
        self.t += 1
        lr_t = self.lr * math.sqrt(1 - self.b2 ** self.t) / (1 - self.b1 ** self.t)
        ps = [p for p in params if p.grad is not None]
        for p in ps:
            if p not in self.state:
                self.state[p] = (torch.zeros_like(p), torch.zeros_like(p))
        gs = [p.grad for p in ps]
        ms = [self.state[p][0] for p in ps]
        vs = [self.state[p][1] for p in ps]
        torch._foreach_mul_(ms, self.b1)
        torch._foreach_add_(ms, gs, alpha=1 - self.b1)
        torch._foreach_mul_(vs, self.b2)
        torch._foreach_addcmul_(vs, gs, gs, value=1 - self.b2)
        den = torch._foreach_sqrt(vs)
        torch._foreach_add_(den, self.eps)
        torch._foreach_addcdiv_(ps, ms, den, value=-lr_t)


# ---------------------------------------------------------------- training

def _zero(*models):
    for m in models:
        for p in m.parameters():
            p.grad = None


def train_gan(X: np.ndarray, Y: np.ndarray, seed: int, variant: str = "2021", device: str = "cuda",
              lambda_cycle: float = 4.0, lambda_id: float = 0.04) -> tuple[Generator, dict]:
    """The notebook's CycleGAN.train(): returns G_AB (SCG -> ECG) and a log of D and G losses."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    T = X.shape[-1]
    d_A, d_B = Discriminator().to(device), Discriminator().to(device)  # built first, as in the notebook
    g_AB, g_BA = Generator(T, variant).to(device), Generator(T, variant).to(device)
    opt = KerasAdam()
    mse, huber = nn.MSELoss(), nn.HuberLoss(delta=1.0)
    log = {"d_loss": [], "g_adv": [], "g_sup": [], "g_cyc": [], "d_out_std": []}
    Xt, Yt = torch.from_numpy(X).to(device), torch.from_numpy(Y).to(device)
    for i in range(len(Xt)):
        A, B = Xt[i:i + 1], Yt[i:i + 1]
        g_AB.eval(); g_BA.eval()
        with torch.no_grad():
            fake_B, fake_A = g_AB(A), g_BA(B)
            rec_A, rec_B = g_BA(fake_B), g_AB(fake_A)
        d_losses = []
        for D, real, fake in ((d_A, A, fake_A), (d_B, B, fake_B)):
            for x, lab in ((real, 1.0), (fake, 0.0)):
                _zero(D)
                out = D(x)
                loss = mse(out, torch.full_like(out, lab))
                loss.backward()
                opt.step(list(D.parameters()))
                d_losses.append(loss.item())
                if lab == 1.0:
                    log["d_out_std"].append(out.std().item())
        g_AB.train(); g_BA.train()
        _zero(g_AB, g_BA, d_A, d_B)
        fB, fA = g_AB(A), g_BA(B)
        rA, rB = g_BA(fB), g_AB(fA)
        vA, vB = d_A(fA), d_B(fB)
        adv = huber(vA, torch.ones_like(vA)) + huber(vB, torch.ones_like(vB))
        sup = huber(fB, B) + huber(fA, A)
        cyc = huber(rA, rec_A) + huber(rB, rec_B)
        (adv + lambda_cycle * sup + lambda_id * cyc).backward()
        opt.step(list(g_AB.parameters()) + list(g_BA.parameters()))
        log["d_loss"].append(float(np.mean(d_losses)))
        log["g_adv"].append(adv.item() / 2)
        log["g_sup"].append(sup.item() / 2)
        log["g_cyc"].append(cyc.item() / 2)
    g_AB.eval()
    return g_AB, log


def train_supervised(X: np.ndarray, Y: np.ndarray, seed: int, variant: str = "2021", device: str = "cuda") -> Generator:
    """Ablation: the same generator, data, optimiser and schedule, with only the direct Huber loss."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    g = Generator(X.shape[-1], variant).to(device)
    opt = KerasAdam()
    huber = nn.HuberLoss(delta=1.0)
    Xt, Yt = torch.from_numpy(X).to(device), torch.from_numpy(Y).to(device)
    g.train()
    for i in range(len(Xt)):
        _zero(g)
        (4.0 * huber(g(Xt[i:i + 1]), Yt[i:i + 1])).backward()
        opt.step(list(g.parameters()))
    g.eval()
    return g


@torch.no_grad()
def predict(g: Generator, X: np.ndarray, device: str = "cuda", batch: int = 64) -> np.ndarray:
    g.eval()
    out = [g(torch.from_numpy(X[i:i + batch]).to(device)).cpu().numpy() for i in range(0, len(X), batch)]
    return np.concatenate(out)
