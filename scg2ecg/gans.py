"""Rung 4: correctly wired GANs, built on the same 1D U-Net generator as the supervised model.

Three set-ups, all trained on windows of the 10 training people of a fold:

  pix2pix          paired. G: SCG -> ECG. Loss = L1(G(s), e) + lam_adv * LSGAN(D(s, G(s))).
                   D sees the SCG with the ECG (conditional), so it judges whether the ECG fits that SCG.
                   With lam_adv = 0 this is exactly the supervised U-Net (same init, data order, optimiser).
  cycle (paired)   G: SCG -> ECG, F: ECG -> SCG, one unconditional discriminator per domain, cycle loss
                   |F(G(s)) - s| + |G(F(e)) - e|, plus the paired L1 terms |G(s) - e| + |F(e) - s|.
  cycle (unpaired) the same without the paired terms; the ECG windows in a batch come from a separate shuffle,
                   so the model never sees an SCG with the ECG recorded at the same time. This is the setting
                   CycleGAN was made for.

What the original model got wrong, and what is different here: the discriminators end in a plain convolution
(no normalisation over their output), so their output depends on the input; the cycle loss compares against
the real input, not the model's own earlier output; the generator output is not normalised, so it can draw a
flat line. Discriminators use spectral normalisation and least-squares GAN losses.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
from torch.utils.data import DataLoader

from .metrics import pearson
from .models import UNet1d
from .windows import WindowDataset, stitch_predict


class PatchD(nn.Module):
    """1D PatchGAN: one real/fake score per patch. Receptive field 435 samples (0.87 s at 500 Hz), about one
    beat, so it judges beat shape rather than the whole window."""

    def __init__(self, cin: int, ch=(32, 64, 128, 256), k: int = 15):
        super().__init__()
        layers, c = [], cin
        for co in ch:
            layers += [spectral_norm(nn.Conv1d(c, co, k, stride=2, padding=k // 2)), nn.LeakyReLU(0.2)]
            c = co
        layers.append(spectral_norm(nn.Conv1d(c, 1, k, padding=k // 2)))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def ls_real(d):
    return F.mse_loss(d.float(), torch.ones_like(d, dtype=torch.float32))


def ls_fake(d):
    return F.mse_loss(d.float(), torch.zeros_like(d, dtype=torch.float32))


@dataclass
class GanConfig:
    kind: str  # 'pix2pix' or 'cycle'
    paired: bool = True
    lam_sup: float = 1.0  # paired L1 (both directions for cycle)
    lam_adv: float = 0.1
    lam_cyc: float = 1.0  # cycle only; CycleGAN's 10:1 cycle-to-adversarial ratio
    max_epochs: int = 15
    patience: int = 3
    stride: int = 1024
    batch: int = 32
    lr_g: float = 1e-3  # same optimiser as the supervised U-Net
    lr_d: float = 1e-3  # probe: higher than pix2pix's 2e-4, else D stays at chance (results/gans/summary.md)

    @property
    def name(self) -> str:
        if self.kind == "pix2pix":
            return f"pix2pix_adv{self.lam_adv:g}"
        return f"cycle_{'paired' if self.paired else 'unpaired'}"


def mean_r(G, segs, length, device) -> float:
    return float(np.mean([pearson(s["y"], stitch_predict(G, s["x"], length, device)) for s in segs]))


def train_gan(cfg: GanConfig, train: list[dict], val: list[dict], seed: int, length: int, device: str):
    """Train one set-up; keep the generator from the epoch with the best mean r on `val` (early stopping).
    For the unpaired cycle this selection step is the only place paired data is used."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    G = UNet1d().to(device)  # created first, so its init matches the supervised U-Net of the same seed
    opt_g_params = list(G.parameters())
    if cfg.kind == "pix2pix":
        D = PatchD(2).to(device)
        Ds = [D]
    else:
        Fn = UNet1d().to(device)
        DE, DS = PatchD(1).to(device), PatchD(1).to(device)
        Ds = [DE, DS]
        opt_g_params += list(Fn.parameters())
    opt_g = torch.optim.AdamW(opt_g_params, lr=cfg.lr_g, weight_decay=1e-4)
    opt_d = torch.optim.Adam([p for d in Ds for p in d.parameters()], lr=cfg.lr_d, betas=(0.5, 0.999))
    ds = WindowDataset(train, length, cfg.stride)
    g_order = torch.Generator().manual_seed(seed)
    g_unpair = torch.Generator().manual_seed(seed + 10_000)
    ac = dict(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda")

    best, best_state, hist, bad = -1.0, None, [], 0
    for epoch in range(cfg.max_epochs):
        ds.set_epoch(seed * 1000 + epoch)
        G.train()
        t0, logs = time.time(), []
        loader = DataLoader(ds, batch_size=cfg.batch, shuffle=True, drop_last=True, generator=g_order)
        others = iter(DataLoader(ds, batch_size=cfg.batch, shuffle=True, drop_last=True, generator=g_unpair)) \
            if not cfg.paired else None
        for s, e in loader:
            s, e = s.to(device, non_blocking=True), e.to(device, non_blocking=True)
            if others is not None:
                e = next(others)[1].to(device, non_blocking=True)  # an ECG from some other time / person
            log = {}
            if cfg.kind == "pix2pix":
                with torch.autocast(**ac):
                    fake = G(s)
                    ld = 0.5 * (ls_real(D(torch.cat([s, e], 1))) + ls_fake(D(torch.cat([s, fake.detach()], 1))))
                opt_d.zero_grad(set_to_none=True)
                ld.backward()
                opt_d.step()
                with torch.autocast(**ac):
                    l_sup = F.l1_loss(fake.float(), e)
                    l_adv = ls_real(D(torch.cat([s, fake], 1)))
                    lg = cfg.lam_sup * l_sup + cfg.lam_adv * l_adv
                log.update(d=ld.item(), sup=l_sup.item(), adv=l_adv.item())
            else:
                with torch.autocast(**ac):
                    fake_e, fake_s = G(s), Fn(e)
                    ld = 0.5 * (ls_real(DE(e)) + ls_fake(DE(fake_e.detach())) +
                                ls_real(DS(s)) + ls_fake(DS(fake_s.detach())))
                opt_d.zero_grad(set_to_none=True)
                ld.backward()
                opt_d.step()
                with torch.autocast(**ac):
                    l_adv = ls_real(DE(fake_e)) + ls_real(DS(fake_s))
                    l_cyc = F.l1_loss(Fn(fake_e).float(), s) + F.l1_loss(G(fake_s).float(), e)
                    lg = cfg.lam_adv * l_adv + cfg.lam_cyc * l_cyc
                    if cfg.paired:
                        l_sup = F.l1_loss(fake_e.float(), e) + F.l1_loss(fake_s.float(), s)
                        lg = lg + cfg.lam_sup * l_sup
                        log["sup"] = l_sup.item()
                log.update(d=ld.item(), adv=l_adv.item(), cyc=l_cyc.item())
            opt_g.zero_grad(set_to_none=True)
            lg.backward()
            opt_g.step()
            logs.append(log)
        r = mean_r(G, val, length, device)
        h = {"epoch": epoch, "val_r": r, "seconds": time.time() - t0}
        h.update({f"train_{k}": float(np.mean([l[k] for l in logs])) for k in logs[0]})
        hist.append(h)
        print(f"    epoch {epoch}: val r={r:.3f} " + " ".join(f"{k}={v:.4f}" for k, v in h.items()
                                                          if k.startswith("train_")) + f" ({h['seconds']:.0f}s)",
              flush=True)
        if r > best:
            best, best_state, bad = r, copy.deepcopy(G.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    G.load_state_dict(best_state)
    G.eval()
    return G, best, hist
