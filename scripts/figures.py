"""D9: the summary figure for the write-up - one unseen person's journey (CEBS subject 6, the U-Net's worst case).

Same 4 s of the person's test part in every panel: the SCG input; the U-Net without calibration; shifted using 10 s
of ECG; fine-tuned on 3 s of ECG; and, last, what the same model draws from white noise, with the heartbeat check's
verdict. r in each title is the mean over 3 seeds for the whole test part (from results/), not this window.

Writes results/figures/journey.png. Run from the repo root:  .venv\\Scripts\\python -m scripts.figures
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scg2ecg.gate import periodicity
from scg2ecg.paths import RESULTS
from scg2ecg.splits import get_splits
from scg2ecg.windows import load_segments, stitch_predict

from .calibration import apply_shift, best_shift, calib_segment, load_unet
from .cheap_calibration import finetune_on
from .simple_models import DEV, FS, LEN, segs

OUT = RESULTS / "figures"
SUBJECT = 6
# reference palette (dataviz skill), light mode; aqua is under 3:1 contrast, so every panel is labelled in text
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#b9b8b3"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"


def mean_r(path, method, calib_s):
    d = pd.read_csv(path)
    d = d[(d.subject == SUBJECT) & (d.method == method) & (d.calib_s == calib_s)]
    return d.r.mean()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sp = get_splits()
    k = next(i for i, f in enumerate(sp["folds"]) if SUBJECT in f["test"])
    m = load_unet(k, 0)
    test = segs([SUBJECT], "late")[0]
    W = 4 * FS
    a = len(test["x"]) // 3
    x = test["x"][a - LEN:a + W + LEN]
    y = test["y"][a:a + W]
    sl = slice(LEN, LEN + W)
    p = stitch_predict(m, x, LEN, DEV)
    clip10 = load_segments([calib_segment(SUBJECT, 10)], FS)[0]
    d = best_shift(clip10["y"], stitch_predict(m, clip10["x"], LEN, DEV))
    clip3 = (clip10["x"][:3 * FS], clip10["y"][:3 * FS])
    ft = finetune_on(m, clip3[0], clip3[1], 0)
    noise = np.random.default_rng(0).standard_normal(len(x)).astype(np.float32)
    pn = stitch_predict(m, noise, LEN, DEV)
    th = pd.read_csv(RESULTS / "gate" / "windows.csv")
    th = th[(th.fold == k) & (th.role == "threshold") & (th.input == "real")].periodicity.quantile(0.05)
    per_real = periodicity(test["x"][a:a + 8 * FS], FS)
    per_noise = periodicity(noise[LEN:LEN + 8 * FS], FS)

    cal, cheap = RESULTS / "calibration" / "per_person.csv", RESULTS / "cheap_calibration" / "per_person.csv"
    r_none = mean_r(cal, "unet", 0)
    r_shift = mean_r(cal, "unet_shift", 10)
    r_ft3 = mean_r(cheap, "ecg_finetune", 3)

    t = np.arange(W) / FS
    panels = [
        ("Input: chest vibration (SCG), normalised", x[sl], INK2, None),
        (f"No calibration (U-Net trained on 10 other people): r = {r_none:.2f}", p[sl], BLUE, y),
        (f"Shifted by {d * 1000 / FS:+.0f} ms, measured on 10 s of this person's ECG: r = {r_shift:.2f}",
         apply_shift(p, d)[sl], ORANGE, y),
        (f"Fine-tuned on 3 s of this person's ECG: r = {r_ft3:.2f}", stitch_predict(ft, x, LEN, DEV)[sl], AQUA, y),
        (f"Same model fed white noise: it still draws ECG-like beats. Heartbeat check: periodicity {per_noise:.2f} "
         f"< threshold {th:.2f}, output withheld (real SCG: {per_real:.2f})", pn[sl], RED, None),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 9.2), sharex=True, facecolor=SURFACE)
    for ax, (title, sig, col, ref) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        if ref is not None:
            ax.plot(t, ref, color=INK, lw=1.0, alpha=0.55, label="real ECG")
        ax.plot(t, sig, color=col, lw=1.6 if ref is not None else 1.0)
        ax.set_title(title, fontsize=9, loc="left", color=INK)
        ax.set_yticks([])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
        ax.tick_params(colors=INK2, labelsize=8)
    axes[1].legend(fontsize=8, loc="upper right", frameon=False, labelcolor=INK2)
    axes[-1].set_xlabel("seconds", color=INK2, fontsize=9)
    fig.suptitle(f"CEBS subject {SUBJECT}, as a person the model has never seen (grey = real ECG)", fontsize=11,
                 x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "journey.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", OUT / "journey.png")


if __name__ == "__main__":
    main()
