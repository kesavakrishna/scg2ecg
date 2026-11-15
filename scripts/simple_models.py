"""Phase 3: ridge regression and the supervised 1D U-Net, scored exactly like the baselines.

Settings (test data is each development person's 'late' part in both):
  unseen  4 folds: train on 10 people ('all' data), choose alpha / early-stop on the 2 tune people
  same    one model on every development person's 'early' data; its last 10% (after a 10 s gap) is the
          validation part for alpha / early stopping

Rows: ridge (sklearn), unet (3 seeds). Fold 0 also fits the ridge as one PyTorch Conv1d to check the two match.
Writes results/simple_models/{per_person.csv, summary.md, ridge_vs_conv1d.txt, examples.png, histories.csv}.
Run from the repo root:  .venv\\Scripts\\python -m scripts.simple_models [--part ridge|unet|summary|all]
"""
from __future__ import annotations

import argparse
import copy
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon
from torch.utils.data import DataLoader

from scg2ecg.metrics import pearson, score
from scg2ecg.models import ConvFIR, RidgeFIR, UNet1d, n_params
from scg2ecg.paths import RESULTS
from scg2ecg.splits import GAP_S, get_splits, segments
from scg2ecg.windows import WindowDataset, load_segments, stitch_predict

FS = 500
LEN = 2048
OUT = RESULTS / "simple_models"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ALPHAS = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
SEEDS = (0, 1, 2)


def segs(subjects, part):
    return load_segments([sg for s in subjects for sg in segments(s, FS)[part]], FS)


def same_person_parts(dev):
    """Every person's 'early' data split into train and a validation tail (last 10%, 10 s gap)."""
    train, val = [], []
    for s in dev:
        early = segments(s, FS)["early"]
        *rest, (name, a, b) = early
        n_val = int(0.10 * (b - a))
        train += rest + [(name, a, b - n_val - int(GAP_S * FS))]
        val += [(name, b - n_val, b)]
    return load_segments(train, FS), load_segments(val, FS)


def mean_r(pred_fn, segments_) -> float:
    return float(np.mean([pearson(s["y"], pred_fn(s["x"])) for s in segments_]))


def score_people(pred_fn, people, base: dict) -> list[dict]:
    rows = []
    for s in people:
        test = segs([s], "late")
        preds = [pred_fn(t["x"]) for t in test]
        rows.append({**base, "subject": s, **score([t["y"] for t in test], preds, [t["r"] for t in test], FS)})
    return rows


# ---------------------------------------------------------------- ridge

def fit_ridge(train, val):
    best = None
    for alpha in ALPHAS:
        m = RidgeFIR(FS, alpha=alpha).fit(train)
        r = mean_r(m.predict, val)
        if best is None or r > best[0]:
            best = (r, alpha, m)
    return best[2], best[1], best[0]


def conv1d_check(train, test_people, sk: RidgeFIR) -> str:
    """Train the same linear filter as one nn.Conv1d with Adam and compare with sklearn."""
    torch.manual_seed(0)
    model = ConvFIR(sk.K).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    dl = DataLoader(WindowDataset(train, LEN, 256), batch_size=64, shuffle=True, drop_last=True)
    lam = sk.alpha / sk.n_fit  # sklearn minimises sum of squares + alpha |w|^2 over n_fit points
    h, step = sk.half, 0
    while step < 6000:
        for xb, yb in dl:
            xb, yb = xb.to(DEV), yb.to(DEV)
            pred = model(xb)[..., h:-h]
            loss = F.mse_loss(pred, yb[..., h:-h]) + lam * model.conv.weight.pow(2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step >= 6000:
                break
    w_t = model.conv.weight.detach().cpu().numpy().ravel()
    lines = [f"kernel correlation sklearn vs Conv1d: {pearson(sk.w, w_t):.4f}",
             f"intercept sklearn {sk.b:+.4f}  Conv1d bias {model.conv.bias.item():+.4f}"]
    for s in test_people:
        for t in segs([s], "late")[:1]:
            p_sk = sk.predict(t["x"])
            with torch.no_grad():
                p_t = model(torch.from_numpy(t["x"])[None, None].to(DEV)).cpu().numpy().ravel()
            lines.append(f"subject {s}: r(sklearn, ECG)={pearson(t['y'], p_sk):.4f}  r(Conv1d, ECG)={pearson(t['y'], p_t):.4f}  "
                         f"r(sklearn, Conv1d)={pearson(p_sk, p_t):.4f}")
    return "\n".join(lines)


def run_ridge(sp) -> list[dict]:
    rows = []
    for k, fold in enumerate(sp["folds"]):
        train, tune = segs(fold["train"], "all"), segs(fold["tune"], "all")
        m, alpha, r_tune = fit_ridge(train, tune)
        print(f"ridge fold {k}: alpha={alpha:g} tune r={r_tune:.3f}", flush=True)
        rows += score_people(m.predict, fold["test"], {"setting": "unseen", "method": "ridge", "fold": k,
                                                        "seed": None, "alpha": alpha})
        if k == 0:
            txt = conv1d_check(train, fold["test"], m)
            (OUT / "ridge_vs_conv1d.txt").write_text(txt + "\n")
            print(txt, flush=True)
    train, val = same_person_parts(sp["dev"])
    m, alpha, r_val = fit_ridge(train, val)
    print(f"ridge same-person: alpha={alpha:g} val r={r_val:.3f}", flush=True)
    rows += score_people(m.predict, sp["dev"], {"setting": "same", "method": "ridge", "fold": None, "seed": None,
                                                "alpha": alpha})
    return rows


# ---------------------------------------------------------------- U-Net

def train_unet(train, val, seed: int, max_epochs: int = 15, patience: int = 3, stride: int = 1024):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = UNet1d().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ds = WindowDataset(train, LEN, stride)
    g = torch.Generator().manual_seed(seed)
    best, best_state, hist, bad = -1.0, None, [], 0
    for epoch in range(max_epochs):
        ds.set_epoch(seed * 1000 + epoch)
        model.train()
        t0, losses = time.time(), []
        for xb, yb in DataLoader(ds, batch_size=32, shuffle=True, drop_last=True, generator=g):
            xb, yb = xb.to(DEV, non_blocking=True), yb.to(DEV, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
                loss = F.l1_loss(model(xb).float(), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        r = mean_r(lambda x: stitch_predict(model, x, LEN, DEV), val)
        hist.append({"epoch": epoch, "train_l1": float(np.mean(losses)), "val_r": r, "seconds": time.time() - t0})
        if r > best:
            best, best_state, bad = r, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best, hist


def run_unet(sp) -> tuple[list[dict], list[dict]]:
    rows, hists = [], []
    for k, fold in enumerate(sp["folds"]):
        train, tune = segs(fold["train"], "all"), segs(fold["tune"], "all")
        for seed in SEEDS:
            t0 = time.time()
            model, r_tune, hist = train_unet(train, tune, seed)
            hists += [{"setting": "unseen", "fold": k, "seed": seed, **h} for h in hist]
            best_ep = max(hist, key=lambda h: h["val_r"])["epoch"]
            print(f"unet fold {k} seed {seed}: best epoch {best_ep}, tune r={r_tune:.3f}, {n_params(model) / 1e6:.2f}M params, "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
            rows += score_people(lambda x: stitch_predict(model, x, LEN, DEV), fold["test"],
                                 {"setting": "unseen", "method": "unet", "fold": k, "seed": seed})
            torch.save(model.state_dict(), OUT / f"unet_unseen_fold{k}_seed{seed}.pt")
    train, val = same_person_parts(sp["dev"])
    for seed in SEEDS:
        model, r_val, hist = train_unet(train, val, seed)
        hists += [{"setting": "same", "fold": None, "seed": seed, **h} for h in hist]
        print(f"unet same-person seed {seed}: val r={r_val:.3f}", flush=True)
        rows += score_people(lambda x: stitch_predict(model, x, LEN, DEV), sp["dev"],
                             {"setting": "same", "method": "unet", "fold": None, "seed": seed})
        torch.save(model.state_dict(), OUT / f"unet_same_seed{seed}.pt")
    return rows, hists


# ---------------------------------------------------------------- summary

def summarize(sp):
    df = pd.read_csv(OUT / "per_person.csv")
    base = pd.read_csv(RESULTS / "baselines" / "per_person.csv")
    base = base[base.method.isin(["avg_beat", "avg_beat_true_r"])]
    allr = pd.concat([base, df], ignore_index=True)
    metrics = ["r", "rmse", "prd", "beat_se", "beat_ppv", "r_timing_abs_ms", "rr_mae_ms"]
    pp = allr.groupby(["setting", "method", "subject"])[metrics].mean().reset_index()  # mean over seeds
    table = pp.groupby(["setting", "method"])[metrics].mean().round(3)
    spread = pp.groupby(["setting", "method"])["r"].agg(["mean", "std", "min", "median", "max"]).round(3)
    seed_sd = df[df.method == "unet"].groupby(["setting", "subject"])["r"].std().groupby("setting").median()

    tests = []
    for setting in ("unseen", "same"):
        piv = pp[pp.setting == setting].pivot(index="subject", columns="method", values="r")
        for a, b in (("unet", "avg_beat"), ("unet", "ridge"), ("ridge", "avg_beat")):
            if a in piv and b in piv:
                d = (piv[a] - piv[b]).dropna()
                p = wilcoxon(d).pvalue if len(d) > 1 and (d != 0).any() else float("nan")
                tests.append(f"- {setting}: {a} beats {b} for {(d > 0).sum()} of {len(d)} people; mean r difference "
                             f"{d.mean():+.3f}; Wilcoxon p = {p:.3g}")
    gap = pp.pivot_table(index=["method", "subject"], columns="setting", values="r").dropna()
    gap = (gap["same"] - gap["unseen"]).groupby("method").agg(["mean", "min", "max"]).round(3)
    worst = pp[(pp.setting == "unseen") & (pp.method == "unet")].nsmallest(3, "r")["subject"].tolist()

    lines = ["# Simple models vs the baseline (generated by scripts/simple_models.py)", "",
             "Same test data for every row: each development person's 'late' part. Seeded rows are averaged over "
             "seeds per person first.", "",
             "## Means over the 16 development people", "", table.to_markdown(), "",
             "## Spread of r across people", "", spread.to_markdown(), "",
             f"Median seed SD of U-Net r per person: {seed_sd.round(3).to_dict()}", "",
             "## Person-by-person", "", *tests, "",
             "## Same-person minus unseen-person r", "", gap.to_markdown(), "",
             f"Three worst people for the unseen U-Net: {worst}."]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return worst


def plot_examples(sp, worst):
    rng = np.random.default_rng(2)
    show = worst + [int(x) for x in rng.choice([s for s in sp["dev"] if s not in worst], 2, replace=False)]
    W = 6 * FS
    fig, axes = plt.subplots(len(show), 1, figsize=(13, 2.4 * len(show)), sharex=True)
    t = np.arange(W) / FS
    same = UNet1d().to(DEV)
    same.load_state_dict(torch.load(OUT / "unet_same_seed0.pt", map_location=DEV))
    for ax, s in zip(axes, show):
        k = next(i for i, f in enumerate(sp["folds"]) if s in f["test"])
        m = UNet1d().to(DEV)
        m.load_state_dict(torch.load(OUT / f"unet_unseen_fold{k}_seed0.pt", map_location=DEV))
        seg = segs([s], "late")[0]
        a = int(rng.integers(0, len(seg["x"]) - W))
        x = seg["x"][a - LEN:a + W + LEN] if a >= LEN else seg["x"][a:a + W + LEN]
        off = LEN if a >= LEN else 0
        pu = stitch_predict(m, x, LEN, DEV)[off:off + W]
        ps = stitch_predict(same, x, LEN, DEV)[off:off + W]
        y = seg["y"][a:a + W]
        ax.plot(t, y, "k", lw=1.0, label="real ECG")
        ax.plot(t, pu, color="#c0392b", lw=0.9, label="U-Net, unseen person")
        ax.plot(t, y - 4, "k", lw=0.6, alpha=0.5)
        ax.plot(t, ps - 4, color="#2471a3", lw=0.9, label="U-Net, same person (offset)")
        ax.set_title(f"subject {s}{' (worst 3)' if s in worst else ''}", fontsize=8, loc="left")
        ax.set_yticks([])
    axes[0].legend(fontsize=7, ncol=3, loc="upper right")
    axes[-1].set_xlabel("seconds (random 6 s of the test part, seed 0 models)")
    fig.tight_layout()
    fig.savefig(OUT / "examples.png", dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["ridge", "unet", "summary", "all"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sp = get_splits()
    path = OUT / "per_person.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if args.part in ("ridge", "all"):
        rows = run_ridge(sp)
        old = pd.concat([old[old.method != "ridge"] if len(old) else old, pd.DataFrame(rows)], ignore_index=True)
        old.to_csv(path, index=False)
    if args.part in ("unet", "all"):
        rows, hists = run_unet(sp)
        old = pd.concat([old[old.method != "unet"] if len(old) else old, pd.DataFrame(rows)], ignore_index=True)
        old.to_csv(path, index=False)
        pd.DataFrame(hists).to_csv(OUT / "histories.csv", index=False)
    if args.part in ("summary", "all") or args.part == "unet":
        worst = summarize(sp)
        if (OUT / "unet_same_seed0.pt").exists():
            plot_examples(sp, worst)


if __name__ == "__main__":
    main()
