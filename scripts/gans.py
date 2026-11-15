"""Phase 4: do GANs add anything over the supervised U-Net, for people the model has never seen?

Unseen-person setting only (4 folds: train on 10 people, early-stop on 2, test on 4), 3 seeds each, same
generator, windows, optimiser and test data as the U-Net in scripts/simple_models.py.

  pix2pix_adv0.1   paired, conditional discriminator, adversarial weight 0.1 next to L1 weight 1
  pix2pix_adv1     the same with a 10x stronger adversarial term
  cycle_paired     CycleGAN with the paired L1 terms added
  cycle_unpaired   CycleGAN proper: SCG and ECG windows drawn independently

Every method, the U-Net (from its saved weights) and the average-heartbeat baseline are also scored for beat
shape at the true R-peaks (scg2ecg.metrics.morphology).

Writes results/gans/{per_person.csv, histories.csv, summary.md, examples.png}.
Run from the repo root:  .venv\\Scripts\\python -m scripts.gans [--part NAME ...]
Resumable: runs with saved scores are skipped; runs with saved weights only are scored (--retrain redoes all).
"""
from __future__ import annotations

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

from scg2ecg.baselines import AverageBeat
from scg2ecg.data import prepared
from scg2ecg.gans import GanConfig, train_gan
from scg2ecg.metrics import morphology, score
from scg2ecg.models import UNet1d
from scg2ecg.paths import RESULTS
from scg2ecg.splits import get_splits, segments
from scg2ecg.windows import stitch_predict

from .simple_models import DEV, FS, LEN, SEEDS, segs

OUT = RESULTS / "gans"
CONFIGS = {
    "pix2pix_adv0.1": GanConfig("pix2pix", lam_adv=0.1),
    "pix2pix_adv1": GanConfig("pix2pix", lam_adv=1.0),
    # two generators and two discriminators per step: half the windows per epoch keeps the run time down
    "cycle_paired": GanConfig("cycle", paired=True, stride=2048),
    "cycle_unpaired": GanConfig("cycle", paired=False, stride=2048, max_epochs=16, patience=5),
}
METRICS = ["r", "rmse", "prd", "beat_se", "beat_ppv", "r_timing_abs_ms", "rr_mae_ms",
           "template_r", "beat_r", "r_amp_ratio"]


def score_people(pred_fn, people, base: dict) -> list[dict]:
    rows = []
    for s in people:
        test = segs([s], "late")
        preds = [pred_fn(t["x"]) for t in test]
        args = ([t["y"] for t in test], preds, [t["r"] for t in test], FS)
        rows.append({**base, "subject": s, **score(*args), **morphology(*args)})
    return rows


def save_rows(path, rows: list[dict], key: dict):
    """Replace the rows matching `key` (e.g. one method / fold / seed) and write the file at once, so an
    interrupted run keeps everything finished before it."""
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=list(key))
    hit = np.ones(len(old), bool)
    for c, v in key.items():
        hit &= (old[c] == v).to_numpy() if c in old else np.zeros(len(old), bool)
    pd.concat([old[~hit], pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)


def run_config(name: str, sp, folds, seeds, out, retrain: bool = False):
    """Train and score each fold / seed. Resumable: a run with scores already saved is skipped, and a run
    with saved weights but no scores (interrupted after training) is scored from the weights."""
    cfg = CONFIGS[name]
    path, hpath = out / "per_person.csv", out / "histories.csv"
    for k in folds:
        fold = sp["folds"][k]
        train = tune = None
        for seed in seeds:
            key = {"method": name, "fold": k, "seed": seed}
            done = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=list(key))
            if not retrain and ((done.method == name) & (done.fold == k) & (done.seed == seed)).any():
                print(f"{name} fold {k} seed {seed}: scores saved, skipping", flush=True)
                continue
            t0 = time.time()
            wpath = out / f"{name}_fold{k}_seed{seed}.pt"
            if wpath.exists() and not retrain:
                print(f"{name} fold {k} seed {seed}: scoring saved weights", flush=True)
                G = UNet1d().to(DEV)
                G.load_state_dict(torch.load(wpath, map_location=DEV))
                G.eval()
                extra = {}
            else:
                print(f"{name} fold {k} seed {seed}", flush=True)
                if train is None:
                    train, tune = segs(fold["train"], "all"), segs(fold["tune"], "all")
                G, r_tune, hist = train_gan(cfg, train, tune, seed, LEN, DEV)
                torch.save(G.state_dict(), wpath)
                save_rows(hpath, [{**key, **h} for h in hist], key)
                extra = {"best_epoch": max(hist, key=lambda h: h["val_r"])["epoch"], "tune_r": r_tune}
            rows = score_people(lambda x: stitch_predict(G, x, LEN, DEV), fold["test"],
                                {"setting": "unseen", **key, **extra})
            save_rows(path, rows, key)
            got = " ".join(f"{r['subject']}:{r['r']:.3f}" for r in rows)
            print(f"  -> {extra}, test r {got} ({(time.time() - t0) / 60:.1f} min)", flush=True)
            del G
            torch.cuda.empty_cache()


def rescore_references(sp, folds, seeds) -> list[dict]:
    """The U-Net from its saved weights and the average-heartbeat baseline, with the beat-shape scores."""
    rows = []
    for k in folds:
        fold = sp["folds"][k]
        for seed in seeds:
            m = UNet1d().to(DEV)
            m.load_state_dict(torch.load(RESULTS / "simple_models" / f"unet_unseen_fold{k}_seed{seed}.pt",
                                         map_location=DEV))
            m.eval()
            rows += score_people(lambda x: stitch_predict(m, x, LEN, DEV), fold["test"],
                                 {"setting": "unseen", "method": "unet", "fold": k, "seed": seed})

        def pieces(people, part):
            out = []
            for s in people:
                for name, a, b in segments(s, FS)[part]:
                    scg, ecg, r = prepared(name, FS)
                    out.append((scg[a:b], ecg[a:b], r[(r >= a) & (r < b)] - a))
            return out

        ab = AverageBeat(FS).fit(pieces(fold["train"], "all"))
        ab.tune(pieces(fold["tune"], "all"))
        for s in fold["test"]:
            test = pieces([s], "late")
            args = ([e for _, e, _ in test], [ab.predict(x)[0] for x, _, _ in test], [r for _, _, r in test], FS)
            rows.append({"setting": "unseen", "method": "avg_beat", "fold": k, "seed": None, "subject": s,
                         **score(*args), **morphology(*args)})
            args = ([e for _, e, _ in test], [e for _, e, _ in test], [r for _, _, r in test], FS)
            rows.append({"setting": "unseen", "method": "real_ecg", "fold": k, "seed": None, "subject": s,
                         **score(*args), **morphology(*args)})
        print(f"references fold {k} done", flush=True)
    return rows


def summarize(out):
    df = pd.read_csv(out / "per_person.csv")
    pp = df.groupby(["method", "subject"])[METRICS].mean().reset_index()  # mean over seeds
    order = [m for m in ["avg_beat", "unet", *CONFIGS, "real_ecg"] if m in set(pp.method)]
    table = pp.groupby("method")[METRICS].mean().reindex(order).round(3)
    spread = pp.groupby("method")["r"].agg(["mean", "std", "min", "median", "max"]).reindex(order).round(3)
    seed_sd = df[df.seed.notna()].groupby(["method", "subject"])["r"].std().groupby("method").median().round(3)
    piv = pp.pivot(index="subject", columns="method", values="r")
    tests = []
    for a in CONFIGS:
        for b in ("unet", "avg_beat"):
            if a in piv and b in piv:
                d = (piv[a] - piv[b]).dropna()
                p = wilcoxon(d).pvalue if len(d) > 5 else float("nan")
                tests.append(f"- {a} vs {b}: {a} better for {(d > 0).sum()} of {len(d)} people; mean r difference "
                             f"{d.mean():+.3f}; Wilcoxon p = {p:.3g}")
    shape = []
    for a in CONFIGS:
        if a in set(pp.method):
            x = pp[pp.method == a].set_index("subject")
            u = pp[pp.method == "unet"].set_index("subject")
            for m in ("r_amp_ratio", "template_r", "beat_r"):
                d = (x[m] - u[m]).dropna()
                p = wilcoxon(d).pvalue if len(d) > 5 else float("nan")
                shape.append(f"- {a} minus unet, {m}: mean {d.mean():+.3f}, higher for {(d > 0).sum()} of {len(d)}; "
                             f"p = {p:.3g}")
    hist = pd.read_csv(out / "histories.csv")
    best = hist.loc[hist.groupby(["method", "fold", "seed"])["val_r"].idxmax()]
    ep = best.groupby("method").agg(best_epoch_mean=("epoch", "mean"), best_epoch_max=("epoch", "max"),
                                    tune_r=("val_r", "mean")).round(2)
    ep["median_seconds_per_epoch"] = hist.groupby("method")["seconds"].median().round(0)

    lines = ["# GANs vs the supervised U-Net, unseen people (generated by scripts/gans.py)", "",
             "Test data: each development person's 'late' part, scored once per seed by the fold that holds the "
             "person out. Seeded rows are averaged over seeds per person first. unet and avg_beat are the "
             "models of results/simple_models and results/baselines, rescored here with the beat-shape metrics.", "",
             "## Means over the 16 development people", "", table.to_markdown(), "",
             "## Spread of r across people", "", spread.to_markdown(), "",
             f"Median seed SD of r per person: {seed_sd.to_dict()}", "",
             "## Person-by-person, Pearson r", "", *tests, "",
             "## Beat shape against the U-Net", "", *shape, "",
             "## Epoch chosen by early stopping (on the 2 tune people)", "", ep.to_markdown(), "",
             "## Pearson r per person", "", piv.reindex(columns=order).round(3).to_markdown(), "",
             "Beat-shape metrics (scg2ecg/metrics.py: morphology) are cut at the database's R-peak marks: "
             "template_r = average output beat vs average real beat; beat_r = median per-beat correlation; "
             "r_amp_ratio = output R amplitude / real (below 1: flattened beats; timing errors also lower it).",
             f"Training histories: histories.csv ({len(hist)} epochs)."]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def plot_examples(sp, out):
    df = pd.read_csv(out / "per_person.csv")
    have = [m for m in CONFIGS if (out / f"{m}_fold0_seed0.pt").exists()]
    pp = df[df.method == "unet"].groupby("subject")["r"].mean()
    show = [int(pp.idxmax()), int((pp - pp.median()).abs().idxmin()), int(pp.idxmin())]  # best, median, worst
    W = 5 * FS
    t = np.arange(W) / FS
    fig, axes = plt.subplots(len(show), 1, figsize=(13, 2.8 * len(show)), sharex=True)
    rng = np.random.default_rng(4)
    colors = ["#c0392b", "#8e44ad", "#d35400", "#16a085"]
    for ax, s in zip(axes, show):
        k = next(i for i, f in enumerate(sp["folds"]) if s in f["test"])
        seg = segs([s], "late")[0]
        a = int(rng.integers(LEN, len(seg["x"]) - W - LEN))
        x = seg["x"][a - LEN:a + W + LEN]
        y = seg["y"][a:a + W]
        models = [("unet", RESULTS / "simple_models" / f"unet_unseen_fold{k}_seed0.pt")] + \
                 [(m, out / f"{m}_fold{k}_seed0.pt") for m in have]
        for j, (m, path) in enumerate(models):
            G = UNet1d().to(DEV)
            G.load_state_dict(torch.load(path, map_location=DEV))
            p = stitch_predict(G, x, LEN, DEV)[LEN:LEN + W]
            ax.plot(t, y - 4 * j, "k", lw=0.7, alpha=0.6)
            ax.plot(t, p - 4 * j, color=(["#2471a3"] + colors)[j], lw=0.9, label=m)
        ax.set_title(f"subject {s} (unet r = {pp[s]:.2f}); black = real ECG", fontsize=8, loc="left")
        ax.set_yticks([])
    axes[0].legend(fontsize=7, ncol=5, loc="upper right")
    axes[-1].set_xlabel("seconds (random 5 s of the test part, seed 0 models, offset per method)")
    fig.tight_layout()
    fig.savefig(out / "examples.png", dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", nargs="+", default=["all"], choices=[*CONFIGS, "references", "summary", "all"])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--max-epochs", type=int, default=None, help="override, for smoke tests")
    ap.add_argument("--out", default=None, help="output folder (default results/gans)")
    ap.add_argument("--retrain", action="store_true", help="ignore saved scores and weights")
    args = ap.parse_args()
    out = OUT if args.out is None else __import__("pathlib").Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.max_epochs:
        for c in CONFIGS.values():
            c.max_epochs = args.max_epochs
    sp = get_splits()
    parts = ["references", *CONFIGS, "summary"] if "all" in args.part else args.part
    path = out / "per_person.csv"
    for part in parts:
        if part == "references":
            rows = rescore_references(sp, args.folds, args.seeds)
            old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["method"])
            keep = old[~old.method.isin(["unet", "avg_beat", "real_ecg"])]
            pd.concat([keep, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)
        elif part in CONFIGS:
            run_config(part, sp, args.folds, args.seeds, out, args.retrain)
    if "summary" in parts:
        summarize(out)
        plot_examples(sp, out)


if __name__ == "__main__":
    main()
