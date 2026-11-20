"""SCG-RHC: heart-failure patients, a third sensor (PhysioNet, ODC-By 1.0). Parts A2, A3, B4 (test) and B5.

Data: 83 right-heart-catheter recordings (~10 min each) of 73 patients. A chest patch below the jugular notch records
a single-lead ECG and a 3-axis accelerometer; the catheter lab records a standard ECG, including lead I. Everything is
at 500 Hz and synchronised. Used here: ECG lead I (target, as in CEBS), the patch's dorsal-ventral axis (the axis
CEBS measures) and, for B5, all three axes. Only recordings marked fine_alignment are used, and only those where the
lead I detector finds a plausible heart rate (30-150 bpm); 17 of the 83 recordings have no lead I channel at all
and are left out. R-peaks: neurokit2 on lead I.
Per recording: calibration clip from 10 s (10 or 30 s long); test = from 60 s, at most 10 min (amended before any
result was produced, for run time: recordings turned out to average 29 min).

Plan, fixed before running (committed with this file):
  A2 transfer   CEBS final U-Net (3 seeds), no calibration; shifted (30 s); fine-tuned (10 s, 30 s; 100 steps);
                the patient's own average beat (10 s, 30 s). r, beat sensitivity / PPV, timing, beat shape; beat r on
                irregular vs regular beats; QRS and QT errors (seed 0).
  B4 test       the three U-Nets of scripts/pooled.py (cebs, sst, pooled), no calibration and fine-tuned on 10 s.
  A3 revisit    patients with a second procedure (another day, the patch re-attached): calibrate on the first
                procedure's clip (30 s), test on the second one's test part; compared with calibrating on the second
                procedure itself.
  B5 axes       U-Nets trained on SCG-RHC patients themselves, 4 folds by patient (early stopping on ~10% of the
                training patients), 1 axis (dorsal-ventral) vs 3 axes, 3 seeds; unseen patients, without and with
                10 s fine-tuning.

Writes results/rhc/{records.csv, transfer.csv, revisit.csv, axes.csv, summary.md}.
Run from the repo root:  .venv\\Scripts\\python -m scripts.rhc [--part prep transfer revisit axes summary]
"""
from __future__ import annotations

import argparse
import copy
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wfdb
from torch.utils.data import DataLoader, Dataset

from scg2ecg.baselines import AverageBeat
from scg2ecg.data import ECG_BAND, SCG_BAND, bandpass, zscore
from scg2ecg.metrics import detect_r_peaks, morphology, pearson, score
from scg2ecg.models import UNet1d
from scg2ecg.paths import CACHE, RESULTS, ROOT
from scg2ecg.windows import normalize_scg, stitch_predict

from .calibration import apply_shift, best_shift
from .clinical import delineate, irregular_flags
from .simple_models import DEV, FS, LEN, SEEDS

RAW = ROOT / "data" / "external" / "scg_rhc" / "processed_data"
OUT = RESULTS / "rhc"
CAL_START_S, TEST_START_S = 10, 60
TEST_LEN_S = 600  # amended before any result: test parts capped at 10 min (recordings average 29 min; compute)
FT_STEPS = 100
AXES = ("patch_ACC_lat", "patch_ACC_hf", "patch_ACC_dv")
METRICS = ["r", "beat_se", "beat_ppv", "r_timing_abs_ms", "template_r", "beat_r", "r_amp_ratio"]
PRE, POST = int(0.25 * FS), int(0.45 * FS)


# ---------------------------------------------------------------- data

def record_list() -> pd.DataFrame:
    rows = []
    for j in sorted(RAW.glob("*.json")):
        name = j.stem
        if not (RAW / f"{name}.dat").exists():
            continue
        meta = json.load(open(j))
        rows.append({"record": name, "patient": int(name[3:6]), "visit": int(name.split("RHC")[1]),
                     "fine_alignment": bool(meta.get("fine_alignment", False)), "nyha": meta.get("NYHAC"),
                     "challenge": meta.get("IsChallenge"), "history": meta.get("history of patient", "")})
    return pd.DataFrame(rows).sort_values(["patient", "visit"]).reset_index(drop=True)


def load(name: str) -> dict:
    """Cached: band-passed, normalised axes (3, n), dorsal-ventral raw band-passed SCG, z-scored lead I, R-peaks."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"rhc_{name}.npz"
    if not path.exists():
        rec = wfdb.rdrecord(str(RAW / name))
        assert rec.fs == FS
        sig = {n: rec.p_signal[:, i].astype(np.float64) for i, n in enumerate(rec.sig_name)}
        ecg = np.nan_to_num(sig["ECG_lead_I"])
        y = zscore(bandpass(ecg - np.median(ecg), *ECG_BAND, FS))
        raw = [bandpass(np.nan_to_num(sig[a]) - np.nanmedian(sig[a]), *SCG_BAND, FS) for a in AXES]
        x3 = np.stack([normalize_scg(a, FS) for a in raw]).astype(np.float32)
        np.savez(path, x3=x3, scg=raw[2].astype(np.float32), y=y, r=detect_r_peaks(y, FS))
    z = np.load(path)
    return {"x3": z["x3"], "x": z["x3"][2], "scg": z["scg"], "y": z["y"], "r": z["r"]}


def cut(d: dict, a: int, b: int | None = None) -> dict:
    b = len(d["y"]) if b is None else b
    r = d["r"][(d["r"] >= a) & (d["r"] < b)] - a
    return {"x3": d["x3"][:, a:b], "x": d["x"][a:b], "scg": d["scg"][a:b], "y": d["y"][a:b], "r": r}


def usable(recs: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for row in recs.itertuples():
        if not row.fine_alignment:
            continue
        if "ECG_lead_I" not in wfdb.rdheader(str(RAW / row.record)).sig_name:  # 17 of 83 have no lead I
            continue
        d = load(row.record)
        minutes = len(d["y"]) / FS / 60
        hr = len(d["r"]) / minutes if minutes else 0
        if minutes >= 3 and 30 <= hr <= 150 and np.std(d["scg"]) > 0:
            keep.append({**row._asdict(), "minutes": round(minutes, 2), "hr": round(hr, 1)})
    return pd.DataFrame(keep).drop(columns=["Index"])


# ---------------------------------------------------------------- scoring helpers

def scored(base, t, pred):
    args = ([t["y"]], [pred], [t["r"]], FS)
    return {**base, **score(*args), **morphology(*args), **irregular_split(t, pred)}


def irregular_split(t, pred) -> dict:
    r = t["r"][(t["r"] > PRE) & (t["r"] < len(t["y"]) - POST)]
    if len(r) < 12:
        return {}
    flags = irregular_flags(r)
    br = np.array([pearson(t["y"][i - PRE:i + POST], pred[i - PRE:i + POST]) for i in r])
    out = {"beat_r_regular": float(np.median(br[~flags]))}
    if flags.sum() >= 3:
        out.update({"beat_r_irregular": float(np.median(br[flags])), "irregular_beats": int(flags.sum())})
    return out


_REAL = {}


def intervals(t, pred) -> dict:
    _, b = delineate((None, pred, t["r"]))
    key = (len(t["y"]), float(t["y"][:1000].sum()))
    if key not in _REAL:
        _REAL[key] = delineate((None, t["y"], t["r"]))[1]
    rb = _REAL[key]
    d = b.merge(rb, on="r", suffixes=("", "_real"))
    return {f"{c}_beat_mae": float(np.nanmedian(np.abs(d[c] - d[f"{c}_real"]))) for c in ("qrs_ms", "qt_ms")} | \
           {"qrs_ms_real": float(np.nanmedian(d.qrs_ms_real))}


def finetune(model, clips, seed, steps=FT_STEPS, lr=1e-4, batch=16, key="x"):
    crop = min(LEN, min(c[key].shape[-1] for c in clips) // 16 * 16)
    m = copy.deepcopy(model).train()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.0)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        xb, yb = [], []
        for _ in range(batch):
            c = clips[rng.integers(len(clips))]
            a = rng.integers(0, c["y"].shape[-1] - crop + 1)
            x = c[key][..., a:a + crop]
            xb.append(x if x.ndim == 2 else x[None])
            yb.append(c["y"][a:a + crop][None])
        xb, yb = torch.from_numpy(np.stack(xb)).to(DEV), torch.from_numpy(np.stack(yb)).to(DEV)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
            loss = F.l1_loss(m(xb).float(), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return m.eval()


def own_beat(clip, test):
    ab = AverageBeat(FS).fit([(clip["scg"], clip["y"], clip["r"])])
    ab.tune([(clip["scg"], clip["y"], clip["r"])])
    return ab.predict(test["scg"])[0]


def load_unets(folder, prefix, cin=1):
    out = []
    for seed in SEEDS:
        p = RESULTS / folder / f"{prefix}_seed{seed}.pt"
        if not p.exists():
            return []
        m = UNet1d(cin=cin).to(DEV)
        m.load_state_dict(torch.load(p, map_location=DEV))
        out.append(m.eval())
    return out


@torch.no_grad()
def predict3(model, x3, length=LEN, hop=None):
    """stitch_predict for a multi-channel input (C, n)."""
    hop = hop or length // 2
    n = x3.shape[-1]
    starts = list(range(0, n - length + 1, hop))
    if starts[-1] != n - length:
        starts.append(n - length)
    w = np.hanning(length).astype(np.float32) + 1e-3
    out, ws = np.zeros(n, np.float32), np.zeros(n, np.float32)
    for i in range(0, len(starts), 64):
        ss = starts[i:i + 64]
        xb = torch.from_numpy(np.stack([x3[:, a:a + length] for a in ss])).to(DEV)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
            yb = model(xb).float().cpu().numpy()[:, 0]
        for a, y in zip(ss, yb):
            out[a:a + length] += w * y
            ws[a:a + length] += w
    return out / ws


# ---------------------------------------------------------------- A2 + B4

def transfer(recs: pd.DataFrame):
    path = OUT / "transfer.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {r["record"] for r in rows}
    cebs = load_unets("holdout", "unet")
    pooled = {k: load_unets("pooled", k) for k in ("cebs", "sst", "pooled")}
    c0 = CAL_START_S * FS
    for row in recs.itertuples():
        if row.record in done:
            continue
        t0 = time.time()
        d = load(row.record)
        test = cut(d, TEST_START_S * FS, (TEST_START_S + TEST_LEN_S) * FS)
        clips = {c: cut(d, c0, c0 + c * FS) for c in (10, 30)}
        base = {"record": row.record, "patient": row.patient, "visit": row.visit}
        for c in (10, 30):
            p = own_beat(clips[c], test)
            extra = intervals(test, p) if c == 30 else {}
            rows.append(scored({**base, "method": "avg_beat_own", "calib_s": c, "seed": None, **extra}, test, p))
        for seed, u in enumerate(cebs):
            p = stitch_predict(u, test["x"], LEN, DEV)
            rows.append(scored({**base, "method": "unet", "calib_s": 0, "seed": seed,
                                **(intervals(test, p) if seed == 0 else {})}, test, p))
            sh = best_shift(clips[30]["y"], stitch_predict(u, clips[30]["x"], LEN, DEV))
            rows.append(scored({**base, "method": "unet_shift", "calib_s": 30, "seed": seed, "shift_ms": sh * 2},
                               test, apply_shift(p, sh)))
            for c in (10, 30):
                ft = finetune(u, [clips[c]], seed)
                pf = stitch_predict(ft, test["x"], LEN, DEV)
                rows.append(scored({**base, "method": "unet_finetune", "calib_s": c, "seed": seed,
                                    **(intervals(test, pf) if seed == 0 and c == 30 else {})}, test, pf))
        for name, models in pooled.items():
            for seed, u in enumerate(models):
                rows.append(scored({**base, "method": f"trained_{name}", "calib_s": 0, "seed": seed}, test,
                                   stitch_predict(u, test["x"], LEN, DEV)))
                ft = finetune(u, [clips[10]], seed)
                rows.append(scored({**base, "method": f"trained_{name}_finetune", "calib_s": 10, "seed": seed}, test,
                                   stitch_predict(ft, test["x"], LEN, DEV)))
        print(f"transfer {row.record} ({time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(path, index=False)


# ---------------------------------------------------------------- A3

def revisit(recs: pd.DataFrame):
    rows = []
    cebs = load_unets("holdout", "unet")
    c0 = CAL_START_S * FS
    for p, g in recs.groupby("patient"):
        if len(g) < 2:
            continue
        first, second = g.iloc[0].record, g.iloc[1].record
        d1, d2 = load(first), load(second)
        test = cut(d2, TEST_START_S * FS, (TEST_START_S + TEST_LEN_S) * FS)
        sources = {"same_visit": cut(d2, c0, c0 + 30 * FS), "previous_visit": cut(d1, c0, c0 + 30 * FS)}
        base = {"patient": p, "calib_record": first, "test_record": second}
        for src, clip in sources.items():
            rows.append(scored({**base, "source": src, "method": "avg_beat_own", "seed": None}, test, own_beat(clip, test)))
            for seed, u in enumerate(cebs):
                sh = best_shift(clip["y"], stitch_predict(u, clip["x"], LEN, DEV))
                rows.append(scored({**base, "source": src, "method": "unet_shift", "seed": seed},
                                   test, apply_shift(stitch_predict(u, test["x"], LEN, DEV), sh)))
                ft = finetune(u, [clip], seed)
                rows.append(scored({**base, "source": src, "method": "unet_finetune", "seed": seed}, test,
                                   stitch_predict(ft, test["x"], LEN, DEV)))
        print(f"revisit patient {p}: {first} -> {second}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "revisit.csv", index=False)


# ---------------------------------------------------------------- B5

class Win3(Dataset):
    def __init__(self, segs, length, stride, key):
        self.segs, self.length, self.stride, self.key = segs, length, stride, key
        self.set_epoch(0)

    def set_epoch(self, e):
        rng = np.random.default_rng(e)
        self.index = []
        for i, s in enumerate(self.segs):
            n = s["y"].shape[-1]
            off = int(rng.integers(0, self.stride))
            self.index += [(i, a) for a in range(off, n - self.length + 1, self.stride)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i, a = self.index[k]
        s = self.segs[i]
        x = s[self.key][..., a:a + self.length]
        x = x if x.ndim == 2 else x[None]
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(s["y"][a:a + self.length][None])


def train_axes(train, val, cin, seed, max_epochs=15, patience=3):
    key = "x3" if cin == 3 else "x"
    torch.manual_seed(seed)
    model = UNet1d(cin=cin).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ds = Win3(train, LEN, 1024, key)
    g = torch.Generator().manual_seed(seed)
    best, state, bad = -1.0, None, 0
    for epoch in range(max_epochs):
        ds.set_epoch(seed * 1000 + epoch)
        model.train()
        for xb, yb in DataLoader(ds, batch_size=32, shuffle=True, drop_last=True, generator=g):
            xb, yb = xb.to(DEV), yb.to(DEV)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
                loss = F.l1_loss(model(xb).float(), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        r = float(np.mean([pearson(v["y"], predict3(model, v[key] if cin == 3 else v["x"][None])) for v in val]))
        if r > best:
            best, state, bad = r, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state)
    return model.eval()


def axes(recs: pd.DataFrame):
    path = OUT / "axes.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {(r["fold"], r["cin"], r["seed"]) for r in rows}
    patients = np.array(sorted(recs.patient.unique()))
    rng = np.random.default_rng(20261119)
    groups = np.array_split(rng.permutation(patients), 4)
    c0 = CAL_START_S * FS
    for k, test_p in enumerate(groups):
        rest = np.array([p for p in patients if p not in set(test_p)])
        tune_p = rest[:max(2, len(rest) // 10)]
        train_p = rest[len(tune_p):]
        seg = lambda ps, a: [cut(load(r), a) for r in recs[recs.patient.isin(ps)].record]
        train = seg(train_p, 0)
        val = [cut(load(r), TEST_START_S * FS, (TEST_START_S + 120) * FS)
               for r in recs[recs.patient.isin(tune_p)].record]  # 2 min per tune recording for early stopping
        tests = recs[recs.patient.isin(test_p)]
        for cin in (1, 3):
            for seed in SEEDS:
                if (k, cin, seed) in done:
                    continue
                t0 = time.time()
                model = train_axes(train, val, cin, seed)
                key = "x3" if cin == 3 else "x"
                for row in tests.itertuples():
                    d = load(row.record)
                    test = cut(d, TEST_START_S * FS, (TEST_START_S + TEST_LEN_S) * FS)
                    pred = lambda m: predict3(m, test[key] if cin == 3 else test["x"][None])
                    base = {"fold": k, "cin": cin, "seed": seed, "record": row.record, "patient": row.patient}
                    rows.append(scored({**base, "method": "no_calibration"}, test, pred(model)))
                    clip = cut(d, c0, c0 + 10 * FS)
                    ft = finetune(model, [clip], seed, key=key)
                    rows.append(scored({**base, "method": "finetune_10s"}, test, pred(ft)))
                print(f"axes fold {k} cin {cin} seed {seed} ({(time.time() - t0) / 60:.1f} min)", flush=True)
                pd.DataFrame(rows).to_csv(path, index=False)


# ---------------------------------------------------------------- summary

def table(df, keys, order=None):
    pp = df.groupby(keys + ["patient"])[[c for c in METRICS if c in df]].mean().reset_index()
    t = pp.groupby(keys)[[c for c in METRICS if c in df]].mean().round(3)
    t.insert(0, "patients", pp.groupby(keys).patient.nunique())
    return t.reindex(order) if order is not None else t


def summarize():
    recs = pd.read_csv(OUT / "records.csv")
    lines = ["# SCG-RHC: heart-failure patients (generated by scripts/rhc.py)", "",
             f"{len(recs)} usable recordings of {recs.patient.nunique()} patients (fine_alignment, 30-150 bpm). Lead I "
             "target, patch dorsal-ventral axis input. Calibration from 10 s; test from 60 s to the end. Per patient "
             "first (first visit only unless stated), then averaged.", ""]
    tr = OUT / "transfer.csv"
    if tr.exists():
        df = pd.read_csv(tr)
        df = df[df.visit == 1]
        df["label"] = df.method + df.calib_s.map(lambda c: "" if c == 0 else f" {int(c)}s")
        order = ["unet", "unet_shift 30s", "avg_beat_own 10s", "avg_beat_own 30s", "unet_finetune 10s",
                 "unet_finetune 30s", "trained_cebs", "trained_sst", "trained_pooled", "trained_cebs_finetune 10s",
                 "trained_sst_finetune 10s", "trained_pooled_finetune 10s"]
        lines += ["## A2: CEBS models on patients", "", table(df, ["label"], order).to_markdown(), ""]
        irr = df.groupby("label")[["beat_r_regular", "beat_r_irregular"]].median().reindex(order[:6]).round(3)
        irr.insert(0, "recordings with >=3 irregular beats", df[df.seed.fillna(0) == 0].groupby("label")
                   .beat_r_irregular.count().reindex(order[:6]))
        lines += ["### Beat shape on regular vs irregular beats (median over recordings)", "", irr.to_markdown(), ""]
        cl = df[df.qrs_ms_beat_mae.notna()].groupby("label")[["qrs_ms_beat_mae", "qt_ms_beat_mae"]].median() \
            .reindex(["unet", "avg_beat_own 30s", "unet_finetune 30s"]).round(1)
        wide = df[(df.label == "avg_beat_own 30s")].qrs_ms_real
        lines += ["### Intervals (median per-beat error, ms; seed 0)", "", cl.to_markdown(), "",
                  f"Real QRS width: median {wide.median():.0f} ms; {int((wide >= 120).sum())} of {len(wide)} recordings "
                  "have QRS >= 120 ms (bundle branch block range).", "",
                  "## B4: does training on more sensors help on an unseen sensor?", "",
                  "trained_*: U-Nets of scripts/pooled.py (CEBS only / SensSmartTech only / both), same recipe.", ""]
        pp = df.groupby(["label", "patient"]).r.mean().unstack(0)
        for a, b in (("trained_pooled", "trained_cebs"), ("trained_pooled", "trained_sst"),
                     ("trained_pooled_finetune 10s", "trained_cebs_finetune 10s"),
                     ("unet_finetune 10s", "avg_beat_own 10s"), ("unet_finetune 30s", "avg_beat_own 30s")):
            if a in pp and b in pp:
                d = (pp[a] - pp[b]).dropna()
                lines.append(f"- {a} vs {b}: better for {(d > 0).sum()} of {len(d)} patients; mean r difference "
                             f"{d.mean():+.3f}")
        lines.append("")
    rv = OUT / "revisit.csv"
    if rv.exists():
        df = pd.read_csv(rv)
        t = table(df, ["method", "source"])
        pp = df.groupby(["method", "source", "patient"]).r.mean().unstack(1)
        lines += ["## A3: does a calibration survive a second procedure (another day, patch re-attached)?", "",
                  t.to_markdown(), ""]
        for m in ("unet_finetune", "avg_beat_own", "unet_shift"):
            d = (pp.loc[m]["previous_visit"] - pp.loc[m]["same_visit"]).dropna()
            lines.append(f"- {m}: previous-visit calibration minus same-visit: mean r {d.mean():+.3f}, lower for "
                         f"{(d < 0).sum()} of {len(d)} patients")
        lines.append("")
    ax = OUT / "axes.csv"
    if ax.exists():
        df = pd.read_csv(ax)
        df = df[df.record.isin(recs[recs.visit == 1].record)]
        t = table(df, ["cin", "method"])
        pp = df.groupby(["cin", "method", "patient"]).r.mean().unstack(0)
        lines += ["## B5: U-Nets trained on patients, 1 axis vs 3 axes (unseen patients, 4 folds)", "", t.to_markdown(), ""]
        for m in ("no_calibration", "finetune_10s"):
            d = (pp.loc[m][3] - pp.loc[m][1]).dropna()
            lines.append(f"- {m}: 3 axes minus 1 axis: mean r {d.mean():+.3f}, better for {(d > 0).sum()} of {len(d)}")
        lines.append("")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", nargs="+", default=["prep", "transfer", "revisit", "axes", "summary"],
                    choices=["prep", "transfer", "revisit", "axes", "summary"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if "prep" in args.part:
        recs = usable(record_list())
        recs.to_csv(OUT / "records.csv", index=False)
        print(f"{len(recs)} usable recordings, {recs.patient.nunique()} patients", flush=True)
    recs = pd.read_csv(OUT / "records.csv")
    if "transfer" in args.part:
        transfer(recs)
    if "revisit" in args.part:
        revisit(recs)
    if "axes" in args.part:
        axes(recs[recs.visit == 1])
    if "summary" in args.part:
        summarize()


if __name__ == "__main__":
    main()
