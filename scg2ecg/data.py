"""Loading CEBS records: anti-aliased resampling, beat annotations, filters, duplicate detection.

CEBS has 20 people, each with a ~5 min rest record (b), a ~50 min music record (m) and a ~5 min
after-music record (p): ECG leads I and II, respiration and one SCG axis, all at 5 kHz.

Two facts about the files shape everything here:
  * For subjects 2-20 the rest record is a byte-identical copy of the after-music record
    (b002.dat == p002.dat, ...). Only subject 1 has three distinct records. `usable_records` keeps
    one copy, so no signal can sit in both a training set and a test set.
  * The .atr files mark every beat twice: channel 0 is the lead I QRS mark, channel 1 the lead II mark.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import wfdb
from scipy.signal import butter, resample_poly, sosfiltfilt

from .paths import CACHE, RAW

FS_RAW = 5000
SUBJECTS = tuple(range(1, 21))
CHANNELS = ("I", "II", "RESP", "SCG")

ECG_BAND = (0.5, 40.0)  # Hz, for the reconstruction target
SCG_BAND = (1.0, 40.0)  # Hz, model / baseline input


@dataclass
class Record:
    name: str
    subject: int
    kind: str  # 'b' rest, 'm' music, 'p' after music
    fs: int
    ecg: np.ndarray  # lead I, mV
    ecg2: np.ndarray  # lead II, mV
    resp: np.ndarray
    scg: np.ndarray
    r_peaks: np.ndarray  # lead I QRS marks from the .atr file, sample indices at `fs`
    r_peaks2: np.ndarray  # lead II QRS marks

    @property
    def n(self) -> int:
        return len(self.ecg)

    @property
    def minutes(self) -> float:
        return self.n / self.fs / 60


def record_name(kind: str, subject: int) -> str:
    return f"{kind}{subject:03d}"


@lru_cache(maxsize=1)
def listed_hashes() -> dict[str, str]:
    """SHA-256 of every file, as published in SHA256SUMS.txt (the data check verifies the files against it)."""
    out = {}
    for line in (RAW / "SHA256SUMS.txt").read_text().splitlines():
        h, name = line.split(maxsplit=1)
        out[name.strip()] = h
    return out


def duplicate_of(name: str) -> str | None:
    """For a rest record whose signal file is identical to the after-music record, the name of that record."""
    if not name.startswith("b"):
        return None
    p = "p" + name[1:]
    h = listed_hashes()
    return p if h[f"{name}.dat"] == h[f"{p}.dat"] else None


def usable_records(subject: int) -> list[str]:
    """Chronological records with distinct signal data for one person."""
    recs = [record_name(k, subject) for k in "bmp"]
    return [r for r in recs if duplicate_of(r) is None]


def load(name: str, fs: int = 500) -> Record:
    """One record at `fs` Hz. Downsampling uses scipy's polyphase resampler, which low-pass filters first."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}_{fs}.npz"
    if not path.exists():
        rec = wfdb.rdrecord(str(RAW / name), return_res=32)
        if rec.fs != FS_RAW or list(rec.sig_name) != list(CHANNELS):
            raise ValueError(f"{name}: unexpected fs={rec.fs} channels={rec.sig_name}")
        x = rec.p_signal.astype(np.float64)
        if fs != FS_RAW:
            if FS_RAW % fs:
                raise ValueError(f"fs must divide {FS_RAW}")
            x = resample_poly(x, 1, FS_RAW // fs, axis=0)
        ann = wfdb.rdann(str(RAW / name), "atr")
        marks = []
        for ch in (0, 1):
            m = np.round(ann.sample[ann.chan == ch] * fs / FS_RAW).astype(np.int64)
            marks.append(m[(m >= 0) & (m < len(x))])
        np.savez(path, x=x.astype(np.float32), r1=marks[0], r2=marks[1])
    z = np.load(path)
    x = z["x"]
    return Record(name=name, subject=int(name[1:]), kind=name[0], fs=fs,
                  ecg=x[:, 0], ecg2=x[:, 1], resp=x[:, 2], scg=x[:, 3],
                  r_peaks=z["r1"], r_peaks2=z["r2"])


def bandpass(x: np.ndarray, lo: float, hi: float, fs: int, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass."""
    sos = butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, np.asarray(x, dtype=np.float64)).astype(np.float32)


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return ((x - x.mean()) / (x.std() + 1e-12)).astype(np.float32)


@lru_cache(maxsize=64)
def prepared(name: str, fs: int = 500) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(scg, ecg_target, r_peaks) for one record.

    scg: band-passed SCG (not normalised; each model normalises its own input windows).
    ecg_target: band-passed lead I, z-scored over the record. This is the scoring convention for every
    model: the target's scale comes from the target record itself, never from other people.
    """
    rec = load(name, fs)
    return bandpass(rec.scg, *SCG_BAND, fs), zscore(bandpass(rec.ecg, *ECG_BAND, fs)), rec.r_peaks
