from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "files"            # CEBS 1.0.0 from PhysioNet (WFDB format)
CACHE = ROOT / "data" / "cache"          # resampled copies, rebuilt on demand
LEGACY_EDF = ROOT / "3-2Project-main" / "3-2Project-main" / "datasets and prev code"
RESULTS = ROOT / "results"
