"""Phase-1 audit: rebuild feature matrices from committed raw data into a temp dir and
compare with the committed data/processed/*_feature_matrix.parquet. Also records the
shape at every hop (loader -> matrix -> HAR design -> Neural-HAR tensors/splits).

Does not write inside the repo except analysis/artifacts/phase1_rebuild_check.json.
Usage: python scripts/audit_rebuild_check.py
"""
from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_pipeline.build_feature_matrix import build_feature_matrix  # noqa: E402
from src.data_pipeline.data_loader import ResearchDataLoader  # noqa: E402

RAW = ROOT / "data_collection" / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "analysis" / "artifacts" / "phase1_rebuild_check.json"
HAR = ["RV_d", "RV_w", "RV_m"]
EXO_HARS = ["vpin_lag1", "obi_sq_lag1", "illiq_lag1", "FinBERT_score_lag1", "sent_surprise_lag1",
            "credit_spread_lag1", "term_slope_lag1", "crypto_fg_lag1"]


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    res = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # sentiment files are inputs read from processed_dir: copy them in
        for p in PROC.glob("sentiment_daily_*.parquet"):
            (tmp / p.name).write_bytes(p.read_bytes())
        for a in ["btc", "spx", "nifty"]:
            base = ResearchDataLoader(str(RAW)).load(a)
            new = build_feature_matrix(a, str(RAW), str(tmp)).sort_index()
            old = pd.read_parquet(PROC / f"{a}_feature_matrix.parquet").sort_index()
            same_cols = list(new.columns) == list(old.columns)
            same_idx = new.index.equals(old.index)
            num = [c for c in old.columns if c != "regime" and c in new.columns]
            maxdiff = (float(np.nanmax(np.abs(new[num].to_numpy(float) - old[num].to_numpy(float))))
                       if same_idx and same_cols else None)
            df = old.dropna(subset=HAR + ["Target_RV"])
            n = len(df)
            exo = [c for c in EXO_HARS if c in df.columns]
            res[a] = {
                "loader_output_shape": list(base.shape),
                "rebuilt_shape": list(new.shape), "committed_shape": list(old.shape),
                "same_columns": same_cols, "same_index": same_idx, "max_abs_numeric_diff": maxdiff,
                "regime_equal": bool((new["regime"].astype(str) == old["regime"].astype(str)).all()) if same_idx else None,
                "HAR_design_shape": [n, len(HAR) + 1],
                "HARS_design_shape": [n, len(HAR) + len(exo) + 1],
                "HARS_rows_with_all_exo_nonnull": int(df[exo].notna().all(axis=1).sum()),
                "HAR_oos_forecasts": n - 252,
                "neural_exo_cols": exo,
                "neural_split_train_val_test": [int(n * 0.8), int(n * 0.1), n - int(n * 0.8) - int(n * 0.1)],
                "neural_test_start": str(df.index[int(n * 0.8) + int(n * 0.1)]),
                "neural_tensor_shapes": {"X_har": [n, 3], "X_exo": [n, len(exo)], "y": [n, 1]},
            }
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
