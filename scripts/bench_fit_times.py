"""Code verification (EXPERIMENT_LOG E1): single-thread fit times on SYNTHETIC data of the real shape
(2,800 rows, 17 features). Output: analysis/artifacts/bench_fit_times.json."""
import time, sys, numpy as np, pandas as pd, torch
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
from test_models import synthetic
from alphavol.models.neural import NeuralHAR
from alphavol.models.gbm import GBM
from alphavol.models.linear import LinearQLIKE, LinearLog
from alphavol.models.garch import GARCH
torch.set_num_threads(1)
df = synthetic(n=2800)
for i in range(12): df[f"f{i}"] = np.random.default_rng(i).normal(size=len(df))
feats = ["RV_d","RV_w","RV_m","exo","exo_missing"]+[f"f{i}" for i in range(12)]
def t(f, n=1):
    s=time.perf_counter(); [f() for _ in range(n)]; return (time.perf_counter()-s)/n
res = {}
res["neural_grn_h32_l2_e60_s"] = t(lambda: NeuralHAR(feats, hidden=32, layers=2, epochs=60).fit(df))
res["neural_grn_h64_l3_e100_s"] = t(lambda: NeuralHAR(feats, hidden=64, layers=3, epochs=100).fit(df))
res["neural_tf_h32_l2_e60_s"] = t(lambda: NeuralHAR(feats, arch="transformer", hidden=32, layers=2, epochs=60).fit(df))
res["gbm_qlike_s"] = t(lambda: GBM(feats).fit(df))
res["linear_ols_s"] = t(lambda: LinearLog(feats).fit(df), 20)
res["linear_qlike_s"] = t(lambda: LinearQLIKE(feats).fit(df), 5)
res["garch_s"] = t(lambda: GARCH().fit(df), 3)
(ROOT / "analysis" / "artifacts" / "bench_fit_times.json").write_text(json.dumps(res, indent=2))
print(json.dumps({k: round(v, 3) for k, v in res.items()}, indent=1))
