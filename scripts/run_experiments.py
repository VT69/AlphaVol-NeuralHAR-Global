"""Step 7: run the pre-registered experiments (analysis/PREREGISTRATION.md, configs/experiments.yaml).

Usage:  python scripts/run_experiments.py [--assets btc spx nifty] [--workers 3]

Guards (refuses to run if any fails):
  * PREREGISTRATION.md and configs/experiments.yaml are committed and unmodified;
  * data/processed_v2/manifest.json lists no pending inputs;
  * the manifest's SHA-256 is recorded in analysis/DECISIONS.md (Step 5 sign-off).
Outputs: analysis/artifacts/results/<asset>/forecasts/*.parquet (incl. per seed), results.json; an
entry per model and asset is appended to analysis/EXPERIMENT_LOG.md (section B).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from alphavol import io as aio  # noqa: E402
from alphavol.config import load_config  # noqa: E402
from alphavol.experiments import evaluate_hypotheses, load_grid, mcs_per_asset, run_asset, summarize_forecasts  # noqa: E402
from alphavol.logging_utils import configure, get_logger  # noqa: E402

log = get_logger("run_experiments")
OUT = ROOT / "analysis" / "artifacts" / "results"


def guards() -> dict:
    dirty = subprocess.run(["git", "status", "--porcelain", "analysis/PREREGISTRATION.md", "configs/experiments.yaml"],
                           cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if dirty:
        raise SystemExit(f"pre-registration files have uncommitted changes:\n{dirty}")
    man_path = ROOT / "data" / "processed_v2" / "manifest.json"
    man = json.loads(man_path.read_text())
    pend = {a: v["pending_omitted"] for a, v in man["assets"].items() if v["pending_omitted"]}
    if pend:
        raise SystemExit(f"datasets still have pending inputs {pend}: finish Steps 3-5 first")
    sha = aio.sha256_file(man_path)
    if sha not in (ROOT / "analysis" / "DECISIONS.md").read_text():
        raise SystemExit(f"dataset manifest sha256 {sha} is not recorded in DECISIONS.md (Step 5 sign-off)")
    probe = ROOT / "analysis" / "artifacts" / "sentiment_probe.json"
    choice = json.loads(probe.read_text())["selection"]["choice"] if probe.exists() else None
    return {"manifest_sha256": sha, "common_start_year": (choice or {}).get("common_start_year")}


def _run(asset: str, common_start_year) -> str:
    import torch
    torch.set_num_threads(1)
    cfg = load_config()
    grid = load_grid()
    df = aio.read_frame(cfg.processed_dir / f"{asset}_dataset.parquet")
    t0 = time.time()
    r = run_asset(asset, df, grid, calendar=cfg.assets[asset].calendar, common_start_year=common_start_year)
    d = OUT / asset / "forecasts"
    for m, f in r.forecasts.items():
        aio.write_frame(f, d / f"{m}.parquet")
    for m, runs in r.per_seed.items():
        if len(runs) > 1:
            for s, f in runs.items():
                aio.write_frame(f, d / "per_seed" / f"{m}__seed{s}.parquet")
    summary = {m: summarize_forecasts(f) for m, f in r.forecasts.items()}
    seed_stats = {m: pd.DataFrame({s: summarize_forecasts(f) for s, f in runs.items()}).T.describe().loc[
        ["mean", "std", "min", "max"]].to_dict() for m, runs in r.per_seed.items() if len(runs) > 1}
    aio.write_json({"sample": r.sample, "summary": summary, "per_seed": seed_stats, "tuning": r.tuning,
                    "timings": r.timings, "wall_seconds": time.time() - t0}, OUT / asset / "run.json")
    return asset


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", nargs="+", default=["btc", "spx", "nifty"])
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    configure()
    g = guards()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for a in ex.map(_run, args.assets, [g["common_start_year"]] * len(args.assets)):
            log.info("finished %s", a)
    log.info("evaluation stage: run scripts/evaluate_results.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
