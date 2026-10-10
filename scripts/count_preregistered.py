"""Count the pre-registered configurations and training runs in configs/experiments.yaml."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "experiments.yaml").read_text())
    seeds = len(cfg["seeds_neural"])
    out = {"model_configs_per_asset": {}, "neural_training_runs": 0, "configurations_total": 0,
           "dm_tests_planned": 0}
    for a in ("btc", "spx", "nifty"):
        models = {k: v for k, v in cfg["models"].items() if a in v.get("assets", ["btc", "spx", "nifty"])}
        out["model_configs_per_asset"][a] = len(models)
        out["configurations_total"] += len(models)
        out["neural_training_runs"] += seeds * sum(v["family"] == "neural" for v in models.values())
        for h in cfg["hypotheses"].values():
            out["dm_tests_planned"] += sum(1 for p in h["pairs"] if all(m in models for m in p))
            out["dm_tests_planned"] += sum(1 for p in h.get("also_report", []) if all(m in models for m in p))
    out["tuning_studies"] = 3 * 5          # assets x {GBM-QLIKE, GBM-MSE, GRN-QLIKE, GRN-MSE, TF-QLIKE}
    (ROOT / "analysis" / "artifacts" / "preregistered_counts.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
