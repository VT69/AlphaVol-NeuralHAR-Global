"""Step 7 evaluation: hypothesis tests (DM + Holm), MCS, summary tables, experiment-log entries.

Reads analysis/artifacts/results/<asset>/forecasts/*.parquet and run.json written by
scripts/run_experiments.py. Writes analysis/artifacts/results/hypotheses.json, mcs_<asset>.csv,
analysis/tables/T_results_*.{csv,tex}; appends section-B rows to analysis/EXPERIMENT_LOG.md.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from alphavol.experiments import AssetRun, evaluate_hypotheses, load_grid, mcs_per_asset  # noqa: E402
from alphavol.logging_utils import configure  # noqa: E402

RES = ROOT / "analysis" / "artifacts" / "results"
TAB = ROOT / "analysis" / "tables"
LOG = ROOT / "analysis" / "EXPERIMENT_LOG.md"


def load_run(asset: str) -> tuple[AssetRun, dict]:
    meta = json.loads((RES / asset / "run.json").read_text())
    fc = {p.stem: pd.read_parquet(p) for p in sorted((RES / asset / "forecasts").glob("*.parquet"))}
    return AssetRun(asset, fc, {}, meta["tuning"], meta["timings"], meta["sample"]), meta


def append_log(runs_meta: dict, commit: str) -> int:
    text = LOG.read_text()
    rows = []
    for a, meta in runs_meta.items():
        for m, s in meta["summary"].items():
            rid = f"P-{a}-{m}"
            if f"| {rid} |" in text:
                continue
            rows.append(f"| {rid} | {date.today()} | pre-registered | see PREREGISTRATION.md | {m} "
                        f"(configs/experiments.yaml; code {commit[:7]}) | {a.upper()} sample {meta['sample']['sample_start']}"
                        f"→dev_end, OOS from {meta['sample']['oos_start']} (T={s['T']}) | QLIKE {s['QLIKE']:.4f}, "
                        f"MSE-log {s['MSE_log']:.4f} | see 05_results.md |")
    if rows:
        marker = "## B. Pre-registered experiments\n"
        text = text.replace("*(none yet; `PREREGISTRATION.md` must be committed first)*\n", "")
        header = ("| ID | Date | Status | Hypothesis | Config | Data | Result | Conclusion |\n"
                  "|---|---|---|---|---|---|---|---|\n")
        if "| ID | Date | Status | Hypothesis |" not in text:
            text = text.replace(marker, marker + "\n" + header)
        idx = text.index(header) + len(header)
        text = text[:idx] + "\n".join(rows) + "\n" + text[idx:]
        LOG.write_text(text)
    return len(rows)


def main() -> int:
    configure()
    grid = load_grid()
    runs, metas = {}, {}
    for a in ("btc", "spx", "nifty"):
        if (RES / a / "run.json").exists():
            runs[a], metas[a] = load_run(a)
    if not runs:
        raise SystemExit("no results found: run scripts/run_experiments.py first")
    hyp = evaluate_hypotheses(runs, grid)
    (RES / "hypotheses.json").write_text(json.dumps(hyp, indent=2, default=float))
    for a, r in runs.items():
        mcs_per_asset(r).to_csv(RES / f"mcs_{a}.csv")
        pd.DataFrame(metas[a]["summary"]).T.sort_values("QLIKE").to_csv(TAB / f"T_results_summary_{a}.csv")
    flat = [{"hypothesis": h, "asset": x["asset"], "A": x["A"], "B": x["B"], "kind": x["kind"], "T": x["T"],
             "QLIKE_pct_change": x["QLIKE"]["pct_change_B_vs_A"], "QLIKE_ci_lo": x["QLIKE"]["pct_change_ci95"][0],
             "QLIKE_ci_hi": x["QLIKE"]["pct_change_ci95"][1], "QLIKE_DM": x["QLIKE"]["stat"],
             "QLIKE_p": x["QLIKE"]["p_value"], "QLIKE_p_holm": x["QLIKE"].get("p_holm"),
             "MSE_DM": x["MSE"]["stat"], "MSE_p": x["MSE"]["p_value"],
             "MDE_mean_diff": x["QLIKE"]["mde_mean_diff_holm_worst"]} for h, xs in hyp.items() for x in xs]
    t = pd.DataFrame(flat)
    t.to_csv(TAB / "T_results_hypotheses.csv", index=False)
    (TAB / "T_results_hypotheses.tex").write_text(t.style.format(precision=4, na_rep="--").hide(axis="index").to_latex(
        hrules=True, caption="Pre-registered comparisons (DM with Newey--West HAC; Holm-adjusted p for tests).",
        label="tab:results_hypotheses"))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    n = append_log(metas, commit)
    print(f"hypotheses evaluated: {len(flat)} comparisons; experiment-log rows added: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
