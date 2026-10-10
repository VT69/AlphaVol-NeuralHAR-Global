# Legacy code path (frozen)

The following files produced the numbers in the original manuscript and are **kept unchanged for
provenance** (analysis/DECISIONS.md D10). They contain defects documented in `analysis/00_repo_map.md`,
`analysis/03_methodology_audit.md` and `analysis/paper_vs_code.md`, and **must not be used for new results**:

`run_paper.py`, `generate_tables.py`, `check_latex.py`, `src/`, `eval/`, `backtest/`, `notebooks/`,
`data_collection/sources/` (fetchers), `data/processed/`, `data/paper_tables/`, `data/*_results/`,
`requirements-legacy.txt`.

The rebuilt pipeline is the `alphavol/` package with tests in `tests/`, configuration in
`configs/pipeline.yaml`, and entry points in `scripts/`. Committed raw data under
`data_collection/data/raw/` is still used as development input, truncated at `dev_end`.
