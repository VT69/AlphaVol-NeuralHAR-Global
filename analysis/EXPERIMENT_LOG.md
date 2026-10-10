# Experiment log

Every experiment is logged here, including failures, dead ends and abandoned ideas. "Experiment" means any model fit or evaluation on **real** data whose result could influence a modelling decision. Pure code verification on synthetic data is listed separately in the last section and never counts as evidence.

Columns: ID · date · status (pre-registered / exploratory / audit / abandoned) · hypothesis · config · data · result · conclusion.

## Counts (kept current)

| Category | Count |
|---|---|
| Audit re-computations on pre-existing artefacts (Phases 1–2, Step 1) | see A-entries |
| Pre-registered experiments | 0 (grid fixed: 92 configurations, 120 neural runs; `PREREGISTRATION.md`) |
| Exploratory experiments | 0 |
| Abandoned | 0 |

## A. Audit computations on pre-existing artefacts (no new models fitted)

| ID | Date | Hypothesis / question | Config | Data | Result | Conclusion |
|---|---|---|---|---|---|---|
| A1 | 2026-10-09 | Does a causal bias correction of HAR close the QLIKE gap to Neural-HAR? | Smearing (Duan 1983) on committed HAR forecasts, residuals ≤ t−2 | committed `forecasts_har_*.npy`, paper trailing windows | SPX 1.455 vs NN 1.567; NIFTY 1.224 vs 1.870; BTC 0.308 vs 0.287 | The headline QLIKE gain is mostly level calibration (`scripts/audit_qlike_bias_check.py`) |
| A2 | 2026-10-09 | Is the SPX/NIFTY target forecastable? | ACF, Ljung–Box, Yule–Walker / signal-share ceilings, in-sample HAR OLS R² | committed RV files, all rows | Linear R² ceiling 0.095 / 0.048 | Target ≈ log squared returns, mostly noise (`scripts/target_quality.py`) |
| A3 | 2026-10-09 | Which code path produced `table5_mcs.csv`? | Match committed HAR mean loss against alignment candidates | committed npy | Exact match to the front-aligned bug for all 3 assets | Paper's main MCS table invalid (`scripts/audit_paper_claims.py`) |
| A4 | 2026-10-10 | Data profile incl. naive baselines on the old test window | Train-mean / random-walk / RV_m forecasts | feature matrices, old test split | SPX train-mean QLIKE 2.750 < paper HAR 3.897 | `scripts/profile_data.py` |
| A5 | 2026-10-10 | Do the metrics match references, and does the training loop behave as described? | Hand cases; OLS reference for DM; `arch.bootstrap.MCS` (R, circular, block 5); MCS size Monte-Carlo (400 reps, B=300, T=300); LR-schedule replay; clamp gradient | synthetic + committed HAR/HAR-S npy | Metrics correct; MCS inclusion matches arch; both over-reject (16.25 % at nominal 10 %); fine-tune LR = 1e-6; 91–96 % SPX/NIFTY predictions in zero-gradient clamp region | Training protocol defective; metric code sound (`scripts/audit_methodology_checks.py`) |
| A6 | 2026-10-10 | How much do the Step 2 fixes change target quality? (interim Step 5) | ACF, Ljung–Box, YW ceiling, signal share, in-sample HAR OLS R² (descriptive; no OOS evaluation) | `data/processed_v2` (dev, ≤ 2026-07-28) vs legacy | In-sample HAR R²: BTC 0.402 → 0.547; SPX 0.090 → 0.508; NIFTY 0.041 → 0.349. YW ceiling: SPX 0.095 → 0.515; NIFTY 0.048 → 0.364 | The GK target makes SPX/NIFTY forecastable; the lag fix adds +0.13 (BTC) in-sample R² (`scripts/target_quality_v2.py`) |

**Note for the paper:** A1–A4 were computed on the old test window (2025-06 → 2026-07). That window is now part of development (R4), so these numbers are **not** held-out evidence.

## B. Pre-registered experiments

*(none yet; `PREREGISTRATION.md` must be committed first)*

## C. Exploratory experiments

*(none yet)*

## D. Abandoned ideas / dead ends

*(none yet)*

## E. Code verification on synthetic data (not evidence)

| ID | Date | Purpose | Script | Result |
|---|---|---|---|---|
| E1 | 2026-10-10 | Fit-time benchmark for the compute plan | `scripts/bench_fit_times.py` → `analysis/artifacts/bench_fit_times.json` | GRN 4.3–7.1 s, Transformer 22.1 s, GBM 0.34 s, OLS 0.008 s per fit (single thread) |
| E2 | 2026-10-10 | Unit/plumbing tests (all synthetic or committed dev data; no model evaluated on real data) | `pytest` (77 tests) | all pass |
