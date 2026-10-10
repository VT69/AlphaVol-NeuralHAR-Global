# Paper vs. code: claims in `paper/main.tex` that the code does not support

**Treat every row marked BLOCKER as a submission blocker.** Scope: `main.tex` plus everything it `\input`s (sections 00–09, appendix, 8 tables). Line numbers refer to the `.tex` source (most paragraphs are one long line) and to the real files under the repo root.

Evidence scripts (run in this session):
`scripts/audit_trace_pipeline.py`, `scripts/audit_rebuild_check.py`, `scripts/audit_qlike_bias_check.py`, `scripts/audit_sentiment_coverage.py`, `scripts/target_quality.py`, `scripts/audit_paper_claims.py`. Their outputs are in `analysis/artifacts/*.json`.

Severity: **BLOCKER** = the claim is false or unverifiable and a reader would draw a wrong conclusion. **MAJOR** = materially misdescribes the method. **MINOR** = imprecise wording.

---

## A. Results the paper reports that cannot be reproduced or are computed wrongly

| # | Claimed text (location) | Actual behaviour | Evidence | Sev. |
|---|---|---|---|---|
| A1 | Every Neural-HAR number: DM stats, MCS, OOS losses, backtest (`05_results.tex`, `07_…`, abstract, conclusion) | **Neural-HAR forecasts and weights are not in the repository.** No Neural-HAR number can be regenerated from committed artefacts. `train_neural_har.py` sets **no seeds** (torch, numpy, Optuna sampler), so retraining will not reproduce them either. | `ls data/processed` (no `forecasts_neural_har_*.npy`; no `data/models/`); `train_neural_har.py:136` (`optuna.create_study` unseeded); no `manual_seed` in the file | BLOCKER |
| A2 | "the exact weights of the out-of-sample models evaluated in this study … available in the public repository" (`appendix.tex:2`) | Not present. `.pth` files would go to `data/models/`, which does not exist. | `train_neural_har.py:45, 328-331` | BLOCKER |
| A3 | "For SPX, the Neural-HAR model is the sole survivor; the baseline HAR is eliminated (MCS p-value = 0.0, `table5_mcs.csv`). For BTC and NIFTY both HAR and Neural-HAR survive" (`05_results.tex:57-58`; also `09_conclusion.tex:5`, abstract) | `table5_mcs.csv` (`tab:mcs`) was produced by the **front-aligned buggy code path** that the paper itself says was fixed. The committed HAR mean losses match *exactly* the QLIKE of `HAR[:n]` vs the first `n` entries of `actuals_{asset}.npy` (HAR-S actuals from a different period): BTC 0.71331, SPX 5.40735, NIFTY 2.13741. The correct end-aligned HAR losses are 0.31840 / 3.89658 / 2.33046. The table compares forecasts and outcomes from **different dates**. | `audit_paper_claims.py → table5_mcs_HAR_alignment.exact_match = front_aligned…` for all three assets | BLOCKER |
| A4 | Caption "MCS: Neural-HAR vs. HAR … Trailing Window N≈275–287" (`tables/table5_mcs.tex`) | As A3: not a trailing window. `tab:mcs` and `tab:mcs_full` also contradict each other on NIFTY (HAR survives in one, is eliminated at p = 0.026 in the other) because they are computed on different data. | `table3_mcs.csv` vs `table5_mcs.csv` | BLOCKER |
| A5 | "Neural-HAR … QLIKE improvements over the HAR baseline that are statistically significant … BTC (DM = +1.98, p = 0.048) and SPX (DM = +5.05) … NIFTY (DM = +2.44)" (`05_results.tex:30-34`; abstract; `01_introduction.tex:9`; `09_conclusion.tex:3`) | The numbers match `table2_dm_test.csv`, but (i) they come from the best of **three** architectures, each evaluated on the same test window (`data/{deep_grn,transformer,ensemble}_results`: SPX QLIKE DM +0.23 / +5.04 / +5.05; BTC −0.27 / +0.83 / +1.98); (ii) a zero-parameter causal bias correction of HAR beats the reported Neural-HAR QLIKE on SPX (1.455 vs 1.567) and NIFTY (1.224 vs 1.870). The improvement is not attributable to sentiment or non-linearity. | `audit_qlike_bias_check.py`; committed variant CSVs | BLOCKER |
| A6 | "The full-sample OOS metrics for the baseline models are reported in Table~\ref{tab:oos_full}" (`05_results.tex:24`); "Regime-conditional OOS forecast accuracy is provided in Table~\ref{tab:regime_oos}" (`06_regime_analysis.tex:5`) | Neither table is `\input` into `main.tex`. **The submitted `paper/main.pdf` prints "Table ??" twice** (PDF text lines 112 and 145). `check_latex.py` reports 0 errors because it scans every `.tex` file on disk, not the ones actually included. | `audit_paper_claims.py → main_pdf_unresolved_refs`; `grep \input main.tex sections/*.tex` | BLOCKER |
| A7 | "the harmonized trailing-window metrics for all three models are in Table~\ref{tab:oos_metrics}" (`05_results.tex:24`); caption "All Models on Harmonised Trailing Window" (`tables/table3_oos_metrics.tex`) | The table contains **only HAR rows**, with N_OOS = 1,119 / 751 / 716. These are the HAR-S forecast counts, not the 275/287/276 trailing window. | `data/paper_tables/table3_oos_metrics.csv` | BLOCKER |
| A8 | "Throughout this paper we rely exclusively on the output of `run_paper.py`, which has always performed correct end-alignment" (`05_results.tex:51`) | False on two counts. (i) Git history: commit `e4527f0` ("fix: align all OOS forecasts to the chronological end…") changed `run_paper.py` *from* `[:n_oos]` front-alignment to `[-min_n:]`. (ii) The paper's own `tab:mcs` comes from `notebooks/paper_tables.py`, not `run_paper.py` (see A3). | `git show e4527f0 -- run_paper.py` | BLOCKER |

## B. Data claims

| # | Claimed text (location) | Actual behaviour | Evidence | Sev. |
|---|---|---|---|---|
| B1 | "for the equity indices, a realized volatility proxy is constructed from daily OHLCV data" (`03_data.tex:3`); the code docstring says Garman–Klass is primary (`compute_rv.py:47-58`) | The SPX/NIFTY target is **log of the squared daily close-to-close return**: `log_RV == log(RV)` and `RV == log_ret²` exactly (max abs diff 0.0). GK is in the file but unused. The committed RV parquet has a different schema from what the current `compute_tradfi_rv` writes, so it was produced by an uncommitted code version. AC(1) = 0.118 (SPX), 0.073 (NIFTY). | `target_quality.py → log_rv_identity`; `analysis/target_quality.md` | BLOCKER |
| B2 | "cross-asset evaluation using high-frequency Bitcoin (BTC), the S&P 500 (SPX), and the NIFTY 50 index" (`00_abstract.tex:2`) | Only BTC uses intraday (5-min) data. SPX/NIFTY use one daily close. | as B1 | MAJOR |
| B3 | "sentiment is measured via a daily aggregate FinBERT score derived from news headlines" (`03_data.tex:9`) and the paper's research question | FinBERT is **0 on 98.5 % (BTC), 92.3 % (SPX), 92.3 % (NIFTY) of rows**. BTC has 0 non-zero rows in train and val (all 40 are in test); SPX 53/2,289 and NIFTY 15/2,204 in train. NIFTY "sentiment" is HDFC Bank ADR news (`tickers=HDB`; 97.5 % of articles mention HDFC). | `audit_sentiment_coverage.py`; `analysis/sentiment_coverage_diagnosis.md`; `fetch_news_extended.py:71` | BLOCKER |
| B4 | "Market microstructure is captured via VPIN, Order Book Imbalance (OBI), and Amihud Illiquidity" (`03_data.tex:9`); "digest non-linear sentiment and microstructure interactions" (`01_introduction.tex:7`) | `illiq_lag1` is an **all-zero placeholder** for every asset (`data_loader.py:220-222`; `illiq` is never lagged, `:214`). SPX/NIFTY have **no** microstructure features at all (no VPIN/OBI columns). For BTC, OBI and VPIN are proxies from 5-min klines (close-location value; bulk-volume classification), not order-book data. | `phase1_trace.json → exo_frac_zero.illiq_lag1 = 1.0`; feature-matrix columns; `compute_microstructure.py:7-74` | BLOCKER |
| B5 | `table2_insample_coefs` reports `illiq_lag1` with p = 0.095 (BTC) and similar values for SPX/NIFTY (`tables/table2_insample_coefs.tex`) | A p-value for an identically-zero regressor is numerically meaningless (pseudo-inverse artefact). | as B4 | MAJOR |
| B6 | "Macroeconomic controls include the term slope and credit spread" (`03_data.tex:9`) | Credit spread exists only from **2023-05-17**. HAR-S drops ~60–74 % of training rows for it (N = 1,095 / 755 / 719 vs 2,744 / 2,862 / 2,755). Neural-HAR fills the gap with 0 (`train_neural_har.py:232`), turning it into a pre/post-2023 time indicator. | `phase1_trace.json → exo_first_valid`; `audit_rebuild_check.py → HARS_rows_with_all_exo_nonnull` | MAJOR |
| B7 | "retail fear/greed indices" (`03_data.tex:9`) | Only the crypto index, for BTC. The CNN equity F&G file has 0 rows. | `phase1_trace.json → raw.fear_greed_index.parquet = [0,0]` | MINOR |

## C. Method claims

| # | Claimed text (location) | Actual behaviour | Evidence | Sev. |
|---|---|---|---|---|
| C1 | HAR: log(RV_t) = α + β_d log(RV_{t−1}) + β_w … (`04_methodology.tex:7`), i.e. a 1-step-ahead model | The target is `log_RV.shift(-1)` and `RV_d = log_RV.shift(1)`, so the code predicts RV_{t+1} from RV_{t−1}, … **and never uses RV_t**. It is a 2-step-ahead HAR. This costs BTC 13 points of in-sample R² (0.402 vs 0.535). The same skip applies to every `_lag1` exogenous feature. | `data_loader.py:98-100`, `compute_rv.py:38-40, 95-97`, `build_feature_matrix.py:124-130`; `target_quality.json` | BLOCKER |
| C2 | "we perform a regime-conditional analysis using a Hidden Markov Model" (`01_introduction.tex:7`); "we filter the data into low, medium, and high volatility regimes using a Hidden Markov Model" (`06_regime_analysis.tex:3`) | **No HMM exists.** Regimes are full-sample `pd.qcut` terciles of `credit_spread − term_slope` (a stress proxy, not volatility), computed on post-2023-05-17 data only. `.ffill().bfill()` then labels **every earlier day "high"** (BTC: 1,933 of 2,744 rows, from 2019-01-23). The terciles use future data (look-ahead), and the regime is contemporaneous, not lagged. | `data_loader.py:200-205`; `phase1_trace.json → regime_first_date` | BLOCKER |
| C3 | Section 6 conclusions: "no robust evidence that sentiment's … content is strictly regime-dependent"; abstract's last sentence | Built on C2 (regimes ≈ "before vs after 2024") and B3 (sentiment ≈ always 0). The test cannot detect the effect it claims to rule out. | C2, B3 | BLOCKER |
| C4 | "a residual neural network takes both the HAR lags and the exogenous features X_{t−1} as input" (`04_methodology.tex:16`) | The network receives **only** `x_exo`. The HAR lags enter only the linear prior. | `neural_har.py:113-117` | MAJOR |
| C5 | "The model computes a linear OLS HAR prior" (`04_methodology.tex:16`) | The HAR layer is *initialised* from OLS (fit on the train split only) and then **trained jointly by gradient descent on QLIKE**, so it is no longer an OLS prior. This is what lets it absorb the log-χ² level bias (A5). | `train_neural_har.py:264, 285`; `neural_har.py:119-123` (weights remain `requires_grad`) | MAJOR |
| C6 | "the walk-forward routine utilizes … an initial minimum training window of 252 days" (`04_methodology.tex:23`) | The first training window is the first 80 % of rows (2,195 / 2,289 / 2,204), with validation on the next 10 %. 252 is the HAR baseline's `min_train`, not Neural-HAR's. | `train_neural_har.py:210-211, 236-238`; `audit_rebuild_check.py` | MAJOR |
| C7 | "At each step, Optuna optimizes the hyperparameters … over 40 trials" (`04_methodology.tex:23`) | Optuna runs **once**, on the initial train/val split, with 20 trials per architecture (`n_trials // 2`). Walk-forward steps only fine-tune 5 epochs with fixed hyperparameters. | `train_neural_har.py:267-276, 189-202` | MAJOR |
| C8 | GRN "Utilizes Gated Linear Units (GLU) to perform soft feature selection" (`04_methodology.tex:19`) | The GLU gates *hidden units* after a shared `Linear(input_dim, hidden)`. There is no per-feature variable-selection network (as in TFT), so "feature selection" is not implemented. | `neural_har.py:18-38` | MINOR |
| C9 | "Feature Tokenizer Transformer (FT-Transformer) and Gated Residual Networks \cite{gorishniy2021revisiting}" (`02_literature.tex:7`) | GRNs come from the Temporal Fusion Transformer (Lim et al., 2021), not Gorishniy et al. There is no `lim2021` entry in `references.bib`. | `paper/references.bib` | MINOR |
| C10 | "These models are estimated via OLS utilizing HAC standard errors" (`04_methodology.tex:13`) | True for the in-sample tables (`paper_tables.py:178-180`, `regime_har.py:69`). The OOS expanding window uses plain OLS (`baseline_har.py:78`). That is fine for point forecasts, but the sentence reads as if it covers both. | — | MINOR |
| C11 | DM test "with the HLN small-sample correction" (`04_methodology.tex:29`) | Implemented with `h = 1` (no autocovariance terms). Because of C1 the forecasts are effectively 2-step, so loss differentials are serially correlated and an HAC (h = 2) variance is required. | `run_paper.py:177`, `dm_test.py:194` (`h=1`) | MAJOR |

## D. Economic-significance claims

| # | Claimed text (location) | Actual behaviour | Evidence | Sev. |
|---|---|---|---|---|
| D1 | "We assume a Quarter-Kelly position sizing framework" (`07_economic_significance.tex:3`) | `kelly_fraction` is passed through but **never used**: `mu_hat` is always `None`, so the weight is pure `target_vol / σ̂`. | `vol_targeting.py:54-59, 81` | BLOCKER (false method statement) |
| D2 | "Net of the 5 bps slippage, this turnover erosion degrades the risk-adjusted return" (`07_…:5`); "the alpha generated by the Neural-HAR model is fully consumed by its turnover" (`08_…:7`) | Annualised slippage from the committed turnover: SPX Neural-HAR 4.61 % vs HAR 1.18 %, a **3.43 pp** difference against a **34.6 pp** return gap. NIFTY: 5.47 pp cost difference vs a 9.37 pp gap. BTC: Neural-HAR has **lower** cost (−0.38 pp) and **higher** return (+7.61 pp). Turnover does not explain the Sharpe gap. Lower leverage does: a higher vol forecast (A5's level shift) mechanically lowers exposure, return and drawdown together. | `audit_paper_claims.py → turnover_cost_attribution` | BLOCKER |
| D3 | "By providing more accurate volatility forecasts, the Neural-HAR strategy produces lower realized annualized volatility … and smaller maximum drawdowns" (`07_…:7`) | The numbers match `table6_backtest.csv`, but the causal attribution to accuracy is unsupported (see D2, A5). A uniformly higher forecast produces the same effect. Also unverifiable: Neural-HAR forecasts are absent (A1). | — | MAJOR |
| D4 | "inversely to the 1-day ahead volatility forecast" (`07_…:3`) | Day *t*'s position uses the forecast of RV_{t+1}, built from data through *t−1* (C1). There is no look-ahead, but the horizon is mismatched by one day. | `run_paper.py:325-332`; C1 | MINOR |

## E. Framing claims that depend on the items above

| # | Claimed text (location) | Problem | Sev. |
|---|---|---|---|
| E1 | "While the HAR-S model improves in-sample R̄² modestly" (`05_results.tex:6`) | The paper's own table shows HAR-S R² **lower** than HAR for every asset: BTC 0.1873 vs 0.4037, SPX 0.0339 vs 0.0885, NIFTY 0.0305 vs 0.0399. It is also estimated on 27–40 % of the sample (N = 1,095 / 755 / 719), so the two are not comparable. | BLOCKER |
| E2 | "FinBERT coefficients for BTC are not identified in the HAR-S regime splits because the BTC news-sentiment series is unavailable for a sufficient in-sample window in those partitions" (`05_results.tex:10-12`) | Understated: BTC sentiment is zero on **every** training row, in every partition and in the full sample. | MAJOR |
| E3 | "The non-linear Neural-HAR model, however, isolates predictive signal" (`01_introduction.tex:9`); "when these features are processed through a non-linear deep learning architecture … we observe a statistically significant improvement" (`09_conclusion.tex:3`) | No ablation (Neural-HAR with exogenous inputs removed) exists. With B3/B4 the exogenous inputs are mostly zeros and a time indicator, and A5 shows the gain is reproducible without them. | BLOCKER |
| E4 | "This window represents a recent, relatively calm market environment" (`08_robustness_limitations.tex:5`) | Partly supported. Test-window mean log variance is lower than train for all three, but significantly so only for BTC (KS 0.209, p = 7.7e−10). SPX (p = 0.032) and NIFTY (p = 0.021) are not significant at 1 % (`profile_data.py → integrity.drift_train_vs_test`). Reword to "calmer than the training period for BTC". | MINOR |
| E5 | `CHANGELOG_vs_internal_notes.md` §6: "Re-running that notebook will now produce results consistent with `run_paper.py`" | The code was fixed; the committed `table5_mcs.csv` used in the paper was never regenerated (A3). | BLOCKER (process) |

---

### Count

| Severity | Count |
|---|---|
| BLOCKER | 19 (A1–A8, B1, B3, B4, C1, C2, C3, D1, D2, E1, E3, E5) |
| MAJOR | 10 |
| MINOR | 6 |

The blockers fall into four groups:
1. **Unreproducible or misaligned results:** A1–A4, A7, A8, E5.
2. **The target and inputs are not what the paper describes:** B1, B3, B4, C1, C2.
3. **The headline effect is explained by a level-bias confound** that the paper does not control for: A5, D2, E3.
4. **Document-integrity errors** a reviewer will see on page one: A6, D1, E1.
