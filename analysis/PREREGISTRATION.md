# Pre-registration: evaluation protocol

**Status:** committed **before any forecasting model has been trained or evaluated on real data** in the rebuilt pipeline. The only real-data computations so far are audit and descriptive ones (`EXPERIMENT_LOG.md` §A: A1–A6). None of them evaluated a candidate model out of sample on the rebuilt data.
**Date:** 2026-10-10. **Code at registration:** `alphavol/` as of commit `33d4be9` (models, harness, tuning, metrics; all tested on synthetic data only).
**Machine-readable companion:** `configs/experiments.yaml` lists every configuration that will be run. The experiment runner refuses configurations not listed there unless they are tagged `exploratory`.
**Amendments:** allowed only as dated, committed sections at the end of this file, **before** the amended analysis is run. Anything not pre-registered is labelled *exploratory* in the log and in the paper.

---

## 1. Research questions and hypotheses

| ID | Question | H0 | H1 (two-sided test) |
|---|---|---|---|
| **H1 (primary)** | Does daily news sentiment improve 1-day-ahead volatility forecasts **beyond the HAR model**? | HAR-BC + SENT and HAR-BC have equal expected QLIKE loss | Expected QLIKE differs |
| H2 (R1) | Does Neural-HAR beat a **bias-corrected** HAR? | Neural-HAR (GRN, QLIKE, ALL features, seed-averaged) and HAR-BC have equal expected QLIKE | Differs |
| H3 | Does sentiment add information **beyond the other covariates**, for each model family? | Model(ALL) and Model(ALL − SENT) have equal expected QLIKE, for Linear-BC, GBM-QLIKE and Neural-GRN-QLIKE | Differs |
| H4 | Is the QLIKE advantage of QLIKE-trained models explained by **level calibration**? | Within each family, (a) the QLIKE-trained model and (b) the log-MSE model **with** smearing have equal expected QLIKE | Differs. Also reported: (c) log-MSE **without** smearing vs (b) |
| H5 | Does each non-sentiment covariate group improve on HAR? | HAR-BC + group G and HAR-BC have equal expected QLIKE (G ∈ macro, implied vol, fear & greed, microstructure, news attention) | Differs |

Unit of analysis: each asset separately (BTC, SPX, NIFTY). Cross-asset pooling is not used for inference.

## 2. Data

* **Development data:** `data/processed_v2/<asset>_dataset.parquet`, rebuilt by `scripts/build_dataset.py` once the author's local fetch (Steps 3–4) has completed. The rebuilt manifest's SHA-256 is recorded in `DECISIONS.md` before Step 7 runs. Rows with target date ≤ 2026-07-28 only.
* **Targets:** log daily variance of the next trading day. BTC: 5-minute realized variance. SPX/NIFTY: Garman–Klass (R2).
* **Sentiment sample S.** The common period selected by the pre-specified probe rules (D25/D28/D33): rows from 1 January of the selected `common_start_year` (and at least 30 days after the asset's first data, for HAR warm-up) to `dev_end`. **All H1–H5 comparisons use the same rows in S for both models**, so model differences are never confounded with sample differences.
  * If the probe selects no admissible source set, H1/H3 are **untestable** and reported as such. H2/H4/H5 then run on the full sample.
  * An asset left without admissible sources is dropped from H1/H3 only.
* **Full sample (secondary, context):** H2, H4 and the naive/HAR baselines are additionally reported on all development rows (2015 → for SPX/NIFTY, 2019 → for BTC), with the same evaluation design.
* **Sealed holdout:** rows with target date > 2026-07-28 (Step 4). Evaluated once, in Step 9 (§8).

### 2.1 Feature sets (fixed)
Per-row transforms only; no estimated parameters. All are defined in `configs/experiments.yaml → feature_sets`.

| Set | Columns |
|---|---|
| HAR | RV_d, RV_w, RV_m |
| SENT (primary sentiment) | sent_mean_score (FinBERT mean P(pos) − P(neg) of the day's articles; NaN on no-news days → training-mean imputation), sent_has_news |
| ATTN (news attention) | log1p(sent_n_articles), log1p(sent_gdelt_volume) (if GDELT selected) |
| MACRO | term_slope, credit_spread (BAA10Y), macro_missing |
| IV (SPX, NIFTY) | iv_logvar = 2·log(IV/100) − log(252) for VIX (SPX) or India VIX (NIFTY), implied_vol_missing |
| FG (BTC) | fear_greed, fear_greed_missing |
| MICRO (BTC) | ofi_abs, vpin, log(amihud), roll, microstructure_missing |
| ALL | HAR ∪ SENT ∪ ATTN ∪ every non-sentiment group available for the asset |
| ALL − SENT | ALL without SENT and without ATTN |

## 3. Evaluation design

**Expanding-window pseudo-out-of-sample evaluation** over the development period, chosen over a rolling window because:

* the HAR coefficients are stable enough that discarding old data mainly adds variance;
* expanding windows are the convention in the HAR literature (Corsi 2009; Patton & Sheppard 2015), which makes comparisons with prior work direct;
* it uses every observation, which matters given the limited sentiment history.

* **Initial training window:** the first **3 calendar years** of the sample (S or full). Out-of-sample (OOS) forecasts cover every subsequent row up to `dev_end`.
* **Forecast origin rule (D34):** the model used at origin t is trained only on rows s < t, and the targets of forecast rows are masked (`alphavol/evaluation.py`, tested).
* **Refit schedules:** linear and naive, every origin; HAR-QLIKE and GARCH, every 21 sessions (BTC: 30 days); GBM and neural, from scratch every 63 sessions (BTC: 91 days) with a fresh optimizer.
* **Preprocessing:** training-window statistics only (imputation, standardization).
* **Sub-period reporting (descriptive, no additional tests):** OOS by calendar year, and separately the old test window 2025-06-05 → 2026-07-28 (now a second validation period, R4).

## 4. Metrics

* **Primary metric:** mean QLIKE on the variance scale (Patton 2011), computed with `alphavol.metrics.qlike`.
* **Primary comparison:** H1, i.e. HAR-BC + SENT vs HAR-BC (both `LinearLog(bias="smearing")`), per asset.
* **Secondary metrics:** MSE of log variance; MAE of log variance; Mincer–Zarnowitz R² of log forecasts; percentage change in mean QLIKE relative to the comparison model.

## 5. Statistical tests

* **Diebold–Mariano** on per-period loss differentials (`alphavol.metrics.dm_test`): Newey–West (Bartlett) HAC variance with lag L = ⌊4·(T/100)^{2/9}⌋ (Newey & West 1994; T = number of OOS forecasts), the HLN factor with h = 1, a Student-t(T−1) reference, two-sided.
* **Effect size:** % change in mean QLIKE, with a 95 % stationary-bootstrap CI (Politis & Romano 1994; mean block length L, 5,000 replications, seed 0).
* **Multiple comparisons:** Holm across the three assets within each hypothesis (H1 is the confirmatory family). Within H3, H4 and H5, Holm across assets × model families (H3, H4) or assets × groups (H5). Unadjusted p-values are always reported as well.
* **Model Confidence Set** (secondary): `arch.bootstrap.MCS`, method R, circular block bootstrap, block length ⌈T^{1/3}⌉, 5,000 replications, α = 0.10, seed 0, on QLIKE. All models in `configs/experiments.yaml` for the asset are included, with p-values for every model. The Step-1 finding that the MCS over-rejects at T ≈ 300 (16 % vs 10 %) is disclosed. MCS is supporting evidence only.
* **Power:** for every DM test, the minimum detectable mean-QLIKE difference at 80 % power, (z_{1−α/2} + z_{0.8})·SE(d̄), is reported, using the Holm-worst α = 0.05/3 for H1.

## 6. Models, baselines and ablations (complete list; see `configs/experiments.yaml`)

| Family | Variants | Loss | Seeds |
|---|---|---|---|
| Naive | Random walk (log v_t); historical mean (QLIKE-optimal constant) | — | — |
| HAR | HAR-OLS (no correction); **HAR-BC** (smearing); HAR-QLIKE; HARQ (BTC only) | log-MSE / QLIKE | — |
| GARCH | GARCH(1,1)-t with causal log-calibration | QMLE | — |
| Linear HAR-X (BC) | HAR+SENT, HAR+ATTN, HAR+MACRO, HAR+IV / HAR+FG, HAR+MICRO, ALL, ALL−SENT | log-MSE + smearing | — |
| GBM | HAR, HAR+SENT, ALL−SENT, ALL (QLIKE); ALL−SENT, ALL (log-MSE + smearing) | gamma (= QLIKE) / log-MSE | 1 (deterministic) |
| Neural-HAR GRN (**primary neural**) | HAR, HAR+SENT, ALL−SENT, ALL (QLIKE); ALL−SENT, ALL (log-MSE + smearing) | QLIKE / log-MSE | 5 (0–4) |
| Neural-HAR FT-Transformer | ALL−SENT, ALL | QLIKE | 5 |
| Neural-HAR ensemble | GRN + Transformer, inverse inner-validation-loss weights (as in the original paper), ALL−SENT, ALL | QLIKE | built from the runs above |

* The **primary neural architecture is GRN**, fixed here a priori (simplest; the original paper's ensemble was chosen after test evaluation, audit V1). Transformer and ensemble are reported as variants, never substituted for the primary.
* **Neural results:** the seed-averaged forecast (mean of log forecasts over seeds 0–4) is the model that enters the tests. The per-seed distribution of every metric and DM statistic is reported (mean ± sd, min, max). The best seed is never reported as a result.
* **Hyperparameter tuning (equal budget, D34):** Optuna TPE (seed 0), **40 trials** for each of GBM-QLIKE, GBM-MSE, GRN-QLIKE, GRN-MSE and Transformer-QLIKE. Tuning happens per asset, on the ALL feature set, and on the **initial training window only** (inner validation = its last 20 %, chronological). The resulting hyperparameters are used for every feature-set variant of that family × loss and are frozen for all refits. Linear, naive and GARCH models have no tuned hyperparameters.
* **Search spaces:**
  * GBM: learning_rate log-U[0.01, 0.2]; max_iter {100, 200, 400}; max_leaf_nodes {7, 15, 31}; min_samples_leaf {20, 40, 80}; l2_regularization log-U[1e-3, 10].
  * Neural: hidden {16, 32, 64}; layers {1, 2, 3}; dropout U[0, 0.4]; lr log-U[1e-4, 3e-3]; weight_decay log-U[1e-6, 1e-3]; batch_size {64, 128}; epochs = best inner-validation epoch within a maximum of 150.

## 7. Decision rule

For each asset, sentiment is declared to **help** if all of the following hold:
1. The H1 DM test is significant at Holm-adjusted p < 0.05 **in favour of** HAR-BC + SENT.
2. The MSE-log DM test for the same pair is **not** significant against HAR-BC + SENT (no significant worsening on the secondary loss).
3. In the holdout (Step 9), the mean QLIKE difference has the **same sign**. Significance is not required there because the holdout has about 50–70 observations and limited power. That power is reported.

Outcomes:
* 1–3 all hold: **"sentiment helps"** for that asset, with the effect size and CI.
* 1–2 hold but 3 fails: **"development-period evidence, not confirmed in the holdout"**.
* Significant in the opposite direction: **"sentiment hurts"**.
* Otherwise: **"no evidence that sentiment helps"**, stated together with the minimum detectable effect (§5), so that a null is not over-read as proof of no effect.

H3 refines H1: "sentiment helps beyond controls" requires the same three conditions for the Linear-BC ALL vs ALL−SENT pair. Results for GBM and Neural are reported under the same rule. Overall statement: sentiment helps "in general" only if H1 holds for at least two of the three assets. Anything weaker is described asset by asset.

## 8. Holdout protocol (Step 9)

* **Freeze first.** The final configuration list (this document plus amendments) and every tuned hyperparameter are committed and recorded in `DECISIONS.md` before unsealing.
* **Integrity check.** Verify the recorded `SHA256SUMS` hash, then rebuild the holdout features from the sealed raw data with the frozen code and compare file hashes. Mismatches are reported.
* **Holdout forecasts.**
  * Linear, naive and GARCH models continue their refit schedule into the holdout. Holdout targets are used only after they are realized, as in development.
  * GBM and neural models are fit once on all development rows (holdout length < one refit interval), with frozen hyperparameters and the same 5 seeds.
* **Evaluation.** Every pre-registered metric and test is computed **once** and reported unedited next to the development results. Any bug found afterwards is fixed and **both** holdout results are reported (integrity rule 4).

## 9. What counts as exploratory

Everything in Step 8: conditional analyses (high-news days, events, volatile periods), attributions/SHAP, alternative sentiment scorers (VADER), alternative windows or targets, economic (vol-targeting) evaluation, and any model not listed in `configs/experiments.yaml`. Exploratory results never enter the §7 decision rule unless a dated amendment pre-registers them and they are then evaluated on data not used to motivate them. In practice that means the holdout, which can be used only once.

## 10. Size of the pre-registered grid and compute plan

From `scripts/count_preregistered.py` → `analysis/artifacts/preregistered_counts.json`:
**92 model configurations** (BTC 32, SPX 30, NIFTY 30), **120 neural training runs** (8 neural configurations × 5 seeds × 3 assets; ensembles reuse those runs), **15 tuning studies** of 40 trials each, and **40 planned DM tests**.
Single-thread fit times on synthetic data of the real shape (`scripts/bench_fit_times.py`, EXPERIMENT_LOG E1): GRN 4.3–7.1 s, Transformer 22.1 s, GBM 0.34 s, OLS 0.008 s. With about 27 refits per OOS run, the grid needs a few hours on 4 cores. No configuration is dropped for compute reasons without an amendment.

---

## Amendments

*(none)*
