# Decisions and binding requirements (human-in-the-loop log)

Every design decision is logged **before** it is acted on: date, options considered, choice, rationale. Binding instructions from the author are logged as requirements (R-numbers). Each status names the commit or step that resolved it. The author reviews this file asynchronously.

## Index

| # | Date | Type | Summary | Status |
|---|---|---|---|---|
| D1 | 2026-10-09 | Author | Work on non-default branch `claude/awesome-curie-6nitg4` (session-bound) instead of `research-audit` | In effect |
| D2 | 2026-10-09 | Author | Phases 1–3 read-only; wait for approval before fixing | **Superseded by D6** (2026-10-10) |
| D3 | 2026-10-09 | Author | Do not change the volatility measure yet | **Superseded by R2** |
| D4 | 2026-10-09 | Author | Sentiment diagnosis top priority | Done (`sentiment_coverage_diagnosis.md`); rebuild ordered (R3) |
| D5 | 2026-10-09 | Author | Deadline unknown | **Resolved 2026-10-10: no deadline** |
| R1 | 2026-10-09 | Author | Neural-HAR retrained with fixed seeds, ≥ 5 runs; DM test vs bias-corrected HAR | Open (Step 7) |
| D6 | 2026-10-10 | Author | Autonomy granted under integrity rules 1–7 (see author brief of 2026-10-10) | In effect |
| R2 | 2026-10-10 | Author | Step 2 fixes: HAR lag, zero-return gap, GK target for SPX/NIFTY, drop regime feature, full-sample credit spread, placeholders, pinned deps, loud failures, logging, unit tests | Open (Step 2) |
| R3 | 2026-10-10 | Author | Rebuild sentiment with comparable density across splits; NIFTY index-level news; missing-indicator, never zero-fill | Open (Step 3) |
| R4 | 2026-10-10 | Author | Fresh holdout from the day after the current data end; hash; touched once in Step 9 | Open (Step 4) |
| R5 | 2026-10-10 | Author | Never use the `.env.example` keys; read keys only from environment variables; never write keys anywhere | In effect |
| D7 | 2026-10-10 | Claude | Network reachability: all data vendors blocked; local-run fetch package | In effect |
| D8 | 2026-10-10 | Claude | Do not use BigQuery/GDELT via the sandbox's GCP credentials | In effect |
| D9 | 2026-10-10 | Claude | Order of work while data is pending | In effect |
| D10 | 2026-10-10 | Claude | New typed package `alphavol/`; legacy code frozen for provenance | In effect |
| D11 | 2026-10-10 | Claude | HAR regressors: log of mean RV over windows ending at t; BTC windows 1/7/30 days, SPX/NIFTY 1/5/22 trading days | In effect |
| D12 | 2026-10-10 | Claude | Target = next trading day's log variance, computed on the full calendar before any filtering; assertion test | In effect |
| D13 | 2026-10-10 | Claude | Invalid-variance days → NaN without cascades: HAR windows use minimum-count rules | In effect |
| D14 | 2026-10-10 | Claude | Dev/holdout boundary D = 2026-07-28; dev rows need target date ≤ D | In effect |
| D15 | 2026-10-10 | Claude | Credit spread = FRED BAA10Y (verify coverage on fetch, assert); absent until fetched, no fill | In effect |
| D16 | 2026-10-10 | Claude | Placeholders: BTC Amihud/Roll properly lagged; removed for SPX/NIFTY | In effect |
| D17 | 2026-10-10 | Claude | Exogenous missing values: bounded causal forward-fill + missing indicators; never zero-fill | In effect |
| D18 | 2026-10-10 | Claude | Single metrics module: DM with Newey–West HAC + Student-t; MCS via `arch` | In effect |
| D19 | 2026-10-10 | Claude | Exact dependency pins; separate fetch-only requirements | In effect |
| D20 | 2026-10-10 | Claude | Information timing: forecast issued at the asset's close of day t; per-source availability lags | In effect |
| D21 | 2026-10-10 | Claude | Sealed-holdout guard enforced in code (`alphavol/io.py`) | In effect |
| D22 | 2026-10-10 | Claude | BTC microstructure recomputed causally from 5-min bars; implied-vol timing | In effect |
| D23 | 2026-10-10 | Claude | Dataset builder: pure function on raw frames; groups explicit; pending groups recorded | In effect |
| D24 | 2026-10-10 | Claude | Intraday SPX/NIFTY data: recommendation only; Garman–Klass stays primary | Recommendation (author decision) |

---

## Detailed entries

### D7 — 2026-10-10 — Data access from the sandbox
* **Observation.** HTTPS probe (`curl -m 12 https://<host>/`) returned no connection (code 000) for every data vendor: api.binance.com, data.binance.vision, query1/2.finance.yahoo.com, fred.stlouisfed.org, api.stlouisfed.org, api.gdeltproject.org, data.gdeltproject.org, content.guardianapis.com, www.alphavantage.co, newsapi.org, huggingface.co, alternative.me, stooq.com, www.nseindia.com, finnhub.io, cryptopanic.com, api.polygon.io, www.dukascopy.com. Reachable: pypi.org, github.com, raw.githubusercontent.com, storage.googleapis.com. No data-vendor API keys exist as environment variables (names checked only, values never read).
* **Options.** (a) Look for mirrors or workarounds; (b) write fetchers here, test them offline against recorded or synthetic responses, and have the author run them locally; (c) stop.
* **Choice.** (b), as integrity rule 6 requires.
* **Rationale.** Rule 6 forbids workarounds. Offline tests (parsers, paging, resume, failure modes) can still be exercised here, which keeps the local run short and loud on failure.
* **Consequence.** Steps 3 (fetch + FinBERT scoring), 4 (holdout fetch) and everything that needs the rebuilt data (5, 7, 8, 9) wait on the author's local run. FinBERT weights are on huggingface.co, so **scoring also has to run locally**.

### D8 — 2026-10-10 — BigQuery GDELT
* **Observation.** The sandbox has gcloud/bq CLIs pre-authenticated to a GCP account. GDELT is published as a BigQuery public dataset.
* **Options.** (a) Query GDELT through BigQuery from here; (b) do not.
* **Choice.** (b), not used.
* **Rationale.** It would be a workaround for a blocked source (rule 6), and it would bill the author's GCP project without consent. It is offered to the author as a **local** option in `sentiment_sources.md`.

### D9 — 2026-10-10 — Order of work while data is pending
* **Options.** (a) Wait idle for data; (b) do everything not requiring new data: Step 1 audit; Step 2 fixes on the committed raw data (except the credit-spread series, which needs a new FRED pull); write and offline-test all fetchers (Steps 3–4); write `PREREGISTRATION.md` (Step 6); build and unit-test the experiment harness **on synthetic data only**; then report with local-run commands.
* **Choice.** (b).
* **Rationale.** No modelling experiment touches real development data before pre-registration is committed. Synthetic-data tests are code verification, not experiments, and are not entered in the experiment log as results.

### D10 — 2026-10-10 — Where the fixes live
* **Options.** (a) Patch every legacy file in place; (b) build a new, typed, tested package and freeze the legacy code; (c) both.
* **Choice.** (b). A new `alphavol/` package is the only code path used from Step 2 on. Legacy files (`run_paper.py`, `src/`, `eval/`, `backtest/`, `notebooks/`) are left untouched and labelled legacy.
* **Rationale.** The legacy pipeline has defects in almost every stage (00–03 audits), and patching would mix fixed and unfixed paths. Keeping it intact lets the paper's old numbers be reproduced for the before/after comparison.

### D11 — 2026-10-10 — HAR regressor definitions
* **Options.** Aggregation: (a) mean of log RV (legacy); (b) log of mean RV level (Corsi 2009; Andersen, Bollerslev & Diebold 2007). Windows for BTC: (c) 5/22 observations (legacy, equity convention); (d) 7/30 calendar days.
* **Choice.** (b) with windows **ending at t**: RV_d = log RV_t, RV_w = log mean(RV_{t−w+1..t}), RV_m = log mean(RV_{t−m+1..t}). w/m = 5/22 trading days for SPX/NIFTY and 7/30 calendar days for BTC (24/7 market).
* **Rationale.** (b) is the standard HAR specification. Averaging logs is a geometric mean, which changes the cascade interpretation. For BTC, a 5-observation window is not a week. The Phase-2 ACF shows 7/14/21-day peaks (`target_quality.md`).

### D12 — 2026-10-10 — Target construction
* **Choice.** Target_t = log variance of the **next trading day of the asset's own calendar**. It is computed by a date-keyed shift on the complete daily series **before** any row filtering. Rows are dropped afterwards only if their own features or target are missing. A unit test asserts that every target equals the next trading day's value (`tests/test_alignment.py`).
* **Rationale.** The legacy shift happened after row drops, which produced targets 35–39 days ahead (Phase 2).

### D13 — 2026-10-10 — Invalid-variance days
* **Options.** (a) Drop the day and let NaN propagate through rolling windows (legacy: 23-row cascades); (b) replace with a floor such as 1e-12 (legacy clip: gives log ≈ −27 outliers); (c) mark as NaN and compute HAR window means over available days with a minimum count (w: ≥ 4/5 or 6/7; m: ≥ 18/22 or 25/30).
* **Choice.** (c). The row whose own variance is invalid has no RV_d and is dropped; neighbouring rows survive.
* **Note.** With Garman–Klass, invalid days occur only when high == low. That count is 0 in the committed SPX/NIFTY data (`target_quality.json → ohlc_quality`), so the rule exists for robustness, not because it changes current numbers.

### D14 — 2026-10-10 — Development / holdout boundary
* **Observation.** In the committed raw data the last day is partial for every asset: BTC 2026-07-29 has 198 of 288 5-min bars (last bar 16:25 UTC); the SPX 2026-07-29 daily bar has volume 1.35 bn vs a 60-day median of 5.48 bn (fetched mid-session); the NIFTY 2026-07-29 bar has volume 0. The legacy matrices used the partial BTC day as a target.
* **Choice.** D = **2026-07-28** for all assets. A development row must have its **target date ≤ D**. The sealed holdout consists of rows whose target date is > D, built from freshly fetched data (raw from 2026-07-29 onwards replaces the partial bars).
* **Rationale.** This prevents any holdout value from entering development through the last row's target, and removes partial-day contamination.
* **BTC incomplete days.** 18 days have < 276 of 288 bars (exchange outages; minimum 168 on 2019-05-15). They are kept and flagged, not rescaled: close-to-close returns across the gap still capture the gap's variance in one return.

### D15 — 2026-10-10 — Credit-spread replacement
* **Options.** (a) Keep ICE BofA HY OAS (only 3 years on FRED: vendor policy); (b) Moody's Baa − 10Y Treasury (FRED `BAA10Y`, daily); (c) drop credit spread.
* **Choice.** (b). The local fetch script asserts that the first valid observation is on or before 2015-01-02 and fails otherwise. **Coverage cannot be verified from the sandbox** (FRED blocked; D7). Until it is fetched and verified, the feature builder raises if `BAA10Y` is requested and absent. It never fills.

### D16 — 2026-10-10 — All-zero placeholder features
* **Choice.** BTC: implement properly by lagging the Amihud illiquidity (`ILLIQ`) and Roll spread (`Roll`) columns already computed from 5-min bars in `btc_obi_daily.parquet` (legacy never lagged them). SPX/NIFTY: **remove**. Index-level "volume" is not tradable dollar volume (^GSPC aggregates constituent share volume; ^NSEI has 29 zero-volume days), so Amihud/Roll are not meaningful.

### D17 — 2026-10-10 — Missing exogenous values
* **Options.** (a) Zero-fill (legacy); (b) drop rows; (c) causal forward-fill with a maximum staleness plus a missing-indicator per feature group; (d) model-specific imputation.
* **Choice.** (c). Maximum staleness: 5 business days for macro and fear/greed, 0 for sentiment (a day without news is "no news", encoded by an indicator and a count of 0, never as sentiment 0 alone). Feature values never use information after t.

### D18 — 2026-10-10 — Metrics implementation
* **Choice.** One module `alphavol/metrics.py`: QLIKE (no prediction clamp in training losses; evaluation clipping only at ±50 as an overflow guard, with a test that it never binds on real data), MSE on log variance, Diebold–Mariano with Newey–West (Bartlett) HAC variance at a lag fixed in the pre-registration, HLN factor, and a Student-t(T−1) reference distribution. The MCS uses `arch.bootstrap.MCS` (reference implementation verified in Step 1) and reports every model's p-value.

### D19 — 2026-10-10 — Dependencies
* **Choice.** `requirements.txt` pins exact versions of the analysis stack (resolved in the audit venv, Python 3.11). `requirements-fetch.txt` pins the fetch-only stack (yfinance, requests, transformers, …) for the author's local run. The legacy `requirements.txt` content is preserved as `requirements-legacy.txt`.

### D20 — 2026-10-10 — Information timing (what is known when the forecast is issued)
* **Rule.** The forecast of the log variance of day t+1 is issued at the **close of day t** on the asset's own clock: BTC 00:00 UTC (end of the UTC day); SPX the NYSE close (16:00 America/New_York); NIFTY the NSE close (15:30 Asia/Kolkata). A feature may enter row t only if it was observable by that time.
* **Per-source lag.**
  * RV, Garman–Klass, and BTC intraday microstructure (OBI², VPIN, Amihud, Roll) for day t: computed from day-t bars, so **lag 0**.
  * US macro from FRED (T10Y2Y, BAA10Y, VIXCLS): observation date d is a US end-of-day value. Lag 0 for SPX (known at the US close); **lag 1 business day for BTC and NIFTY** (not published by 00:00 UTC; the US close of day t happens after the NSE close). For simplicity and safety, **lag 1 for all assets**.
  * Crypto Fear & Greed (Alternative.me), stamped 00:00 UTC of date d: **lag 1**.
  * News sentiment for row t: articles published in (close_{t−1}, close_t], on the asset's clock (Step 3).
* **Options considered.** (a) Uniform lag 1 for everything (legacy: discards day-t information for RV-derived features, which is the HAR mis-specification); (b) per-source timing as above.
* **Choice.** (b). Each rule has a unit test that perturbs data after the cutoff and asserts that row t is unchanged.

### D21 — 2026-10-10 — Sealed-holdout guard
* **Choice.** All holdout files live under `data/sealed/`. `alphavol.io.read_frame` raises `SealedDataError` for any path under `data/sealed/` unless `analysis/UNSEAL_RECORD.json` exists and the caller passes `allow_sealed=True`. That record is created only in Step 9, after the final configurations are frozen and committed. The holdout build script writes files and SHA-256 hashes only; it prints no row counts, dates or statistics.

### D22 — 2026-10-10 — BTC microstructure and implied-volatility features
* **Observation.** Legacy `btc_vpin_daily.parquet` sets the VPIN bucket size from the **whole-sample** average daily volume (`compute_microstructure.py:58-60`, mild look-ahead). Legacy "OBI" is a close-location proxy, although Binance klines contain actual taker-buy (aggressor-side) volume.
* **Options.** (a) Reuse legacy files (lag them); (b) recompute causally in `alphavol/microstructure.py`.
* **Choice.** (b), computed per UTC day from that day's 5-min bars (lag 0, D20):
  * `ofi_abs`: |Σ taker-buy − Σ taker-sell| / Σ volume (aggressor order-flow imbalance).
  * `vpin`: bulk-volume-classification VPIN (Easley, López de Prado & O'Hara 2012). Bucket size V* = (trailing 30-day mean daily volume up to the previous day) / 50; 50-bucket rolling window; value at the day's last completed bucket.
  * `amihud`: mean over bars of |r| / quote volume (USDT) × 1e9.
  * `roll`: 2·√(−cov(r_t, r_{t−1})) within the day when the covariance is negative, else 0 (Roll 1984).
  Legacy files are no longer used.
* **Implied volatility (optional group; use decided in the pre-registration).** SPX: FRED `VIXCLS`. The VIX closes at 16:15 ET, after the 16:00 SPX close, so **lag 1**. NIFTY: India VIX (to be fetched locally, `^INDIAVIX`). It closes with the NSE, so lag 0. BTC: Deribit DVOL exists only from 2023-11-03 in the repo (a 1,000-row API cap), so it is excluded for coverage, the same failure mode as the ICE spread.

### D23 — 2026-10-10 — Dataset builder design
* **Choice.** `alphavol.features.build_dataset(asset, raw: RawInputs, cfg, groups, pending)` is a pure function of in-memory raw frames. Loading is separate, so the same code builds development data and (Step 4) the sealed holdout. Every feature group must be listed explicitly. A requested group whose input is missing raises `MissingDataError` unless the group is in `pending`, in which case it is omitted and recorded in the manifest. The output keeps exogenous NaNs plus `<group>_missing` indicators; imputation happens inside models using training-window statistics only.

### D24 — 2026-10-10 — Is genuine 5-minute data for SPX/NIFTY accessible? (research; no switch)
* **Findings** (web search 2026-10-10; vendor sites unreachable from the sandbox, so terms not fully verified):
  * **Oxford-Man Realized Library:** discontinued, last S&P 500 values June 2022, no replacement planned; its data may not be redistributed ([Realized Library](https://realized.oxford-man.ox.ac.uk/)). Not usable for 2015–2026.
  * **SPX — FirstRate Data:** index product with 1/5-minute bars from 2008-01-02 to July 2026, one-off purchase plus optional update subscription ([firstratedata.com/i/index/SPX](https://firstratedata.com/i/index/SPX)). The SPX price was not shown; a comparable product is $299.95. Licence terms for publication were not confirmed.
  * **SPX — VOLARE** (arXiv:2602.19732, "VOLatility Archive for Realized Estimates"): coverage and licence **not verified**.
  * **NIFTY:** no verified licensed source found. Community datasets exist (Kaggle "NSE – Nifty 50 Index Minute data", listed CC0; GitHub repositories) but their upstream provenance is undocumented. NSE index data is exchange-proprietary, so a CC0 label on a repackaged copy does not establish the right to use it in a publication.
* **Options.** (a) Buy FirstRate SPX 5-minute data and use intraday RV for SPX only; (b) use community NIFTY minute data despite unclear licensing; (c) keep Garman–Klass for both and report intraday RV as future work, or as a robustness check on SPX if (a) is bought.
* **Recommendation.** (c) now. (a) if the author is willing to purchase and confirms the licence permits publishing derived results. **Do not use (b)** in a published paper without written licence confirmation. **No switch has been made**; GK remains the primary target per R2.
