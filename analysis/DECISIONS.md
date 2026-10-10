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
