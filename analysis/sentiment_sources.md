# Sentiment sources: candidates, documented claims, and how coverage will be verified

**Status (2026-10-10): coverage NOT YET VERIFIED.** Every vendor is unreachable from the analysis sandbox (DECISIONS.md D7). The author's local run of `scripts/local/run_local.py probe` measures coverage by **querying**, and its output (`analysis/artifacts/sentiment_probe.json`) will replace the "documented" columns below. Selection follows rules fixed **before** any coverage data existed: D25 (density criterion), D28 (subset choice) and D33 (feasibility).

## Candidates

| Source | Access | Documented history (claim and source) | Rate limit (documented/claimed) | Terms / licence | Reachable from sandbox | Planned use |
|---|---|---|---|---|---|---|
| **GDELT DOC 2.0 API** | no key | "searches back to January 1, 2017" per the 2017 launch post ([GDELT blog](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/)); later sources describe a rolling 3-month window, where older requests "may still return data, but it's not guaranteed" ([py-gdelt docs](https://rbozydar.github.io/py-gdelt/user-guide/rest-apis/)). **Conflicting; must be probed.** | rate-limited, quota undocumented ([GDELT blog](https://blog.gdeltproject.org/ukraine-api-rate-limiting-web-ngrams-3-0/)) | Open data; articles belong to publishers (links, titles only) | No | Primary candidate for all assets. Daily top-250 relevance sample + uncapped daily volume (D32). NIFTY query: Indian market terms, `sourcecountry:india` |
| **The Guardian Open Platform** | free developer key | Archive from 1999 (commonly stated; **not verified**) | not verified (developer-tier limits not found in search; assumed 500/day in D33, overridable) | Developer terms; attribution | No | Candidate (BTC, SPX). Thin Indian-market coverage expected |
| **NYT Article Search** | free key | Archive back to 1851 (commonly stated) | 500/day and 5/min per a third-party summary; older guides say 4,000/day and 10/min ([Jentic](https://jentic.com/apis/nytimes.com/times-newswire-api), [D-Lab](https://dlab.berkeley.edu/news/getting-started-nyt-api)): **conflicting** | NYT developer terms (not verified) | No | Candidate (BTC, SPX), subject to the D33 feasibility filter |
| **Alpha Vantage NEWS_SENTIMENT** | free key | Earliest date undocumented; `time_from`/`time_to` supported ([parameter reference](https://qveris.ai/tools/alphavantage.news_sentiment.query.v1.467a92c0)) | 25/day (assumed in legacy code) | Vendor terms | No | **Ineligible** (quota, D32); availability probed only |
| GDELT GKG via BigQuery | GCP billing | 2015-02-18 onward (GDELT 2.0) | billed per TB scanned | Open data | Technically (pre-authenticated gcloud), but **not used** (D8) | Option for the author to run locally if the DOC API lacks pre-2017 history |
| Legacy AlphaVantage/Finnhub file | — | 785 BTC articles over 42 days, etc. | — | — | — | **Not used** (R3; `sentiment_coverage_diagnosis.md`) |

## NIFTY: index-level Indian news (R3)
The legacy NIFTY series was HDFC Bank ADR news. The new NIFTY queries use market-level terms (Sensex, Nifty, Dalal Street, Indian stock market). GDELT is restricted to Indian sources (`sourcecountry:india`). Whether this yields comparable density across 2015–2026 is exactly what the probe measures. If no source passes D25 for NIFTY, NIFTY is excluded from the sentiment hypotheses and the exclusion is reported (D25).

## What will be reported once data arrives
* Per source × asset: probe counts per quarter, eligibility (D33), D25 outcome, the D28 subset table.
* Per asset: articles per trading day and the share of `has_news` rows **per split** of the pre-registered design, plus the share of GDELT days that hit the 250 cap.
* FinBERT model revision (commit hash) and label map, from `analysis/artifacts/finbert_manifest.json`.
* A Phase-2-style profile of the new sentiment features (Step 3 completion), including drift tests across splits.
