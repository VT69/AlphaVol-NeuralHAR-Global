# Why is sentiment coverage so sparse? (priority diagnosis)

**Scripts that produced every number below:** `scripts/audit_sentiment_coverage.py` → `analysis/artifacts/sentiment_coverage.json`, and `scripts/audit_trace_pipeline.py` → `analysis/artifacts/phase1_trace.json`. The URL-parsing check was run inline with `requests.Request(...).prepare()`; its output is quoted in §2.4.
**Status:** diagnosis only. No collection code was changed and nothing was re-fetched. The audit container's egress proxy blocks alphavantage.co, newsapi.org and huggingface.co, so a re-fetch cannot be done from here anyway.

---

## 1. Bottom line

1. **The 43-day BTC window is not an API free-tier history limit. It is a count cap.** `fetch_alphavantage_news` makes **one** request per asset for the *latest* ≤ 1,000 articles. It passes no `time_from`/`time_to`, does no pagination, and the run **overwrites** the file instead of accumulating. BTC news is dense (18.78 articles/day), so the newest articles cover only 41.8 days (2026-06-18 → 2026-07-30).
   *Decisive evidence:* the same call, with the same key, on the same day, returns very different start dates per asset: BTC 2026-06-18, NIFTY 2023-07-03, SPX 2014-04-08. A time-based history limit would truncate all three at the same date. A count cap truncates in proportion to how much news each ticker has, which is what we see.
2. **The NIFTY "sentiment" is HDFC Bank news.** The query ticker is `HDB`, the HDFC Bank ADR (`fetch_news_extended.py:71`). 97.5 % of the 1,000 NIFTY articles mention "hdfc"; only 35.7 % match a NIFTY/Indian-market keyword. NIFTY also hit the 1,000-article cap exactly.
3. **Two of the four news sources never ran, and nothing failed loudly.** The committed file contains only `AlphaVantage` and `Finnhub` rows. Guardian (the one source with deep free history back to 2010) and CryptoPanic produced zero rows. A missing key only logs a warning (`fetch_news_extended.py:415-443`), and `run()` still writes the file.
4. **NewsAPI never reaches FinBERT.** `newsapi_headlines.parquet` (198 rows) is written by `fetch_sentiment.py`, but `sentiment_engine.py` reads only `extended_news_headlines.parquet` (`sentiment_engine.py:101`). The NewsAPI free (Developer) tier does have a one-month history limit [pricing](https://newsapi.org/pricing), but that limit has **no effect** on the model inputs.
5. **Net effect on the experiment.** Non-zero `FinBERT_score_lag1` rows per Neural-HAR split:

| Asset | Train (rows / non-zero) | Val | Test |
|---|---|---|---|
| BTC | 2,195 / **0** | 274 / **0** | 275 / 40 |
| SPX | 2,289 / 53 | 286 / 14 | 287 / 151 |
| NIFTY | 2,204 / 15 | 275 / 30 | 276 / 166 |

A model cannot learn a weight for a feature that is constantly 0 throughout training (BTC). For SPX and NIFTY, about 2 % and 0.7 % of training rows carry any signal. **The paper's question ("does sentiment add information?") cannot be answered with this data, in either direction.** The current results neither support nor refute it.

## 2. Root causes, ranked by impact

### 2.1 Count cap mistaken for history, with no time-windowing and no accumulation (BLOCKER)

| Evidence | Location |
|---|---|
| `start_date` parameter is accepted but **never used**. `start_dt` and `now_dt` are computed and discarded. | `fetch_news_extended.py:52, 75-76` |
| Request has `sort=LATEST`, `limit=1000`, and no `time_from`/`time_to` | `fetch_news_extended.py:87-93` |
| The code comment acknowledges the problem ("gets ~1000 most recent articles per asset… Re-run weekly to accumulate") … | `fetch_news_extended.py:79-82` |
| … but the output is overwritten, not appended, so weekly re-runs would never accumulate | `fetch_news_extended.py:457-458` |
| `run(start_date="2019-01-01")` default is therefore meaningless for AlphaVantage | `fetch_news_extended.py:394` |

Observed footprint (`sentiment_coverage.json → extended`):

| Asset (AV ticker) | Articles | First → last | Span (days) | Articles / day | In last 60 days | Cap hit? |
|---|---|---|---|---|---|---|
| BTC (`CRYPTO:BTC`) | 785 | 2026-06-18 → 2026-07-30 | 41.8 | 18.78 | 785 (100 %) | Window fully determined by recency |
| NIFTY (`HDB`) | 1,000 | 2023-07-03 → 2026-07-29 | 1,122.6 | 0.89 | 168 | **Yes, exactly 1,000** |
| SPX (`SPY`) | 543 | 2014-04-08 → 2026-07-29 | 4,495.6 | 0.12 | 90 | No: SPY-tagged articles are rare |

Why BTC has 785 rather than 1,000 is **NOT DETERMINABLE** from the saved data: the raw API response was not stored. Possible causes are a short response or intra-asset `(url, title)` duplicates removed at `fetch_news_extended.py:454`.

What the API allows: third-party references to the Alpha Vantage `NEWS_SENTIMENT` documentation list `time_from`/`time_to` (format `YYYYMMDDTHHMM`), `sort` ∈ {LATEST, EARLIEST, RELEVANCE}, and `limit` default 50, max 1000 ([search summary](https://qveris.ai/tools/alphavantage.news_sentiment.query.v1.467a92c0); [Wealth-Lab](https://wealth-lab.com/blog/news-sentiment)). Back-history therefore has to be obtained by walking explicit time windows. I could not reach alphavantage.co to confirm the earliest available date or the current free-tier quota (egress blocked). The script's own comment assumes 25 requests/day.

### 2.2 Ticker / query mismatches (BLOCKER for NIFTY, MAJOR for others)

| Asset | Query | Problem | Location |
|---|---|---|---|
| NIFTY | AV `tickers=HDB` | HDFC Bank ADR stands in for the index. 97.5 % of articles mention HDFC; 35.7 % mention NIFTY/Indian-market terms. | `fetch_news_extended.py:71` |
| SPX | AV `tickers=SPY` | ETF-tagged articles are sparse (0.12/day over 12 years). 94.7 % are relevant to US markets, so the problem is coverage, not relevance. | `:70` |
| BTC | AV `CRYPTO:BTC` | 77.3 % keyword-relevant. Fine. | `:69` |
| SPX / NIFTY | Finnhub `category=general` | The *same* general-news feed is assigned to **both** SPX and NIFTY. Cross-asset dedup on `(url,title)` keeps the first copy, so all 100 go to SPX and NIFTY gets **0**. Only 21 % of these are relevant to SPX. They span 2.65 days. | `:258-261, 267-291, 454` |
| BTC | Finnhub `company-news` with `symbol=BINANCE:BTCUSDT` | `company-news` takes equity symbols. The crypto pair returns nothing usable, which is silently turned into `[]` (0 BTC Finnhub rows observed). | `:263, 297-302` |
| SPX (NewsAPI) | `q="S&P 500" OR "stock market crash"` interpolated into the URL without encoding | The `&` splits the parameter. Verified by URL parse: `q` becomes `'"S'`, plus a stray key `'P 500" OR "stock market crash"'`. The SPX request is malformed and **no SPX rows exist** in `newsapi_headlines.parquet`. | `fetch_sentiment.py:64-72` |

### 2.3 Silent error swallowing (MAJOR)

| Pattern | Location |
|---|---|
| AV response without `feed` (rate-limit / bad-key JSON) → `warning` + `continue`. The asset is silently missing. | `fetch_news_extended.py:99-102` |
| Any AV exception → `logger.error`, then continue | `:130-131` |
| Guardian: on first error → `break` (no retry) | `:186-188, 231-233` |
| Finnhub: `status_code != 200` → `[]` with no log line; whole block in one `try` | `:274, 302, 322-323` |
| CryptoPanic: non-200 → `break`, no log | `:352-353` |
| Missing API key → warning only. `run()` still writes the file if *any* source returned data. | `:405-443, 445-458` |
| NewsAPI: non-200 → silently skipped (no `else` branch). No pagination, no `from`/`to`. | `fetch_sentiment.py:73-84` |
| 24-hour cache check: re-running within a day does nothing | `fetch_sentiment.py:56`, `utils/cache.py:5-17` |
| No retry or backoff anywhere for AV, Finnhub or NewsAPI. Only the GDELT fetcher retries. | — |

Neither the extended-news file nor `manifest.json` records which sources succeeded. The extended-news file is not even in the manifest list (`fetch_all.py`), so there is no audit trail of the partial failure.

### 2.4 Downstream encoding hides the gap (MAJOR)

* Days without news are set to `0.0` twice: `sentiment_engine.py:90` (`resample('D').mean().fillna(0)`) and `build_feature_matrix.py:118-120`. A zero is indistinguishable from "perfectly neutral news". There is no missingness indicator and no article-count feature.
* `sent_surprise` is then computed on mostly-zero series (`build_feature_matrix.py:85`), so the "surprise" fires mainly when news *appears*.
* In-sample, `FinBERT_score_lag1` for BTC is 0 on every row with complete HAR-S exogenous data. `table4_regime_coefficients.csv` correspondingly shows BTC FinBERT β = 0.0 with an empty p-value.

### 2.5 Not verifiable from here

* **FinBERT label order.** `sentiment_engine.py:69-71` assumes index 0 = positive, 1 = negative, 2 = neutral for `ProsusAI/finbert`. huggingface.co is blocked from this container, so I could not read the model's `id2label`. **NOT VERIFIED.** It must be checked before any re-scoring. A swapped order would flip the sign of every score.
* **Cross-validation signal available but unused.** 100 % of AV rows carry `av_sentiment_score` (`sentiment_coverage.json`). It could be used to sanity-check the FinBERT sign at zero cost.

## 3. Recommendation (decision needed: this changes the data, so it is your call)

The sentiment question can only be tested once the training window contains sentiment. Options, roughly from least to most effort:

| Option | What | Pros | Cons / risk |
|---|---|---|---|
| **A. Re-fetch AV by time window** | Loop monthly windows with `time_from`/`time_to` and `sort=EARLIEST`; append and dedup; log per-window counts; fail loudly on a missing `feed`. Replace `HDB` with India topic/keyword queries or a different source for NIFTY. | Same vendor, so the code change is small. Pre-scored `av_sentiment_score` comes for free. | Free quota (assumed 25 requests/day in the script) gives ~25 windows/day, i.e. ~1–2 weeks of runs for 3 assets × ~90 months. Earliest AV coverage unconfirmed (could not reach docs). |
| **B. Turn on Guardian** (code already written) | Set `GUARDIAN_API_KEY`; the existing `fetch_guardian_news` paginates from 2019. | Deep history, full text, already implemented. | General press, not finance-wire. Relevance filtering needed. Weak for NIFTY. |
| **C. GDELT tone (historical)** | GDELT 2.0 GKG/Events daily tone per query, back to 2015. | Long, consistent daily coverage for all three markets. A standard sentiment proxy in finance papers. | The free DOC API in `fetch_sentiment.py` is limited to `timespan=3months`. Full history needs BigQuery (billing) or bulk files. Not FinBERT. |
| **D. Narrow the claim instead** | Keep the data, restrict sentiment experiments to the dense window (2025-06 → 2026-07 for SPX/NIFTY), and state the limitation. | No re-fetch. | ~1 year of data, no proper train/test split for sentiment. Probably not publishable as a sentiment paper. |

**My recommendation: A + B for SPX/BTC, A (fixed NIFTY query) + C for NIFTY.** Whichever data you choose, also: (i) add a `news_count` feature and a `has_news` indicator instead of zero-filling, (ii) store the raw API responses and a per-source coverage log in the manifest, and (iii) verify the FinBERT label order against `av_sentiment_score` before re-scoring.

Until the data is fixed, the paper cannot claim anything about sentiment. The defensible framing is "HAR vs. ML on microstructure/macro features", with sentiment reported as untested.
