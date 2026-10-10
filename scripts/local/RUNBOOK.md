# Local data run — exact commands

**Why local:** every data vendor and huggingface.co is blocked from the analysis sandbox
(`analysis/DECISIONS.md` D7). All code below is tested offline (`tests/`, including an end-to-end
run of all four stages with fake vendors).
**Resumable:** every API response is cached under `data/cache/`. After any interruption, re-run the
same command; already-fetched windows are not requested again.

## 0. Before you start
* **Rotate** the NewsAPI and FRED keys committed in `data_collection/.env.example`. They are public in git history. The new pipeline never reads that file.
* Python **3.11**, git, about 5 GB free disk (the response cache is the largest part).

## 1. Keys (free; environment variables only, never files)
| Variable | Where to get it | Required? |
|---|---|---|
| `GUARDIAN_API_KEY` | https://open-platform.theguardian.com/access/ (developer key) | yes |
| `NYT_API_KEY` | https://developer.nytimes.com/get-started (enable "Article Search API") | yes |
| `ALPHAVANTAGE_API_KEY` | https://www.alphavantage.co/support/#api-key | optional (availability probe only) |

GDELT, FRED (`fredgraph.csv`), Binance market data, Yahoo and Alternative.me need no key.

```bash
# macOS/Linux (current shell only)
export GUARDIAN_API_KEY='...'
export NYT_API_KEY='...'
# Windows PowerShell
#   $env:GUARDIAN_API_KEY = '...'; $env:NYT_API_KEY = '...'
```

## 2. Setup
```bash
git fetch origin claude/awesome-curie-6nitg4
git checkout claude/awesome-curie-6nitg4 && git pull
python3.11 -m venv .venv-fetch && source .venv-fetch/bin/activate   # Windows: .venv-fetch\Scripts\activate
pip install -r requirements-fetch.txt
python -m pytest -q                                                # expect: all tests pass, no network used
```

## 3. Stages (run in order)
| # | Command | What it does | Rough runtime* |
|---|---|---|---|
| 1 | `python scripts/local/run_local.py probe` | Counts articles per source × asset in one 7-day window per quarter (2015 → 2026-07), then applies the pre-specified rules D25 + D28 + D33 to choose sources. Writes `analysis/artifacts/sentiment_probe.json`. | ~45 min |
| 2 | `python scripts/local/run_local.py fetch-dev` | FRED (T10Y2Y, BAA10Y, VIXCLS; **fails if BAA10Y does not start by 2015-01-02**), Crypto Fear & Greed, India VIX, and news from the chosen sources up to the close of 2026-07-28. | ~20 h (GDELT ≈ 1 request per asset per day at 5.5 s) |
| 3 | `python scripts/local/run_local.py score-dev` | FinBERT (+VADER) scoring. Writes daily sentiment and text-free per-article scores to `data/raw_v2/`, and the model revision to `analysis/artifacts/finbert_manifest.json`. | 1–3 h CPU |
| 4 | `python scripts/local/run_local.py seal-holdout` | Fetches everything from 2026-07-29 to yesterday, scores with the **same** FinBERT revision, builds holdout features, writes `data/sealed/SHA256SUMS`. **Prints only the manifest hash.** | 1–2 h |

\*Estimates. Vendor rate limits are undocumented or unverified. If stage 1 reports that no source
set is admissible, stop and send me `sentiment_probe.json`; that result itself goes into the paper.

**Holdout discipline (integrity rule 4):** after stage 4, do **not** open, plot, summarise or
describe anything in `data/sealed/`. Just send me the printed `SHA256SUMS sha256 = …` line.

## 4. Commit and push
```bash
git add analysis/artifacts/sentiment_probe.json analysis/artifacts/news_fetch_log_dev.json \
        analysis/artifacts/finbert_manifest.json data/raw_v2 data/sealed
git commit -m "data: probe, dev macro/sentiment refresh, sealed holdout"
git push origin claude/awesome-curie-6nitg4
```
Never commit `data/cache/`. It holds raw API responses and headline text (gitignored, D29).

## 5. If something fails
Failures are loud by design. A stage stops with a `FetchError` or `MissingDataError` that names
the source and a **redacted** URL. Re-running resumes from the cache. Common cases:
* `HTTP 401/403` → key missing or invalid for that source.
* `exhausted N retries` → rate limiting. Wait and re-run.
* `first valid observation … is after required 2015-01-02` → a FRED series lacks history. Send me the message; I will log a decision before changing anything.
