"""Local data run (author's machine): vendors and Hugging Face are unreachable from the analysis
sandbox (DECISIONS.md D7). Four stages, run in order; each is resumable via the HTTP cache.

  probe         coverage probe of news sources + D25/D28/D33 selection → analysis/artifacts/sentiment_probe.json
  fetch-dev     FRED (T10Y2Y, BAA10Y, VIXCLS), Crypto F&G, India VIX, news (selected sources) up to dev_end
  score-dev     FinBERT (+ VADER) scoring and daily aggregation for the development period
  seal-holdout  fetch everything after dev_end, score, build holdout features, write SHA256SUMS (prints hashes only)

Keys are read from environment variables only: GUARDIAN_API_KEY, NYT_API_KEY (and optional
ALPHAVANTAGE_API_KEY for the availability probe). Never use the keys in data_collection/.env.example.
See scripts/local/RUNBOOK.md for exact commands.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from alphavol import io as aio  # noqa: E402
from alphavol.config import ASSETS, load_config  # noqa: E402
from alphavol.features import daily_variance, load_raw_inputs  # noqa: E402
from alphavol.fetch.http import FetchError, MissingCredentialError  # noqa: E402
from alphavol.fetch.market import assert_coverage, binance_klines, fear_greed, fred_series, yahoo_daily  # noqa: E402
from alphavol.fetch.news import (NYT, AlphaVantage, ArticleStore, FetchLog, GdeltDoc, Guardian,  # noqa: E402
                                 month_windows, probe_windows, select_sources)
from alphavol.logging_utils import configure, get_logger  # noqa: E402
from alphavol.sentiment import (FinbertScorer, aggregate_daily, assign_to_trading_day, close_times,  # noqa: E402
                                dedup_articles, score_articles)

log = get_logger("run_local")
UTC = timezone.utc
CFG = load_config()
D = CFG.dev_end
ART = ROOT / "analysis" / "artifacts"
RAW2 = CFG.extra_raw_dir
CACHE_ARTICLES = ROOT / "data" / "cache" / "articles"
FRED_CODES = ["T10Y2Y", "BAA10Y", "VIXCLS"]
PAGE_SIZE = {"guardian": 200, "nyt": 10}


def _git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def _trading_days_in(asset: str, s: datetime, e: datetime) -> int:
    days = pd.date_range(s, e - timedelta(seconds=1), freq="D")
    return len(days) if asset == "btc" else int((days.dayofweek < 5).sum())


def _close_of(asset: str, day: pd.Timestamp) -> pd.Timestamp:
    return close_times(pd.DatetimeIndex([day]), CFG.assets[asset])[0]


# ── Stage 1: probe ─────────────────────────────────────────────────────────────────────
def stage_probe(args) -> None:
    windows = probe_windows(datetime(2015, 1, 1, tzinfo=UTC), (D + pd.Timedelta(days=1)).to_pydatetime())
    sources = {"gdelt": GdeltDoc(), "guardian": Guardian(), "nyt": NYT()}
    quota = {"guardian": args.guardian_quota, "nyt": args.nyt_quota}
    rows, status = [], {}
    for name, src in sources.items():
        got = []
        try:
            for asset in ASSETS:
                for s, e in windows:
                    n = src.count(asset, s, e)
                    got.append({"source": name, "asset": asset, "window_start": pd.Timestamp(s),
                                "articles": n, "trading_days": _trading_days_in(asset, s, e),
                                "articles_per_trading_day": n / _trading_days_in(asset, s, e)})
            status[name] = "ok"
            rows += got
        except MissingCredentialError as exc:
            status[name] = f"skipped: {exc}"
        except FetchError as exc:
            status[name] = f"error: {exc}"
        log.info("probe %s: %s", name, status[name])
    try:                                   # Alpha Vantage: availability only (D32), one window per year
        av = AlphaVantage()
        avail = {}
        for asset in ("btc", "spx"):
            for s, e in [w for w in windows if w[0].month == 1]:
                avail[f"{asset}|{s:%Y}"] = av.count(asset, s, e)
        status["alphavantage"] = {"eligible": False, "reason": "quota (D32)", "yearly_window_counts": avail}
    except (MissingCredentialError, FetchError) as exc:
        status["alphavantage"] = {"eligible": False, "error": str(exc)}
    probe = pd.DataFrame(rows)
    feas = {}
    for name in [n for n, st in status.items() if st == "ok"]:
        sub = probe[probe["source"] == name]
        if name == "gdelt":
            est = len(ASSETS) * ((D - pd.Timestamp("2015-01-01", tz="UTC")).days + 1 + 12 * 12)
            feas[name] = {"estimated_requests": int(est), "quota_per_day": None, "eligible": True}
            continue
        est = 0
        for asset, g in sub.groupby("asset"):
            per_day = g.sort_values("window_start")["articles"].sum() / (7 * len(g))
            months = (D.year - 2015) * 12 + D.month
            est += months * math.ceil(per_day * 30.4 / PAGE_SIZE[name])
        feas[name] = {"estimated_requests": int(est), "quota_per_day": quota[name],
                      "eligible": est <= 14 * quota[name]}
    eligible = [n for n, f in feas.items() if f["eligible"]]
    selection = (select_sources(probe[probe["source"].isin(eligible)], assets=ASSETS, end_year=D.year)
                 if eligible else {"choice": None, "outcomes": []})
    out = {"run_at": datetime.now(UTC).isoformat(), "dev_end": str(D.date()), "git_commit": _git_commit(),
           "rule": "DECISIONS.md D25 + D28 + D33", "status": status, "feasibility": feas,
           "windows": [[str(a), str(b)] for a, b in windows],
           "probe": probe.assign(window_start=probe["window_start"].astype(str)).to_dict("records") if len(probe) else [],
           "selection": selection}
    aio.write_json(out, ART / "sentiment_probe.json")
    log.info("probe written; selected sources: %s", (selection.get("choice") or {}).get("sources"))


def _selection() -> dict:
    p = ART / "sentiment_probe.json"
    if not p.exists():
        raise SystemExit("run the 'probe' stage first")
    sel = json.loads(p.read_text())["selection"]
    if not sel.get("choice"):
        raise SystemExit("probe selected no admissible source set (D28); news fetch skipped. Report this result.")
    return sel["choice"]


def _fetch_news(asset: str, sources: list[str], start: datetime, end: datetime, store: ArticleStore,
                flog: FetchLog, volume_out: list) -> None:
    for name in sources:
        if name == "gdelt":
            g = GdeltDoc()
            day = start
            while day < end:
                store.append(g.articles(asset, day, min(day + timedelta(days=1), end), flog, sample=True))
                day += timedelta(days=1)
            for ms, me in month_windows(start, end):
                volume_out.append(g.daily_volume(asset, ms, me))
        else:
            src = Guardian() if name == "guardian" else NYT()
            for ms, me in month_windows(start, end):
                store.append(src.articles(asset, ms, me, flog))


# ── Stage 2: development fetch ─────────────────────────────────────────────────────────
def stage_fetch_dev(args) -> None:
    end_d = D.strftime("%Y-%m-%d")
    fred = pd.concat([fred_series(c, "2014-01-01", end_d, cache_tag="dev") for c in FRED_CODES], axis=1)
    assert_coverage(fred["BAA10Y"], first_on_or_before="2015-01-02")        # D15
    assert_coverage(fred["T10Y2Y"], first_on_or_before="2015-01-02")
    assert_coverage(fred["VIXCLS"], first_on_or_before="2015-01-02")
    aio.write_frame(fred[fred.index <= D], RAW2 / "macro" / "fred.parquet")
    fg = fear_greed(cache_tag="dev")
    aio.write_frame(fg[fg.index <= D].rename("value").to_frame(), RAW2 / "macro" / "crypto_fear_greed.parquet")
    iv = yahoo_daily("^INDIAVIX", "2014-01-01", end_d)
    aio.write_frame(iv[iv.index <= D], RAW2 / "ohlcv" / "india_vix_1d.parquet")
    choice = _selection()
    start = datetime(int(choice["common_start_year"]), 1, 1, tzinfo=UTC) - timedelta(days=45)  # buffer for windows
    store = ArticleStore(CACHE_ARTICLES / "dev")
    flog = FetchLog()
    for asset in ASSETS:
        vol: list = []
        end = (_close_of(asset, D) + pd.Timedelta(seconds=1)).to_pydatetime()
        _fetch_news(asset, choice["sources"], start, end, store, flog, vol)
        if vol:
            v = pd.concat(vol).groupby(level=0).sum()
            aio.write_frame(v[v.index <= D].to_frame(), RAW2 / "sentiment" / f"gdelt_volume_{asset}.parquet")
    aio.write_json({"sources": choice["sources"], "start": str(start), "requests": flog.requests,
                    "capped_windows": len(flog.truncations),
                    "capped_by_asset": pd.Series([t.asset for t in flog.truncations]).value_counts().to_dict()},
                   ART / "news_fetch_log_dev.json")


# ── Stage 3: scoring ────────────────────────────────────────────────────────────────────
def _score_and_aggregate(asset: str, arts: pd.DataFrame, trading_dates: pd.DatetimeIndex, scorer, vader,
                         volume: pd.Series | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = CFG.assets[asset]
    arts = dedup_articles(arts)
    scored = score_articles(arts, scorer)
    scored["vader"] = [vader.polarity_scores(str(t))["compound"] for t in scored["title"]]
    agg = aggregate_daily(scored, trading_dates, spec)
    scored["t"] = assign_to_trading_day(scored["published_at"], trading_dates, spec)
    agg["vader_mean"] = scored.dropna(subset=["t"]).groupby("t")["vader"].mean().reindex(agg.index)
    if volume is not None and len(volume):
        # UTC day d's volume is known at the end of d (d+1 00:00 UTC): assign causally (D20)
        tday = assign_to_trading_day(pd.Series(volume.index + pd.Timedelta(days=1)), trading_dates, spec)
        agg["gdelt_volume"] = pd.Series(volume.to_numpy(), index=tday.to_numpy()).groupby(level=0).sum().reindex(agg.index)
    derived = scored.assign(url_sha256=scored["url"].map(lambda u: hashlib.sha256(str(u).encode()).hexdigest()))
    derived = derived[["source", "asset", "published_at", "t", "domain", "url_sha256", "p_pos", "p_neg", "p_neu", "vader"]]
    return agg, derived


def _write_derived(derived: pd.DataFrame, root: Path, asset: str) -> None:
    for year, g in derived.groupby(derived["published_at"].dt.year):
        aio.write_frame(g.reset_index(drop=True), root / "sentiment" / "articles_scored" / asset / f"{year}.parquet")


def _scorers(revision: str | None):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    scorer = FinbertScorer(revision=revision)
    manifest = scorer.load()
    return scorer, manifest, SentimentIntensityAnalyzer()


def stage_score_dev(args) -> None:
    scorer, manifest, vader = _scorers(args.revision)
    store = ArticleStore(CACHE_ARTICLES / "dev")
    for asset in ASSETS:
        spec = CFG.assets[asset]
        variance, _ = daily_variance(asset, spec, load_raw_inputs(asset, CFG).prices)
        tdates = variance.index[variance.index <= D]
        arts = store.load(asset)
        arts = arts[pd.to_datetime(arts["published_at"], utc=True) <= _close_of(asset, D)]
        vp = RAW2 / "sentiment" / f"gdelt_volume_{asset}.parquet"
        vol = pd.read_parquet(vp).iloc[:, 0] if vp.exists() else None
        agg, derived = _score_and_aggregate(asset, arts, tdates, scorer, vader, vol)
        aio.write_frame(agg, RAW2 / "sentiment" / f"sentiment_daily_{asset}.parquet")
        _write_derived(derived, RAW2, asset)
    aio.write_json({**manifest, "scored_at": datetime.now(UTC).isoformat(), "git_commit": _git_commit()},
                   ART / "finbert_manifest.json")


# ── Stage 4: holdout (prints hashes only) ───────────────────────────────────────────────
def stage_seal_holdout(args) -> None:
    import logging

    from alphavol.holdout import build_holdout_features, sealed_raw_dir, write_sha256sums, write_seal_info

    end = pd.Timestamp(args.end, tz="UTC") if args.end else (pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=1))
    if end <= D:
        raise SystemExit(f"holdout end {end.date()} must be after dev_end {D.date()}")
    logging.getLogger("alphavol").setLevel(logging.WARNING)       # no content-bearing INFO lines
    raw = sealed_raw_dir(CFG)
    tag = f"seal-{end.date()}"
    start = (D + pd.Timedelta(days=1)).to_pydatetime()
    btc = binance_klines(start, (end + pd.Timedelta(days=1)).to_pydatetime())
    aio.write_frame(btc, raw / "ohlcv" / "btc_5min.parquet")
    for name, tic in (("spx", "^GSPC"), ("nifty", "^NSEI"), ("india_vix", "^INDIAVIX")):
        aio.write_frame(yahoo_daily(tic, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")),
                        raw / "ohlcv" / f"{name}_1d.parquet")
    fred = pd.concat([fred_series(c, "2014-01-01", end.strftime("%Y-%m-%d"), cache_tag=tag) for c in FRED_CODES], axis=1)
    aio.write_frame(fred[fred.index > D], raw / "macro" / "fred.parquet")
    fg = fear_greed(cache_tag=tag)
    aio.write_frame(fg[fg.index > D].rename("value").to_frame(), raw / "macro" / "crypto_fear_greed.parquet")
    fin = json.loads((ART / "finbert_manifest.json").read_text())
    scorer, manifest, vader = _scorers(fin["resolved_revision"])
    if manifest["resolved_revision"] != fin["resolved_revision"]:
        raise SystemExit("FinBERT revision differs from the development scoring run")
    choice = _selection()
    store = ArticleStore(CACHE_ARTICLES / "sealed")
    flog = FetchLog()
    for asset in ASSETS:
        spec = CFG.assets[asset]
        if asset == "btc":
            tdates = pd.date_range(D + pd.Timedelta(days=1), end, freq="D", tz="UTC")
        else:
            tdates = pd.read_parquet(raw / "ohlcv" / f"{asset}_1d.parquet").index
        n_start = _close_of(asset, D).to_pydatetime()
        n_end = (close_times(pd.DatetimeIndex([tdates[-1]]), spec)[0] + pd.Timedelta(seconds=1)).to_pydatetime()
        vol: list = []
        _fetch_news(asset, choice["sources"], n_start.replace(hour=0, minute=0, second=0), n_end, store, flog, vol)
        arts = store.load(asset)
        pub = pd.to_datetime(arts["published_at"], utc=True)
        arts = arts[(pub > _close_of(asset, D)) & (pub <= n_end)]
        volume = pd.concat(vol).groupby(level=0).sum() if vol else None
        agg, derived = _score_and_aggregate(asset, arts, pd.DatetimeIndex(tdates), scorer, vader, volume)
        aio.write_frame(agg, raw / "sentiment" / f"sentiment_daily_{asset}.parquet")
        _write_derived(derived, raw, asset)
    build_holdout_features(CFG, pending=set(args.pending))
    write_seal_info(CFG, {"holdout_end": str(end.date()), "dev_end": str(D.date()), "git_commit": _git_commit(),
                          "sealed_at": datetime.now(UTC).isoformat(), "finbert_revision": fin["resolved_revision"],
                          "news_sources": choice["sources"], "pending": sorted(args.pending)})
    manifest_sha = write_sha256sums(CFG)
    sys.stdout.write(f"SEALED. Record this in analysis/DECISIONS.md:\n  SHA256SUMS sha256 = {manifest_sha}\n"
                     f"  holdout_end = {end.date()}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--nyt-quota", type=int, default=500)
    p.add_argument("--guardian-quota", type=int, default=500)
    sub.add_parser("fetch-dev")
    s = sub.add_parser("score-dev")
    s.add_argument("--revision", default=None, help="pin a ProsusAI/finbert commit hash")
    h = sub.add_parser("seal-holdout")
    h.add_argument("--end", default=None, help="last holdout day (default: yesterday UTC)")
    h.add_argument("--pending", nargs="*", default=[])
    args = ap.parse_args()
    configure()
    {"probe": stage_probe, "fetch-dev": stage_fetch_dev, "score-dev": stage_score_dev,
     "seal-holdout": stage_seal_holdout}[args.stage](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
