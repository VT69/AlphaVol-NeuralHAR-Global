"""Phase 2 — exhaustive, deterministic dataset profile.

Profiles every modelling dataset (the per-asset feature matrices that HAR, HAR-S and
Neural-HAR consume), per split (train / val / test, the chronological 80/10/10 split used by
``src/models/train_neural_har.py``) and combined, plus the news-headline text corpus and an
inventory of every raw source file. No sampling is used anywhere (all data fits in memory).

Outputs
-------
analysis/artifacts/data_profile.json            machine-readable profile (dashboard input)
analysis/artifacts/feature_matrices.parquet     long-format copy of the matrices + split labels
analysis/artifacts/news_headlines.parquet       headlines + split labels (dashboard explorer)
analysis/tables/T1..T6 *.csv / *.tex            paper-ready tables (LaTeX booktabs)
analysis/figures/profile/*.png|svg              all Phase-2 figures (300 dpi)
analysis/01_data_profile.md                     human-readable dossier (auto-generated sections
                                                plus analysis/notes/01_data_profile_interpretation.md)

Usage:  python scripts/profile_data.py            (≈1–2 min on 4 CPU cores)
Seed:   SEED=0 for mutual information, t-SNE and langdetect.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.cluster import hierarchy  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.feature_selection import f_regression, mutual_info_regression  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from statsmodels.stats.outliers_influence import variance_inflation_factor  # noqa: E402

# ── Configuration ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_collection" / "data" / "raw"
PROC = ROOT / "data" / "processed"
AN = ROOT / "analysis"
ART, TAB, FIG = AN / "artifacts", AN / "tables", AN / "figures" / "profile"
NOTES = AN / "notes" / "01_data_profile_interpretation.md"
ASSETS = ["btc", "spx", "nifty"]
TRAIN_FRAC, VAL_FRAC = 0.80, 0.10          # mirrors train_neural_har.train_asset defaults
SEED = 0
CORE = ["RV_d", "RV_w", "RV_m", "Target_RV"]
TARGET = "Target_RV"
# Okabe–Ito colour-blind-safe palette
C = {"train": "#0072B2", "val": "#E69F00", "test": "#D55E00", "combined": "#555555",
     "btc": "#0072B2", "spx": "#D55E00", "nifty": "#009E73"}
SEMANTIC = {
    "log_RV": "continuous (same-day log variance; not a model input)",
    "RV_d": "continuous (HAR daily lag)", "RV_w": "continuous (HAR weekly lag)",
    "RV_m": "continuous (HAR monthly lag)", "Target_RV": "target (continuous, log variance t+1)",
    "vpin_lag1": "continuous (bounded [0,1])", "obi_sq_lag1": "continuous (bounded [0,1])",
    "illiq_lag1": "continuous (placeholder)", "roll_spread_lag1": "continuous (placeholder)",
    "FinBERT_score_lag1": "continuous (bounded [-1,1], zero-inflated)",
    "sent_surprise_lag1": "continuous (zero-inflated)",
    "credit_spread_lag1": "continuous (percent)", "term_slope_lag1": "continuous (percent)",
    "crypto_fg_lag1": "continuous/ordinal (bounded [0,1])",
    "gmsi": "continuous (credit_spread - term_slope; not lagged)",
    "regime": "ordinal categorical (low<medium<high)",
}
BOUNDS = {"vpin_lag1": (0, 1), "obi_sq_lag1": (0, 1), "crypto_fg_lag1": (0, 1),
          "FinBERT_score_lag1": (-1, 1), "credit_spread_lag1": (0, None)}
DATA_PULL_DATE = pd.Timestamp("2026-07-29", tz="UTC")   # manifest.json pull timestamps
log = logging.getLogger("profile_data")


# ── Small helpers ──────────────────────────────────────────────────────────────
def jsonable(o: Any) -> Any:
    """Recursively convert numpy / pandas scalars to JSON-safe Python types."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if (math.isnan(o) or math.isinf(o)) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, np.datetime64)):
        return str(o)
    return o


def save_fig(fig: plt.Figure, name: str) -> str:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return f"figures/profile/{name}.png"


def save_table(df: pd.DataFrame, name: str, caption: str, index: bool = False) -> None:
    """CSV + LaTeX booktabs (Styler.to_latex with hrules)."""
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{name}.csv", index=index)
    d = df if index else df.reset_index(drop=True)
    fmt = {c: "{:.4g}" for c in d.columns if pd.api.types.is_float_dtype(d[c])}
    sty = d.style.format(fmt, na_rep="--")
    if not index:
        sty = sty.hide(axis="index")
    tex = sty.to_latex(hrules=True, caption=caption, label=f"tab:{name.lower()}",
                       position="htbp", position_float="centering")
    tex = tex.replace("_", r"\_").replace(r"\_\_", "__").replace("%", r"\%")
    # undo escaping inside LaTeX commands that legitimately contain underscores
    tex = tex.replace(r"\label{tab:" + name.lower().replace("_", r"\_") + "}",
                      r"\label{tab:" + name.lower() + "}")
    (TAB / f"{name}.tex").write_text(tex)


def md_table(df: pd.DataFrame, floatfmt: str = "{:.4g}", index: bool = False) -> str:
    """Render a DataFrame as a GitHub markdown table without extra dependencies."""
    d = df.reset_index() if index else df
    def f(v: Any) -> str:
        if isinstance(v, (float, np.floating)):
            return "—" if pd.isna(v) else floatfmt.format(v)
        return str(v)
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join(["---"] * len(d.columns)) + "|"
    rows = ["| " + " | ".join(f(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


# ── Loading & splitting ───────────────────────────────────────────────────────
def load_matrix(asset: str) -> pd.DataFrame:
    """Load a feature matrix exactly as the models do (drop rows missing HAR core / target)."""
    df = pd.read_parquet(PROC / f"{asset}_feature_matrix.parquet").sort_index()
    return df.dropna(subset=CORE)


def split_labels(n: int) -> np.ndarray:
    ntr, nva = int(n * TRAIN_FRAC), int(n * VAL_FRAC)
    lab = np.array(["test"] * n, dtype=object)
    lab[:ntr], lab[ntr:ntr + nva] = "train", "val"
    return lab


# ── Univariate profiles ────────────────────────────────────────────────────────
def numeric_profile(s: pd.Series) -> dict:
    x = s.to_numpy(dtype=float)
    finite = x[np.isfinite(x)]
    n_all, n = len(x), len(finite)
    out: dict = {"count": n, "non_null": int(s.notna().sum()), "null": int(s.isna().sum()),
                 "null_pct": 100 * float(s.isna().mean()) if n_all else None,
                 "unique": int(s.nunique()), "cardinality_ratio": s.nunique() / max(n_all, 1),
                 "inf_count": int(np.isinf(x).sum()),
                 "zero_count": int((finite == 0).sum()), "negative_count": int((finite < 0).sum())}
    if n == 0:
        return out
    q = np.percentile(finite, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    mean, std = finite.mean(), finite.std(ddof=1) if n > 1 else 0.0
    vc = pd.Series(finite).value_counts()
    modes = vc[vc == vc.iloc[0]].index.tolist()
    iqr = q[5] - q[3]
    out.update({
        "mean": mean, "median": float(np.median(finite)),
        "mode": modes[:10], "mode_count": int(vc.iloc[0]), "n_modes": len(modes),
        "std": std, "variance": std ** 2, "min": finite.min(), "max": finite.max(),
        "range": finite.max() - finite.min(),
        **{f"p{p}": v for p, v in zip([1, 5, 10, 25, 50, 75, 90, 95, 99], q)},
        "iqr": iqr,
        "skewness": float(stats.skew(finite, bias=False)) if n > 2 and std > 0 else None,
        "kurtosis_excess": float(stats.kurtosis(finite, bias=False)) if n > 3 and std > 0 else None,
        "cv": std / abs(mean) if mean != 0 else None,
    })
    for k in (1.5, 3.0):
        lo, hi = q[3] - k * iqr, q[5] + k * iqr
        cnt = int(((finite < lo) | (finite > hi)).sum())
        out[f"outliers_iqr_{k}"] = cnt
        out[f"outliers_iqr_{k}_pct"] = 100 * cnt / n
    if std > 0:
        z = (finite - mean) / std
        out["outliers_z3"] = int((np.abs(z) > 3).sum())
        out["outliers_z3_pct"] = 100 * out["outliers_z3"] / n
    if std > 0 and n >= 8:
        if n <= 5000:
            w, p = stats.shapiro(finite)
            out["normality"] = {"test": "Shapiro-Wilk", "stat": w, "p": p}
        else:
            k2, p = stats.normaltest(finite)
            out["normality"] = {"test": "D'Agostino K2", "stat": k2, "p": p}
        out["normality"]["interpretation"] = ("reject normality (p<0.05)" if out["normality"]["p"] < 0.05
                                              else "cannot reject normality")
    else:
        out["normality"] = {"test": None, "interpretation": "constant or too few values"}
    out["is_constant"] = bool(std == 0)
    return out


def categorical_profile(s: pd.Series) -> dict:
    s = s.astype("object")
    vc = s.value_counts(dropna=False)
    freq = vc / vc.sum()
    p = freq.to_numpy(dtype=float)
    return {"non_null": int(s.notna().sum()), "null": int(s.isna().sum()),
            "unique": int(s.nunique()), "cardinality_ratio": s.nunique() / max(len(s), 1),
            "value_counts": {str(k): int(v) for k, v in vc.items()},
            "relative_freq": {str(k): float(v) for k, v in freq.items()},
            "mode": str(vc.index[0]), "entropy_bits": float(-(p * np.log2(p)).sum()),
            "max_entropy_bits": float(np.log2(len(vc))) if len(vc) > 1 else 0.0,
            "rare_categories_lt1pct": int((freq < 0.01).sum()),
            "singletons": int((vc == 1).sum()),
            "whitespace_or_case_variants": int(len(vc) - s.dropna().astype(str).str.strip().str.lower().nunique())}


TOKEN_RE = re.compile(r"[A-Za-z0-9$%&']+")


def text_profile(s: pd.Series) -> dict:
    s = s.fillna("").astype(str)
    chars = s.str.len()
    toks = s.map(lambda t: TOKEN_RE.findall(t.lower()))
    ntok = toks.map(len)
    vocab = Counter(t for ts in toks for t in ts)
    def dist(v: pd.Series) -> dict:
        return {"mean": v.mean(), "median": v.median(), "std": v.std(), "min": int(v.min()), "max": int(v.max()),
                **{f"p{p}": float(np.percentile(v, p)) for p in (5, 25, 75, 95, 99)}}
    return {"n": int(len(s)), "empty": int((chars == 0).sum()), "char_length": dist(chars),
            "token_length": dist(ntok), "vocab_size": len(vocab),
            "top_tokens": vocab.most_common(25),
            "duplicate_texts": int(s[s != ""].duplicated().sum())}


# ── Integrity helpers ─────────────────────────────────────────────────────────
def row_hash(df: pd.DataFrame) -> pd.Series:
    """Content hash of a row (numerics rounded to 8 dp) — excludes the date index."""
    r = df.copy()
    for c in r.columns:
        if pd.api.types.is_numeric_dtype(r[c]):
            r[c] = r[c].round(8)
    keys = ["|".join(map(str, row)) for row in r.to_numpy(dtype=object)]  # NaN -> 'nan' explicitly
    return pd.Series([hashlib.sha1(k.encode()).hexdigest() for k in keys], index=r.index)


def target_alignment(asset: str, df: pd.DataFrame) -> dict:
    """Compare Target_RV with raw log_RV on the next *raw* trading day."""
    raw = pd.read_parquet(RAW / "realized_vol" / f"{asset}_rv_daily.parquet").sort_index()
    raw.index = pd.to_datetime(raw.index, utc=True)
    nxt = raw["log_RV"].shift(-1)
    nxt_date = raw.index.to_series().shift(-1)
    common = df.index.intersection(raw.index)
    expected = nxt.reindex(common)
    got = df.loc[common, TARGET]
    bad = ~np.isclose(got, expected, equal_nan=False) | expected.isna()
    gaps = (df.index.to_series().shift(-1) - df.index.to_series()).dt.days.reindex(common)
    rows = []
    for d in common[bad.to_numpy()]:
        rows.append({"date": str(d.date()), "target_is_logRV_of": str(df.index[df.index.get_loc(d) + 1].date())
                     if df.index.get_loc(d) + 1 < len(df) else None,
                     "next_raw_trading_day": str(nxt_date.loc[d].date()) if pd.notna(nxt_date.loc[d]) else None,
                     "days_ahead": float(gaps.loc[d]) if pd.notna(gaps.loc[d]) else None})
    return {"rows_checked": int(len(common)), "misaligned_targets": int(bad.sum()), "examples": rows[:20]}


def calendar_gaps(asset: str, df: pd.DataFrame) -> list[dict]:
    raw = pd.read_parquet(RAW / "realized_vol" / f"{asset}_rv_daily.parquet").sort_index()
    raw.index = pd.to_datetime(raw.index, utc=True)
    g = df.index.to_series().diff().dt.days
    out = []
    for d, days in g[g > 5].items():
        prev = df.index[df.index.get_loc(d) - 1]
        mid = raw.loc[(raw.index > prev) & (raw.index < d)]
        zero = mid.index[mid.get("log_ret", pd.Series(dtype=float)) == 0]
        out.append({"from": str(prev.date()), "to": str(d.date()), "days": float(days),
                    "raw_rows_dropped": int(len(mid)),
                    "zero_return_days": [str(z.date()) for z in zero],
                    "cause": "zero close-to-close return -> log(0)=NaN -> rolling(22) NaN for 22 rows -> rows dropped"
                    if len(zero) else "unknown"})
    return out


def missingness(df: pd.DataFrame, cols: list[str]) -> dict:
    m = df[cols].isna()
    out = {"null_counts": m.sum().to_dict(), "null_pct": (100 * m.mean()).to_dict()}
    nz = [c for c in cols if 0 < m[c].mean() < 1]
    if len(nz) >= 2:
        out["missingness_correlation"] = m[nz].astype(int).corr().round(4).to_dict()
    # MAR/MNAR-looking diagnostics: missingness vs time and vs observed target
    out["mechanism"] = {}
    t = np.arange(len(df))
    for c in nz:
        miss = m[c].to_numpy()
        r_time = stats.pointbiserialr(miss, t)
        y = df[TARGET].to_numpy()
        mw = stats.mannwhitneyu(y[miss], y[~miss]) if miss.sum() and (~miss).sum() else None
        first_valid, last_valid = df[c].first_valid_index(), df[c].last_valid_index()
        contiguous_block = bool(m[c].loc[first_valid:last_valid].sum() == 0) if first_valid is not None else None
        out["mechanism"][c] = {
            "corr_missing_vs_time": r_time.statistic, "p_time": r_time.pvalue,
            "target_mannwhitney_p": mw.pvalue if mw else None,
            "first_valid": str(first_valid), "last_valid": str(last_valid),
            "missing_only_outside_[first,last]_valid": contiguous_block,
            "verdict": ("structural / time-dependent (not MCAR): missing as a contiguous block in time"
                        if contiguous_block and r_time.pvalue < 0.01 else
                        "time-dependent (not MCAR)" if r_time.pvalue < 0.01 else "no time dependence detected"),
        }
    return out


def drift_tests(a: pd.DataFrame, b: pd.DataFrame, num: list[str], cat: list[str]) -> list[dict]:
    rows = []
    for c in num:
        x, y = a[c].dropna(), b[c].dropna()
        if len(x) < 5 or len(y) < 5 or (x.nunique() <= 1 and y.nunique() <= 1):
            rows.append({"feature": c, "test": "KS", "stat": None, "p": None, "note": "constant or too few"})
            continue
        ks = stats.ks_2samp(x, y)
        rows.append({"feature": c, "test": "KS", "stat": ks.statistic, "p": ks.pvalue,
                     "mean_a": x.mean(), "mean_b": y.mean(), "std_a": x.std(), "std_b": y.std()})
    for c in cat:
        ct = pd.crosstab(np.r_[np.zeros(len(a)), np.ones(len(b))],
                         np.r_[a[c].astype(str).to_numpy(), b[c].astype(str).to_numpy()])
        if ct.shape[1] < 2:
            rows.append({"feature": c, "test": "chi2", "stat": None, "p": None, "note": "single category"})
            continue
        chi = stats.chi2_contingency(ct)
        rows.append({"feature": c, "test": "chi2", "stat": chi.statistic, "p": chi.pvalue, "dof": chi.dof})
    return rows


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    ct = pd.crosstab(x, y)
    if min(ct.shape) < 2:
        return float("nan")
    chi2 = stats.chi2_contingency(ct, correction=False).statistic
    n = ct.to_numpy().sum()
    return float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))


# ── Figures ────────────────────────────────────────────────────────────────────
def fig_numeric(asset: str, col: str, df: pd.DataFrame, splits: np.ndarray) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.0), gridspec_kw={"width_ratios": [2.2, 1, 1]})
    x = df[col]
    ax = axes[0]
    if x.dropna().nunique() > 1:
        bins = min(60, max(10, int(np.sqrt(x.notna().sum()))))
        ax.hist(x.dropna(), bins=bins, density=True, color="#999999", alpha=0.55, label="all")
        try:
            kde = stats.gaussian_kde(x.dropna())
            gx = np.linspace(x.min(), x.max(), 300)
            ax.plot(gx, kde(gx), color="#222222", lw=1.2, label="KDE")
        except Exception:  # noqa: BLE001 - singular KDE for near-constant data
            pass
    else:
        ax.text(0.5, 0.5, f"constant = {x.dropna().iloc[0] if x.notna().any() else 'NaN'}",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_title(f"{asset.upper()} · {col} · histogram + KDE", fontsize=9)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=7)
    data = [x[splits == s].dropna().to_numpy() for s in ("train", "val", "test")]
    ok = [d if len(d) else np.array([np.nan]) for d in data]
    axes[1].boxplot(ok, tick_labels=["train", "val", "test"], showfliers=True,
                    flierprops={"markersize": 2})
    axes[1].set_title("box by split", fontsize=9)
    if all(len(d) > 1 and np.nanstd(d) > 0 for d in data):
        vp = axes[2].violinplot(data, showmedians=True)
        for b, s in zip(vp["bodies"], ("train", "val", "test")):
            b.set_facecolor(C[s]); b.set_alpha(0.6)
        axes[2].set_xticks([1, 2, 3], ["train", "val", "test"])
    else:
        axes[2].text(0.5, 0.5, "violin n/a\n(constant split)", ha="center", va="center",
                     transform=axes[2].transAxes, fontsize=8)
    axes[2].set_title("violin by split", fontsize=9)
    for a_ in axes:
        a_.tick_params(labelsize=7)
    fig.tight_layout()
    return save_fig(fig, f"num_{asset}_{col}")


def fig_corr(asset: str, corr: pd.DataFrame, method: str) -> str:
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(corr)), corr.index, fontsize=7)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.iat[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5,
                        color="white" if abs(v) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"{asset.upper()} · {method} correlation (train split)", fontsize=9)
    fig.tight_layout()
    return save_fig(fig, f"corr_{method.lower()}_{asset}")


def fig_missing(asset: str, df: pd.DataFrame, cols: list[str], zero_as_missing: list[str]) -> list[str]:
    m = df[cols].isna().copy()
    for c in zero_as_missing:
        if c in m:
            m[c + " (==0)"] = (df[c].fillna(0) == 0)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.imshow(m.T.to_numpy(dtype=float), aspect="auto", cmap="Greys", interpolation="nearest")
    ax.set_yticks(range(m.shape[1]), m.columns, fontsize=7)
    ticks = np.linspace(0, len(df) - 1, 6).astype(int)
    ax.set_xticks(ticks, [str(df.index[i].date()) for i in ticks], fontsize=7)
    ax.set_title(f"{asset.upper()} · missingness matrix (black = missing; '(==0)' rows = zero-filled)", fontsize=9)
    fig.tight_layout()
    paths = [save_fig(fig, f"missing_matrix_{asset}")]
    var = [c for c in m.columns if 0 < m[c].mean() < 1]
    if len(var) >= 2:
        corr = m[var].astype(int).corr().fillna(0)
        dist = (1 - corr.abs()).to_numpy(copy=True)
        np.fill_diagonal(dist, 0)
        z = hierarchy.linkage(dist[np.triu_indices(len(var), 1)], "average")
        fig, ax = plt.subplots(figsize=(6, 3))
        hierarchy.dendrogram(z, labels=var, ax=ax, orientation="right", color_threshold=0)
        ax.tick_params(labelsize=7)
        ax.set_title(f"{asset.upper()} · missingness dendrogram (1-|corr|)", fontsize=9)
        fig.tight_layout()
        paths.append(save_fig(fig, f"missing_dendrogram_{asset}"))
    return paths


# ── Main per-asset profile ──────────────────────────────────────────────────────
def profile_asset(asset: str) -> tuple[dict, pd.DataFrame]:
    df = load_matrix(asset)
    splits = split_labels(len(df))
    num_cols = [c for c in df.columns if c != "regime"]
    model_inputs = ["RV_d", "RV_w", "RV_m"] + [c for c in df.columns if c.endswith("_lag1")]
    res: dict = {"asset": asset, "n_rows": len(df), "n_cols": df.shape[1],
                 "date_min": str(df.index.min()), "date_max": str(df.index.max()),
                 "memory_bytes": int(df.memory_usage(deep=True).sum()),
                 "file": f"data/processed/{asset}_feature_matrix.parquet",
                 "file_bytes": (PROC / f"{asset}_feature_matrix.parquet").stat().st_size,
                 "split_sizes": dict(Counter(splits)),
                 "split_ranges": {s: [str(df.index[splits == s].min().date()), str(df.index[splits == s].max().date())]
                                  for s in ("train", "val", "test")}}
    # Column inventory
    inv = []
    for c in df.columns:
        s = df[c]
        inv.append({"column": c, "dtype": str(s.dtype), "semantic": SEMANTIC.get(c, "continuous"),
                    "non_null": int(s.notna().sum()), "null": int(s.isna().sum()),
                    "null_pct": 100 * float(s.isna().mean()), "unique": int(s.nunique()),
                    "cardinality_ratio": float(s.nunique() / len(s)),
                    "is_model_input": c in model_inputs})
    res["columns"] = inv
    # Univariate per split + combined
    res["numeric"], res["categorical"] = {}, {}
    for s_name in ("train", "val", "test", "combined"):
        sub = df if s_name == "combined" else df[splits == s_name]
        res["numeric"][s_name] = {c: numeric_profile(sub[c]) for c in num_cols}
        res["categorical"][s_name] = {"regime": categorical_profile(sub["regime"])}
    # Target diagnostics (regression: distribution + transform diagnostics)
    y = df[TARGET]
    rv = np.exp(y)
    bc_lambda = stats.boxcox(rv.to_numpy())[1]
    res["target"] = {
        "definition": "Target_RV = log_RV shifted -1 (log daily variance on the next row)",
        "log_scale": numeric_profile(y),
        "raw_variance_scale": {k: v for k, v in numeric_profile(rv).items() if k in
                               ("mean", "median", "std", "min", "max", "skewness", "kurtosis_excess")},
        "boxcox_lambda_on_raw_variance": bc_lambda,
        "interpretation": "Box-Cox lambda near 0 supports the log transform",
    }
    # "Majority-class" analogue for regression: naive forecasts on the test split
    test = splits == "test"
    ytr, yte = y[splits == "train"], y[test]
    lr_today = df["log_RV"]
    naive = {"train_mean": np.full(test.sum(), ytr.mean()),
             "random_walk_logRV_t (uses day t)": lr_today[test].to_numpy(),
             "random_walk_RV_d (info set as coded)": df.loc[test, "RV_d"].to_numpy(),
             "RV_m (22-day mean)": df.loc[test, "RV_m"].to_numpy()}
    def qlike(p: np.ndarray, a: np.ndarray) -> float:
        av, pv = np.exp(np.clip(a, -15, 15)), np.exp(np.clip(p, -15, 15))
        return float(np.mean(av / pv - np.log(av / pv) - 1))
    res["naive_baselines_test"] = {k: {"MSE": float(np.mean((yte.to_numpy() - v) ** 2)),
                                       "QLIKE": qlike(v, yte.to_numpy()),
                                       "R2_oos_vs_train_mean": 1 - float(np.sum((yte.to_numpy() - v) ** 2)
                                                                         / np.sum((yte.to_numpy() - ytr.mean()) ** 2))}
                                   for k, v in naive.items()}
    # Multivariate (train split only — no peeking at test)
    tr = df[splits == "train"]
    feats = [c for c in model_inputs if tr[c].nunique() > 1]
    const_feats = [c for c in model_inputs if tr[c].nunique() <= 1]
    corr_cols = feats + ["log_RV", TARGET]
    pear, spear = tr[corr_cols].corr("pearson"), tr[corr_cols].corr("spearman")
    res["correlation"] = {"pearson": pear.round(4).to_dict(), "spearman": spear.round(4).to_dict(),
                          "constant_features_excluded": const_feats}
    res["figures"] = {"corr": [fig_corr(asset, pear, "Pearson"), fig_corr(asset, spear, "Spearman")]}
    X = tr[feats].ffill().fillna(0.0)
    mi = mutual_info_regression(X, tr[TARGET], random_state=SEED)
    fstat, fp = f_regression(X, tr[TARGET])
    assoc = pd.DataFrame({"feature": feats, "mutual_info": mi, "f_stat": fstat, "f_p": fp,
                          "pearson_with_target": [pear.loc[f, TARGET] for f in feats],
                          "spearman_with_target": [spear.loc[f, TARGET] for f in feats]})
    assoc["leakage_flag"] = (assoc["spearman_with_target"].abs() > 0.95)
    assoc = assoc.sort_values("mutual_info", ascending=False)
    res["feature_target_association"] = assoc.to_dict("records")
    res["feature_target_association_note"] = ("train split; NaNs forward-filled then 0-filled exactly as "
                                              "train_neural_har.py does; mutual_info random_state=0")
    # Regime (categorical) vs target and vs binned target
    yq = pd.qcut(tr[TARGET], 3, labels=["low", "mid", "high"])
    groups = [tr.loc[tr["regime"] == r, TARGET] for r in tr["regime"].astype(str).unique()]
    kw = stats.kruskal(*groups) if len(groups) > 1 else None
    res["regime_vs_target"] = {"cramers_v_regime_vs_target_tercile": cramers_v(tr["regime"].astype(str), yq),
                               "kruskal_p": kw.pvalue if kw else None,
                               "regime_counts_train": tr["regime"].astype(str).value_counts().to_dict()}
    # VIF
    vif_X = X.loc[:, X.std() > 0]
    vif_X = (vif_X - vif_X.mean()) / vif_X.std()
    res["vif"] = {c: float(variance_inflation_factor(np.c_[np.ones(len(vif_X)), vif_X.to_numpy()], i + 1))
                  for i, c in enumerate(vif_X.columns)}
    res["vif_note"] = f"train split; constant features excluded: {const_feats}"
    # Pairplot of top-k informative features
    top = assoc["feature"].head(4).tolist()
    k = len(top) + 1
    fig, axes = plt.subplots(k, k, figsize=(2.0 * k, 2.0 * k))
    cols_pp = top + [TARGET]
    for i, ci in enumerate(cols_pp):
        for j, cj in enumerate(cols_pp):
            ax = axes[i, j]
            if i == j:
                ax.hist(tr[ci].dropna(), bins=30, color=C[asset], alpha=0.7)
            else:
                ax.scatter(tr[cj], tr[ci], s=1.5, alpha=0.35, color=C[asset], rasterized=True)
            ax.tick_params(labelsize=5)
            if i == k - 1:
                ax.set_xlabel(cj, fontsize=6)
            if j == 0:
                ax.set_ylabel(ci, fontsize=6)
    fig.suptitle(f"{asset.upper()} · pairplot of top-{len(top)} features by MI + target (train)", fontsize=9)
    fig.tight_layout()
    res["figures"]["pairplot"] = save_fig(fig, f"pairplot_{asset}")

    # Integrity
    integ: dict = {}
    content = row_hash(df.drop(columns=["regime"]))
    integ["exact_duplicate_rows_incl_date"] = int(df.reset_index().duplicated().sum())
    integ["exact_duplicate_rows_content_only"] = int(content.duplicated().sum())
    Z = StandardScaler().fit_transform(df[feats].ffill().fillna(0.0))
    nn = NearestNeighbors(n_neighbors=2).fit(Z)
    d, idx = nn.kneighbors(Z)
    eps = 0.01 * float(np.median(d[:, 1]))
    near = d[:, 1] < eps
    integ["near_duplicate_rows"] = {"definition": f"nearest-neighbour distance in standardised model-input space < 1% of median NN distance ({eps:.4g})",
                                    "count": int(near.sum()),
                                    "examples": [{"date": str(df.index[i].date()), "nn_date": str(df.index[idx[i, 1]].date()),
                                                  "distance": float(d[i, 1])} for i in np.where(near)[0][:10]]}
    # Split overlap: ID, content hash, near-duplicate similarity
    ov = []
    sets = {s: df.index[splits == s] for s in ("train", "val", "test")}
    hashes = {s: set(content[splits == s]) for s in ("train", "val", "test")}
    for a_, b_ in (("train", "val"), ("train", "test"), ("val", "test")):
        Za, Zb = Z[splits == a_], Z[splits == b_]
        nn_a = NearestNeighbors(n_neighbors=2).fit(Za)
        within = nn_a.kneighbors(Za)[0][:, 1]
        cross = nn_a.kneighbors(Zb, n_neighbors=1)[0][:, 0]
        thr = float(np.percentile(within, 1))
        ov.append({"pair": f"{a_}-{b_}", "id_overlap": int(len(sets[a_].intersection(sets[b_]))),
                   "content_hash_overlap": int(len(hashes[a_] & hashes[b_])),
                   "near_dup_threshold_p1_within": thr,
                   "near_dup_cross_count": int((cross < thr).sum()),
                   "near_dup_cross_pct": 100 * float((cross < thr).mean())})
    integ["split_overlap"] = ov
    # Temporal ordering
    integ["temporal"] = {
        "index_monotonic_increasing": bool(df.index.is_monotonic_increasing),
        "duplicate_dates": int(df.index.duplicated().sum()),
        "train_max_lt_val_min": bool(sets["train"].max() < sets["val"].min()),
        "val_max_lt_test_min": bool(sets["val"].max() < sets["test"].min()),
        "future_dates_after_pull": int((df.index > DATA_PULL_DATE).sum()),
        "rolling_window_rows_straddling_into_test": int(min(22, (splits == "test").sum())),
        "note": "RV_w/RV_m of the first 22 test rows average log_RV from val dates by construction (not leakage)",
    }
    integ["calendar_gaps_gt_5d"] = calendar_gaps(asset, df)
    integ["target_alignment"] = target_alignment(asset, df)
    # Value consistency
    checks = {}
    for c, (lo, hi) in BOUNDS.items():
        if c in df:
            v = df[c].dropna()
            checks[c] = {"below_min": int((v < lo).sum()) if lo is not None else 0,
                         "above_max": int((v > hi).sum()) if hi is not None else 0, "bounds": [lo, hi]}
    checks["constant_placeholders"] = [c for c in df.columns if c != "regime" and df[c].nunique(dropna=True) <= 1]
    checks["regime_categories"] = sorted(df["regime"].astype(str).unique().tolist())
    checks["log_RV_extreme_low_lt_mean_minus_5sd"] = int((df["log_RV"] < df["log_RV"].mean() - 5 * df["log_RV"].std()).sum())
    checks["unit_note"] = ("credit_spread and term_slope are in percentage points; gmsi = credit_spread - term_slope "
                           "mixes a spread level with a curve slope (not a standardised index)")
    integ["value_checks"] = checks
    # Missingness (true NaN) and 'zero as missing' for sentiment
    sent = [c for c in ("FinBERT_score_lag1", "sent_surprise_lag1") if c in df]
    integ["missingness"] = missingness(df, num_cols)
    integ["zero_as_missing"] = {c: {s: float((df.loc[splits == s, c].fillna(0) == 0).mean())
                                    for s in ("train", "val", "test")} for c in sent}
    res["figures"]["missing"] = fig_missing(asset, df, num_cols, sent)
    # Drift
    num_drift = [c for c in model_inputs + [TARGET]]
    integ["drift_train_vs_test"] = drift_tests(df[splits == "train"], df[splits == "test"], num_drift, ["regime"])
    integ["drift_train_vs_val"] = drift_tests(df[splits == "train"], df[splits == "val"], num_drift, ["regime"])
    integ["drift_val_vs_test"] = drift_tests(df[splits == "val"], df[splits == "test"], num_drift, ["regime"])
    res["integrity"] = integ

    # Figures: per-numeric, target per split, drift overlays, PCA/t-SNE, regime bars
    res["figures"]["numeric"] = [fig_numeric(asset, c, df, splits) for c in num_cols]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for s_name in ("train", "val", "test"):
        ax.hist(y[splits == s_name], bins=50, density=True, histtype="step", lw=1.4, color=C[s_name],
                label=f"{s_name} (n={int((splits == s_name).sum())})")
    ax.set_xlabel("Target_RV (log variance, next day)")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title(f"{asset.upper()} · target distribution per split", fontsize=9)
    res["figures"]["target_by_split"] = save_fig(fig, f"target_by_split_{asset}")
    dr = pd.DataFrame([r for r in integ["drift_train_vs_test"] if r.get("stat") is not None and r["test"] == "KS"])
    top_drift = dr.sort_values("stat", ascending=False)["feature"].head(4).tolist()
    fig, axes = plt.subplots(1, len(top_drift), figsize=(3.0 * len(top_drift), 2.8))
    for ax, c in zip(np.atleast_1d(axes), top_drift):
        for s_name in ("train", "test"):
            v = df.loc[splits == s_name, c].dropna()
            ax.hist(v, bins=40, density=True, histtype="stepfilled", alpha=0.45, color=C[s_name], label=s_name)
        r = dr.set_index("feature").loc[c]
        ax.set_title(f"{c}\nKS={r['stat']:.2f}, p={r['p']:.1e}", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.legend(frameon=False, fontsize=6)
    fig.suptitle(f"{asset.upper()} · train vs test, top drifting features", fontsize=9)
    fig.tight_layout()
    res["figures"]["drift"] = save_fig(fig, f"drift_top_{asset}")
    # PCA + t-SNE on standardised model inputs
    pca = PCA().fit(Z)
    res["pca"] = {"explained_variance_ratio": pca.explained_variance_ratio_.tolist(), "features": feats}
    P = pca.transform(Z)[:, :2]
    ts = TSNE(n_components=2, random_state=SEED, init="pca", perplexity=30).fit_transform(Z)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    axes[0].bar(range(1, len(feats) + 1), pca.explained_variance_ratio_, color="#999999")
    axes[0].plot(range(1, len(feats) + 1), np.cumsum(pca.explained_variance_ratio_), "-o", ms=3, color="#222222")
    axes[0].set_title("PCA scree (bars) + cumulative", fontsize=9)
    for ax, E, nm in ((axes[1], P, "PCA"), (axes[2], ts, "t-SNE")):
        sc = ax.scatter(E[:, 0], E[:, 1], c=y, cmap="viridis", s=3, rasterized=True)
        ax.set_title(f"{nm} 2-D, colour = Target_RV", fontsize=9)
        ax.tick_params(labelsize=6)
    fig.colorbar(sc, ax=axes[2], shrink=0.8)
    fig.suptitle(f"{asset.upper()} · dimensionality view (all splits, standardised model inputs)", fontsize=9)
    fig.tight_layout()
    res["figures"]["pca_tsne"] = save_fig(fig, f"pca_tsne_{asset}")
    emb = pd.DataFrame({"pca1": P[:, 0], "pca2": P[:, 1], "tsne1": ts[:, 0], "tsne2": ts[:, 1]}, index=df.index)
    # Does the dominant structure in input space track data availability rather than market state?
    from sklearn.cluster import KMeans
    km = KMeans(2, random_state=SEED, n_init=10).fit(ts)
    if "credit_spread_lag1" in df:
        ct = pd.crosstab(km.labels_, df["credit_spread_lag1"].notna().to_numpy())
        res["tsne_cluster_vs_credit_available"] = {f"cluster{int(i)}": {str(bool(k)): int(v) for k, v in row.items()}
                                                   for i, row in ct.iterrows()}
    fig, ax = plt.subplots(figsize=(5, 2.8))
    vc = pd.crosstab(splits, df["regime"].astype(str)).reindex(index=["train", "val", "test"])
    vc.plot.bar(ax=ax, color=["#D55E00", "#E69F00", "#0072B2"][:vc.shape[1]], rot=0)
    ax.set_title(f"{asset.upper()} · regime counts by split", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    res["figures"]["regime_bar"] = save_fig(fig, f"regime_by_split_{asset}")

    long = df.copy()
    long["split"] = splits
    long["asset"] = asset
    long = long.join(emb)
    long["regime"] = long["regime"].astype(str)
    return res, long.reset_index(names="date")


# ── News corpus profile ────────────────────────────────────────────────────────
def profile_news(split_starts: dict[str, dict[str, pd.Timestamp]]) -> tuple[dict, pd.DataFrame]:
    from langdetect import DetectorFactory, detect
    DetectorFactory.seed = SEED
    p = RAW / "sentiment" / "extended_news_headlines.parquet"
    n = pd.read_parquet(p)
    n["date"] = pd.to_datetime(n["date"], utc=True)
    n["text"] = n["text_finbert"].fillna(n["title"]).fillna("")
    def lang(t: str) -> str:
        try:
            return detect(t) if t.strip() else "empty"
        except Exception:  # noqa: BLE001 - langdetect raises on symbol-only text
            return "unknown"
    n["lang"] = n["text"].map(lang)
    # assign the Neural-HAR split of the *row that would use this headline* (lag-1: next day)
    def split_of(r: pd.Series) -> str:
        s = split_starts.get(r["asset"].lower())
        if s is None:
            return "n/a"
        d = r["date"].floor("D") + pd.Timedelta(days=1)
        return "test" if d >= s["test"] else "val" if d >= s["val"] else "train"
    n["split"] = n.apply(split_of, axis=1)
    res: dict = {"file": str(p.relative_to(ROOT)), "file_bytes": p.stat().st_size,
                 "memory_bytes": int(n.memory_usage(deep=True).sum()), "shape": list(n.shape),
                 "columns": [{"column": c, "dtype": str(n[c].dtype), "non_null": int(n[c].notna().sum()),
                              "null_pct": 100 * float(n[c].isna().mean()), "unique": int(n[c].nunique())}
                             for c in n.columns],
                 "categorical": {c: categorical_profile(n[c]) for c in ("asset", "source", "av_sentiment_label", "split")},
                 "language": n["lang"].value_counts().to_dict(),
                 "av_sentiment_score": numeric_profile(n["av_sentiment_score"])}
    res["text"] = {}
    for a, g in n.groupby("asset"):
        res["text"][a] = {"title": text_profile(g["title"]), "text_finbert": text_profile(g["text"])}
        tr_vocab = set(t for x in g.loc[g.split == "train", "text"] for t in TOKEN_RE.findall(x.lower()))
        te_tokens = [t for x in g.loc[g.split == "test", "text"] for t in TOKEN_RE.findall(x.lower())]
        res["text"][a]["oov_rate_test_vs_train_vocab"] = (float(np.mean([t not in tr_vocab for t in te_tokens]))
                                                          if te_tokens and tr_vocab else None)
        res["text"][a]["split_counts"] = g["split"].value_counts().to_dict()
    res["text_all"] = text_profile(n["text"])
    res["duplicate_title_across_splits"] = int(n.groupby("title")["split"].nunique().gt(1).sum())
    res["oov_note"] = ("FinBERT WordPiece OOV not computed: huggingface.co is blocked in the audit container; "
                       "tokens are lower-cased regex word tokens [A-Za-z0-9$%&']+")
    return res, n.drop(columns=["description"])


# ── Raw inventory ──────────────────────────────────────────────────────────────
def raw_inventory() -> list[dict]:
    rows = []
    for p in sorted(RAW.rglob("*.parquet")):
        pf = pq.ParquetFile(p)
        rows.append({"file": str(p.relative_to(ROOT)), "bytes_on_disk": p.stat().st_size,
                     "rows": pf.metadata.num_rows, "cols": pf.metadata.num_columns})
        if pf.metadata.num_rows and pf.metadata.num_rows <= 1_000_000:
            df = pd.read_parquet(p)
            rows[-1]["memory_bytes"] = int(df.memory_usage(deep=True).sum())
            rows[-1]["null_pct_overall"] = 100 * float(df.isna().to_numpy().mean()) if df.size else None
            idx = df.index
            if isinstance(idx, pd.DatetimeIndex):
                rows[-1]["start"], rows[-1]["end"] = str(idx.min().date()), str(idx.max().date())
            else:
                rows[-1]["index"] = f"{type(idx).__name__} (not date-indexed)"
    return rows


# ── Orchestration ──────────────────────────────────────────────────────────────
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.time()
    for d in (ART, TAB, FIG):
        d.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    profile: dict = {"generated_by": "scripts/profile_data.py", "seed": SEED,
                     "split_rule": f"chronological {TRAIN_FRAC:.0%}/{VAL_FRAC:.0%}/{1 - TRAIN_FRAC - VAL_FRAC:.0%} "
                                   "(train_neural_har.train_asset defaults)",
                     "sampling": "none (all rows used)", "assets": {}}
    longs, starts = [], {}
    for a in ASSETS:
        log.info("profiling %s", a)
        res, long = profile_asset(a)
        profile["assets"][a] = res
        longs.append(long)
        idx = load_matrix(a).index
        lab = split_labels(len(idx))
        starts[a] = {"val": idx[lab == "val"].min(), "test": idx[lab == "test"].min()}
    log.info("profiling news")
    profile["news"], news = profile_news(starts)
    profile["raw_inventory"] = raw_inventory()
    fred = pd.read_parquet(RAW / "macro" / "fred_series.parquet")
    profile["fred_first_valid"] = {c: str(fred[c].first_valid_index().date()) for c in fred.columns}
    profile["fred_last_date"] = str(fred.index.max().date())
    profile["runtime_seconds"] = round(time.time() - t0, 1)

    pd.concat(longs, ignore_index=True).to_parquet(ART / "feature_matrices.parquet")
    news.to_parquet(ART / "news_headlines.parquet")
    (ART / "data_profile.json").write_text(json.dumps(jsonable(profile), indent=1))
    build_tables(profile)
    write_markdown(profile)
    log.info("done in %.1fs", time.time() - t0)


# ── Tables ─────────────────────────────────────────────────────────────────────
def build_tables(P: dict) -> None:
    A = P["assets"]
    t1 = []
    for a, r in A.items():
        t1.append({"Dataset": f"{a.upper()} feature matrix", "Rows": r["n_rows"], "Cols": r["n_cols"],
                   "Start": r["date_min"][:10], "End": r["date_max"][:10],
                   "Train/Val/Test": "/".join(str(r["split_sizes"][s]) for s in ("train", "val", "test")),
                   "Test start": r["split_ranges"]["test"][0],
                   "Disk (KB)": round(r["file_bytes"] / 1024, 1), "Memory (KB)": round(r["memory_bytes"] / 1024, 1)})
    nw = P["news"]
    t1.append({"Dataset": "News headlines (FinBERT input)", "Rows": nw["shape"][0], "Cols": nw["shape"][1],
               "Start": None, "End": None, "Train/Val/Test": "/".join(str(nw["categorical"]["split"]["value_counts"].get(s, 0))
                                                                     for s in ("train", "val", "test")),
               "Test start": None, "Disk (KB)": round(nw["file_bytes"] / 1024, 1),
               "Memory (KB)": round(nw["memory_bytes"] / 1024, 1)})
    save_table(pd.DataFrame(t1), "T1_dataset_summary", "Modelling datasets: size, period and chronological splits.")

    t2 = []
    for a, r in A.items():
        for c, s in r["numeric"]["combined"].items():
            t2.append({"Asset": a.upper(), "Feature": c, "N": s["count"], "Null%": s["null_pct"],
                       "Mean": s.get("mean"), "Std": s.get("std"), "Min": s.get("min"), "P1": s.get("p1"),
                       "Median": s.get("median"), "P99": s.get("p99"), "Max": s.get("max"),
                       "Skew": s.get("skewness"), "ExKurt": s.get("kurtosis_excess"),
                       "Zero%": 100 * s["zero_count"] / max(s["count"], 1),
                       "Out|z|>3%": s.get("outliers_z3_pct"),
                       "Normality p": (s.get("normality") or {}).get("p")})
    save_table(pd.DataFrame(t2), "T2_numeric_stats", "Univariate statistics of all numeric columns (all rows).")

    t3 = []
    for a, r in A.items():
        for s_name in ("train", "val", "test", "combined"):
            cp = r["categorical"][s_name]["regime"]
            t3.append({"Asset": a.upper(), "Split": s_name, **{f"n_{k}": v for k, v in cp["value_counts"].items()},
                       "Mode": cp["mode"], "Entropy (bits)": cp["entropy_bits"], "Rare<1%": cp["rare_categories_lt1pct"]})
    for c in ("asset", "source", "av_sentiment_label"):
        cp = nw["categorical"][c]
        t3.append({"Asset": "news", "Split": c, "Mode": cp["mode"], "Entropy (bits)": cp["entropy_bits"],
                   "Rare<1%": cp["rare_categories_lt1pct"],
                   **{f"n_{k}": v for k, v in cp["value_counts"].items()}})
    save_table(pd.DataFrame(t3), "T3_categorical_stats", "Categorical columns: counts, mode and entropy.")

    t4 = []
    for a, r in A.items():
        for s_name in ("train", "val", "test"):
            s = r["numeric"][s_name][TARGET]
            z = r["integrity"]["zero_as_missing"].get("FinBERT_score_lag1", {}).get(s_name)
            t4.append({"Asset": a.upper(), "Split": s_name, "N": s["count"],
                       "From": r["split_ranges"][s_name][0], "To": r["split_ranges"][s_name][1],
                       "Target mean": s["mean"], "Target std": s["std"], "Target P5": s["p5"], "Target P95": s["p95"],
                       "FinBERT==0 %": 100 * z if z is not None else None,
                       **{f"regime_{k}": v for k, v in r["categorical"][s_name]["regime"]["value_counts"].items()}})
    save_table(pd.DataFrame(t4), "T4_split_distribution", "Target and regime distribution per chronological split.")

    t5 = []
    for a, r in A.items():
        mi = r["integrity"]["missingness"]
        for c, v in mi["null_pct"].items():
            mech = mi["mechanism"].get(c, {})
            t5.append({"Asset": a.upper(), "Feature": c, "Null%": v,
                       "FirstValid": (mech.get("first_valid") or "")[:10] or None,
                       "Corr(missing,time)": mech.get("corr_missing_vs_time"),
                       "Verdict": mech.get("verdict", "complete" if v == 0 else "")})
        for c, d in r["integrity"]["zero_as_missing"].items():
            t5.append({"Asset": a.upper(), "Feature": f"{c} == 0 (zero-filled)",
                       "Null%": 100 * np.mean(list(d.values())), "FirstValid": None,
                       "Corr(missing,time)": None,
                       "Verdict": "disguised missingness: train/val/test = " + "/".join(f"{100 * d[s]:.1f}%" for s in ("train", "val", "test"))})
    t5 = pd.DataFrame(t5)
    t5 = t5[(t5["Null%"] > 0) | t5["Feature"].str.contains("==")]
    save_table(t5, "T5_missingness", "Missingness by feature (true NaN and zero-filled), with mechanism diagnostics.")

    t6 = []
    for a, r in A.items():
        it = r["integrity"]
        for o in it["split_overlap"]:
            t6.append({"Asset": a.upper(), "Check": f"overlap {o['pair']}", "ID": o["id_overlap"],
                       "Content hash": o["content_hash_overlap"], "Near-dup": o["near_dup_cross_count"],
                       "Result": "PASS" if o["id_overlap"] == 0 and o["content_hash_overlap"] == 0 else "FAIL"})
        t6.append({"Asset": a.upper(), "Check": "chronological split order", "ID": None, "Content hash": None,
                   "Near-dup": None, "Result": "PASS" if it["temporal"]["train_max_lt_val_min"] and it["temporal"]["val_max_lt_test_min"] else "FAIL"})
        ta = it["target_alignment"]
        t6.append({"Asset": a.upper(), "Check": "Target_RV = next raw trading day", "ID": ta["misaligned_targets"],
                   "Content hash": None, "Near-dup": None, "Result": "PASS" if ta["misaligned_targets"] == 0 else "FAIL"})
        flag = [x["feature"] for x in r["feature_target_association"] if x["leakage_flag"]]
        t6.append({"Asset": a.upper(), "Check": "feature-target abs(Spearman) > 0.95", "ID": len(flag), "Content hash": None,
                   "Near-dup": None, "Result": "PASS" if not flag else "FLAG: " + ",".join(flag)})
        t6.append({"Asset": a.upper(), "Check": "regime terciles use full sample (look-ahead)", "ID": None,
                   "Content hash": None, "Near-dup": None, "Result": "FAIL (data_loader.py:203)"})
    save_table(pd.DataFrame(t6), "T6_leakage_check", "Leakage and integrity checks per asset.")


# ── Markdown dossier ───────────────────────────────────────────────────────────
def write_markdown(P: dict) -> None:
    A, N = P["assets"], P["news"]
    L: list[str] = []
    w = L.append
    w("# 01 — Data Profile (Phase 2)\n")
    w(f"*Generated by `scripts/profile_data.py` (seed {P['seed']}, runtime {P['runtime_seconds']} s). "
      f"Every number below is read from `analysis/artifacts/data_profile.json`. Sampling: {P['sampling']}. "
      f"Split rule: {P['split_rule']}.*\n")
    w("Sections 1–9 are generated automatically; section 10 (interpretation, provenance, and "
      "**what the data can and cannot support**) is maintained in `analysis/notes/01_data_profile_interpretation.md` "
      "and spliced in at the end.\n")

    w("## 1. Datasets, shape and splits (T1)\n")
    w(md_table(pd.read_csv(TAB / "T1_dataset_summary.csv")))
    w("\n### Raw source inventory\n")
    ri = pd.DataFrame(P["raw_inventory"])
    ri["bytes_on_disk"] = (ri["bytes_on_disk"] / 1024).round(1)
    ri = ri.rename(columns={"bytes_on_disk": "disk_KB"})
    ri["memory_bytes"] = (ri.get("memory_bytes") / 1024).round(1)
    ri = ri.rename(columns={"memory_bytes": "memory_KB"})
    w(md_table(ri.fillna(""), floatfmt="{:.1f}"))
    w(f"\nFRED series first valid date: `{P['fred_first_valid']}`; FRED pull ends **{P['fred_last_date']}**.\n")

    for a, r in A.items():
        w(f"## 2.{ASSETS.index(a) + 1} {a.upper()} — column inventory\n")
        w(md_table(pd.DataFrame(r["columns"]), floatfmt="{:.3g}"))
        w("")

    w("## 3. Univariate statistics — numeric (T2; per-split detail in JSON)\n")
    t2 = pd.read_csv(TAB / "T2_numeric_stats.csv")
    w(md_table(t2, floatfmt="{:.4g}"))
    w("\nPer-split versions of every statistic (train / val / test / combined, including all quantiles, "
      "IQR- and z-outliers, modes, CV and normality tests) are in `data_profile.json → assets.<asset>.numeric.<split>`. "
      "Normality: Shapiro–Wilk (all splits have n ≤ 5000).\n")
    for a, r in A.items():
        w(f"### {a.upper()} — outliers and normality (combined)\n")
        rows = []
        for c, s in r["numeric"]["combined"].items():
            nm = s.get("normality") or {}
            rows.append({"feature": c, "IQR1.5": s.get("outliers_iqr_1.5"), "IQR1.5 %": s.get("outliers_iqr_1.5_pct"),
                         "IQR3": s.get("outliers_iqr_3.0"), "IQR3 %": s.get("outliers_iqr_3.0_pct"),
                         "|z|>3": s.get("outliers_z3"), "|z|>3 %": s.get("outliers_z3_pct"),
                         "CV": s.get("cv"), "test": nm.get("test"), "p": nm.get("p"),
                         "verdict": nm.get("interpretation")})
        w(md_table(pd.DataFrame(rows), floatfmt="{:.3g}"))
        w("")

    w("## 4. Categorical and text columns (T3)\n")
    w(md_table(pd.read_csv(TAB / "T3_categorical_stats.csv").fillna(""), floatfmt="{:.3g}"))
    w("\n### News text\n")
    tr = []
    for a, t in N["text"].items():
        tf = t["text_finbert"]
        tr.append({"asset": a, "n": tf["n"], "chars mean": tf["char_length"]["mean"], "chars p95": tf["char_length"]["p95"],
                   "tokens mean": tf["token_length"]["mean"], "tokens median": tf["token_length"]["median"],
                   "tokens max": tf["token_length"]["max"], "vocab": tf["vocab_size"], "dup texts": tf["duplicate_texts"],
                   "OOV test vs train": t["oov_rate_test_vs_train_vocab"],
                   "train/val/test": "/".join(str(t["split_counts"].get(s, 0)) for s in ("train", "val", "test"))})
    w(md_table(pd.DataFrame(tr), floatfmt="{:.3g}"))
    w(f"\nLanguage (langdetect, seed {P['seed']}): `{N['language']}`. {N['oov_note']}. "
      f"Titles appearing in more than one split: {N['duplicate_title_across_splits']}.\n")

    w("## 5. Target variable\n")
    for a, r in A.items():
        t = r["target"]
        ls, rs = t["log_scale"], t["raw_variance_scale"]
        w(f"**{a.upper()}** — log scale: mean {ls['mean']:.4g}, std {ls['std']:.4g}, skew {ls['skewness']:.3g}, "
          f"excess kurtosis {ls['kurtosis_excess']:.3g}; raw variance scale: skew {rs['skewness']:.3g}, "
          f"excess kurtosis {rs['kurtosis_excess']:.3g}; Box–Cox λ on raw variance = {t['boxcox_lambda_on_raw_variance']:.3f}.\n")
    w("\n### What score is meaningful? Naive forecasts on the test split\n")
    rows = []
    for a, r in A.items():
        for k, v in r["naive_baselines_test"].items():
            rows.append({"asset": a.upper(), "naive forecast": k, **v})
    w(md_table(pd.DataFrame(rows), floatfmt="{:.4g}"))
    w("")

    w("## 6. Multivariate and relational (train split only)\n")
    for a, r in A.items():
        w(f"### {a.upper()}\n")
        w(f"![Pearson]({r['figures']['corr'][0]}) ![Spearman]({r['figures']['corr'][1]})\n")
        w("Feature–target association (sorted by mutual information):\n")
        w(md_table(pd.DataFrame(r["feature_target_association"]), floatfmt="{:.3g}"))
        w(f"\nVIF: `{ {k: round(v, 2) for k, v in r['vif'].items()} }` ({r['vif_note']}).  ")
        rv = r["regime_vs_target"]
        w(f"Regime vs target: Cramér's V (target terciles) = {rv['cramers_v_regime_vs_target_tercile']:.3f}, "
          f"Kruskal–Wallis p = {rv['kruskal_p'] if rv['kruskal_p'] is not None else float('nan'):.3g}; "
          f"train regime counts `{rv['regime_counts_train']}`.\n")
        w(f"![pairplot]({r['figures']['pairplot']})\n")

    w("## 7. Integrity and quality (T5, T6)\n")
    w(md_table(pd.read_csv(TAB / "T6_leakage_check.csv").fillna(""), floatfmt="{:.3g}"))
    w("")
    for a, r in A.items():
        it = r["integrity"]
        w(f"### {a.upper()}\n")
        w(f"- Exact duplicate rows (incl. date): {it['exact_duplicate_rows_incl_date']}; content-only: "
          f"{it['exact_duplicate_rows_content_only']}; near-duplicates: {it['near_duplicate_rows']['count']} "
          f"({it['near_duplicate_rows']['definition']}).")
        for o in it["split_overlap"]:
            w(f"- Split overlap {o['pair']}: ID {o['id_overlap']}, content hash {o['content_hash_overlap']}, "
              f"near-dup {o['near_dup_cross_count']} ({o['near_dup_cross_pct']:.2f}% of the later split below the "
              f"1st percentile of within-split NN distance).")
        tmp = it["temporal"]
        w(f"- Temporal: monotonic {tmp['index_monotonic_increasing']}, duplicate dates {tmp['duplicate_dates']}, "
          f"train<val {tmp['train_max_lt_val_min']}, val<test {tmp['val_max_lt_test_min']}, "
          f"dates after data pull {tmp['future_dates_after_pull']}. {tmp['note']}.")
        ta = it["target_alignment"]
        w(f"- **Target alignment:** {ta['misaligned_targets']} of {ta['rows_checked']} rows have a Target_RV that is not "
          f"the next raw trading day's log RV. Examples: `{ta['examples'][:4]}`")
        for g in it["calendar_gaps_gt_5d"]:
            w(f"- Calendar gap {g['from']} → {g['to']} ({g['days']:.0f} days, {g['raw_rows_dropped']} raw rows dropped; "
              f"zero-return day(s) {g['zero_return_days']}): {g['cause']}.")
        vc = it["value_checks"]
        w(f"- Value checks: constant placeholder columns `{vc['constant_placeholders']}`; bounds "
          f"`{ {k: v for k, v in vc.items() if isinstance(v, dict)} }`; log_RV below mean−5sd: "
          f"{vc['log_RV_extreme_low_lt_mean_minus_5sd']}. {vc['unit_note']}.")
        w(f"- Zero-filled sentiment share by split: `{it['zero_as_missing']}`")
        for c, m in it["missingness"]["mechanism"].items():
            w(f"- Missingness `{c}`: {m['verdict']} (first valid {m['first_valid'][:10]}, last valid "
              f"{m['last_valid'][:10]}, corr with time {m['corr_missing_vs_time']:.3f}).")
        for f_ in r["figures"]["missing"]:
            w(f"\n![missing]({f_})")
        w("")
    w("\n### Distribution drift — train vs test (KS for numerics, χ² for regime)\n")
    rows = []
    for a, r in A.items():
        for d in r["integrity"]["drift_train_vs_test"]:
            rows.append({"asset": a.upper(), "feature": d["feature"], "test": d["test"], "stat": d.get("stat"),
                         "p": d.get("p"), "non-iid (p<0.01)": (d.get("p") is not None and d["p"] < 0.01)})
    w(md_table(pd.DataFrame(rows), floatfmt="{:.3g}"))
    w("\nTrain-vs-val and val-vs-test drift tests: `data_profile.json → integrity.drift_*`.\n")

    w("## 8. Figures index\n")
    for a, r in A.items():
        w(f"**{a.upper()}:** target by split ![t]({r['figures']['target_by_split']}) drift ![d]({r['figures']['drift']}) "
          f"PCA/t-SNE ![p]({r['figures']['pca_tsne']}) regime ![r]({r['figures']['regime_bar']})\n")
        w("Per-feature histogram+KDE / box / violin: " + ", ".join(f"[{Path(f).stem}]({f})" for f in r["figures"]["numeric"]) + "\n")
    w("## 9. PCA\n")
    for a, r in A.items():
        ev = np.cumsum(r["pca"]["explained_variance_ratio"])
        w(f"- {a.upper()}: {len(ev)} standardised inputs; cumulative variance of first 3 PCs = "
          f"{', '.join(f'{v:.3f}' for v in ev[:3])}.")
    w("")
    if NOTES.exists():
        w(NOTES.read_text())
    else:
        w("## 10. Interpretation\n\n*(pending: analysis/notes/01_data_profile_interpretation.md not found)*\n")
    (AN / "01_data_profile.md").write_text("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
