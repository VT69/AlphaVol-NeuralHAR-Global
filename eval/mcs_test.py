"""
Model Confidence Set (MCS) — Hansen, Lunde & Nason (2011)
==========================================================
Reference: Hansen, P.R., Lunde, A., and Nason, J.M. (2011).
           "The Model Confidence Set." Econometrica, 79(2), 453–497.

The MCS procedure iteratively eliminates the worst-performing model
until the null of equal predictive ability (EPA) cannot be rejected
at the specified confidence level alpha. The surviving set forms the MCS.

We implement the TR (range) and TMax statistics and use bootstrap
(circular block bootstrap) for critical values.

Usage:
    from eval.mcs_test import mcs_test, print_mcs_results

    # losses: pd.DataFrame shape (T, M) — T periods × M models
    # lower is better (e.g. QLIKE, MSE, MAE)
    results = mcs_test(losses, alpha=0.10, B=1000, block_size=5)
    print_mcs_results(results)
"""

import numpy as np
import pandas as pd
import warnings
from typing import Union


# ─────────────────────────────────────────────────────────────────────────────
# Core statistics
# ─────────────────────────────────────────────────────────────────────────────

def _loss_diffs(losses: np.ndarray) -> np.ndarray:
    """
    Compute pairwise loss differentials d_{ij,t} = L_{i,t} - L_{j,t}.
    Returns array of shape (T, M, M).
    """
    T, M = losses.shape
    d = losses[:, :, None] - losses[:, None, :]   # (T, M, M)
    return d


def _d_bar(d: np.ndarray) -> np.ndarray:
    """Mean loss differential over T. Shape (M, M)."""
    return d.mean(axis=0)


def _T_R(d: np.ndarray, d_bar: np.ndarray, var_d_bar: np.ndarray) -> float:
    """
    TR statistic — max absolute standardized mean loss differential.
    TR = max_{i,j} |d̄_{ij}| / sqrt(var(d̄_{ij}))
    """
    M = d_bar.shape[0]
    stats = []
    for i in range(M):
        for j in range(i + 1, M):
            v = var_d_bar[i, j]
            if v > 1e-20:
                stats.append(abs(d_bar[i, j]) / np.sqrt(v))
    return float(np.max(stats)) if stats else 0.0


def _T_max(d_bar: np.ndarray, var_d_bar: np.ndarray) -> float:
    """
    TMax statistic — max over i of standardized mean loss vs model set mean.
    t_i = d̄_i. / sqrt(var(d̄_i.)) where d̄_i. = (1/M) Σ_j d̄_{ij}
    """
    M = d_bar.shape[0]
    t_vals = []
    for i in range(M):
        d_i_dot = d_bar[i, :].mean()
        # variance of d̄_i. using diagonal block of var matrix
        v = var_d_bar[i, :].mean() / M
        if v > 1e-20:
            t_vals.append(d_i_dot / np.sqrt(v))
    return float(np.max(t_vals)) if t_vals else 0.0


def _bootstrap_variance(d: np.ndarray, block_size: int = 5,
                          B: int = 1000, seed: int = 42) -> np.ndarray:
    """
    Circular block bootstrap estimate of variance of d̄_{ij}.
    Returns var_d_bar of shape (M, M).
    """
    rng = np.random.default_rng(seed)
    T, M, _ = d.shape
    boot_means = np.zeros((B, M, M))

    n_blocks = int(np.ceil(T / block_size))
    for b in range(B):
        starts = rng.integers(0, T, size=n_blocks)
        idx = np.concatenate([
            np.arange(s, s + block_size) % T for s in starts
        ])[:T]
        boot_means[b] = d[idx].mean(axis=0)

    return boot_means.var(axis=0, ddof=1)


def _eliminate_worst(losses_sub: np.ndarray,
                       model_names: list,
                       stat: str = "TR",
                       B: int = 1000,
                       block_size: int = 5,
                       seed: int = 42) -> tuple:
    """
    Identify the model to eliminate (worst under TR or TMax).
    Returns (eliminated_index_in_sub, t_stat_value).
    """
    d = _loss_diffs(losses_sub)
    db = _d_bar(d)
    var_db = _bootstrap_variance(d, block_size, B, seed)

    M = losses_sub.shape[1]

    if stat == "TR":
        # Eliminate i* = argmax_i (max_j t_{ij})
        max_t = np.full(M, -np.inf)
        for i in range(M):
            for j in range(M):
                if i == j:
                    continue
                v = var_db[i, j]
                if v > 1e-20:
                    max_t[i] = max(max_t[i], abs(db[i, j]) / np.sqrt(v))
        # Eliminate the one with the largest max-t (worst model)
        # but sign matters: eliminate the worst (highest mean loss)
        mean_loss = losses_sub.mean(axis=0)
        worst = int(np.argmax(mean_loss))
        t_val = _T_R(d, db, var_db)
    else:
        # TMax: t_i for each model
        t_i = np.zeros(M)
        for i in range(M):
            d_i_dot = db[i, :].mean()
            v = var_db[i, :].mean() / M
            t_i[i] = d_i_dot / np.sqrt(max(v, 1e-20))
        worst = int(np.argmax(t_i))
        t_val = float(t_i[worst])

    return worst, t_val


def _bootstrap_critical_value(d: np.ndarray,
                                var_db: np.ndarray,
                                stat: str,
                                B: int,
                                block_size: int,
                                seed: int) -> np.ndarray:
    """
    Simulate the null distribution of T_stat under H0 (EPA).
    Returns sorted array of B bootstrap test statistics.
    """
    rng = np.random.default_rng(seed + 1)
    T, M, _ = d.shape
    n_blocks = int(np.ceil(T / block_size))
    boot_stats = np.zeros(B)

    for b in range(B):
        starts = rng.integers(0, T, size=n_blocks)
        idx = np.concatenate([
            np.arange(s, s + block_size) % T for s in starts
        ])[:T]
        d_boot = d[idx]
        db_boot = d_boot.mean(axis=0)
        # Centre under H0
        db_centred = db_boot - d.mean(axis=0)

        if stat == "TR":
            stats = []
            for i in range(M):
                for j in range(i + 1, M):
                    v = var_db[i, j]
                    if v > 1e-20:
                        stats.append(abs(db_centred[i, j]) / np.sqrt(v))
            boot_stats[b] = max(stats) if stats else 0.0
        else:
            t_i = np.zeros(M)
            for i in range(M):
                d_i = db_centred[i, :].mean()
                v = var_db[i, :].mean() / M
                t_i[i] = d_i / np.sqrt(max(v, 1e-20))
            boot_stats[b] = t_i.max()

    return np.sort(boot_stats)


# ─────────────────────────────────────────────────────────────────────────────
# Main MCS procedure
# ─────────────────────────────────────────────────────────────────────────────

def mcs_test(losses: Union[pd.DataFrame, np.ndarray],
             alpha: float = 0.10,
             B: int = 1000,
             block_size: int = 5,
             stat: str = "TR",
             seed: int = 42) -> pd.DataFrame:
    """
    Model Confidence Set procedure (Hansen et al. 2011).

    Parameters
    ----------
    losses      : (T × M) array/DataFrame of period-by-period losses.
                  Lower is better (QLIKE, MSE, MAE).
    alpha       : Significance level for elimination (default 0.10).
    B           : Bootstrap replications (default 1000).
    block_size  : Circular block bootstrap block length (default 5).
    stat        : Test statistic: 'TR' (range) or 'TMax'.
    seed        : RNG seed for reproducibility.

    Returns
    -------
    pd.DataFrame with columns:
        model         : model name
        mean_loss     : average period loss
        MCS_pvalue    : p-value at elimination step (NaN if in MCS)
        in_MCS        : bool — True if model survives at level alpha
        rank          : MCS rank (1 = best; eliminated models get rank > M_MCS)
        eliminated_at : step at which eliminated (NaN if in MCS)
    """
    if isinstance(losses, pd.DataFrame):
        names = list(losses.columns)
        L = losses.values.astype(float)
    else:
        L = np.asarray(losses, dtype=float)
        names = [f"M{i}" for i in range(L.shape[1])]

    T, M = L.shape
    if M < 2:
        raise ValueError("Need at least 2 models for MCS.")

    surviving = list(range(M))
    results = {i: {"model": names[i],
                   "mean_loss": float(L[:, i].mean()),
                   "MCS_pvalue": np.nan,
                   "in_MCS": True,
                   "rank": np.nan,
                   "eliminated_at": np.nan}
               for i in range(M)}

    step = 0
    p_values = []   # store (model_idx, p_value) for each elimination

    while len(surviving) > 1:
        step += 1
        L_sub = L[:, surviving]
        d = _loss_diffs(L_sub)
        db = _d_bar(d)
        var_db = _bootstrap_variance(d, block_size, B, seed)
        boot_dist = _bootstrap_critical_value(d, var_db, stat, B, block_size, seed)

        # Observed test statistic
        if stat == "TR":
            t_obs = _T_R(d, db, var_db)
        else:
            M_sub = L_sub.shape[1]
            t_i = np.zeros(M_sub)
            for i in range(M_sub):
                d_i_dot = db[i, :].mean()
                v = var_db[i, :].mean() / M_sub
                t_i[i] = d_i_dot / np.sqrt(max(v, 1e-20))
            t_obs = float(t_i.max())

        p_val = float(np.mean(boot_dist >= t_obs))
        p_val = max(p_val, max([r["MCS_pvalue"] for r in results.values()
                                 if not np.isnan(r["MCS_pvalue"])], default=0.0))

        if p_val >= alpha:
            break   # All remaining models in MCS

        # Eliminate worst
        worst_sub, _ = _eliminate_worst(L_sub, [names[s] for s in surviving],
                                         stat, B, block_size, seed)
        worst_global = surviving[worst_sub]
        results[worst_global]["in_MCS"] = False
        results[worst_global]["MCS_pvalue"] = p_val
        results[worst_global]["eliminated_at"] = step
        results[worst_global]["rank"] = len(surviving)
        surviving.pop(worst_sub)

    # Assign ranks to surviving models (by mean loss)
    survivors_sorted = sorted(surviving, key=lambda i: results[i]["mean_loss"])
    for rank_pos, idx in enumerate(survivors_sorted, start=1):
        results[idx]["rank"] = rank_pos
        results[idx]["MCS_pvalue"] = 1.0   # convention: p=1 for last step

    df_out = pd.DataFrame(list(results.values()))
    df_out = df_out.sort_values("rank").reset_index(drop=True)
    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# Printing utilities
# ─────────────────────────────────────────────────────────────────────────────

def print_mcs_results(df: pd.DataFrame,
                       title: str = "Model Confidence Set Results",
                       alpha: float = 0.10):
    """Pretty-print the MCS table for inclusion in the paper."""
    print(f"\n{'='*65}")
    print(f"  {title}  (alpha = {alpha})")
    print(f"{'='*65}")
    cols = ["model", "mean_loss", "MCS_pvalue", "in_MCS", "rank"]
    print(df[cols].to_string(index=False))
    print(f"{'='*65}")
    n_mcs = df["in_MCS"].sum()
    mcs_models = df.loc[df["in_MCS"], "model"].tolist()
    print(f"  MCS contains {n_mcs} model(s): {', '.join(mcs_models)}")
    print(f"{'='*65}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)
    T = 500

    # Simulate log(RV) losses (QLIKE-like, positive)
    # Model A: worst, Model B: medium, Model C: best
    base = np.abs(np.random.normal(0.5, 0.3, T))
    losses_A = base + np.abs(np.random.normal(0.20, 0.1, T))
    losses_B = base + np.abs(np.random.normal(0.10, 0.1, T))
    losses_C = base + np.abs(np.random.normal(0.02, 0.1, T))

    L = pd.DataFrame({"HAR": losses_A, "HAR-S": losses_B, "Neural-HAR": losses_C})

    print("Running MCS (TR statistic, alpha=0.10, B=500)...")
    result = mcs_test(L, alpha=0.10, B=500, block_size=5, stat="TR")
    print_mcs_results(result, title="Self-Test MCS", alpha=0.10)
