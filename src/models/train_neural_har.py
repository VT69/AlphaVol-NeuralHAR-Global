"""
Neural-HAR Training Runner
===========================
Trains the GRN-corrected Neural-HAR on real feature matrices.

Strategy:
  - Single in-sample train (full expanding window is too slow on CPU)
  - HAR weights initialized from the pre-fitted OLS betas (warm start)
  - GRN residual learned on FinBERT + microstructure features
  - Saves OOS forecasts as  data/processed/forecasts_neural_har_{asset}.npy

Usage:
    python src/models/train_neural_har.py --asset btc
    python src/models/train_neural_har.py --asset spx
    python src/models/train_neural_har.py --asset nifty
    python src/models/train_neural_har.py   # all assets
"""

import os
import sys
import argparse
import logging
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.models.neural_har import NeuralHAR, train_model, qlike_loss

PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
MODEL_DIR     = os.path.join(ROOT, "data", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

ASSETS  = ["btc", "spx", "nifty"]
HAR_COLS = ["RV_d", "RV_w", "RV_m"]

# Exogenous candidates — use whatever is available per asset
EXO_CANDIDATES = [
    "FinBERT_score_lag1",
    "sent_surprise_lag1",
    "vpin_lag1",
    "obi_sq_lag1",
    "illiq_lag1",
    "crypto_fg_lag1",
    "term_slope_lag1",
    "credit_spread_lag1",
]


def _get_har_ols_init(df: pd.DataFrame, har_cols, target="Target_RV"):
    """Fit OLS HAR to get warm-start weights for the neural layer."""
    mask = df[har_cols + [target]].notnull().all(axis=1)
    X = sm.add_constant(df.loc[mask, har_cols])
    y = df.loc[mask, target]
    res = sm.OLS(y, X).fit()
    intercept = float(res.params["const"])
    betas = np.array([float(res.params[c]) for c in har_cols], dtype=np.float32)
    return betas, intercept


def train_asset(asset: str,
                train_frac: float = 0.80,
                val_frac: float = 0.10,
                epochs: int = 100,
                lr: float = 5e-4,
                hidden_dim: int = 32,
                dropout: float = 0.3,
                batch_size: int = 64):
    """
    Full training pipeline for one asset.

    Train / Val / Test split (chronological):
        train_frac=0.80, val_frac=0.10 -> test_frac=0.10

    Returns numpy arrays of OOS (test) predictions and actuals.
    """
    logger.info(f"\n{'='*65}")
    logger.info(f"  Neural-HAR Training | Asset: {asset.upper()}")
    logger.info(f"{'='*65}")

    # ── Load feature matrix ──────────────────────────────────────────
    path = os.path.join(PROCESSED_DIR, f"{asset}_feature_matrix.parquet")
    if not os.path.exists(path):
        logger.error(f"  Feature matrix not found: {path}")
        return None, None
    df = pd.read_parquet(path).sort_index()

    # ── Select exogenous features available for this asset ───────────
    exo_cols = [c for c in EXO_CANDIDATES if c in df.columns]
    logger.info(f"  HAR cols : {HAR_COLS}")
    logger.info(f"  EXO cols : {exo_cols}")

    needed = HAR_COLS + exo_cols + ["Target_RV"]
    df = df.dropna(subset=["RV_d", "RV_w", "RV_m", "Target_RV"])

    # Forward-fill sparse exogenous columns (e.g. weekly FRED data)
    for c in exo_cols:
        if c in df.columns:
            df[c] = df[c].ffill().fillna(0.0)

    df = df.dropna(subset=HAR_COLS + ["Target_RV"])
    logger.info(f"  Dataset  : {len(df)} rows | {df.index.min().date()} -> {df.index.max().date()}")

    # ── Chronological split ──────────────────────────────────────────
    n        = len(df)
    n_train  = int(n * train_frac)
    n_val    = int(n * val_frac)
    n_test   = n - n_train - n_val

    train_df = df.iloc[:n_train]
    val_df   = df.iloc[n_train : n_train + n_val]
    test_df  = df.iloc[n_train + n_val:]

    logger.info(f"  Split    : Train={n_train} | Val={n_val} | Test={n_test}")

    # ── Warm-start HAR weights from OLS ─────────────────────────────
    betas, intercept = _get_har_ols_init(train_df, HAR_COLS)
    logger.info(f"  OLS init : const={intercept:.4f}  betas={betas.round(4).tolist()}")

    # ── Scale exogenous features (fit on train only!) ────────────────
    if exo_cols:
        scaler = StandardScaler()
        train_exo = scaler.fit_transform(train_df[exo_cols].values.astype(np.float32))
        val_exo   = scaler.transform(val_df[exo_cols].values.astype(np.float32))
        test_exo  = scaler.transform(test_df[exo_cols].values.astype(np.float32))
    else:
        # No exogenous — use zeros
        train_exo = np.zeros((n_train, 1), dtype=np.float32)
        val_exo   = np.zeros((n_val, 1),   dtype=np.float32)
        test_exo  = np.zeros((n_test, 1),  dtype=np.float32)
        exo_cols  = ["zero"]

    num_exo = train_exo.shape[1]

    def _to_tensor(df_split, exo_arr):
        X_har = torch.FloatTensor(df_split[HAR_COLS].values)
        X_exo = torch.FloatTensor(exo_arr)
        y     = torch.FloatTensor(df_split["Target_RV"].values.reshape(-1, 1))
        return TensorDataset(X_har, X_exo, y)

    train_ds = _to_tensor(train_df, train_exo)
    val_ds   = _to_tensor(val_df,   val_exo)
    test_ds  = _to_tensor(test_df,  test_exo)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=256,         shuffle=False)

    # ── Build model ───────────────────────────────────────────────────
    model_path = os.path.join(MODEL_DIR, f"neural_har_{asset}.pth")
    model = NeuralHAR(
        num_har_features=3,
        num_exo_features=num_exo,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
    model.init_har_weights(
        beta_weights=betas.reshape(1, -1),
        intercept=intercept,
    )
    logger.info(f"  {model}")

    # ── Train ─────────────────────────────────────────────────────────
    # Temporarily patch early_stopper save path
    from src.models.neural_har import EarlyStopping
    stopper = EarlyStopping(patience=15, save_path=model_path)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    torch.manual_seed(42)
    np.random.seed(42)

    best_val = np.inf
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for bx_har, bx_exo, by in train_loader:
            optimizer.zero_grad()
            pred = model(bx_har, bx_exo)
            loss = qlike_loss(pred, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx_har, bx_exo, by in val_loader:
                val_loss += qlike_loss(model(bx_har, bx_exo), by).item()
        val_loss /= len(val_loader)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch+1:>3}/{epochs} | Train QLIKE: {train_loss:.5f} | Val QLIKE: {val_loss:.5f}")

        stopper(val_loss, model)
        if stopper.stop:
            logger.info(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best weights
    model.load_state_dict(torch.load(model_path, weights_only=True))

    # ── OOS Evaluation (test set) ─────────────────────────────────────
    model.eval()
    preds_list, acts_list = [], []
    with torch.no_grad():
        for bx_har, bx_exo, by in test_loader:
            preds_list.append(model(bx_har, bx_exo).squeeze().numpy())
            acts_list.append(by.squeeze().numpy())

    preds_arr = np.concatenate(preds_list)
    acts_arr  = np.concatenate(acts_list)

    # ── OOS Metrics ───────────────────────────────────────────────────
    av = np.exp(np.clip(acts_arr,  -15, 15))
    pv = np.exp(np.clip(preds_arr, -15, 15))
    qlike_oos = float(np.mean(av / pv - np.log(av / pv) - 1))
    mse_oos   = float(np.mean((acts_arr - preds_arr) ** 2))
    mae_oos   = float(np.mean(np.abs(acts_arr - preds_arr)))

    logger.info(f"\n  OOS Test Results ({n_test} periods):")
    logger.info(f"    QLIKE : {qlike_oos:.4f}")
    logger.info(f"    MSE   : {mse_oos:.4f}")
    logger.info(f"    MAE   : {mae_oos:.4f}")

    # ── Save forecasts ────────────────────────────────────────────────
    fc_path  = os.path.join(PROCESSED_DIR, f"forecasts_neural_har_{asset}.npy")
    act_path = os.path.join(PROCESSED_DIR, f"actuals_neural_{asset}.npy")
    np.save(fc_path,  preds_arr)
    np.save(act_path, acts_arr)
    logger.info(f"  Saved forecasts -> {fc_path}")
    logger.info(f"  Saved model     -> {model_path}")

    return preds_arr, acts_arr


def main():
    parser = argparse.ArgumentParser(description="Train Neural-HAR on real feature matrices.")
    parser.add_argument("--asset", type=str, default=None,
                        choices=["btc", "spx", "nifty"],
                        help="Asset to train on. Omit for all assets.")
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--lr",         type=float, default=5e-4)
    parser.add_argument("--hidden_dim", type=int,   default=32)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--train_frac", type=float, default=0.80)
    args = parser.parse_args()

    assets = [args.asset] if args.asset else ASSETS

    for asset in assets:
        train_asset(
            asset,
            train_frac=args.train_frac,
            epochs=args.epochs,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            batch_size=args.batch_size,
        )

    logger.info("\nNeural-HAR training complete for all assets.")
    logger.info("Run 'python run_paper.py --skip_build --step dm' to include Neural-HAR in DM/MCS tests.")


if __name__ == "__main__":
    main()
