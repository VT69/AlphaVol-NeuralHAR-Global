"""
Neural-HAR Training Runner
===========================
Trains the GRN-corrected Neural-HAR on real feature matrices.

Strategy:
  - Hyperparameter tuning with Optuna on the initial Train/Val split.
  - Walk-Forward (Expanding Window) fine-tuning for the Out-Of-Sample test set.
  - HAR weights initialized from the pre-fitted OLS betas (warm start).
  - GRN residual learned on FinBERT + microstructure features.
  - Saves OOS forecasts as data/processed/forecasts_neural_har_{asset}.npy

Usage:
    python src/models/train_neural_har.py --asset btc
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
import optuna

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.models.neural_har import NeuralHAR, qlike_loss

PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
MODEL_DIR     = os.path.join(ROOT, "data", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

ASSETS  = ["btc", "spx", "nifty"]
HAR_COLS = ["RV_d", "RV_w", "RV_m"]

# Exogenous candidates
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
    mask = df[har_cols + [target]].notnull().all(axis=1)
    X = sm.add_constant(df.loc[mask, har_cols])
    y = df.loc[mask, target]
    res = sm.OLS(y, X).fit()
    intercept = float(res.params["const"])
    betas = np.array([float(res.params[c]) for c in har_cols], dtype=np.float32)
    return betas, intercept


def _to_tensor(df_split, exo_arr):
    X_har = torch.FloatTensor(df_split[HAR_COLS].values)
    X_exo = torch.FloatTensor(exo_arr)
    y     = torch.FloatTensor(df_split["Target_RV"].values.reshape(-1, 1))
    return TensorDataset(X_har, X_exo, y)


def run_optuna_tuning(train_loader, val_loader, num_exo, betas, intercept, n_trials=15):
    """Run Optuna to find best hyperparameters."""
    logger.info(f"  Starting Optuna Hyperparameter Tuning ({n_trials} trials)...")
    
    def objective(trial):
        hidden_dim = trial.suggest_categorical("hidden_dim", [16, 32, 64])
        dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        
        model = NeuralHAR(num_har_features=3, num_exo_features=num_exo, hidden_dim=hidden_dim, dropout=dropout)
        model.init_har_weights(beta_weights=betas.reshape(1, -1), intercept=intercept)
        
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        best_val = np.inf
        patience = 5
        patience_counter = 0
        
        for epoch in range(30): # short training for HPO
            model.train()
            for bx_har, bx_exo, by in train_loader:
                optimizer.zero_grad()
                loss = qlike_loss(model(bx_har, bx_exo), by)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for bx_har, bx_exo, by in val_loader:
                    val_loss += qlike_loss(model(bx_har, bx_exo), by).item()
            val_loss /= len(val_loader)
            
            if val_loss < best_val:
                best_val = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
                    
        return best_val

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    logger.info(f"  Best params found: {best_params} (Val QLIKE: {study.best_value:.4f})")
    return best_params


def train_asset(asset: str,
                train_frac: float = 0.80,
                val_frac: float = 0.10,
                n_trials: int = 15,
                step_size: int = 21):
    """
    Full training pipeline with HPO and Walk-Forward Expanding Window.
    """
    logger.info(f"\n{'='*65}")
    logger.info(f"  Neural-HAR | Asset: {asset.upper()} | Expanding Window: {step_size} days")
    logger.info(f"{'='*65}")

    path = os.path.join(PROCESSED_DIR, f"{asset}_feature_matrix.parquet")
    if not os.path.exists(path):
        logger.error(f"  Feature matrix not found: {path}")
        return None, None
    df = pd.read_parquet(path).sort_index()

    exo_cols = [c for c in EXO_CANDIDATES if c in df.columns]
    df = df.dropna(subset=["RV_d", "RV_w", "RV_m", "Target_RV"])
    for c in exo_cols:
        if c in df.columns:
            df[c] = df[c].ffill().fillna(0.0)
    df = df.dropna(subset=HAR_COLS + ["Target_RV"])

    n        = len(df)
    n_train  = int(n * train_frac)
    n_val    = int(n * val_frac)
    n_test   = n - n_train - n_val

    # Initial static split for HPO
    train_df = df.iloc[:n_train]
    val_df   = df.iloc[n_train : n_train + n_val]
    
    logger.info(f"  Total    : {n} rows | {df.index.min().date()} -> {df.index.max().date()}")
    logger.info(f"  Split    : Train={n_train} | Val={n_val} | Test={n_test}")

    # Scale Exogenous (fit on initial train)
    scaler = StandardScaler()
    if exo_cols:
        scaler.fit(train_df[exo_cols].values.astype(np.float32))
        train_exo = scaler.transform(train_df[exo_cols].values.astype(np.float32))
        val_exo   = scaler.transform(val_df[exo_cols].values.astype(np.float32))
    else:
        train_exo = np.zeros((n_train, 1), dtype=np.float32)
        val_exo   = np.zeros((n_val, 1),   dtype=np.float32)
        exo_cols = ["zero"]
    num_exo = train_exo.shape[1]

    train_ds = _to_tensor(train_df, train_exo)
    val_ds   = _to_tensor(val_df,   val_exo)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)

    betas, intercept = _get_har_ols_init(train_df, HAR_COLS)

    # 1. Hyperparameter Tuning
    best_p = run_optuna_tuning(train_loader, val_loader, num_exo, betas, intercept, n_trials=n_trials)
    
    # 2. Walk-Forward Expanding Window Evaluation
    logger.info("  Starting Walk-Forward Expanding Window Evaluation...")
    
    # Instantiate best model
    model = NeuralHAR(num_har_features=3, num_exo_features=num_exo, 
                      hidden_dim=best_p["hidden_dim"], dropout=best_p["dropout"])
    model.init_har_weights(beta_weights=betas.reshape(1, -1), intercept=intercept)
    optimizer = optim.Adam(model.parameters(), lr=best_p["lr"], weight_decay=best_p["weight_decay"])
    
    # Initial Pre-train on Train+Val
    logger.info("  Pre-training on initial in-sample window (Train + Val)...")
    init_df = df.iloc[:n_train + n_val]
    init_exo = scaler.transform(init_df[exo_cols].values.astype(np.float32)) if exo_cols != ["zero"] else np.zeros((len(init_df), 1), dtype=np.float32)
    init_loader = DataLoader(_to_tensor(init_df, init_exo), batch_size=64, shuffle=False)
    
    for epoch in range(50):
        model.train()
        for bx_har, bx_exo, by in init_loader:
            optimizer.zero_grad()
            loss = qlike_loss(model(bx_har, bx_exo), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
    # Walk-Forward
    test_start = n_train + n_val
    preds_list = []
    acts_list = []
    
    while test_start < n:
        test_end = min(test_start + step_size, n)
        
        # OOS Predict
        model.eval()
        oos_df = df.iloc[test_start:test_end]
        oos_exo = scaler.transform(oos_df[exo_cols].values.astype(np.float32)) if exo_cols != ["zero"] else np.zeros((len(oos_df), 1), dtype=np.float32)
        oos_ds = _to_tensor(oos_df, oos_exo)
        oos_loader = DataLoader(oos_ds, batch_size=max(len(oos_df), 1), shuffle=False)
        
        with torch.no_grad():
            for bx_har, bx_exo, by in oos_loader:
                preds = model(bx_har, bx_exo).squeeze().numpy()
                acts = by.squeeze().numpy()
                if preds.ndim == 0:
                    preds = np.array([preds])
                    acts = np.array([acts])
                preds_list.append(preds)
                acts_list.append(acts)
                
        # Expanding Window Finetune
        if test_end < n:
            expand_df = df.iloc[:test_end]
            expand_exo = scaler.transform(expand_df[exo_cols].values.astype(np.float32)) if exo_cols != ["zero"] else np.zeros((len(expand_df), 1), dtype=np.float32)
            expand_loader = DataLoader(_to_tensor(expand_df, expand_exo), batch_size=64, shuffle=False)
            
            # Finetune for 5 epochs
            model.train()
            for epoch in range(5):
                for bx_har, bx_exo, by in expand_loader:
                    optimizer.zero_grad()
                    loss = qlike_loss(model(bx_har, bx_exo), by)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
        test_start = test_end
        
    preds_arr = np.concatenate(preds_list)
    acts_arr  = np.concatenate(acts_list)

    # Metrics
    av = np.exp(np.clip(acts_arr,  -15, 15))
    pv = np.exp(np.clip(preds_arr, -15, 15))
    qlike_oos = float(np.mean(av / pv - np.log(av / pv) - 1))
    mse_oos   = float(np.mean((acts_arr - preds_arr) ** 2))
    mae_oos   = float(np.mean(np.abs(acts_arr - preds_arr)))

    logger.info(f"\n  OOS Test Results ({n_test} periods):")
    logger.info(f"    QLIKE : {qlike_oos:.4f}")
    logger.info(f"    MSE   : {mse_oos:.4f}")
    logger.info(f"    MAE   : {mae_oos:.4f}")

    # Save
    fc_path  = os.path.join(PROCESSED_DIR, f"forecasts_neural_har_{asset}.npy")
    act_path = os.path.join(PROCESSED_DIR, f"actuals_neural_{asset}.npy")
    np.save(fc_path,  preds_arr)
    np.save(act_path, acts_arr)
    
    model_path = os.path.join(MODEL_DIR, f"neural_har_{asset}.pth")
    torch.save(model.state_dict(), model_path)
    
    logger.info(f"  Saved forecasts -> {fc_path}")
    logger.info(f"  Saved model     -> {model_path}")

    return preds_arr, acts_arr


def main():
    parser = argparse.ArgumentParser(description="Train Neural-HAR on real feature matrices with Walk-Forward.")
    parser.add_argument("--asset", type=str, default=None,
                        choices=["btc", "spx", "nifty"],
                        help="Asset to train on. Omit for all assets.")
    parser.add_argument("--n_trials", type=int, default=15)
    parser.add_argument("--step_size", type=int, default=21)
    args = parser.parse_args()

    assets = [args.asset] if args.asset else ASSETS

    for asset in assets:
        train_asset(
            asset,
            n_trials=args.n_trials,
            step_size=args.step_size,
        )

    logger.info("\nNeural-HAR training complete for all assets.")
    logger.info("Run 'python run_paper.py --skip_build --step dm' to include Neural-HAR in DM/MCS tests.")

if __name__ == "__main__":
    main()
