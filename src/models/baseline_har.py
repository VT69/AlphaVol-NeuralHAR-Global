import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging

# Setup Logging (Issue 12)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def qlike_loss_numpy(pred_log_rv, actual_log_rv):
    """
    QLIKE = mean(actual_var/pred_var - log(actual_var/pred_var) - 1)
    Calculated on variance (exp(log_RV)).
    """
    actual_var = np.exp(actual_log_rv)
    pred_var = np.exp(np.clip(pred_log_rv, -10, 10))
    return np.mean(actual_var / pred_var - np.log(actual_var / pred_var) - 1)

def expanding_window_forecast(df, min_train=252):
    """
    Performs an expanding window out-of-sample (OOS) forecast. (Issue 2)
    """
    logger.info(f"Starting expanding window forecast. Minimum training size: {min_train} days.")
    
    # Issue 1: Target and features must be log(RV)
    X = np.log(df[['RV_d', 'RV_w', 'RV_m']] + 1e-6)
    y = np.log(df['Target_RV'] + 1e-6)
    X = sm.add_constant(X)
    
    forecasts = []
    actuals = []
    
    # Expanding window loop
    for t in range(min_train, len(df)):
        # Train on data strictly prior to time t
        X_train = X.iloc[:t]
        y_train = y.iloc[:t]
        
        # Test strictly on time t
        X_test = X.iloc[t:t+1]
        y_test = y.iloc[t]
        
        # Fit model and predict
        model = sm.OLS(y_train, X_train).fit()
        pred = model.predict(X_test).values[0]
        
        forecasts.append(pred)
        actuals.append(y_test)
        
    return np.array(actuals), np.array(forecasts), model

def run_baseline_evaluation(processed_file_path):
    logger.info("1. Loading Feature Matrix...")
    df = pd.read_parquet(processed_file_path)
    
    logger.info("2. Executing Expanding Window OLS...")
    actuals, forecasts, final_model = expanding_window_forecast(df)
    
    logger.info("3. Calculating Evaluation Metrics...")
    mse = mean_squared_error(actuals, forecasts)
    mae = mean_absolute_error(actuals, forecasts)
    qlike = qlike_loss_numpy(forecasts, actuals)
    
    logger.info("--- HAR Model Final OOS Metrics ---")
    logger.info(f"Baseline Mean Squared Error (MSE): {mse:.4f}")
    logger.info(f"Baseline Mean Absolute Error (MAE): {mae:.4f}")
    logger.info(f"Baseline QLIKE Loss: {qlike:.4f}")
    
    return final_model, actuals, forecasts

# Example execution:
# final_model, actuals, forecasts = run_baseline_evaluation('har_features_v1.parquet')
