"""
Classical Corsi (2009) OLS benchmark for Realized Volatility
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def train_baseline_har(processed_file_path):
    print("🧠 1. Loading Feature Matrix...")
    df = pd.read_parquet(processed_file_path)
    
    # Define our inputs (X) and target (y)
    X = df[['RV_d', 'RV_w', 'RV_m']]
    y = df['Target_RV']
    
    # Statsmodels requires us to explicitly add a constant (beta_0)
    X = sm.add_constant(X)
    
    print("✂️ 2. Performing Chronological Train/Test Split (80/20)...")
    split_index = int(len(df) * 0.8)
    
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    print(f"Training on dates: {X_train.index[0].date()} to {X_train.index[-1].date()}")
    print(f"Testing on dates: {X_test.index[0].date()} to {X_test.index[-1].date()}")
    
    print("⚙️ 3. Fitting the OLS Regression Model...")
    model = sm.OLS(y_train, X_train)
    results = model.fit()
    
    # Print the mathematical summary
    print("\n--- HAR Model Summary ---")
    print(results.summary().tables[1]) # Only prints the coefficient table for clean output
    
    print("\n🔮 4. Evaluating on the Unseen Test Set...")
    predictions = results.predict(X_test)
    
    # Calculate baseline metrics
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    
    print(f"Baseline Mean Squared Error (MSE): {mse:.4f}")
    print(f"Baseline Mean Absolute Error (MAE): {mae:.4f}")
    
    return results, y_test, predictions

# Assuming PROCESSED_HAR_FILE is defined from previous cells
# results, y_test, predictions = train_baseline_har(PROCESSED_HAR_FILE)
