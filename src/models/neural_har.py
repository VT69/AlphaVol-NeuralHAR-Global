"""
Hybrid TFT/GRN Model for Volatility Forecasting
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ==========================================
# 1. THE ARCHITECTURE: Gated Residual Network
# ==========================================
class GatedResidualNetwork(nn.Module):
    """
    Core component of the Temporal Fusion Transformer.
    Applies non-linear processing only where necessary, bypassing it via 
    residual connections when linear relationships (like classical HAR) suffice.
    """
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(GatedResidualNetwork, self).__init__()
        
        # Primary dense layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        
        # Secondary dense layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Gated Linear Unit (GLU) for dynamic feature selection
        self.glu = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GLU()
        )
        
        # Residual connection projection (if input and hidden dim differ)
        self.res_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Non-linear path
        h = self.fc1(x)
        h = self.elu(h)
        h = self.fc2(h)
        h = self.dropout(h)
        h = self.glu(h)
        
        # Residual connection
        res = self.res_proj(x)
        
        # Add and Norm
        return self.layer_norm(h + res)


class NeuralHAR(nn.Module):
    """
    Hybrid Volatility Predictor: Fuses traditional HAR lags with Exogenous Alpha features.
    """
    def __init__(self, num_har_features=3, num_exo_features=2, hidden_dim=32):
        super(NeuralHAR, self).__init__()
        
        total_features = num_har_features + num_exo_features
        
        # The Brain
        self.grn = GatedResidualNetwork(input_dim=total_features, hidden_dim=hidden_dim)
        
        # The Final Output (Predicting a single continuous value: RV)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.grn(x)
        out = self.regressor(h)
        return out

# ==========================================
# 2. THE DATA PIPELINE & TRAINING LOOP
# ==========================================
def prepare_dataloaders(df, target_col='Target_RV', test_size=0.2, batch_size=64):
    """
    Converts the Pandas dataframe into PyTorch tensors with chronological splitting.
    """
    # 1. Chronological Split (No time travel!)
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    # 2. Separate Features (X) and Target (y)
    X_train = train_df.drop(columns=[target_col]).values
    y_train = train_df[target_col].values.reshape(-1, 1)
    
    X_test = test_df.drop(columns=[target_col]).values
    y_test = test_df[target_col].values.reshape(-1, 1)
    
    # 3. Scale Features (Neural nets require normalized inputs)
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # 4. Convert to PyTorch Tensors
    train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled), torch.FloatTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False) # Sequential batches
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, y_test

def train_model(model, train_loader, test_loader, epochs=50, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    print(f"{'Epoch':<10} | {'Train Loss (MSE)':<20} | {'Val Loss (MSE)':<20}")
    print("-" * 55)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation Phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                preds = model(batch_X)
                loss = criterion(preds, batch_y)
                val_loss += loss.item()
        val_loss /= len(test_loader)
        
        if (epoch + 1) % 10 == 0:
            print(f"{epoch+1:<10} | {train_loss:<20.4f} | {val_loss:<20.4f}")
            
    return model

# ==========================================
# 3. EXECUTION BLOCK (For Colab Integration)
# ==========================================
if __name__ == "__main__":
    print("🧠 Initializing Neural-HAR Engine...")
    
    # Simulate loading your Data Lake feature matrix
    # In practice, load: pd.read_parquet('/content/drive/MyDrive/AlphaVol_Data/processed/har_features_v1.parquet')
    
    # ⚠️ MOCK DATA FOR TESTING SCRIPT EXECUTION
    print("⚠️ Generating mock data to test architecture...")
    np.random.seed(42)
    mock_data = pd.DataFrame({
        'RV_d': np.random.rand(1000) * 10,
        'RV_w': np.random.rand(1000) * 12,
        'RV_m': np.random.rand(1000) * 15,
        'Sentiment': np.random.randn(1000), # Exogenous 1
        'Macro_VIX': np.random.rand(1000) * 30, # Exogenous 2
        'Target_RV': np.random.rand(1000) * 12 # Target
    })
    
    # 1. Prepare Data
    train_loader, test_loader, y_actual = prepare_dataloaders(mock_data)
    
    # 2. Instantiate Model
    # 3 HAR features (d, w, m) + 2 Exogenous (Sentiment, VIX)
    model = NeuralHAR(num_har_features=3, num_exo_features=2, hidden_dim=64)
    
    # 3. Train
    print("\n🚀 Beginning Deep Learning Training Loop...")
    trained_model = train_model(model, train_loader, test_loader, epochs=50, lr=0.005)
    
    print("\n✅ Training Complete. Model is ready for TensorRT quantization (C++ Inference).")
    
    # Save the weights to your Google Drive
    # torch.save(trained_model.state_dict(), '/content/drive/MyDrive/AlphaVol_Data/models/neural_har_weights.pth')
