import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import logging

# Setup Logging (Issue 12)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# 1. THE ARCHITECTURE
# ==========================================
class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, dropout=0.3): # Issue 8
        super(GatedResidualNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.glu = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GLU()
        )
        
        self.res_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.dropout(self.fc2(self.elu(self.fc1(x))))
        h = self.glu(h)
        res = self.res_proj(x)
        return self.layer_norm(h + res)

class DeepGRN(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, dropout=0.3, num_layers=2):
        super(DeepGRN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GatedResidualNetwork(input_dim, hidden_dim, dropout))
        for _ in range(num_layers - 1):
            self.layers.append(GatedResidualNetwork(hidden_dim, hidden_dim, dropout))
            
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class NeuralHAR(nn.Module):
    """
    Issue 5: HAR prior + GRN correction separated correctly.
    """
    def __init__(self, num_har_features=3, num_exo_features=2, hidden_dim=16, dropout=0.3, num_layers=2):
        super().__init__()
        
        # HAR prior — linear weights
        self.har_linear = nn.Linear(num_har_features, 1, bias=True)
        
        # GRN learns ONLY the residual from exogenous features
        self.grn = DeepGRN(num_exo_features, hidden_dim, dropout, num_layers)
        self.residual_head = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, x_har, x_exo):
        har_prior = self.har_linear(x_har)
        grn_out = self.grn(x_exo)
        correction = self.residual_head(grn_out)
        return har_prior + correction 
    
    def init_har_weights(self, beta_weights, intercept):
        """Initialize HAR linear layer with pre-trained OLS estimates."""
        with torch.no_grad():
            self.har_linear.weight.data = torch.tensor(beta_weights, dtype=torch.float32)
            self.har_linear.bias.data = torch.tensor([intercept], dtype=torch.float32)

    def __repr__(self): # Issue 10
        total_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (f"NeuralHAR(har_features={self.har_linear.in_features}, "
                f"exo_features={self.grn.layers[0].fc1.in_features}, "
                f"total_params={total_params}, trainable={trainable})")

# ==========================================
# 2. METRICS & TRAINING TOOLS
# ==========================================
def qlike_loss(pred, actual):
    """Issue 4: QLIKE Loss function for volatility."""
    actual_var = torch.exp(actual)
    pred_var = torch.exp(pred.clamp(min=-10, max=10))
    return torch.mean(actual_var/pred_var - torch.log(actual_var/pred_var) - 1)

class EarlyStopping:
    """Issue 7: Early Stopping to prevent overfitting."""
    def __init__(self, patience=10, min_delta=1e-5, save_path='best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0
        self.stop = False
        self.save_path = save_path
    
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

# ==========================================
# 3. THE DATA PIPELINE & TRAINING LOOP
# ==========================================
def train_model(model, train_loader, val_loader, epochs=50, lr=0.001):
    # Issue 11: Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    early_stopper = EarlyStopping(patience=10)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_x_har, batch_x_exo, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_x_har, batch_x_exo)
            
            # Issue 4: QLIKE loss
            loss = qlike_loss(predictions, batch_y)
            loss.backward()
            
            # Issue 6: Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x_har, batch_x_exo, batch_y in val_loader:
                preds = model(batch_x_har, batch_x_exo)
                loss = qlike_loss(preds, batch_y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch+1:<4} | Train QLIKE: {train_loss:<10.4f} | Val QLIKE: {val_loss:<10.4f}")
            
        early_stopper(val_loss, model)
        if early_stopper.stop:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
            
    # Load best weights
    model.load_state_dict(torch.load(early_stopper.save_path, weights_only=True))
    return model

# ==========================================
# 4. EXECUTION BLOCK (Mock Testing)
# ==========================================
if __name__ == "__main__":
    logger.info("Initializing Neural-HAR Engine...")
    
    # Issue 9: Realistic Mock Data Generation
    np.random.seed(42)
    n = 1000
    log_rv = np.zeros(n)
    for t in range(1, n):
        log_rv[t] = 0.1 + 0.7 * log_rv[t-1] + np.random.normal(0, 0.3)
        
    mock_df = pd.DataFrame({
        'RV_d': log_rv,
        'RV_w': pd.Series(log_rv).rolling(5).mean().fillna(log_rv[0]).values,
        'RV_m': pd.Series(log_rv).rolling(22).mean().fillna(log_rv[0]).values,
        'Sentiment': np.random.normal(0, 1, n), 
        'Macro_VIX': np.random.beta(2, 5, n), 
        'Target_RV': np.roll(log_rv, -1) # Issue 1: Target is already log(RV)
    }).iloc[:-1] # Drop the last rolled row
    
    # Split Data (Static split for testing architecture only - Issue 2/3 note applied)
    split_idx = int(len(mock_df) * 0.8)
    train_df = mock_df.iloc[:split_idx]
    test_df = mock_df.iloc[split_idx:]
    
    har_cols = ['RV_d', 'RV_w', 'RV_m']
    exo_cols = ['Sentiment', 'Macro_VIX']
    
    # Issue 3 Warning: In the final expanding window script, scaler must be refit per window.
    scaler_exo = StandardScaler()
    train_exo_scaled = scaler_exo.fit_transform(train_df[exo_cols].values)
    test_exo_scaled = scaler_exo.transform(test_df[exo_cols].values)
    
    train_dataset = TensorDataset(
        torch.FloatTensor(train_df[har_cols].values),
        torch.FloatTensor(train_exo_scaled),
        torch.FloatTensor(train_df['Target_RV'].values.reshape(-1, 1))
    )
    
    test_dataset = TensorDataset(
        torch.FloatTensor(test_df[har_cols].values),
        torch.FloatTensor(test_exo_scaled),
        torch.FloatTensor(test_df['Target_RV'].values.reshape(-1, 1))
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Instantiate Model
    model = NeuralHAR(num_har_features=3, num_exo_features=2, hidden_dim=16, dropout=0.3, num_layers=2)
    logger.info(f"Model Structure: {model}")
    
    logger.info("Beginning Deep Learning Training Loop...")
    trained_model = train_model(model, train_loader, test_loader, epochs=50, lr=0.005)
    
    logger.info("Training Complete.")
