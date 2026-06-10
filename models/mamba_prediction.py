import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import math

import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------
# Minimal Pure PyTorch Mamba / State Space Model (SSM) Implementation
# -----------------------------------------------------------------------

class MambaBlock(nn.Module):
    """
    A simplified, pure-PyTorch implementation of a Mamba (Selective SSM) block.
    This avoids the complex CUDA compilation of the official mamba-ssm package 
    while retaining the core State Space Model recurrence.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2 + 1, bias=False)
        
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        
        # S4D initialization for A matrix
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).unsqueeze(0) # (1, d_state)
        self.A_log = nn.Parameter(torch.log(A)) # Learned parameter
        
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape
        
        # 1. Project input (x to x and z)
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1) # (B, L, d_inner)
        
        # 2. 1D Convolution
        x_conv = x_proj.transpose(1, 2) # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[..., :seq_len] # truncate padding
        x_conv = x_conv.transpose(1, 2) # (B, L, d_inner)
        
        x_act = F.silu(x_conv)
        
        # 3. State Space Model (SSM) Parameters
        x_dbl = self.x_proj(x_act) # (B, L, d_state * 2 + 1)
        delta, B, C = torch.split(x_dbl, [1, self.d_state, self.d_state], dim=-1)
        
        # Softplus to ensure positive delta
        delta = F.softplus(delta) # (B, L, 1)
        # Prevent numerical explosion (clamping delta)
        delta = torch.clamp(delta, min=1e-5, max=5.0)
        
        delta = self.dt_proj(delta) # (B, L, d_inner)
        
        # Discretize A and B (Zero-order hold approximation)
        self.A_log.data = torch.clamp(self.A_log.data, max=0.0) # Prevent explosion
        A = -torch.exp(self.A_log) # (1, d_state) - ensure negative eigenvalues
        A = A.view(1, 1, 1, self.d_state) # (1, 1, 1, d_state)
        
        # Protect exponential from blowing up
        delta_A = torch.exp(torch.clamp(delta.unsqueeze(-1) * A, min=-20.0, max=0.0)) # (B, L, d_inner, d_state)
        delta_B = delta.unsqueeze(-1) * B.unsqueeze(2) # (B, L, d_inner, d_state)
        
        # 4. Sequential Scan (Unrolled purely in PyTorch for simplicity, 
        # normally optimized via parallel associative scan in CUDA)
        h = torch.zeros(batch, self.d_inner, self.d_state, device=x.device)
        y = []
        for t in range(seq_len):
            xt = x_act[:, t].unsqueeze(-1) # (B, d_inner, 1)
            delta_A_t = delta_A[:, t] # (B, d_inner, d_state)
            delta_B_t = delta_B[:, t] # (B, d_inner, d_state)
            
            h = delta_A_t * h + delta_B_t * xt
            h = torch.clamp(h, min=-1e3, max=1e3) # Prevent unbounded growth
            
            C_t = C[:, t].unsqueeze(1) # (B, 1, d_state)
            y_t = torch.sum(h * C_t, dim=-1) # (B, d_inner)
            y.append(y_t)
            
        y = torch.stack(y, dim=1) # (B, L, d_inner)
        
        # Skip connection based on D
        y = y + x_act * self.D
        
        # 5. Gated output
        y = y * F.silu(z)
        out = self.out_proj(y)
        
        return out

class MambaTimeSeries(nn.Module):
    def __init__(self, input_size=1, d_model=32, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Linear(input_size, d_model)
        
        self.layers = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=16, expand=2) 
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        # Sequence pooling (take last timestep for forecasting)
        out = self.dropout(x[:, -1, :]) 
        out = self.fc(out)
        return out

# -----------------------------------------------------------------------

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def smape_metric(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def autoregressive_mc_predict(model, initial_seq, horizon, num_samples, scaler, anchor_price, device):
    """
    Performs true multi-step out-of-sample forecasting by feeding back predictions.
    Generates `num_samples` Monte Carlo dropout paths and compounds their returns 
    into absolute price paths to extract correct fanning percentiles.
    """
    model.train() # Enable dropout for MC sampling
    
    # initial_seq shape: (1, seq_len, 1) -> expand to (num_samples, seq_len, 1)
    current_seqs = initial_seq.repeat(num_samples, 1, 1).to(device)
    
    predicted_returns_scaled = []
    
    with torch.no_grad():
        for step in range(horizon):
            # Predict next step (returns scaled)
            # next_preds shape: (num_samples, 1)
            next_preds = model(current_seqs)
            predicted_returns_scaled.append(next_preds.cpu().numpy())
            
            # Autoregressive slide: drop oldest, append predicted
            next_preds_expanded = next_preds.unsqueeze(1) # (num_samples, 1, 1)
            current_seqs = torch.cat([current_seqs[:, 1:, :], next_preds_expanded], dim=1)
            
    # Combine predictions: (horizon, num_samples, 1) -> (num_samples, horizon)
    predicted_returns_scaled = np.array(predicted_returns_scaled).squeeze(-1).T
    
    # Inverse transform to get actual Log Returns
    flat_scaled = predicted_returns_scaled.reshape(-1, 1)
    flat_returns = scaler.inverse_transform(flat_scaled)
    predicted_returns = flat_returns.reshape(num_samples, horizon)
    
    # Cumulative sum over the horizon
    cumulative_returns = np.cumsum(predicted_returns, axis=1) # (num_samples, horizon)
    
    # Reconstruct absolute price paths: P_t+k = P_t * exp(Cumulative Return)
    price_paths = anchor_price * np.exp(cumulative_returns)
    
    # Extract prediction percentiles
    mean_price = np.mean(price_paths, axis=0)
    lower_price = np.percentile(price_paths, 5, axis=0) # 90% confidence lower bound
    upper_price = np.percentile(price_paths, 95, axis=0) # 90% confidence upper bound
    
    return mean_price, lower_price, upper_price

from sklearn.model_selection import TimeSeriesSplit

def main():
    print("Loading data...")
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(_repo_root, "data", "base_df_post_eda_fixed.csv")
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # 1. Transform Prices to Log Returns to remove upward drift
    # Log_Return = ln(Price_t / Price_t-1)
    df['Log_Return'] = np.log(df['Price'] / df['Price'].shift(1))
    df = df.dropna().reset_index(drop=True) # Drop the first NaN row
    
    # Optional: Keep raw prices around for reconstruction and plotting
    raw_prices = df['Price'].values
    target_returns = df['Log_Return'].values.reshape(-1, 1)
    dates = df['Date']
    
    seq_length = 60
    horizon_size = 30 
    
    print("\n--- Running Time-Series Cross Validation (3 Splits) ---")
    tscv = TimeSeriesSplit(n_splits=3, test_size=horizon_size)
    
    fold_metrics = []
    
    # We will save the very last fold for the final plotting visualization
    best_model = None
    last_fold_plot_data = {}
    
    scaler = MinMaxScaler()
    
    # Store complete scaled dataset (can fit just on train incrementally if strict, 
    # but for simplicity we fit on all past folds)
    fold_idx = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    for train_index, test_index in tscv.split(target_returns):
        print(f"--- Fold {fold_idx} ---")
        
        # Slicing the data
        # Note: TimeSeriesSplit provides indices. 
        # test_index usually has length = test_size (30).
        # We need seq_length rows BEFORE test_index to build the input sequence for the first test point.
        
        train_returns_raw = target_returns[train_index]
        # For the test set to generate full sequences, we prepend the last seq_length points of train
        test_returns_raw = target_returns[np.concatenate((train_index[-seq_length:], test_index))]
        
        # Fit scaler ONLY on this fold's training data to prevent leakage
        scaler_fold = MinMaxScaler()
        train_data_scaled = scaler_fold.fit_transform(train_returns_raw)
        test_data_scaled = scaler_fold.transform(test_returns_raw)
        
        X_train, y_train = create_sequences(train_data_scaled, seq_length)
        X_test, y_test = create_sequences(test_data_scaled, seq_length)
        
        train_dataset = TimeSeriesDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Grid Search space
        d_models = [16, 32]
        learning_rates = [0.0001, 0.0005]
        epoch_options = [20, 40]
        
        fold_best_model = None
        fold_best_loss = float('inf')
        fold_best_params = {}
        
        # Hyperparameter Tuning Loop
        for d_m in d_models:
            for lr in learning_rates:
                for eps in epoch_options:
                    model = MambaTimeSeries(input_size=1, d_model=d_m, num_layers=2, dropout=0.2).to(device)
                    criterion = nn.MSELoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    
                    for epoch in range(eps):
                        model.train()
                        epoch_loss = 0
                        for batch_X, batch_y in train_loader:
                            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                            optimizer.zero_grad()
                            outputs = model(batch_X)
                            loss = criterion(outputs, batch_y)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()
                            epoch_loss += loss.item()
                            
                    avg_train_loss = epoch_loss / len(train_loader)
                    if avg_train_loss < fold_best_loss:
                        fold_best_loss = avg_train_loss
                        fold_best_model = model
                        fold_best_params = {'d_model': d_m, 'lr': lr, 'epochs': eps}
                        
        print(f"Fold {fold_idx} Best Params: {fold_best_params} (Train Loss: {fold_best_loss:.4f})")
        
        # Evaluate Best Fold Model Autoregressively
        fold_best_model.eval()
        
        # The true anchor price is the EXACT price from the day immediately preceding the 30-day forecast horizon
        anchor_price = raw_prices[test_index[0] - 1]
        actual_prices = raw_prices[test_index]
        
        # The initial sequence is the very last 60 days of the Training Set Fold
        initial_seq_array = train_data_scaled[-seq_length:].reshape(1, seq_length, 1)
        initial_seq_tensor = torch.tensor(initial_seq_array, dtype=torch.float32).to(device)
        
        mean_prices, lower_prices, upper_prices = autoregressive_mc_predict(
            model=model, 
            initial_seq=initial_seq_tensor, 
            horizon=horizon_size, 
            num_samples=50, 
            scaler=scaler_fold, 
            anchor_price=anchor_price, 
            device=device
        )
        
        predicted_prices = mean_prices
        
        mae = mean_absolute_error(actual_prices, predicted_prices)
        rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
        mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
        smape_val = smape_metric(actual_prices, predicted_prices)
        
        fold_metrics.append({'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'SMAPE': smape_val})
        print(f"Fold {fold_idx} Metrics | MAPE: {mape:.2f}% | SMAPE: {smape_val:.2f}%")
        
        # Save last fold for plot
        if fold_idx == 3:            
            last_fold_plot_data = {
                'train_idx': train_index,
                'test_idx': test_index,
                'actual_prices': actual_prices,
                'predicted_prices': predicted_prices,
                'lower_bound': lower_prices,
                'upper_bound': upper_prices
            }
            
        fold_idx += 1
        
    print("\n--- Average Final CV Metrics ---")
    avg_mape = np.mean([fm['MAPE'] for fm in fold_metrics])
    avg_smape = np.mean([fm['SMAPE'] for fm in fold_metrics])
    avg_mae = np.mean([fm['MAE'] for fm in fold_metrics])
    avg_rmse = np.mean([fm['RMSE'] for fm in fold_metrics])
    print(f"Mamba CV Avg | MAE: {avg_mae:.2f} | RMSE: {avg_rmse:.2f} | MAPE: {avg_mape:.2f}% | SMAPE: {avg_smape:.2f}%")
            
    print("\nGenerating model comparison plot for final fold (Returns Transformed)...")
    plt.figure(figsize=(14, 7))
    
    # Plot last 100 days of train from the last CV fold
    last_train_idx = last_fold_plot_data['train_idx']
    last_test_idx = last_fold_plot_data['test_idx']
    
    train_dates = dates.iloc[last_train_idx][-100:]
    train_prices = raw_prices[last_train_idx][-100:]
    
    test_dates = dates.iloc[last_test_idx]
    
    plt.plot(train_dates, train_prices, label='Actual (Train History)', color='black', linewidth=2)
    plt.plot(test_dates, last_fold_plot_data['actual_prices'], label='Actual (Test)', color='blue', linewidth=2)
    
    plt.plot(test_dates, last_fold_plot_data['predicted_prices'], label='Mamba Mean Prediction (from Returns)', linestyle='--', color='red')
    plt.fill_between(test_dates, last_fold_plot_data['lower_bound'].flatten(), last_fold_plot_data['upper_bound'].flatten(), color='red', alpha=0.2, label='90% Prediction Interval')
            
    plt.title('KSE100 Prediction: Mamba (Cross-Validated Return Transformation)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_img = os.path.join(_repo_root, "results", "pytorch_mamba_prediction.png")
    plt.savefig(output_img, dpi=300)
    print(f"Saved plot to {output_img}")

if __name__ == "__main__":
    main()
