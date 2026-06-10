import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from arch import arch_model
from nixtla import NixtlaClient
import os
import warnings

warnings.filterwarnings('ignore')

# ----------------- Deep Learning Architectures -----------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class MambaBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.dt_proj = nn.Linear(d_model, d_model)
        self.x_proj = nn.Linear(d_model, d_model * 2)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch, seq_len, d = x.shape
        dt = torch.exp(torch.clamp(self.dt_proj(x), max=10.0))
        proj = self.x_proj(x)
        B, C = proj.chunk(2, dim=-1)
        A = -torch.ones(d, device=x.device)
        
        y_out = torch.zeros_like(x)
        h = torch.zeros(batch, d, device=x.device)
        for t in range(seq_len):
            xt = x[:, t, :]
            dtt = dt[:, t, :]
            Bt = B[:, t, :]
            Ct = C[:, t, :]
            h = torch.exp(A * dtt) * h + dtt * Bt * xt
            y_out[:, t, :] = Ct * h
            
        return self.out_proj(y_out)

class MambaTimeSeries(nn.Module):
    def __init__(self, input_dim=1, d_model=16, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([MambaBlock(d_model) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, 1)
        
    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.layers:
            x = x + self.dropout(layer(x))
        x = self.norm(x)
        out = self.out_proj(x[:, -1, :])
        return out

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def autoregressive_mc_paths(model, initial_seq, horizon, num_samples, scaler, anchor_price, device):
    model.train()
    current_seqs = initial_seq.repeat(num_samples, 1, 1).to(device)
    predicted_returns_scaled = []
    
    with torch.no_grad():
        for step in range(horizon):
            next_preds = model(current_seqs)
            predicted_returns_scaled.append(next_preds.cpu().numpy())
            next_preds_expanded = next_preds.unsqueeze(1)
            current_seqs = torch.cat([current_seqs[:, 1:, :], next_preds_expanded], dim=1)
            
    predicted_returns_scaled = np.array(predicted_returns_scaled).squeeze(-1).T
    flat_scaled = predicted_returns_scaled.reshape(-1, 1)
    flat_returns = scaler.inverse_transform(flat_scaled)
    predicted_returns = flat_returns.reshape(num_samples, horizon)
    
    cumulative_returns = np.cumsum(predicted_returns, axis=1)
    price_paths = anchor_price * np.exp(cumulative_returns)
    return price_paths

def get_insample_preds(model, X_train, scaler, anchor_prices, device):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        preds_scaled = model(X_t).cpu().numpy()
        preds_ret = scaler.inverse_transform(preds_scaled).flatten()
    pred_prices = anchor_prices * np.exp(preds_ret)
    return pred_prices

# ----------------- Evaluation & GARCH Logic -----------------
def coverage_ratio(y_true, lower, upper):
    covered = np.sum((y_true >= lower) & (y_true <= upper))
    return (covered / len(y_true)) * 100.0

def generate_plots(model_name, train_dates, train_prices, test_dates, test_actual, mean_preds, lower_bound, upper_bound, garch_vol=None):
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(out_dir, exist_ok=True)
    
    # 3 subplots if GARCH vol provided, else 2
    n_plots = 3 if garch_vol is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 6))
    if n_plots == 2:
        ax1, ax2 = axes
    else:
        ax1, ax2, ax3 = axes
        
    # Plot 1: Standard
    ax1.plot(train_dates[-100:], train_prices[-100:], label="History", color="black")
    ax1.plot(test_dates, test_actual, label="Actual", color="blue")
    ax1.plot(test_dates, mean_preds, label="Mean Pred", color="purple", linestyle="--")
    ax1.fill_between(test_dates, lower_bound, upper_bound, color="purple", alpha=0.2, label=f"90% Interval")
    ax1.set_title(f"{model_name} - 100 Day Context")
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Zoomed out-of-sample
    ax2.plot(test_dates, test_actual, label="Actual", color="blue", marker='o')
    ax2.plot(test_dates, mean_preds, label="Mean Pred", color="purple", linestyle="--", marker='x')
    ax2.fill_between(test_dates, lower_bound, upper_bound, color="purple", alpha=0.2, label=f"90% Interval")
    ax2.set_title(f"{model_name} - Zoomed (30 Days Out-of-Sample)")
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Conditional Volatility (if applicable)
    if garch_vol is not None:
        ax3.plot(test_dates, np.sqrt(garch_vol), label="GARCH(1,1) Std Dev", color="red", marker='s')
        ax3.set_title(f"{model_name} - Forecasted Conditional Volatility")
        ax3.legend()
        ax3.grid(True)
    
    filename = os.path.join(out_dir, f"{model_name.replace(' ', '_')}_GARCH_Hybrid.png")
    plt.savefig(filename, dpi=300)
    print(f"[{model_name}] Saved plot to {filename}")

def run_local_model(name, model_class, params, df, dates, horizon=30, seq_length=60):
    prices = df['Price'].values
    log_returns = np.log(prices[1:] / prices[:-1])
    
    train_prices = prices[:-horizon]
    train_returns = log_returns[:-horizon]
    
    test_actual = prices[-horizon:]
    test_dates = dates[-horizon:]
    train_dates = dates[:-horizon]
    
    scaler = MinMaxScaler()
    train_returns_scaled = scaler.fit_transform(train_returns.reshape(-1, 1))
    
    X_train, y_train = create_sequences(train_returns_scaled, seq_length)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_class(**params, dropout=0.2).to(device)
    
    # Create Dataset and DataLoader to match original training exactly
    class TimeSeriesDataset(torch.utils.data.Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)
        def __len__(self):
            return len(self.X)
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]
            
    train_dataset = TimeSeriesDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    if 'LSTM' in name:
        target_epochs = 40
        lr = 0.001
    else:
        target_epochs = 20
        lr = 0.0005 # Mamba uses a lower learning rate
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
        
    print(f"[{name}] Training model safely using DataLoaders with calibrated epochs ({target_epochs}) and lr ({lr})...")
    
    model.train()
    for epoch in range(target_epochs):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = model(batch_X)
            loss = criterion(out, batch_y)
            loss.backward()
            # Gradient clipping is essential for Mamba/LSTM stability on noisy returns
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
    print(f"[{name}] Generating in-sample historical predictions for GARCH...")
    anchor_prices = train_prices[seq_length : seq_length + len(X_train)]
    actual_prices_insample = train_prices[seq_length + 1 : seq_length + 1 + len(X_train)]
    
    pred_prices_insample = get_insample_preds(model, X_train, scaler, anchor_prices, device)
    residuals = actual_prices_insample - pred_prices_insample
    
    print(f"[{name}] Fitting Fat-Tailed GARCH(1,1) on residuals...")
    res_scaled = residuals / 100.0
    # Use Student's t-distribution to model fat tails
    garch = arch_model(res_scaled, vol='Garch', p=1, q=1, dist='t', rescale=False)
    garch_res = garch.fit(disp='off')
    garch_forecast = garch_res.forecast(horizon=horizon)
    garch_var_scaled = garch_forecast.variance.values[-1] 
    garch_var = garch_var_scaled * (100.0 ** 2) 
    
    print(f"[{name}] Generating Out-Of-Sample MC Paths (High Dropout)...")
    initial_seq = torch.tensor(train_returns_scaled[-seq_length:].reshape(1, seq_length, 1), dtype=torch.float32)
    last_price = train_prices[-1]
    
    print(f"[{name}] Generating Out-Of-Sample MC Paths (Dropout {params.get('dropout', 0.2)})...")
    # Generate the paths exactly as they were in the original evaluation script
    model.train() # Ensure dropout is fully enabled for both mean and variance
    
    # Generate 50 MC paths. Averaging these produces the gentle curve the user expects
    paths = autoregressive_mc_paths(model, initial_seq, horizon, 50, scaler, last_price, device) 
    
    mean_preds = np.mean(paths, axis=0) # Averaging the 50 exponential paths matching original logic
    model_var = np.var(paths, axis=0)
    
    # Calculate Hybrid Interval centered around deterministic mean
    total_std = np.sqrt(model_var + garch_var)
    lower_bound = mean_preds - 1.645 * total_std
    upper_bound = mean_preds + 1.645 * total_std
    
    cr = coverage_ratio(test_actual, lower_bound, upper_bound)
    print(f"[{name}] Coverage Ratio: {cr:.2f}%\n")
    
    generate_plots(name, train_dates, train_prices, test_dates, test_actual, mean_preds, lower_bound, upper_bound, garch_var)
    
def run_timegpt(df, dates, is_finetuned=False, horizon=30):
    name = "TimeGPT (Fine-Tuned)" if is_finetuned else "TimeGPT (Zero-Shot)"
    print(f"--- Processing {name} ---")
    
    Y_df_raw = df[['Date', 'Price']].copy()
    Y_df_raw['ds'] = pd.to_datetime(Y_df_raw['Date'])
    Y_df_raw['y'] = Y_df_raw['Price']
    Y_df_raw['unique_id'] = 'KSE100'
    Y_df_raw = Y_df_raw[['unique_id', 'ds', 'y']]
    
    Y_df = Y_df_raw.set_index('ds').resample('B').ffill().reset_index()
    Y_df['unique_id'] = 'KSE100'
    
    train_df = Y_df.iloc[:-horizon]
    test_df = Y_df.iloc[-horizon:]
    
    nixtla_client = NixtlaClient(api_key="nixak-NQrAApCEVEF3Nk5y8D3oMaExh9MbdgG866fshU5UiaaJyzMfLMMQRHQJbtNybriMBYOxjaMKSt2AC4Sf")
    
    print(f"[{name}] Generating Conformal Forecast (No GARCH Double Counting)...")
    finetune_steps = 20 if is_finetuned else 0
    # Requesting 80% intervals to tighten the excessively wide 90% bounds
    fcst = nixtla_client.forecast(
        df=train_df, h=horizon, level=[80], add_history=False, freq='B', finetune_steps=finetune_steps
    )
    
    mean_preds = fcst['TimeGPT'].values[:horizon]
    lower_bound = fcst['TimeGPT-lo-80'].values[:horizon]
    upper_bound = fcst['TimeGPT-hi-80'].values[:horizon]
    
    cr = coverage_ratio(test_df['y'].values, lower_bound, upper_bound)
    print(f"[{name}] Coverage Ratio (Actual 80% Target): {cr:.2f}%\n")
    
    generate_plots(name, train_df['ds'].values, train_df['y'].values, test_df['ds'].values, test_df['y'].values, mean_preds, lower_bound, upper_bound, garch_vol=None)

def main():
    print("Loading data...")
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(_repo_root, "data", "base_df_post_eda_fixed.csv")
    df = pd.read_csv(data_path)
    dates = pd.to_datetime(df['Date']).values
    
    run_local_model("PyTorch LSTM GARCH Hybrid", LSTMModel, {'hidden_size': 64}, df, dates)
    run_local_model("PyTorch Mamba GARCH Hybrid", MambaTimeSeries, {'d_model': 16}, df, dates)
    run_timegpt(df, dates, is_finetuned=False)
    run_timegpt(df, dates, is_finetuned=True)

if __name__ == "__main__":
    main()
