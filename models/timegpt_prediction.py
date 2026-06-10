import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from nixtla import NixtlaClient

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

def main():
    api_key = "nixak-NQrAApCEVEF3Nk5y8D3oMaExh9MbdgG866fshU5UiaaJyzMfLMMQRHQJbtNybriMBYOxjaMKSt2AC4Sf"
    nixtla_client = NixtlaClient(api_key=api_key)
    
    # Test connection
    try:
        nixtla_client.validate_api_key()
        print("Successfully connected to Nixtla TimeGPT API.")
    except Exception as e:
        print(f"Failed to connect to API: {e}")
        return

    print("Loading data...")
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(_repo_root, "data", "base_df_post_eda_fixed.csv")
    df = pd.read_csv(data_path)
    df['ds'] = pd.to_datetime(df['Date'])
    df['y'] = df['Price']
    df['unique_id'] = 'KSE100'
    
    Y_df_raw = df[['unique_id', 'ds', 'y']].sort_values(by=['unique_id', 'ds']).reset_index(drop=True)
    Y_df_raw = Y_df_raw.set_index('ds')
    
    # Fill missing business days
    Y_df = Y_df_raw.resample('B').ffill().reset_index()
    Y_df['unique_id'] = 'KSE100'
    
    print("\n--- Running Time-Series Cross Validation with Fine-Tuning (3 Splits) ---")
    horizon = 30
    tscv = TimeSeriesSplit(n_splits=3, test_size=horizon)
    
    fold_metrics = []
    last_fold_results = None
    
    fold_idx = 1
    for train_index, test_index in tscv.split(Y_df):
        print(f"--- Fold {fold_idx} ---")
        
        train_df = Y_df.iloc[train_index]
        test_df = Y_df.iloc[test_index]
        
        try:
            # Using Fine-Tuning for specialized performance
            fcst = nixtla_client.forecast(
                df=train_df,
                h=horizon,
                level=[90],
                time_col='ds',
                target_col='y',
                freq='B',
                finetune_steps=20
            )
            
            y_true = test_df['y'].values
            y_pred = fcst['TimeGPT'].values[:horizon]
            
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            smape_val = smape(y_true, y_pred)
            
            fold_metrics.append({'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'SMAPE': smape_val})
            print(f"Fold {fold_idx} Metrics | MAPE: {mape:.2f}% | SMAPE: {smape_val:.2f}%")
            
            if fold_idx == 3:
                last_fold_results = {
                    'train_df': train_df,
                    'test_df': test_df,
                    'fcst': fcst
                }
                
        except Exception as e:
            print(f"Error in Fold {fold_idx}: {e}")
            
        fold_idx += 1

    if not fold_metrics:
        print("No successful folds.")
        return

    # Average Metrics
    avg_metrics = pd.DataFrame(fold_metrics).mean()
    print("\n--- Average Final CV Metrics (TimeGPT Zero-Shot) ---")
    print(f"TimeGPT CV Avg | MAE: {avg_metrics['MAE']:.2f} | RMSE: {avg_metrics['RMSE']:.2f} | MAPE: {avg_metrics['MAPE']:.2f}% | SMAPE: {avg_metrics['SMAPE']:.2f}%")

    print("\nGenerating model comparison plot for final fold...")
    if last_fold_results:
        plt.figure(figsize=(14, 8))
        
        train_df = last_fold_results['train_df']
        test_df = last_fold_results['test_df']
        fcst = last_fold_results['fcst']
        
        # Plot last 100 days of train
        train_dates = train_df['ds'].iloc[-100:]
        train_y = train_df['y'].iloc[-100:]
        
        plt.plot(train_dates, train_y, label='Actual (Train History)', color='black', linewidth=2)
        plt.plot(test_df['ds'], test_df['y'], label='Actual (Test)', color='blue', linewidth=2)
        
        plt.plot(test_df['ds'], fcst['TimeGPT'].values[:horizon], label='TimeGPT Forecast', linestyle='--', color='purple')
        plt.fill_between(test_df['ds'], fcst['TimeGPT-lo-90'].values[:horizon], fcst['TimeGPT-hi-90'].values[:horizon], 
                         color='purple', alpha=0.15, label='90% Confidence Interval')
                         
        plt.title('KSE100 Prediction: TimeGPT (Final CV Fold)')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_img = os.path.join(_repo_root, "results", "timegpt_prediction.png")
        plt.savefig(output_img, dpi=300)
        print(f"Saved plot to {output_img}")

if __name__ == "__main__":
    main()
