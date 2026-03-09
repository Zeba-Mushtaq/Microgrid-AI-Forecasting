"""
Generate Net Load and EMS plots from trained models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

# Load the test data
print("Loading data...")
load_df = pd.read_csv('data/load/L1-train.csv')
solar_df = pd.read_csv('data/solar/train1.csv')

# Quick timestamp parsing
def parse_timestamp_load(ts):
    parts = str(ts).strip().split()
    date_part = parts[0]
    year = int(date_part[-4:])
    day_of_year = int(date_part[:-4])
    base_date = pd.Timestamp(year=year, month=1, day=1)
    date = base_date + pd.Timedelta(days=day_of_year - 1)
    hour_min = parts[1].split(':')
    return date.replace(hour=int(hour_min[0]), minute=int(hour_min[1]))

load_df['TIMESTAMP'] = load_df['TIMESTAMP'].apply(parse_timestamp_load)
solar_df['TIMESTAMP'] = pd.to_datetime(solar_df['TIMESTAMP'], format='%Y%m%d %H:%M')

# Aggregate wind data
import glob
wind_files = glob.glob('data/wind/Task1_W_Zone*.csv')
dfs = []
for file in wind_files:
    df = pd.read_csv(file)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='%Y%m%d %H:%M')
    dfs.append(df)
wind_df = pd.concat(dfs, ignore_index=True)
wind_df_agg = wind_df.groupby('TIMESTAMP').agg({'TARGETVAR': 'mean'}).reset_index()

# Prepare a small sample for prediction (use first 1000 points for each)
from sklearn.preprocessing import MinMaxScaler

def prepare_and_predict(df, target_col, model_path, n_samples=1000):
    df = df.copy()
    df['hour'] = df['TIMESTAMP'].dt.hour
    df['day_of_week'] = df['TIMESTAMP'].dt.dayofweek
    df['day_of_year'] = df['TIMESTAMP'].dt.dayofyear
    df['month'] = df['TIMESTAMP'].dt.month
    df['season'] = df['month'].apply(lambda x: (x%12 + 3)//3 - 1)
    
    feature_cols = ['hour', 'day_of_week', 'day_of_year', 'season']
    X = df[feature_cols].values[:n_samples]
    y = df[target_col].values[:n_samples].reshape(-1, 1)
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # Load model and predict
    model = keras.models.load_model(model_path, compile=False)
    y_pred_scaled = model.predict(X_scaled, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    return y_pred.flatten()

print("Generating predictions...")
n_samples = 500

# Get predictions
load_pred = prepare_and_predict(load_df[load_df['LOAD'].notna()], 'LOAD', 'models/ann_load.h5', n_samples)
solar_pred = prepare_and_predict(solar_df, 'POWER', 'models/ann_solar.h5', n_samples)
wind_pred = prepare_and_predict(wind_df_agg, 'TARGETVAR', 'models/ann_wind.h5', n_samples)

# Calculate net load
print("Calculating net load...")
net_load = load_pred - (solar_pred + wind_pred)

# Plot net load profile
print("Plotting net load profile...")
plt.figure(figsize=(15, 6))
time_steps = range(n_samples)

plt.plot(time_steps, load_pred, label='Load Demand', linewidth=2)
plt.plot(time_steps, solar_pred, label='Solar Generation', linewidth=2)
plt.plot(time_steps, wind_pred, label='Wind Generation', linewidth=2)
plt.plot(time_steps, net_load, label='Net Load', linewidth=2, linestyle='--', color='black')

plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.xlabel('Time Steps')
plt.ylabel('Power (normalized units)')
plt.title('Net Load Profile: Load - (Solar + Wind)', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/net_load_profile.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved net load profile plot")

# Plot EMS decisions
print("Plotting EMS decisions...")
grid_import = np.where(net_load > 0, net_load, 0)
grid_export = np.where(net_load < 0, -net_load, 0)

fig, axes = plt.subplots(2, 1, figsize=(15, 8))

# Import
axes[0].fill_between(time_steps, 0, grid_import, 
                      alpha=0.6, color='red', label='Grid Import (Net Load > 0)')
axes[0].set_ylabel('Import Power')
axes[0].set_title('Energy Management System: Grid Import', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Export
axes[1].fill_between(time_steps, 0, grid_export, 
                      alpha=0.6, color='green', label='Grid Export (Net Load < 0)')
axes[1].set_ylabel('Export Power')
axes[1].set_xlabel('Time Steps')
axes[1].set_title('Energy Management System: Grid Export', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/ems_decisions.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved EMS decisions plot")

# Print EMS statistics
total_import = np.sum(grid_import)
total_export = np.sum(grid_export)

print(f"\nEMS Statistics:")
print(f"Total Grid Import: {total_import:.4f} units")
print(f"Total Grid Export: {total_export:.4f} units")
print(f"Net Grid Exchange: {total_import - total_export:.4f} units")
print(f"Import hours: {np.sum(net_load > 0)} / {len(net_load)}")
print(f"Export hours: {np.sum(net_load < 0)} / {len(net_load)}")

print("\n[DONE] All plots generated successfully!")
