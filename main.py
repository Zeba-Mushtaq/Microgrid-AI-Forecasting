"""
AI-Based Forecasting Models for Load Demand, Renewable Generation and 
Microgrid Control Mechanisms for Adaptive Energy Dispatch

Author: Abubakar
Date: March 2026
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('plots', exist_ok=True)


# ============================================================================
# 1. DATA LOADING & PREPROCESSING
# ============================================================================

def load_load_data():
    """Load load demand data"""
    print("Loading load data...")
    df = pd.read_csv('data/load/L1-train.csv')
    
    # Custom timestamp parsing for format like "112001 1:00"
    # Format is DDYYYY H:MM where DD is day of year and YYYY is year
    def parse_timestamp(ts):
        parts = str(ts).strip().split()
        date_part = parts[0]
        time_part = parts[1]
        
        # Extract day of year and year
        # Format: last 4 digits are year, rest are day of year
        year = int(date_part[-4:])
        day_of_year = int(date_part[:-4])
        
        # Parse time
        hour_min = time_part.split(':')
        hour = int(hour_min[0])
        minute = int(hour_min[1])
        
        # Create timestamp from year and day of year
        base_date = pd.Timestamp(year=year, month=1, day=1)
        date = base_date + pd.Timedelta(days=day_of_year - 1)
        
        return date.replace(hour=hour, minute=minute)
    
    df['TIMESTAMP'] = df['TIMESTAMP'].apply(parse_timestamp)
    
    # Handle missing LOAD values - interpolate
    df['LOAD'] = pd.to_numeric(df['LOAD'], errors='coerce')
    df['LOAD'] = df['LOAD'].interpolate(method='linear')
    
    # Drop rows where LOAD is still missing
    df = df.dropna(subset=['LOAD'])
    
    return df


def load_solar_data():
    """Load solar generation data"""
    print("Loading solar data...")
    df = pd.read_csv('data/solar/train1.csv')
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='%Y%m%d %H:%M')
    
    # Handle missing values
    df['POWER'] = df['POWER'].interpolate(method='linear')
    df = df.dropna(subset=['POWER'])
    
    return df


def load_wind_data():
    """Load and combine all wind data files"""
    print("Loading wind data...")
    wind_files = glob.glob('data/wind/Task1_W_Zone*.csv')
    
    dfs = []
    for file in wind_files:
        df = pd.read_csv(file)
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='%Y%m%d %H:%M')
        dfs.append(df)
    
    # Combine all zones
    wind_df = pd.concat(dfs, ignore_index=True)
    
    # Handle missing values
    wind_df['TARGETVAR'] = wind_df['TARGETVAR'].interpolate(method='linear')
    wind_df = wind_df.dropna(subset=['TARGETVAR'])
    
    return wind_df


def extract_time_features(df, timestamp_col='TIMESTAMP'):
    """Extract time-based features from timestamp"""
    df['hour'] = df[timestamp_col].dt.hour
    df['day_of_week'] = df[timestamp_col].dt.dayofweek
    df['day_of_year'] = df[timestamp_col].dt.dayofyear
    
    # Extract season (0: Winter, 1: Spring, 2: Summer, 3: Fall)
    df['month'] = df[timestamp_col].dt.month
    df['season'] = df['month'].apply(lambda x: (x%12 + 3)//3 - 1)
    
    return df


def preprocess_data(df, target_col, feature_cols=None):
    """
    Preprocess data: extract features, normalize, and split
    
    Args:
        df: DataFrame
        target_col: Name of target column
        feature_cols: List of feature columns (if None, use time features)
    
    Returns:
        X_train, X_test, y_train, y_test, scaler_X, scaler_y
    """
    # Extract time features
    df = extract_time_features(df)
    
    # Define features
    if feature_cols is None:
        feature_cols = ['hour', 'day_of_week', 'day_of_year', 'season']
    
    X = df[feature_cols].values
    y = df[target_col].values.reshape(-1, 1)
    
    # Normalize features
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # Split data (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, shuffle=False
    )
    
    return X_train, X_test, y_train, y_test, scaler_X, scaler_y


def prepare_lstm_data(X, y, lookback=24):
    """Prepare data for LSTM (create sequences)"""
    X_lstm, y_lstm = [], []
    
    for i in range(lookback, len(X)):
        X_lstm.append(X[i-lookback:i])
        y_lstm.append(y[i])
    
    return np.array(X_lstm), np.array(y_lstm)


# ============================================================================
# 2. FORECASTING MODELS
# ============================================================================

def build_ann_model(input_dim):
    """Build Artificial Neural Network model"""
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def build_lstm_model(input_shape):
    """Build LSTM model"""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def train_ann(X_train, y_train, X_test, y_test, name='ann'):
    """Train ANN model"""
    print(f"Training ANN for {name}...")
    
    model = build_ann_model(X_train.shape[1])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )
    
    return model, history


def train_lstm(X_train, y_train, X_test, y_test, name='lstm', lookback=24):
    """Train LSTM model"""
    print(f"Training LSTM for {name}...")
    
    # Prepare LSTM data
    X_train_lstm, y_train_lstm = prepare_lstm_data(X_train, y_train, lookback)
    X_test_lstm, y_test_lstm = prepare_lstm_data(X_test, y_test, lookback)
    
    model = build_lstm_model((lookback, X_train.shape[1]))
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train_lstm, y_train_lstm,
        validation_data=(X_test_lstm, y_test_lstm),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )
    
    return model, history, X_test_lstm, y_test_lstm


def train_random_forest(X_train, y_train, name='rf'):
    """Train Random Forest model"""
    print(f"Training Random Forest for {name}...")
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train.ravel())
    
    return model


def persistence_model(y_test):
    """Baseline: Persistence model (t+1 = t)"""
    # Simply shift the test data by one step
    y_pred = np.roll(y_test, 1)
    y_pred[0] = y_test[0]  # First prediction is same as first actual
    return y_pred


# ============================================================================
# 3. EVALUATION METRICS
# ============================================================================

def calculate_metrics(y_true, y_pred):
    """Calculate MAE, RMSE, MAPE"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # MAPE (avoid division by zero)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    
    return mae, rmse, mape


def evaluate_all_models(models_dict, X_test, y_test, scaler_y, model_type='load'):
    """Evaluate all models and return results"""
    results = []
    predictions = {}
    
    for model_name, model_data in models_dict.items():
        if model_name == 'Persistence':
            y_pred_scaled = model_data
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
        elif model_name == 'LSTM':
            model = model_data['model']
            X_test_lstm = model_data['X_test']
            y_pred_scaled = model.predict(X_test_lstm, verbose=0)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
            # Adjust y_test for LSTM (due to lookback)
            y_test_adj = y_test[24:]  # Skip first 24 points
            y_true = scaler_y.inverse_transform(y_test_adj.reshape(-1, 1))
        else:
            model = model_data
            y_pred_scaled = model.predict(X_test)
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
            y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1))
        
        # For Persistence and non-LSTM models
        if model_name != 'LSTM':
            y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1))
        
        mae, rmse, mape = calculate_metrics(y_true, y_pred)
        
        results.append({
            'Model Type': model_type.capitalize(),
            'Model': model_name,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        })
        
        predictions[model_name] = {
            'y_true': y_true.flatten(),
            'y_pred': y_pred.flatten()
        }
    
    return results, predictions


# ============================================================================
# 4. PLOTTING FUNCTIONS
# ============================================================================

def plot_predictions(predictions_dict, model_type='Load', save_path='plots/'):
    """Plot actual vs predicted for all models"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    model_names = list(predictions_dict.keys())
    
    for idx, model_name in enumerate(model_names):
        if idx < 4:
            ax = axes[idx]
            data = predictions_dict[model_name]
            
            # Plot only first 200 points for clarity
            n_points = min(200, len(data['y_true']))
            
            ax.plot(data['y_true'][:n_points], label='Actual', linewidth=2)
            ax.plot(data['y_pred'][:n_points], label='Predicted', linewidth=2, alpha=0.7)
            ax.set_title(f'{model_name} - {model_type}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel(f'{model_type} Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}{model_type.lower()}_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {model_type} predictions plot")


def plot_net_load(load, solar, wind, save_path='plots/'):
    """Plot net load profile"""
    net_load = load - (solar + wind)
    
    plt.figure(figsize=(15, 6))
    
    # Plot first 500 points for clarity
    n_points = min(500, len(load))
    time_steps = range(n_points)
    
    plt.plot(time_steps, load[:n_points], label='Load Demand', linewidth=2)
    plt.plot(time_steps, solar[:n_points], label='Solar Generation', linewidth=2)
    plt.plot(time_steps, wind[:n_points], label='Wind Generation', linewidth=2)
    plt.plot(time_steps, net_load[:n_points], label='Net Load', linewidth=2, linestyle='--', color='black')
    
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Time Steps')
    plt.ylabel('Power (normalized units)')
    plt.title('Net Load Profile: Load - (Solar + Wind)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_path}net_load_profile.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved net load profile plot")


def plot_ems_decisions(net_load, save_path='plots/'):
    """Plot EMS grid import/export decisions"""
    grid_import = np.where(net_load > 0, net_load, 0)
    grid_export = np.where(net_load < 0, -net_load, 0)
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    
    # Plot first 500 points
    n_points = min(500, len(net_load))
    time_steps = range(n_points)
    
    # Import
    axes[0].fill_between(time_steps, 0, grid_import[:n_points], 
                          alpha=0.6, color='red', label='Grid Import (Net Load > 0)')
    axes[0].set_ylabel('Import Power')
    axes[0].set_title('Energy Management System: Grid Import', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Export
    axes[1].fill_between(time_steps, 0, grid_export[:n_points], 
                          alpha=0.6, color='green', label='Grid Export (Net Load < 0)')
    axes[1].set_ylabel('Export Power')
    axes[1].set_xlabel('Time Steps')
    axes[1].set_title('Energy Management System: Grid Export', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}ems_decisions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved EMS decisions plot")


# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================

def train_models_for_dataset(df, target_col, feature_cols, dataset_name):
    """Train all models for a specific dataset"""
    print(f"\n{'='*80}")
    print(f"Processing {dataset_name} Dataset")
    print(f"{'='*80}")
    
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = preprocess_data(
        df, target_col, feature_cols
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train models
    models_dict = {}
    
    # 1. ANN
    ann_model, _ = train_ann(X_train, y_train, X_test, y_test, dataset_name)
    models_dict['ANN'] = ann_model
    ann_model.save(f'models/ann_{dataset_name.lower()}.h5')
    
    # 2. LSTM
    lstm_model, _, X_test_lstm, y_test_lstm = train_lstm(
        X_train, y_train, X_test, y_test, dataset_name
    )
    models_dict['LSTM'] = {'model': lstm_model, 'X_test': X_test_lstm}
    lstm_model.save(f'models/lstm_{dataset_name.lower()}.h5')
    
    # 3. Random Forest
    rf_model = train_random_forest(X_train, y_train, dataset_name)
    models_dict['Random Forest'] = rf_model
    
    # Save Random Forest using joblib
    import joblib
    joblib.dump(rf_model, f'models/rf_{dataset_name.lower()}.pkl')
    
    # 4. Persistence Model
    y_pred_persistence = persistence_model(y_test)
    models_dict['Persistence'] = y_pred_persistence
    
    # Evaluate models
    results, predictions = evaluate_all_models(
        models_dict, X_test, y_test, scaler_y, dataset_name
    )
    
    # Plot predictions
    plot_predictions(predictions, dataset_name)
    
    # Return best model predictions (using ANN for simplicity)
    y_pred_best = ann_model.predict(X_test, verbose=0)
    y_pred_best = scaler_y.inverse_transform(y_pred_best)
    
    return results, y_pred_best.flatten()


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("AI-BASED FORECASTING FOR MICROGRID ENERGY MANAGEMENT")
    print("="*80 + "\n")
    
    # ========================================================================
    # STEP 1: Load Data
    # ========================================================================
    load_df = load_load_data()
    solar_df = load_solar_data()
    wind_df = load_wind_data()
    
    print(f"\nData loaded successfully!")
    print(f"Load data: {load_df.shape[0]} records")
    print(f"Solar data: {solar_df.shape[0]} records")
    print(f"Wind data: {wind_df.shape[0]} records")
    
    # ========================================================================
    # STEP 2: Train Models for Each Dataset
    # ========================================================================
    all_results = []
    
    # LOAD FORECASTING
    load_feature_cols = ['hour', 'day_of_week', 'day_of_year', 'season']
    load_results, load_predictions = train_models_for_dataset(
        load_df, 'LOAD', load_feature_cols, 'Load'
    )
    all_results.extend(load_results)
    
    # SOLAR FORECASTING
    solar_feature_cols = ['hour', 'day_of_week', 'day_of_year', 'season']
    solar_results, solar_predictions = train_models_for_dataset(
        solar_df, 'POWER', solar_feature_cols, 'Solar'
    )
    all_results.extend(solar_results)
    
    # WIND FORECASTING (aggregate all zones)
    wind_df_agg = wind_df.groupby('TIMESTAMP').agg({
        'TARGETVAR': 'mean',
        'U10': 'mean',
        'V10': 'mean',
        'U100': 'mean',
        'V100': 'mean'
    }).reset_index()
    
    wind_feature_cols = ['hour', 'day_of_week', 'day_of_year', 'season']
    wind_results, wind_predictions = train_models_for_dataset(
        wind_df_agg, 'TARGETVAR', wind_feature_cols, 'Wind'
    )
    all_results.extend(wind_results)
    
    # ========================================================================
    # STEP 3: Display Results Table
    # ========================================================================
    print("\n" + "="*80)
    print("MODEL EVALUATION RESULTS")
    print("="*80 + "\n")
    
    results_df = pd.DataFrame(all_results)
    print(results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv('results/model_evaluation_results.csv', index=False)
    print("\n[DONE] Results saved to results/model_evaluation_results.csv")
    
    # ========================================================================
    # STEP 4: Net Load Calculation
    # ========================================================================
    print("\n" + "="*80)
    print("NET LOAD CALCULATION")
    print("="*80)
    
    # Ensure all predictions have the same length (use minimum length)
    min_len = min(len(load_predictions), len(solar_predictions), len(wind_predictions))
    
    load_pred = load_predictions[:min_len]
    solar_pred = solar_predictions[:min_len]
    wind_pred = wind_predictions[:min_len]
    
    net_load = load_pred - (solar_pred + wind_pred)
    
    print(f"\nNet Load Statistics:")
    print(f"Mean Net Load: {np.mean(net_load):.4f}")
    print(f"Max Net Load: {np.max(net_load):.4f}")
    print(f"Min Net Load: {np.min(net_load):.4f}")
    
    # ========================================================================
    # STEP 5: Energy Management System (EMS)
    # ========================================================================
    print("\n" + "="*80)
    print("ENERGY MANAGEMENT SYSTEM (EMS)")
    print("="*80)
    
    grid_import = np.where(net_load > 0, net_load, 0)
    grid_export = np.where(net_load < 0, -net_load, 0)
    
    total_import = np.sum(grid_import)
    total_export = np.sum(grid_export)
    
    print(f"\nEMS Statistics:")
    print(f"Total Grid Import: {total_import:.4f} units")
    print(f"Total Grid Export: {total_export:.4f} units")
    print(f"Net Grid Exchange: {total_import - total_export:.4f} units")
    print(f"Import hours: {np.sum(net_load > 0)} / {len(net_load)}")
    print(f"Export hours: {np.sum(net_load < 0)} / {len(net_load)}")
    
    # ========================================================================
    # STEP 6: Generate All Plots
    # ========================================================================
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80 + "\n")
    
    plot_net_load(load_pred, solar_pred, wind_pred)
    plot_ems_decisions(net_load)
    
    # ========================================================================
    # COMPLETION
    # ========================================================================
    print("\n" + "="*80)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\n[DONE] Models trained and saved in 'models/' folder")
    print("[DONE] Results saved in 'results/' folder")
    print("[DONE] Plots saved in 'plots/' folder")
    print("\nAll tasks completed!")


if __name__ == "__main__":
    main()
