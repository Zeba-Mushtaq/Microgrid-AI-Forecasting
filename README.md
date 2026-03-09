# AI-Based Forecasting Models for Load Demand, Renewable Generation and Microgrid Control Mechanisms for Adaptive Energy Dispatch

## Project Overview
This project implements AI-based forecasting models for microgrid energy management, including load demand forecasting, renewable energy generation forecasting (solar and wind), and an energy management system for adaptive energy dispatch.

## Project Structure
```
MicrogridProject/
├── data/
│   ├── load/     → Load demand data
│   ├── solar/    → Solar generation data
│   └── wind/     → Wind generation data (10 zones)
├── models/       → Trained machine learning models
│   ├── ann_load.h5, ann_solar.h5, ann_wind.h5
│   ├── lstm_load.h5, lstm_solar.h5, lstm_wind.h5
│   └── rf_load.pkl, rf_solar.pkl, rf_wind.pkl
├── results/      → Evaluation results
│   └── model_evaluation_results.csv
├── plots/        → Visualization outputs
│   ├── load_predictions.png
│   ├── solar_predictions.png
│   ├── wind_predictions.png
│   ├── net_load_profile.png
│   └── ems_decisions.png
├── main.py       → Main project pipeline
└── generate_plots.py  → Additional plotting script
```

## Implementation Details

### 1. Data Loading & Preprocessing
- **Load Data**: Custom timestamp parsing (DDYYYY format) for day-of-year format
- **Solar Data**: Standard datetime parsing
- **Wind Data**: Aggregated from 10 zones, averaged across zones
- **Preprocessing Steps**:
  - Missing value handling using linear interpolation
  - Min-Max normalization for all features
  - Time feature extraction: hour, day of week, day of year, season

### 2. Forecasting Models
Four models were trained for each dataset (Load, Solar, Wind):

#### a. Artificial Neural Network (ANN)
- Architecture: 128→64→32→1 neurons with Dropout
- Activation: ReLU
- Optimizer: Adam
- Loss: MSE

#### b. LSTM (Long Short-Term Memory)
- Architecture: 128 LSTM→64 LSTM→32 Dense→1
- Lookback window: 24 time steps
- Activation: ReLU for dense layers
- Optimizer: Adam

#### c. Random Forest
- 100 estimators
- Max depth: 20
- Min samples split: 5

#### d. Persistence Model (Baseline)
- Simple baseline: prediction at t+1 = actual at t

### 3. Evaluation Metrics
Models were evaluated using:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **MAPE** (Mean Absolute Percentage Error)

### 4. Model Performance

| Dataset | Model | MAE | RMSE | MAPE |
|---------|-------|-----|------|------|
| **Load** | ANN | 26.23 | 34.52 | 16.23% |
| | LSTM | 25.71 | 34.36 | 15.63% |
| | Random Forest | 29.75 | 39.74 | 17.89% |
| | Persistence | 8.45 | 10.79 | 5.47% |
| **Solar** | ANN | 0.082 | 0.135 | - |
| | LSTM | 0.080 | 0.131 | - |
| | Random Forest | 0.050 | 0.091 | - |
| | Persistence | 0.073 | 0.122 | - |
| **Wind** | ANN | 0.236 | 0.268 | 202.1% |
| | LSTM | 0.236 | 0.266 | 213.9% |
| | Random Forest | 0.280 | 0.349 | 283.4% |
| | Persistence | 0.034 | 0.048 | 13.1% |

**Key Findings**:
- For **Load**: Persistence model performs best (indicating high autocorrelation)
- For **Solar**: Random Forest achieves lowest error
- For **Wind**: Persistence model is most accurate

### 5. Net Load Calculation
Net Load = Load Demand - (Solar Generation + Wind Generation)

### 6. Energy Management System (EMS)
Simple rule-based dispatch:
- **Net Load > 0**: Import power from grid (deficit)
- **Net Load < 0**: Export power to grid (surplus)

### 7. Visualizations
Generated plots in `plots/` folder:
1. **Load predictions**: Actual vs Predicted for all models
2. **Solar predictions**: Actual vs Predicted for all models
3. **Wind predictions**: Actual vs Predicted for all models
4. **Net load profile**: Combined load, solar, wind, and net load
5. **EMS decisions**: Grid import/export visualization

## Dependencies
```
- pandas
- numpy
- matplotlib
- scikit-learn
- tensorflow/keras
- joblib
```

## Usage

### Run Complete Pipeline
```bash
python main.py
```

This will:
1. Load and preprocess all datasets
2. Train all models (ANN, LSTM, RF, Persistence)
3. Evaluate and save results
4. Generate prediction plots
5. Save trained models

### Generate Additional Plots
```bash
python generate_plots.py
```

Generates net load and EMS decision plots from saved models.

## Key Features
✅ Multi-model comparison (ANN, LSTM, Random Forest, Persistence)
✅ Comprehensive evaluation metrics (MAE, RMSE, MAPE)
✅ Time-series feature engineering
✅ Net load calculation and analysis
✅ Energy management system logic
✅ Professional visualizations
✅ Model persistence (saved for future use)

## Future Enhancements
- Implement advanced EMS strategies (battery storage, demand response)
- Hyperparameter tuning for better model performance
- Ensemble methods combining multiple models
- Real-time forecasting capabilities
- Economic dispatch optimization
- Uncertainty quantification

## Author
Abubakar - March 2026

## Thesis Title
"AI-Based Forecasting Models for Load Demand, Renewable Generation and Microgrid Control Mechanisms for Adaptive Energy Dispatch"
