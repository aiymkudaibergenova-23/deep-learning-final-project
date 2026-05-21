# Week 02 Progress Report

## Task
Daily Temperature Forecasting using Simple RNN (Baseline).

## Dataset
- Source: Daily Climate Time Series Data (Kaggle)
- File: DailyDelhiClimateTrain.csv
- Shape: 1462 rows, 5 columns (date, meantemp, humidity, wind_speed, meanpressure)
- Missing values: none

## EDA Summary
- Target variable: meantemp (daily mean temperature in Celsius)
- Temperature ranges from 6°C to 38.7°C
- Clear seasonal pattern visible in the time series plot

## Data Preparation
- Feature: meantemp only
- Normalization: MinMaxScaler to range [0, 1]
- Sliding window: last 7 days → predict next day
- Dataset split: 70% train / 15% val / 15% test
  - Train: 1018 samples
  - Val: 218 samples
  - Test: 219 samples

## Baseline Model: Simple RNN
- Architecture: RNN(input=1, hidden=32) + Linear(32, 1)
- Optimizer: Adam (lr=0.001)
- Loss: MSELoss
- Epochs: 50

## Results
| Model | MAE | RMSE |
|-------|-----|------|
| Simple RNN | 4.44 | 5.06 |

## Observations
- Loss decreased from 0.18 to 0.024 over 50 epochs
- Model converged around epoch 30-40
- Baseline established for comparison with LSTM and GRU
