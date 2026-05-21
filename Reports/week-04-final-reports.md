# Final Report: Daily Temperature Forecasting

## Project Overview
This project compares three recurrent neural network architectures —
Simple RNN, LSTM, and GRU — for daily temperature forecasting.

## Dataset
- Source: Daily Climate Time Series Data (Kaggle)
- File: DailyDelhiClimateTrain.csv
- Size: 1462 rows, 5 columns
- Target: meantemp (daily mean temperature, °C)
- Missing values: none

## Methodology
- Feature: meantemp only
- Normalization: MinMaxScaler [0, 1]
- Sliding window: 7 days → predict next day
- Split: 70% train / 15% val / 15% test
- Framework: PyTorch
- Optimizer: Adam (lr=0.001)
- Loss: MSELoss
- Epochs: 50

## Model Architectures

| Model | Layers |
|-------|--------|
| Simple RNN | RNN(1, 32) + Linear(32, 1) |
| LSTM | LSTM(1, 32) + Linear(32, 1) |
| GRU | GRU(1, 32) + Linear(32, 1) |

## Results

| Model | MAE | RMSE |
|-------|-----|------|
| Simple RNN | **1.65** | **2.14** |
| LSTM | 3.48 | 4.64 |
| GRU | 3.32 | 4.37 |

## Key Findings
1. Simple RNN achieved the best results on this dataset
2. LSTM and GRU underperformed due to small dataset size (~1400 samples)
3. Complex models with more parameters tend to overfit on small datasets
4. A sliding window of 7 days captures the seasonal temperature pattern well

## Conclusion
For small time series datasets, simpler models can outperform complex ones.
Simple RNN is the best model for this task with MAE=1.65 and RMSE=2.14.
