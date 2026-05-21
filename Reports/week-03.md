# Week 03 Progress Report

## Task
Comparison of Simple RNN, LSTM, and GRU for daily temperature forecasting.

## Models Compared
All models use the same architecture pattern:
- Input size: 1
- Hidden size: 32
- Output: Linear(32, 1)
- Optimizer: Adam (lr=0.001)
- Loss: MSELoss
- Epochs: 50

## Results

| Model | MAE | RMSE |
|-------|-----|------|
| Simple RNN | 1.65 | 2.14 |
| LSTM | 3.48 | 4.64 |
| GRU | 3.32 | 4.37 |

## Analysis
- Simple RNN outperformed LSTM and GRU on this dataset
- Likely reason: the dataset is small (~1400 samples); complex models like LSTM and GRU require more data to show their advantage
- LSTM and GRU have more parameters and tend to overfit on small datasets
- Sliding window of 7 days is sufficient for Simple RNN to capture the temperature pattern

## Conclusion
For small time series datasets, simpler models can outperform complex ones.
Simple RNN achieved the best results with MAE=1.65 and RMSE=2.14.
