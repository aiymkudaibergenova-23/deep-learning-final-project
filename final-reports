# Final Report: Daily Temperature Forecasting: RNN vs LSTM vs GRU

## 1. Project Title
Daily Temperature Forecasting: RNN vs LSTM vs GRU

## 2. Problem Statement
Weather forecasting is important for agriculture, transportation,
and daily planning. This project predicts the next day's mean
temperature using the previous 7 days of data. We compare three
recurrent neural network architectures — Simple RNN, LSTM, and GRU
— to find which works best on a small real-world dataset.

## 3. Dataset Description
- Source: Daily Climate Time Series Data (Kaggle)
- Link: https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data
- File: DailyDelhiClimateTrain.csv
- Size: 1462 rows, 5 columns
- Features: date, meantemp, humidity, wind_speed, meanpressure
- Target: meantemp (daily mean temperature in Celsius)
- Missing values: none

## 4. Data Preprocessing
- Used only meantemp as input feature
- Normalization: MinMaxScaler to range [0, 1]
- Sliding window: last 7 days → predict next day
- Sequences: X shape (1455, 7, 1), y shape (1455, 1)
- Split (no shuffling — time series order preserved):
  - Train: 1018 samples (70%)
  - Validation: 218 samples (15%)
  - Test: 219 samples (15%)
- Converted to PyTorch tensors (float32)

## 5. Model Architecture
All models follow the same pattern:

| Model | Layers |
|-------|--------|
| Simple RNN | RNN(input=1, hidden=32) + Linear(32, 1) |
| LSTM | LSTM(input=1, hidden=32) + Linear(32, 1) |
| GRU | GRU(input=1, hidden=32) + Linear(32, 1) |

## 6. Training Setup
- Framework: PyTorch
- Optimizer: Adam (lr=0.001)
- Loss function: MSELoss
- Epochs: 50
- Batch: full batch training
- No shuffling (time series data)

## 7. Evaluation Metrics
- **MAE (Mean Absolute Error)**: average absolute difference
  between predicted and real values. Lower is better.
- **RMSE (Root Mean Squared Error)**: penalizes large errors
  more than MAE. Lower is better.
- Both metrics are computed on the original scale (°C)
  after inverse transformation with MinMaxScaler.

## 8. Results

| Model | MAE | RMSE |
|-------|-----|------|
| Simple RNN | **1.65** | **2.14** |
| LSTM | 3.48 | 4.64 |
| GRU | 3.32 | 4.37 |

## 9. Error Analysis
- Simple RNN makes larger errors during seasonal transitions
  (e.g. winter to spring, summer to autumn)
- LSTM and GRU overfit on the small training set,
  leading to worse generalization on the test set
- All models struggle with sudden temperature changes
- Simple RNN errors are more stable and consistent

## 10. Limitations
- Small dataset (~1400 samples) limits model complexity
- Only one feature (meantemp) used — adding humidity,
  wind_speed could improve results
- No hyperparameter tuning was performed
- Window size of 7 days was chosen manually,
  not optimized
- Models were not tested on other cities or time periods

## 11. Conclusion
Simple RNN outperformed LSTM and GRU on this dataset.
The main reason is the small size of the dataset —
complex models with more parameters tend to overfit
when data is limited. For small time series datasets,
simpler models can achieve better generalization.
Best result: Simple RNN with MAE=1.65 and RMSE=2.14.

## 12. References
- Daily Climate Time Series Data:
  https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data
- PyTorch Documentation: https://pytorch.org/docs/
- Hochreiter, S., Schmidhuber, J. (1997). Long Short-Term Memory.
  Neural Computation, 9(8), 1735-1780.
- Cho, K. et al. (2014). Learning Phrase Representations using
  RNN Encoder-Decoder for Statistical Machine Translation.
