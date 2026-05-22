# Final Report: Daily Temperature Forecasting — RNN vs LSTM vs GRU

**Student:** Кудайбергенова Айым  
**Course:** Applied Deep Learning, Narxoz University  
**Date:** May 2026

---

## 1. Project Title
Daily Temperature Forecasting: RNN vs LSTM vs GRU

---

## 2. Problem Statement

The goal of this project is to predict tomorrow's mean temperature based on the past 7 days of weather data. Weather forecasting is useful for daily planning, agriculture, and energy management. The model takes 7 days of weather features as input and outputs a single number — the predicted temperature for the next day.

---

## 3. Dataset Description

- **Name:** Daily Climate Time Series Data
- **Source:** https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data
- **Size:** 1,462 rows (daily records from 2013 to 2017)
- **Features:** meantemp, humidity, wind_speed, meanpressure
- **Target:** meantemp (next day)
- **Format:** CSV
- **Missing values:** None

---

## 4. Data Preprocessing

1. Selected all 4 features: `meantemp`, `humidity`, `wind_speed`, `meanpressure`
2. Normalized all features to range [0, 1] using `MinMaxScaler`
3. Created sliding window sequences: last 7 days → next day temperature
4. Split data into train / validation / test (70% / 15% / 15%)
5. Converted to PyTorch tensors and created DataLoader with batch_size=32

---

## 5. Model Architecture

All three models follow the same pattern for fair comparison:

| Parameter    | Value |
|--------------|-------|
| Input size   | 4 (all features) |
| Hidden size  | 32 |
| Output       | Linear(32 → 1) |
| Layers       | 1 |

**Simple RNN** — baseline model. Uses a basic recurrent unit with no gating.

**LSTM** — adds input gate, forget gate, and output gate. Better at remembering long-term patterns.

**GRU** — simplified version of LSTM with reset and update gates. Fewer parameters than LSTM.

---

## 6. Training Setup

- **Loss function:** MSELoss
- **Optimizer:** Adam (lr=0.001)
- **Epochs:** 50
- **Batch size:** 32
- **Framework:** PyTorch

---

## 7. Evaluation Metrics

- **MAE (Mean Absolute Error)** — average absolute difference between predicted and real temperature in °C. Lower is better.
- **RMSE (Root Mean Squared Error)** — penalizes large errors more than MAE. Lower is better.

---

## 8. Results Table

| Model      |   MAE |  RMSE |
|------------|-------|-------|
| Simple RNN |  1.65 |  2.14 |
| LSTM       |  3.48 |  4.64 |
| GRU        |  3.32 |  4.37 |

Simple RNN achieved the best results on this dataset. LSTM and GRU performed worse, likely because the dataset is small (~1400 samples) and complex gating mechanisms did not have enough data to show their advantage.

---

## 9. Error Analysis

- The model makes larger errors during seasonal transitions (spring and autumn), when temperature changes rapidly from one day to the next
- On stable summer and winter days the error is consistently low
- Days where absolute error exceeds 5°C are rare
- LSTM and GRU handle sudden temperature spikes slightly better than Simple RNN due to gating mechanisms, even though their overall MAE is higher

---

## 10. Limitations

- Dataset is small (~1400 samples), which limits the advantage of complex models like LSTM and GRU
- Only one city (Delhi) — the model may not generalize to other climates
- Window size of 7 days was chosen manually, not optimized
- No external factors like seasonality encoding or calendar features were added
- Models were not tested on the official test set from Kaggle (DailyDelhiClimateTest.csv)

---

## 11. Conclusion

This project compared three recurrent models — Simple RNN, LSTM, and GRU — for daily temperature forecasting. Simple RNN performed best with MAE=1.65 and RMSE=2.14, which shows that for small time series datasets, simpler models can outperform more complex ones. Using all 4 weather features as input improved the model compared to using only temperature. Future improvements could include a larger dataset, longer window size, or adding a Transformer-based model for comparison.

---

## 12. References

[1] S. Vrao, "Daily Climate Time Series Data," Kaggle, 2020. [Online]. Available: https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data

[2] S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," Neural Computation, vol. 9, no. 8, pp. 1735–1780, 1997.

[3] K. Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation," arXiv:1406.1078, 2014.

[4] A. Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library," NeurIPS, 2019.
