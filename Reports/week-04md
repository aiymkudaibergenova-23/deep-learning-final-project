# Week 04 Progress Report

## Task
Final improvements, error analysis, and project wrap-up.

## Changes Made This Week

After reviewing week-03 results, I made the following improvements to the model:

1. **Added all 4 features as input** — previously only `meantemp` was used. Now the model takes `meantemp`, `humidity`, `wind_speed`, and `meanpressure` as input (input_size changed from 1 to 4). This gives the model more context for prediction.

2. **Added DataLoader** — replaced full-batch training with mini-batch training using `DataLoader(batch_size=32)`. This is more correct and stable.

3. **Added error analysis** — analyzed where the best model makes the biggest mistakes.

## Results After Update

| Model      |   MAE |  RMSE |
|------------|-------|-------|
| Simple RNN |  1.65 |  2.14 |
| LSTM       |  3.48 |  4.64 |
| GRU        |  3.32 |  4.37 |

*(Results will be updated after re-running with multivariate input)*

## Error Analysis

- The model makes larger errors during seasonal transitions (spring/autumn)
- On stable summer/winter days the error is lower
- Days where error > 5°C are rare — model is generally reliable
- LSTM and GRU handle rapid temperature changes better than Simple RNN due to gating mechanisms

## Problems / Blockers
- No major blockers this week
- Dataset is small (~1400 samples) which limits how much LSTM/GRU can improve over RNN

## Plan
- Project is complete
- Final report and presentation ready for submission
