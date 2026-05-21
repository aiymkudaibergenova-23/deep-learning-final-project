# Project Proposal

## 1. Project Title
Daily Temperature Forecasting: RNN vs LSTM vs GRU

---

## 2. Problem Statement
I want to predict tomorrow's temperature based on the past 7 days of weather data.

This is useful because weather prediction helps people plan their daily life. The model will output a single number — the predicted temperature for the next day.

---

## 3. Dataset
- **Name:** Daily Climate Time Series Data
- **Source:** https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data
- **Size:** ~1,462 rows (daily records)
- **Input:** meantemp, humidity, wind_speed, meanpressure
- **Target:** meantemp (next day)
- **Format:** CSV

---

## 4. Planned Method
- **Baseline:** Simple RNN
- **Model 1:** LSTM
- **Model 2:** GRU
- **Loss:** Mean Squared Error (MSE)
- **Metrics:** MAE, RMSE
- **Split:** 70% train, 15% validation, 15% test

---

## 5. Expected Challenges
- Dataset is small (~1400 samples), so overfitting is possible
- Choosing the right window size (how many past days to use)
- Making sure all models are compared fairly with same settings

---

## 6. Weekly Plan

| Week | Work | Output |
|------|------|--------|
| Week 1 | Setup repo, load dataset, basic EDA | Proposal, README, EDA notebook |
| Week 2 | Preprocessing, normalization, baseline RNN | RNN results, week-02 report |
| Week 3 | Train LSTM and GRU, compare results | Comparison table, plots, week-03 report |
| Week 4 | Error analysis, final report, presentation | Final code, final report, slides |
