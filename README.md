# Daily Temperature Forecasting: RNN vs LSTM vs GRU

## Project Description
This project predicts the next day's temperature based on the past 7 days of weather data.
I compare three recurrent neural network models to see which one performs best.

## Dataset
- Name: Daily Climate Time Series Data
- Source: https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data
- Size: ~1,462 daily records
- Features: meantemp, humidity, wind_speed, meanpressure

## Models
- Simple RNN — baseline model
- LSTM — improved model 1
- GRU — improved model 2

## Evaluation Metrics
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)

## Tools
- Python, PyTorch, Google Colab

## How to Run
1. Download dataset from Kaggle link above
2. Open notebooks/ folder in Google Colab
3. Run the notebook step by step
