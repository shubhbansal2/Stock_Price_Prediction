# Hybrid ARIMA-LSTM Stock Forecasting

One-day-ahead forecasts of daily closing prices for **Reliance Industries (RELIANCE.NS)** and the **NIFTY 50 index (^NSEI)**, 2015-2024, using a hybrid of ARIMA and an LSTM.

## Method

The hybrid follows Zhang (2003):

```
forecast(t) = ARIMA forecast(t) + LSTM forecast of ARIMA residual(t)
```

1. **Data**: adjusted daily closes from Yahoo Finance, log-transformed.
2. **Split (chronological)**: train 2015-2021, validation 2022, test 2023-2024.
3. **Stationarity**: ADF test on the training period (log price is non-stationary, log returns are stationary), so d = 1.
4. **ARIMA(p, 1, q)**: p and q chosen by lowest AIC on the training period. Parameters are fitted on train only; through validation and test the model is updated one day at a time with each new observation (walk-forward, no refit on future data).
5. **LSTM on residuals**: the previous 60 ARIMA residuals predict the next residual. LSTM(64) -> Dropout(0.3) -> Dense(25) -> Dense(1). Scaler fitted on training residuals; early stopping on validation.
6. **Baselines**: naive (tomorrow = today), ARIMA only, and an LSTM-only model trained on the previous 60 log returns.
7. **Metrics** in price units on the test period: MAPE, RMSE, MASE (scaled by the in-sample naive error) and directional accuracy.

## Run

```
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute ARIMA_LSTM_hybrid.ipynb --output ARIMA_LSTM_hybrid.ipynb
```

Results are written to `results/metrics.csv` and `results/summary.csv`. To run offline, put CSV files with `Date` and `Close` columns in `data/` (`data/RELIANCE.NS.csv`, `data/NSEI.csv`).

## Results

Run the notebook to fill this in.

| Ticker | Model | MAPE % | RMSE | MASE | Direction % |
| --- | --- | --- | --- | --- | --- |
| RELIANCE.NS | naive | | | | |
| RELIANCE.NS | ARIMA | | | | |
| RELIANCE.NS | LSTM | | | | |
| RELIANCE.NS | hybrid | | | | |
| ^NSEI | naive | | | | |
| ^NSEI | ARIMA | | | | |
| ^NSEI | LSTM | | | | |
| ^NSEI | hybrid | | | | |

## Version history

`archive/ARIMA_LSTM_model_v1.ipynb` is the first version. It had three problems that this version fixes:

- look-ahead leakage: the LSTM input used the next day's ARIMA residual (`shift(-1)`), and the scaler was fitted on the full series;
- the test set doubled as the validation set;
- no baselines, and metrics were computed on scaled values instead of prices.
