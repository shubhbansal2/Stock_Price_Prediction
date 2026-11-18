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

Test period: 2023-01-02 to 2024-12-31, 490 trading days, one-day-ahead forecasts. RMSE in INR (Reliance) and index points (NIFTY). "vs naive" = model MAE / naive MAE (below 1 beats naive).

| Ticker | Model | MAPE % | RMSE | vs naive | Direction % |
| --- | --- | --- | --- | --- | --- |
| RELIANCE.NS | naive | 0.929 | 16.87 | 1.000 | - |
| RELIANCE.NS | ARIMA(0,1,2) | 0.931 | 16.84 | 1.002 | 49.0 |
| RELIANCE.NS | LSTM | 0.930 | 16.87 | 1.001 | 50.6 |
| RELIANCE.NS | hybrid | 0.931 | 16.86 | 1.001 | 50.2 |
| ^NSEI | naive | 0.550 | 168.18 | 1.000 | - |
| ^NSEI | ARIMA(0,1,0) | 0.550 | 168.18 | 1.000 | - |
| ^NSEI | LSTM | 0.547 | 167.86 | 0.995 | 55.6 |
| ^NSEI | hybrid | 0.547 | 167.86 | 0.995 | 55.6 |

**Takeaway:** none of the models meaningfully beats the naive forecast. The hybrid's MAPE is 0.2% *higher* than naive for Reliance and 0.5% lower for NIFTY, differences far too small to matter. Directional accuracy is near a coin flip: about 50% for Reliance (50.8% of test days were up days) and 55.6% for NIFTY, below the 56.2% you'd get by always predicting "up". This is what weak-form market efficiency predicts for daily closing prices. AIC picked ARIMA(0,1,0) for NIFTY, i.e. a random walk.

Other checks: ADF on training-period log price p = 0.88 (Reliance) and 0.94 (NIFTY), non-stationary; on log returns p < 1e-19, stationary. Full outputs are in the executed notebook and `results/`.

## Version history

`archive/ARIMA_LSTM_model_v1.ipynb` is the first version. It had three problems that this version fixes:

- look-ahead leakage: the LSTM input used the next day's ARIMA residual (`shift(-1)`), and the scaler was fitted on the full series;
- the test set doubled as the validation set;
- no baselines, and metrics were computed on scaled values instead of prices.
