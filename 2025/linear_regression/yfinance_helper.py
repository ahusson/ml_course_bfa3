import yfinance as yf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

start_date = "2018-01-01"
end_date = "2023-12-31"

nvda_data = yf.download("NVDA", start=start_date, end=end_date)
sp500_data = yf.download("^GSPC", start=start_date, end=end_date)

nvda_data["Daily Return"] = nvda_data["Close"].pct_change()
sp500_data["Daily Return"] = sp500_data["Close"].pct_change()

returns = pd.DataFrame({
    "NVDA": nvda_data["Daily Return"],
    "S&P500": sp500_data["Daily Return"]
}).dropna()

X = returns["S&P500"].values.reshape(-1, 1)
y = returns["NVDA"].values
