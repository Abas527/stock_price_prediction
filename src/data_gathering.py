import yfinance as yf
import pandas as pd


ticker=yf.Ticker("AAPL")

data=ticker.history(period="10y")

data.to_csv("data/Apple_stock_price.csv")

