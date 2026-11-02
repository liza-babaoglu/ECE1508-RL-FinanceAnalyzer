from pathlib import Path
import pandas as pd

def prepare_env_prices(raw_data, tickers):
    close_prices = raw_data["Close"]
    env_prices = close_prices.copy()
    env_prices = env_prices[tickers]

    return env_prices