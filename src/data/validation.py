import pandas as pd

def validate_prices(prices, expected_tickers):
    if (not prices.index.is_monotonic_increasing):
        raise ValueError("Prices index is not sorted in ascending order")
    if (not prices.index.is_unique):
        raise ValueError("Prices index is not unique")
    if(prices.index.hasnans): #hasnas: check if there are any NaN values in the index
        raise ValueError("Prices index has NaN values")

def summarize_coverage(prices):
    coverage = []
    for ticker in prices.columns:
        coverage.append({
            "ticker": ticker,
            "First Valid Date": prices[ticker].first_valid_index(),
            "Last Valid Date": prices[ticker].last_valid_index(),
            "Valid Prices": prices[ticker].count(),
            "Missing Prices": prices[ticker].isna().sum()
        })
    coverage = pd.DataFrame(coverage)
    return coverage  