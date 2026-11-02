from pathlib import Path
from .download import download_raw_data
from .preprocessing import prepare_env_prices
from .data_check import validate_prices, summarize_coverage
from .split import create_split
from . import config


def main():
    raw_data, meta_data = download_raw_data(config.TICKERS, config.START, config.END)

    env_prices = prepare_env_prices(raw_data, config.TICKERS)

    validate_prices(env_prices, config.TICKERS)

    coverage_summary = summarize_coverage(env_prices)
    print("Coverage Summary:")
    print(coverage_summary)

    
    train_prices = create_split(env_prices, config.TRAIN_SPLIT[0], config.TRAIN_SPLIT[1], window=30)
    val_prices = create_split(env_prices, config.VAL_SPLIT[0], config.VAL_SPLIT[1], window=30)
    test_prices = create_split(env_prices, config.TEST_SPLIT[0], config.TEST_SPLIT[1], window=30)

    # Save the processed data to CSV files
    env_prices.to_csv(config.PROCESSED_DATA_DIR / "env_prices.csv")

    train_prices.to_csv(config.PROCESSED_DATA_DIR / "train_prices.csv")
    val_prices.to_csv(config.PROCESSED_DATA_DIR / "val_prices.csv")
    test_prices.to_csv(config.PROCESSED_DATA_DIR / "test_prices.csv")


if  __name__ == "__main__":
    main()