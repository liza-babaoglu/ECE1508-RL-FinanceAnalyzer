from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data/raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data/processed"

TICKERS = ["SPY", "TLT", "GLD"]

START = "2007-01-01"
END = "2026-01-01"

TRAIN_SPLIT = ("2008-01-01", "2019-01-01")
VAL_SPLIT = ("2019-01-01", "2021-01-01")
TEST_SPLIT = ("2021-01-01", "2026-01-01")