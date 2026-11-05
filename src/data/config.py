from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data/raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data/processed"

# These stocks are good for our starting point. (we can always add/remove more)

# SPY = US stocks (S&P 500) 
# TLT = Long-term US Treasury bonds
# GLD = Gold
# EFA = International stocks
# VNQ = US real estate 
# CCJ = Cameco (Canadian uranium miner, NYSE listing) |
# TD = TD Bank (Canadian bank, NYSE listing) |

# 1. Canadian stocks such as CCO (uranium) and TD (bank) but chose their US equivalent
# to have a smoother currency integration in USD (maybe: in the future consider CAD as well...)

TICKERS = ["SPY", "TLT", "GLD", "EFA", "VNQ", "CCJ", "TD"] # chosing USD equivalent

START = "2007-01-01"
END = "2026-01-01"

# Never shuffle time series: train on the past, test on the future.
# 2. These include the covid crash, the US housing crash etc... for now we are not modifying anything
# until we run some tests, analyze the results...
# 3. I also realized that CCJ had a downward trend 2008–2018 but then boomed in 2021–2025...

TRAIN_SPLIT = ("2008-01-01", "2019-01-01")
VAL_SPLIT = ("2019-01-01", "2021-01-01")
TEST_SPLIT = ("2021-01-01", "2026-01-01")

TRADING_DAYS_PER_YEAR = 252   # one training episode = one trading year

#Coverage Summary:
#   ticker First Valid Date Last Valid Date  Valid Prices  Missing Prices
# 0    SPY       2007-01-03      2025-12-31          4780               0
# 1    TLT       2007-01-03      2025-12-31          4780               0
# 2    GLD       2007-01-03      2025-12-31          4780               0
# 3    EFA       2007-01-03      2025-12-31          4780               0
# 4    VNQ       2007-01-03      2025-12-31          4780               0
# 5    CCJ       2007-01-03      2025-12-31          4780               0
# 6     TD       2007-01-03      2025-12-31          4780               0