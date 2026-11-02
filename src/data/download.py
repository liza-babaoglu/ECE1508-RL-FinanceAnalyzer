import yfinance as yf
import json
from datetime import datetime, timezone
from .config import RAW_DATA_DIR


def download_raw_data(tickers, start, end, interval="1d", auto_adjust=True):
    data_file = RAW_DATA_DIR / "raw_data.csv"

    raw_data = yf.download(tickers=tickers, start=start, end=end, interval=interval, auto_adjust=auto_adjust)


    if raw_data.empty:
        raise ValueError("No data downloaded!")
    raw_data.to_csv(data_file)

    meta_data = {
        "requested_tickers": tickers,
        "start_date": start,
        "end_date": end,
        "interval": interval,
        "auto_adjust": auto_adjust, #to adjust for splits and dividends
        "download_at": datetime.now(timezone.utc).isoformat(),
        "yf_version": yf.__version__,
        "data_file": str(data_file)
    }
    json_string = json.dumps(meta_data, indent=2)
    (RAW_DATA_DIR / "meta_data.json").write_text(json_string, encoding="utf-8")

    return raw_data, meta_data