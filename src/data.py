"""
Data fetching module with local CSV caching.
Priority: Cache → Yahoo Finance
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import datetime
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _get_cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.upper()}.csv"


def _load_from_cache(ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    cache_path = _get_cache_path(ticker)
    if not cache_path.exists():
        return None

    df = pd.read_csv(cache_path)
    date_col = 'Date' if 'Date' in df.columns else 'date'
    if date_col not in df.columns:
        raise ValueError(f"Cache file for {ticker} is missing a Date column")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    df.index.name = 'Date'

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    df = df[(df.index >= start_dt) & (df.index <= end_dt)]

    # Cache must cover both ends of the requested range (7-day tolerance for weekends/holidays)
    if len(df) == 0:
        return None
    if df.index.min() > start_dt + pd.Timedelta(days=10):
        return None
    if df.index.max() < end_dt - pd.Timedelta(days=7):
        return None

    logger.info(f"Loaded {ticker} from cache: {len(df)} rows")
    return df


def _save_to_cache(ticker: str, df: pd.DataFrame) -> None:
    cache_path = _get_cache_path(ticker)
    df_to_save = df.reset_index()
    if 'Date' not in df_to_save.columns and 'date' in df_to_save.columns:
        df_to_save = df_to_save.rename(columns={'date': 'Date'})
    df_to_save.to_csv(cache_path, index=False)
    logger.info(f"Cached {ticker} to {cache_path}")


def _download_from_yfinance(ticker: str, start_date: str, end_date: str, max_retries: int = 3) -> Optional[pd.DataFrame]:
    for attempt in range(max_retries):
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)

            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                logger.info(f"Downloaded {ticker} from Yahoo Finance: {len(df)} rows")
                return df
        except Exception as e:
            error_msg = str(e)
            if 'rate limit' in error_msg.lower() or 'YFRateLimitError' in error_msg:
                wait_time = (attempt + 1) * 30
                logger.warning(f"Yahoo Finance rate limited for {ticker}, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                logger.warning(f"Yahoo Finance failed for {ticker}: {e}")
                break

    return None


def download_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    use_cache: bool = True,
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Download historical stock data.
    Priority: local cache → Yahoo Finance.
    Raises ValueError if any ticker fails to download.
    """
    data_frames = []

    for ticker in tickers:
        df = None

        if use_cache and not force_refresh:
            df = _load_from_cache(ticker, start_date, end_date)

        if df is None:
            df = _download_from_yfinance(ticker, start_date, end_date)
            if df is None:
                raise ValueError(f"Failed to download data for {ticker} from Yahoo Finance")
            _save_to_cache(ticker, df)

        df = df.copy()
        df['ticker'] = ticker
        df = df.reset_index()
        df.set_index(['ticker', 'Date'], inplace=True)
        data_frames.append(df)

    if not data_frames:
        raise ValueError("No data downloaded for any ticker")

    return pd.concat(data_frames)



def clear_cache(ticker: Optional[str] = None) -> None:
    if ticker:
        cache_path = _get_cache_path(ticker)
        if cache_path.exists():
            cache_path.unlink()
            logger.info(f"Cleared cache for {ticker}")
    else:
        for cache_file in CACHE_DIR.glob("*.csv"):
            cache_file.unlink()
        logger.info("Cleared all cache")
