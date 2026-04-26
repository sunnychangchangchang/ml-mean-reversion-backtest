"""
Feature engineering module.
Creates technical indicators and features for the ML model.
"""

import pandas as pd
import numpy as np
from typing import List


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features for all tickers in the dataset.

    Computed columns (all backward-looking, no look-ahead bias):
    - return_1d: 1-day return (ML feature)
    - return_5d: 5-day return (signal trigger only — NOT an ML feature)
    - return_10d: 10-day return (ML feature)
    - volatility_5d: 5-day rolling volatility of daily returns (ML feature)
    - volatility_10d: 10-day rolling volatility of daily returns (ML feature)
    - rsi_14: 14-day RSI using Wilder's smoothing (ML feature)
    - distance_ma20: (Close - MA20) / MA20 (ML feature)
    - distance_ma200: (Close - MA200) / MA200 (ML feature)
    - gap_open: (Open[t] - Close[t-1]) / Close[t-1] — overnight gap (ML feature)

    Args:
        df: DataFrame with OHLCV data, indexed by (ticker, Date)

    Returns:
        DataFrame with added feature columns
    """
    df = df.copy()

    # Calculate returns
    df['return_1d'] = df.groupby('ticker')['Close'].pct_change(1)
    df['return_5d'] = df.groupby('ticker')['Close'].pct_change(5)
    df['return_10d'] = df.groupby('ticker')['Close'].pct_change(10)

    # Calculate volatility (rolling std of returns)
    df['volatility_5d'] = df.groupby('ticker')['return_1d'].transform(
        lambda x: x.rolling(window=5, min_periods=5).std()
    )
    df['volatility_10d'] = df.groupby('ticker')['return_1d'].transform(
        lambda x: x.rolling(window=10, min_periods=10).std()
    )

    # Overnight gap: (Open[t] - Close[t-1]) / Close[t-1]
    # Known at the start of day t; relevant because the trade executes Open→Close intraday.
    prev_close = df.groupby('ticker')['Close'].shift(1)
    df['gap_open'] = (df['Open'] - prev_close) / prev_close

    # Calculate RSI (14-day)
    df['rsi_14'] = df.groupby('ticker')['Close'].transform(
        lambda x: calculate_rsi(x, window=14)
    )

    # Moving averages: MA20, MA200
    ma20 = df.groupby('ticker')['Close'].transform(
        lambda x: x.rolling(window=20, min_periods=20).mean()
    )
    ma200 = df.groupby('ticker')['Close'].transform(
        lambda x: x.rolling(window=200, min_periods=200).mean()
    )
    df['distance_ma20'] = (df['Close'] - ma20) / ma20
    df['distance_ma200'] = (df['Close'] - ma200) / ma200

    return df


def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI using Wilder's EMA smoothing (com = window - 1)."""
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)

    # Wilder's smoothing: EMA with alpha = 1/window (com = window - 1).
    # Standard SMA gives a different result from industry RSI — this is the correct formula.
    avg_gain = gains.ewm(com=window - 1, adjust=False, min_periods=window).mean()
    avg_loss = losses.ewm(com=window - 1, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss          # avg_loss == 0 → rs = inf → rsi = 100 (correct)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add target column: 1 if Close[t+1] > Open[t+1], else 0. Final row per ticker stays NaN."""
    df = df.copy()
    
    # Calculate next day's open and close
    df['next_open'] = df.groupby('ticker')['Open'].shift(-1)
    df['next_close'] = df.groupby('ticker')['Close'].shift(-1)
    
    # Create target only where the next day's open and close are known.
    # The final row for each ticker has no observable t+1 outcome and must stay NaN.
    has_next_bar = df['next_open'].notna() & df['next_close'].notna()
    df['target'] = np.nan
    df.loc[has_next_bar, 'target'] = (
        df.loc[has_next_bar, 'next_close'] > df.loc[has_next_bar, 'next_open']
    ).astype(int)
    
    # Drop the helper columns
    df = df.drop(columns=['next_open', 'next_close'])
    
    return df


def get_feature_columns() -> List[str]:
    """Return the ML feature column names (excludes return_5d, which is the signal trigger)."""
    # return_5d is intentionally excluded: it is the signal trigger itself.
    # Keeping it as a feature would let the ML model learn the signal condition
    # rather than independently assessing market context, making the two
    # strategies non-independent and overstating ML's added value.
    return [
        'return_1d',
        'return_10d',
        'volatility_5d',
        'volatility_10d',
        'rsi_14',
        'distance_ma20',
        'distance_ma200',
        'gap_open',
    ]
