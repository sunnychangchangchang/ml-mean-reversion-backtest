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
    - volatility_5d: 5-day rolling volatility of daily returns (ML feature)
    - volatility_10d: 10-day rolling volatility of daily returns (ML feature)
    - volume_zscore: 20-day z-score of volume (ML feature)
    - rsi_14: 14-day RSI using Wilder's smoothing (ML feature)
    - distance_ma20: (Close - MA20) / MA20 (ML feature)
    
    Args:
        df: DataFrame with OHLCV data, indexed by (ticker, Date)
    
    Returns:
        DataFrame with added feature columns
    """
    # Group by ticker to calculate features
    df = df.copy()
    
    # Calculate returns
    df['return_1d'] = df.groupby('ticker')['Close'].pct_change(1)
    df['return_5d'] = df.groupby('ticker')['Close'].pct_change(5)
    
    # Calculate volatility (rolling std of returns)
    df['volatility_5d'] = df.groupby('ticker')['return_1d'].transform(
        lambda x: x.rolling(window=5, min_periods=5).std()
    )
    df['volatility_10d'] = df.groupby('ticker')['return_1d'].transform(
        lambda x: x.rolling(window=10, min_periods=10).std()
    )
    
    # Calculate volume z-score
    df['volume_zscore'] = df.groupby('ticker')['Volume'].transform(
        lambda x: (x - x.rolling(window=20, min_periods=20).mean()) /
                  x.rolling(window=20, min_periods=20).std()
    )
    # Replace inf (std=0 edge case) with NaN so downstream dropna handles it cleanly
    df['volume_zscore'] = df['volume_zscore'].replace([np.inf, -np.inf], np.nan)
    
    # Calculate RSI (14-day)
    df['rsi_14'] = df.groupby('ticker')['Close'].transform(
        lambda x: calculate_rsi(x, window=14)
    )
    
    # Calculate distance from MA20 (intermediate ma20 dropped immediately after use)
    ma20 = df.groupby('ticker')['Close'].transform(
        lambda x: x.rolling(window=20, min_periods=20).mean()
    )
    df['distance_ma20'] = (df['Close'] - ma20) / ma20

    return df


def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index.
    
    Args:
        series: Price series
        window: RSI window period
    
    Returns:
        RSI values
    """
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
    """
    Create target variable for ML model.
    
    Target: y = 1 if Close[t+1] > Open[t+1], else 0
    
    This represents whether the next day's close is higher than its open,
    i.e., whether the stock had a positive intraday return.
    
    Args:
        df: DataFrame with price data
    
    Returns:
        DataFrame with added target column
    """
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
    """
    Return list of feature column names.
    
    Returns:
        List of feature column names
    """
    # return_5d is intentionally excluded: it is the signal trigger itself.
    # Keeping it as a feature would let the ML model learn the signal condition
    # rather than independently assessing market context, making the two
    # strategies non-independent and overstating ML's added value.
    return [
        'return_1d',
        'volatility_5d',
        'volatility_10d',
        'volume_zscore',
        'rsi_14',
        'distance_ma20',
    ]
