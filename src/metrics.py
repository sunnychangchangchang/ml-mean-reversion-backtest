"""
Metrics module.
Provides functions to calculate and display performance metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


def compare_strategies(
    raw_metrics: Dict,
    lr_metrics: Dict,
    xgb_metrics: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Compare Raw, LR, and optionally XGBoost strategy metrics.
    """
    metric_names = {
        'cumulative_return': 'Cumulative Return',
        'annualized_return': 'Annualized Return',
        'volatility': 'Volatility (ann.)',
        'sharpe_ratio': 'Sharpe Ratio (all-period)',
        'active_sharpe': 'Sharpe Ratio (active-period)',
        'max_drawdown': 'Max Drawdown',
        'win_rate': 'Win Rate',
        'trade_count': 'Trade Count',
        'avg_trade_return': 'Avg Trade Return',
        'exposure_ratio': 'Exposure Ratio',
    }

    pct_keys = {'cumulative_return', 'annualized_return', 'max_drawdown',
                'win_rate', 'avg_trade_return', 'exposure_ratio', 'volatility'}
    float_keys = {'sharpe_ratio', 'active_sharpe'}

    def _fmt(val, key):
        if key in pct_keys:
            return f"{val:.2%}"
        if key in float_keys:
            return f"{val:.2f}"
        if key == 'trade_count':
            return f"{int(val)}"
        return str(val)

    rows = []
    for key, label in metric_names.items():
        row = {
            'Metric': label,
            'Raw': _fmt(raw_metrics.get(key, 0), key),
            'LR (L1)': _fmt(lr_metrics.get(key, 0), key),
        }
        if xgb_metrics is not None:
            row['XGBoost'] = _fmt(xgb_metrics.get(key, 0), key)
        rows.append(row)

    return pd.DataFrame(rows)



def get_complete_equity(
    equity_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    initial_capital: float,
) -> pd.DataFrame:
    """
    Reconstruct a complete daily equity curve by forward-filling capital
    over every trading date present in reference_df.
    Needed for accurate yearly returns and other calendar-based metrics.
    """
    if isinstance(reference_df.index, pd.MultiIndex):
        all_dates = sorted(reference_df.index.get_level_values('Date').unique())
    else:
        all_dates = sorted(reference_df.index.unique())

    equity_map = dict(zip(equity_df['date'], equity_df['capital'])) if len(equity_df) > 0 else {}

    complete, cap = [], initial_capital
    for d in all_dates:
        if d in equity_map:
            cap = equity_map[d]
        complete.append({'date': d, 'capital': cap})
    return pd.DataFrame(complete)


def get_yearly_returns(equity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute calendar-year returns from a complete daily equity curve.
    equity_df must have 'date' and 'capital' columns with every trading day present.
    """
    if len(equity_df) == 0:
        return pd.DataFrame(columns=['Year', 'Return', 'TradingDays', 'Partial'])

    df = equity_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df['year'] = df['date'].dt.year

    year_end_cap = df.groupby('year')['capital'].last()
    year_start_cap = year_end_cap.shift(1)
    year_start_cap.iloc[0] = df['capital'].iloc[0]

    year_days = df.groupby('year').size()

    result = (year_end_cap / year_start_cap - 1).reset_index()
    result.columns = ['Year', 'Return']
    result['TradingDays'] = year_days.reindex(result['Year']).values
    # A full calendar year has ~252 trading days; flag years with fewer as partial
    result['Partial'] = result['TradingDays'] < 200
    return result


def compute_spy_equity(
    spy_close: pd.Series,
    initial_capital: float,
    start_date: str,
) -> pd.DataFrame:
    """
    Normalise SPY close prices to an equity curve starting at initial_capital
    from start_date onwards. spy_close must be indexed by Date.
    """
    start_ts = pd.Timestamp(start_date)
    s = spy_close[spy_close.index >= start_ts].sort_index()
    if len(s) == 0:
        return pd.DataFrame(columns=['date', 'capital'])
    capital = initial_capital * (1 + s.pct_change().fillna(0)).cumprod()
    return pd.DataFrame({'date': capital.index, 'capital': capital.values})