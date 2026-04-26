"""
Walk-forward validation module.
Retrains model at each fold boundary; capital carries over between folds.
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)

from src.model import train_model, predict_proba
from src.backtest import run_backtest_raw, run_backtest_ml, calculate_metrics


def generate_folds(
    df: pd.DataFrame,
    min_train_years: int = 3,
    test_window_months: int = 12,
) -> List[Dict]:
    """
    Generate expanding-window walk-forward folds.
    train: start → fold boundary (grows each fold)
    test:  fold boundary → boundary + test_window (no overlap)
    """
    if isinstance(df.index, pd.MultiIndex):
        all_dates = sorted(df.index.get_level_values('Date').unique())
    else:
        all_dates = sorted(df.index.unique())

    start_dt = pd.Timestamp(all_dates[0])
    end_dt = pd.Timestamp(all_dates[-1])
    first_test_start = start_dt + pd.DateOffset(years=min_train_years)

    folds, test_start = [], first_test_start
    while test_start < end_dt:
        test_end = min(test_start + pd.DateOffset(months=test_window_months), end_dt)
        folds.append({'train_end': test_start, 'test_start': test_start, 'test_end': test_end})
        test_start = test_end
    return folds


def run_walk_forward(
    df: pd.DataFrame,
    feature_cols: List[str],
    initial_capital: float = 100000,
    max_positions: int = 3,
    max_weight: float = 0.5,
    transaction_cost: float = 0.001,
    return_threshold: float = -0.05,
    ml_prob_threshold: Optional[float] = 0.55,
    model_type: str = 'lr',
    min_train_years: int = 3,
    test_window_months: int = 12,
    risk_free_rate: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, List[Dict]]:
    """
    Walk-forward validation with capital continuity between folds.

    At each fold:
      1. Retrain model on all history up to the fold boundary (expanding window)
      2. Predict only on the test window — strictly out-of-sample
      3. Carry end capital forward to the next fold

    ml_prob_threshold=None  →  raw strategy (skips model training entirely)

    Returns
    -------
    equity_df      : trade-day equity curve stitched across all folds (for charts)
    trades_df      : all trades
    overall_metrics: metrics on the complete stitched out-of-sample curve
    fold_results   : list of per-fold summary dicts (for the consistency table)
    """
    folds = generate_folds(df, min_train_years, test_window_months)
    if not folds:
        raise ValueError("Not enough data for walk-forward with the given parameters")

    dates_level = (
        df.index.get_level_values('Date') if isinstance(df.index, pd.MultiIndex) else df.index
    )

    all_equity: List[pd.DataFrame] = []
    all_trades: List[pd.DataFrame] = []
    fold_results: List[Dict] = []
    current_capital = initial_capital

    for i, fold in enumerate(folds):
        # Intermediate folds use [start, end) so the boundary date falls in the next fold.
        # The last fold uses [start, end] to include the final trading day of the dataset.
        is_last = (i == len(folds) - 1)
        if is_last:
            mask = (dates_level >= fold['test_start']) & (dates_level <= fold['test_end'])
        else:
            mask = (dates_level >= fold['test_start']) & (dates_level < fold['test_end'])
        df_test = df[mask]
        if len(df_test) == 0:
            continue

        kwargs = dict(
            initial_capital=current_capital,
            max_positions=max_positions,
            max_weight=max_weight,
            transaction_cost=transaction_cost,
            return_threshold=return_threshold,
            risk_free_rate=risk_free_rate,
        )

        if ml_prob_threshold is not None:
            # Train on all history up to fold boundary, then predict on test window only
            train_end_str = fold['train_end'].strftime('%Y-%m-%d')
            try:
                fold_model, fold_scaler, _ = train_model(df, feature_cols, train_end_str, model_type)
                df_test_preds = predict_proba(fold_model, fold_scaler, df_test, feature_cols)
            except ValueError as e:
                logger.warning(f"Walk-forward fold {i + 1}: skipping — {e}")
                fold_results.append(_empty_fold(i + 1, fold))
                continue
            equity, trades, fold_metrics = run_backtest_ml(
                df_test_preds, **kwargs, ml_prob_threshold=ml_prob_threshold
            )
        else:
            # Raw strategy: no model needed
            equity, trades, fold_metrics = run_backtest_raw(df_test, **kwargs)

        if len(equity) > 0:
            current_capital = equity['capital'].iloc[-1]
            all_equity.append(equity)
        if len(trades) > 0:
            all_trades.append(trades)

        fold_results.append({
            'Fold': i + 1,
            'Test Period': (
                f"{fold['test_start'].strftime('%Y-%m')} → "
                f"{fold['test_end'].strftime('%Y-%m')}"
            ),
            'Return': fold_metrics.get('cumulative_return', 0.0),
            'Sharpe': fold_metrics.get('sharpe_ratio', 0.0),
            'Max DD': fold_metrics.get('max_drawdown', 0.0),
            'Win Rate': fold_metrics.get('win_rate', 0.0),
            'Trades': int(fold_metrics.get('trade_count', 0)),
        })

    if not all_equity:
        return pd.DataFrame(), pd.DataFrame(), {}, fold_results

    combined_equity = pd.concat(all_equity, ignore_index=True)
    combined_trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    # Rebuild complete daily equity curve (ffill flat days) for accurate overall metrics
    first_test_start = folds[0]['test_start']
    all_test_dates = sorted({d for d in dates_level if pd.Timestamp(d) >= first_test_start})

    equity_map = dict(zip(combined_equity['date'], combined_equity['capital']))
    complete, cap = [], initial_capital
    for d in all_test_dates:
        if d in equity_map:
            cap = equity_map[d]
        complete.append({'date': d, 'capital': cap})
    complete_equity_df = pd.DataFrame(complete)

    overall_metrics = calculate_metrics(
        complete_equity_df,
        combined_trades,
        initial_capital,
        transaction_cost,
        trade_days=len(combined_equity),
        risk_free_rate=risk_free_rate,
    )

    return combined_equity, combined_trades, overall_metrics, fold_results


def _empty_fold(i: int, fold: Dict) -> Dict:
    return {
        'Fold': i,
        'Test Period': (
            f"{fold['test_start'].strftime('%Y-%m')} → "
            f"{fold['test_end'].strftime('%Y-%m')}"
        ),
        'Return': 0.0, 'Sharpe': 0.0, 'Max DD': 0.0, 'Win Rate': 0.0, 'Trades': 0,
    }
