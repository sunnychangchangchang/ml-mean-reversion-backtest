"""
Backtesting module.
Implements portfolio-level backtesting with proper signal generation.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)


def generate_signals(
    df: pd.DataFrame,
    return_threshold: float = -0.05,
    ml_prob_threshold: float = 0.55,
) -> pd.DataFrame:
    """
    Generate ML-filtered trading signals.

    Signal generation (NO LOOK-AHEAD BIAS):
    - At day t close, we only have access to data up to day t
    - Signal is generated based on return_5d = Close[t] / Close[t-5] - 1
    - Entry happens at Open[t+1], exit at Close[t+1]

    Rule: return_5d < threshold AND ml_probability > ml_prob_threshold
    """
    df = df.copy()
    df['raw_signal'] = (df['return_5d'] < return_threshold).astype(int)
    df['signal'] = (
        (df['raw_signal'] == 1) & (df['ml_probability'] > ml_prob_threshold)
    ).astype(int)
    return df


def calculate_intraday_return(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate intraday return: Close[t+1] / Open[t+1] - 1
    
    This is the return from entering at Open[t+1] and exiting at Close[t+1].
    
    Args:
        df: DataFrame with price data
    
    Returns:
        DataFrame with intraday_return column
    """
    df = df.copy()
    
    # Get next day's open and close
    df['next_open'] = df.groupby('ticker')['Open'].shift(-1)
    df['next_close'] = df.groupby('ticker')['Close'].shift(-1)
    
    # Calculate intraday return
    df['intraday_return'] = df['next_close'] / df['next_open'] - 1
    
    # Drop helper columns
    df = df.drop(columns=['next_open', 'next_close'])
    
    return df


def run_backtest(
    df: pd.DataFrame,
    initial_capital: float = 100000,
    max_positions: int = 3,
    max_weight: float = 0.5,
    transaction_cost: float = 0.001,
    rank_col: str = 'ml_probability',
    risk_free_rate: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Run portfolio-level backtest.

    Portfolio rules:
    - Long only, no leverage
    - Max positions: max_positions
    - Max weight per position: max_weight
    - If more signals than max_positions, pick top by rank_col

    Args:
        df: DataFrame with signals and intraday returns
        initial_capital: Initial capital amount
        max_positions: Maximum number of concurrent positions
        max_weight: Maximum weight per position (e.g., 0.5 for 50%)
        transaction_cost: Round-trip transaction cost (e.g., 0.001 for 0.1%)
        rank_col: Column used to rank and select positions when signals exceed max_positions

    Returns:
        Tuple of (equity_curve, trade_log, metrics)
    """
    # All trading dates in the backtest period (before any signal filtering)
    if isinstance(df.index, pd.MultiIndex):
        all_dates = sorted(df.index.get_level_values('Date').unique())
    else:
        all_dates = sorted(df['Date'].unique() if 'Date' in df.columns else df.index.unique())

    # Calculate intraday returns if not already done
    if 'intraday_return' not in df.columns:
        df = calculate_intraday_return(df)

    # Filter to only rows with signals
    signals = df[df['signal'] == 1].copy()

    if len(signals) == 0:
        logger.warning("No trading signals generated!")
        return pd.DataFrame(), pd.DataFrame(), calculate_metrics(
            pd.DataFrame(), pd.DataFrame(), initial_capital,
            trade_days=0, risk_free_rate=risk_free_rate,
        )

    # Sort by date then rank_col descending (highest score first)
    signals = signals.sort_values(['Date', rank_col], ascending=[True, False])
    
    # Initialize portfolio
    capital = initial_capital
    equity_curve = []
    trade_log = []
    
    # Group by date
    for date, group in signals.groupby('Date'):
        # Get available signals for this date
        day_signals = group.copy()
        
        # Handle both single and multi-ticker data
        # Reset index to make 'ticker' accessible as a column
        if 'ticker' not in day_signals.columns:
            day_signals = day_signals.reset_index()
        
        # Select top signals by rank_col
        if len(day_signals) > max_positions:
            day_signals = day_signals.nlargest(max_positions, rank_col)
        
        # Calculate position sizes
        n_positions = len(day_signals)
        if n_positions == 0:
            continue

        # Equal weight among positions, capped at max_weight
        weight = min(max_weight, 1.0 / n_positions)

        # Snapshot capital at start of day so all positions are sized consistently.
        # Without this, the second trade's size would depend on the first trade's P&L,
        # producing unequal position sizes and intra-day compounding.
        start_of_day_capital = capital

        # Execute trades
        for idx, row in day_signals.iterrows():
            ticker = row['ticker']
            intraday_ret = row['intraday_return']

            if pd.isna(intraday_ret):
                continue

            # Position sized off start-of-day capital, not running capital
            position_value = start_of_day_capital * weight
            
            # Apply transaction costs (entry + exit = round-trip)
            net_return = intraday_ret - 2 * transaction_cost
            
            # Calculate P&L
            pnl = position_value * net_return
            
            # Update capital
            capital += pnl
            
            # Record trade
            trade_log.append({
                'date': date,
                'ticker': ticker,
                'weight': weight,
                'intraday_return': intraday_ret,
                'transaction_cost': transaction_cost,
                'net_return': net_return,
                'pnl': pnl,
                'capital_before': capital - pnl,
                'capital_after': capital
            })
        
        # Record equity
        equity_curve.append({
            'date': date,
            'capital': capital
        })
    
    # Trade-day-only equity curve (used for charts)
    equity_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(trade_log)

    # Complete daily equity curve: fill flat days with last known capital.
    # This gives accurate annualized return, volatility, and Sharpe ratio.
    equity_by_date = dict(zip(equity_df['date'], equity_df['capital']))
    complete = []
    current_capital = initial_capital
    for d in all_dates:
        if d in equity_by_date:
            current_capital = equity_by_date[d]
        complete.append({'date': d, 'capital': current_capital})
    complete_equity_df = pd.DataFrame(complete)

    metrics = calculate_metrics(
        complete_equity_df,
        trades_df,
        initial_capital,
        trade_days=len(equity_df),
        risk_free_rate=risk_free_rate,
    )

    return equity_df, trades_df, metrics


def run_backtest_raw(
    df: pd.DataFrame,
    initial_capital: float = 100000,
    max_positions: int = 3,
    max_weight: float = 0.5,
    transaction_cost: float = 0.001,
    return_threshold: float = -0.05,
    risk_free_rate: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Run backtest for raw strategy (no ML filter). Ranks signals by absolute 5-day drop."""
    # Generate raw signal; rank by magnitude of the 5-day drop (no ML used)
    df = df.copy()
    df['raw_signal'] = (df['return_5d'] < return_threshold).astype(int)
    df['signal'] = df['raw_signal']
    df['_raw_rank'] = df['return_5d'].abs()

    return run_backtest(
        df,
        initial_capital=initial_capital,
        max_positions=max_positions,
        max_weight=max_weight,
        transaction_cost=transaction_cost,
        rank_col='_raw_rank',
        risk_free_rate=risk_free_rate,
    )


def run_backtest_ml(
    df: pd.DataFrame,
    initial_capital: float = 100000,
    max_positions: int = 3,
    max_weight: float = 0.5,
    transaction_cost: float = 0.001,
    return_threshold: float = -0.05,
    ml_prob_threshold: float = 0.55,
    risk_free_rate: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Run backtest for ML-filtered strategy. Ranks signals by ml_probability."""
    # Generate signals with ML filter
    df = generate_signals(
        df,
        return_threshold=return_threshold,
        ml_prob_threshold=ml_prob_threshold,
    )
    
    return run_backtest(
        df,
        initial_capital=initial_capital,
        max_positions=max_positions,
        max_weight=max_weight,
        transaction_cost=transaction_cost,
        rank_col='ml_probability',
        risk_free_rate=risk_free_rate,
    )


def calculate_metrics(
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    initial_capital: float,
    trade_days: int = None,
    risk_free_rate: float = 0.0,
) -> Dict:
    """Calculate performance metrics from a complete equity curve and trade log."""
    if len(equity_df) == 0 or len(trades_df) == 0:
        return {
            'cumulative_return': 0.0,
            'annualized_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'active_sharpe': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'trade_count': 0,
            'avg_trade_return': 0.0,
            'exposure_ratio': 0.0,
            'final_capital': initial_capital,
        }
    
    final_capital = equity_df['capital'].iloc[-1]
    
    # Cumulative return
    cumulative_return = (final_capital / initial_capital) - 1
    
    # Calculate daily returns from equity curve
    equity_df = equity_df.copy()
    equity_df['daily_return'] = equity_df['capital'].pct_change()
    
    # Annualized return (assuming ~252 trading days)
    n_days = len(equity_df)
    years = n_days / 252
    if years > 0:
        annualized_return = (final_capital / initial_capital) ** (1 / years) - 1
    else:
        annualized_return = 0.0
    
    # Volatility (annualized)
    daily_vol = equity_df['daily_return'].std()
    volatility = daily_vol * np.sqrt(252)
    
    # Sharpe ratio (all-period, includes flat days at 0 return)
    if volatility > 0:
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility
    else:
        sharpe_ratio = 0.0

    # Active-period Sharpe: only on days the strategy held a position.
    # All-period Sharpe is depressed by flat days (0 return lowers measured vol),
    # making low-exposure strategies look artificially good vs always-invested benchmarks.
    # Active-period Sharpe isolates the strategy's edge when it is actually deployed.
    #
    # Annualize by actual active-days-per-year, not 252. Using 252 would assume the
    # strategy is always deployed and inflate Sharpe by sqrt(1 / exposure_ratio).
    active_returns = equity_df.loc[equity_df['daily_return'] != 0, 'daily_return'].dropna()
    if len(active_returns) > 1:
        n_active = len(active_returns)
        n_total = len(equity_df)
        active_per_year = n_active * 252 / n_total  # avg active trading days per year
        active_ann_return = (1 + active_returns.mean()) ** active_per_year - 1
        active_vol = active_returns.std() * np.sqrt(active_per_year)
        active_sharpe = (active_ann_return - risk_free_rate) / active_vol if active_vol > 0 else 0.0
    else:
        active_sharpe = 0.0
    
    # Max drawdown
    equity_df['peak'] = equity_df['capital'].cummax()
    equity_df['drawdown'] = (equity_df['capital'] - equity_df['peak']) / equity_df['peak']
    max_drawdown = equity_df['drawdown'].min()
    
    # Win rate
    winning_trades = (trades_df['pnl'] > 0).sum()
    total_trades = len(trades_df)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
    
    # Trade count
    trade_count = total_trades
    
    # Average trade return
    avg_trade_return = trades_df['net_return'].mean()
    
    # Exposure ratio (days with active positions / total trading days)
    exposure_ratio = (trade_days / n_days) if (trade_days is not None and n_days > 0) else 1.0
    
    return {
        'cumulative_return': cumulative_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'active_sharpe': active_sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'trade_count': trade_count,
        'avg_trade_return': avg_trade_return,
        'exposure_ratio': exposure_ratio,
        'final_capital': final_capital,
    }
