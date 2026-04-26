"""
Visualization module.
Creates charts for equity curves, drawdowns, and trade distributions.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Tuple


def plot_equity_curves(
    raw_equity: pd.DataFrame,
    ml_equity: pd.DataFrame,
    initial_capital: float = 100000,
    spy_equity: pd.DataFrame = None,
) -> go.Figure:
    """Plot equity curves for both strategies plus optional SPY benchmark."""
    fig = go.Figure()

    if len(raw_equity) > 0:
        fig.add_trace(go.Scatter(
            x=raw_equity['date'], y=raw_equity['capital'],
            mode='lines', name='Raw Strategy',
            line=dict(color='steelblue', width=2)
        ))

    if len(ml_equity) > 0:
        fig.add_trace(go.Scatter(
            x=ml_equity['date'], y=ml_equity['capital'],
            mode='lines', name='ML Strategy',
            line=dict(color='seagreen', width=2)
        ))

    if spy_equity is not None and len(spy_equity) > 0:
        fig.add_trace(go.Scatter(
            x=spy_equity['date'], y=spy_equity['capital'],
            mode='lines', name='SPY Buy & Hold',
            line=dict(color='orange', width=2, dash='dash')
        ))

    fig.add_hline(y=initial_capital, line_dash='dash', line_color='gray',
                  annotation_text='Initial Capital')

    fig.update_layout(
        title='Equity Curves',
        xaxis_title='Date', yaxis_title='Capital ($)',
        hovermode='x unified', template='plotly_white',
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01)
    )
    return fig


def plot_yearly_returns(
    raw_yearly: pd.DataFrame,
    ml_yearly: pd.DataFrame,
    spy_yearly: pd.DataFrame = None,
) -> go.Figure:
    """Grouped bar chart of calendar-year returns for all strategies."""
    fig = go.Figure()

    # Collect all years; use ML returns (most complete) to determine partial years
    years = set()
    for df in [raw_yearly, ml_yearly, spy_yearly]:
        if df is not None and len(df) > 0:
            years.update(df['Year'].astype(str).tolist())
    years = sorted(years)

    # Build a partial-year lookup from whichever df has the Partial column
    partial_years = set()
    for df in [ml_yearly, raw_yearly]:
        if df is not None and 'Partial' in df.columns:
            partial_years = set(df.loc[df['Partial'], 'Year'].astype(str))
            break
    x_labels = [f"{y}*" if y in partial_years else y for y in years]

    def _add_bars(df, name, color):
        if df is None or len(df) == 0:
            return
        year_map = dict(zip(df['Year'].astype(str), df['Return'] * 100))
        fig.add_trace(go.Bar(
            x=x_labels,
            y=[year_map.get(y, None) for y in years],
            name=name,
            marker_color=color,
            opacity=0.8,
        ))

    _add_bars(raw_yearly, 'Raw Strategy', 'steelblue')
    _add_bars(ml_yearly, 'ML Strategy', 'seagreen')
    _add_bars(spy_yearly, 'SPY Buy & Hold', 'orange')

    fig.add_hline(y=0, line_color='black', line_width=1)
    fig.update_layout(
        title='Yearly Returns',
        xaxis_title='Year',
        yaxis_title='Return (%)',
        yaxis=dict(ticksuffix='%'),
        barmode='group',
        template='plotly_white',
    )
    return fig


def plot_drawdowns(
    equity_df: pd.DataFrame,
    strategy_name: str = 'Strategy'
) -> go.Figure:
    """
    Plot drawdown chart.
    
    Args:
        equity_df: DataFrame with equity curve
        strategy_name: Name for the strategy
    
    Returns:
        Plotly figure
    """
    if len(equity_df) == 0:
        fig = go.Figure()
        fig.update_layout(title=f'{strategy_name} - No Data')
        return fig
    
    # Calculate drawdowns
    df = equity_df.copy()
    df['peak'] = df['capital'].cummax()
    df['drawdown'] = (df['capital'] - df['peak']) / df['peak'] * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['drawdown'],
        mode='lines',
        name='Drawdown',
        fill='tozeroy',
        fillcolor='rgba(255, 0, 0, 0.3)',
        line=dict(color='red', width=1)
    ))
    
    fig.update_layout(
        title=f'{strategy_name} - Drawdown',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def plot_trade_distribution(
    trades_df: pd.DataFrame,
    strategy_name: str = 'Strategy'
) -> go.Figure:
    """
    Plot histogram of trade returns.
    
    Args:
        trades_df: DataFrame with trade log
        strategy_name: Name for the strategy
    
    Returns:
        Plotly figure
    """
    if len(trades_df) == 0:
        fig = go.Figure()
        fig.update_layout(title=f'{strategy_name} - No Trades')
        return fig
    
    fig = go.Figure()
    
    # Histogram of P&L
    fig.add_trace(go.Histogram(
        x=trades_df['pnl'],
        nbinsx=30,
        name='P&L',
        marker_color='steelblue',
        opacity=0.7
    ))
    
    # Add vertical line at 0
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    
    fig.update_layout(
        title=f'{strategy_name} - Trade P&L Distribution',
        xaxis_title='P&L ($)',
        yaxis_title='Count',
        template='plotly_white'
    )
    
    return fig


def plot_calibration_curve(cal_data: dict) -> go.Figure:
    """
    Reliability diagram: predicted probability vs actual positive rate.
    A perfectly calibrated model follows the diagonal.
    """
    fig = go.Figure()

    # Perfect calibration reference line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines', name='Perfect Calibration',
        line=dict(color='gray', dash='dash', width=1)
    ))

    # Model calibration curve
    ece = cal_data['ece']
    brier = cal_data['brier_score']
    fig.add_trace(go.Scatter(
        x=cal_data['mean_predicted_value'],
        y=cal_data['fraction_of_positives'],
        mode='lines+markers',
        name=f'Logistic Regression  (ECE={ece:.3f}, Brier={brier:.3f})',
        line=dict(color='steelblue', width=2),
        marker=dict(size=8),
    ))

    # Shade the gap between model and perfect calibration
    fig.add_trace(go.Scatter(
        x=np.concatenate([cal_data['mean_predicted_value'],
                          cal_data['mean_predicted_value'][::-1]]),
        y=np.concatenate([cal_data['fraction_of_positives'],
                          cal_data['mean_predicted_value'][::-1]]),
        fill='toself',
        fillcolor='rgba(70, 130, 180, 0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
    ))

    fig.update_layout(
        title='Calibration Curve (Reliability Diagram)',
        xaxis=dict(title='Mean Predicted Probability', range=[0, 1]),
        yaxis=dict(title='Fraction of Positives', range=[0, 1]),
        template='plotly_white',
        legend=dict(yanchor='top', y=0.15, xanchor='left', x=0.01),
    )
    return fig


