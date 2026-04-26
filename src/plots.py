"""
Visualization module.
Creates charts for equity curves, drawdowns, and trade distributions.
"""

import pandas as pd
import plotly.graph_objects as go
import numpy as np


def plot_equity_curves(
    raw_equity: pd.DataFrame,
    lr_equity: pd.DataFrame,
    initial_capital: float = 100000,
    spy_equity: pd.DataFrame = None,
    xgb_equity: pd.DataFrame = None,
) -> go.Figure:
    """Plot equity curves for Raw, LR, optional XGBoost, and SPY benchmark."""
    fig = go.Figure()

    if len(raw_equity) > 0:
        fig.add_trace(go.Scatter(
            x=raw_equity['date'], y=raw_equity['capital'],
            mode='lines', name='Raw Strategy',
            line=dict(color='steelblue', width=2)
        ))

    if lr_equity is not None and len(lr_equity) > 0:
        fig.add_trace(go.Scatter(
            x=lr_equity['date'], y=lr_equity['capital'],
            mode='lines', name='LR (L1)',
            line=dict(color='seagreen', width=2)
        ))

    if xgb_equity is not None and len(xgb_equity) > 0:
        fig.add_trace(go.Scatter(
            x=xgb_equity['date'], y=xgb_equity['capital'],
            mode='lines', name='XGBoost',
            line=dict(color='mediumpurple', width=2)
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
    lr_yearly: pd.DataFrame,
    spy_yearly: pd.DataFrame = None,
    xgb_yearly: pd.DataFrame = None,
) -> go.Figure:
    """Grouped bar chart of calendar-year returns for all strategies."""
    fig = go.Figure()

    all_dfs = [raw_yearly, lr_yearly, xgb_yearly, spy_yearly]
    years = set()
    for df in all_dfs:
        if df is not None and len(df) > 0:
            years.update(df['Year'].astype(str).tolist())
    years = sorted(years)

    partial_years = set()
    for df in [lr_yearly, raw_yearly]:
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
    _add_bars(lr_yearly,  'LR (L1)',      'seagreen')
    _add_bars(xgb_yearly, 'XGBoost',      'mediumpurple')
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
    """Plot drawdown (%) over time from the equity curve."""
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
    """Plot histogram of per-trade P&L in dollars."""
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
    """Plot a reliability diagram from check_calibration_oos() output."""
    fig = go.Figure()
    if not cal_data or not cal_data.get('bins'):
        fig.update_layout(title='Calibration - No Data', template='plotly_white')
        return fig

    bins = cal_data['bins']
    x = [r['avg_proba'] for r in bins]
    y = [r['avg_actual'] for r in bins]
    sizes = np.sqrt([max(1, r['count']) for r in bins])

    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers+lines',
        name='OOS calibration',
        marker=dict(size=sizes, color='steelblue', opacity=0.8),
        line=dict(color='steelblue', width=2),
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Perfect calibration',
        line=dict(color='gray', dash='dash'),
    ))
    fig.update_layout(
        title=f"Calibration (OOS) — ECE={cal_data.get('ece', float('nan')):.4f}, "
              f"Brier={cal_data.get('brier_score', float('nan')):.4f}",
        xaxis_title='Mean predicted probability',
        yaxis_title='Empirical positive rate',
        template='plotly_white',
        hovermode='closest',
    )
    return fig



