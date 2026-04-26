"""
Visualization module.
Creates charts for equity curves, drawdowns, and trade distributions.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple


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


def plot_feature_winrate(
    df: pd.DataFrame,
    feature_columns: List[str],
    n_bins: int = 8,
) -> go.Figure:
    """
    For each feature, bin into quantiles and plot mean win rate per bin.

    A roughly straight line → linear relationship → LR is sufficient.
    A curve, plateau, or reversal → non-linear → XGBoost may add value.
    """
    valid = df.dropna(subset=feature_columns + ['target']).copy()
    n_features = len(feature_columns)
    n_cols = 3
    n_rows = int(np.ceil(n_features / n_cols))

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=feature_columns,
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
    )

    base_rate = float(valid['target'].mean())

    for i, feat in enumerate(feature_columns):
        row, col = divmod(i, n_cols)
        row += 1
        col += 1

        try:
            valid['_bin'] = pd.qcut(valid[feat], q=n_bins, duplicates='drop')
            grouped     = valid.groupby('_bin', observed=True)['target'].agg(['mean', 'count'])
            bin_centers = [iv.mid for iv in grouped.index]
            win_rates   = grouped['mean'].values
            counts      = grouped['count'].values
        except Exception:
            continue

        colors = ['seagreen' if w >= base_rate else 'tomato' for w in win_rates]

        fig.add_trace(go.Bar(
            x=list(range(len(bin_centers))),
            y=win_rates,
            marker_color=colors,
            customdata=np.stack([bin_centers, counts], axis=1),
            hovertemplate=(
                'Bin centre: %{customdata[0]:.3f}<br>'
                'Win rate: %{y:.1%}<br>'
                'Count: %{customdata[1]}<extra></extra>'
            ),
            showlegend=False,
        ), row=row, col=col)

        fig.add_hline(
            y=base_rate, line_dash='dash', line_color='gray',
            line_width=1, row=row, col=col,
        )

        fig.update_xaxes(showticklabels=False, row=row, col=col)
        fig.update_yaxes(tickformat='.0%', row=row, col=col)

    fig.update_layout(
        title='Feature vs Win Rate (quantile bins) — dashed = base rate',
        height=280 * n_rows,
        template='plotly_white',
        margin=dict(t=60, b=20),
    )
    return fig


