"""
Streamlit UI for ML-Enhanced Mean Reversion Strategy.
"""

import streamlit as st
import pandas as pd
from datetime import datetime

from src import data, features, model, metrics, plots, walkforward, i18n
from src.constants import (
    INITIAL_CAPITAL,
    MAX_POSITIONS,
    MAX_WEIGHT,
    ML_PROB_THRESHOLD,
    TRANSACTION_COST,
    TEST_WINDOW_MONTHS,
    MIN_TRAIN_YEARS,
)


st.set_page_config(
    page_title="ML Mean Reversion Strategy",
    page_icon="📈",
    layout="wide"
)


_METRIC_DEFS = [
    ('cumreturn',    'cumulative_return',  ':.2%'),
    ('annreturn',    'annualized_return',  ':.2%'),
    ('vol',          'volatility',         ':.2%'),
    ('sharpe',       'sharpe_ratio',       ':.2f'),
    ('active_sharpe','active_sharpe',      ':.2f'),
    ('maxdd',        'max_drawdown',       ':.2%'),
    ('winrate',      'win_rate',           ':.1%'),
    ('trades',       'trade_count',        ':d'),
    ('avgtrade',     'avg_trade_return',   ':.2%'),
    ('exposure',     'exposure_ratio',     ':.1%'),
]

def _fmt_metric(val, fmt: str) -> str:
    if fmt == ':d':
        return str(int(val))
    return format(float(val), fmt.lstrip(':'))

def _render_metrics_table(T, raw_m, lr_m, xgb_m):
    rows = ""
    for i, (suffix, key, fmt) in enumerate(_METRIC_DEFS):
        label     = T[f'metric_label_{suffix}']
        help_text = T[f'metric_help_{suffix}']
        raw_val   = _fmt_metric(raw_m.get(key, 0), fmt)
        lr_val    = _fmt_metric(lr_m.get(key, 0), fmt)
        xgb_val   = _fmt_metric(xgb_m.get(key, 0), fmt)
        bg = "rgba(255,255,255,0.03)" if i % 2 == 0 else "transparent"
        rows += (
            f'<tr style="background:{bg}">'
            f'<td style="padding:8px 12px;cursor:help" title="{help_text}">'
            f'{label}&nbsp;<span style="opacity:0.45;font-size:11px">(?)</span></td>'
            f'<td style="padding:8px 12px">{raw_val}</td>'
            f'<td style="padding:8px 12px">{lr_val}</td>'
            f'<td style="padding:8px 12px">{xgb_val}</td>'
            f'</tr>'
        )
    st.markdown(
        f'<table style="width:100%;border-collapse:collapse;font-size:14px">'
        f'<thead><tr style="border-bottom:1px solid rgba(250,250,250,0.2)">'
        f'<th style="text-align:left;padding:8px 12px;font-weight:600">Metric</th>'
        f'<th style="text-align:left;padding:8px 12px;font-weight:600">Raw</th>'
        f'<th style="text-align:left;padding:8px 12px;font-weight:600">LR (L1)</th>'
        f'<th style="text-align:left;padding:8px 12px;font-weight:600">XGBoost</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>',
        unsafe_allow_html=True,
    )

def _format_fold_table(fold_results: list) -> pd.DataFrame:
    rows = []
    for r in fold_results:
        rows.append({
            'Fold': r['Fold'],
            'Test Period': r['Test Period'],
            'Return': f"{r['Return']:.1%}",
            'Sharpe': f"{r['Sharpe']:.2f}",
            'Max DD': f"{r['Max DD']:.1%}",
            'Win Rate': f"{r['Win Rate']:.1%}",
            'Trades': r['Trades'],
        })
    return pd.DataFrame(rows)


def main():
    # ── Language toggle (top of sidebar) ──────────────────────────────────────
    if 'lang' not in st.session_state:
        st.session_state.lang = 'en'

    col_en, col_zh = st.sidebar.columns(2)
    if col_en.button('English', use_container_width=True,
                     type='primary' if st.session_state.lang == 'en' else 'secondary'):
        st.session_state.lang = 'en'
        st.rerun()
    if col_zh.button('中文', use_container_width=True,
                     type='primary' if st.session_state.lang == 'zh' else 'secondary'):
        st.session_state.lang = 'zh'
        st.rerun()
    st.sidebar.divider()

    T = i18n.get(st.session_state.lang)

    st.title(T['title'])

    # ── Sidebar ────────────────────────────────────────────────────────────────
    st.sidebar.header(T['sidebar_header'])

    st.sidebar.subheader(T['data_sub'])
    tickers = st.sidebar.multiselect(
        T['tickers_label'],
        ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'GOOGL', 'META', 'NFLX'],
        default=['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN']
    )
    start_date = st.sidebar.date_input(T['start_label'], datetime(2015, 1, 1))
    end_date   = st.sidebar.date_input(T['end_label'],   datetime.today())
    use_cache     = st.sidebar.checkbox(T['cache_label'], value=True)
    force_refresh = st.sidebar.button(T['refresh_btn'])

    st.sidebar.subheader(T['strategy_sub'])
    return_threshold = st.sidebar.slider(
        T['threshold_label'],
        min_value=-20, max_value=-1, value=-5, step=1, format="%d%%"
    ) / 100
    st.sidebar.caption(T['threshold_caption'])

    st.sidebar.subheader(T['benchmark_sub'])
    risk_free_rate = st.sidebar.slider(
        T['rfr_label'],
        min_value=0.0, max_value=10.0, value=4.0, step=0.25, format="%.2f%%"
    ) / 100
    st.sidebar.caption(T['rfr_caption'])

    st.sidebar.divider()
    st.sidebar.markdown(T['fixed_header'])
    st.sidebar.markdown(T['fixed_body'].format(
        ml_thr=ML_PROB_THRESHOLD,
        tc=TRANSACTION_COST,
        tw=TEST_WINDOW_MONTHS,
        mty=MIN_TRAIN_YEARS,
        ic=INITIAL_CAPITAL,
        mp=MAX_POSITIONS,
        mw=MAX_WEIGHT,
    ))

    # ── Main page ──────────────────────────────────────────────────────────────
    st.warning(T['surv_warning'])

    with st.expander(T['how_to_title'], expanded=False):
        st.markdown(T['how_to_body'].format(ml_thr=ML_PROB_THRESHOLD))

    st.markdown(T['strategy_desc'].format(
        thr=return_threshold * 100,
        ml_thr=ML_PROB_THRESHOLD,
    ))

    if not tickers:
        st.error(T['no_tickers_err'])
        return
    if start_date >= end_date:
        st.error(T['date_order_err'])
        return

    with st.spinner(T['spinner_text']):
        try:
            start_str = start_date.strftime('%Y-%m-%d')
            end_str   = end_date.strftime('%Y-%m-%d')

            st.info(T['downloading_info'])
            df_raw  = data.download_data(tickers, start_str, end_str,
                                         use_cache=use_cache, force_refresh=force_refresh)
            spy_raw = data.download_data(['SPY'], start_str, end_str,
                                         use_cache=use_cache, force_refresh=force_refresh)
            spy_close = spy_raw.reset_index(level='ticker', drop=True)['Close']
            st.success(T['downloaded_ok'].format(rows=len(df_raw), n=len(tickers)))

            st.info(T['features_info'])
            df = features.create_features(df_raw)
            df = features.create_target(df)
            feature_cols = features.get_feature_columns()
            st.success(T['features_ok'])

            st.info(T['wf_info'])
            wf_kwargs = dict(
                feature_cols=feature_cols,
                initial_capital=INITIAL_CAPITAL,
                max_positions=MAX_POSITIONS,
                max_weight=MAX_WEIGHT,
                transaction_cost=TRANSACTION_COST,
                return_threshold=return_threshold,
                min_train_years=MIN_TRAIN_YEARS,
                test_window_months=TEST_WINDOW_MONTHS,
                risk_free_rate=risk_free_rate,
            )
            raw_equity, raw_trades, raw_metrics_dict, raw_folds, _ = \
                walkforward.run_walk_forward(df, ml_prob_threshold=None, **wf_kwargs)
            lr_equity, lr_trades, lr_metrics_dict, lr_folds, lr_oos_preds = \
                walkforward.run_walk_forward(df, ml_prob_threshold=ML_PROB_THRESHOLD,
                                             model_type='lr', **wf_kwargs)
            xgb_equity, xgb_trades, xgb_metrics_dict, xgb_folds, xgb_oos_preds = \
                walkforward.run_walk_forward(df, ml_prob_threshold=ML_PROB_THRESHOLD,
                                             model_type='xgb', **wf_kwargs)

            backtest_start = (
                pd.Timestamp(start_date) + pd.DateOffset(years=MIN_TRAIN_YEARS)
            ).strftime('%Y-%m-%d')
            df_backtest_period = df[
                df.index.get_level_values('Date') >= pd.Timestamp(backtest_start)
            ]
            ref_lr_model,  ref_lr_scaler,  ref_train_data = model.train_model(
                df, feature_cols, backtest_start, model_type='lr'
            )
            ref_xgb_model, ref_xgb_scaler, _ = model.train_model(
                df, feature_cols, backtest_start, model_type='xgb'
            )
            st.success(T['wf_ok'])

            spy_equity    = metrics.compute_spy_equity(spy_close, INITIAL_CAPITAL, backtest_start)
            raw_complete  = metrics.get_complete_equity(raw_equity,  df_backtest_period, INITIAL_CAPITAL)
            lr_complete   = metrics.get_complete_equity(lr_equity,   df_backtest_period, INITIAL_CAPITAL)
            xgb_complete  = metrics.get_complete_equity(xgb_equity,  df_backtest_period, INITIAL_CAPITAL)
            raw_yearly    = metrics.get_yearly_returns(raw_complete)
            lr_yearly     = metrics.get_yearly_returns(lr_complete)
            xgb_yearly    = metrics.get_yearly_returns(xgb_complete)
            spy_yearly    = metrics.get_yearly_returns(spy_equity)

            # ── Results ───────────────────────────────────────────────────────
            st.header(T['summary_hdr'])
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric(T['raw_capital_metric'],
                        f"${raw_metrics_dict.get('final_capital', 0):,.0f}")
            col2.metric(T['ml_capital_metric'],
                        f"${lr_metrics_dict.get('final_capital', 0):,.0f}")
            col3.metric('XGBoost Final Capital',
                        f"${xgb_metrics_dict.get('final_capital', 0):,.0f}")
            col4.metric(T['raw_trades_metric'], int(raw_metrics_dict.get('trade_count', 0)))
            col5.metric(T['ml_trades_metric'],  int(lr_metrics_dict.get('trade_count', 0)))
            col6.metric('XGBoost Trades',       int(xgb_metrics_dict.get('trade_count', 0)))

            st.divider()
            st.header(T['folds_hdr'])
            st.caption(T['folds_caption'])
            tab_lr, tab_xgb = st.tabs(['LR (L1)', 'XGBoost'])
            with tab_lr:
                if lr_folds:
                    st.dataframe(_format_fold_table(lr_folds),
                                 use_container_width=True, hide_index=True)
                else:
                    st.info(T['no_folds_info'])
            with tab_xgb:
                if xgb_folds:
                    st.dataframe(_format_fold_table(xgb_folds),
                                 use_container_width=True, hide_index=True)
                else:
                    st.info(T['no_folds_info'])

            st.divider()
            st.header(T['metrics_hdr'])
            st.caption(T['metrics_caption'])
            _render_metrics_table(T, raw_metrics_dict, lr_metrics_dict, xgb_metrics_dict)

            st.divider()
            st.header(T['viz_hdr'])

            st.subheader(T['equity_sub'])
            st.plotly_chart(
                plots.plot_equity_curves(raw_equity, lr_equity, INITIAL_CAPITAL,
                                         spy_equity=spy_equity, xgb_equity=xgb_equity),
                use_container_width=True
            )

            st.subheader(T['yearly_sub'])
            st.plotly_chart(
                plots.plot_yearly_returns(raw_yearly, lr_yearly, spy_yearly, xgb_yearly=xgb_yearly),
                use_container_width=True
            )
            if lr_yearly is not None and lr_yearly.get('Partial', pd.Series()).any():
                st.caption(T['partial_caption'])

            col1, col2 = st.columns(2)
            with col1:
                st.subheader(T['drawdown_sub'])
                tab_dd_lr, tab_dd_xgb = st.tabs(['LR (L1)', 'XGBoost'])
                with tab_dd_lr:
                    st.plotly_chart(plots.plot_drawdowns(lr_equity, "LR (L1)"),
                                    use_container_width=True)
                with tab_dd_xgb:
                    st.plotly_chart(plots.plot_drawdowns(xgb_equity, "XGBoost"),
                                    use_container_width=True)
            with col2:
                st.subheader(T['pnl_sub'])
                tab_pnl_lr, tab_pnl_xgb = st.tabs(['LR (L1)', 'XGBoost'])
                with tab_pnl_lr:
                    st.plotly_chart(plots.plot_trade_distribution(lr_trades, "LR (L1)"),
                                    use_container_width=True)
                with tab_pnl_xgb:
                    st.plotly_chart(plots.plot_trade_distribution(xgb_trades, "XGBoost"),
                                    use_container_width=True)

            st.divider()
            st.header(T['trades_hdr'])
            tab_tr_lr, tab_tr_xgb = st.tabs(['LR (L1)', 'XGBoost'])
            for tab, trades_df in [(tab_tr_lr, lr_trades), (tab_tr_xgb, xgb_trades)]:
                with tab:
                    if len(trades_df) > 0:
                        t = trades_df.copy().iloc[::-1].reset_index(drop=True)
                        t['date']            = pd.to_datetime(t['date']).dt.strftime('%Y-%m-%d')
                        t['intraday_return'] = t['intraday_return'].apply(lambda x: f"{x:.2%}")
                        t['net_return']      = t['net_return'].apply(lambda x: f"{x:.2%}")
                        t['pnl']             = t['pnl'].apply(lambda x: f"${x:.2f}")
                        st.dataframe(
                            t[['date', 'ticker', 'intraday_return', 'net_return', 'pnl']],
                            use_container_width=True, hide_index=True, height=400
                        )
                    else:
                        st.info(T['no_trades_info'])

            st.divider()
            st.header(T['insights_hdr'])

            st.subheader(T['cal_sub'])
            st.markdown(T['cal_desc'].format(ml_thr=ML_PROB_THRESHOLD))
            with st.expander(T['cal_expander'], expanded=False):
                st.markdown(T['cal_body'].format(ml_thr=ML_PROB_THRESHOLD))

            tab_cal_lr, tab_cal_xgb = st.tabs(['LR (L1)', 'XGBoost'])
            for tab, oos_preds, ref_m, ref_s in [
                (tab_cal_lr,  lr_oos_preds,  ref_lr_model,  ref_lr_scaler),
                (tab_cal_xgb, xgb_oos_preds, ref_xgb_model, ref_xgb_scaler),
            ]:
                with tab:
                    if oos_preds is not None:
                        cal_data = model.check_calibration_oos(
                            oos_preds['proba'], oos_preds['actual']
                        )
                    else:
                        cal_data = model.check_calibration(
                            ref_m, ref_s, ref_train_data, feature_cols
                        )
                    if cal_data:
                        col_cal, col_info = st.columns([2, 1])
                        with col_cal:
                            st.plotly_chart(plots.plot_calibration_curve(cal_data),
                                            use_container_width=True)
                        with col_info:
                            ece          = cal_data['ece']
                            brier        = cal_data['brier_score']
                            base         = cal_data['base_rate']
                            brier_random = base * (1 - base)
                            st.metric("ECE",         f"{ece:.4f}",   help=T['cal_ece_help'])
                            st.metric("Brier Score", f"{brier:.4f}",
                                      delta=f"{brier - brier_random:+.4f} vs random ({brier_random:.4f})",
                                      delta_color="inverse",
                                      help=T['cal_brier_help'])
                            st.metric("Base Rate",   f"{base:.1%}",  help=T['cal_base_help'])
                            st.divider()
                            if ece < 0.03:
                                st.success(T['cal_excellent'].format(ece=ece, ml_thr=ML_PROB_THRESHOLD))
                            elif ece < 0.05:
                                st.success(T['cal_good'].format(ece=ece, ml_thr=ML_PROB_THRESHOLD))
                            elif ece < 0.08:
                                st.warning(T['cal_moderate'].format(ece=ece))
                            else:
                                st.error(T['cal_poor'].format(ece=ece))

            st.subheader(T['fi_sub'])
            st.caption(T['fi_caption'].format(date=backtest_start))
            tab_fi_lr, tab_fi_xgb = st.tabs(['LR (L1)', 'XGBoost'])
            with tab_fi_lr:
                st.dataframe(model.get_feature_importance(ref_lr_model, feature_cols),
                             use_container_width=True, hide_index=True)
            with tab_fi_xgb:
                st.dataframe(model.get_feature_importance(ref_xgb_model, feature_cols),
                             use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Error: {e}")
            import traceback
            st.code(traceback.format_exc())

    st.divider()
    st.caption(T['footer'])


if __name__ == "__main__":
    main()
