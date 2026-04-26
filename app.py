"""
Streamlit UI for ML-Enhanced Mean Reversion Strategy.
"""

import traceback

import streamlit as st
import pandas as pd
import numpy as np
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


def _format_fold_table(fold_results: list) -> pd.DataFrame:
    rows = []
    for r in fold_results:
        rows.append({
            'Fold': str(r['Fold']),
            'Test Period': r['Test Period'],
            'Return': f"{r['Return']:.1%}",
            'Sharpe': f"{r['Sharpe']:.2f}",
            'Max DD': f"{r['Max DD']:.1%}",
            'Win Rate': f"{r['Win Rate']:.1%}",
            'Trades': str(r['Trades']),
        })
    return pd.DataFrame(rows)


def _style_left(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    return df.style.set_properties(**{'text-align': 'left'}).set_table_styles(
        [{'selector': 'th', 'props': [('text-align', 'left')]}]
    )


def _ml_diag_summary_markdown(T: dict, m: dict, folds: list) -> str:
    """Return a short, high-signal interpretation of OOS ML diagnostics."""
    n = int(m.get('ml_oos_n', 0) or 0)
    if n <= 0:
        return T['ml_diag_summary_no_data']

    auc = float(m.get('ml_oos_auc', np.nan))
    ll = float(m.get('ml_oos_logloss', np.nan))
    brier = float(m.get('ml_oos_brier', np.nan))
    p = float(m.get('ml_oos_base_rate', np.nan))

    # Baseline: constant predictor p (i.e., "always guess the base rate").
    eps = 1e-12
    p0 = float(np.clip(p, eps, 1 - eps))
    ll_base = float(-(p0 * np.log(p0) + (1 - p0) * np.log(1 - p0)))
    brier_base = float(p0 * (1 - p0))  # E[(p-y)^2] when y~Bernoulli(p)

    ll_impr = ll_base - ll
    brier_impr = brier_base - brier

    # Fold stability stats (AUC).
    aucs = []
    ns = []
    for r in (folds or []):
        if 'ML AUC' not in r:
            continue
        v = r.get('ML AUC')
        if v is None or not np.isfinite(v):
            continue
        aucs.append(float(v))
        ns.append(int(r.get('ML N', 0) or 0))
    if len(aucs) >= 2:
        auc_mean = float(np.mean(aucs))
        auc_std = float(np.std(aucs, ddof=1))
        frac_gt_05 = float(np.mean(np.array(aucs) > 0.5))
        auc_best = float(np.max(aucs))
        auc_worst = float(np.min(aucs))
    elif len(aucs) == 1:
        auc_mean = auc_best = auc_worst = float(aucs[0])
        auc_std = 0.0
        frac_gt_05 = 1.0 if auc_mean > 0.5 else 0.0
    else:
        auc_mean = auc_std = frac_gt_05 = auc_best = auc_worst = np.nan

    # AUC interpretation buckets (finance-appropriate).
    if np.isnan(auc):
        auc_band = T['ml_diag_auc_undef']
    elif auc < 0.505:
        auc_band = T['ml_diag_auc_weak']
    elif auc < 0.53:
        auc_band = T['ml_diag_auc_small']
    elif auc < 0.58:
        auc_band = T['ml_diag_auc_meaningful']
    else:
        auc_band = T['ml_diag_auc_suspicious']

    # Improvement interpretation (tiny improvements are normal in finance).
    def _imp_word(x: float) -> str:
        if not np.isfinite(x):
            return T['ml_diag_imp_na']
        if x > 0.002:
            return T['ml_diag_imp_clear']
        if x > 0.0005:
            return T['ml_diag_imp_small']
        if x >= -0.0005:
            return T['ml_diag_imp_none']
        return T['ml_diag_imp_worse']

    ll_word = _imp_word(ll_impr)
    brier_word = _imp_word(brier_impr)
    if np.isfinite(auc) and auc < 0.53 and ll_impr <= 0.0005 and brier_impr <= 0.0005:
        verdict = T['ml_diag_verdict_weak']
    elif np.isfinite(auc) and auc >= 0.53 and (ll_impr > 0.0005 or brier_impr > 0.0005):
        verdict = T['ml_diag_verdict_promising']
    else:
        verdict = T['ml_diag_verdict_mixed']

    return T['ml_diag_summary'].format(
        n=n,
        base_rate=p,
        auc=auc,
        auc_band=auc_band,
        auc_mean=auc_mean,
        auc_std=auc_std,
        auc_best=auc_best,
        auc_worst=auc_worst,
        frac_gt_05=frac_gt_05,
        logloss=ll,
        logloss_base=ll_base,
        logloss_impr=ll_impr,
        logloss_imp_word=ll_word,
        brier=brier,
        brier_base=brier_base,
        brier_impr=brier_impr,
        brier_imp_word=brier_word,
        verdict=verdict,
    )


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
        tc=2 * TRANSACTION_COST,
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
            lr_equity,  lr_trades,  lr_metrics_dict,  lr_folds,  _  = \
                walkforward.run_walk_forward(df, ml_prob_threshold=ML_PROB_THRESHOLD,
                                             model_type='lr', **wf_kwargs)
            xgb_equity, xgb_trades, xgb_metrics_dict, xgb_folds, _ = \
                walkforward.run_walk_forward(df, ml_prob_threshold=ML_PROB_THRESHOLD,
                                             model_type='xgb', **wf_kwargs)

            backtest_start = (
                pd.Timestamp(start_date) + pd.DateOffset(years=MIN_TRAIN_YEARS)
            ).strftime('%Y-%m-%d')
            df_backtest_period = df[
                df.index.get_level_values('Date') >= pd.Timestamp(backtest_start)
            ]
            ref_lr_model,  _ = model.train_model(df, feature_cols, backtest_start, 'lr')
            ref_xgb_model, _ = model.train_model(df, feature_cols, backtest_start, 'xgb')
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
            col3.metric(T['xgb_capital_metric'],
                        f"${xgb_metrics_dict.get('final_capital', 0):,.0f}")
            col4.metric(T['raw_trades_metric'], int(raw_metrics_dict.get('trade_count', 0)))
            col5.metric(T['ml_trades_metric'],  int(lr_metrics_dict.get('trade_count', 0)))
            col6.metric(T['xgb_trades_metric'], int(xgb_metrics_dict.get('trade_count', 0)))

            st.divider()
            st.header(T['folds_hdr'])
            st.caption(T['folds_caption'])
            tab_lr, tab_xgb = st.tabs(['LR (L1)', 'XGBoost'])
            with tab_lr:
                if lr_folds:
                    st.dataframe(_style_left(_format_fold_table(lr_folds)),
                                 use_container_width=True, hide_index=True)
                else:
                    st.info(T['no_folds_info'])
            with tab_xgb:
                if xgb_folds:
                    st.dataframe(_style_left(_format_fold_table(xgb_folds)),
                                 use_container_width=True, hide_index=True)
                else:
                    st.info(T['no_folds_info'])

            st.divider()
            st.header(T['metrics_hdr'])
            st.caption(T['metrics_caption'])
            st.dataframe(
                metrics.compare_strategies(raw_metrics_dict, lr_metrics_dict, xgb_metrics_dict),
                use_container_width=True, hide_index=True,
            )

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
            if lr_yearly is not None and lr_yearly['Partial'].any():
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

            st.subheader(T['ml_diag_sub'])
            st.caption(T['ml_diag_caption'])
            tab_ml_lr, tab_ml_xgb = st.tabs(['LR (L1)', 'XGBoost'])
            for tab, m_dict, folds in [(tab_ml_lr, lr_metrics_dict, lr_folds), (tab_ml_xgb, xgb_metrics_dict, xgb_folds)]:
                with tab:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric('OOS AUC', f"{m_dict.get('ml_oos_auc', 0):.3f}" if m_dict.get('ml_oos_n', 0) else "—")
                    c2.metric('OOS LogLoss', f"{m_dict.get('ml_oos_logloss', 0):.4f}" if m_dict.get('ml_oos_n', 0) else "—")
                    c3.metric('OOS Brier', f"{m_dict.get('ml_oos_brier', 0):.4f}" if m_dict.get('ml_oos_n', 0) else "—")
                    c4.metric('OOS Base Rate', f"{m_dict.get('ml_oos_base_rate', 0):.1%}" if m_dict.get('ml_oos_n', 0) else "—")

                    st.markdown(_ml_diag_summary_markdown(T, m_dict, folds))

                    if folds:
                        diag_rows = []
                        for r in folds:
                            if 'ML N' not in r:
                                continue
                            diag_rows.append({
                                'Fold': r.get('Fold'),
                                'Test Period': r.get('Test Period'),
                                'N': r.get('ML N', 0),
                                'Base Rate': r.get('ML Base Rate', 0.0),
                                'AUC': r.get('ML AUC', 0.0),
                                'LogLoss': r.get('ML LogLoss', 0.0),
                                'Brier': r.get('ML Brier', 0.0),
                            })
                        if diag_rows:
                            df_diag = pd.DataFrame(diag_rows)
                            df_diag['Fold'] = df_diag['Fold'].astype(str)
                            df_diag['Base Rate'] = df_diag['Base Rate'].apply(lambda x: f"{x:.1%}")
                            df_diag['AUC'] = df_diag['AUC'].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
                            df_diag['LogLoss'] = df_diag['LogLoss'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
                            df_diag['Brier'] = df_diag['Brier'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
                            st.dataframe(_style_left(df_diag), use_container_width=True, hide_index=True)
                        else:
                            st.info(T['no_ml_diag_info'])
                    else:
                        st.info(T['no_ml_diag_info'])

            st.subheader(T['fi_sub'])
            tab_fi_lr, tab_fi_xgb = st.tabs(['LR (L1)', 'XGBoost'])
            with tab_fi_lr:
                st.caption(T['fi_caption_lr'].format(date=backtest_start))
                st.dataframe(model.get_feature_importance(ref_lr_model, feature_cols),
                             use_container_width=True, hide_index=True)
            with tab_fi_xgb:
                st.caption(T['fi_caption_xgb'].format(date=backtest_start))
                st.dataframe(model.get_feature_importance(ref_xgb_model, feature_cols),
                             use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Error: {e}")
            st.code(traceback.format_exc())

    st.divider()
    st.caption(T['footer'])


if __name__ == "__main__":
    main()
