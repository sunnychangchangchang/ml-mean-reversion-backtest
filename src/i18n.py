"""
UI translations.  T = i18n.get(lang)  →  T['key']
"""

_EN = {
    # ── page ──────────────────────────────────────────────────────────────────
    'title': '📈 ML-Enhanced Mean Reversion Strategy',

    # ── sidebar ───────────────────────────────────────────────────────────────
    'sidebar_header':       '⚙️ Parameters',
    'data_sub':             'Data',
    'tickers_label':        'Tickers',
    'start_label':          'Start Date',
    'end_label':            'End Date',
    'cache_label':          'Use Cache',
    'refresh_btn':          '🔄 Refresh Data',
    'strategy_sub':         'Strategy Hypothesis',
    'threshold_label':      'Mean Reversion Threshold (%)',
    'threshold_caption':    (
        'Buy when the 5-day return of a stock drops below this level. '
        'Set based on your market view — do not tune to maximise Sharpe.'
    ),
    'benchmark_sub':        'Benchmark',
    'rfr_label':            'Annual Risk-Free Rate (%)',
    'rfr_caption':          'Current T-bill rate, used in Sharpe calculation.',
    'fixed_header':         '**Fixed Parameters** *(not tunable)*',
    'fixed_body':           (
        '- ML Threshold: **{ml_thr:.2f}** — applies to both LR and XGBoost  \n'
        '- Transaction Cost: **{tc:.1%}** round-trip  \n'
        '- Test Window: **{tw} months** per fold  \n'
        '- Min Training Period: **{mty} years**  \n'
        '- Max Positions: **{mp}** — risk constraint  \n'
        '- Max Weight: **{mw:.0%}** — risk constraint  \n'
        '- Initial Capital: **${ic:,.0f}** (display only)'
    ),

    # ── main page ─────────────────────────────────────────────────────────────
    'surv_warning': (
        '⚠️ **Survivorship Bias**: All tickers are currently large, successful companies. '
        'Results are likely optimistic compared to a real investable universe that includes '
        'delisted and historically representative stocks.'
    ),
    'how_to_title':  '📋 How to use this tool correctly',
    'how_to_body': """\
### Correct workflow

**1. Set parameters before looking at results.**
The only parameter that defines the strategy *hypothesis* is the **threshold**.
Set it based on your market view — for example, *"I believe large-cap stocks that fall
more than 5 % in 5 days tend to recover intraday."*
Do not adjust it after seeing the backtest output.

**2. Run the backtest once.**
The walk-forward validation gives you one out-of-sample equity curve.
Treat it as your single test result.

**3. Check robustness — don't optimise.**
After you have a result, you may vary the threshold slightly (e.g., −4 % to −7 %) to
check whether performance holds across a range.
If the strategy only works at exactly one value, it is fragile and likely data-mined.

**4. Read the fold table, not just the headline.**
A single cumulative return figure is easy to get lucky on.
The fold table shows whether the strategy is *consistently* profitable across different
market regimes. Winning in 6 of 8 folds is more credible than a big win in 2 folds.

---

### What each control does

| Control | Purpose | What NOT to do |
|---------|---------|----------------|
| Threshold | Defines the mean-reversion hypothesis | Tune to maximise Sharpe |
| Risk-Free Rate | Sharpe denominator — set to current T-bill rate | Change to make Sharpe look better |

---

### Why some parameters are fixed

- **ML Threshold = {ml_thr:.2f}**: shared by both LR (L1) and XGBoost.
  Tuning it on backtest results is overfitting.
- **Transaction Cost = 0.1 %**: Realistic round-trip estimate for liquid large-cap US equities.
- **Max Positions / Max Weight**: Portfolio risk constraints, not strategy parameters.
- **Test Window = 12 months**: Industry-standard walk-forward cadence.

---

### Limitations

- **Survivorship bias**: Only currently-traded large-caps are available.
- **Market impact**: Entry at Open[t+1] assumes a perfect fill at the opening price.
- **Regime risk**: The strategy is tested on a broad equity bull market; performance in
  a prolonged bear market or liquidity crisis is unknown.
""",
    'strategy_desc': (
        '**Strategy**: buy when the 5-day return of a stock drops below **{thr:.0f}%**, '
        'then apply ML filters (Logistic Regression and XGBoost, probability > {ml_thr:.0%}) to select which signals to trade. '
        'Entry at Open[t+1], exit at Close[t+1].'
    ),
    'no_tickers_err':   'Please select at least one ticker.',
    'date_order_err':   'Start Date must be before End Date.',
    'spinner_text':     'Loading data and running walk-forward backtest...',
    'downloading_info': '📥 Downloading data...',
    'downloaded_ok':    'Loaded {rows:,} rows for {n} tickers',
    'features_info':    '🔧 Computing features...',
    'features_ok':      'Features ready',
    'wf_info':          '🤖 Running walk-forward validation...',
    'wf_ok':            'Walk-forward complete!',

    # ── results section headers ───────────────────────────────────────────────
    'summary_hdr':          '📋 Summary',
    'raw_capital_metric':   'Raw Final Capital',
    'ml_capital_metric':    'LR Final Capital',
    'raw_trades_metric':    'Raw Trades',
    'ml_trades_metric':     'LR Trades',
    'folds_hdr':            '🔄 Walk-Forward Fold Results (by Model)',
    'folds_caption': (
        'Each row is a separate out-of-sample period with its own freshly-trained model. '
        'Consistent positive returns across folds is stronger evidence than a high overall Sharpe.'
    ),
    'no_folds_info':        'No completed folds — check date range and min training period.',
    'metrics_hdr':          '📊 Performance Metrics',
    'metrics_caption': (
        'All-period Sharpe includes flat days (cash); '
        'active-period Sharpe reflects only days with open positions — '
        'more comparable to always-invested benchmarks like SPY.'
    ),
    'viz_hdr':              '📈 Visualizations',
    'equity_sub':           'Equity Curves',
    'yearly_sub':           'Yearly Returns vs SPY',
    'partial_caption':      '* Partial year — backtest started mid-year; not directly comparable to full years.',
    'drawdown_sub':         'Drawdown',
    'pnl_sub':              'Trade P&L Distribution',
    'trades_hdr':           '📝 Recent Trades',
    'no_trades_info':       'No trades generated — try a less restrictive threshold.',

    # ── ML insights ───────────────────────────────────────────────────────────
    'insights_hdr':     '🔍 ML Model Insights',
    'fi_sub':           'Feature Importance',
    'fi_caption_lr':    (
        'L1 Logistic Regression coefficients — reference model trained on all data before {date}. '
        'Positive = feature increases P(Close[t+1] > Open[t+1]). '
        'Features with coefficient = 0 were zeroed out by L1 regularization.'
    ),
    'fi_caption_xgb':   (
        'XGBoost feature importance (mean gain) — reference model trained on all data before {date}. '
        'Higher = more useful for splitting decisions across all trees.'
    ),
    'footer': (
        '**Look-ahead bias prevention**: signals use only data available at day *t* close · '
        'trades execute at Open[t+1] · each walk-forward fold trains strictly on past data.'
    ),
}

_ZH = {
    # ── page ──────────────────────────────────────────────────────────────────
    'title': '📈 機器學習均值回歸策略',

    # ── sidebar ───────────────────────────────────────────────────────────────
    'sidebar_header':       '⚙️ 參數設定',
    'data_sub':             '資料',
    'tickers_label':        '股票代號',
    'start_label':          '起始日期',
    'end_label':            '結束日期',
    'cache_label':          '使用快取',
    'refresh_btn':          '🔄 重新下載資料',
    'strategy_sub':         '策略假設',
    'threshold_label':      '均值回歸門檻 (%)',
    'threshold_caption':    (
        '當股票的 5 日報酬率低於此門檻時產生訊號。'
        '依據你對市場的看法設定——不要為了提高 Sharpe 而調整。'
    ),
    'benchmark_sub':        '基準比較',
    'rfr_label':            '年化無風險利率 (%)',
    'rfr_caption':          '當前公債利率，用於計算 Sharpe Ratio。',
    'fixed_header':         '**固定參數** *（不可調整）*',
    'fixed_body':           (
        '- ML 門檻：**{ml_thr:.2f}** — LR 與 XGBoost 共用  \n'
        '- 交易成本：**{tc:.1%}** 來回  \n'
        '- 測試窗口：每折 **{tw} 個月**  \n'
        '- 最小訓練期間：**{mty} 年**  \n'
        '- 最大持倉數：**{mp}** — 風險限制  \n'
        '- 最大權重：**{mw:.0%}** — 風險限制  \n'
        '- 初始資本：**${ic:,.0f}**（僅供顯示）'
    ),

    # ── main page ─────────────────────────────────────────────────────────────
    'surv_warning': (
        '⚠️ **倖存者偏差**：本工具中的股票均為目前仍在交易的大型知名公司。'
        '相較於包含下市股票的真實可投資宇宙，回測結果可能過於樂觀。'
    ),
    'how_to_title':  '📋 如何正確使用這個工具',
    'how_to_body': """\
### 正確流程

**1. 在看結果之前先設定參數。**
唯一定義策略「假設」的參數是**門檻值**。
根據你的市場觀點設定——例如：
*「我認為大型股在 5 天內下跌超過 5%，隔日盤中會有回彈。」*
**不要在看到回測結果後再調整它。**

**2. 只跑一次回測。**
Walk-forward 驗證給你的是一條 out-of-sample equity curve。
把它當作你唯一的測試結果。

**3. 做穩健性檢查，不是最佳化。**
在有了結果之後，你可以稍微改變門檻值（例如 −4% 到 −7%）
來確認績效是否在一個合理範圍內都成立。
如果只有某一個精確值有效，策略是脆弱的，可能是 data mining 的結果。

**4. 看各折表格，不是只看整體數字。**
單一的累計報酬率很容易靠運氣達到。
各折表格顯示策略在不同市場環境下是否穩定獲利。
8 折中 6 折正報酬，比 2 折大賺 6 折小虧更具說服力。

---

### 每個控制項的作用

| 控制項 | 用途 | 不應做的事 |
|--------|------|-----------|
| 門檻值 | 定義均值回歸假設 | 調整以最大化 Sharpe |
| 無風險利率 | Sharpe 分母——設為當前公債利率 | 改動以讓 Sharpe 看起來更好 |

---

### 為什麼有些參數是固定的

- **ML 門檻 = {ml_thr:.2f}**：LR（L1）和 XGBoost 共用，根據回測 Sharpe 調整就是 overfitting。
- **交易成本 = 0.1%**：大型美股的實際來回成本估算，不是可以優化的參數。
- **最大持倉數 / 最大權重**：投資組合風險限制，不是策略參數。
- **測試窗口 = 12 個月**：業界標準的 walk-forward 步進節奏。

---

### 需要注意的限制

- **倖存者偏差**：只有目前仍在交易的大型股可用。
- **市場衝擊**：假設以 Open[t+1] 完美成交，實際執行可能更差。
- **市場環境風險**：策略在大多頭市場中訓練和測試，在長期熊市或流動性危機中的表現未知。
""",
    'strategy_desc': (
        '**策略**：當股票 5 日報酬率低於 **{thr:.0f}%** 時產生訊號，'
        '再透過 ML 篩選器（Logistic Regression 和 XGBoost，機率 > {ml_thr:.0%}）決定是否進場。'
        '進場於 Open[t+1]，出場於 Close[t+1]。'
    ),
    'no_tickers_err':   '請至少選擇一檔股票。',
    'date_order_err':   '起始日期必須早於結束日期。',
    'spinner_text':     '資料載入中，執行 walk-forward 回測...',
    'downloading_info': '📥 下載資料中...',
    'downloaded_ok':    '已載入 {n} 支股票共 {rows:,} 筆資料',
    'features_info':    '🔧 計算特徵中...',
    'features_ok':      '特徵計算完成',
    'wf_info':          '🤖 執行 walk-forward 驗證...',
    'wf_ok':            'Walk-forward 完成！',

    # ── results section headers ───────────────────────────────────────────────
    'summary_hdr':          '📋 策略摘要',
    'raw_capital_metric':   '原始策略最終資本',
    'ml_capital_metric':    'LR 策略最終資本',
    'raw_trades_metric':    '原始策略交易次數',
    'ml_trades_metric':     'LR 策略交易次數',
    'folds_hdr':            '🔄 Walk-Forward 各折結果（依模型分頁）',
    'folds_caption': (
        '每一折都是獨立的 out-of-sample 期間，使用當折重新訓練的模型。'
        '各折結果一致才具有真正的說服力，整體 Sharpe 高但各折不穩定則不可信。'
    ),
    'no_folds_info':        '沒有完成的折——請確認日期範圍和最小訓練期間的設定。',
    'metrics_hdr':          '📊 績效指標',
    'metrics_caption': (
        'All-period Sharpe 含閒置天（持現金）；'
        'Active-period Sharpe 僅計算持倉日——'
        '後者更適合與 SPY 等永遠持倉的基準比較。'
    ),
    'viz_hdr':              '📈 視覺化',
    'equity_sub':           '資產曲線',
    'yearly_sub':           '年度報酬 vs SPY',
    'partial_caption':      '* 非完整年——回測期間內起訖於年度中，不宜與完整年直接比較。',
    'drawdown_sub':         'Drawdown',
    'pnl_sub':              '交易損益分布',
    'trades_hdr':           '📝 最近交易記錄',
    'no_trades_info':       '未產生任何交易——請嘗試放寬門檻值。',

    # ── ML insights ───────────────────────────────────────────────────────────
    'insights_hdr':     '🔍 ML 模型分析',
    'fi_sub':           '特徵重要性',
    'fi_caption_lr':    (
        'L1 邏輯回歸係數——參考模型以 {date} 前所有資料訓練。'
        '正值代表該特徵提升 P(Close[t+1] > Open[t+1])。'
        '係數為 0 的特徵已被 L1 正則化自動歸零。'
    ),
    'fi_caption_xgb':   (
        'XGBoost 特徵重要性（mean gain）——參考模型以 {date} 前所有資料訓練。'
        '數值越高代表該特徵在所有決策樹的分裂中平均貢獻越大。'
    ),
    'footer': (
        '**避免前視偏差**：訊號僅使用第 *t* 日收盤時已知資料 · '
        '交易於 Open[t+1] 進場 · 每折 walk-forward 模型僅以過去資料訓練。'
    ),
}


def get(lang: str) -> dict:
    """Return translation dict for the given language code ('en' or 'zh')."""
    return _ZH if lang == 'zh' else _EN
