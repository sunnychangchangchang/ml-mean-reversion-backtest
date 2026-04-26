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
    'portfolio_sub':        'Portfolio Risk Controls',
    'max_pos_label':        'Max Concurrent Positions',
    'max_weight_label':     'Max Weight per Position (%)',
    'benchmark_sub':        'Benchmark',
    'rfr_label':            'Annual Risk-Free Rate (%)',
    'rfr_caption':          'Current T-bill rate, used in Sharpe calculation.',
    'wf_sub':               'Walk-Forward Design',
    'min_train_label':      'Min Training Period (years)',
    'min_train_caption':    (
        'Minimum history the model must see before its first prediction. '
        'Longer = more stable model, shorter OOS window.'
    ),
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
| Max Positions | Concentration limit | Lower to cherry-pick best trades |
| Max Weight | Single-position risk cap | Increase to hide poor diversification |
| Risk-Free Rate | Sharpe denominator — set to current T-bill rate | Change to make Sharpe look better |
| Min Training Period | Walk-forward stability | Lower just to get more OOS data |

---

### Why some parameters are fixed

- **ML Threshold = {ml_thr:.2f}**: shared by both Logistic Regression (L1) and XGBoost.
  OOS calibration confirms LR probabilities are accurate —
  {ml_thr:.2f} is the chosen threshold. Tuning it further on backtest results is overfitting.
- **Transaction Cost = 0.1 %**: Realistic round-trip estimate for liquid large-cap US equities.
  It is not a free parameter.
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
    'insights_hdr':         '🔍 ML Model Insights',
    'cal_sub':              'Probability Calibration',
    'cal_desc': (
        'Evaluated on **out-of-sample walk-forward predictions** — '
        'the same predictions used in the backtest, not training data. '
        'A well-calibrated model justifies the current threshold of **{ml_thr:.2f}**.'
    ),
    'cal_expander':         '📖 How to read calibration metrics',
    'cal_body': """\
**Calibration** means: when the model says probability = 0.60, about 60 % of those days
actually close up. If the model is not calibrated, its probability output is meaningless
and no threshold can be trusted.

**The chart (Reliability Diagram)**
- Perfect calibration = points lie on the grey diagonal
- Above diagonal: model is underconfident
- Below diagonal: model is overconfident

**ECE (Expected Calibration Error)** — mean gap between predicted and actual rate

| ECE | Meaning |
|-----|---------|
| < 0.03 | Excellent — threshold {ml_thr:.2f} is well-justified |
| 0.03 – 0.05 | Good — threshold {ml_thr:.2f} is reasonable |
| 0.05 – 0.08 | Moderate — use with caution |
| > 0.08 | Poor — threshold needs CV-based tuning |

**Brier Score** — MSE of probabilities.
Compare to random baseline (= base_rate × (1 − base_rate)).
If Brier ≈ random, the model has learned nothing.
""",
    'cal_ece_help':         'Expected Calibration Error — lower is better.',
    'cal_brier_help':       'MSE of probability predictions. Lower is better.',
    'cal_base_help':        'Fraction of days where Close[t+1] > Open[t+1].',

    # ── Performance metric labels & tooltips ─────────────────────────────────
    'metric_label_cumreturn':    'Cumulative Return',
    'metric_label_annreturn':    'Annualized Return',
    'metric_label_vol':          'Volatility (ann.)',
    'metric_label_sharpe':       'Sharpe (all-period)',
    'metric_label_active_sharpe':'Sharpe (active-period)',
    'metric_label_maxdd':        'Max Drawdown',
    'metric_label_winrate':      'Win Rate',
    'metric_label_trades':       'Trade Count',
    'metric_label_avgtrade':     'Avg Trade Return',
    'metric_label_exposure':     'Exposure Ratio',

    'metric_help_cumreturn':     '(Final capital / Initial capital) − 1.',
    'metric_help_annreturn':     'Geometric annualized return: (final / initial)^(1/years) − 1.',
    'metric_help_vol':           'Annualized std of daily returns on the full equity curve, including flat cash days.',
    'metric_help_sharpe':        '(Annualized return − risk-free rate) / volatility. Flat cash days are included in the denominator.',
    'metric_help_active_sharpe': 'Sharpe computed only on active trading days, annualized by actual active-days-per-year — not 252. Avoids inflating Sharpe for low-exposure strategies.',
    'metric_help_maxdd':         'Largest peak-to-trough decline in portfolio value.',
    'metric_help_winrate':       'Fraction of trades with positive net P&L after transaction costs.',
    'metric_help_trades':        'Total individual position entries across the full backtest.',
    'metric_help_avgtrade':      'Mean net return per trade after round-trip transaction costs.',
    'metric_help_exposure':      'Fraction of total calendar trading days with at least one open position.',
    'cal_excellent':        'ECE = {ece:.4f} — Excellent. Threshold {ml_thr:.2f} is well-justified.',
    'cal_good':             'ECE = {ece:.4f} — Good. Threshold {ml_thr:.2f} is reasonable.',
    'cal_moderate':         'ECE = {ece:.4f} — Moderate. Proceed with caution.',
    'cal_poor':             'ECE = {ece:.4f} — Poor. Threshold needs CV-based tuning.',
    'eda_sub':    'Feature vs Win Rate (EDA)',
    'eda_caption': (
        'Each bar is a quantile bin of the feature value; bar height = win rate (target=1 rate) in that bin. '
        'Dashed line = overall base rate. '
        'A roughly flat or monotone pattern suggests a linear relationship — LR should suffice. '
        'A curve or reversal suggests non-linearity — XGBoost may add value.'
    ),
    'fi_sub':               'Feature Importance',
    'fi_caption':           (
        'Logistic Regression coefficients from the reference model '
        '(trained on all data before {date}). '
        'Positive = feature increases P(Close[t+1] > Open[t+1]).'
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
    'portfolio_sub':        '投資組合風險控制',
    'max_pos_label':        '最大同時持倉數',
    'max_weight_label':     '單一持倉最大權重 (%)',
    'benchmark_sub':        '基準比較',
    'rfr_label':            '年化無風險利率 (%)',
    'rfr_caption':          '當前公債利率，用於計算 Sharpe Ratio。',
    'wf_sub':               'Walk-Forward 設計',
    'min_train_label':      '最小訓練期間（年）',
    'min_train_caption':    (
        '模型在第一次預測前至少需要看到多少歷史資料。'
        '越長 = 模型越穩定，但 OOS 回測窗口越短。'
    ),
    'fixed_header':         '**固定參數** *（不可調整）*',
    'fixed_body':           (
        '- ML 門檻：**{ml_thr:.2f}** — LR 與 XGBoost 共用  \n'
        '- 交易成本：**{tc:.1%}** 來回  \n'
        '- 測試窗口：每折 **{tw} 個月**  \n'
        '- 最小訓練期間：**{mty} 年** — LR 收斂所需樣本量已足夠  \n'
        '- 初始資本：**${ic:,.0f}**（僅供顯示，不影響相對績效）'
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
| 最大持倉數 | 集中度限制 | 降低以挑選最佳交易 |
| 最大權重 | 單一部位風險上限 | 提高以掩蓋分散化不足 |
| 無風險利率 | Sharpe 分母——設為當前公債利率 | 改動以讓 Sharpe 看起來更好 |
| 最小訓練期間 | Walk-forward 穩定性 | 降低只是為了得到更多 OOS 資料 |

---

### 為什麼有些參數是固定的

- **ML 門檻 = {ml_thr:.2f}**：LR（L1）和 XGBoost 共用同一個門檻。
  OOS 校準測試確認 LR 機率是可信的——
  {ml_thr:.2f} 是目前使用的門檻。根據回測 Sharpe 進一步調整門檻就是 overfitting。
- **交易成本 = 0.1%**：大型美股的實際來回成本估算，不是可以優化的參數。
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
    'insights_hdr':         '🔍 ML 模型分析',
    'cal_sub':              '機率校準（Probability Calibration）',
    'cal_desc': (
        '此結果使用 **walk-forward out-of-sample 預測**評估——'
        '與回測使用的是同一批預測，並非訓練資料。'
        '校準良好代表目前使用的門檻 **{ml_thr:.2f}** 是有依據的。'
    ),
    'cal_expander':         '📖 如何解讀校準指標',
    'cal_body': """\
**校準（Calibration）** 的意思是：當模型輸出機率 = 0.60，約 60% 的那些天真的會收漲。
如果模型沒有校準，機率輸出就沒有意義，任何門檻值都不可信。

**圖表（Reliability Diagram）**
- 完美校準 = 點落在灰色對角線上
- 對角線上方：模型信心不足（實際正率高於預測）
- 對角線下方：模型過度自信

**ECE（Expected Calibration Error）** — 預測機率與實際正率的平均絕對差距

| ECE | 意義 |
|-----|------|
| < 0.03 | 優秀 — 門檻 {ml_thr:.2f} 完全合理 |
| 0.03 – 0.05 | 良好 — 門檻 {ml_thr:.2f} 可接受 |
| 0.05 – 0.08 | 一般 — 需謹慎 |
| > 0.08 | 不佳 — 需透過訓練期 CV 選擇門檻 |

**Brier Score** — 機率預測的均方誤差，越低越好。
參考基準：隨機猜測 = base_rate × (1 − base_rate)。
若 Brier ≈ 基準，代表模型幾乎沒有學到任何有用資訊。
""",
    'cal_ece_help':         'Expected Calibration Error，越低越好。',
    'cal_brier_help':       '機率預測的均方誤差，越低越好。',
    'cal_base_help':        'Close[t+1] > Open[t+1] 的天數比例（訓練期）。',

    # ── 績效指標標籤與說明 ────────────────────────────────────────────────────
    'metric_label_cumreturn':    '累計報酬',
    'metric_label_annreturn':    '年化報酬',
    'metric_label_vol':          '年化波動率',
    'metric_label_sharpe':       'Sharpe（全期）',
    'metric_label_active_sharpe':'Sharpe（持倉期）',
    'metric_label_maxdd':        '最大回撤',
    'metric_label_winrate':      '勝率',
    'metric_label_trades':       '交易次數',
    'metric_label_avgtrade':     '平均交易報酬',
    'metric_label_exposure':     '曝險比率',

    'metric_help_cumreturn':     '（最終資本 / 初始資本）− 1。',
    'metric_help_annreturn':     '幾何年化報酬：(最終 / 初始)^(1/年數) − 1。',
    'metric_help_vol':           '完整資產曲線（含閒置天）日報酬率的年化標準差。',
    'metric_help_sharpe':        '（年化報酬 − 無風險利率）/ 年化波動率，包含閒置天。',
    'metric_help_active_sharpe': '僅計算持倉日的 Sharpe，以實際每年平均持倉天數年化（非固定 252）。避免低曝險策略的 Sharpe 虛高。',
    'metric_help_maxdd':         '資產曲線中最大的峰值到谷底跌幅。',
    'metric_help_winrate':       '扣除交易成本後，盈利交易佔總次數的比例。',
    'metric_help_trades':        '回測期間的總進場次數。',
    'metric_help_avgtrade':      '扣除來回交易成本後，每筆交易的平均淨報酬。',
    'metric_help_exposure':      '至少持有一個倉位的交易日佔總交易日的比例。',
    'cal_excellent':        'ECE = {ece:.4f} — 優秀。使用門檻 {ml_thr:.2f} 完全合理。',
    'cal_good':             'ECE = {ece:.4f} — 良好。門檻 {ml_thr:.2f} 可接受。',
    'cal_moderate':         'ECE = {ece:.4f} — 一般。請謹慎解讀。',
    'cal_poor':             'ECE = {ece:.4f} — 不佳。應透過訓練期 CV 重新選擇門檻。',
    'eda_sub':    'Feature vs 勝率（EDA）',
    'eda_caption': (
        '每個 bar 是 feature 值的 quantile bin；bar 高度 = 該 bin 內的勝率（target=1 比例）。'
        '虛線 = 整體基礎勝率。'
        '幾乎平坦或單調 → 線性關係，LR 已足夠。'
        '有曲線或反轉 → 非線性，XGBoost 可能有額外幫助。'
    ),
    'fi_sub':               '特徵重要性',
    'fi_caption':           (
        '以 {date} 前所有訓練資料訓練的參考模型之邏輯回歸係數。'
        '正值代表該特徵提升 P(Close[t+1] > Open[t+1]) 的機率。'
    ),
    'footer': (
        '**避免前視偏差**：訊號僅使用第 *t* 日收盤時已知資料 · '
        '交易於 Open[t+1] 進場 · 每折 walk-forward 模型僅以過去資料訓練。'
    ),
}


def get(lang: str) -> dict:
    """Return translation dict for the given language code ('en' or 'zh')."""
    return _ZH if lang == 'zh' else _EN
