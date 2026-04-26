# ML-Enhanced Mean Reversion Strategy

A compact quant research project for testing a short-horizon mean-reversion hypothesis with machine-learning filters and walk-forward out-of-sample validation.

The strategy is deliberately simple: identify large 5-day selloffs, enter at the next open, exit at the next close, and use a classifier only as a trade filter. Two ML models are compared side-by-side — **Logistic Regression (L1)** for interpretability and **XGBoost** for non-linear pattern capture — while the raw rule serves as the baseline.

## Strategy Summary

| Item | Implementation |
| --- | --- |
| Universe | Cached US large-cap research universe: `AAPL`, `MSFT`, `NVDA`, `TSLA`, `AMZN` |
| Hypothesis | Stocks with sharp short-term declines may mean-revert intraday on the next session |
| Raw signal | `return_5d < -5%` by default |
| ML target | `1` if `Close[t+1] > Open[t+1]`, otherwise `0` |
| Models | LR (L1, C=0.5, liblinear) and XGBoost (max_depth=3, n_estimators=300) |
| ML threshold | `0.55` — above the 0.50 natural boundary to filter for higher-confidence trades |
| Execution | Signal at close of day `t`; enter `Open[t+1]`; exit `Close[t+1]` |
| Cost model | `0.05%` one-way transaction cost (`0.10%` round-trip) |
| Portfolio | Long only, no leverage, max 3 concurrent positions, max 50% per position |
| Validation | Expanding-window walk-forward; 3-year minimum training history, 12-month test windows |
| Benchmark | SPY buy-and-hold normalized to the same out-of-sample start date |

## Why This Is Structured This Way

The mean-reversion rule defines the economic hypothesis. The ML models do not create the primary signal; they estimate whether a candidate trade has positive next-day intraday direction given the current market context.

The probability threshold is fixed at `0.55` rather than optimized against backtest Sharpe. Calibration is evaluated on the same walk-forward out-of-sample predictions used by the backtest, confirming the threshold is meaningful. Two models are compared to distinguish what is learnable by linear versus non-linear classifiers.

## Project Structure

```text
ml-mean-reversion-backtest/
├── app.py                    # Streamlit research UI
├── requirements.txt          # Python dependencies
├── README.md                 # Project overview
├── spec.md                   # Research and implementation specification
├── data/cache/               # Local OHLCV cache
├── notebooks/
│   └── research_demo.ipynb   # Reproducible notebook workflow
└── src/
    ├── data.py               # Data loading with local cache and Yahoo Finance fallback
    ├── features.py           # Feature engineering and target labeling
    ├── model.py              # LR (L1) and XGBoost training, importance, calibration
    ├── backtest.py           # Portfolio backtest and metrics
    ├── walkforward.py        # Expanding-window out-of-sample validation
    ├── metrics.py            # Strategy comparison (Raw / LR / XGBoost), yearly returns
    ├── plots.py              # Plotly visualizations
    └── i18n.py               # English / Chinese UI copy
```

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run The App

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal, typically `http://localhost:8501`.

The sidebar controls are intentionally limited:

- `Use Cache`: load `data/cache/{TICKER}.csv` first when the requested date range is covered.
- `Refresh Data`: force a fresh Yahoo Finance download and overwrite cache files.
- `Mean Reversion Threshold`: the main economic hypothesis parameter.
- `Risk-Free Rate`: used for Sharpe calculation.

Fixed parameters (not exposed as controls to prevent overfitting):

- Initial capital: `$100,000`
- ML probability threshold: `0.55`
- Transaction cost: `0.10%` round-trip (`0.05%` one-way)
- Test window: `12` months
- Minimum training period: `3` years
- Max positions: `3`
- Max weight per position: `50%`

## Run The Notebook

```bash
jupyter notebook notebooks/research_demo.ipynb
```

The notebook mirrors the production research path:

1. Load cached or downloaded data.
2. Build features and target labels.
3. Run raw, LR-filtered, and XGBoost-filtered walk-forward backtests.
4. Compare metrics, folds, equity curves, drawdowns, yearly returns, and calibration.
5. Inspect reference-model feature importance and recent trades.

## Feature Set

Features are computed independently per ticker using only information available by the close of day `t`:

| Feature | Definition |
| --- | --- |
| `return_1d` | 1-day close-to-close return |
| `return_10d` | 10-day close-to-close return |
| `volatility_5d` | 5-day rolling std of `return_1d` |
| `volatility_10d` | 10-day rolling std of `return_1d` |
| `rsi_14` | 14-day RSI (Wilder smoothing) |
| `distance_ma20` | `(Close − MA20) / MA20` |
| `distance_ma200` | `(Close − MA200) / MA200` |
| `gap_open` | `(Open[t] − Close[t−1]) / Close[t−1]` — overnight gap |

`return_5d` is intentionally excluded from ML features because it is the raw signal trigger. L1 regularization on LR automatically zeroes out features that provide no incremental predictive value.

## Backtest Mechanics

For each signal date:

1. At close of day `t`, compute features and raw signal.
2. In each ML strategy, trade only if `ml_probability > 0.55`.
3. If more names qualify than allowed, select the highest-ranked names.
4. Enter at `Open[t+1]`.
5. Exit at `Close[t+1]`.
6. Subtract round-trip transaction cost.
7. Update portfolio capital at the end of the trading day.

The equity curve used for annualized return, volatility, Sharpe, and drawdown is forward-filled over all trading days so inactive days are represented as cash.

## Reported Metrics

- Cumulative return
- Annualized return
- Annualized volatility
- All-period Sharpe ratio (includes flat cash days)
- Active-period Sharpe ratio (annualized by actual active-days-per-year, not 252)
- Max drawdown
- Win rate
- Trade count
- Average trade return
- Exposure ratio
- Final capital
- Calendar-year returns versus SPY
- Out-of-sample calibration: ECE, Brier score, base rate

## Research Caveats

- The universe is survivorship-biased because it uses currently listed large-cap equities.
- Daily OHLC data assumes fills at the next open and next close; real execution can differ.
- The cost model is simple and does not include market impact.
- The strategy is tested on a small liquid equity universe, not a full institutional investment universe.
- LR is intentionally interpretable; XGBoost adds non-linear capacity but reduces interpretability.

## Professional Pitch

This project tests whether short-term large-cap selloffs contain a next-session intraday reversal premium. Two ML models — a regularized Logistic Regression for transparency and XGBoost for non-linear pattern capture — act as calibrated filters on the raw mean-reversion signal. All reported results use strictly out-of-sample expanding-window walk-forward predictions with portfolio constraints, transaction costs, SPY comparison, and explicit controls against look-ahead bias and parameter overfitting.
