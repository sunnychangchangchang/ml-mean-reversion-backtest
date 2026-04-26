# ML-Enhanced Mean Reversion Strategy

A compact quant research project for testing a short-horizon mean-reversion hypothesis with an interpretable machine-learning filter and walk-forward out-of-sample validation.

The strategy is deliberately simple: identify large 5-day selloffs, enter at the next open, exit at the next close, and use Logistic Regression only as a trade filter. The project is designed to make the research process auditable: no look-ahead bias, explicit execution timing, portfolio-level constraints, transaction costs, and benchmark comparison.

## Strategy Summary

| Item | Implementation |
| --- | --- |
| Universe | Cached US large-cap research universe: `AAPL`, `MSFT`, `NVDA`, `TSLA`, `AMZN` |
| Hypothesis | Stocks with sharp short-term declines may mean-revert intraday on the next session |
| Raw signal | `return_5d < -5%` by default |
| ML target | `1` if `Close[t+1] > Open[t+1]`, otherwise `0`; unknown final observations remain `NaN` |
| Model | Standardized Logistic Regression |
| ML threshold | Fixed at `0.50`, the natural calibrated probability boundary |
| Execution | Signal at close of day `t`; enter `Open[t+1]`; exit `Close[t+1]` |
| Cost model | `0.10%` round-trip transaction cost |
| Portfolio | Long only, no leverage, max 3 concurrent positions, max 50% per position |
| Validation | Expanding-window walk-forward; default 3-year minimum training history and 12-month test windows |
| Benchmark | SPY buy-and-hold normalized to the same out-of-sample start date |

## Why This Is Structured This Way

The mean-reversion rule defines the economic hypothesis. The ML model does not forecast price levels or create the primary signal; it estimates whether a candidate trade has positive next-day intraday direction.

The probability threshold is fixed at `0.50` rather than optimized against backtest Sharpe. This avoids turning the ML layer into a tuned rule. Calibration is evaluated on the same walk-forward out-of-sample predictions used by the backtest, so the threshold choice can be assessed without leaking test performance into model selection.

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
    ├── model.py              # Logistic Regression, feature importance, calibration
    ├── backtest.py           # Portfolio backtest and metrics
    ├── walkforward.py        # Expanding-window out-of-sample validation
    ├── metrics.py            # Strategy comparison, yearly returns, SPY benchmark
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
- `Max Positions` and `Max Weight`: portfolio risk controls.
- `Risk-Free Rate`: used for Sharpe calculation.

Fixed parameters in the app:

- Initial capital: `$100,000`
- ML probability threshold: `0.50`
- Transaction cost: `0.10%`
- Test window: `12` months
- Minimum training period: `3` years

## Run The Notebook

```bash
jupyter notebook notebooks/research_demo.ipynb
```

The notebook mirrors the production research path:

1. Load cached or downloaded data.
2. Build features and target labels.
3. Run raw and ML-filtered walk-forward backtests.
4. Compare metrics, folds, equity curves, drawdowns, yearly returns, and calibration.
5. Inspect reference-model coefficients and recent trades.

## Feature Set

Features are computed independently per ticker and use only information available by the close of day `t`:

- `return_1d`
- `volatility_5d`
- `volatility_10d`
- `volume_zscore`
- `rsi_14`
- `distance_ma20`

`return_5d` is intentionally excluded from the ML feature set because it is already the raw signal trigger. Keeping it out makes the model a filter on market context rather than a model that relearns the rule.

## Backtest Mechanics

For each signal date:

1. At close of day `t`, compute features and raw signal.
2. In the ML strategy, trade only if `ml_probability > 0.50`.
3. If more names qualify than allowed, select the highest-ranked names.
4. Enter at `Open[t+1]`.
5. Exit at `Close[t+1]`.
6. Subtract round-trip transaction cost.
7. Update portfolio capital at the end of the trading day.

The equity curve used for annualized return, volatility, Sharpe, and drawdown is forward-filled over all trading days, so inactive days are represented as cash.

## Reported Metrics

- Cumulative return
- Annualized return
- Annualized volatility
- Sharpe ratio across all days
- Active-period Sharpe on days with exposure
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
- The ML model is intentionally interpretable; it is not intended to maximize leaderboard performance.

## Professional Pitch

This project tests whether short-term large-cap selloffs contain a next-session intraday reversal premium. The raw rule captures the economic hypothesis, while Logistic Regression acts as a calibrated filter on technical context. The result is evaluated through expanding-window walk-forward validation with portfolio constraints, costs, SPY comparison, and explicit controls against look-ahead bias and parameter overfitting.
