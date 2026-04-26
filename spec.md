# Research Specification: ML-Filtered Intraday Mean Reversion

## Objective

Build a transparent quant research workflow for a short-horizon US equity mean-reversion strategy.

The implementation must satisfy four requirements:

- The trading hypothesis is explicit and not data-mined inside the UI.
- The ML model is a filter, not a black-box alpha engine.
- All model predictions used in the reported backtest are out-of-sample.
- Portfolio returns reflect execution timing, transaction costs, concentration limits, and inactive cash days.

## Economic Hypothesis

Large, liquid equities that sell off sharply over a short window may exhibit next-session intraday mean reversion.

The raw hypothesis is:

```text
return_5d[t] < threshold
```

Default threshold:

```text
-5%
```

The threshold is the main research assumption. It may be varied for robustness checks, but should not be optimized to maximize Sharpe.

## Universe

Supported tickers in the Streamlit UI:

```text
AAPL, MSFT, NVDA, TSLA, AMZN, GOOGL, META, NFLX
```

Default universe:

```text
AAPL, MSFT, NVDA, TSLA, AMZN
```

Benchmark:

```text
SPY buy-and-hold
```

Known limitation: this is a survivorship-biased large-cap universe.

## Data

Data fields:

- `Open`
- `High`
- `Low`
- `Close`
- `Volume`

Data loading behavior:

1. If `Use Cache` is enabled, check `data/cache/{TICKER}.csv`.
2. Use the cached file only if it covers the requested date range with a small tolerance for weekends and holidays.
3. If cache is missing or stale, download with Yahoo Finance through `yfinance`.
4. Save successful downloads back to `data/cache/`.
5. `Refresh Data` bypasses the cache and forces a new download.

The current implementation does not use Stooq. `pandas-datareader` remains in dependencies but is not part of the active data path.

## Features

All features are computed per ticker using information available by the close of day `t`.

| Feature | Definition |
| --- | --- |
| `return_1d` | 1-day close-to-close return |
| `return_5d` | 5-day close-to-close return, used as the raw signal trigger |
| `volatility_5d` | 5-day rolling standard deviation of `return_1d` |
| `volatility_10d` | 10-day rolling standard deviation of `return_1d` |
| `volume_zscore` | 20-day z-score of volume |
| `rsi_14` | 14-day RSI using Wilder smoothing |
| `distance_ma20` | `(Close - MA20) / MA20` |

ML feature columns:

```text
return_1d
volatility_5d
volatility_10d
volume_zscore
rsi_14
distance_ma20
```

`return_5d` is excluded from ML features because it is the signal definition. This keeps the classifier from simply relearning the entry rule.

## Target

The model predicts next-session intraday direction:

```text
target[t] = 1 if Close[t+1] > Open[t+1]
target[t] = 0 if Close[t+1] <= Open[t+1]
```

If `Open[t+1]` or `Close[t+1]` is unavailable, `target[t] = NaN`.

Rows with missing feature values or missing targets are excluded from model training and calibration. Rows with valid features may still receive a live-style probability estimate even when the next outcome is not yet known.

## Model

Model:

```text
StandardScaler + LogisticRegression
```

Rationale:

- Interpretable coefficients.
- Native probability output.
- Fast retraining in walk-forward validation.
- Lower overfitting risk than complex nonlinear models on a small universe.

The ML probability threshold is fixed:

```text
0.50
```

This is the natural decision boundary for a calibrated binary classifier. It is evaluated using out-of-sample walk-forward calibration rather than optimized on realized trading performance.

## Signal Logic

Raw strategy:

```text
raw_signal[t] = return_5d[t] < threshold
```

ML-filtered strategy:

```text
signal[t] = raw_signal[t] and ml_probability[t] > 0.50
```

If the number of valid signals exceeds `max_positions`:

- Raw strategy ranks candidates by absolute 5-day decline magnitude.
- ML strategy ranks candidates by `ml_probability`.

## Execution Assumptions

Signal timestamp:

```text
Close of day t
```

Entry:

```text
Open[t+1]
```

Exit:

```text
Close[t+1]
```

Gross trade return:

```text
Close[t+1] / Open[t+1] - 1
```

Net trade return:

```text
gross_trade_return - transaction_cost
```

Default round-trip cost:

```text
0.10%
```

## Portfolio Construction

Initial capital:

```text
$100,000
```

Rules:

- Long only.
- No leverage.
- Maximum concurrent positions: `3`.
- Maximum single-name weight: `50%`.
- Equal weight across selected names, capped by max weight.
- Position sizing uses start-of-day capital, not capital after earlier trades on the same day.

Allocation examples:

| Selected signals | Weight per position |
| --- | --- |
| 1 | 50% |
| 2 | 50% |
| 3 | 33.33% |

If no trades are taken in a period, capital remains unchanged.

## Walk-Forward Validation

The production research path uses expanding-window walk-forward validation.

Defaults:

```text
Minimum training period: 3 years
Test window: 12 months
```

For each fold:

1. Train on all available history before the fold start.
2. Predict only within the fold test window.
3. Run the portfolio backtest on that fold.
4. Carry ending capital into the next fold.
5. Store fold-level metrics for consistency analysis.

The reported ML backtest is stitched from out-of-sample predictions only.

## Metrics

Primary metrics:

- Cumulative return
- Annualized return
- Annualized volatility
- All-period Sharpe ratio
- Active-period Sharpe ratio
- Max drawdown
- Win rate
- Trade count
- Average trade return
- Exposure ratio
- Final capital

Additional diagnostics:

- Fold-level return, Sharpe, drawdown, win rate, and trade count.
- Calendar-year returns for raw strategy, ML strategy, and SPY.
- Trade P&L distribution.
- Logistic Regression coefficient table.
- Out-of-sample calibration curve.
- Expected Calibration Error.
- Brier score versus random baseline.

## UI Requirements

Sidebar controls:

- Language toggle: English / Chinese.
- Ticker selection.
- Start date.
- End date.
- Use Cache.
- Refresh Data.
- Mean Reversion Threshold.
- Max Concurrent Positions.
- Max Weight per Position.
- Annual Risk-Free Rate.

Fixed in UI:

- Initial capital.
- ML probability threshold.
- Transaction cost.
- Test window length.
- Minimum training years.

Main page:

- Survivorship-bias warning.
- Correct-use guide.
- Strategy description.
- Summary metrics.
- Walk-forward fold table.
- Raw versus ML metric comparison.
- Equity curves with SPY.
- Yearly return bars.
- ML drawdown.
- ML trade P&L distribution.
- Recent ML trades.
- OOS calibration diagnostics.
- Feature importance.

## Bias Controls

Look-ahead control:

- Features use data available by close of day `t`.
- Target and intraday return reference `t+1`, but only for training labels and realized backtest P&L.
- Trades execute after signal formation, at `Open[t+1]`.
- Walk-forward folds train only on history before the test window.

Overfitting control:

- ML threshold is fixed at `0.50`.
- Transaction cost is fixed.
- Test window length is fixed.
- Threshold changes should be interpreted as robustness checks, not optimization.

Reporting control:

- Metrics are calculated on complete daily equity curves with flat cash days included.
- Active-period Sharpe is reported separately to isolate performance on deployed capital.

## Limitations

- Survivorship-biased universe.
- Small number of names.
- Daily bars do not capture intraday liquidity or opening auction execution quality.
- Transaction cost is static and does not model market impact.
- No borrow, shorting, tax, or cash interest model.
- No regime filter, sector neutrality, or volatility targeting.

## Extension Ideas

- Expand to a historical point-in-time universe.
- Add liquidity and volatility filters.
- Use intraday data to model open execution and slippage.
- Add regime features or market-state filters.
- Test cross-sectional ranking models.
- Add pre-registered robustness bands for the return threshold.
