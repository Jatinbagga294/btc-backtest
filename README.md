# BTC Backtesting & Strategy Research

An event-driven backtesting engine for cryptocurrency trading strategies, and a
walk-forward validation harness that tells you whether a strategy actually
works or whether you just overfit it.

Built over 4,440,799 rows of 1-minute BTCUSDT data from August 2017 to
January 2026, resampled into seven timeframes with 40+ engineered features per
bar.

## What it does

- **Scale:** 4,440,799 rows of 1-minute BTCUSDT data (534 MB), August 2017 to
  January 2026, resampled into seven timeframes.
- **Features:** 40+ engineered per bar, from moving averages and momentum to
  volatility, candle structure, volume pressure and fair value gaps.
- **Engine:** event-driven, bar by bar, with realistic fills, fees on both
  sides, long and short, and intrabar stop-loss and take-profit.
- **Validation:** walk-forward. Parameters are tuned on one window and scored on
  the next unseen one, so results reflect out-of-sample performance rather than
  curve fitting.
- **Baseline:** every result is reported next to buy-and-hold over the same
  window.

## Status

This is an ongoing research project. The engine and validation framework are
done. The strategies in `strategies.py` are simple reference implementations
(EMA crossover, RSI reversion, trend-filtered RSI) used to test the harness end
to end. New strategies plug in as a single function and go through the same
validation.

Baseline runs are saved in `results/`, and you can reproduce them with
`python run_walkforward.py --timeframe 1d`.

## What is in here

```
backtester/
  backtester.py    event-driven engine: long/short, stop-loss, take-profit, fees
  validation.py    walk-forward splitter and out-of-sample aggregation
  baselines.py     buy-and-hold, Sharpe, max drawdown
  strategies.py    signal functions and their parameter grids
  data.py          feature file loading
run_walkforward.py validate every strategy, print the table, write JSON
tests/             14 tests covering the engine, metrics and fold construction
```

### The engine

Bar-by-bar simulation, not vectorised. Slower, but it models things vectorised
backtests routinely get wrong:

- Stop-loss and take-profit are checked against each bar's actual high and low,
  so an intrabar stop triggers on the bar it really would have.
- Fees are charged on both entry and exit, and applied to the entry price rather
  than deducted afterwards.
- Long and short both supported, with the stop and target mirrored correctly.
- Open positions are force-closed at the end of the window, so no phantom
  unrealised gain inflates the result.

### Features

40+ per bar: EMA (9/21/50/100/200), RSI, ATR, MACD with signal and histogram,
Bollinger Bands with width and position, candle anatomy (body ratio, upper and
lower wick), volume ratio and buy pressure, and bullish/bearish fair value gap
detection.

### Metrics

Annualised Sharpe on daily returns, maximum drawdown from a rolling equity peak,
profit factor, win rate, total return, plus the full trade log and equity curve.
Sharpe assumes a zero risk-free rate, which is the usual simplification and
flatters the strategy slightly.

## Running it

```bash
pip install -r requirements-dev.txt
pytest                                    # 14 tests, no data needed
python run_walkforward.py --timeframe 1d
```

Market data is not committed. See [data/README.md](data/README.md) for how to
rebuild it from Binance's public archive.

## What I would do next

- Test on out-of-sample assets. Everything here is BTC, so the conclusions may
  be about Bitcoin's particular trend rather than about the strategies.
- Add slippage. Fees are modelled, market impact is not, so real results would
  be worse than these.
- Position sizing. Every trade currently risks the full account, which makes
  drawdowns worse than a real risk model would allow.
- More strategies: model-driven signals, regime detection, and combinations
  of the existing features.
