# BTC Backtesting & Strategy Research

An event-driven backtesting engine for cryptocurrency trading strategies, and a
walk-forward validation harness that tells you whether a strategy actually
works or whether you just overfit it.

Built over 4,440,799 rows of 1-minute BTCUSDT data from August 2017 to
January 2026, resampled into seven timeframes with 40+ engineered features per
bar.

## The headline result

**None of the strategies tested beat buy-and-hold out of sample.** Not one.

Walk-forward validated, 14 folds, anchored 12-month train into 6-month test:

### Daily bars

| Strategy | Folds beating B&H | Strategy return | Buy & hold | Worst drawdown | Trades |
|---|---|---|---|---|---|
| EMA crossover | 2 / 14 | **-97.5%** | +1,801% | -52.9% | 863 |
| RSI reversion | 3 / 14 | **-80.8%** | +1,801% | -28.4% | 131 |
| Trend-filtered RSI | 2 / 6 | **-19.9%** | +235% | -6.5% | 10 |

### 4-hour bars

| Strategy | Folds beating B&H | Strategy return | Buy & hold | Worst drawdown | Trades |
|---|---|---|---|---|---|
| EMA crossover | 1 / 14 | **-98.4%** | +1,646% | -66.0% | 1,927 |
| RSI reversion | 3 / 14 | **-98.0%** | +1,646% | -47.5% | 529 |
| Trend-filtered RSI | 5 / 14 | **-45.2%** | +1,646% | -15.0% | 102 |

Reproduce with `python run_walkforward.py --timeframe 1d`.

## Why that is the interesting result

The first version of this project reported large positive returns. Those numbers
were not real, and the reason is the point of the whole repo.

**Parameters were tuned on the same data they were scored on.** Pick the best
EMA pair across the full nine years and of course it looks profitable, because
the choice already used the answer. Walk-forward validation removes that: each
fold selects parameters using only the training window, then scores on the next
window it has never seen. The returns collapse.

**Fees compound faster than edge.** At 0.1% per side, the EMA crossover pays
0.2% per round trip and takes 1,927 trades on 4-hour bars. That is roughly 385%
in cumulative fees. Even a strategy that is right more often than not dies to
transaction costs at that frequency.

**Trading less loses less.** The clearest pattern in both tables is that the
strategy with the fewest trades performs best, and the ranking by trade count is
exactly the ranking by return. The trend filter cut EMA crossover's 1,927 trades
to 102 and its loss from -98.4% to -45.2%.

**Bitcoin rose roughly 18x over the test period.** Any strategy that is out of
the market part of the time starts from behind. Beating buy-and-hold in a
sustained bull market is a much harder bar than making money, which is why
buy-and-hold is the baseline here rather than zero.

A backtest that does not report a baseline is not reporting anything.

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
- A strategy with an actual hypothesis behind it. EMA crossovers and RSI
  thresholds are the null hypothesis, and the null hypothesis held.
