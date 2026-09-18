"""Baselines a strategy has to beat to be worth anything.

A backtest that reports "+180% return" means nothing on its own. Bitcoin rose
roughly 100x over this period, so a strategy can look spectacular while being
far worse than doing nothing. Every result in this project is reported against
buy-and-hold over the identical window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def buy_and_hold(df: pd.DataFrame, initial_capital: float = 10_000,
                 fee: float = 0.001) -> dict:
    """Buy at the first bar, hold to the last. One entry fee, one exit fee.

    Deliberately charged the same fees as a traded strategy so the comparison is
    like for like.
    """
    if df.empty:
        return {"error": "empty window"}

    entry = df.iloc[0]["close"] * (1 + fee)
    exit_ = df.iloc[-1]["close"] * (1 - fee)
    units = initial_capital / entry
    final = units * exit_

    equity = pd.DataFrame(
        {"equity": units * df["close"]},
        index=df.index,
    )
    return {
        "strategy": "buy_and_hold",
        "total_return_pct": round((final - initial_capital) / initial_capital * 100, 2),
        "final_capital": round(final, 2),
        "max_drawdown_pct": round(max_drawdown(equity["equity"]), 2),
        "sharpe_ratio": round(sharpe(equity["equity"]), 2),
        "total_trades": 1,
        "equity_curve": equity,
    }


def max_drawdown(equity: pd.Series) -> float:
    """Worst peak-to-trough decline, as a negative percentage."""
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    return float(((equity - peak) / peak * 100).min())


def sharpe(equity: pd.Series, periods_per_year: int = 365) -> float:
    """Annualised Sharpe on daily returns, risk-free rate assumed zero.

    Zero is the standard simplification for crypto backtests, and it flatters
    the strategy slightly. Worth remembering when reading the number.
    """
    if equity.empty:
        return 0.0
    daily = equity.resample("1D").last().pct_change().dropna()
    if len(daily) < 2 or daily.std() == 0:
        return 0.0
    return float(daily.mean() / daily.std() * np.sqrt(periods_per_year))
