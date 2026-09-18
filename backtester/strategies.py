"""Signal functions.

A strategy is any callable that takes one row and returns 1 (go long),
-1 (go short) or 0 (do nothing). Keeping the interface this small is what lets
the backtester, the walk-forward validator and the tests all share one code
path.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any


def ema_crossover(params: dict[str, Any]) -> Callable:
    """Long when the fast EMA is above the slow one, short when below.

    The oldest trend-following idea there is, included as a reference point.
    """
    fast = params.get("fast", "ema_9")
    slow = params.get("slow", "ema_21")

    def signal(row) -> int:
        if row[fast] > row[slow]:
            return 1
        if row[fast] < row[slow]:
            return -1
        return 0

    return signal


def rsi_reversion(params: dict[str, Any]) -> Callable:
    """Buy oversold, sell overbought. Mean reversion."""
    low = params.get("oversold", 30)
    high = params.get("overbought", 70)

    def signal(row) -> int:
        rsi = row["rsi"]
        if rsi < low:
            return 1
        if rsi > high:
            return -1
        return 0

    return signal


def trend_filtered_rsi(params: dict[str, Any]) -> Callable:
    """RSI reversion, but only in the direction of the longer trend.

    Pure mean reversion gets destroyed in a strong trend, since it keeps
    shorting a market that keeps going up. This only takes the reversion trade
    when it agrees with price relative to the 200 EMA.
    """
    low = params.get("oversold", 30)
    high = params.get("overbought", 70)
    trend = params.get("trend", "ema_200")

    def signal(row) -> int:
        uptrend = row["close"] > row[trend]
        if uptrend and row["rsi"] < low:
            return 1
        if not uptrend and row["rsi"] > high:
            return -1
        return 0

    return signal


REGISTRY = {
    "ema_crossover": ema_crossover,
    "rsi_reversion": rsi_reversion,
    "trend_filtered_rsi": trend_filtered_rsi,
}

GRIDS = {
    "ema_crossover": [
        {"fast": "ema_9", "slow": "ema_21"},
        {"fast": "ema_9", "slow": "ema_50"},
        {"fast": "ema_21", "slow": "ema_50"},
        {"fast": "ema_50", "slow": "ema_200"},
    ],
    "rsi_reversion": [
        {"oversold": 20, "overbought": 80},
        {"oversold": 30, "overbought": 70},
        {"oversold": 35, "overbought": 65},
    ],
    "trend_filtered_rsi": [
        {"oversold": 20, "overbought": 80, "trend": "ema_200"},
        {"oversold": 30, "overbought": 70, "trend": "ema_200"},
        {"oversold": 30, "overbought": 70, "trend": "ema_100"},
    ],
}
