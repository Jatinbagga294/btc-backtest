"""Walk-forward validation.

The failure mode this exists to prevent: tune a strategy's parameters on the
whole dataset, report the result, and call it a 90% win rate. That number is
meaningless, because the parameters already saw every bar they are being scored
on.

Walk-forward splits the timeline into consecutive blocks and always tests on a
window the optimiser has not seen:

    train 2018 -> test 2019
    train 2018-2019 -> test 2020
    train 2018-2020 -> test 2021
    ...

Only the test windows count. The headline number is the concatenation of
out-of-sample results, which is the closest a backtest gets to honest.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .backtester import Backtester
from .baselines import buy_and_hold


@dataclass
class Fold:
    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    params: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    baseline: dict[str, Any] = field(default_factory=dict)

    @property
    def beat_baseline(self) -> bool:
        return self.result.get("total_return_pct", 0) > self.baseline.get("total_return_pct", 0)


def rolling_windows(
    df: pd.DataFrame, train_months: int = 12, test_months: int = 6, anchored: bool = True
) -> Iterator[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Yield (train_start, train_end, test_start, test_end) tuples.

    anchored=True grows the training window from a fixed start (more data each
    fold). anchored=False slides a fixed-width window, which adapts faster to
    regime change but trains on less.
    """
    start = df.index.min()
    end = df.index.max()

    train_start = start
    train_end = train_start + pd.DateOffset(months=train_months)

    while train_end + pd.DateOffset(months=test_months) <= end:
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)
        yield train_start, train_end, test_start, test_end

        train_end = test_end
        if not anchored:
            train_start = train_end - pd.DateOffset(months=train_months)


def walk_forward(
    df: pd.DataFrame,
    strategy_factory: Callable[[dict[str, Any]], Callable],
    param_grid: list[dict[str, Any]],
    train_months: int = 12,
    test_months: int = 6,
    anchored: bool = True,
    initial_capital: float = 10_000,
    fee: float = 0.001,
) -> list[Fold]:
    """Fit parameters on each training window, score on the next unseen window.

    strategy_factory takes a params dict and returns a signal function, so the
    same code path handles a fixed strategy (one entry in param_grid) and a
    tuned one (many).
    """
    folds: list[Fold] = []

    for i, (tr_s, tr_e, te_s, te_e) in enumerate(
        rolling_windows(df, train_months, test_months, anchored)
    ):
        train = df[(df.index >= tr_s) & (df.index < tr_e)]
        test = df[(df.index >= te_s) & (df.index < te_e)]
        if train.empty or test.empty:
            continue

        # --- fit: pick the best params on the TRAINING window only -----------
        best_params, best_score = param_grid[0], float("-inf")
        for params in param_grid:
            res = Backtester(train, initial_capital, fee).run(strategy_factory(params))
            score = (float("-inf") if "error" in res
                     else res.get("total_return_pct", float("-inf")))
            if score > best_score:
                best_params, best_score = params, score

        # --- score: run those params on the untouched TEST window ------------
        result = Backtester(test, initial_capital, fee).run(strategy_factory(best_params))
        folds.append(
            Fold(
                index=i,
                train_start=tr_s, train_end=tr_e,
                test_start=te_s, test_end=te_e,
                params=best_params,
                result=result,
                baseline=buy_and_hold(test, initial_capital, fee),
            )
        )

    return folds


def summarise(folds: list[Fold]) -> dict[str, Any]:
    """Aggregate the out-of-sample folds into one honest verdict."""
    scored = [f for f in folds if "error" not in f.result]
    if not scored:
        return {"error": "no fold produced any trades"}

    strat_returns = [f.result["total_return_pct"] for f in scored]
    base_returns = [f.baseline["total_return_pct"] for f in scored]
    wins = sum(f.beat_baseline for f in scored)

    # Compounding each fold's return gives the equity path of actually trading
    # this thing fold after fold, which is the number that matters.
    compounded = 1.0
    for r in strat_returns:
        compounded *= (1 + r / 100)
    base_compounded = 1.0
    for r in base_returns:
        base_compounded *= (1 + r / 100)

    return {
        "folds": len(scored),
        "folds_beating_baseline": wins,
        "beat_rate_pct": round(wins / len(scored) * 100, 1),
        "mean_fold_return_pct": round(sum(strat_returns) / len(strat_returns), 2),
        "mean_baseline_return_pct": round(sum(base_returns) / len(base_returns), 2),
        "compounded_return_pct": round((compounded - 1) * 100, 2),
        "baseline_compounded_return_pct": round((base_compounded - 1) * 100, 2),
        "total_trades": sum(f.result.get("total_trades", 0) for f in scored),
        "worst_fold_return_pct": round(min(strat_returns), 2),
        "worst_drawdown_pct": round(min(f.result.get("max_drawdown_pct", 0) for f in scored), 2),
    }
