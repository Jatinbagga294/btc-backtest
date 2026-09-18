"""Tests for the backtest engine and the validation layer."""
import numpy as np
import pandas as pd
import pytest

from backtester.backtester import Backtester
from backtester.baselines import buy_and_hold, max_drawdown, sharpe
from backtester.validation import rolling_windows, summarise


def make_df(closes, start="2020-01-01", freq="1D"):
    idx = pd.date_range(start, periods=len(closes), freq=freq)
    c = np.array(closes, dtype=float)
    return pd.DataFrame(
        {"open": c, "high": c * 1.02, "low": c * 0.98, "close": c,
         "ema_9": c, "ema_21": c, "ema_50": c, "ema_100": c, "ema_200": c,
         "rsi": np.full(len(c), 50.0)},
        index=idx,
    )


# --- baselines -------------------------------------------------------------

def test_buy_and_hold_on_a_doubling_market():
    r = buy_and_hold(make_df([100, 150, 200]), initial_capital=10_000, fee=0)
    assert r["total_return_pct"] == pytest.approx(100, abs=0.01)


def test_buy_and_hold_charges_fees():
    with_fee = buy_and_hold(make_df([100, 200]), fee=0.001)["total_return_pct"]
    without = buy_and_hold(make_df([100, 200]), fee=0)["total_return_pct"]
    assert with_fee < without


def test_max_drawdown_is_negative_and_correct():
    eq = pd.Series([100, 120, 60, 90], index=pd.date_range("2020-01-01", periods=4))
    # peak 120 -> trough 60 is -50%
    assert max_drawdown(eq) == pytest.approx(-50.0)


def test_max_drawdown_of_a_rising_line_is_zero():
    eq = pd.Series([100, 110, 120], index=pd.date_range("2020-01-01", periods=3))
    assert max_drawdown(eq) == pytest.approx(0.0)


def test_sharpe_of_a_flat_curve_is_zero():
    eq = pd.Series([100.0] * 10, index=pd.date_range("2020-01-01", periods=10))
    assert sharpe(eq) == 0.0


def test_empty_inputs_do_not_crash():
    empty = pd.DataFrame(columns=["close"], index=pd.DatetimeIndex([]))
    assert "error" in buy_and_hold(empty)
    assert max_drawdown(pd.Series(dtype=float)) == 0.0


# --- engine ----------------------------------------------------------------

def test_no_signal_means_no_trades():
    r = Backtester(make_df([100, 110, 120])).run(lambda row: 0)
    assert "error" in r


def test_take_profit_closes_a_winning_long():
    # +4% take profit; a 10% jump must trigger it.
    df = make_df([100, 110, 120])
    r = Backtester(df, fee=0, tp_pct=0.04, sl_pct=0.02).run(lambda row: 1)
    assert r["total_trades"] >= 1
    assert r["trades"].iloc[0]["result"] == "tp"
    assert r["total_return_pct"] > 0


def test_stop_loss_closes_a_losing_long():
    df = make_df([100, 80, 70])
    r = Backtester(df, fee=0, tp_pct=0.04, sl_pct=0.02).run(lambda row: 1)
    assert r["trades"].iloc[0]["result"] == "sl"
    assert r["total_return_pct"] < 0


def test_fees_reduce_return():
    df = make_df([100, 110, 120])
    free = Backtester(df, fee=0).run(lambda row: 1)["total_return_pct"]
    paid = Backtester(df, fee=0.001).run(lambda row: 1)["total_return_pct"]
    assert paid < free


def test_date_filter_restricts_the_window():
    df = make_df([100] * 30 + [200] * 30)
    r = Backtester(df, fee=0).run(lambda row: 1, start_date="2020-01-01",
                                  end_date="2020-01-15")
    assert r["equity_curve"].index.max() <= pd.Timestamp("2020-01-15")


# --- walk-forward ----------------------------------------------------------

def test_windows_never_overlap_train_and_test():
    df = make_df([100] * 1200)
    for tr_s, tr_e, te_s, te_e in rolling_windows(df, train_months=12, test_months=6):
        assert tr_e <= te_s, "training window must end before the test window starts"
        assert tr_s < tr_e and te_s < te_e


def test_anchored_windows_grow_and_sliding_windows_do_not():
    df = make_df([100] * 1500)
    anchored = list(rolling_windows(df, 12, 6, anchored=True))
    sliding = list(rolling_windows(df, 12, 6, anchored=False))
    assert anchored[0][0] == anchored[-1][0], "anchored keeps one fixed start"
    assert sliding[0][0] < sliding[-1][0], "sliding moves its start forward"


def test_summarise_reports_an_error_when_nothing_traded():
    assert "error" in summarise([])
