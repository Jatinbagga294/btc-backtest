import numpy as np
import pandas as pd


class Backtester:
    def __init__(self, df, initial_capital=10000, fee=0.001, sl_pct=0.02, tp_pct=0.04):
        """
        df             : DataFrame with features, indexed by datetime
        initial_capital: starting cash in USDT
        fee            : trading fee per side (0.001 = 0.1%)
        sl_pct         : stop loss % (0.02 = 2%)
        tp_pct         : take profit % (0.04 = 4%)
        """
        self.df = df.copy()
        self.initial_capital = initial_capital
        self.fee = fee
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct

    def run(self, signal_func, start_date=None, end_date=None):
        """
        signal_func: function that takes a row and returns 1 (buy), -1 (sell/short), 0 (no trade)
        start_date / end_date: optional strings like "2022-01-01" to filter the window
        """
        df = self.df.copy()

        # Apply date range filter
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]

        capital = self.initial_capital
        position = 0        # 1 = long, -1 = short, 0 = flat
        entry_price = 0
        entry_time = None
        stop_loss = 0
        take_profit = 0

        trades = []
        equity_curve = []

        for timestamp, row in df.iterrows():
            current_price = row["close"]
            equity_curve.append({"time": timestamp, "equity": capital})

            # Check if we're in a trade — manage exit first
            if position != 0:
                hit_sl = False
                hit_tp = False

                if position == 1:  # Long
                    if row["low"] <= stop_loss:
                        hit_sl = True
                        exit_price = stop_loss
                    elif row["high"] >= take_profit:
                        hit_tp = True
                        exit_price = take_profit

                elif position == -1:  # Short
                    if row["high"] >= stop_loss:
                        hit_sl = True
                        exit_price = stop_loss
                    elif row["low"] <= take_profit:
                        hit_tp = True
                        exit_price = take_profit

                if hit_sl or hit_tp:
                    # Calculate PnL
                    if position == 1:
                        pnl_pct = (exit_price - entry_price) / entry_price
                    else:
                        pnl_pct = (entry_price - exit_price) / entry_price

                    pnl_pct -= 2 * self.fee  # Entry + exit fees
                    pnl = capital * pnl_pct
                    capital += pnl

                    trades.append({
                        "entry_time": entry_time,
                        "exit_time": timestamp,
                        "direction": "long" if position == 1 else "short",
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl": pnl,
                        "pnl_pct": pnl_pct * 100,
                        "result": "sl" if hit_sl else "tp",
                        "capital_after": capital
                    })

                    position = 0
                    continue

            # No position — check for new signal
            if position == 0:
                signal = signal_func(row)

                if signal == 1:  # Long entry
                    position = 1
                    entry_price = current_price * (1 + self.fee)
                    entry_time = timestamp
                    stop_loss = entry_price * (1 - self.sl_pct)
                    take_profit = entry_price * (1 + self.tp_pct)

                elif signal == -1:  # Short entry
                    position = -1
                    entry_price = current_price * (1 - self.fee)
                    entry_time = timestamp
                    stop_loss = entry_price * (1 + self.sl_pct)
                    take_profit = entry_price * (1 - self.tp_pct)

        # Force close any open position at end
        if position != 0:
            exit_price = df.iloc[-1]["close"]
            if position == 1:
                pnl_pct = (exit_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - exit_price) / entry_price
            pnl_pct -= 2 * self.fee
            pnl = capital * pnl_pct
            capital += pnl
            trades.append({
                "entry_time": entry_time,
                "exit_time": df.index[-1],
                "direction": "long" if position == 1 else "short",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "pnl_pct": pnl_pct * 100,
                "result": "open_close",
                "capital_after": capital
            })

        trades_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame(equity_curve).set_index("time")

        return self._analyze(trades_df, equity_df, capital)

    def _analyze(self, trades_df, equity_df, final_capital):
        if trades_df.empty:
            return {"error": "No trades generated"}

        total_trades = len(trades_df)
        winners = trades_df[trades_df["pnl"] > 0]
        losers = trades_df[trades_df["pnl"] <= 0]

        win_rate = len(winners) / total_trades * 100
        avg_win = winners["pnl_pct"].mean() if len(winners) > 0 else 0
        avg_loss = losers["pnl_pct"].mean() if len(losers) > 0 else 0
        # Guard on the SUM, not the count: losers whose pnl sums to zero
        # divide by zero and emit a RuntimeWarning before returning inf.
        gross_loss = abs(losers["pnl"].sum())
        profit_factor = (winners["pnl"].sum() / gross_loss) if gross_loss > 0 else float("inf")

        total_return = (final_capital - self.initial_capital) / self.initial_capital * 100

        # Max drawdown
        equity = equity_df["equity"]
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()

        # Sharpe ratio (simplified, daily)
        daily_returns = equity_df["equity"].resample("1D").last().pct_change().dropna()
        sharpe = (
            (daily_returns.mean() / daily_returns.std()) * np.sqrt(365)
            if daily_returns.std() > 0 else 0
        )

        results = {
            "total_trades": total_trades,
            "win_rate": round(win_rate, 2),
            "avg_win_pct": round(avg_win, 2),
            "avg_loss_pct": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "total_return_pct": round(total_return, 2),
            "final_capital": round(final_capital, 2),
            "max_drawdown_pct": round(max_drawdown, 2),
            "sharpe_ratio": round(sharpe, 2),
            "trades": trades_df,
            "equity_curve": equity_df
        }

        return results


def print_results(results):
    if "error" in results:
        print(f"Error: {results['error']}")
        return

    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    print(f"Total Trades     : {results['total_trades']}")
    print(f"Win Rate         : {results['win_rate']}%")
    print(f"Avg Win          : {results['avg_win_pct']}%")
    print(f"Avg Loss         : {results['avg_loss_pct']}%")
    print(f"Profit Factor    : {results['profit_factor']}")
    print(f"Total Return     : {results['total_return_pct']}%")
    print(f"Final Capital    : ${results['final_capital']}")
    print(f"Max Drawdown     : {results['max_drawdown_pct']}%")
    print(f"Sharpe Ratio     : {results['sharpe_ratio']}")
    print("="*50)