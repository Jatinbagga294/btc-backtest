"""Run walk-forward validation on every strategy and print the results table.

    python run_walkforward.py --timeframe 1d
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from backtester.data import load
from backtester.strategies import GRIDS, REGISTRY
from backtester.validation import summarise, walk_forward

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeframe", default="1d")
    ap.add_argument("--train-months", type=int, default=12)
    ap.add_argument("--test-months", type=int, default=6)
    ap.add_argument("--sliding", action="store_true",
                    help="slide a fixed training window instead of growing it")
    args = ap.parse_args()

    df = load(args.timeframe)
    print(f"{len(df):,} bars  {df.index.min().date()} to {df.index.max().date()}  "
          f"({args.timeframe})")
    print(f"train {args.train_months}m -> test {args.test_months}m, "
          f"{'sliding' if args.sliding else 'anchored'}\n")

    rows = []
    for name, factory in REGISTRY.items():
        folds = walk_forward(
            df, factory, GRIDS[name],
            train_months=args.train_months,
            test_months=args.test_months,
            anchored=not args.sliding,
        )
        s = summarise(folds)
        if "error" in s:
            print(f"  {name:<20} {s['error']}")
            continue
        s["strategy"] = name
        rows.append(s)

    if not rows:
        print("No strategy produced a tradeable result.")
        return 1

    hdr = (f"{'strategy':<20}{'folds':>6}{'beat b&h':>10}{'strat %':>12}"
           f"{'b&h %':>12}{'worst DD':>10}{'trades':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['strategy']:<20}{r['folds']:>6}"
              f"{r['folds_beating_baseline']:>4}/{r['folds']:<5}"
              f"{r['compounded_return_pct']:>12,.1f}"
              f"{r['baseline_compounded_return_pct']:>12,.1f}"
              f"{r['worst_drawdown_pct']:>10.1f}"
              f"{r['total_trades']:>8}")

    out = Path("results") / f"walkforward_{args.timeframe}.json"
    out.parent.mkdir(exist_ok=True)
    # numpy int64/float64 leak in from pandas sums and json cannot encode them.
    out.write_text(json.dumps(rows, indent=2, default=float), encoding="utf-8")
    print(f"\nwritten to {out}")

    best = max(rows, key=lambda r: r["compounded_return_pct"])
    bh = best["baseline_compounded_return_pct"]
    print(f"\nBest out-of-sample: {best['strategy']} at "
          f"{best['compounded_return_pct']:,.1f}% vs buy-and-hold {bh:,.1f}%.")
    print("Beating buy-and-hold" if best["compounded_return_pct"] > bh
          else "None of these beat buy-and-hold out of sample.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
