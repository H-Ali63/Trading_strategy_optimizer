"""Sanity-check script -- run this before pytest to see if your optimize() works.

This runs the hand-worked example from README.md (the "Worked Example" section)
and prints your function's output next to the expected values.

Usage:
    python sanity_check.py

You should see all PASS markers. Any FAIL marker tells you which area of your
implementation needs fixing.
"""

from __future__ import annotations

import math
import sys

import pandas as pd

try:
    from optimizer import optimize
except (ImportError, NotImplementedError) as e:
    print(f"[FAIL] Could not import optimize() from optimizer.py: {e}")
    print("       Make sure you've implemented the function (replaced the raise).")
    sys.exit(1)


def mark(ok: bool) -> str:
    return "[PASS]" if ok else "[FAIL]"


def run_worked_example() -> None:
    """The 4-trade example from README.md, section 'Worked example'."""
    trades = pd.DataFrame(
        [
            {"trade_id": 1, "entry_time": "2024-01-01", "exit_time": "2024-01-02",
             "pnl": 10.0, "mae": 2.0, "mfe": 15.0},
            {"trade_id": 2, "entry_time": "2024-01-03", "exit_time": "2024-01-04",
             "pnl": -8.0, "mae": 10.0, "mfe": 1.0},
            {"trade_id": 3, "entry_time": "2024-01-05", "exit_time": "2024-01-06",
             "pnl":  3.0, "mae":  6.0, "mfe": 20.0},
            {"trade_id": 4, "entry_time": "2024-01-07", "exit_time": "2024-01-08",
             "pnl":  4.0, "mae":  1.0, "mfe":  3.0},
        ]
    )

    print("=" * 64)
    print("WORKED EXAMPLE -- SL=5, TP=12")
    print("=" * 64)
    print()
    print("Input trades:")
    print(trades[["trade_id", "pnl", "mae", "mfe"]].to_string(index=False))
    print()
    print("Expected adjusted PnL per trade:")
    print("  Trade 1: TP fires (mfe=15 >= 12)           ->  +12")
    print("  Trade 2: SL fires (mae=10 >= 5)            ->   -5")
    print("  Trade 3: SL fires (priority, both could)   ->   -5")
    print("  Trade 4: neither fires                     ->   +4")
    print()

    try:
        result = optimize(trades, stop_losses=[5.0], take_profits=[12.0], top_n=1)
    except Exception as e:
        print(f"[FAIL] optimize() raised: {type(e).__name__}: {e}")
        sys.exit(1)

    if not result:
        print("[FAIL] optimize() returned an empty list - expected 1 entry")
        sys.exit(1)

    row = result[0]

    expected_total = 6.0   # 12 - 5 - 5 + 4
    expected_stop = 2
    expected_tp = 1
    # mean=1.5, ddof=0 std=sqrt(((12-1.5)^2 + (-5-1.5)^2 + (-5-1.5)^2 + (4-1.5)^2)/4)
    #        = sqrt((110.25 + 42.25 + 42.25 + 6.25)/4) = sqrt(50.25) ~= 7.0887
    # sharpe = 1.5 / 7.0887 ~= 0.2116
    expected_sharpe_pop = 1.5 / math.sqrt(50.25)        # ddof=0
    expected_sharpe_sample = 1.5 / math.sqrt(50.25 * 4 / 3)  # ddof=1

    print("Output (top result):")
    for k in ("stop_loss", "take_profit", "sharpe", "total_pnl", "stopped_out", "took_profit"):
        v = row.get(k, "<missing>")
        print(f"  {k:<14} = {v}")
    print()

    print("Per-field checks:")
    print(f"  {mark(set(row.keys()) == {'stop_loss','take_profit','sharpe','total_pnl','stopped_out','took_profit'})}  result keys are exactly the 6 required keys")
    print(f"  {mark(math.isclose(row.get('stop_loss', 0), 5.0))}  stop_loss == 5.0")
    print(f"  {mark(math.isclose(row.get('take_profit', 0), 12.0))}  take_profit == 12.0")
    print(f"  {mark(math.isclose(row.get('total_pnl', 0), expected_total))}  total_pnl == 6.0  (12 - 5 - 5 + 4)")
    print(f"  {mark(row.get('stopped_out') == expected_stop)}  stopped_out == 2  (trades 2 and 3)")
    print(f"  {mark(row.get('took_profit') == expected_tp)}  took_profit == 1  (only trade 1 - trade 3 hit SL first)")

    sharpe_ok = (
        math.isclose(row.get("sharpe", 0), expected_sharpe_pop, abs_tol=1e-3)
        or math.isclose(row.get("sharpe", 0), expected_sharpe_sample, abs_tol=1e-3)
    )
    print(f"  {mark(sharpe_ok)}  sharpe ~= {expected_sharpe_pop:.4f} (ddof=0) or {expected_sharpe_sample:.4f} (ddof=1)")
    print()


def run_empty_inputs() -> None:
    print("=" * 64)
    print("EMPTY INPUTS -- should return [] without crashing")
    print("=" * 64)
    print()

    empty_df = pd.DataFrame(columns=["trade_id", "entry_time", "exit_time", "pnl", "mae", "mfe"])
    nonempty_df = pd.DataFrame(
        [{"trade_id": 1, "entry_time": "2024-01-01", "exit_time": "2024-01-02",
          "pnl": 1.0, "mae": 0.5, "mfe": 1.5}]
    )

    try:
        r1 = optimize(empty_df, [5.0], [10.0], top_n=5)
        print(f"  {mark(r1 == [])}  empty DataFrame -> [] (got {r1!r})")
    except Exception as e:
        print(f"  [FAIL]  empty DataFrame raised: {type(e).__name__}: {e}")

    try:
        r2 = optimize(nonempty_df, [], [10.0], top_n=5)
        print(f"  {mark(r2 == [])}  empty stop_losses -> [] (got {r2!r})")
    except Exception as e:
        print(f"  [FAIL]  empty stop_losses raised: {type(e).__name__}: {e}")

    try:
        r3 = optimize(nonempty_df, [5.0], [], top_n=5)
        print(f"  {mark(r3 == [])}  empty take_profits -> [] (got {r3!r})")
    except Exception as e:
        print(f"  [FAIL]  empty take_profits raised: {type(e).__name__}: {e}")
    print()


def run_ordering_check() -> None:
    print("=" * 64)
    print("ORDERING -- top entry must have the highest Sharpe")
    print("=" * 64)
    print()

    df = pd.DataFrame(
        [
            {"trade_id": i, "entry_time": "2024-01-01", "exit_time": "2024-01-02",
             "pnl": (-1.0) ** i * (i + 1) * 0.5, "mae": 2.0 + i * 0.1, "mfe": 2.0 + i * 0.1}
            for i in range(20)
        ]
    )

    try:
        result = optimize(df, stop_losses=[1, 2, 3, 4], take_profits=[1, 2, 3, 4], top_n=8)
    except Exception as e:
        print(f"  [FAIL] optimize() raised: {type(e).__name__}: {e}")
        return

    sharpes = [r["sharpe"] for r in result]
    is_sorted = sharpes == sorted(sharpes, reverse=True)
    print(f"  {mark(is_sorted)}  results sorted by sharpe DESC")
    print(f"      sharpe values: {[round(s, 4) for s in sharpes]}")
    print()


def main() -> None:
    print()
    run_worked_example()
    run_empty_inputs()
    run_ordering_check()
    print("=" * 64)
    print("Done. If you see any [FAIL], fix that area before running pytest.")
    print("When all checks pass, run: pytest tests/ -v")
    print("=" * 64)


if __name__ == "__main__":
    main()
