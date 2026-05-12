"""Public tests — visible to the candidate.

These cover the basic happy path. Hidden tests run by the reviewer cover
edge cases and performance.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from optimizer import optimize


DATA_PATH = Path(__file__).parent.parent / "data" / "trades.csv"


@pytest.fixture(scope="module")
def trades() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, parse_dates=["entry_time", "exit_time"])


def _toy_trades() -> pd.DataFrame:
    # 4 hand-crafted trades for predictable assertions.
    return pd.DataFrame(
        [
            # winner: pnl=10, MFE=15, MAE=2  -> TP=12 fires (mfe>=12), adjusted=+12
            {"trade_id": 1, "entry_time": "2024-01-01", "exit_time": "2024-01-02",
             "pnl": 10.0, "mae": 2.0, "mfe": 15.0},
            # loser: pnl=-8, MFE=1, MAE=10  -> SL=5 fires (mae>=5), adjusted=-5
            {"trade_id": 2, "entry_time": "2024-01-03", "exit_time": "2024-01-04",
             "pnl": -8.0, "mae": 10.0, "mfe": 1.0},
            # both could fire (mae>=SL AND mfe>=TP) -> SL has priority, adjusted=-5
            {"trade_id": 3, "entry_time": "2024-01-05", "exit_time": "2024-01-06",
             "pnl": 3.0, "mae": 6.0, "mfe": 20.0},
            # neither fires -> adjusted = pnl = +4
            {"trade_id": 4, "entry_time": "2024-01-07", "exit_time": "2024-01-08",
             "pnl": 4.0, "mae": 1.0, "mfe": 3.0},
        ]
    )


# ---------------------------------------------------------------------------
# 1. Result structure
# ---------------------------------------------------------------------------

def test_result_structure(trades: pd.DataFrame) -> None:
    """Every result dict has exactly the required keys with correct types."""
    result = optimize(trades, stop_losses=[5, 10, 15], take_profits=[5, 10, 15], top_n=5)

    assert isinstance(result, list)
    assert len(result) > 0

    required_keys = {"stop_loss", "take_profit", "sharpe", "total_pnl", "stopped_out", "took_profit"}
    for row in result:
        assert isinstance(row, dict)
        assert set(row.keys()) == required_keys, f"unexpected keys: {set(row.keys()) ^ required_keys}"
        assert isinstance(row["stop_loss"], (int, float))
        assert isinstance(row["take_profit"], (int, float))
        assert isinstance(row["sharpe"], float)
        assert isinstance(row["total_pnl"], (int, float))
        assert isinstance(row["stopped_out"], int)
        assert isinstance(row["took_profit"], int)


# ---------------------------------------------------------------------------
# 2. Returns top-N entries
# ---------------------------------------------------------------------------

def test_returns_top_n(trades: pd.DataFrame) -> None:
    sls = [5, 10, 15, 20]
    tps = [5, 10, 15, 20]

    result_5 = optimize(trades, sls, tps, top_n=5)
    result_3 = optimize(trades, sls, tps, top_n=3)

    assert len(result_5) == 5
    assert len(result_3) == 3


# ---------------------------------------------------------------------------
# 3. Loose SL/TP -> all trades unchanged
# ---------------------------------------------------------------------------

def test_loose_params_leave_trades_unchanged() -> None:
    """If SL and TP are larger than any MAE/MFE, no trade is modified."""
    df = _toy_trades()
    # MAE max is 10, MFE max is 20 -> SL=1000, TP=1000 should fire on nothing.
    result = optimize(df, stop_losses=[1000.0], take_profits=[1000.0], top_n=5)

    assert len(result) == 1
    row = result[0]
    assert row["stopped_out"] == 0
    assert row["took_profit"] == 0
    assert math.isclose(row["total_pnl"], df["pnl"].sum())


# ---------------------------------------------------------------------------
# 4. Manual SL/TP arithmetic
# ---------------------------------------------------------------------------

def test_sl_tp_applied_correctly() -> None:
    """Hand-calculate the adjusted PnL with SL=5, TP=12 on the toy trades."""
    df = _toy_trades()
    result = optimize(df, stop_losses=[5.0], take_profits=[12.0], top_n=1)
    row = result[0]

    # Trade 1: TP fires -> +12
    # Trade 2: SL fires -> -5
    # Trade 3: SL fires (priority over TP) -> -5
    # Trade 4: neither -> +4
    expected_total_pnl = 12 - 5 - 5 + 4  # = 6
    expected_stopped = 2
    expected_took_profit = 1  # only trade 1 (trade 3 is SL because of priority)

    assert math.isclose(row["total_pnl"], expected_total_pnl)
    assert row["stopped_out"] == expected_stopped
    assert row["took_profit"] == expected_took_profit


# ---------------------------------------------------------------------------
# 5. Ordering: top result has the best Sharpe
# ---------------------------------------------------------------------------

def test_ordering_best_sharpe_first(trades: pd.DataFrame) -> None:
    """Top entry must have the highest Sharpe in the returned list."""
    result = optimize(trades, stop_losses=[5, 10, 15], take_profits=[5, 10, 15], top_n=9)

    sharpes = [row["sharpe"] for row in result]
    assert sharpes == sorted(sharpes, reverse=True), f"not sorted desc: {sharpes}"
