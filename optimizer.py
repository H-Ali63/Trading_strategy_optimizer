"""Trading strategy optimizer implementation."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


_REQUIRED_NUMERIC_COLUMNS = ("pnl", "mae", "mfe")


def _clean_parameter_values(values: object) -> np.ndarray:
    """Convert parameter inputs to finite, positive Python-float candidates."""
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return np.empty(0, dtype=np.float64)

    if array.ndim == 0:
        array = array.reshape(1)

    return array[np.isfinite(array) & (array > 0.0)]


def _sharpe(adjusted_pnl: np.ndarray) -> float:
    """Compute mean / population std, returning 0.0 for degenerate cases."""
    if adjusted_pnl.size <= 1:
        return 0.0

    std = float(np.std(adjusted_pnl, ddof=0))
    if std == 0.0 or not math.isfinite(std):
        return 0.0

    sharpe = float(np.mean(adjusted_pnl) / std)
    return sharpe if math.isfinite(sharpe) else 0.0


def optimize(
    trades_df: pd.DataFrame,
    stop_losses: list[float],
    take_profits: list[float],
    top_n: int = 5,
) -> list[dict]:
    """Return the top-N (stop_loss, take_profit) combinations by Sharpe.

    Invalid trade rows are ignored: pnl, mae, and mfe must be numeric and
    finite, and mae/mfe must be non-negative. The input DataFrame is never
    mutated.
    """
    try:
        top_limit = int(top_n)
    except (TypeError, ValueError):
        return []

    sl_values = _clean_parameter_values(stop_losses)
    tp_values = _clean_parameter_values(take_profits)

    if not isinstance(trades_df, pd.DataFrame) or trades_df.empty or top_limit <= 0:
        return []

    if any(column not in trades_df.columns for column in _REQUIRED_NUMERIC_COLUMNS):
        return []

    if sl_values.size == 0 or tp_values.size == 0:
        return []

    try:
        values = np.column_stack(
            [
                pd.to_numeric(trades_df[column], errors="coerce").to_numpy(
                    dtype=np.float64,
                    copy=False,
                )
                for column in _REQUIRED_NUMERIC_COLUMNS
            ]
        )
    except (TypeError, ValueError):
        return []
    valid_rows = np.isfinite(values).all(axis=1)
    valid_rows &= values[:, 1] >= 0.0
    valid_rows &= values[:, 2] >= 0.0

    if not bool(valid_rows.any()):
        return []

    clean_values = values[valid_rows]
    pnl = clean_values[:, 0]
    mae = clean_values[:, 1]
    mfe = clean_values[:, 2]

    results: list[dict] = []

    for sl in sl_values:
        sl_float = float(sl)
        sl_hit = mae >= sl_float
        not_sl_hit = ~sl_hit
        stopped_out = int(np.count_nonzero(sl_hit))

        for tp in tp_values:
            tp_float = float(tp)

            # SL has priority, so TP is counted only when SL did not trigger.
            tp_hit = not_sl_hit & (mfe >= tp_float)
            took_profit = int(np.count_nonzero(tp_hit))

            adjusted = pnl.copy()
            adjusted[sl_hit] = -sl_float
            adjusted[tp_hit] = tp_float

            results.append(
                {
                    "stop_loss": sl_float,
                    "take_profit": tp_float,
                    "sharpe": _sharpe(adjusted),
                    "total_pnl": float(np.sum(adjusted)),
                    "stopped_out": stopped_out,
                    "took_profit": took_profit,
                }
            )

    results.sort(
        key=lambda row: (
            -row["sharpe"],
            -row["total_pnl"],
            row["stop_loss"],
            row["take_profit"],
        )
    )

    return results[:top_limit]
