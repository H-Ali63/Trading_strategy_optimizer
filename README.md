# Mini Optimizer — Backend Python Take-Home Assignment

Welcome! This is a **1-2 day coding assignment**. We're looking for clean, working
Python code — not perfection. Submit what you have when time's up.

> **You don't need any finance background.** Everything you need is explained
> below in plain English with examples.

---

## Table of Contents

1. [What you're building (in plain English)](#what-youre-building-in-plain-english)
2. [The trading concepts you need (with examples)](#the-trading-concepts-you-need-with-examples)
3. [The exact rules](#the-exact-rules)
4. [What to implement](#what-to-implement)
5. [Worked example — do this by hand once](#worked-example--do-this-by-hand-once)
6. [Getting started (setup)](#getting-started-setup)
7. [Constraints](#constraints)
8. [Edge cases](#edge-cases)
9. [Submitting](#submitting)
10. [Tips & common pitfalls](#tips--common-pitfalls)
11. [FAQ](#faq)

---

## What you're building (in plain English)

Imagine a trader has a CSV of every trade they made last year — 5,000 trades.
For each trade we know: when it opened, when it closed, how much they made or
lost, the worst point during the trade (when they were losing the most), and
the best point during the trade (when they were winning the most).

The trader wants to know: **"If I had used a stop-loss of $5 and a take-profit
of $10 on every trade, would my results have been better?"** And they want
to try every combination of stop-loss and take-profit values to find the best
pair.

Your job: write a function that takes the trades + a list of stop-loss values
+ a list of take-profit values, and returns the **top 5 combinations** ranked
by how good they would have been (measured by Sharpe ratio — explained below).

That's the whole task. It's a number-crunching problem with some edge cases.

---

## The trading concepts you need (with examples)

You don't need to know finance — just these four ideas:

### 1. PnL (Profit and Loss)

How much money you made (positive) or lost (negative) on a trade.
- `pnl = +10` → made $10
- `pnl = -3` → lost $3

### 2. MAE — Maximum Adverse Excursion

The **deepest underwater** the trade went *during* its lifetime — even if it
recovered later. **Always ≥ 0.**

> Example: You buy at $100. Price dips to $92 (you're temporarily down $8),
> then recovers to $103 (you exit up $3). The trade's `pnl = +3`, but the
> `mae = 8` — because at the worst point, you were $8 in the hole.

### 3. MFE — Maximum Favourable Excursion

The **highest point** the trade reached *during* its lifetime — even if it
came back down. **Always ≥ 0.**

> Example: You buy at $100. Price spikes to $115 (you're temporarily up $15),
> then drops back to $103 by close. The trade's `pnl = +3`, but the
> `mfe = 15` — because at the best point, you were $15 ahead.

### 4. Stop-Loss (SL) and Take-Profit (TP)

Automatic exit rules a trader sets *before* the trade:

- **Stop-Loss (SL):** "If I'm ever down by $5, get me out — limit my loss to $5."
- **Take-Profit (TP):** "If I'm ever up by $10, get me out — lock in $10 of profit."

The whole point of the assignment is asking: *"would adding these rules to
past trades have produced better results?"*

### 5. Sharpe Ratio (just a number — explained)

Sharpe ratio measures **how good a series of profits is, relative to how
volatile it is**. A series of small, steady gains has a high Sharpe. A series
that swings wildly between huge wins and huge losses has a low Sharpe — even
if the average is the same.

The formula for this exercise (simplified — no annualisation, no risk-free rate):

```
sharpe = mean(pnl_series) / std(pnl_series)
```

- Higher = better (the strategy is more consistent).
- If `std == 0` (all trades have the same PnL), return `sharpe = 0.0`. Don't crash. Don't return NaN/inf.
- If there are no trades, return `sharpe = 0.0`.
- Use either population (`ddof=0`) or sample (`ddof=1`) standard deviation — pick one, be consistent.

---

## The exact rules

For each trade, given a candidate `stop_loss` (positive number) and
`take_profit` (positive number):

1. **If `mae >= stop_loss`** → the stop-loss would have triggered. The trade exits at a loss of `stop_loss`. **Adjusted PnL = `-stop_loss`**.
2. **Else if `mfe >= take_profit`** → the take-profit would have triggered. The trade exits at a profit of `take_profit`. **Adjusted PnL = `+take_profit`**.
3. **Otherwise** → neither rule triggered. The trade played out normally. **Adjusted PnL = `pnl`** (the original value).

> **Important:** Stop-loss has priority over take-profit. If a trade could
> have triggered both (it went deep underwater AND high in profit), we
> conservatively assume the stop-loss fired first. This is a simplifying
> assumption — in real markets you'd look at the order of events, but here
> we keep it simple.

You compute the Adjusted PnL for **every trade** under a given `(SL, TP)`
combination, then compute the Sharpe ratio of that series. Repeat for every
combination, return the top 5.

---

## What to implement

A single function in `optimizer.py`. The signature and return shape are fixed
— the tests check them exactly:

```python
import pandas as pd

def optimize(
    trades_df: pd.DataFrame,
    stop_losses: list[float],
    take_profits: list[float],
    top_n: int = 5,
) -> list[dict]:
    """Return the top-N parameter combinations sorted by Sharpe (desc)."""
    ...
```

### Input

- **`trades_df`** — pandas DataFrame with columns:
  | column      | type     | meaning                                                      |
  |-------------|----------|--------------------------------------------------------------|
  | `trade_id`  | int      | unique identifier                                            |
  | `entry_time`| timestamp| when the position was opened                                 |
  | `exit_time` | timestamp| when the position was closed                                 |
  | `pnl`       | float    | realised profit/loss (no SL/TP applied) — can be `+` or `-`  |
  | `mae`       | float    | Maximum Adverse Excursion (always ≥ 0)                       |
  | `mfe`       | float    | Maximum Favourable Excursion (always ≥ 0)                    |

  Physical invariants you can rely on:
  - `mfe >= max(0, pnl)` (the trade reached at least its final gain)
  - `mae >= max(0, -pnl)` (the trade went at least as deep as its final loss)

- **`stop_losses`** — list of positive floats to test, e.g. `[2.0, 5.0, 10.0]`.
- **`take_profits`** — list of positive floats to test, e.g. `[5.0, 10.0, 20.0]`.
- **`top_n`** — how many of the best combinations to return (default 5).

### Output

A list of dicts, each with **exactly** these 6 keys:

```python
[
    {
        "stop_loss":    5.0,    # float — the SL value for this combo
        "take_profit": 10.0,    # float — the TP value for this combo
        "sharpe":       0.234,  # float — Sharpe ratio (mean / std of adjusted PnL)
        "total_pnl":  1234.56,  # float — sum of adjusted PnL across all trades
        "stopped_out":   312,   # int   — how many trades hit the SL
        "took_profit":   587,   # int   — how many trades hit the TP
    },
    ...
]
```

### Ordering rules (must be deterministic)

1. **Primary:** sort by `sharpe` **descending** (best first).
2. **Tie-breaker:** by `total_pnl` **descending**.
3. **Further tie-breaker:** by `(stop_loss, take_profit)` **ascending**.

If the parameter grid has fewer than `top_n` combinations (e.g. 3 SLs × 1 TP = 3 combos, but `top_n=5`), return all of them — don't pad with empty entries.

---

## Worked example — do this by hand once

Before you write any code, **work through this by hand** to make sure you
understand the rules. Then add it as your own pytest test once you've got
the basics working.

### Input — 4 trades

| trade_id | pnl   | mae  | mfe  |
|----------|-------|------|------|
| 1        | +10   | 2    | 15   |
| 2        | -8    | 10   | 1    |
| 3        | +3    | 6    | 20   |
| 4        | +4    | 1    | 3    |

### Apply `stop_loss = 5`, `take_profit = 12`

| trade_id | mae ≥ 5? | mfe ≥ 12? | which fires?       | adjusted pnl |
|----------|---------|----------|---------------------|--------------|
| 1        | no (2)   | yes (15) | TP fires            | **+12**      |
| 2        | yes (10) | no (1)   | SL fires            | **-5**       |
| 3        | yes (6)  | yes (20) | SL fires (priority) | **-5**       |
| 4        | no (1)   | no (3)   | neither             | **+4**       |

### Compute the metrics

- `adjusted = [+12, -5, -5, +4]`
- `total_pnl = 12 - 5 - 5 + 4 = 6`
- `stopped_out = 2` (trades 2 and 3)
- `took_profit = 1` (only trade 1 — trade 3 triggered SL first)
- `mean(adjusted) = 6 / 4 = 1.5`
- `std(adjusted, ddof=0) ≈ 7.05` (work it out yourself!)
- `sharpe ≈ 1.5 / 7.05 ≈ 0.213`

That's one combination. Now imagine doing this for `32 × 32 = 1024`
different `(SL, TP)` pairs in under 30 seconds. **That's the assignment.**

---

## Getting started (setup)

### 1. Install Python 3.10 or newer

Check with: `python --version` — anything 3.10+ works.

### 2. (Recommended) Create a virtual environment

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux/git-bash:
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs `pandas`, `numpy`, and `pytest`.

### 4. Confirm the test runner works

```bash
pytest tests/ -v
```

You should see **5 tests fail** with `NotImplementedError`. That's correct —
you haven't written anything yet. Your job is to make them pass.

### 5. Open `optimizer.py` and replace the `raise NotImplementedError`

Implement `optimize(...)` per the spec.

### 6. As you work, run tests frequently

```bash
pytest tests/ -v                                    # all public tests
pytest tests/test_public.py::test_returns_top_n -v  # one specific test
pytest tests/ -v -x                                 # stop on first failure
```

### 7. Try the worked example as a sanity check

```bash
python sanity_check.py
```

This runs the by-hand example from above and prints your function's output
side-by-side with the expected values. Useful for debugging without running
pytest.

---

## Constraints

- **Performance:** A `32 × 32 = 1024`-trial sweep over the 5000-row CSV must finish in **under 30 seconds** on a normal laptop. The reviewer's hidden test enforces this. If you wrote `for trade in trades_df.iterrows():` you'll fail this test. Use numpy vectorisation (see [Tips](#tips--common-pitfalls) below).
- **No optimisation libraries.** No `optuna`, `scikit-optimize`, `hyperopt`, `bayesian-optimization`, etc. — write the search yourself. `pandas` and `numpy` are fine (and expected).
- **Standard library + `pandas` + `numpy` only** for the optimiser. You can use other libraries for personal testing if you want, but the submission's `optimizer.py` must run with just `requirements.txt`.

---

## Edge cases

These aren't trick questions — they're things a real optimiser deals with
every day. The reviewer's hidden tests will hit them all:

| Case | Expected behaviour |
|------|-------------------|
| Empty trades DataFrame | Return `[]` |
| Empty `stop_losses` or empty `take_profits` | Return `[]` |
| Single trade | Sharpe = 0.0 (std is undefined for one number) |
| All trades are losses | Sharpe is negative — that's fine, just don't crash |
| Some rows have `NaN` in pnl/mae/mfe | Drop those rows, continue with the rest |
| Param grid smaller than `top_n` | Return all combos, don't pad |
| Multiple combos tie on Sharpe | Apply the deterministic tie-breakers (see above) |

---

## Submitting

1. **Add a `NOTES.md`** (about half a page) covering:
   - What approach did you take? (Naive grid search? Smarter search? Why?)
   - Where did you trade readability for speed, or vice versa?
   - **What would you do differently with another day?** (This matters — we'd rather hear "I didn't get to X because Y" than a list of half-finished features.)
2. Make sure all 5 public tests pass: `pytest tests/ -v`
3. Zip the entire folder (keep the structure intact).
4. Email it back. Include your name in the filename: `mini-optimizer-yourname.zip`.

**Partial submissions are fine.** We grade what you turned in. Honest notes
about what you didn't finish are valued.

---

## Tips & common pitfalls

### What "vectorisation" means and why you need it

The naive approach to this problem is **3 nested loops** (over SLs, over TPs,
over trades). With 32 × 32 × 5000 = 5.1 million inner iterations, this is
slow if written as pure Python loops or with `df.iterrows()` — easily 60+
seconds.

**Vectorised** code does the same work using array operations. With numpy
broadcasting, the same calculation runs in milliseconds because the loops
happen in compiled C code under the hood.

Rough relative speeds for this problem:

| Approach | Approx time | Will it pass the perf test? |
|----------|-------------|------------------------------|
| `df.iterrows()` inside 2 loops | 60-120s | No |
| `df.apply()` inside 2 loops | 30-60s | Borderline |
| List comprehensions + `np.array` | 5-15s | Yes |
| Fully vectorised broadcasting | <1s | Yes |

You don't need to be a numpy wizard. Even a "halfway vectorised" solution
(loop over the SL/TP grid, but use numpy arrays for the trade-level math)
will easily pass.

### Common pitfalls

| Bug | What you'll see |
|-----|----------------|
| Forgot SL has priority over TP | `test_sl_tp_applied_correctly` fails (wrong counts) |
| Compared `mae > sl` instead of `mae >= sl` | Off-by-one trade counts on edge values |
| `total_pnl` is summed before applying SL/TP | Returns `pnl.sum()` regardless of SL/TP |
| Sharpe returns `inf` or `NaN` when std=0 | `test_single_trade_degenerate_sharpe` fails |
| Sort ascending instead of descending | `test_ordering_best_sharpe_first` fails — top entry has worst Sharpe |
| Result dict has extra/missing keys | `test_result_structure` fails |
| Returns numpy types instead of Python `float`/`int` | Possible — wrap with `float(...)` / `int(...)` |
| Pads with empty entries when grid is small | `test_grid_smaller_than_top_n_returns_all` fails |


---

## FAQ

**Q: Can I use ChatGPT / Copilot / Claude to help?**
A: We can't stop you, but: (a) we read your code carefully, and AI-generated
solutions usually miss the subtle edge cases (especially the SL-priority and
the determinism rule); (b) your `NOTES.md` will need to discuss tradeoffs you
actually understood; (c) if you use AI, *credit it honestly* in `NOTES.md` and
focus on showing what you customised. Honesty scores better than pretending
you wrote everything from scratch.

**Q: Is the goal to write the fastest possible solution?**
A: No — **correctness first**, then "fast enough" (under 30s for 1024 trials).
A 5-second correct solution beats a 0.1-second solution that fails 3 edge cases.

**Q: What if I think the spec is ambiguous?**
A: Make the most reasonable assumption, write it down in `NOTES.md`, and
move on. Reading specs and resolving ambiguity yourself is part of the job.

**Q: Do I need to add more tests?**
A: Optional, but it's a positive signal if you do. The reviewer will run
both your public tests and a private hidden test set covering edge cases.

**Q: Can I refactor `optimizer.py` into multiple files?**
A: Yes. Add helper modules if it makes the code cleaner. The public entry
point `optimize()` in `optimizer.py` must stay where it is.

**Q: Should I write a CLI?**
A: No. We only call `optimize()` programmatically from the tests. Don't
spend time on argparse / typer / etc.

**Q: How will I know if my solution is "good enough" to submit?**
A: All 5 public tests pass + your `NOTES.md` honestly describes what you
did. That's a complete submission.

**Q: What if pytest doesn't find `optimizer.py`?**
A: Run pytest from the **candidate root folder** (the one containing
`optimizer.py`), not from inside `tests/`. The tests `import optimizer`,
which only works if Python's cwd contains it.

---

Good luck — have fun with it. We're rooting for you.
