"""
scripts/backfill_snapshots.py — Seed ML training data from historical base strategy trades.

For each completed position in trader.db, fetches the OHLCV candles as they
looked at entry time (using Birdeye's time_to param) and saves them as
signal_charts + strategy_outcomes so ChartMLScorer has training data
immediately instead of waiting for live chart-filtered trades to accumulate.

Outcomes are saved under the base strategy name (e.g. "quick_pop") because
chart variants (e.g. "quick_pop_managed") are configured to train on their
base strategy's unfiltered outcomes — avoiding selection bias.

How it works
------------
1. Load all CLOSED positions for the source strategy from trader.db.
2. For each, fetch ML candles + 1m candles at the entry timestamp.
3. Save a signal_chart + strategy_outcome with the known outcome:
     - pnl_pct       realized_pnl / usd_size × 100
     - sell_reason   why the position closed (STOP_LOSS, TP1, TRAILING_STOP …)
     - hold_secs     seconds from entry to close
     - max_gain_pct  (highest_price / entry_price − 1) × 100

Usage
-----
    source venv/bin/activate
    python scripts/backfill_snapshots.py                        # backfill all 3 base strategies
    python scripts/backfill_snapshots.py --strategy quick_pop   # one strategy only
    python scripts/backfill_snapshots.py --dry-run              # preview without saving
    python scripts/backfill_snapshots.py --delay 1.5            # slower, avoid rate limits
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sqlite3
import sys
from datetime import datetime, timezone

import aiohttp
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trader.analysis.chart import OHLCV_BARS, compute_chart_context
from trader.analysis.ml_scorer import ML_OHLCV_BARS, ML_OHLCV_INTERVAL
from trader.config import Config
from trader.persistence.database import TradeDatabase
from trader.pricing.birdeye import BirdeyePriceClient

DB_PATH  = os.getenv("DB_PATH", "trader.db")
SEP = "-" * 72

# Base strategies to backfill. Outcomes are saved under the base strategy name
# so chart variants (which set ml_training_strategy to the base) can find them.
BACKFILL_STRATEGIES = ["quick_pop", "trend_rider", "infinite_moonbag"]


# ---------------------------------------------------------------------------
# Load completed positions from DB
# ---------------------------------------------------------------------------

def load_closed_positions(db_path: str, strategy: str) -> list[dict]:
    """Return all CLOSED positions for strategy with enough data to backfill."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT mint, symbol, entry_price, realized_pnl_usd, usd_size,
               sell_reason, opened_at, closed_at, highest_price
          FROM positions
         WHERE strategy = ? AND status = 'CLOSED'
           AND opened_at IS NOT NULL
           AND closed_at IS NOT NULL
           AND realized_pnl_usd IS NOT NULL
           AND opened_at LIKE '20%'
           AND closed_at LIKE '20%'
         ORDER BY opened_at ASC
        """,
        (strategy,),
    ).fetchall()
    conn.close()

    positions = []
    for mint, symbol, entry_price, realized_pnl, usd_size, sell_reason, \
            opened_at, closed_at, highest_price in rows:

        entry_ts = datetime.fromisoformat(opened_at)
        if entry_ts.tzinfo is None:
            entry_ts = entry_ts.replace(tzinfo=timezone.utc)

        closed_ts = datetime.fromisoformat(closed_at)
        if closed_ts.tzinfo is None:
            closed_ts = closed_ts.replace(tzinfo=timezone.utc)

        positions.append({
            "mint":          mint,
            "symbol":        symbol,
            "entry_price":   entry_price,
            "entry_ts":      entry_ts,
            "entry_unix":    int(entry_ts.timestamp()),
            "realized_pnl":  realized_pnl,
            "usd_size":      usd_size or 30.0,
            "sell_reason":   sell_reason or "UNKNOWN",
            "hold_secs":     (closed_ts - entry_ts).total_seconds(),
            "max_gain_pct":  ((highest_price / entry_price) - 1.0) * 100.0
                             if highest_price and entry_price else 0.0,
        })

    return positions


def already_backfilled(db_path: str, strategy: str, mint: str, entry_unix: int) -> bool:
    """True if a strategy_outcomes row for this (strategy, mint, entry time ±60s) already exists."""
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        """
        SELECT so.id
          FROM strategy_outcomes so
          JOIN signal_charts sc ON so.signal_chart_id = sc.id
         WHERE so.strategy = ? AND sc.mint = ?
           AND ABS(CAST(strftime('%s', sc.ts) AS INTEGER) - ?) <= 60
         LIMIT 1
        """,
        (strategy, mint, entry_unix),
    ).fetchone()
    conn.close()
    return row is not None


# ---------------------------------------------------------------------------
# Main async backfill loop
# ---------------------------------------------------------------------------

async def backfill_strategy(
    strategy: str,
    args: argparse.Namespace,
    cfg: Config,
    db,
    session: aiohttp.ClientSession,
) -> tuple[int, int, int]:
    """Backfill one base strategy. Returns (saved, skipped_exists, skipped_no_data)."""
    positions = load_closed_positions(DB_PATH, strategy)
    if not positions:
        print(f"[WARN] No closed {strategy} positions found in {DB_PATH}")
        return 0, 0, 0

    print(f"\n{SEP}")
    print(f"  Strategy : {strategy}  ({len(positions)} closed position(s))")
    print(SEP)

    birdeye = BirdeyePriceClient(cfg, session)
    saved = skipped_no_data = skipped_exists = 0

    for pos in positions:
        symbol     = pos["symbol"]
        mint       = pos["mint"]
        entry_unix = pos["entry_unix"]
        pnl_pct    = pos["realized_pnl"] / pos["usd_size"] * 100.0

        print(
            f"  {symbol:<12} {pos['entry_ts'].strftime('%Y-%m-%d %H:%M')} "
            f"| pnl={pnl_pct:+6.1f}% | {pos['sell_reason']}"
        )

        if args.dry_run:
            continue

        if already_backfilled(DB_PATH, strategy, mint, entry_unix):
            print(f"    [SKIP] already exists")
            skipped_exists += 1
            continue

        # Fetch ML candles at entry time
        ml_candles = await birdeye.get_ohlcv(
            mint, bars=ML_OHLCV_BARS, interval=ML_OHLCV_INTERVAL, time_to=entry_unix,
        )

        # Fetch 1m chart-filter candles at entry time
        chart_candles = await birdeye.get_ohlcv(
            mint, bars=OHLCV_BARS, interval="1m", time_to=entry_unix,
        )

        if not ml_candles:
            print(f"    [SKIP] no candle data (token may be delisted)")
            skipped_no_data += 1
            await asyncio.sleep(args.delay)
            continue

        chart_ctx = (
            compute_chart_context(chart_candles, pos["entry_price"])
            if chart_candles else None
        )

        # Save candles once, then outcome under base strategy name.
        # Pass the original entry timestamp so KNN recency weighting reflects
        # the true age of the trade rather than the backfill run time.
        signal_chart_id = db.save_signal_chart(
            symbol=symbol,
            mint=mint,
            entry_price=pos["entry_price"],
            candles=ml_candles,
            chart_ctx=chart_ctx,
            ml_score=None,
            pair_stats=None,
            candles_1m=chart_candles if chart_candles else None,
            ts=pos["entry_ts"].isoformat(),
        )

        outcome_id = db.save_strategy_outcome(
            signal_chart_id=signal_chart_id,
            strategy=strategy,   # save under base strategy name for unbiased training
            entered=True,
        )

        db.update_strategy_outcome(
            outcome_id,
            pnl_pct=pnl_pct,
            sell_reason=pos["sell_reason"],
            hold_secs=pos["hold_secs"],
            max_gain_pct=pos["max_gain_pct"],
        )

        print(f"    [SAVED] chart_id={signal_chart_id} outcome_id={outcome_id}"
              f" | ml_bars={len(ml_candles)}"
              + (f" | 1m_bars={len(chart_candles)}" if chart_candles else ""))
        saved += 1

        await asyncio.sleep(args.delay)

    return saved, skipped_exists, skipped_no_data


async def run(args: argparse.Namespace, cfg: Config) -> None:
    strategies = (
        [args.strategy] if args.strategy else BACKFILL_STRATEGIES
    )

    if args.dry_run:
        print("[DRY RUN] No data will be saved.")

    db = TradeDatabase(path=DB_PATH) if not args.dry_run else None

    total_saved = total_exists = total_no_data = 0

    async with aiohttp.ClientSession() as session:
        for strategy in strategies:
            saved, exists, no_data = await backfill_strategy(
                strategy, args, cfg, db, session,
            )
            total_saved   += saved
            total_exists  += exists
            total_no_data += no_data

    print(f"\n{SEP}")
    if args.dry_run:
        print("  DRY RUN complete — nothing saved")
    else:
        print(f"  Strategies processed     : {', '.join(strategies)}")
        print(f"  Saved                    : {total_saved}")
        print(f"  Skipped (already exists) : {total_exists}")
        print(f"  Skipped (no candle data) : {total_no_data}")
    print(SEP)

    if db:
        db.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill ML training snapshots from historical base strategy trades."
    )
    parser.add_argument(
        "--strategy",
        choices=BACKFILL_STRATEGIES,
        default=None,
        help="Backfill one strategy only (default: all three base strategies)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview positions without fetching candles or saving to DB",
    )
    parser.add_argument(
        "--delay", type=float, default=1.0, metavar="SECS",
        help="Seconds to wait between Birdeye calls (default: 1.0)",
    )
    args = parser.parse_args()

    if not os.path.exists(DB_PATH):
        print(f"[ERROR] Database not found: {DB_PATH}", file=sys.stderr)
        sys.exit(1)

    try:
        cfg = Config.load()
    except ValueError as exc:
        print(f"[ERROR] Config: {exc}", file=sys.stderr)
        sys.exit(1)

    asyncio.run(run(args, cfg))


if __name__ == "__main__":
    main()
