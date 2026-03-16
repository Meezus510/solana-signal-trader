"""
scripts/backfill_snapshots.py — Seed ML training data from historical quick_pop trades.

For each completed quick_pop position in trader.db, fetches the OHLCV candles
as they looked at entry time (using Birdeye's time_to param) and saves them as
chart_snapshots so ChartMLScorer has training data immediately instead of
waiting for live quick_pop_chart trades to accumulate.

How it works
------------
1. Load all CLOSED quick_pop positions from trader.db.
2. For each, fetch 40 × 15s candles + 20 × 1m candles at the entry timestamp.
3. Save a chart_snapshot with strategy='quick_pop_chart' and the known outcome:
     - pnl_pct       realized_pnl / usd_size × 100
     - sell_reason   why the position closed (STOP_LOSS, TP1, TRAILING_STOP …)
     - hold_secs     seconds from entry to close
     - max_gain_pct  (highest_price / entry_price − 1) × 100

Usage
-----
    source venv/bin/activate
    python scripts/backfill_snapshots.py            # backfill all
    python scripts/backfill_snapshots.py --dry-run  # preview without saving
    python scripts/backfill_snapshots.py --delay 1.5  # slower, avoid rate limits
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

# Strategy whose closed positions we read from
SOURCE_STRATEGY = "quick_pop"
# Strategy we save snapshots under (what the ML scorer queries)
TARGET_STRATEGY = "quick_pop_chart_ml"


# ---------------------------------------------------------------------------
# Load completed positions from DB
# ---------------------------------------------------------------------------

def load_closed_positions(db_path: str) -> list[dict]:
    """Return all CLOSED quick_pop positions with enough data to backfill."""
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
         ORDER BY opened_at ASC
        """,
        (SOURCE_STRATEGY,),
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


def already_backfilled(db_path: str, mint: str, entry_unix: int) -> bool:
    """True if a snapshot for this exact (mint, entry time ±60s) already exists."""
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        """
        SELECT id FROM chart_snapshots
         WHERE strategy = ? AND mint = ?
           AND ABS(CAST(strftime('%s', ts) AS INTEGER) - ?) <= 60
         LIMIT 1
        """,
        (TARGET_STRATEGY, mint, entry_unix),
    ).fetchone()
    conn.close()
    return row is not None


# ---------------------------------------------------------------------------
# Main async backfill loop
# ---------------------------------------------------------------------------

async def run(args: argparse.Namespace, cfg: Config) -> None:
    positions = load_closed_positions(DB_PATH)
    if not positions:
        print(f"[WARN] No closed {SOURCE_STRATEGY} positions found in {DB_PATH}")
        return

    print(f"Found {len(positions)} closed {SOURCE_STRATEGY} position(s) to process.")
    if args.dry_run:
        print("[DRY RUN] No data will be saved.\n")

    db = TradeDatabase(path=DB_PATH) if not args.dry_run else None

    saved = skipped_no_data = skipped_exists = 0

    async with aiohttp.ClientSession() as session:
        birdeye = BirdeyePriceClient(cfg, session)

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

            # Skip if already backfilled
            if already_backfilled(DB_PATH, mint, entry_unix):
                print(f"    [SKIP] snapshot already exists")
                skipped_exists += 1
                continue

            # --- Fetch 15s ML candles at entry time ---
            ml_candles = await birdeye.get_ohlcv(
                mint,
                bars=ML_OHLCV_BARS,
                interval=ML_OHLCV_INTERVAL,
                time_to=entry_unix,
            )

            # --- Fetch 1m chart-filter candles at entry time ---
            chart_candles = await birdeye.get_ohlcv(
                mint,
                bars=OHLCV_BARS,
                interval="1m",
                time_to=entry_unix,
            )

            if not ml_candles:
                print(f"    [SKIP] no 15s candle data (token may be delisted)")
                skipped_no_data += 1
                await asyncio.sleep(args.delay)
                continue

            chart_ctx = (
                compute_chart_context(chart_candles, pos["entry_price"])
                if chart_candles else None
            )

            snapshot_id = db.save_chart_snapshot(
                strategy=TARGET_STRATEGY,
                symbol=symbol,
                mint=mint,
                entry_price=pos["entry_price"],
                candles=ml_candles,
                chart_ctx=chart_ctx,
                entered=True,
                ml_score=None,
            )

            db.update_chart_snapshot_outcome(
                snapshot_id,
                pnl_pct=pnl_pct,
                sell_reason=pos["sell_reason"],
                hold_secs=pos["hold_secs"],
                max_gain_pct=pos["max_gain_pct"],
            )

            print(f"    [SAVED] id={snapshot_id} | 15s_bars={len(ml_candles)}"
                  + (f" | 1m_bars={len(chart_candles)}" if chart_candles else ""))
            saved += 1

            await asyncio.sleep(args.delay)

    print()
    print(SEP)
    if args.dry_run:
        print(f"  DRY RUN complete — {len(positions)} position(s) previewed, nothing saved")
    else:
        print(f"  Saved   : {saved}")
        print(f"  Skipped (already exists) : {skipped_exists}")
        print(f"  Skipped (no candle data) : {skipped_no_data}")
        total_snapshots = saved + skipped_exists
        print(f"  Total chart_snapshots for {TARGET_STRATEGY}: {total_snapshots}")
    print(SEP)

    if db:
        db.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill ML training snapshots from historical quick_pop trades."
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
