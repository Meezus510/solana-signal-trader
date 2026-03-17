#!/usr/bin/env python3
"""
scripts/fix_backfill_timestamps.py — Remove backfill rows with wrong timestamps.

The first backfill run (2026-03-17T09:39–09:41) stamped all training rows with
the time the script ran instead of the original trade entry time.  This breaks
KNN recency weighting because every training row appears to be from the same
moment.

This script:
  1. Shows a preview of the bad rows to be deleted
  2. Shows what live data will be preserved
  3. Deletes the bad rows (with --fix flag)
  4. Prints instructions to re-run backfill with correct timestamps

Usage:
    python scripts/fix_backfill_timestamps.py           # dry-run preview only
    python scripts/fix_backfill_timestamps.py --fix     # actually delete
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

DB_PATH = os.getenv("DB_PATH", "trader.db")
SEP = "=" * 72

# The 2-minute window when the bad backfill ran
BAD_TS_FROM = "2026-03-17T09:39"
BAD_TS_TO   = "2026-03-17T09:41"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove bad-timestamp backfill rows and re-seed training data."
    )
    parser.add_argument(
        "--fix", action="store_true",
        help="Actually delete the bad rows (default: dry-run preview only)",
    )
    args = parser.parse_args()

    if not os.path.exists(DB_PATH):
        print(f"[ERROR] Database not found: {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)

    # ------------------------------------------------------------------
    # 1. Preview bad rows
    # ------------------------------------------------------------------
    bad_charts = conn.execute(
        """
        SELECT sc.id, sc.ts, sc.symbol,
               sc.candle_count,
               ROUND(LENGTH(sc.candles_json) * 1.0 / MAX(sc.candle_count, 1)) AS bpc
          FROM signal_charts sc
         WHERE sc.ts >= ? AND sc.ts <= ?
         ORDER BY sc.ts
        """,
        (BAD_TS_FROM, BAD_TS_TO),
    ).fetchall()

    bad_outcomes = conn.execute(
        """
        SELECT so.id, so.strategy, so.outcome_pnl_pct
          FROM strategy_outcomes so
          JOIN signal_charts sc ON so.signal_chart_id = sc.id
         WHERE sc.ts >= ? AND sc.ts <= ?
        """,
        (BAD_TS_FROM, BAD_TS_TO),
    ).fetchall()

    print(f"\n{SEP}")
    print(f"  BAD ROWS  (ts between {BAD_TS_FROM} and {BAD_TS_TO})")
    print(SEP)

    if not bad_charts:
        print("  No bad rows found — nothing to do.")
        conn.close()
        return

    print(f"  signal_charts to delete    : {len(bad_charts)}")
    print(f"  strategy_outcomes to delete: {len(bad_outcomes)}")
    print()
    print(f"  {'chart_id':8}  {'ts':19}  {'symbol':12}  {'candles':7}  bpc")
    print(f"  {'-'*8}  {'-'*19}  {'-'*12}  {'-'*7}  ---")
    for chart_id, ts, symbol, candle_count, bpc in bad_charts:
        src = "moralis-10s" if bpc and bpc > 400 else "birdeye-1m "
        print(f"  {chart_id:<8}  {str(ts)[:19]}  {str(symbol):<12}  {candle_count or 0:<7}  {src}")

    # ------------------------------------------------------------------
    # 2. Preview live data that will be kept
    # ------------------------------------------------------------------
    kept = conn.execute(
        """
        SELECT COUNT(*),
               SUM(CASE WHEN ROUND(LENGTH(candles_json)*1.0/MAX(candle_count,1)) > 400
                         THEN 1 ELSE 0 END) AS moralis_count,
               MIN(ts), MAX(ts)
          FROM signal_charts
         WHERE ts < ? OR ts > ?
        """,
        (BAD_TS_FROM, BAD_TS_TO),
    ).fetchone()

    total_kept, moralis_kept, first_ts, last_ts = kept

    print(f"\n{SEP}")
    print("  DATA PRESERVED (not touched)")
    print(SEP)
    print(f"  signal_charts rows kept    : {total_kept}")
    print(f"  Of which Moralis-10s       : {moralis_kept}  ← live trading candles")
    if first_ts:
        print(f"  Date range                 : {str(first_ts)[:16]} → {str(last_ts)[:16]}")

    # ------------------------------------------------------------------
    # 3. Delete or show instructions
    # ------------------------------------------------------------------
    if not args.fix:
        print(f"\n{SEP}")
        print("  DRY RUN — nothing deleted")
        print(SEP)
        print("  Re-run with --fix to apply the delete:")
        print("    python scripts/fix_backfill_timestamps.py --fix")
        conn.close()
        return

    print(f"\n{SEP}")
    print("  DELETING BAD ROWS...")
    print(SEP)

    bad_chart_ids = [r[0] for r in bad_charts]
    placeholders = ",".join("?" * len(bad_chart_ids))

    deleted_outcomes = conn.execute(
        f"DELETE FROM strategy_outcomes WHERE signal_chart_id IN ({placeholders})",
        bad_chart_ids,
    ).rowcount

    deleted_charts = conn.execute(
        f"DELETE FROM signal_charts WHERE id IN ({placeholders})",
        bad_chart_ids,
    ).rowcount

    conn.commit()
    conn.close()

    print(f"  Deleted strategy_outcomes  : {deleted_outcomes}")
    print(f"  Deleted signal_charts      : {deleted_charts}")

    print(f"\n{SEP}")
    print("  NEXT STEP — re-run backfill with correct timestamps:")
    print(SEP)
    print("    python scripts/backfill_snapshots.py --strategy quick_pop")
    print()
    print("  Then verify training data:")
    print("    python scripts/check_quick_pop_ml.py")
    print()


if __name__ == "__main__":
    main()
