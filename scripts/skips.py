#!/usr/bin/env python3
"""
scripts/skips.py — Show skipped signals and why they were rejected.

Usage:
    python scripts/skips.py                              # all strategies, last 30
    python scripts/skips.py quick_pop_managed           # one strategy
    python scripts/skips.py quick_pop_managed --limit 50
    python scripts/skips.py quick_pop_managed --today
    python scripts/skips.py quick_pop_managed --reason ML_SKIP
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DB_PATH = os.getenv("DB_PATH", "trader.db")
SEP = "=" * 90

VALID_STRATEGIES = [
    "quick_pop",
    "quick_pop_managed",
    "trend_rider",
    "trend_rider_managed",
    "moonbag_managed",
]

VALID_REASONS = ["ML_SKIP", "CHART_SKIP", "POLICY_BLK"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Show skipped signals and skip reasons.")
    parser.add_argument(
        "strategy",
        nargs="?",
        default=None,
        metavar="STRATEGY",
        help=f"Strategy to filter. One of: {', '.join(VALID_STRATEGIES)}",
    )
    parser.add_argument(
        "--limit", type=int, default=30, metavar="N",
        help="Number of recent skips to show (default: 30)",
    )
    parser.add_argument(
        "--today", action="store_true",
        help="Show only today's skips (UTC)",
    )
    parser.add_argument(
        "--reason", default=None, metavar="REASON",
        help=f"Filter by skip reason: {', '.join(VALID_REASONS)}",
    )
    parser.add_argument(
        "--db", default=DB_PATH, metavar="PATH",
        help=f"Path to trader.db (default: {DB_PATH})",
    )
    args = parser.parse_args()

    if args.strategy and args.strategy not in VALID_STRATEGIES:
        print(f"[ERROR] Unknown strategy '{args.strategy}'", file=sys.stderr)
        print(f"  Valid options: {', '.join(VALID_STRATEGIES)}", file=sys.stderr)
        sys.exit(1)

    if args.reason and args.reason not in VALID_REASONS:
        print(f"[ERROR] Unknown reason '{args.reason}'", file=sys.stderr)
        print(f"  Valid options: {', '.join(VALID_REASONS)}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.db):
        print(f"[ERROR] Database not found: {args.db}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(args.db)

    where_parts = ["so.entered = 0"]
    params: list = []

    if args.strategy:
        where_parts.append("so.strategy = ?")
        params.append(args.strategy)
    if args.today:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        where_parts.append("sc.ts LIKE ?")
        params.append(f"{today}%")
    if args.reason:
        where_parts.append("so.skip_reason = ?")
        params.append(args.reason)

    where = "WHERE " + " AND ".join(where_parts)
    params.append(args.limit)

    rows = conn.execute(
        f"""
        SELECT sc.ts, so.strategy, sc.symbol, sc.source_channel,
               so.ml_score, so.skip_reason,
               aod.decision, aod.agent_reason
          FROM strategy_outcomes so
          JOIN signal_charts sc ON so.signal_chart_id = sc.id
          LEFT JOIN ai_override_decisions aod ON aod.signal_chart_id = sc.id
                                              AND aod.strategy = so.strategy
         {where}
         ORDER BY sc.ts DESC
         LIMIT ?
        """,
        params,
    ).fetchall()
    conn.close()

    label = args.strategy or "all strategies"
    suffix = f" ({args.reason})" if args.reason else ""
    suffix += " (today)" if args.today else ""

    print(f"\n{SEP}")
    print(f"  SKIPPED SIGNALS — {label}{suffix}  [{len(rows)} shown]")
    print(SEP)

    if not rows:
        print("  No skipped signals found.")
        print(SEP)
        return

    # Count by reason
    reason_counts: dict[str, int] = {}
    for row in rows:
        r = row[5] or "unknown"
        reason_counts[r] = reason_counts.get(r, 0) + 1

    print(f"  {'ts':16}  {'strategy':26}  {'symbol':12}  {'ch':14}  {'score':6}  {'reason':11}  {'ai_decision':16}  agent_reason")
    print(f"  {'-'*16}  {'-'*26}  {'-'*12}  {'-'*14}  {'-'*6}  {'-'*11}  {'-'*16}  -----------")

    for ts, strategy, symbol, channel, ml_score, skip_reason, ai_decision, agent_reason in rows:
        score_str  = f"{ml_score:.2f}" if ml_score is not None else "  —  "
        reason_str = skip_reason or "unknown"
        ai_str     = ai_decision or "—"
        agent_str  = (agent_reason or "")[:60]
        ch_str     = (channel or "")[:14]
        print(
            f"  {str(ts)[:16]}  {strategy:<26}  {str(symbol):<12}  {ch_str:<14}  "
            f"{score_str:<6}  {reason_str:<11}  {ai_str:<16}  {agent_str}"
        )

    print()
    print(f"  Skip breakdown:")
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"    {reason:<12} {count}")
    print(SEP)


if __name__ == "__main__":
    main()
