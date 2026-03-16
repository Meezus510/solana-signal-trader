"""
scripts/trades.py — Print and filter the trade event log by strategy.

Reads trades.log (or the DB trades table) and prints every event for the
requested strategy.  Safe to run while the bot is live.

Usage:
    python scripts/trades.py                          # all strategies
    python scripts/trades.py quick_pop                # one strategy
    python scripts/trades.py quick_pop --db           # read from DB instead of log
    python scripts/trades.py quick_pop --today        # today only
"""

import argparse
import os
import sqlite3
import sys
from datetime import datetime, timezone

SEP = "=" * 90
LOG_PATH = os.getenv("TRADES_LOG_PATH", "trades.log")
DB_PATH  = os.getenv("DB_PATH", "trader.db")

VALID_STRATEGIES = [
    "quick_pop",
    "quick_pop_chart",
    "trend_rider",
    "trend_rider_chart",
    "infinite_moonbag",
    "infinite_moonbag_chart",
]


# ---------------------------------------------------------------------------
# Log file reader
# ---------------------------------------------------------------------------

def print_from_log(strategy: str | None, today_only: bool) -> None:
    if not os.path.exists(LOG_PATH):
        print(f"[ERROR] trades.log not found: {LOG_PATH}", file=sys.stderr)
        sys.exit(1)

    today_prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d") if today_only else None
    filter_tag   = f"strategy={strategy}" if strategy else None

    print(SEP)
    label = strategy or "all strategies"
    print(f"  TRADE LOG — {label}" + (" (today)" if today_only else ""))
    print(SEP)

    count = 0
    with open(LOG_PATH) as f:
        for line in f:
            line = line.rstrip()
            if today_prefix and not line.startswith(today_prefix):
                continue
            # exact match: "strategy=quick_pop" must not partially match "strategy=quick_pop_chart"
            if filter_tag and not _exact_strategy_match(line, strategy):
                continue
            print(f"  {line}")
            count += 1

    print(SEP)
    print(f"  {count} event(s)")
    print(SEP)


def _exact_strategy_match(line: str, strategy: str) -> bool:
    """Match 'strategy=<name>' at end of line to avoid partial matches."""
    return line.endswith(f"strategy={strategy}")


# ---------------------------------------------------------------------------
# DB reader (alternative — more structured output)
# ---------------------------------------------------------------------------

def print_from_db(strategy: str | None, today_only: bool) -> None:
    if not os.path.exists(DB_PATH):
        print(f"[ERROR] Database not found: {DB_PATH}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)

    where_clauses = []
    params: list = []
    if strategy:
        where_clauses.append("strategy = ?")
        params.append(strategy)
    if today_only:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        where_clauses.append("ts LIKE ?")
        params.append(f"{today}%")

    where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    rows = conn.execute(
        f"""
        SELECT ts, event, symbol, mint, price, quantity, pnl, strategy
          FROM trades
          {where}
         ORDER BY ts ASC
        """,
        params,
    ).fetchall()
    conn.close()

    print(SEP)
    label = strategy or "all strategies"
    print(f"  TRADE LOG (DB) — {label}" + (" (today)" if today_only else ""))
    print(SEP)

    for ts, event, symbol, mint, price, qty, pnl, strat in rows:
        sign = "+" if pnl >= 0 else ""
        print(
            f"  {ts[:19]} | {event:<18} | {symbol:<12} | {mint:<46} "
            f"| price=${price:<14.8f} | qty={qty:<14.4f} | pnl=${sign}{pnl:+.4f} | strategy={strat}"
        )

    print(SEP)
    print(f"  {len(rows)} event(s)")
    print(SEP)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Filter and print trade events by strategy.")
    parser.add_argument(
        "strategy",
        nargs="?",
        default=None,
        metavar="STRATEGY",
        help=f"Strategy name to filter. One of: {', '.join(VALID_STRATEGIES)}",
    )
    parser.add_argument(
        "--db",
        action="store_true",
        help="Read from trader.db instead of trades.log",
    )
    parser.add_argument(
        "--today",
        action="store_true",
        help="Show only today's events (UTC)",
    )
    args = parser.parse_args()

    if args.strategy and args.strategy not in VALID_STRATEGIES:
        print(f"[ERROR] Unknown strategy '{args.strategy}'", file=sys.stderr)
        print(f"  Valid options: {', '.join(VALID_STRATEGIES)}", file=sys.stderr)
        sys.exit(1)

    if args.db:
        print_from_db(args.strategy, args.today)
    else:
        print_from_log(args.strategy, args.today)


if __name__ == "__main__":
    main()
