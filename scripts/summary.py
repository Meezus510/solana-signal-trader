"""
summary.py — Print a PnL + positions snapshot from the live database.

Reads trader.db directly (safe while the bot is running — WAL mode allows
concurrent reads) and reconstructs the same summary that prints on Ctrl+C.

Usage (on the server):
    cd ~/solana-signal-trader
    source venv/bin/activate
    python summary.py

No Birdeye call is made — open positions show last known price from the DB.
"""

import os
import sqlite3
import sys
import logging

from dotenv import load_dotenv

load_dotenv()

from trader.config import Config
from trader.persistence.database import TradeDatabase
from trader.trading.engine import MultiStrategyEngine
from trader.utils.logging import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

SEP = "=" * 72


def print_closed_positions(db_path: str) -> None:
    """Query and print all closed positions directly from the DB."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT strategy, symbol, entry_price, realized_pnl_usd, sell_reason,
               opened_at, closed_at
        FROM positions
        WHERE status = 'CLOSED'
        ORDER BY strategy, closed_at
        """
    ).fetchall()

    totals = conn.execute(
        """
        SELECT strategy,
               COUNT(*) AS total,
               SUM(CASE WHEN realized_pnl_usd > 0 THEN 1 ELSE 0 END) AS wins,
               SUM(realized_pnl_usd) AS pnl
        FROM positions
        WHERE status = 'CLOSED'
        GROUP BY strategy
        ORDER BY strategy
        """
    ).fetchall()

    grand_total = conn.execute(
        """
        SELECT COUNT(*),
               SUM(CASE WHEN realized_pnl_usd > 0 THEN 1 ELSE 0 END),
               SUM(realized_pnl_usd)
        FROM positions
        WHERE status = 'CLOSED'
        """
    ).fetchone()

    conn.close()

    if not rows:
        print(SEP)
        print("  CLOSED POSITIONS — none yet")
        print(SEP)
        return

    print(SEP)
    print("  CLOSED POSITIONS (realized PnL from fully exited trades)")
    print(SEP)

    current_strategy = None
    for strategy, symbol, entry_price, pnl, reason, opened_at, closed_at in rows:
        if strategy != current_strategy:
            current_strategy = strategy
            print(f"\n  [{strategy}]")
        sign = "+" if pnl >= 0 else ""
        print(
            f"    {symbol:<12} entry=${entry_price:<12.8f} "
            f"pnl=${sign}{pnl:+.4f}  exit={reason or '—':<18} closed={closed_at[:16] if closed_at else '?'}"
        )

    print()
    print(SEP)
    print("  CLOSED POSITIONS SUMMARY")
    print(SEP)
    for strategy, total, wins, pnl in totals:
        win_rate = (wins / total * 100) if total else 0.0
        print(
            f"  [{strategy:<20}]  closed={total:>3}  wins={wins:>3}  "
            f"win_rate={win_rate:>5.1f}%  realized=${pnl:+.4f}"
        )

    total_count, total_wins, total_pnl = grand_total
    total_win_rate = (total_wins / total_count * 100) if total_count else 0.0
    print(SEP)
    print(
        f"  GLOBAL                       closed={total_count:>3}  wins={total_wins:>3}  "
        f"win_rate={total_win_rate:>5.1f}%  realized=${total_pnl:+.4f}"
    )
    print(SEP)
    print()


def print_ai_balance(db_path: str) -> None:
    """Print AI balance, live-trading strategies, and progress toward the $300 target."""
    conn = sqlite3.connect(db_path)

    pnl_row = conn.execute(
        """
        SELECT COALESCE(SUM(outcome_pnl_usd), 0.0)
          FROM strategy_outcomes
         WHERE is_live = 1 AND closed = 1 AND entered = 1
           AND outcome_pnl_usd IS NOT NULL
        """
    ).fetchone()

    live_strategies = conn.execute(
        """
        SELECT DISTINCT strategy FROM strategy_outcomes
         WHERE is_live = 1 AND entered = 1
        """
    ).fetchall()

    conn.close()

    start    = 1000.0
    target   = 300.0
    deadline = "2026-04-10"
    pnl      = pnl_row[0] if pnl_row else 0.0
    balance  = start + pnl
    needed   = max(0.0, target - pnl)
    pct      = (pnl / target * 100) if target else 0.0
    live     = [r[0] for r in live_strategies] if live_strategies else []

    print(SEP)
    print("  AI BALANCE")
    print(SEP)
    print(f"  Balance:          ${balance:.2f}  (started at ${start:.2f})")
    print(f"  Profit so far:    ${pnl:+.2f}")
    print(f"  Target:           ${target:.2f} by {deadline}")
    print(f"  Still needed:     ${needed:.2f}  ({pct:.1f}% of target reached)")
    print(f"  Live strategies:  {', '.join(live) if live else 'none (all paper trading)'}")
    print(SEP)
    print()


def main() -> None:
    try:
        cfg = Config.load()
    except ValueError as exc:
        print(f"[ERROR] Config: {exc}", file=sys.stderr)
        sys.exit(1)

    db_path = os.getenv("DB_PATH", "trader.db")
    if not os.path.exists(db_path):
        print(f"[ERROR] Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    # Print closed-position PnL first (direct DB query — no runner needed)
    print_ai_balance(db_path)
    print_closed_positions(db_path)

    db = TradeDatabase(path=db_path)

    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _here)
    sys.path.insert(0, os.path.dirname(_here))
    from run import build_runners

    runners = build_runners(cfg, db=db)

    for runner in runners:
        saved = db.load_portfolio(runner.name)
        if saved:
            available_cash, starting_cash = saved
            runner.restore_cash(available_cash, starting_cash)
        runner.restore_positions(db.load_open_positions(runner.name))

    db.close()

    # BirdeyePriceClient is not needed — print_summary uses last_price from DB
    engine = MultiStrategyEngine(cfg=cfg, runners=runners, birdeye_client=None)
    engine.print_summary()


if __name__ == "__main__":
    main()
