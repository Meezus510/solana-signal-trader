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
    """
    Query and print all closed strategy outcomes.
    Reads from strategy_outcomes (which has simulated DRY_RUN PnL) joined to
    signal_charts for symbol/entry price. Falls back to positions table for
    live-traded closes where outcome_pnl_usd may live there instead.
    """
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT so.strategy,
               sc.symbol,
               sc.entry_price,
               so.outcome_pnl_pct,
               so.outcome_pnl_usd,
               so.outcome_sell_reason,
               so.outcome_hold_secs,
               sc.peak_pnl_pct,
               sc.ts
          FROM strategy_outcomes so
          JOIN signal_charts sc ON sc.id = so.signal_chart_id
         WHERE so.entered = 1
           AND so.closed  = 1
           AND so.outcome_pnl_pct IS NOT NULL
         ORDER BY so.strategy, sc.ts
        """
    ).fetchall()

    totals = conn.execute(
        """
        SELECT so.strategy,
               COUNT(*)                                                      AS total,
               SUM(CASE WHEN so.outcome_pnl_pct > 0 THEN 1 ELSE 0 END)     AS wins,
               SUM(CASE WHEN so.outcome_pnl_pct <= 0 THEN 1 ELSE 0 END)    AS losses,
               ROUND(AVG(so.outcome_pnl_pct), 2)                            AS avg_pnl,
               COALESCE(SUM(so.outcome_pnl_usd), 0.0)                       AS total_usd
          FROM strategy_outcomes so
         WHERE so.entered = 1
           AND so.closed  = 1
           AND so.outcome_pnl_pct IS NOT NULL
         GROUP BY so.strategy
         ORDER BY so.strategy
        """
    ).fetchall()

    open_counts = conn.execute(
        """
        SELECT so.strategy, COUNT(*) AS open
          FROM strategy_outcomes so
         WHERE so.entered = 1
           AND so.closed  = 0
         GROUP BY so.strategy
        """
    ).fetchall()
    open_by_strategy = {r[0]: r[1] for r in open_counts}

    grand = conn.execute(
        """
        SELECT COUNT(*),
               SUM(CASE WHEN outcome_pnl_pct > 0 THEN 1 ELSE 0 END),
               ROUND(AVG(outcome_pnl_pct), 2),
               COALESCE(SUM(outcome_pnl_usd), 0.0)
          FROM strategy_outcomes
         WHERE entered = 1 AND closed = 1 AND outcome_pnl_pct IS NOT NULL
        """
    ).fetchone()

    conn.close()

    if not rows:
        print(SEP)
        print("  CLOSED OUTCOMES — none yet")
        print(SEP)
        return

    print(SEP)
    print("  CLOSED OUTCOMES  (simulated via price_history.py candles)")
    print(SEP)

    current_strategy = None
    for strategy, symbol, entry_price, pnl_pct, pnl_usd, reason, hold_secs, peak_pct, ts in rows:
        if strategy != current_strategy:
            current_strategy = strategy
            print(f"\n  [{strategy}]")
        hold_str = f"{hold_secs/60:.0f}m" if hold_secs else "—"
        peak_str = f"peak={peak_pct:+.1f}%" if peak_pct is not None else ""
        usd_str  = f"${pnl_usd:+.2f}" if pnl_usd is not None else "n/a"
        print(
            f"    {str(symbol):<10}  entry={entry_price:.8f}  "
            f"pnl={pnl_pct:+.1f}% ({usd_str})  "
            f"exit={str(reason or '—'):<12}  hold={hold_str:<6}  {peak_str}"
        )

    print()
    print(SEP)
    print("  SUMMARY BY STRATEGY")
    print(SEP)
    for strategy, total, wins, losses, avg_pnl, total_usd in totals:
        win_rate = (wins / total * 100) if total else 0.0
        still_open = open_by_strategy.get(strategy, 0)
        print(
            f"  [{strategy:<28}]  "
            f"closed={total:>3}  open={still_open:>3}  "
            f"W/L={wins}/{losses}  win%={win_rate:>5.1f}  "
            f"avg={avg_pnl:>+6.1f}%  realized={total_usd:>+8.2f}"
        )

    total_count, total_wins, avg_pnl, total_usd = grand
    total_losses = total_count - total_wins
    total_win_rate = (total_wins / total_count * 100) if total_count else 0.0
    print(SEP)
    print(
        f"  {'GLOBAL':<30}  "
        f"closed={total_count:>3}  "
        f"W/L={total_wins}/{total_losses}  win%={total_win_rate:>5.1f}  "
        f"avg={avg_pnl:>+6.1f}%  realized={total_usd:>+8.2f}"
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

    db = TradeDatabase(path=db_path, read_only=True)

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
        runner.restore_closed_positions(db.load_closed_positions(runner.name))

    db.close()

    # BirdeyePriceClient is not available here — open positions show entry price,
    # not current market price. Unrealized PnL will read $0 for all open positions.
    print(SEP)
    print("  OPEN POSITIONS  (prices shown are entry prices — not live)")
    print(SEP)
    engine = MultiStrategyEngine(cfg=cfg, runners=runners, birdeye_client=None)
    engine.print_summary()


if __name__ == "__main__":
    main()
