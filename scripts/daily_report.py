"""
scripts/daily_report.py — Daily PnL report from the live database.

Shows closed-position PnL for a given day (default: today UTC), a per-strategy
breakdown, AI balance progress, current SOL price, and open-position counts.

Optionally delivers the report as a Telegram message when TG_BOT_TOKEN and
TG_REPORT_CHAT_ID are set in the environment.

Environment variables:
    DB_PATH              — path to trader.db (default: trader.db)
    BIRDEYE_API_KEY      — required for live SOL price fetch
    TG_BOT_TOKEN         — Telegram bot token for delivery (optional)
    TG_REPORT_CHAT_ID    — chat / user ID to receive the report (optional)

Usage:
    python scripts/daily_report.py                  # today (UTC)
    python scripts/daily_report.py --date 2026-03-23
    python scripts/daily_report.py --send-telegram  # also send via Telegram bot
    python scripts/daily_report.py --all-time       # include all-time totals section
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sqlite3
import sys
from datetime import datetime, timezone, timedelta

import aiohttp
from dotenv import load_dotenv

load_dotenv()

DB_PATH  = os.getenv("DB_PATH", "trader.db")
SEP      = "=" * 72

# SOL native mint — used to fetch the SOL/USD price from Birdeye
_SOL_MINT = "So11111111111111111111111111111111111111112"

# AI balance targets (mirrors summary.py)
_AI_START    = 1000.0
_AI_TARGET   = 300.0
_AI_DEADLINE = "2026-04-10"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _day_bounds(date_str: str) -> tuple[str, str]:
    """Return ISO prefix strings for the start and end of a UTC day."""
    d = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    start = d.strftime("%Y-%m-%d")
    end   = (d + timedelta(days=1)).strftime("%Y-%m-%d")
    return start, end


def _fetch_daily_rows(conn: sqlite3.Connection, date_start: str, date_end: str) -> list:
    return conn.execute(
        """
        SELECT so.strategy,
               sc.symbol,
               sc.entry_price,
               so.outcome_pnl_pct,
               so.outcome_pnl_usd,
               so.outcome_sell_reason,
               so.outcome_hold_secs,
               sc.peak_pnl_pct,
               so.is_live,
               sc.ts
          FROM strategy_outcomes so
          JOIN signal_charts sc ON sc.id = so.signal_chart_id
         WHERE so.entered = 1
           AND so.closed  = 1
           AND so.outcome_pnl_pct IS NOT NULL
           AND sc.ts >= ? AND sc.ts < ?
         ORDER BY so.strategy, sc.ts
        """,
        (date_start, date_end),
    ).fetchall()


def _fetch_daily_totals(conn: sqlite3.Connection, date_start: str, date_end: str) -> list:
    return conn.execute(
        """
        SELECT so.strategy,
               COUNT(*)                                                   AS total,
               SUM(CASE WHEN so.outcome_pnl_pct > 0 THEN 1 ELSE 0 END)  AS wins,
               SUM(CASE WHEN so.outcome_pnl_pct <= 0 THEN 1 ELSE 0 END) AS losses,
               ROUND(AVG(so.outcome_pnl_pct), 2)                         AS avg_pnl,
               COALESCE(SUM(so.outcome_pnl_usd), 0.0)                    AS total_usd,
               SUM(CASE WHEN so.is_live = 1 THEN 1 ELSE 0 END)           AS live_count
          FROM strategy_outcomes so
          JOIN signal_charts sc ON sc.id = so.signal_chart_id
         WHERE so.entered = 1
           AND so.closed  = 1
           AND so.outcome_pnl_pct IS NOT NULL
           AND sc.ts >= ? AND sc.ts < ?
         GROUP BY so.strategy
         ORDER BY so.strategy
        """,
        (date_start, date_end),
    ).fetchall()


def _fetch_grand_daily(conn: sqlite3.Connection, date_start: str, date_end: str) -> tuple:
    return conn.execute(
        """
        SELECT COUNT(*),
               SUM(CASE WHEN so.outcome_pnl_pct > 0 THEN 1 ELSE 0 END),
               ROUND(AVG(so.outcome_pnl_pct), 2),
               COALESCE(SUM(so.outcome_pnl_usd), 0.0)
          FROM strategy_outcomes so
          JOIN signal_charts sc ON sc.id = so.signal_chart_id
         WHERE so.entered = 1 AND so.closed = 1 AND so.outcome_pnl_pct IS NOT NULL
           AND sc.ts >= ? AND sc.ts < ?
        """,
        (date_start, date_end),
    ).fetchone()


def _fetch_open_counts(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute(
        """
        SELECT so.strategy, COUNT(*) AS open
          FROM strategy_outcomes so
         WHERE so.entered = 1 AND so.closed = 0
         GROUP BY so.strategy
        """
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def _fetch_ai_balance(conn: sqlite3.Connection) -> tuple[float, list[str]]:
    pnl = conn.execute(
        """
        SELECT COALESCE(SUM(outcome_pnl_usd), 0.0)
          FROM strategy_outcomes
         WHERE is_live = 1 AND closed = 1 AND entered = 1
           AND outcome_pnl_usd IS NOT NULL
        """
    ).fetchone()[0]

    live = [
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT strategy FROM strategy_outcomes WHERE is_live=1 AND entered=1"
        ).fetchall()
    ]
    return pnl, live


def _fetch_alltime_totals(conn: sqlite3.Connection) -> list:
    return conn.execute(
        """
        SELECT so.strategy,
               COUNT(*),
               SUM(CASE WHEN so.outcome_pnl_pct > 0 THEN 1 ELSE 0 END),
               SUM(CASE WHEN so.outcome_pnl_pct <= 0 THEN 1 ELSE 0 END),
               ROUND(AVG(so.outcome_pnl_pct), 2),
               COALESCE(SUM(so.outcome_pnl_usd), 0.0)
          FROM strategy_outcomes so
         WHERE so.entered = 1 AND so.closed = 1 AND so.outcome_pnl_pct IS NOT NULL
         GROUP BY so.strategy
         ORDER BY so.strategy
        """
    ).fetchall()


# ---------------------------------------------------------------------------
# SOL price
# ---------------------------------------------------------------------------

async def _fetch_sol_price() -> float | None:
    """Fetch the current SOL/USD price from Birdeye. Returns None on failure."""
    api_key = os.getenv("BIRDEYE_API_KEY", "").strip()
    if not api_key:
        return None
    url     = f"https://public-api.birdeye.so/defi/price?address={_SOL_MINT}"
    headers = {
        "X-API-KEY": api_key,
        "x-chain": "solana",
        "accept": "application/json",
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("data", {}).get("value")
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Report builders
# ---------------------------------------------------------------------------

def build_report(db_path: str, date_str: str, include_alltime: bool, sol_price: float | None = None) -> str:
    """Return the full daily report as a plain-text string."""
    date_start, date_end = _day_bounds(date_str)

    conn = sqlite3.connect(db_path)
    rows        = _fetch_daily_rows(conn, date_start, date_end)
    totals      = _fetch_daily_totals(conn, date_start, date_end)
    grand       = _fetch_grand_daily(conn, date_start, date_end)
    open_counts = _fetch_open_counts(conn)
    ai_pnl, live_strats = _fetch_ai_balance(conn)

    alltime_totals = _fetch_alltime_totals(conn) if include_alltime else []
    conn.close()

    lines: list[str] = []

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sol_str = f"   SOL: ${sol_price:,.2f}" if sol_price else ""
    lines += [
        SEP,
        f"  DAILY REPORT  —  {date_str} (UTC)",
        f"  Generated:  {generated_at}{sol_str}",
        SEP,
        "",
    ]

    # ------------------------------------------------------------------
    # AI Balance
    # ------------------------------------------------------------------
    balance = _AI_START + ai_pnl
    needed  = max(0.0, _AI_TARGET - ai_pnl)
    pct     = (ai_pnl / _AI_TARGET * 100) if _AI_TARGET else 0.0
    lines += [
        SEP,
        "  AI BALANCE",
        SEP,
        f"  Balance:       ${balance:.2f}  (started at ${_AI_START:.2f})",
        f"  Profit so far: ${ai_pnl:+.2f}",
        f"  Target:        ${_AI_TARGET:.2f} by {_AI_DEADLINE}",
        f"  Still needed:  ${needed:.2f}  ({pct:.1f}% of target reached)",
        f"  Live strats:   {', '.join(live_strats) if live_strats else 'none (paper trading)'}",
        SEP,
        "",
    ]

    # ------------------------------------------------------------------
    # Daily closed positions detail
    # ------------------------------------------------------------------
    lines += [
        SEP,
        f"  CLOSED POSITIONS  —  {date_str}",
        SEP,
    ]

    if not rows:
        lines.append("  (none closed today)")
    else:
        current_strategy = None
        for strategy, symbol, entry_price, pnl_pct, pnl_usd, reason, hold_secs, peak_pct, is_live, ts in rows:
            if strategy != current_strategy:
                current_strategy = strategy
                lines.append(f"\n  [{strategy}]")
            hold_str = f"{hold_secs/60:.0f}m" if hold_secs else "—"
            peak_str = f"peak={peak_pct:+.1f}%" if peak_pct is not None else ""
            usd_str  = f"${pnl_usd:+.2f}" if pnl_usd is not None else "n/a"
            live_tag = " [LIVE]" if is_live else ""
            lines.append(
                f"    {str(symbol):<10}  entry={entry_price:.8f}  "
                f"pnl={pnl_pct:+.1f}% ({usd_str})  "
                f"exit={str(reason or '—'):<12}  hold={hold_str:<6}  {peak_str}{live_tag}"
            )

    lines.append("")

    # ------------------------------------------------------------------
    # Daily summary by strategy
    # ------------------------------------------------------------------
    if totals:
        lines += [
            SEP,
            f"  DAILY SUMMARY BY STRATEGY  —  {date_str}",
            SEP,
        ]
        for strategy, total, wins, losses, avg_pnl, total_usd, live_count in totals:
            win_rate   = (wins / total * 100) if total else 0.0
            still_open = open_counts.get(strategy, 0)
            live_tag   = f"  live={live_count}" if live_count else ""
            lines.append(
                f"  [{strategy:<28}]  "
                f"closed={total:>3}  open={still_open:>3}  "
                f"W/L={wins}/{losses}  win%={win_rate:>5.1f}  "
                f"avg={avg_pnl:>+6.1f}%  realized={total_usd:>+8.2f}{live_tag}"
            )

        total_count, total_wins, avg_pnl, total_usd = grand
        total_losses   = (total_count or 0) - (total_wins or 0)
        total_win_rate = (total_wins / total_count * 100) if total_count else 0.0
        lines += [
            SEP,
            f"  {'DAILY TOTAL':<30}  "
            f"closed={total_count:>3}  "
            f"W/L={total_wins}/{total_losses}  win%={total_win_rate:>5.1f}  "
            f"avg={avg_pnl:>+6.1f}%  realized={total_usd:>+8.2f}",
            SEP,
            "",
        ]

    # ------------------------------------------------------------------
    # All-time totals (optional)
    # ------------------------------------------------------------------
    if include_alltime and alltime_totals:
        lines += [
            SEP,
            "  ALL-TIME SUMMARY BY STRATEGY",
            SEP,
        ]
        grand_at = (0, 0, 0.0, 0.0)
        for strategy, total, wins, losses, avg_pnl, total_usd in alltime_totals:
            win_rate   = (wins / total * 100) if total else 0.0
            still_open = open_counts.get(strategy, 0)
            lines.append(
                f"  [{strategy:<28}]  "
                f"closed={total:>3}  open={still_open:>3}  "
                f"W/L={wins}/{losses}  win%={win_rate:>5.1f}  "
                f"avg={avg_pnl:>+6.1f}%  realized={total_usd:>+8.2f}"
            )
            grand_at = (
                grand_at[0] + total,
                grand_at[1] + wins,
                0.0,
                grand_at[3] + total_usd,
            )
        g_total, g_wins, _, g_usd = grand_at
        g_losses   = g_total - g_wins
        g_win_rate = (g_wins / g_total * 100) if g_total else 0.0
        lines += [
            SEP,
            f"  {'ALL-TIME TOTAL':<30}  "
            f"closed={g_total:>3}  "
            f"W/L={g_wins}/{g_losses}  win%={g_win_rate:>5.1f}  "
            f"realized={g_usd:>+8.2f}",
            SEP,
            "",
        ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Optional Telegram delivery
# ---------------------------------------------------------------------------

async def _send_telegram(text: str) -> None:
    """Send report text via Telegram Bot API (chunked to respect 4096-char limit)."""
    bot_token = os.getenv("TG_BOT_TOKEN", "").strip()
    chat_id   = os.getenv("TG_REPORT_CHAT_ID", "").strip()

    if not bot_token or not chat_id:
        print(
            "[WARN] TG_BOT_TOKEN or TG_REPORT_CHAT_ID not set — skipping Telegram delivery.",
            file=sys.stderr,
        )
        return

    url   = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    chunk = 4000
    parts = [text[i : i + chunk] for i in range(0, len(text), chunk)]

    async with aiohttp.ClientSession() as session:
        for part in parts:
            payload = {"chat_id": chat_id, "text": part, "parse_mode": "HTML"}
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    print(f"[ERROR] Telegram API {resp.status}: {body}", file=sys.stderr)
                else:
                    print("[INFO] Report sent via Telegram.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Daily PnL report")
    parser.add_argument(
        "--date",
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="UTC date to report (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--all-time",
        action="store_true",
        help="Append an all-time summary section.",
    )
    parser.add_argument(
        "--send-telegram",
        action="store_true",
        help="Also deliver the report via Telegram bot (requires TG_BOT_TOKEN + TG_REPORT_CHAT_ID).",
    )
    parser.add_argument(
        "--db",
        default=DB_PATH,
        help=f"Path to trader.db (default: {DB_PATH}).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"[ERROR] Database not found: {args.db}", file=sys.stderr)
        sys.exit(1)

    try:
        datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        print(f"[ERROR] Invalid date format: {args.date!r}. Use YYYY-MM-DD.", file=sys.stderr)
        sys.exit(1)

    async def _run() -> None:
        sol_price = await _fetch_sol_price()
        report    = build_report(args.db, args.date, include_alltime=args.all_time, sol_price=sol_price)
        print(report)

        # Save to reports/YYYY-MM-DD.txt alongside the database
        import pathlib
        reports_dir = pathlib.Path(args.db).parent / "reports"
        reports_dir.mkdir(exist_ok=True)
        report_file = reports_dir / f"{args.date}.txt"
        report_file.write_text(report, encoding="utf-8")
        print(f"[INFO] Report saved to {report_file}")

        if args.send_telegram:
            await _send_telegram(report)

    asyncio.run(_run())


if __name__ == "__main__":
    main()
