"""
backtest_chart.py — Replay chart filter against historical trades.log.

Parses trades.log, fetches historical OHLCV for each signal at the exact
time it was entered, runs the chart filter, then prints a side-by-side
comparison in summary.py style:

    BASELINE     — enters every signal (current behaviour)
    CHART FILTER — skips entries when pump_ratio >= 3.5x or volume dying

Usage:
    python backtest_chart.py [--log trades.log]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import aiohttp
from dotenv import load_dotenv

from trader.analysis.chart import OHLCV_BARS, compute_chart_context
from trader.config import Config
from trader.pricing.birdeye import BirdeyePriceClient

load_dotenv()

SEP  = "=" * 72
DASH = "-" * 52


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

_LOG_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"       # timestamp
    r" \| (\S+)\s+"                                  # event
    r"\| (\S+)\s+"                                   # symbol
    r"\| (\S+)\s+"                                   # mint
    r"\| price=\$([0-9.]+)"                          # price
    r" \| qty=([0-9.]+)"                             # qty
    r" \| pnl=\$([+\-]?[0-9.]+)"                    # pnl
    r"(?: \| strategy=(\S+))?"                       # strategy (optional)
)


@dataclass
class LogTrade:
    ts: str
    event: str
    symbol: str
    mint: str
    price: float
    qty: float
    pnl: float
    strategy: str


def parse_log(path: str) -> list[LogTrade]:
    trades: list[LogTrade] = []
    with open(path) as fh:
        for line in fh:
            m = _LOG_RE.search(line)
            if not m:
                continue
            ts, event, symbol, mint, price, qty, pnl, strategy = m.groups()
            trades.append(LogTrade(
                ts=ts, event=event.strip(), symbol=symbol, mint=mint,
                price=float(price), qty=float(qty), pnl=float(pnl),
                strategy=strategy or "default",
            ))
    return trades


# ---------------------------------------------------------------------------
# Position aggregation  (per symbol × strategy)
# ---------------------------------------------------------------------------

@dataclass
class PositionResult:
    symbol: str
    mint: str
    strategy: str
    entry_price: float
    entry_ts: str
    total_pnl: float = 0.0
    exit_reason: str = "OPEN"
    is_win: bool = False


def aggregate_positions(trades: list[LogTrade]) -> list[PositionResult]:
    """Roll up all trade events into per-(symbol, strategy) position results."""
    # Collect buy timestamps + all sell PnL per position
    entries: dict[tuple, PositionResult] = {}
    for t in trades:
        key = (t.symbol, t.mint, t.strategy)
        if t.event == "BUY":
            entries[key] = PositionResult(
                symbol=t.symbol, mint=t.mint,
                strategy=t.strategy,
                entry_price=t.price, entry_ts=t.ts,
            )
        elif key in entries:
            entries[key].total_pnl += t.pnl
            entries[key].exit_reason = t.event

    for pos in entries.values():
        pos.is_win = pos.total_pnl > 0

    return list(entries.values())


# ---------------------------------------------------------------------------
# Chart verdict per unique signal
# ---------------------------------------------------------------------------

async def fetch_verdicts(
    positions: list[PositionResult],
    birdeye: BirdeyePriceClient,
) -> dict[str, bool]:
    """
    Returns mint → should_enter mapping.
    Fetches OHLCV once per unique mint (shared across strategies).
    """
    seen: dict[str, bool] = {}
    unique = {(p.mint, p.entry_ts, p.entry_price) for p in positions}

    for mint, ts_str, entry_price in sorted(unique, key=lambda x: x[1]):
        if mint in seen:
            continue
        dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        time_to = int(dt.timestamp())
        candles = await birdeye.get_ohlcv(mint, bars=OHLCV_BARS, time_to=time_to)
        ctx = compute_chart_context(candles, entry_price)
        # If no data, default to entering (same as live fallback)
        seen[mint] = ctx.should_enter if ctx else True
        await asyncio.sleep(0.4)  # avoid rate limiting

    return seen


# ---------------------------------------------------------------------------
# Summary printer  (mirrors summary.py output style)
# ---------------------------------------------------------------------------

def _strategy_block(label: str, positions: list[PositionResult], starting_cash: float) -> None:
    closed = [p for p in positions if p.exit_reason != "OPEN"]
    wins   = [p for p in closed if p.is_win]
    total_pnl  = sum(p.total_pnl for p in closed)
    win_rate   = (len(wins) / len(closed) * 100) if closed else 0.0
    net_pnl_pct = (total_pnl / starting_cash * 100) if starting_cash else 0.0

    print(SEP)
    print(f"  Strategy: {label}")
    print(SEP)
    for p in closed:
        print(
            f"  {'[WIN] ' if p.is_win else '[LOSS]'} {p.symbol:<12} "
            f"entry=${p.entry_price:<12.8f} "
            f"pnl=${p.total_pnl:+.4f}  "
            f"exit={p.exit_reason:<20} entered={p.entry_ts}"
        )
    print(f"  {DASH}")
    print(f"  Closed      : {len(closed):>3}")
    print(f"  Wins        : {len(wins):>3}   ({win_rate:.1f}%)")
    print(f"  Realized PnL: ${total_pnl:+.4f}  ({net_pnl_pct:+.2f}% of ${starting_cash:.0f})")
    print(SEP)
    print()


def print_comparison(
    positions: list[PositionResult],
    verdicts: dict[str, bool],
    starting_cash: float,
) -> None:
    strategies = sorted({p.strategy for p in positions})

    # ---- BASELINE (all signals) ----
    print()
    print("=" * 72)
    print("  GROUP A — BASELINE  (enters every signal)")
    print("=" * 72)
    for strat in strategies:
        strat_pos = [p for p in positions if p.strategy == strat]
        _strategy_block(strat, strat_pos, starting_cash)

    baseline_closed = [p for p in positions if p.exit_reason != "OPEN"]
    baseline_wins   = [p for p in baseline_closed if p.is_win]
    baseline_pnl    = sum(p.total_pnl for p in baseline_closed)
    baseline_wr     = (len(baseline_wins) / len(baseline_closed) * 100) if baseline_closed else 0.0
    baseline_total_cash = starting_cash * len(strategies)
    baseline_pnl_pct = (baseline_pnl / baseline_total_cash * 100) if baseline_total_cash else 0.0

    print(SEP)
    print(f"  GLOBAL — BASELINE  ({len(strategies)} strategies)")
    print(SEP)
    print(f"  Signals entered : {len({p.mint for p in positions})}")
    print(f"  Signals skipped : 0  (no filter)")
    print(f"  Total closed    : {len(baseline_closed)}")
    print(f"  Win rate        : {baseline_wr:.1f}%")
    print(f"  Total realized  : ${baseline_pnl:+.4f}  ({baseline_pnl_pct:+.2f}% on ${baseline_total_cash:.0f})")
    print(SEP)

    # ---- CHART FILTER ----
    # Mirror strategy names (append _chart suffix for display)
    print()
    print()
    print("=" * 72)
    print("  GROUP B — CHART FILTER  (skips late pumps + dying volume)")
    print("=" * 72)

    chart_positions = [p for p in positions if verdicts.get(p.mint, True)]
    skipped_mints   = {p.mint for p in positions if not verdicts.get(p.mint, True)}
    skipped_symbols = sorted({p.symbol for p in positions if p.mint in skipped_mints})

    print(f"\n  Skipped signals  : {', '.join(skipped_symbols) or 'none'}")
    print(f"  Entered signals  : {', '.join(sorted({p.symbol for p in chart_positions}))}\n")

    for strat in strategies:
        strat_pos = [p for p in chart_positions if p.strategy == strat]
        _strategy_block(f"{strat}_chart", strat_pos, starting_cash)

    chart_closed = [p for p in chart_positions if p.exit_reason != "OPEN"]
    chart_wins   = [p for p in chart_closed if p.is_win]
    chart_pnl    = sum(p.total_pnl for p in chart_closed)
    chart_wr     = (len(chart_wins) / len(chart_closed) * 100) if chart_closed else 0.0
    chart_pnl_pct = (chart_pnl / baseline_total_cash * 100) if baseline_total_cash else 0.0

    print(SEP)
    print(f"  GLOBAL — CHART FILTER  ({len(strategies)} strategies)")
    print(SEP)
    print(f"  Signals entered : {len({p.mint for p in chart_positions})}")
    print(f"  Signals skipped : {len(skipped_mints)}  {skipped_symbols}")
    print(f"  Total closed    : {len(chart_closed)}")
    print(f"  Win rate        : {chart_wr:.1f}%")
    print(f"  Total realized  : ${chart_pnl:+.4f}  ({chart_pnl_pct:+.2f}% on ${baseline_total_cash:.0f})")
    print(SEP)

    # ---- DELTA ----
    delta = chart_pnl - baseline_pnl
    print()
    print(SEP)
    print("  CHART FILTER vs BASELINE")
    print(SEP)
    print(f"  Baseline realized   : ${baseline_pnl:+.4f}")
    print(f"  Chart filter realized: ${chart_pnl:+.4f}")
    print(f"  Delta               : ${delta:+.4f}  ({'better' if delta > 0 else 'worse'} with chart filter)")
    print(f"  Signals skipped     : {len(skipped_mints)} of {len({p.mint for p in positions})}")
    skipped_win = [p for p in positions if p.mint in skipped_mints and p.is_win and p.exit_reason != 'OPEN']
    skipped_loss = [p for p in positions if p.mint in skipped_mints and not p.is_win and p.exit_reason != 'OPEN']
    print(f"  Skipped winners     : {len(skipped_win)//len(strategies) if strategies else 0}  ({', '.join(sorted({p.symbol for p in skipped_win}))})")
    print(f"  Skipped losers      : {len(skipped_loss)//len(strategies) if strategies else 0}  ({', '.join(sorted({p.symbol for p in skipped_loss}))})")
    print(SEP)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(cfg: Config, log_path: str) -> None:
    trades = parse_log(log_path)
    if not trades:
        print(f"No trades found in {log_path}")
        return

    positions = aggregate_positions(trades)
    strategies = sorted({p.strategy for p in positions})
    print(f"\nParsed {len(trades)} trade events → {len(positions)} positions "
          f"across {len(strategies)} strategies: {', '.join(strategies)}")

    async with aiohttp.ClientSession() as http:
        birdeye = BirdeyePriceClient(cfg=cfg, session=http)
        print(f"Fetching OHLCV for {len({p.mint for p in positions})} unique tokens…\n")
        verdicts = await fetch_verdicts(positions, birdeye)

    # print per-token verdict table
    print(SEP)
    print(f"  {'SYMBOL':<12} {'CHART':<6} {'PUMP':>6}  REASON")
    print(SEP)
    seen = set()
    for p in positions:
        if p.mint in seen:
            continue
        seen.add(p.mint)
        entered = verdicts.get(p.mint, True)
        # find a context summary — use symbol's first position
        print(f"  {p.symbol:<12} {'BUY ' if entered else 'SKIP'}")
    print(SEP)

    print_comparison(positions, verdicts, starting_cash=cfg.starting_cash_usd)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="trades.log")
    args = ap.parse_args()

    try:
        cfg = Config.load()
    except ValueError as exc:
        print(f"Config error: {exc}")
        return

    await run(cfg, args.log)


if __name__ == "__main__":
    asyncio.run(main())
