#!/usr/bin/env python3
"""
scripts/price_history.py — Continuous post-signal price tracker + dry-run outcome simulator.

For every signal_charts row where price_tracking_done=0, fetches the next 10 minutes of
Birdeye OHLCV data and updates peak/trough/snapshot if new extremes are found.
Tracking stops when:
  - 3 consecutive 10-min intervals show no new high or low  (STALE_THRESHOLD)
  - The total window reaches 240 minutes (MAX_WINDOW_MIN, ~4 hours)
  - No candles returned for MAX_ATTEMPTS consecutive fetches (dead/rugged coin)

Outcome simulation (DRY_RUN=true only):
  On the FIRST fetch for a signal, simulates sell outcomes for any entered+unclosed
  strategy_outcomes rows. Uses the first 10-min candle window (same data as the
  initial price history write). Subsequent fetches do not re-simulate.

Usage
-----
    python scripts/price_history.py                  # run once now
    python scripts/price_history.py --loop           # poll every 10 min
    python scripts/price_history.py --max-window 120 # stop tracking after 2h
    python scripts/price_history.py --dry-run        # fetch + compute, no DB writes
    python scripts/price_history.py --no-sim         # skip outcome simulation
    python scripts/price_history.py --db trader.db   # explicit DB path

Environment
-----------
    BIRDEYE_API_KEY  required
    DB_PATH          optional, defaults to trader.db
    DRY_RUN          if "true"/"1", outcome simulation is enabled
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

_LOG_FMT = "%(asctime)s [%(name)s] %(levelname)s %(message)s"
_LOG_DATE = "%H:%M:%S"
logging.basicConfig(level=logging.INFO, format=_LOG_FMT, datefmt=_LOG_DATE)
_fh = logging.FileHandler("price_history.log", encoding="utf-8")
_fh.setFormatter(logging.Formatter(_LOG_FMT, _LOG_DATE))
logging.getLogger().addHandler(_fh)
logger = logging.getLogger("price_history")

import aiohttp

from trader.config import Config
from trader.persistence.database import TradeDatabase
from trader.pricing.birdeye import BirdeyePriceClient

DB_PATH               = os.getenv("DB_PATH", "trader.db")
IS_DRY_RUN            = os.getenv("DRY_RUN", "").strip().lower() in ("1", "true", "yes")
POLL_INTERVAL_SECONDS = 600    # 10 minutes between loop runs
MAX_BATCH             = 20     # Birdeye calls per run
INTER_REQUEST_SLEEP   = 0.5    # seconds between Birdeye calls
MAX_WINDOW_MIN        = 1440   # base window: 24 hours
DAILY_EXTENSION_PCT   = 100.0  # if price is >100% above entry at each 24h checkpoint, extend another 24h
STALE_THRESHOLD       = 3      # consecutive no-movement intervals → mark done
MAX_ATTEMPTS          = 3      # dead-coin threshold (no candles returned N times)

# Adaptive fetch step: check more often when the coin is fresh.
#   0–60 min  tracked → 10 min step  (catching the initial pump/dump)
#   1–4 hours tracked → 20 min step  (still active but settling)
#   4h+       tracked → 60 min step  (long runners, hourly is enough)
_STEP_SCHEDULE = [
    (60,   10),   # up to 60 min tracked  → 10-min step
    (240,  20),   # up to 240 min tracked → 20-min step
    (None, 60),   # beyond that           → 60-min step
]


def _fetch_step(window_min: int) -> int:
    """Return the fetch step in minutes given how much has already been tracked."""
    for threshold, step in _STEP_SCHEDULE:
        if threshold is None or window_min < threshold:
            return step
    return 60

_STRATEGY_CONFIG_PATH = Path(__file__).parent.parent / "strategy_config.json"


# ---------------------------------------------------------------------------
# Strategy config helpers
# ---------------------------------------------------------------------------

def _load_strategy_config() -> dict:
    try:
        return json.loads(_STRATEGY_CONFIG_PATH.read_text())
    except Exception as exc:
        logger.warning("[price_history] Could not load strategy_config.json: %s", exc)
        return {}


def _iso(unix: int) -> str:
    return datetime.fromtimestamp(unix, tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _get_pending_rows(db_path: str, max_attempts: int) -> list[tuple]:
    """
    Return signal_charts rows where price_tracking_done=0 and the next
    FETCH_STEP_MIN window is available.

    Initial fetch:     peak_price IS NULL  AND signal is at least FETCH_STEP_MIN minutes old
    Subsequent fetch:  peak_price IS NOT NULL AND (signal_ts + price_window_min) is
                       at least FETCH_STEP_MIN minutes in the past

    Returns list of (id, mint, entry_price, ts, price_window_min,
                     peak_price, trough_price, price_stale_count).
    """
    conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
    try:
        return conn.execute(
            """
            SELECT id, mint, entry_price, ts, price_window_min,
                   peak_price, trough_price, price_stale_count, price_checkpoint_price
              FROM signal_charts
             WHERE price_tracking_done = 0
               AND fetch_attempts < ?
               AND (
                     -- Initial fetch: wait 10 min after signal
                     (peak_price IS NULL
                      AND datetime(ts) < datetime('now', '-10 minutes'))
                     OR
                     -- Subsequent fetch: wait the adaptive step beyond current watermark
                     (peak_price IS NOT NULL
                      AND datetime(ts, '+' || price_window_min || ' minutes') < datetime('now', '-' ||
                          CASE
                              WHEN price_window_min < 60  THEN 10
                              WHEN price_window_min < 240 THEN 20
                              ELSE 60
                          END || ' minutes'))
                   )
             ORDER BY ts ASC
             LIMIT ?
            """,
            (max_attempts, MAX_BATCH),
        ).fetchall()
    finally:
        conn.close()


def _get_unclosed_outcomes(db_path: str, signal_chart_id: int) -> list[tuple]:
    """
    Return (outcome_id, strategy_name, usd_size) for entered+unclosed strategy_outcomes
    rows linked to this signal_chart_id. usd_size comes from the positions table.
    """
    conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
    try:
        return conn.execute(
            """
            SELECT so.id, so.strategy, p.usd_size
              FROM strategy_outcomes so
              JOIN signal_charts sc ON sc.id = so.signal_chart_id
              LEFT JOIN positions p
                     ON p.mint = sc.mint AND p.strategy = so.strategy
             WHERE so.signal_chart_id = ?
               AND so.entered = 1
               AND so.closed  = 0
            """,
            (signal_chart_id,),
        ).fetchall()
    finally:
        conn.close()


def _close_position_in_db(
    db_path: str,
    mint: str,
    strategy: str,
    sell_reason: str,
    closed_at: str,
    exit_price: float,
) -> None:
    """Mark the positions table row as CLOSED so it isn't restored on restart.

    Also zeroes remaining_quantity and records the simulated exit price and PnL
    so the row is financially consistent (not left with a phantom open quantity).
    """
    conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
    try:
        conn.execute(
            """
            UPDATE positions
               SET status              = 'CLOSED',
                   sell_reason         = ?,
                   closed_at           = ?,
                   last_price          = ?,
                   realized_pnl_usd    = realized_pnl_usd
                                         + (? - entry_price) * remaining_quantity,
                   total_proceeds_usd  = total_proceeds_usd
                                         + ? * remaining_quantity,
                   remaining_quantity  = 0
             WHERE mint=? AND strategy=? AND status='OPEN'
            """,
            (sell_reason, closed_at, exit_price, exit_price, exit_price, mint, strategy),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Outcome simulation (dry-run only, initial fetch only)
# ---------------------------------------------------------------------------

def _simulate_outcome(
    candles,
    entry_price: float,
    stop_loss_pct: float,
    tp_multiple: float,
) -> tuple[str, float, int | None, float, int | None, float, int | None]:
    """
    Walk 1m candles in chronological order and return
    (sell_reason, exit_price, exit_candle_unix,
     peak_price, peak_unix, trough_price, trough_unix).

    peak_price / trough_price are the highest high and lowest low seen
    across all candles up to and including the exit candle.

    Logic:
      - If a candle's low  <= stop_loss_price → STOP_LOSS
      - If a candle's high >= tp_price        → TP1
      - If both in the same candle            → stop loss wins (conservative)
      - If neither after all candles          → TIMEOUT at final close price
    """
    stop_price = entry_price * (1.0 - stop_loss_pct)
    tp_price   = entry_price * tp_multiple

    peak_price   = entry_price
    peak_unix    = candles[0].unix_time if candles else None
    trough_price = entry_price
    trough_unix  = candles[0].unix_time if candles else None

    for candle in candles:
        if candle.high > peak_price:
            peak_price = candle.high
            peak_unix  = candle.unix_time
        if candle.low < trough_price:
            trough_price = candle.low
            trough_unix  = candle.unix_time

        hit_stop = candle.low  <= stop_price
        hit_tp   = candle.high >= tp_price

        if hit_stop:
            return "STOP_LOSS", stop_price, candle.unix_time, peak_price, peak_unix, trough_price, trough_unix
        if hit_tp:
            return "TP1", tp_price, candle.unix_time, peak_price, peak_unix, trough_price, trough_unix

    exit_price = candles[-1].close if candles else entry_price
    exit_unix  = candles[-1].unix_time if candles else None
    return "TIMEOUT", exit_price, exit_unix, peak_price, peak_unix, trough_price, trough_unix


def _simulate_and_write(
    db: TradeDatabase,
    db_path: str,
    signal_chart_id: int,
    mint: str,
    entry_price: float,
    ts_str: str,
    candles,
    strategy_cfg: dict,
    dry_run_only: bool,
) -> None:
    """
    For each entered+unclosed strategy_outcomes row linked to this signal,
    simulate the sell outcome from the candles and persist results.
    Only runs on the initial fetch and only when IS_DRY_RUN.
    """
    rows = _get_unclosed_outcomes(db_path, signal_chart_id)
    if not rows:
        return

    try:
        ts_dt = datetime.fromisoformat(ts_str)
    except Exception:
        return
    if ts_dt.tzinfo is None:
        ts_dt = ts_dt.replace(tzinfo=timezone.utc)

    for outcome_id, strategy_name, usd_size in rows:
        cfg           = strategy_cfg.get(strategy_name, {})
        stop_loss_pct = cfg.get("stop_loss_pct", 0.30)
        tp_levels     = cfg.get("tp_levels", [[2.0, 0.5]])
        tp_multiple   = tp_levels[0][0] if tp_levels else 2.0

        sell_reason, exit_price, exit_unix, peak_price, peak_unix, trough_price, trough_unix = _simulate_outcome(
            candles, entry_price, stop_loss_pct, tp_multiple
        )

        pnl_pct      = (exit_price   / entry_price - 1.0) * 100.0
        pnl_usd      = (pnl_pct / 100.0) * usd_size if usd_size is not None else None
        max_gain_pct = (peak_price   / entry_price - 1.0) * 100.0 if entry_price else 0.0
        peak_pnl_pct   = max_gain_pct
        trough_pnl_pct = (trough_price / entry_price - 1.0) * 100.0 if entry_price else 0.0
        peak_ts_str    = _iso(peak_unix)   if peak_unix   else None
        trough_ts_str  = _iso(trough_unix) if trough_unix else None

        if exit_unix:
            exit_dt   = datetime.fromtimestamp(exit_unix, tz=timezone.utc)
            closed_at = exit_dt.isoformat()
        else:
            exit_dt   = ts_dt + timedelta(minutes=len(candles))
            closed_at = exit_dt.isoformat()
        hold_secs = (exit_dt - ts_dt).total_seconds()

        logger.info(
            "[sim] signal_chart=%d strategy=%-20s | %-10s pnl=%+.1f%% peak=%+.1f%% trough=%+.1f%% hold=%.0fs",
            signal_chart_id, strategy_name, sell_reason, pnl_pct, peak_pnl_pct, trough_pnl_pct, hold_secs,
        )

        if not dry_run_only:
            db.update_strategy_outcome(
                outcome_id, pnl_pct, sell_reason, hold_secs, max_gain_pct,
                pnl_usd=pnl_usd,
                position_peak_price=peak_price,
                position_peak_ts=peak_ts_str,
                position_peak_pnl_pct=peak_pnl_pct,
                position_trough_price=trough_price,
                position_trough_ts=trough_ts_str,
                position_trough_pnl_pct=trough_pnl_pct,
            )
            _close_position_in_db(db_path, mint, strategy_name, sell_reason, closed_at, exit_price)


# ---------------------------------------------------------------------------
# Main batch processor
# ---------------------------------------------------------------------------

async def _process_batch(
    rows: list[tuple],
    db: TradeDatabase,
    db_path: str,
    dry_run_write: bool,
    simulate: bool,
    strategy_cfg: dict,
    max_attempts: int,
    max_window_min: int,
) -> int:
    """
    For each pending signal_charts row:
      - Fetch the next FETCH_STEP_MIN candles from the current watermark
      - Update peak/trough/snapshot if new extremes found
      - Advance watermark; increment stale counter if no new extreme
      - Mark price_tracking_done when stale or window cap reached
    Returns count of rows updated.
    """
    updated = 0
    async with aiohttp.ClientSession() as session:
        cfg    = Config.load()
        client = BirdeyePriceClient(cfg, session)

        for row_id, mint, entry_price, ts_str, current_window_min, stored_peak, stored_trough, stale_count, checkpoint_price in rows:
            try:
                ts_dt = datetime.fromisoformat(ts_str)
            except Exception:
                logger.warning("[skip] row %d — unparseable ts: %r", row_id, ts_str)
                continue

            if ts_dt.tzinfo is None:
                ts_dt = ts_dt.replace(tzinfo=timezone.utc)
            signal_unix = int(ts_dt.timestamp())

            is_initial = stored_peak is None
            step = _fetch_step(current_window_min if not is_initial else 0)

            # For initial fetch, always start from the signal timestamp.
            # For subsequent fetches, advance from the current watermark.
            if is_initial:
                time_to = signal_unix + step * 60
            else:
                time_to = signal_unix + current_window_min * 60 + step * 60

            candles = await client.get_ohlcv(
                mint_address=mint,
                bars=step,
                interval="1m",
                time_to=time_to,
            )

            if not candles:
                attempts = db.increment_fetch_attempts(row_id) if not dry_run_write else 1
                if attempts >= max_attempts:
                    logger.warning(
                        "[dead] row %d (%s) — no candles after %d attempt(s), dropping",
                        row_id, mint[:8], attempts,
                    )
                else:
                    logger.warning(
                        "[price_history] row %d (%s) — no candles (attempt %d/%d)",
                        row_id, mint[:8], attempts, max_attempts,
                    )
                await asyncio.sleep(INTER_REQUEST_SLEEP)
                continue

            # Strip any candles that predate the signal — Birdeye fetches a
            # slightly wider window (+5 bars) so pre-signal candles can appear.
            candles = [c for c in candles if c.unix_time >= signal_unix]
            if not candles:
                logger.warning("[price_history] row %d (%s) — all candles predate signal, skipping", row_id, mint[:8])
                await asyncio.sleep(INTER_REQUEST_SLEEP)
                continue

            highs  = [c.high  for c in candles]
            lows   = [c.low   for c in candles]
            closes = [c.close for c in candles]

            batch_peak   = max(highs)
            batch_trough = min(lows)
            snapshot_price = closes[-1]
            snapshot_ts_str = _iso(candles[-1].unix_time)

            new_window_min = (step if is_initial else current_window_min + step)

            if is_initial:
                # First fetch: write all price history fields fresh
                peak_price     = batch_peak
                trough_price   = batch_trough
                peak_ts        = _iso(candles[highs.index(batch_peak)].unix_time)
                trough_ts      = _iso(candles[lows.index(batch_trough)].unix_time)
                peak_pnl_pct   = (peak_price   / entry_price - 1.0) * 100.0 if entry_price else None
                trough_pnl_pct = (trough_price / entry_price - 1.0) * 100.0 if entry_price else None
                new_stale      = 0

                logger.info(
                    "[price_history] row %d %s | window=%dmin peak=%+.1f%% trough=%+.1f%% snap=%.8f",
                    row_id, mint[:8], new_window_min,
                    peak_pnl_pct or 0, trough_pnl_pct or 0, snapshot_price,
                )

                if not dry_run_write:
                    db.save_price_history(
                        signal_chart_id=row_id,
                        peak_price=peak_price,
                        peak_price_ts=peak_ts,
                        peak_pnl_pct=peak_pnl_pct,
                        trough_price=trough_price,
                        trough_price_ts=trough_ts,
                        trough_pnl_pct=trough_pnl_pct,
                        snapshot_price=snapshot_price,
                        snapshot_ts=snapshot_ts_str,
                        price_window_min=new_window_min,
                    )
                    updated += 1

                # Simulate sell outcomes against this candle window.
                # _simulate_and_write only acts on entered+unclosed outcomes,
                # so once a strategy hits TP/SL it won't be simulated again.
                # When all outcomes for this signal are closed, subsequent
                # fetches will find no unclosed rows and skip simulation.
                if simulate:
                    _simulate_and_write(
                        db=db,
                        db_path=db_path,
                        signal_chart_id=row_id,
                        mint=mint,
                        entry_price=entry_price,
                        ts_str=ts_str,
                        candles=candles,
                        strategy_cfg=strategy_cfg,
                        dry_run_only=dry_run_write,
                    )

            else:
                # Subsequent fetch: update peak/trough only if new extremes found
                has_new_peak   = batch_peak   > stored_peak
                has_new_trough = batch_trough < stored_trough
                has_movement   = has_new_peak or has_new_trough

                if has_movement:
                    new_stale = 0
                else:
                    new_stale = stale_count + 1

                # At each 24h boundary, compare current price to the price at
                # the previous checkpoint (rolling 24h window):
                #   - first checkpoint (24h): compare vs entry_price
                #   - subsequent checkpoints (48h, 72h, ...): compare vs price saved at prior checkpoint
                # If >100% gain over that 24h window, extend another 24h and
                # save current snapshot as the new checkpoint baseline.
                at_daily_checkpoint = (new_window_min % MAX_WINDOW_MIN == 0
                                       and new_window_min >= MAX_WINDOW_MIN)
                new_checkpoint_price = None  # only set when we save a new baseline
                if at_daily_checkpoint:
                    baseline = checkpoint_price if checkpoint_price is not None else entry_price
                    gain_pct = (snapshot_price / baseline - 1.0) * 100.0 if baseline else 0.0
                    if gain_pct > DAILY_EXTENSION_PCT:
                        logger.info(
                            "[price_history] row %d %s — %dh checkpoint: +%.1f%% vs prior 24h → extending",
                            row_id, mint[:8], new_window_min // 60, gain_pct,
                        )
                        new_checkpoint_price = snapshot_price  # save as next baseline
                        done = new_stale >= STALE_THRESHOLD
                    else:
                        logger.info(
                            "[price_history] row %d %s — %dh checkpoint: +%.1f%% ≤ %.0f%% → stopping",
                            row_id, mint[:8], new_window_min // 60, gain_pct, DAILY_EXTENSION_PCT,
                        )
                        done = True
                else:
                    done = new_stale >= STALE_THRESHOLD

                peak_price     = batch_peak   if has_new_peak   else stored_peak
                trough_price   = batch_trough if has_new_trough else stored_trough
                peak_ts        = _iso(candles[highs.index(batch_peak)].unix_time) if has_new_peak   else None
                trough_ts      = _iso(candles[lows.index(batch_trough)].unix_time) if has_new_trough else None
                peak_pnl_pct   = (peak_price   / entry_price - 1.0) * 100.0 if entry_price else None
                trough_pnl_pct = (trough_price / entry_price - 1.0) * 100.0 if entry_price else None

                status = "DONE" if done else f"stale={new_stale}/{STALE_THRESHOLD}"
                logger.info(
                    "[price_history] row %d %s | window=%dmin peak=%+.1f%% trough=%+.1f%% snap=%.8f [%s]",
                    row_id, mint[:8], new_window_min,
                    peak_pnl_pct or 0, trough_pnl_pct or 0, snapshot_price, status,
                )

                if not dry_run_write:
                    # Only pass new ts values when we actually have a new extreme;
                    # advance_price_watermark's SQL CASE will keep the old values otherwise.
                    db.advance_price_watermark(
                        signal_chart_id=row_id,
                        peak_price=batch_peak,
                        peak_price_ts=peak_ts or _iso(candles[highs.index(batch_peak)].unix_time),
                        peak_pnl_pct=peak_pnl_pct,
                        trough_price=batch_trough,
                        trough_price_ts=trough_ts or _iso(candles[lows.index(batch_trough)].unix_time),
                        trough_pnl_pct=trough_pnl_pct,
                        snapshot_price=snapshot_price,
                        snapshot_ts=snapshot_ts_str,
                        new_window_min=new_window_min,
                        stale_count=new_stale,
                        done=done,
                        checkpoint_price=new_checkpoint_price,
                    )
                    updated += 1

                # Keep simulating on each subsequent window until TP/SL is hit.
                # Once all outcomes are closed, _simulate_and_write is a no-op.
                if simulate:
                    _simulate_and_write(
                        db=db,
                        db_path=db_path,
                        signal_chart_id=row_id,
                        mint=mint,
                        entry_price=entry_price,
                        ts_str=ts_str,
                        candles=candles,
                        strategy_cfg=strategy_cfg,
                        dry_run_only=dry_run_write,
                    )

            await asyncio.sleep(INTER_REQUEST_SLEEP)

    return updated


async def run_once(
    db_path: str,
    dry_run_write: bool,
    simulate: bool,
    max_attempts: int = MAX_ATTEMPTS,
    max_window_min: int = MAX_WINDOW_MIN,
) -> int:
    db = TradeDatabase(db_path)  # opens connection + runs migrations before any raw sqlite3 queries
    rows = _get_pending_rows(db_path, max_attempts)
    if not rows:
        logger.info("[price_history] nothing to process")
        db.close()
        return 0

    logger.info(
        "[price_history] processing %d row(s) | adaptive_step | max_window=%dmin | sim=%s",
        len(rows), max_window_min, simulate,
    )

    strategy_cfg = _load_strategy_config() if simulate else {}
    try:
        updated = await _process_batch(
            rows, db, db_path, dry_run_write, simulate, strategy_cfg, max_attempts, max_window_min,
        )
    finally:
        db.close()

    suffix = " (dry-run write skipped)" if dry_run_write else ""
    logger.info("[price_history] updated %d row(s)%s", updated, suffix)
    return updated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Continuous post-signal price tracker")
    p.add_argument("--loop",        action="store_true",
                   help="Poll every 10 minutes")
    p.add_argument("--max-window",  type=int, default=MAX_WINDOW_MIN, metavar="MIN",
                   help=f"Hard cap: stop tracking after this many minutes (default: {MAX_WINDOW_MIN} = 24h)")
    p.add_argument("--db",          default=DB_PATH, metavar="PATH",
                   help="SQLite DB path (default: $DB_PATH or trader.db)")
    p.add_argument("--dry-run",     action="store_true",
                   help="Fetch and compute but do not write to DB")
    p.add_argument("--no-sim",      action="store_true",
                   help="Skip dry-run outcome simulation even if DRY_RUN=true")
    p.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS, metavar="N",
                   help=f"Drop a coin after N failed OHLCV fetches (default: {MAX_ATTEMPTS})")
    return p.parse_args()


def main() -> None:
    args     = _parse_args()
    simulate = not args.no_sim

    if IS_DRY_RUN and simulate:
        logger.info("[price_history] DRY_RUN=true — outcome simulation enabled")
    elif IS_DRY_RUN:
        logger.info("[price_history] DRY_RUN=true but --no-sim passed — simulation off")

    if args.loop:
        logger.info("[price_history] starting loop (interval=%ds)", POLL_INTERVAL_SECONDS)
        while True:
            try:
                asyncio.run(run_once(args.db, args.dry_run, simulate, args.max_attempts, args.max_window))
            except Exception:
                logger.exception("[price_history] unhandled error — continuing")
            time.sleep(POLL_INTERVAL_SECONDS)
    else:
        asyncio.run(run_once(args.db, args.dry_run, simulate, args.max_attempts, args.max_window))


if __name__ == "__main__":
    main()
