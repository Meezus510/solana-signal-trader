#!/usr/bin/env python3
"""
scripts/run_openai_manager.py — Trigger loop for the OpenAI manager.

Recommended policy:
  - run after enough closed trades to learn from realized outcomes, or
  - run after enough evaluated / blocked signals to detect regime shifts, or
  - run on a time heartbeat during active opportunity flow
This lets the agent react even when strict filters block most trades.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from trader.agents.openai_manager import run as manager_run
from trader.agents.strategy_tuner import load_config, save_owned_config

STRATEGY = "open_ai_managed"
META_PREFIX = "openai_manager_"


def _closed_count(db_path: str) -> int:
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT COUNT(*) FROM strategy_outcomes WHERE strategy=? AND entered=1 AND closed=1",
        (STRATEGY,),
    ).fetchone()
    conn.close()
    return row[0] if row else 0


def _signal_counts(db_path: str) -> tuple[int, int]:
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS total_signals,
            SUM(CASE WHEN entered = 0 THEN 1 ELSE 0 END) AS blocked_signals
        FROM strategy_outcomes
        WHERE strategy = ?
        """,
        (STRATEGY,),
    ).fetchone()
    conn.close()
    if not row:
        return 0, 0
    return int(row[0] or 0), int(row[1] or 0)


def _last_run_meta() -> tuple[int, int, int, datetime | None]:
    cfg = load_config()
    meta = cfg.get("_meta", {})
    closed = int(meta.get(f"{META_PREFIX}last_closed_count", 0))
    signals = int(meta.get(f"{META_PREFIX}last_signal_count", 0))
    blocked = int(meta.get(f"{META_PREFIX}last_blocked_count", 0))
    ts_raw = meta.get(f"{META_PREFIX}last_run_at")
    if not ts_raw:
        return closed, signals, blocked, None
    ts = datetime.fromisoformat(ts_raw)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return closed, signals, blocked, ts


def _store_meta(current_closed: int, current_signals: int, current_blocked: int) -> None:
    cfg = load_config()
    original = json.loads(json.dumps(cfg))
    cfg.setdefault("_meta", {})
    cfg["_meta"][f"{META_PREFIX}last_closed_count"] = current_closed
    cfg["_meta"][f"{META_PREFIX}last_signal_count"] = current_signals
    cfg["_meta"][f"{META_PREFIX}last_blocked_count"] = current_blocked
    cfg["_meta"][f"{META_PREFIX}last_run_at"] = datetime.now(timezone.utc).isoformat()
    save_owned_config(
        original,
        cfg,
        owned_strategy=STRATEGY,
        allowed_meta_prefixes=(META_PREFIX,),
    )


def should_trigger(
    db_path: str,
    every_closed: int,
    min_hours: float,
    min_closed_hours: int,
    every_signals: int,
    every_blocked: int,
    blocked_rate: float,
    heartbeat_signals: int,
    bootstrap_signals: int,
) -> tuple[bool, dict]:
    current_closed = _closed_count(db_path)
    current_signals, current_blocked = _signal_counts(db_path)
    last_closed, last_signals, last_blocked, last_run = _last_run_meta()
    delta_closed = current_closed - last_closed
    delta_signals = current_signals - last_signals
    delta_blocked = current_blocked - last_blocked
    now = datetime.now(timezone.utc)
    hours = None if last_run is None else (now - last_run).total_seconds() / 3600.0
    block_rate_recent = (delta_blocked / delta_signals) if delta_signals > 0 else 0.0

    reasons: list[str] = []
    if last_run is None and current_signals >= bootstrap_signals:
        reasons.append("bootstrap_signals")
    if delta_closed >= every_closed:
        reasons.append("closed_trades")
    if delta_signals >= every_signals:
        reasons.append("signal_flow")
    if delta_blocked >= every_blocked and block_rate_recent >= blocked_rate:
        reasons.append("high_block_rate")
    if (
        hours is not None
        and hours >= min_hours
        and (delta_closed >= min_closed_hours or delta_signals >= heartbeat_signals)
    ):
        reasons.append("heartbeat")

    trigger = bool(reasons)

    return trigger, {
        "current_closed": current_closed,
        "last_closed": last_closed,
        "new_closed": delta_closed,
        "current_signals": current_signals,
        "last_signals": last_signals,
        "new_signals": delta_signals,
        "current_blocked": current_blocked,
        "last_blocked": last_blocked,
        "new_blocked": delta_blocked,
        "recent_block_rate": round(block_rate_recent, 4),
        "hours_since_last_run": hours,
        "trigger_reasons": reasons,
    }


def run_once(
    db_path: str,
    every_closed: int,
    min_hours: float,
    min_closed_hours: int,
    every_signals: int,
    every_blocked: int,
    blocked_rate: float,
    heartbeat_signals: int,
    bootstrap_signals: int,
    dry_run: bool,
) -> None:
    trigger, status = should_trigger(
        db_path,
        every_closed,
        min_hours,
        min_closed_hours,
        every_signals,
        every_blocked,
        blocked_rate,
        heartbeat_signals,
        bootstrap_signals,
    )
    print(json.dumps(status, indent=2, default=str))
    if not trigger:
        print("No trigger.")
        return

    delta = manager_run(db_path=db_path, dry_run=dry_run)
    print(json.dumps(delta, indent=2, default=list))
    if not dry_run:
        _store_meta(
            status["current_closed"],
            status["current_signals"],
            status["current_blocked"],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the OpenAI strategy manager.")
    parser.add_argument("--db", default="trader.db")
    parser.add_argument("--every-closed", type=int, default=12)
    parser.add_argument("--min-hours", type=float, default=3.0)
    parser.add_argument("--min-closed-hours", type=int, default=10)
    parser.add_argument("--every-signals", type=int, default=40)
    parser.add_argument("--every-blocked", type=int, default=30)
    parser.add_argument("--blocked-rate", type=float, default=0.85)
    parser.add_argument("--heartbeat-signals", type=int, default=20)
    parser.add_argument("--bootstrap-signals", type=int, default=30)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--poll-sec", type=int, default=900)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.loop:
        run_once(
            args.db,
            args.every_closed,
            args.min_hours,
            args.min_closed_hours,
            args.every_signals,
            args.every_blocked,
            args.blocked_rate,
            args.heartbeat_signals,
            args.bootstrap_signals,
            args.dry_run,
        )
        return

    while True:
        run_once(
            args.db,
            args.every_closed,
            args.min_hours,
            args.min_closed_hours,
            args.every_signals,
            args.every_blocked,
            args.blocked_rate,
            args.heartbeat_signals,
            args.bootstrap_signals,
            args.dry_run,
        )
        time.sleep(args.poll_sec)


if __name__ == "__main__":
    main()
