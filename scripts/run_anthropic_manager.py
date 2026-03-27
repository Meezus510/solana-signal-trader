#!/usr/bin/env python3
"""
Trigger loop for the Anthropic-managed strategy controller.
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

from trader.agents.anthropic_manager import run as manager_run
from trader.agents.strategy_tuner import load_config, save_owned_config

STRATEGY = "anthropic_managed"
META_PREFIX = "anthropic_manager_"


def _closed_count(db_path: str) -> int:
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT COUNT(*) FROM strategy_outcomes WHERE strategy=? AND entered=1 AND closed=1",
        (STRATEGY,),
    ).fetchone()
    conn.close()
    return row[0] if row else 0


def _last_run_meta() -> tuple[int, datetime | None]:
    cfg = load_config()
    meta = cfg.get("_meta", {})
    count = int(meta.get(f"{META_PREFIX}last_closed_count", 0))
    ts_raw = meta.get(f"{META_PREFIX}last_run_at")
    if not ts_raw:
        return count, None
    ts = datetime.fromisoformat(ts_raw)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return count, ts


def _store_meta(current_closed: int) -> None:
    cfg = load_config()
    original = json.loads(json.dumps(cfg))
    cfg.setdefault("_meta", {})
    cfg["_meta"][f"{META_PREFIX}last_closed_count"] = current_closed
    cfg["_meta"][f"{META_PREFIX}last_run_at"] = datetime.now(timezone.utc).isoformat()
    save_owned_config(
        original,
        cfg,
        owned_strategy=STRATEGY,
        allowed_meta_prefixes=(META_PREFIX,),
    )


def should_trigger(db_path: str, every_closed: int, min_hours: float, min_closed_hours: int) -> tuple[bool, dict]:
    current = _closed_count(db_path)
    last_count, last_run = _last_run_meta()
    delta = current - last_count
    now = datetime.now(timezone.utc)
    hours = None if last_run is None else (now - last_run).total_seconds() / 3600.0

    trigger = delta >= every_closed
    if not trigger and hours is not None and hours >= min_hours and delta >= min_closed_hours:
        trigger = True

    return trigger, {
        "current_closed": current,
        "last_closed": last_count,
        "new_closed": delta,
        "hours_since_last_run": hours,
    }


def run_once(db_path: str, every_closed: int, min_hours: float, min_closed_hours: int, dry_run: bool) -> None:
    trigger, status = should_trigger(db_path, every_closed, min_hours, min_closed_hours)
    print(json.dumps(status, indent=2, default=str))
    if not trigger:
        print("No trigger.")
        return

    delta = manager_run(db_path=db_path, dry_run=dry_run)
    print(json.dumps(delta, indent=2, default=list))
    if not dry_run:
        _store_meta(status["current_closed"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Anthropic strategy manager.")
    parser.add_argument("--db", default="trader.db")
    parser.add_argument("--every-closed", type=int, default=25)
    parser.add_argument("--min-hours", type=float, default=6.0)
    parser.add_argument("--min-closed-hours", type=int, default=10)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--poll-sec", type=int, default=900)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.loop:
        run_once(args.db, args.every_closed, args.min_hours, args.min_closed_hours, args.dry_run)
        return

    while True:
        run_once(args.db, args.every_closed, args.min_hours, args.min_closed_hours, args.dry_run)
        time.sleep(args.poll_sec)


if __name__ == "__main__":
    main()
