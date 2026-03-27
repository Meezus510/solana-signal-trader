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


def _closed_count(db_path: str) -> tuple[int, int]:
    """Returns (managed_closed, base_closed) counts."""
    conn = sqlite3.connect(db_path)
    managed = conn.execute(
        "SELECT COUNT(*) FROM strategy_outcomes WHERE strategy=? AND entered=1 AND closed=1",
        (STRATEGY,),
    ).fetchone()[0]
    base = conn.execute(
        "SELECT COUNT(*) FROM strategy_outcomes WHERE strategy=? AND entered=1 AND closed=1",
        ("quick_pop",),
    ).fetchone()[0]
    conn.close()
    return managed, base


def _last_run_meta() -> tuple[int, int, datetime | None]:
    cfg = load_config()
    meta = cfg.get("_meta", {})
    managed_count = int(meta.get(f"{META_PREFIX}last_closed_count", 0))
    base_count = int(meta.get(f"{META_PREFIX}last_base_closed_count", 0))
    ts_raw = meta.get(f"{META_PREFIX}last_run_at")
    if not ts_raw:
        return managed_count, base_count, None
    ts = datetime.fromisoformat(ts_raw)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return managed_count, base_count, ts


def _store_meta(current_managed: int, current_base: int) -> None:
    cfg = load_config()
    original = json.loads(json.dumps(cfg))
    cfg.setdefault("_meta", {})
    cfg["_meta"][f"{META_PREFIX}last_closed_count"] = current_managed
    cfg["_meta"][f"{META_PREFIX}last_base_closed_count"] = current_base
    cfg["_meta"][f"{META_PREFIX}last_run_at"] = datetime.now(timezone.utc).isoformat()
    save_owned_config(
        original,
        cfg,
        owned_strategy=STRATEGY,
        allowed_meta_prefixes=(META_PREFIX,),
    )


def should_trigger(db_path: str, every_closed: int, min_hours: float, min_closed_hours: int) -> tuple[bool, dict]:
    current_managed, current_base = _closed_count(db_path)
    last_managed, last_base, last_run = _last_run_meta()
    managed_delta = current_managed - last_managed
    base_delta = current_base - last_base
    now = datetime.now(timezone.utc)
    hours = None if last_run is None else (now - last_run).total_seconds() / 3600.0

    # Trigger if enough managed trades closed
    trigger = managed_delta >= every_closed
    # OR: time-based fallback with low bar (1 managed trade + time elapsed)
    if not trigger and hours is not None and hours >= min_hours and managed_delta >= min_closed_hours:
        trigger = True
    # OR: base strategy is moving fast but managed strategy is mostly blocked
    if not trigger and base_delta >= every_closed * 5 and managed_delta >= 1:
        trigger = True

    return trigger, {
        "current_closed": current_managed,
        "last_closed": last_managed,
        "new_closed": managed_delta,
        "base_new_closed": base_delta,
        "hours_since_last_run": hours,
    }


def run_once(db_path: str, every_closed: int, min_hours: float, min_closed_hours: int, dry_run: bool) -> None:
    current_managed, current_base = _closed_count(db_path)
    trigger, status = should_trigger(db_path, every_closed, min_hours, min_closed_hours)
    print(json.dumps(status, indent=2, default=str))
    if not trigger:
        print("No trigger.")
        return

    delta = manager_run(db_path=db_path, dry_run=dry_run)
    print(json.dumps(delta, indent=2, default=list))
    if not dry_run:
        _store_meta(current_managed, current_base)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Anthropic strategy manager.")
    parser.add_argument("--db", default="trader.db")
    parser.add_argument("--every-closed", type=int, default=10)
    parser.add_argument("--min-hours", type=float, default=3.0)
    parser.add_argument("--min-closed-hours", type=int, default=1)
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
