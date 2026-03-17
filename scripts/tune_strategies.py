#!/usr/bin/env python3
"""
scripts/tune_strategies.py — Strategy parameter tuner feedback loop.

Monitors total buy signals received per strategy. When N new signals have
accumulated since the last tune, fires Agent E (strategy_tuner) to analyze
performance and propose + apply parameter adjustments to strategy_config.json.

"Signal" means any row in strategy_outcomes for the strategy — including
skipped signals (entered=0). This lets the tuner fire even when chart/ML
filters are blocking most entries.

Changes take effect on the next hot-reload cycle (strategy_config.json mtime
is watched by the engine) or bot restart.

Usage
-----
    python scripts/tune_strategies.py                          # tune all 5 now
    python scripts/tune_strategies.py --strategy trend_rider   # tune one strategy
    python scripts/tune_strategies.py --every 15               # trigger after 15 new signals
    python scripts/tune_strategies.py --loop                   # poll every 60s
    python scripts/tune_strategies.py --dry-run                # preview, no writes

Environment
-----------
    ANTHROPIC_API_KEY  required
    DB_PATH            optional, defaults to trader.db
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tune_strategies")

from trader.agents.strategy_tuner import (
    CONTROLLED_STRATEGIES,
    load_config,
    run as tuner_run,
    save_config,
)

DB_PATH = os.getenv("DB_PATH", "trader.db")
POLL_INTERVAL_SECONDS = 60
MIN_EVERY = 5
MAX_EVERY = 50
SEP = "=" * 60

CONTROLLED_LIST = sorted(CONTROLLED_STRATEGIES)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _get_signal_count(db_path: str, strategy: str) -> int:
    """
    Count total buy signals received for a strategy (entered OR skipped).

    Uses COUNT(*) on all strategy_outcomes rows — including skipped signals
    (entered=0) — so the tuner fires even when filters block most entries.
    """
    try:
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT COUNT(*) FROM strategy_outcomes WHERE strategy=?",
            (strategy,),
        ).fetchone()
        conn.close()
        return row[0] if row else 0
    except sqlite3.OperationalError:
        return 0


# ---------------------------------------------------------------------------
# Tune logic
# ---------------------------------------------------------------------------

def _should_tune(
    strategy: str,
    db_path: str,
    meta: dict,
    every: int,
) -> tuple[bool, int]:
    """
    Returns (should_tune, current_count).
    Fires when current_count - baseline >= every.
    """
    current = _get_signal_count(db_path, strategy)
    baseline = meta.get("trades_at_last_tune", {}).get(strategy, 0)
    return (current - baseline) >= every, current


def _update_meta(strategy: str, new_count: int, config: dict) -> None:
    """Update _meta in the config dict in-place (caller saves to disk)."""
    if "_meta" not in config:
        config["_meta"] = {}
    meta = config["_meta"]
    if "trades_at_last_tune" not in meta:
        meta["trades_at_last_tune"] = {}
    meta["trades_at_last_tune"][strategy] = new_count
    meta["last_tuned_at"] = datetime.now(timezone.utc).isoformat()


def run_once(
    strategies: list[str],
    db_path: str,
    every: int,
    dry_run: bool,
) -> None:
    """Check all strategies and fire the tuner for any that have hit the threshold."""
    try:
        config = load_config()
    except FileNotFoundError:
        logger.error("strategy_config.json not found — run the bot first to create it")
        return

    meta = config.get("_meta", {})
    any_tuned = False

    for strategy in strategies:
        should, current_count = _should_tune(strategy, db_path, meta, every)

        print(f"\n{SEP}")
        print(f"Strategy: {strategy}")
        baseline = meta.get("trades_at_last_tune", {}).get(strategy, 0)
        new_signals = current_count - baseline
        print(f"  Signals since last tune: {new_signals} / {every} needed")
        print(SEP)

        if not should:
            print(f"  Skipping — not enough new signals yet ({new_signals}/{every})")
            continue

        print(f"  Threshold reached — running tuner...")
        try:
            delta = tuner_run(strategy, db_path=db_path, dry_run=dry_run)
        except Exception as exc:
            logger.exception("Tuner failed for %s: %s", strategy, exc)
            continue

        if not delta:
            print(f"  No changes proposed (insufficient data or no improvements found).")
            continue

        reason = delta.pop("reason", "")
        print(f"\n  Proposed changes:")
        for k, v in delta.items():
            current_val = config.get(strategy, {}).get(k, "?")
            print(f"    {k}: {current_val} → {v}")
        if reason:
            print(f"\n  Reason: {reason}")

        if not dry_run:
            # Reload config after tuner_run() may have written it
            config = load_config()
            _update_meta(strategy, current_count, config)
            save_config(config)
            print(f"\n  Applied. strategy_config.json updated.")
            print(f"  Changes take effect on the next incoming signal (hot-reload).")
        else:
            print(f"\n  [DRY RUN] No changes written.")

        any_tuned = True

    if not any_tuned:
        print(f"\n  No strategies reached the tune threshold.")


def run_loop(
    strategies: list[str],
    db_path: str,
    every: int,
    dry_run: bool,
) -> None:
    """Poll every POLL_INTERVAL_SECONDS and fire the tuner when thresholds are met."""
    print(f"[loop] Watching {len(strategies)} strategy(s). Polling every {POLL_INTERVAL_SECONDS}s.")
    print(f"[loop] Will tune when {every} new buy signals accumulate. Press Ctrl+C to stop.\n")
    try:
        while True:
            run_once(strategies, db_path, every, dry_run)
            time.sleep(POLL_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\n[loop] Stopped.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strategy parameter tuner feedback loop."
    )
    parser.add_argument(
        "--strategy",
        choices=CONTROLLED_LIST + ["all"],
        default="all",
        help="Strategy to tune, or 'all' (default: all)",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=10,
        metavar="N",
        help=f"Tune after N new buy signals (entered+skipped, default: 10, range {MIN_EVERY}-{MAX_EVERY})",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help=f"Poll every {POLL_INTERVAL_SECONDS}s and tune automatically",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show proposed changes without writing strategy_config.json",
    )
    parser.add_argument(
        "--db",
        default=DB_PATH,
        metavar="PATH",
        help=f"Path to trader.db (default: {DB_PATH})",
    )
    args = parser.parse_args()

    if not (MIN_EVERY <= args.every <= MAX_EVERY):
        parser.error(f"--every must be between {MIN_EVERY} and {MAX_EVERY}")

    strategies = CONTROLLED_LIST if args.strategy == "all" else [args.strategy]

    if args.dry_run:
        print("[DRY RUN] No changes will be written.\n")

    if args.loop:
        run_loop(strategies, args.db, args.every, args.dry_run)
    else:
        run_once(strategies, args.db, args.every, args.dry_run)


if __name__ == "__main__":
    main()
