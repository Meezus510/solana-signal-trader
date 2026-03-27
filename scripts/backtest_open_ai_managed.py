#!/usr/bin/env python3
"""
Backward-compatible wrapper for the generic managed strategy backtest engine.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from trader.analysis.managed_backtest import (
    backtest_managed_config,
    backtest_managed_mode,
    load_managed_config,
    resolve_managed_config,
)
from trader.strategies.registry import _OPEN_AI_MANAGED_BASES, _OPEN_AI_MANAGED_MODES

DB_PATH = "trader.db"
STRATEGY = "open_ai_managed"


def load_open_ai_config(config_path: str = "strategy_config.json") -> dict:
    return load_managed_config(STRATEGY, config_path=config_path)


def resolve_open_ai_config(raw_cfg: dict) -> tuple[str, dict]:
    return resolve_managed_config(STRATEGY, raw_cfg)


def backtest_config(db_path: str, cfg: dict) -> dict:
    return backtest_managed_config(db_path, STRATEGY, cfg)


def backtest_named_mode(db_path: str, base_strategy: str, mode: str) -> dict:
    return backtest_managed_mode(db_path, STRATEGY, base_strategy, mode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward backtest for open_ai_managed.")
    parser.add_argument("--db", default=DB_PATH)
    parser.add_argument("--base-strategy", choices=sorted(_OPEN_AI_MANAGED_BASES), default=None)
    parser.add_argument("--mode", default="current", choices=["current", *sorted(next(iter(_OPEN_AI_MANAGED_MODES.values())).keys())])
    args = parser.parse_args()

    if args.mode == "current":
        raw = load_open_ai_config()
        if args.base_strategy:
            raw["base_strategy"] = args.base_strategy
        metrics = backtest_config(args.db, raw)
        print(f"Current open_ai_managed config on {metrics['base_strategy']}:")
    else:
        base = args.base_strategy or load_open_ai_config().get("base_strategy", "quick_pop")
        metrics = backtest_named_mode(args.db, base, args.mode)
        print(f"{base} / {args.mode}")

    print(f"  entered    : {metrics['entered']}")
    print(f"  blocked    : {metrics['blocked']} ({metrics['block_rate']:.1%})")
    print(f"  win_rate   : {metrics['win_rate']:.1%}")
    print(f"  total_pnl  : ${metrics['total_pnl_usd']:+.2f}")
    print(f"  avg/trade  : ${metrics['avg_pnl_per_trade_usd']:+.2f}")


if __name__ == "__main__":
    main()
