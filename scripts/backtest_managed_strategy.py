#!/usr/bin/env python3
"""
CLI wrapper for generic AI-managed strategy backtests.
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
)
from trader.strategies.registry import MANAGED_STRATEGY_SPECS, get_managed_strategy_spec

DB_PATH = "trader.db"


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward backtest for a managed strategy.")
    parser.add_argument("--db", default=DB_PATH)
    parser.add_argument("--strategy", required=True, choices=sorted(MANAGED_STRATEGY_SPECS))
    parser.add_argument("--base-strategy", default=None)
    parser.add_argument("--mode", default="current")
    parser.add_argument("--date-from", dest="date_from", default=None)
    parser.add_argument("--date-to", dest="date_to", default=None)
    args = parser.parse_args()

    spec = get_managed_strategy_spec(args.strategy)
    allowed_modes = sorted(set().union(*(mode_table.keys() for mode_table in spec.modes.values())))
    if args.mode != "current" and args.mode not in allowed_modes:
        raise SystemExit(f"--mode must be one of: current, {', '.join(allowed_modes)}")

    if args.mode == "current":
        raw = load_managed_config(args.strategy)
        if args.base_strategy:
            raw["base_strategy"] = args.base_strategy
        metrics = backtest_managed_config(
            args.db,
            args.strategy,
            raw,
            date_from=args.date_from,
            date_to=args.date_to,
        )
        print(f"Current {args.strategy} config on {metrics['base_strategy']}:")
    else:
        base = args.base_strategy or load_managed_config(args.strategy).get("base_strategy", spec.default_base)
        metrics = backtest_managed_mode(
            args.db,
            args.strategy,
            base,
            args.mode,
            date_from=args.date_from,
            date_to=args.date_to,
        )
        print(f"{args.strategy} :: {base} / {args.mode}")

    print(f"  entered    : {metrics['entered']}")
    print(f"  blocked    : {metrics['blocked']} ({metrics['block_rate']:.1%})")
    print(f"  win_rate   : {metrics['win_rate']:.1%}")
    print(f"  total_pnl  : ${metrics['total_pnl_usd']:+.2f}")
    print(f"  avg/trade  : ${metrics['avg_pnl_per_trade_usd']:+.2f}")


if __name__ == "__main__":
    main()
