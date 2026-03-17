#!/usr/bin/env python3
"""
scripts/run_agents.py — Run AI agents to analyze performance and suggest config changes.

Usage:
    python scripts/run_agents.py --agent threshold
    python scripts/run_agents.py --agent exit
    python scripts/run_agents.py --agent policy
    python scripts/run_agents.py --agent all
    python scripts/run_agents.py --agent threshold --db /path/to/trader.db

The threshold and exit agents print a proposed delta as JSON (no auto-apply).
The policy agent runs against a hardcoded example signal context and prints the decision.

Environment:
    ANTHROPIC_API_KEY  required — Claude API key
    (other .env vars loaded automatically via python-dotenv)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

# Current live config for quick_pop_chart_ml — update these when you change registry.py
CURRENT_CONFIG = {
    "ml_min_score":            5.0,
    "ml_high_score_threshold": 8.0,
    "ml_max_score_threshold":  9.5,
    "ml_size_multiplier":      2.0,
    "ml_max_size_multiplier":  3.0,
    "stop_loss_pct":           0.20,
    "trailing_stop_pct":       0.22,
    "timeout_minutes":         45.0,
}


def run_threshold(db_path: str, strategy: str) -> dict:
    from trader.agents.threshold import run
    print(f"\n{'='*60}")
    print("AGENT B — ML Threshold Optimizer")
    print(f"{'='*60}")
    delta = run(db_path=db_path, strategy=strategy, current_config=CURRENT_CONFIG)
    if delta:
        reason = delta.pop("reason", "")
        print(f"\nProposed changes:")
        for k, v in delta.items():
            current = CURRENT_CONFIG.get(k, "?")
            print(f"  {k}: {current} → {v}")
        if reason:
            print(f"\nReason: {reason}")
        delta["reason"] = reason
    else:
        print("No changes proposed (insufficient data or no improvements found).")
    return delta


# Hardcoded example signal contexts for testing policy agent without live wiring.
# Edit these to exercise different code paths.
EXAMPLE_SIGNAL_CONTEXTS = {
    "normal": {
        "ml_score": 7.2,
        "used_moralis_10s": True,
        "used_birdeye_fallback": False,
        "pair_stats_available": True,
        "liquidity_usd": 45_000,
        "slippage_bps": 90,
    },
    "degraded": {
        "ml_score": 6.1,
        "used_moralis_10s": False,
        "used_birdeye_fallback": True,   # fallback candles
        "pair_stats_available": False,   # missing pair stats
        "liquidity_usd": 18_000,         # thin market
        "slippage_bps": 230,             # high slippage
    },
    "hard_block": {
        "ml_score": 8.5,
        "used_moralis_10s": True,
        "used_birdeye_fallback": False,
        "pair_stats_available": True,
        "liquidity_usd": 2_000,          # below hard floor → always blocked
        "slippage_bps": 80,
    },
}


def run_policy(strategy: str, context_name: str = "normal") -> dict:
    from trader.agents.policy import propose_policy_decision
    ctx = EXAMPLE_SIGNAL_CONTEXTS.get(context_name, EXAMPLE_SIGNAL_CONTEXTS["normal"])
    print(f"\n{'='*60}")
    print("AGENT A — Per-Signal Policy")
    print(f"{'='*60}")
    print(f"\nSignal context ({context_name}):")
    for k, v in ctx.items():
        print(f"  {k}: {v}")
    decision = propose_policy_decision(signal_context=ctx, strategy=strategy)
    print(f"\nDecision:")
    print(f"  allow_trade:                {decision['allow_trade']}")
    print(f"  buy_size_multiplier:        {decision['buy_size_multiplier']}")
    print(f"  effective_score_adjustment: {decision['effective_score_adjustment']}")
    print(f"  reason_codes:               {decision['reason_codes']}")
    return decision


def run_exit(db_path: str, strategy: str) -> dict:
    from trader.agents.exit_tuner import run
    print(f"\n{'='*60}")
    print("AGENT D — Exit Parameter Tuner")
    print(f"{'='*60}")
    delta = run(db_path=db_path, strategy=strategy, current_config=CURRENT_CONFIG)
    if delta:
        reason = delta.pop("reason", "")
        print(f"\nProposed changes:")
        for k, v in delta.items():
            current = CURRENT_CONFIG.get(k, "?")
            print(f"  {k}: {current} → {v}")
        if reason:
            print(f"\nReason: {reason}")
        delta["reason"] = reason
    else:
        print("No changes proposed (insufficient data or no improvements found).")
    return delta


def main() -> None:
    parser = argparse.ArgumentParser(description="Run trading performance agents")
    parser.add_argument(
        "--agent",
        choices=["policy", "threshold", "exit", "all"],
        default="all",
        help="Which agent to run (default: all)",
    )
    parser.add_argument(
        "--context",
        choices=list(EXAMPLE_SIGNAL_CONTEXTS.keys()),
        default="normal",
        help="Example signal context for --agent policy (default: normal)",
    )
    parser.add_argument(
        "--db",
        default="trader.db",
        help="Path to trader.db (default: trader.db)",
    )
    parser.add_argument(
        "--strategy",
        default="quick_pop_chart_ml",
        help="Strategy to analyze (default: quick_pop_chart_ml)",
    )
    args = parser.parse_args()

    results = {}

    if args.agent in ("policy", "all"):
        results["policy"] = run_policy(args.strategy, args.context)

    if args.agent in ("threshold", "all"):
        results["threshold"] = run_threshold(args.db, args.strategy)

    if args.agent in ("exit", "all"):
        results["exit"] = run_exit(args.db, args.strategy)

    print(f"\n{'='*60}")
    print("FULL DELTA JSON (apply to registry.py manually after review):")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
