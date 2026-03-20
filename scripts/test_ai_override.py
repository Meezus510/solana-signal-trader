#!/usr/bin/env python3
"""
scripts/test_ai_override.py — Live test for the AI override agent.

Runs propose_ai_override against several hand-crafted scenarios that represent
real skip situations:  strong-but-filtered, weak-and-correct, ambiguous, chart-pumped.
Prints the agent's decision for each and whether it matches the expected outcome.

Optionally loads historical context from the real DB if trader.db exists.

Usage:
    python scripts/test_ai_override.py
    python scripts/test_ai_override.py --db path/to/trader.db
    python scripts/test_ai_override.py --scenario ml_strong   # single scenario
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

from trader.agents.ai_override import propose_ai_override, summarize_candles

SEP  = "=" * 72
SEP2 = "-" * 72


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------
# Each scenario has:
#   skip_reason  : ML_SKIP | CHART_SKIP | POLICY_BLK
#   signal_context: same dict the engine builds
#   expect       : "override" | "reject" | "reanalyze" | "any"
#   label        : short description
# ---------------------------------------------------------------------------

def _candles_rising(n=8):
    """Simulate steadily rising price candles."""
    base = 0.00100
    return [
        {"open": base * (1 + i * 0.02), "close": base * (1 + (i + 1) * 0.02), "volume": 500 + i * 50}
        for i in range(n)
    ]

def _candles_dying(n=8):
    """Simulate a pump that's fading — volume collapsing."""
    base = 0.00300
    return [
        {"open": base * (1 - i * 0.03), "close": base * (1 - (i + 0.5) * 0.03), "volume": max(10, 800 - i * 100)}
        for i in range(n)
    ]

def _candles_flat(n=8):
    """Flat price, moderate volume."""
    base = 0.00200
    return [{"open": base, "close": base * 1.005, "volume": 300} for _ in range(n)]


SCENARIOS: list[dict] = [
    # -------------------------------------------------------------------------
    # 1. ML_SKIP — score slightly below threshold but chart looks strong
    #    Strong buy pressure, rising volume, pump not excessive
    #    → Expected: OVERRIDE
    # -------------------------------------------------------------------------
    {
        "label":       "ml_strong",
        "description": "ML_SKIP but strong chart + high buys — should override",
        "skip_reason": "ML_SKIP",
        "expect":      "override",
        "signal_context": {
            "ml_score":       2.1,   # just below threshold of 2.5
            "ml_min_score":   2.5,
            "pump_ratio":     1.9,   # modest pump
            "pump_ratio_max": 3.5,
            "vol_trend":      "RISING",
            "chart_reason":   "pump=1.9x — vol RISING",
            "ml_source":      "moralis/10s",
            "source_channel": "WizzyTrades",
            "pair_stats": {
                "buys_5m":               48,
                "sells_5m":              12,
                "price_change_5m_pct":   18.5,
                "liquidity_change_1h_pct": 12.0,
                "buy_volume_1h":         8500,
                "total_volume_1h":       11000,
            },
            "candles_summary": summarize_candles(_candles_rising()),
        },
    },

    # -------------------------------------------------------------------------
    # 2. ML_SKIP — score well below threshold, volume dying
    #    → Expected: REJECT
    # -------------------------------------------------------------------------
    {
        "label":       "ml_weak",
        "description": "ML_SKIP with dying volume and low score — should reject",
        "skip_reason": "ML_SKIP",
        "expect":      "reject",
        "signal_context": {
            "ml_score":       0.7,   # well below threshold
            "ml_min_score":   2.5,
            "pump_ratio":     2.8,
            "pump_ratio_max": 3.5,
            "vol_trend":      "DYING",
            "chart_reason":   "pump=2.8x — vol DYING",
            "ml_source":      "moralis/10s",
            "source_channel": "WizzyTrades",
            "pair_stats": {
                "buys_5m":               6,
                "sells_5m":              31,
                "price_change_5m_pct":   -8.2,
                "liquidity_change_1h_pct": -15.0,
                "buy_volume_1h":         900,
                "total_volume_1h":       4200,
            },
            "candles_summary": summarize_candles(_candles_dying()),
        },
    },

    # -------------------------------------------------------------------------
    # 3. CHART_SKIP — pump slightly over limit but strong fresh buys
    #    Volume still rising, lots of buy pressure — second leg possible
    #    → Expected: override (debatable, but agent should lean toward it)
    # -------------------------------------------------------------------------
    {
        "label":       "chart_fresh_pump",
        "description": "CHART_SKIP — pump just over limit but very fresh buys",
        "skip_reason": "CHART_SKIP",
        "expect":      "any",
        "signal_context": {
            "ml_score":       3.8,
            "ml_min_score":   2.5,
            "pump_ratio":     3.7,   # just over 3.5 max
            "pump_ratio_max": 3.5,
            "vol_trend":      "RISING",
            "chart_reason":   "pump=3.7x >= 3.5x",
            "ml_source":      "moralis/10s",
            "source_channel": "WizzyTrades",
            "pair_stats": {
                "buys_5m":               65,
                "sells_5m":              9,
                "price_change_5m_pct":   32.0,
                "liquidity_change_1h_pct": 25.0,
                "buy_volume_1h":         18000,
                "total_volume_1h":       22000,
            },
            "candles_summary": summarize_candles(_candles_rising()),
        },
    },

    # -------------------------------------------------------------------------
    # 4. CHART_SKIP — massive pump + dying volume
    #    Classic "already over" scenario
    #    → Expected: REJECT
    # -------------------------------------------------------------------------
    {
        "label":       "chart_dead",
        "description": "CHART_SKIP — large pump + dying volume — should reject",
        "skip_reason": "CHART_SKIP",
        "expect":      "reject",
        "signal_context": {
            "ml_score":       None,
            "ml_min_score":   2.5,
            "pump_ratio":     5.2,
            "pump_ratio_max": 3.5,
            "vol_trend":      "DYING",
            "chart_reason":   "pump=5.2x >= 3.5x + vol dying",
            "ml_source":      "moralis/10s",
            "source_channel": "WizzyCasino",
            "pair_stats": {
                "buys_5m":               3,
                "sells_5m":              44,
                "price_change_5m_pct":   -22.0,
                "liquidity_change_1h_pct": -30.0,
                "buy_volume_1h":         400,
                "total_volume_1h":       8800,
            },
            "candles_summary": summarize_candles(_candles_dying()),
        },
    },

    # -------------------------------------------------------------------------
    # 5. ML_SKIP — no candle data, pair stats unavailable
    #    → Expected: REANALYZE (incomplete data)
    # -------------------------------------------------------------------------
    {
        "label":       "ml_nodata",
        "description": "ML_SKIP — missing pair stats and candles — should reanalyze",
        "skip_reason": "ML_SKIP",
        "expect":      "reanalyze",
        "signal_context": {
            "ml_score":       1.9,
            "ml_min_score":   2.5,
            "pump_ratio":     2.2,
            "pump_ratio_max": 3.5,
            "vol_trend":      "FLAT",
            "chart_reason":   "pump=2.2x — vol FLAT",
            "ml_source":      "birdeye/1m",   # fallback — lower quality
            "source_channel": "WizzyTrades",
            "pair_stats":     None,           # unavailable
            "candles_summary": {},            # no candles
        },
    },

    # -------------------------------------------------------------------------
    # 6. POLICY_BLK — blocked for data quality (birdeye fallback + no pair stats)
    #    Chart itself looks fine
    #    → Expected: override (data block, not fundamental issue)
    # -------------------------------------------------------------------------
    {
        "label":       "policy_data_quality",
        "description": "POLICY_BLK — data quality block, chart fine — may override",
        "skip_reason": "POLICY_BLK",
        "expect":      "any",
        "signal_context": {
            "ml_score":       4.1,
            "ml_min_score":   2.5,
            "pump_ratio":     2.0,
            "pump_ratio_max": 3.5,
            "vol_trend":      "RISING",
            "chart_reason":   "pump=2.0x — vol RISING",
            "ml_source":      "birdeye/1m",
            "source_channel": "WizzyTrades",
            "pair_stats":     None,
            "candles_summary": summarize_candles(_candles_flat()),
        },
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_scenario(scenario: dict, db_path: str | None, training_strategy: str) -> dict:
    """Call the agent and return the decision + metadata."""
    t0 = time.monotonic()
    decision = propose_ai_override(
        skip_reason=scenario["skip_reason"],
        signal_context=scenario["signal_context"],
        strategy="quick_pop_managed",
        db_path=db_path,
        training_strategy=training_strategy,
    )
    elapsed = time.monotonic() - t0
    return {**decision, "_elapsed_ms": round(elapsed * 1000)}


def check_expectation(decision: dict, expect: str) -> tuple[bool, str]:
    """Return (passed, label) based on expected vs actual decision."""
    if expect == "any":
        return True, "✓ (any)"
    if expect == "override" and decision["override"]:
        return True, "✓"
    if expect == "reject" and not decision["override"] and decision["reanalyze_after_seconds"] == 0.0:
        return True, "✓"
    if expect == "reanalyze" and not decision["override"] and decision["reanalyze_after_seconds"] > 0:
        return True, "✓"
    actual = "OVERRIDE" if decision["override"] else (
        f"REANALYZE({decision['reanalyze_after_seconds']:.0f}s)"
        if decision["reanalyze_after_seconds"] > 0 else "REJECT"
    )
    return False, f"✗  expected {expect.upper()} got {actual}"


def print_scenario_result(scenario: dict, decision: dict) -> bool:
    passed, check_label = check_expectation(decision, scenario["expect"])

    action = "OVERRIDE" if decision["override"] else (
        f"REANALYZE in {decision['reanalyze_after_seconds']:.0f}s"
        if decision["reanalyze_after_seconds"] > 0 else "REJECT"
    )

    ctx = scenario["signal_context"]
    ml  = ctx.get("ml_score")
    ml_str = f"{ml:.2f}" if ml is not None else "None"

    print(f"\n{SEP2}")
    print(f"  Scenario : {scenario['label']}")
    print(f"  Desc     : {scenario['description']}")
    print(f"  Skip     : {scenario['skip_reason']}  |  ml_score={ml_str}  |  pump={ctx.get('pump_ratio')}x  |  vol={ctx.get('vol_trend')}")
    print(f"  Decision : {action}")
    print(f"  Reason   : {decision['reason']}")
    print(f"  Expect   : {scenario['expect'].upper():10}  {check_label}  ({decision['_elapsed_ms']}ms)")
    return passed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Live test for the AI override agent")
    parser.add_argument("--db", default=os.getenv("DB_PATH", "trader.db"),
                        help="Path to trader.db for historical context")
    parser.add_argument("--training-strategy", default="quick_pop",
                        help="Base strategy for historical context (default: quick_pop)")
    parser.add_argument("--scenario", default=None,
                        help="Run a single scenario by label (e.g. ml_strong)")
    args = parser.parse_args()

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        print("[ERROR] ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    db_path = args.db if os.path.exists(args.db) else None
    if db_path:
        print(f"[DB] Using historical context from: {db_path}")
    else:
        print(f"[DB] {args.db} not found — running without historical context")

    scenarios = SCENARIOS
    if args.scenario:
        scenarios = [s for s in SCENARIOS if s["label"] == args.scenario]
        if not scenarios:
            labels = [s["label"] for s in SCENARIOS]
            print(f"[ERROR] Unknown scenario '{args.scenario}'. Available: {labels}", file=sys.stderr)
            sys.exit(1)

    print(f"\n{SEP}")
    print(f"  AI Override Agent — live test  ({len(scenarios)} scenario(s))")
    print(f"  Model: claude-haiku-4-5-20251001  |  strategy: quick_pop_managed")
    print(SEP)

    passed_count = 0
    total_ms = 0

    for scenario in scenarios:
        decision = run_scenario(scenario, db_path, args.training_strategy)
        ok = print_scenario_result(scenario, decision)
        if ok:
            passed_count += 1
        total_ms += decision["_elapsed_ms"]

    print(f"\n{SEP}")
    print(f"  Results : {passed_count}/{len(scenarios)} matched expectations")
    print(f"  Total   : {total_ms}ms  |  avg {total_ms // len(scenarios)}ms per call")
    print(SEP)

    if passed_count < len(scenarios):
        sys.exit(1)


if __name__ == "__main__":
    main()
