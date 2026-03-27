#!/usr/bin/env python3
"""
scripts/check_shadow_override.py — Shadow AI override accuracy report.

Shadow decisions are about SKIPPED signals (ML_SKIP / CHART_SKIP / POLICY_BLK),
so strategy_outcomes will never be entered/closed for them. Instead we use the
price history (peak_pnl_pct / trough_pnl_pct from signal_charts) together with
each strategy's stop_loss_pct and TP multiple to estimate what would have happened
if the trade had been taken.

Verdict logic (conservative — stop loss wins if both would have been hit):
  SHADOW_OVERRIDE (AI wanted to buy a skipped signal):
    ✓ RIGHT  → peak would have hit TP before trough hit stop
    ✗ WRONG  → trough would have hit stop loss first
  SHADOW_REJECT (AI agreed to skip):
    ✓ RIGHT  → trough would have hit stop loss (AI saved money by agreeing to skip)
    ✗ WRONG  → peak would have hit TP (AI agreed to skip something that would have won)

Usage:
    python scripts/check_shadow_override.py
    python scripts/check_shadow_override.py --strategy quick_pop_managed
    python scripts/check_shadow_override.py --limit 50
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

DB_PATH     = os.getenv("DB_PATH", "trader.db")
CONFIG_PATH = Path(__file__).parent.parent / "strategy_config.json"
SEP = "=" * 80


def _load_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text())
    except Exception:
        return {}


def _strategy_params(cfg: dict, strategy: str) -> tuple[float, float]:
    """Return (stop_loss_pct, tp_pct) for a strategy."""
    s = cfg.get(strategy, {})
    stop_loss_pct = s.get("stop_loss_pct", 0.30)
    tp_levels     = s.get("tp_levels", [[2.0, 1.0]])
    tp_multiple   = tp_levels[0][0] if tp_levels else 2.0
    tp_pct        = (tp_multiple - 1.0) * 100.0
    return stop_loss_pct * 100.0, tp_pct


def _verdict(decision: str, peak_pnl: float | None, trough_pnl: float | None,
             stop_pct: float, tp_pct: float) -> str:
    """Estimate correctness using price history. Conservative: stop wins if both hit."""
    if peak_pnl is None or trough_pnl is None:
        return "pending"

    would_stop = trough_pnl <= -stop_pct
    would_tp   = peak_pnl   >= tp_pct

    if would_stop and would_tp:
        outcome = "STOP"   # conservative: stop wins
    elif would_stop:
        outcome = "STOP"
    elif would_tp:
        outcome = "TP"
    else:
        outcome = "TIMEOUT"  # neither hit — expired at snapshot price

    if decision == "SHADOW_OVERRIDE":
        # AI wanted to buy — was it right?
        correct = outcome in ("TP",)
        return ("✓ RIGHT " if correct else "✗ WRONG ") + f"({outcome})"
    else:
        # AI agreed to skip — was it right to skip?
        correct = outcome in ("STOP", "TIMEOUT")
        return ("✓ RIGHT " if correct else "✗ WRONG ") + f"({outcome})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Shadow AI override accuracy report")
    parser.add_argument("--strategy", default=None)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    if not os.path.exists(DB_PATH):
        print(f"[ERROR] Database not found: {DB_PATH}")
        sys.exit(1)

    cfg = _load_config()
    conn = sqlite3.connect(DB_PATH)
    strat_filter = f"AND aod.strategy = '{args.strategy}'" if args.strategy else ""

    # ------------------------------------------------------------------
    # 1. Per-decision rows with price history
    # ------------------------------------------------------------------
    rows = conn.execute(
        f"""
        SELECT aod.ts, aod.strategy, aod.symbol, aod.decision,
               aod.skip_reason, aod.ml_score, aod.pump_ratio,
               sc.peak_pnl_pct, sc.trough_pnl_pct,
               sc.price_tracking_done,
               aod.agent_reason
          FROM ai_override_decisions aod
          LEFT JOIN signal_charts sc ON sc.id = aod.signal_chart_id
         WHERE aod.decision IN ('SHADOW_OVERRIDE', 'SHADOW_REJECT')
               {strat_filter}
         ORDER BY aod.ts DESC
         LIMIT ?
        """,
        (args.limit,),
    ).fetchall()

    print(f"\n{SEP}")
    print("  SHADOW DECISIONS — individual calls  (price history verdict)")
    print(SEP)

    if not rows:
        print("  No shadow override decisions found.")
    else:
        hdr = f"  {'ts':16}  {'sym':10}  {'decision':16}  {'score':5}  {'peak':7}  {'trough':7}  verdict"
        print(hdr)
        print(f"  {'-'*16}  {'-'*10}  {'-'*16}  {'-'*5}  {'-'*7}  {'-'*7}  -------")

        for ts, strategy, symbol, decision, skip_reason, ml_score, pump, peak, trough, done, reason in rows:
            stop_pct, tp_pct = _strategy_params(cfg, strategy)
            verdict  = _verdict(decision, peak, trough, stop_pct, tp_pct)
            score_s  = f"{ml_score:.2f}" if ml_score is not None else "—"
            peak_s   = f"{peak:+.0f}%" if peak is not None else "—"
            trough_s = f"{trough:+.0f}%" if trough is not None else "—"
            tracking = "" if done else " [tracking]"
            print(
                f"  {str(ts)[:16]}  {str(symbol):<10}  {decision:<16}  "
                f"{score_s:<5}  {peak_s:<7}  {trough_s:<7}  {verdict}{tracking}"
            )
            if reason and decision == "SHADOW_OVERRIDE":
                print(f"      ↳ {reason[:120]}")

    # ------------------------------------------------------------------
    # 2. Accuracy summary
    # ------------------------------------------------------------------
    all_rows = conn.execute(
        f"""
        SELECT aod.strategy, aod.decision,
               sc.peak_pnl_pct, sc.trough_pnl_pct, sc.price_tracking_done
          FROM ai_override_decisions aod
          LEFT JOIN signal_charts sc ON sc.id = aod.signal_chart_id
         WHERE aod.decision IN ('SHADOW_OVERRIDE', 'SHADOW_REJECT')
               {strat_filter}
        """,
    ).fetchall()

    # Group by strategy + decision
    from collections import defaultdict
    buckets: dict[tuple, list] = defaultdict(list)
    for strategy, decision, peak, trough, done in all_rows:
        buckets[(strategy, decision)].append((peak, trough, done))

    print(f"\n{SEP}")
    print("  ACCURACY SUMMARY  (stop_loss wins if both TP and SL would have been hit)")
    print(SEP)

    for (strategy, decision), entries in sorted(buckets.items()):
        stop_pct, tp_pct = _strategy_params(cfg, strategy)
        total    = len(entries)
        resolved = sum(1 for p, t, _ in entries if p is not None and t is not None)
        pending  = total - resolved

        verdicts = [_verdict(decision, p, t, stop_pct, tp_pct) for p, t, _ in entries
                    if p is not None and t is not None]
        correct  = sum(1 for v in verdicts if v.startswith("✓"))

        accuracy = (correct / resolved * 100) if resolved else 0.0
        peak_pnls   = [p for p, t, _ in entries if p is not None]
        trough_pnls = [t for p, t, _ in entries if t is not None]

        print(f"\n  [{strategy}]  {decision}")
        print(f"    Total : {total}  |  resolved: {resolved}  |  pending: {pending}")
        print(f"    Accuracy : {correct}/{resolved}  ({accuracy:.1f}%)")
        if peak_pnls:
            avg_peak   = sum(peak_pnls) / len(peak_pnls)
            avg_trough = sum(trough_pnls) / len(trough_pnls)
            print(f"    Avg peak : {avg_peak:+.1f}%   avg trough: {avg_trough:+.1f}%")
            print(f"    TP thresh: +{tp_pct:.0f}%   SL thresh: -{stop_pct:.0f}%")
        if decision == "SHADOW_OVERRIDE":
            print(f"    → AI wanted to BUY these skipped signals. Right if they would have hit TP.")
        else:
            print(f"    → AI agreed to SKIP. Right if they would have stopped out or timed out.")

    conn.close()
    print()


if __name__ == "__main__":
    main()
