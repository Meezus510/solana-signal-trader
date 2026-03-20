#!/usr/bin/env python3
"""
scripts/ask_agent.py — Ask the trading AI a free-form question about the system.

Gathers live stats from trader.db (trade history, ML scores, config, agent log)
and sends them to Claude with your question.  Useful for getting a holistic
review, data gaps, or recommendations without having to parse raw DB output.

Usage:
    python scripts/ask_agent.py
    python scripts/ask_agent.py --question "What should I do next to improve results?"
    python scripts/ask_agent.py --question "Is the ML filter helping or hurting?"
    python scripts/ask_agent.py --question "Which strategy is performing best and why?"
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

DB_PATH    = os.getenv("DB_PATH", "trader.db")
CONFIG_PATH = Path("strategy_config.json")
LOG_PATH    = Path(os.getenv("AGENT_LOG_PATH", "logs/agent_actions.log"))
SEP = "=" * 72

DEFAULT_QUESTION = (
    "Based on everything you can see about this trading system — "
    "the trade history, ML scores, config changes, and signal outcomes — "
    "give me a full review: what is working, what is not, what data gaps "
    "are limiting your accuracy, and your top 3 concrete recommendations "
    "to improve performance."
)


# ---------------------------------------------------------------------------
# Data gathering
# ---------------------------------------------------------------------------

def gather_trade_summary(conn: sqlite3.Connection) -> dict:
    rows = conn.execute(
        """
        SELECT strategy,
               COUNT(*)                                                  AS total,
               SUM(CASE WHEN realized_pnl_usd > 0 THEN 1 ELSE 0 END)   AS wins,
               ROUND(SUM(realized_pnl_usd), 4)                          AS total_pnl,
               ROUND(AVG(realized_pnl_usd), 4)                          AS avg_pnl,
               ROUND(MIN(realized_pnl_usd), 4)                          AS worst,
               ROUND(MAX(realized_pnl_usd), 4)                          AS best
          FROM positions
         WHERE status = 'CLOSED'
         GROUP BY strategy
         ORDER BY strategy
        """
    ).fetchall()
    return [
        {
            "strategy":  r[0],
            "trades":    r[1],
            "wins":      r[2],
            "win_rate":  round(r[2] / r[1] * 100, 1) if r[1] else 0,
            "total_pnl": r[3],
            "avg_pnl":   r[4],
            "worst":     r[5],
            "best":      r[6],
        }
        for r in rows
    ]


def gather_ml_summary(conn: sqlite3.Connection) -> dict:
    scored = conn.execute(
        """
        SELECT so.strategy,
               COUNT(*)                                              AS total,
               SUM(CASE WHEN so.entered = 1 THEN 1 ELSE 0 END)     AS entered,
               SUM(CASE WHEN so.entered = 0 THEN 1 ELSE 0 END)     AS skipped,
               ROUND(MIN(sc.ml_score), 2)                           AS min_score,
               ROUND(MAX(sc.ml_score), 2)                           AS max_score,
               ROUND(AVG(sc.ml_score), 2)                           AS avg_score
          FROM strategy_outcomes so
          JOIN signal_charts sc ON so.signal_chart_id = sc.id
         WHERE sc.ml_score IS NOT NULL
         GROUP BY so.strategy
        """
    ).fetchall()
    return [
        {
            "strategy": r[0], "total": r[1], "entered": r[2],
            "skipped":  r[3], "min_score": r[4], "max_score": r[5], "avg_score": r[6],
        }
        for r in scored
    ]


def gather_training_data(conn: sqlite3.Connection) -> dict:
    rows = conn.execute(
        """
        SELECT so.strategy,
               COUNT(*)                                                     AS total,
               SUM(CASE WHEN so.outcome_pnl_pct > 0 THEN 1 ELSE 0 END)    AS wins,
               ROUND(AVG(so.outcome_pnl_pct), 2)                           AS avg_pnl,
               ROUND(MIN(so.outcome_pnl_pct), 2)                           AS worst,
               ROUND(MAX(so.outcome_pnl_pct), 2)                           AS best,
               MIN(sc.ts), MAX(sc.ts)
          FROM strategy_outcomes so
          JOIN signal_charts sc ON so.signal_chart_id = sc.id
         WHERE so.closed = 1
           AND so.outcome_pnl_pct IS NOT NULL
         GROUP BY so.strategy
        """
    ).fetchall()
    return [
        {
            "strategy":  r[0],
            "rows":      r[1],
            "wins":      r[2],
            "win_rate":  round(r[2] / r[1] * 100, 1) if r[1] else 0,
            "avg_pnl":   r[3],
            "worst":     r[4],
            "best":      r[5],
            "date_from": (r[6] or "")[:16],
            "date_to":   (r[7] or "")[:16],
        }
        for r in rows
    ]


def gather_recent_signals(conn: sqlite3.Connection, limit: int = 20) -> list:
    rows = conn.execute(
        """
        SELECT sc.ts, sc.symbol, so.strategy, sc.ml_score, so.entered,
               so.outcome_pnl_pct, so.outcome_sell_reason,
               ROUND(LENGTH(sc.candles_json)*1.0/MAX(sc.candle_count,1)) AS bpc
          FROM strategy_outcomes so
          JOIN signal_charts sc ON so.signal_chart_id = sc.id
         WHERE so.closed = 1
         ORDER BY sc.ts DESC
         LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [
        {
            "ts":         (r[0] or "")[:16],
            "symbol":     r[1],
            "strategy":   r[2],
            "ml_score":   r[3],
            "entered":    bool(r[4]),
            "pnl_pct":    r[5],
            "sell_reason": r[6],
            "candle_src": "moralis-10s" if r[7] and r[7] > 400 else "birdeye-1m",
        }
        for r in rows
    ]


def gather_agent_log(max_lines: int = 30) -> list[str]:
    if not LOG_PATH.exists():
        return ["(no agent actions logged yet)"]
    lines = LOG_PATH.read_text().strip().splitlines()
    return lines[-max_lines:]


def load_current_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open() as f:
        cfg = json.load(f)
    cfg.pop("_meta", None)
    return cfg


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

AUTONOMOUS_CONTROL = """
===== WHAT THE STRATEGY TUNER CONTROLS AUTONOMOUSLY =====
The strategy_tuner agent runs on a schedule and writes directly to strategy_config.json.
It does NOT need human approval. When you recommend changes, clearly distinguish:
  - AUTONOMOUS: things the tuner will handle itself on its next run
  - MANUAL: things that require a human to act (e.g. code changes, infrastructure)

STRATEGY PERMISSIONS:
  quick_pop, trend_rider, infinite_moonbag (BASE strategies)
    → NOT autonomously controlled — changes require manual edits by a human.
      The tuner uses their closed trade outcomes as ML training data only.

  trend_rider_managed, moonbag_managed (CHART variants — FULL CONTROL)
    → Can autonomously change: TP levels, stop_loss_pct, trailing_stop_pct,
      timeout_minutes, pump_ratio_max, use_reanalyze + all delay params,
      use_ml_filter (CAN ENABLE ML FILTERING), all ML params, live_trading

  quick_pop_managed (ML variant — ML ONLY)
    → Can autonomously change: ml_min_score, ml_high/max_score_threshold,
      ml_size_multiplier, ml_max_size_multiplier, ml_k, ml_halflife_days,
      ml_score_low_pct, ml_score_high_pct, live_trading
    → CANNOT change: TP levels, stop_loss_pct, trailing_stop_pct, chart/reanalyze settings

KEY AUTONOMOUS LEVERS:
  live_trading      — can be set true/false for any controlled strategy at any time
  use_ml_filter     — can be enabled for chart variants when >= 20 scored trades exist
  use_chart_filter  — can be toggled for chart variants based on filter performance
  use_reanalyze     — can be enabled/disabled based on whether re-entries outperform

GUARDRAIL BANDS (tuner is clamped to these ranges):
  ml_min_score:          [2.0, 7.0]
  stop_loss_pct:         [0.10, 0.35]
  trailing_stop_pct:     [0.10, 0.40]
  timeout_minutes:       [20.0, 120.0]
"""


def build_prompt(data: dict, question: str) -> str:
    return f"""You are the AI analyst for a live Solana memecoin trading bot.
Below is a complete snapshot of the system state. Answer the user's question
based only on this data — be specific, quantitative, and actionable.
{AUTONOMOUS_CONTROL}
===== CURRENT CONFIG =====
{json.dumps(data['config'], indent=2)}

===== TRADE PERFORMANCE BY STRATEGY =====
{json.dumps(data['trade_summary'], indent=2)}

===== ML SCORE SUMMARY (scored signals only) =====
{json.dumps(data['ml_summary'], indent=2)}

===== ML TRAINING DATA BY STRATEGY =====
{json.dumps(data['training_data'], indent=2)}

===== LAST 20 CLOSED SIGNALS =====
{json.dumps(data['recent_signals'], indent=2)}

===== LAST 30 AGENT ACTIONS (config changes made autonomously) =====
{chr(10).join(data['agent_log'])}

===== USER QUESTION =====
{question}
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask the trading AI a free-form question about the system."
    )
    parser.add_argument(
        "--question", "-q",
        default=DEFAULT_QUESTION,
        help="What to ask (default: full system review)",
    )
    parser.add_argument(
        "--db", default=DB_PATH,
        help=f"Path to trader.db (default: {DB_PATH})",
    )
    args = parser.parse_args()

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        print("[ERROR] ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.db):
        print(f"[ERROR] Database not found: {args.db}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(args.db)
    data = {
        "config":         load_current_config(),
        "trade_summary":  gather_trade_summary(conn),
        "ml_summary":     gather_ml_summary(conn),
        "training_data":  gather_training_data(conn),
        "recent_signals": gather_recent_signals(conn),
        "agent_log":      gather_agent_log(),
    }
    conn.close()

    prompt = build_prompt(data, args.question)

    print(f"\n{SEP}")
    print(f"  Asking Claude about: {args.question[:80]}{'...' if len(args.question) > 80 else ''}")
    print(f"{SEP}\n")

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )

    print(message.content[0].text)
    print()


if __name__ == "__main__":
    main()
