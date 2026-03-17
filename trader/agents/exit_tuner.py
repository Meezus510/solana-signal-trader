"""
trader/agents/exit_tuner.py — Exit parameter tuner (Agent D).

Analyzes per-sell-reason performance from chart_snapshots and proposes
adjustments to stop_loss_pct, trailing_stop_pct, and timeout_minutes.

Key insight this agent exploits:
- Many STOP_LOSS exits with low avg_max_gain → stop may be too wide (lower stop_loss_pct)
- Many STOP_LOSS exits with high avg_max_gain (token mooned after stop) → stop may be too tight
- TIMEOUT exits with positive avg_pnl → timeout too aggressive (raise timeout_minutes)
- TRAILING_STOP exits with low avg_pnl relative to avg_max_gain → trail too loose

Usage:
    from trader.agents.exit_tuner import run
    delta = run(db_path="trader.db", strategy="quick_pop_chart_ml", current_config={...})
"""

from __future__ import annotations

import json
import logging
import os

import anthropic

from trader.agents.base import (
    GUARDRAILS,
    query_exit_stats,
    query_recent_trades,
    summarise_guardrails,
    validate_delta,
)

logger = logging.getLogger(__name__)

_MODEL = "claude-haiku-4-5-20251001"


def run(
    db_path: str = "trader.db",
    strategy: str = "quick_pop_chart_ml",
    current_config: dict | None = None,
) -> dict:
    """
    Query exit-reason stats, ask Claude Haiku for exit parameter adjustments,
    validate against guardrails, and return the clamped delta.

    Parameters
    ----------
    db_path : str
        Path to trader.db.
    strategy : str
        Strategy name to analyze.
    current_config : dict | None
        Current exit parameter values.
        Expected keys: stop_loss_pct, trailing_stop_pct, timeout_minutes

    Returns
    -------
    dict
        Validated delta with any subset of exit parameter keys plus "reason".
        Returns empty dict if fewer than 20 closed trades exist.
    """
    exit_stats = query_exit_stats(db_path, strategy)
    total_trades = sum(s["count"] for s in exit_stats)

    if total_trades < 20:
        logger.info(
            "[exit_tuner_agent] Only %d closed trades for %s — need 20 to run",
            total_trades, strategy,
        )
        return {}

    recent = query_recent_trades(db_path, strategy, limit=20)

    if current_config is None:
        current_config = {k: (lo + hi) / 2 for k, (lo, hi) in GUARDRAILS.items()}

    prompt = _build_prompt(exit_stats, recent, current_config, total_trades, strategy)

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set — required for exit tuner agent")

    client = anthropic.Anthropic(api_key=api_key)

    logger.info("[exit_tuner_agent] Calling %s for strategy=%s (%d trades)", _MODEL, strategy, total_trades)

    message = client.messages.create(
        model=_MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = message.content[0].text.strip()
    logger.debug("[exit_tuner_agent] Raw response: %s", raw_text)

    delta = _parse_response(raw_text)
    validated = validate_delta(delta)

    logger.info("[exit_tuner_agent] Proposed delta (after guardrails): %s", validated)
    return validated


def _build_prompt(
    exit_stats: list[dict],
    recent: list[dict],
    current_config: dict,
    total_trades: int,
    strategy: str,
) -> str:
    exit_text = json.dumps(exit_stats, indent=2)
    recent_text = json.dumps(recent[-10:], indent=2)
    guardrails_text = summarise_guardrails()

    current_text = "\n".join(
        f"  {k}: {current_config.get(k, 'N/A')}"
        for k in ["stop_loss_pct", "trailing_stop_pct", "timeout_minutes"]
    )

    return f"""You are a parameter tuner for a Solana paper trading bot.
Strategy: {strategy}

EXIT CONFIGURATION:
  stop_loss_pct:    initial stop below entry price (e.g. 0.20 = −20%)
  trailing_stop_pct: trail stop below highest price seen (activates after TP1)
  timeout_minutes:  exit if position is not gaining after this many minutes

TOTAL CLOSED TRADES ANALYZED: {total_trades}

EXIT REASON BREAKDOWN:
{exit_text}

Fields:
- avg_pnl_pct: average realized PnL% (negative = losses)
- avg_hold_secs: average seconds held before this exit type
- avg_max_gain_pct: average highest unrealized gain% seen during the trade

LAST 10 TRADES (most recent first):
{recent_text}

CURRENT CONFIGURATION:
{current_text}

ALLOWED ADJUSTMENT BANDS (Python enforces these — you cannot exceed them):
{guardrails_text}

TASK:
Analyze the exit data. Diagnose the largest source of PnL leakage and propose
up to 2 parameter changes. Consider:

1. STOP_LOSS exits: if avg_max_gain_pct is high (>15%), token often ran before stopping
   → stop may be too tight. If avg_max_gain_pct is low (<5%), losses are genuine
   → stop may be too wide.
2. TRAILING_STOP exits: compare avg_pnl_pct to avg_max_gain_pct. Large gap →
   trailing stop is too loose (lower trailing_stop_pct).
3. TIMEOUT exits with positive avg_pnl_pct → timeout is cutting winners short
   (raise timeout_minutes).

Return ONLY a JSON object with the keys you want to change and a "reason" string.
Do not include keys you are not changing.
Example: {{"stop_loss_pct": 0.18, "reason": "STOP_LOSS exits show avg_max_gain_pct of 4.2% — losses are genuine, tightening stop from 0.20 to 0.18 cuts downside"}}

Respond with valid JSON only. No markdown, no explanation outside the JSON."""


def _parse_response(raw: str) -> dict:
    """Extract the JSON object from the agent response."""
    text = raw
    if "```" in text:
        start = text.find("{")
        end = text.rfind("}") + 1
        text = text[start:end]

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error("[exit_tuner_agent] Failed to parse response as JSON: %s\nRaw: %s", exc, raw)
        return {}
