"""
trader/agents/threshold.py — ML threshold optimizer (Agent B).

Analyzes score-bucket performance from chart_snapshots and proposes
adjustments to the ML gating thresholds (ml_min_score,
ml_high_score_threshold, ml_max_score_threshold) and size multipliers.

Usage:
    from trader.agents.threshold import run
    delta = run(db_path="trader.db", current_config={...})
    # delta is validated against GUARDRAILS before being returned

Architecture note:
    This function calls Claude Haiku once, reads the response JSON, then
    runs it through validate_delta() before returning. The agent never
    touches the DB directly — it only sees the summary data passed in the
    prompt.
"""

from __future__ import annotations

import json
import logging
import os

import anthropic

from trader.agents.base import (
    GUARDRAILS,
    query_score_buckets,
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
    Query score-bucket performance, ask Claude Haiku for threshold adjustments,
    validate against guardrails, and return the clamped delta.

    Parameters
    ----------
    db_path : str
        Path to trader.db.
    strategy : str
        Strategy name to analyze (must have chart_snapshots rows).
    current_config : dict | None
        Current ML threshold values. If None, guardrail midpoints are used.
        Expected keys: ml_min_score, ml_high_score_threshold, ml_max_score_threshold,
                        ml_size_multiplier, ml_max_size_multiplier

    Returns
    -------
    dict
        Validated delta with any subset of the ML threshold keys plus a
        "reason" string from the agent.
        Returns empty dict if fewer than 20 closed snapshots exist.
    """
    buckets = query_score_buckets(db_path, strategy)
    total_trades = sum(b["count"] for b in buckets)

    if total_trades < 20:
        logger.info(
            "[threshold_agent] Only %d closed snapshots for %s — need 20 to run",
            total_trades, strategy,
        )
        return {}

    recent = query_recent_trades(db_path, strategy, limit=20)

    # Use midpoints if no current config provided
    if current_config is None:
        current_config = {k: (lo + hi) / 2 for k, (lo, hi) in GUARDRAILS.items()}

    prompt = _build_prompt(buckets, recent, current_config, total_trades)

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set — required for threshold agent")

    client = anthropic.Anthropic(api_key=api_key)

    logger.info("[threshold_agent] Calling %s with %d buckets (%d trades)", _MODEL, len(buckets), total_trades)

    message = client.messages.create(
        model=_MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = message.content[0].text.strip()
    logger.debug("[threshold_agent] Raw response: %s", raw_text)

    delta = _parse_response(raw_text)
    validated = validate_delta(delta)

    logger.info("[threshold_agent] Proposed delta (after guardrails): %s", validated)
    return validated


def _build_prompt(
    buckets: list[dict],
    recent: list[dict],
    current_config: dict,
    total_trades: int,
) -> str:
    buckets_text = json.dumps(buckets, indent=2)
    recent_text = json.dumps(recent[-10:], indent=2)  # last 10 for brevity
    guardrails_text = summarise_guardrails()

    current_text = "\n".join(
        f"  {k}: {current_config.get(k, 'N/A')}"
        for k in [
            "ml_min_score",
            "ml_high_score_threshold",
            "ml_max_score_threshold",
            "ml_size_multiplier",
            "ml_max_size_multiplier",
        ]
    )

    return f"""You are a parameter tuner for a Solana paper trading bot.
The bot uses a KNN ML scorer (0–10) to gate entries on the quick_pop_chart_ml strategy.
Scores below ml_min_score → skip trade.
Scores >= ml_high_score_threshold → double position size.
Scores >= ml_max_score_threshold → triple position size.

TOTAL CLOSED TRADES ANALYZED: {total_trades}

SCORE-BUCKET PERFORMANCE (each bucket is a 1-point range):
{buckets_text}

LAST 10 TRADES (most recent first):
{recent_text}

CURRENT CONFIGURATION:
{current_text}

ALLOWED ADJUSTMENT BANDS (Python enforces these — you cannot exceed them):
{guardrails_text}

TASK:
Analyze the score-bucket data. Focus on:
1. Which buckets have consistently negative avg_pnl_pct and low win_rate → raise ml_min_score to exclude them
2. Which bucket floor reliably marks high-confidence trades → set ml_high_score_threshold there
3. Whether the current size multipliers match actual outperformance

Return ONLY a JSON object with the keys you want to change and a "reason" string.
Do not include keys you are not changing.
Example: {{"ml_min_score": 5.5, "reason": "Buckets 5.0-5.9 show 28% win rate — raising floor reduces losers without losing volume"}}

Respond with valid JSON only. No markdown, no explanation outside the JSON."""


def _parse_response(raw: str) -> dict:
    """Extract the JSON object from the agent response."""
    # Strip markdown code fences if present
    text = raw
    if "```" in text:
        start = text.find("{")
        end = text.rfind("}") + 1
        text = text[start:end]

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error("[threshold_agent] Failed to parse response as JSON: %s\nRaw: %s", exc, raw)
        return {}
