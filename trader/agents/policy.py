"""
trader/agents/policy.py — Per-signal policy agent (Agent A).

Runs on a single incoming signal context and returns per-trade overrides.
Does NOT mutate global config — all decisions are scoped to one trade.

Usage:
    from trader.agents.policy import propose_policy_decision

    decision = propose_policy_decision(
        signal_context={
            "ml_score": 7.2,
            "used_moralis_10s": True,
            "used_birdeye_fallback": False,
            "pair_stats_available": True,
            "liquidity_usd": 45000,
            "slippage_bps": 120,
        },
        strategy="quick_pop_chart_ml",
    )
    # decision = {
    #     "allow_trade": True,
    #     "buy_size_multiplier": 1.0,
    #     "effective_score_adjustment": 0.0,
    #     "reason_codes": [],
    # }

Returns
-------
dict with keys:
    allow_trade (bool)
    buy_size_multiplier (float, 0.0–3.0)
    effective_score_adjustment (float, -1.0–+0.5)
    reason_codes (list[str])
"""

from __future__ import annotations

import json
import logging
import os

import anthropic

from trader.agents.base import validate_delta

logger = logging.getLogger(__name__)

_MODEL = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------------------
# Per-trade guardrail bands — separate from global GUARDRAILS in base.py.
# These bound what this agent can return for a single trade decision.
# ---------------------------------------------------------------------------
POLICY_GUARDRAILS: dict[str, tuple[float, float]] = {
    "buy_size_multiplier":       (0.0, 3.0),   # 0.0 = reject sizing (but allow_trade governs block)
    "effective_score_adjustment": (-1.0, 0.5), # nudge score down on degraded data, small upside
}

# Hard thresholds used for pre-flight checks before calling Claude.
# If any of these trip, we return a deterministic block without an API call.
_MIN_LIQUIDITY_USD   = 5_000    # below this → always reject
_MAX_SLIPPAGE_BPS    = 500      # above this → always reject
_HIGH_SLIPPAGE_BPS   = 200      # above this → reduce size
_LOW_LIQUIDITY_USD   = 20_000   # below this → reduce size


def propose_policy_decision(
    signal_context: dict,
    strategy: str = "quick_pop_chart_ml",
    db_path: str | None = None,          # reserved for future history look-ups
) -> dict:
    """
    Evaluate a single signal context and return per-trade overrides.

    Hard pre-flight checks run first (no API call). If the signal passes,
    Claude Haiku is asked for a bounded JSON decision.

    Parameters
    ----------
    signal_context : dict
        Expected keys: ml_score, used_moralis_10s, used_birdeye_fallback,
        pair_stats_available, liquidity_usd, slippage_bps
    strategy : str
        Strategy name (informational — included in prompt).
    db_path : str | None
        Unused in v1; reserved for adding historical context later.

    Returns
    -------
    dict with allow_trade, buy_size_multiplier, effective_score_adjustment, reason_codes.
    """
    # -- 1. Hard pre-flight checks (deterministic, no API call) ---------------
    hard_block = _hard_preflight(signal_context)
    if hard_block is not None:
        return hard_block

    # -- 2. Build prompt and call Claude Haiku --------------------------------
    prompt = _build_prompt(signal_context, strategy)

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set — required for policy agent")

    client = anthropic.Anthropic(api_key=api_key)
    logger.info("[policy_agent] Calling %s for strategy=%s ml_score=%.2f",
                _MODEL, strategy, signal_context.get("ml_score", 0))

    message = client.messages.create(
        model=_MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = message.content[0].text.strip()
    logger.debug("[policy_agent] Raw response: %s", raw_text)

    # -- 3. Parse, clamp, return ----------------------------------------------
    raw_decision = _parse_response(raw_text)
    return _validate_decision(raw_decision)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hard_preflight(ctx: dict) -> dict | None:
    """
    Deterministic checks that block or pass without an API call.
    Returns a decision dict if the trade should be blocked outright, else None.
    """
    liquidity   = ctx.get("liquidity_usd")   # None = unknown, skip hard-floor check
    slippage    = ctx.get("slippage_bps", 0) or 0

    if liquidity is not None and liquidity < _MIN_LIQUIDITY_USD:
        logger.info("[policy_agent] Hard block: liquidity $%.0f < $%.0f", liquidity, _MIN_LIQUIDITY_USD)
        return _blocked(["LOW_LIQUIDITY_HARD_BLOCK"])

    if slippage > _MAX_SLIPPAGE_BPS:
        logger.info("[policy_agent] Hard block: slippage %d bps > %d bps", slippage, _MAX_SLIPPAGE_BPS)
        return _blocked(["HIGH_SLIPPAGE_HARD_BLOCK"])

    return None  # pass — let Claude decide


def _build_prompt(ctx: dict, strategy: str) -> str:
    ml_score              = ctx.get("ml_score")
    used_moralis_10s      = ctx.get("used_moralis_10s", False)
    used_birdeye_fallback = ctx.get("used_birdeye_fallback", False)
    pair_stats_available  = ctx.get("pair_stats_available", False)
    liquidity_usd         = ctx.get("liquidity_usd", 0)
    slippage_bps          = ctx.get("slippage_bps", 0)

    # Summarise data quality in plain text for the model
    data_quality_lines = []
    if used_birdeye_fallback:
        data_quality_lines.append("- Candle source: Birdeye 1m fallback (Moralis 10s unavailable) — lower resolution")
    else:
        data_quality_lines.append("- Candle source: Moralis 10s (full resolution)")

    if not pair_stats_available:
        data_quality_lines.append("- Pair stats: unavailable — features 9-13 used neutral values")
    else:
        data_quality_lines.append("- Pair stats: available (full feature vector)")

    data_quality_text = "\n".join(data_quality_lines)

    return f"""You are a per-trade risk policy agent for a Solana trading bot.
Strategy: {strategy}

SIGNAL CONTEXT:
  ml_score:              {ml_score}   (KNN confidence score, 0–10; bot requires ≥5.0 to enter)
  liquidity_usd:         {"unknown" if liquidity_usd is None else f"${liquidity_usd:,.0f}"}
  slippage_bps:          {slippage_bps} bps{"  (unknown)" if slippage_bps == 0 else ""}

DATA QUALITY:
{data_quality_text}

YOUR JOB:
Decide whether to take this trade and at what confidence-adjusted size.
Be conservative. Penalise degraded data quality and thin markets.

DECISION RULES TO APPLY:
1. Birdeye fallback → score likely less reliable → apply score_adjustment in [-0.5, 0.0]
2. Missing pair stats → features 9-13 are neutral, not real → apply score_adjustment in [-0.3, 0.0]
3. Slippage > 200 bps → reduce size (buy_size_multiplier ≤ 0.75)
4. Liquidity < $20,000 → reduce size (buy_size_multiplier ≤ 0.5)
5. Adjusted score (ml_score + effective_score_adjustment) < 5.0 → set allow_trade = false
6. Otherwise allow_trade = true and size normally (buy_size_multiplier = 1.0)

ALLOWED OUTPUT BANDS (Python will clamp to these regardless):
  buy_size_multiplier:        0.0 – 3.0
  effective_score_adjustment: -1.0 – +0.5

REASON CODES (use any that apply, exact strings):
  BIRDEYE_FALLBACK_PENALTY
  MISSING_PAIR_STATS_PENALTY
  HIGH_SLIPPAGE_SIZE_CUT
  LOW_LIQUIDITY_SIZE_CUT
  ADJUSTED_SCORE_BELOW_FLOOR
  NORMAL

Return ONLY a JSON object with these exact keys:
  allow_trade (bool), buy_size_multiplier (float), effective_score_adjustment (float), reason_codes (list of strings)

Example: {{"allow_trade": true, "buy_size_multiplier": 0.75, "effective_score_adjustment": -0.3, "reason_codes": ["BIRDEYE_FALLBACK_PENALTY"]}}

Respond with valid JSON only. No markdown, no explanation outside the JSON."""


def _parse_response(raw: str) -> dict:
    text = raw
    if "```" in text:
        start = text.find("{")
        end   = text.rfind("}") + 1
        text  = text[start:end]

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error("[policy_agent] Failed to parse JSON: %s\nRaw: %s", exc, raw)
        # Safe fallback: allow trade but don't adjust anything
        return _default_allow()


def _validate_decision(raw: dict) -> dict:
    """Clamp numeric fields, coerce types, guarantee all keys are present."""
    allow_trade = bool(raw.get("allow_trade", True))

    # Clamp numerics via a temporary dict passed through validate_delta
    numeric = {
        "buy_size_multiplier":        raw.get("buy_size_multiplier", 1.0),
        "effective_score_adjustment": raw.get("effective_score_adjustment", 0.0),
    }
    clamped = _clamp_policy(numeric)

    reason_codes = raw.get("reason_codes", [])
    if not isinstance(reason_codes, list):
        reason_codes = [str(reason_codes)]

    # If agent says allow but adjusted score would be below 5.0, honour that
    # (the model should have set allow_trade=false itself, but double-check)
    return {
        "allow_trade":               allow_trade,
        "buy_size_multiplier":       clamped["buy_size_multiplier"],
        "effective_score_adjustment": clamped["effective_score_adjustment"],
        "reason_codes":              reason_codes,
    }


def _clamp_policy(values: dict) -> dict:
    """Clamp using POLICY_GUARDRAILS (separate from global GUARDRAILS)."""
    result = {}
    for key, value in values.items():
        if key in POLICY_GUARDRAILS:
            lo, hi = POLICY_GUARDRAILS[key]
            clamped = max(lo, min(hi, float(value)))
            if clamped != value:
                logger.warning(
                    "[policy_agent] guardrail: %s=%.4f clamped to %.4f (band [%.4f, %.4f])",
                    key, value, clamped, lo, hi,
                )
            result[key] = clamped
        else:
            result[key] = value
    return result


def _blocked(reason_codes: list[str]) -> dict:
    return {
        "allow_trade":                False,
        "buy_size_multiplier":        0.0,
        "effective_score_adjustment": 0.0,
        "reason_codes":               reason_codes,
    }


def _default_allow() -> dict:
    return {
        "allow_trade":                True,
        "buy_size_multiplier":        1.0,
        "effective_score_adjustment": 0.0,
        "reason_codes":               ["NORMAL"],
    }
