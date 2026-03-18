"""
trader/agents/ai_override.py — AI Override Agent.

Called when a signal is filtered out (ML_SKIP, CHART_SKIP, POLICY_BLK).
Analyzes all available chart data, historical outcomes, and skipped-signal
performance to decide whether to override the skip decision.

Usage:
    from trader.agents.ai_override import propose_ai_override

    decision = propose_ai_override(
        skip_reason="ML_SKIP",
        signal_context={
            "ml_score": 1.8,
            "ml_min_score": 2.5,
            "pump_ratio": 2.1,
            "pump_ratio_max": 3.5,
            "vol_trend": "RISING",
            "chart_reason": "pump=2.1x — vol RISING",
            "ml_source": "moralis/10s",
            "source_channel": "WizzyTrades",
            "pair_stats": {...},
            "candles_summary": {...},
        },
        strategy="quick_pop_chart_ml",
        db_path="/path/to/trades.db",
        training_strategy="quick_pop",
    )
    # decision = {
    #     "override": False,
    #     "reanalyze_after_seconds": 120.0,
    #     "reason": "Volume unclear, re-checking in 2 min",
    # }

Returns
-------
dict with keys:
    override (bool)                  — True = enter the trade now
    reanalyze_after_seconds (float)  — >0 = schedule a delayed re-check instead
                                       Mutually exclusive with override=True.
    reason (str)                     — brief explanation (≤120 chars)
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

_MODEL = "claude-haiku-4-5-20251001"

# Cap agent-requested reanalysis delays to 10 minutes
_MAX_REANALYZE_DELAY_S = 600.0

# ---------------------------------------------------------------------------
# Historical context cache — keyed by (strategy, training_strategy).
# Score buckets and skipped stats change slowly; TTL of 5 min is fine.
# ---------------------------------------------------------------------------
_context_cache: dict[str, tuple[float, dict]] = {}
_CACHE_TTL_S = 300.0


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def propose_ai_override(
    skip_reason: str,
    signal_context: dict,
    strategy: str,
    db_path: str | None = None,
    training_strategy: str | None = None,
) -> dict:
    """
    Evaluate a filtered-out signal and decide whether to override the skip.

    Parameters
    ----------
    skip_reason : str
        Why the signal was filtered: "ML_SKIP", "CHART_SKIP", or "POLICY_BLK".
    signal_context : dict
        All available signal data: ml_score, chart context, pair stats,
        candles summary, source channel, etc.
    strategy : str
        Strategy name (e.g. "quick_pop_chart_ml").
    db_path : str | None
        Path to the SQLite DB. When provided, historical context (score buckets,
        recent trade outcomes, previously-skipped signal outcomes) is added to
        the prompt so the agent can calibrate its override decisions.
    training_strategy : str | None
        The base strategy whose outcomes feed the ML scorer (e.g. "quick_pop").
        Used to pull historical win rates and skipped-signal outcomes.

    Returns
    -------
    dict with override (bool), reanalyze_after_seconds (float), reason (str).
    override=True and reanalyze_after_seconds>0 are mutually exclusive —
    the validator clears reanalyze_after_seconds when override is True.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set — required for AI override agent")

    hist = _query_historical_context(db_path, strategy, training_strategy) if db_path else None

    prompt = _build_prompt(skip_reason, signal_context, strategy, hist)

    client = anthropic.Anthropic(api_key=api_key)
    logger.info(
        "[ai_override] Calling %s | strategy=%s | skip=%s | ml_score=%s | hist=%s",
        _MODEL, strategy, skip_reason,
        signal_context.get("ml_score"),
        "yes" if hist else "no",
    )

    message = client.messages.create(
        model=_MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = message.content[0].text.strip()
    logger.debug("[ai_override] Raw response: %s", raw_text)

    return _parse_and_validate(raw_text)


def summarize_candles(candles: list[Any]) -> dict:
    """
    Produce a compact candles summary dict for the AI override prompt.
    Accepts either dicts with open/close/volume keys or objects with those attrs.
    Returns at most the last 8 candles.
    """
    if not candles:
        return {}

    recent = []
    for c in candles[-8:]:
        try:
            if isinstance(c, dict):
                o  = c.get("open", 0) or 0
                cl = c.get("close", 0) or 0
                v  = c.get("volume", 0) or 0
            else:
                o  = getattr(c, "open", 0) or 0
                cl = getattr(c, "close", 0) or 0
                v  = getattr(c, "volume", 0) or 0
            recent.append({"o": round(float(o), 8), "c": round(float(cl), 8), "v": round(float(v), 2)})
        except Exception:
            continue

    return {"recent": recent}


# ---------------------------------------------------------------------------
# Historical context query
# ---------------------------------------------------------------------------

def log_override_decision(
    strategy: str,
    symbol: str,
    skip_reason: str,
    decision: dict,
    signal_context: dict,
    shadow: bool = False,
) -> None:
    """
    Append one line to logs/ai_override_{strategy}.log.

    Every decision is logged — OVERRIDE, REJECT, REANALYZE, and their SHADOW_* variants.
    This is the human-readable companion to the DB table; the DB is used for queries,
    the log file for tailing in real time.
    """
    override   = decision.get("override", False)
    reanalyze  = decision.get("reanalyze_after_seconds", 0.0)
    reason     = decision.get("reason", "")

    prefix = "SHADOW_" if shadow else ""
    if override:
        action = f"{prefix}OVERRIDE"
    elif reanalyze > 0:
        action = f"{prefix}REANALYZE"
    else:
        action = f"{prefix}REJECT"

    ml_score   = signal_context.get("ml_score")
    pump_ratio = signal_context.get("pump_ratio")
    vol_trend  = signal_context.get("vol_trend", "?")

    score_str = f"{ml_score:.2f}" if ml_score is not None else "None"
    pump_str  = f"{pump_ratio:.2f}x" if pump_ratio is not None else "n/a"
    delay_str = f"  delay={reanalyze:.0f}s" if reanalyze > 0 else ""

    ts   = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = (
        f"{ts} | {action:<16} | {symbol:<12} | {skip_reason:<11} | "
        f"score={score_str:<5}  pump={pump_str:<6}  vol={vol_trend}{delay_str} | "
        f"{reason}\n"
    )

    log_dir  = Path(os.getenv("AGENT_LOG_DIR", "logs"))
    log_path = log_dir / f"ai_override_{strategy}.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as f:
            f.write(line)
    except OSError as exc:
        logger.warning("[ai_override] Could not write override log: %s", exc)


def _query_historical_context(
    db_path: str,
    strategy: str,
    training_strategy: str | None,
) -> dict | None:
    """
    Pull three compact data sets from the DB for context, with a 5-min cache.
      1. score_buckets   — win rate + avg PnL per ML score bucket (training strategy)
      2. recent_trades   — last 15 closed trades (training strategy)
      3. skipped_stats   — what the base strategy made on signals we previously skipped
    Returns None on any error so the agent still runs without historical context.
    """
    cache_key = f"{strategy}:{training_strategy or strategy}"
    now = time.monotonic()
    if cache_key in _context_cache:
        expires_at, cached = _context_cache[cache_key]
        if now < expires_at:
            return cached

    try:
        from trader.agents.base import query_score_buckets, query_recent_trades, query_skipped_stats

        base = training_strategy or strategy

        result = {
            "score_buckets": query_score_buckets(db_path, strategy=base),
            "recent_trades": query_recent_trades(db_path, strategy=base, limit=15, scored_only=True),
            "skipped_stats": query_skipped_stats(db_path, strategy=strategy, base_strategy=base),
        }
        _context_cache[cache_key] = (now + _CACHE_TTL_S, result)
        return result
    except Exception as exc:
        logger.warning("[ai_override] Failed to load historical context: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(skip_reason: str, ctx: dict, strategy: str, hist: dict | None) -> str:
    ml_score        = ctx.get("ml_score")
    ml_min_score    = ctx.get("ml_min_score", 5.0)
    pump_ratio      = ctx.get("pump_ratio")
    pump_ratio_max  = ctx.get("pump_ratio_max", 3.5)
    vol_trend       = ctx.get("vol_trend", "unknown")
    chart_reason    = ctx.get("chart_reason", "n/a")
    ml_source       = ctx.get("ml_source", "none")
    source_channel  = ctx.get("source_channel", "unknown")
    pair_stats      = ctx.get("pair_stats") or {}
    candles_summary = ctx.get("candles_summary") or {}

    # ML score line
    if ml_score is not None:
        ml_line = f"{ml_score:.2f} (threshold was {ml_min_score})"
    else:
        ml_line = f"None (threshold={ml_min_score})"

    # Pump ratio line
    pump_line = f"{pump_ratio:.2f}x (max allowed={pump_ratio_max}x)" if pump_ratio is not None else "n/a"

    # Pair stats block
    if pair_stats:
        buys  = pair_stats.get("buys_5m", "n/a")
        sells = pair_stats.get("sells_5m", "n/a")
        pc5m  = pair_stats.get("price_change_5m_pct", "n/a")
        liq   = pair_stats.get("liquidity_change_1h_pct", "n/a")
        bvol  = pair_stats.get("buy_volume_1h", "n/a")
        tvol  = pair_stats.get("total_volume_1h", "n/a")
        bvol_str = f"${bvol:,.0f}" if isinstance(bvol, (int, float)) else str(bvol)
        tvol_str = f"${tvol:,.0f}" if isinstance(tvol, (int, float)) else str(tvol)
        pc5m_str = f"{pc5m:.1f}%" if isinstance(pc5m, (int, float)) else str(pc5m)
        liq_str  = f"{liq:.1f}%"  if isinstance(liq,  (int, float)) else str(liq)
        ps_block = (
            f"  buys_5m / sells_5m:        {buys} / {sells}\n"
            f"  price_change_5m:           {pc5m_str}\n"
            f"  liquidity_change_1h:       {liq_str}\n"
            f"  buy_vol_1h / total_vol_1h: {bvol_str} / {tvol_str}"
        )
    else:
        ps_block = "  (pair stats unavailable)"

    # Candles block
    recent = candles_summary.get("recent", [])
    if recent:
        lines = [f"  Last {len(recent)} candle(s) [open, close, vol]:"]
        for c in recent:
            lines.append(f"    o={c['o']}  c={c['c']}  v={c['v']}")
        cs_block = "\n".join(lines)
    else:
        cs_block = "  (candle data unavailable)"

    # Historical context block
    hist_block = _format_historical_context(hist, ml_score)

    return f"""You are an AI override agent for a Solana trading bot (strategy: {strategy}).
A buy signal was FILTERED OUT. Decide if the filter was wrong and this trade is worth entering.

SKIP REASON: {skip_reason}

SIGNAL DATA:
  source_channel:   {source_channel}
  ml_score:         {ml_line}
  ml_candle_source: {ml_source}
  pump_ratio:       {pump_line}
  vol_trend:        {vol_trend}
  chart_reason:     {chart_reason}

PAIR STATS (Moralis, 5m/1h window):
{ps_block}

RECENT CANDLES:
{cs_block}
{hist_block}
STRATEGY PROFILE:
  - Fast scalp: targets 1.5x–2x within 45 minutes
  - Ideal entry: early momentum, high buy activity, rising volume
  - Stop loss: 20% below entry; hard exit if no 1.49x gain after 45 min

OVERRIDE DECISION GUIDE:
  ML_SKIP   → override if chart+pair stats show strong momentum AND score was only slightly below threshold
  CHART_SKIP → override if pump was modest AND buy pressure is still strong (not dying)
  POLICY_BLK → override only if blocked for data quality reasons (not a hard liquidity block)

WHEN TO REANALYZE INSTEAD (set reanalyze_after_seconds > 0, leave override=false):
  - Candle data incomplete (<5 bars) — check again in 90–120s
  - Momentum ambiguous but not dead — check again in 60–180s

WHEN TO REJECT ENTIRELY (override=false, reanalyze_after_seconds=0):
  - Volume clearly dying
  - Pump already >4x with no fresh buy pressure
  - ML score very low (<1.0) with no supporting chart evidence

IMPORTANT: override and reanalyze are mutually exclusive. Set EITHER override=true OR
reanalyze_after_seconds>0, never both. If overriding, set reanalyze_after_seconds=0.

Return ONLY valid JSON (no markdown, no text outside JSON):
  override (bool)
  reanalyze_after_seconds (float — 0 if override=true, max 600)
  reason (str, ≤80 chars)

Example A (override): {{"override": true, "reanalyze_after_seconds": 0, "reason": "Strong buys, score only 0.3 below floor"}}
Example B (reanalyze): {{"override": false, "reanalyze_after_seconds": 120.0, "reason": "Volume unclear, rechecking in 2 min"}}
Example C (reject): {{"override": false, "reanalyze_after_seconds": 0, "reason": "Volume dying, no edge"}}"""


def _format_historical_context(hist: dict | None, current_ml_score: float | None) -> str:
    """Format historical context into a compact prompt block."""
    if not hist:
        return ""

    lines = ["\nHISTORICAL CONTEXT (base strategy outcomes):"]

    # Score bucket performance
    buckets = hist.get("score_buckets", [])
    if buckets:
        lines.append("  ML score bucket performance (win_rate | avg_pnl | trades):")
        for b in buckets:
            marker = " ← current signal" if (
                current_ml_score is not None
                and b["bucket"].split("-")[0] <= f"{current_ml_score:.1f}"
                and f"{current_ml_score:.1f}" <= b["bucket"].split("-")[1]
            ) else ""
            lines.append(
                f"    score {b['bucket']}: {b['win_rate']*100:.0f}% wins | "
                f"avg {b['avg_pnl_pct']:+.1f}% pnl | "
                f"avg {b['avg_max_gain_pct']:+.1f}% peak | "
                f"n={b['count']}{marker}"
            )

    # Previously-skipped signal outcomes
    skipped = hist.get("skipped_stats", {})
    if skipped and skipped.get("base_entered", 0) > 0:
        lines.append(
            f"  Previously skipped signals ({skipped['total_skipped']} total skipped, "
            f"{skipped['base_entered']} tracked by base strategy):"
        )
        lines.append(
            f"    profitable: {skipped['profitable_pct']:.0f}% | "
            f"avg pnl: {skipped['avg_phantom_pnl']:+.1f}% | "
            f"avg peak: {skipped.get('avg_max_gain', 0) or 0:+.1f}%"
        )
        samples = skipped.get("sample_outcomes", [])
        if samples:
            lines.append("    Recent skipped samples [pnl% | peak% | exit | ml_score]:")
            for s in samples[:5]:
                lines.append(
                    f"      {s['symbol']:<10} {s['pnl_pct']:+6.1f}% | "
                    f"{(s['max_gain_pct'] or 0):+6.1f}% peak | "
                    f"{s['sell_reason'] or '?':<12} | score={s['ml_score']}"
                )

    # Recent trade outcomes (last 10 for brevity)
    recent = hist.get("recent_trades", [])
    if recent:
        lines.append(f"  Last {min(len(recent), 10)} closed trades [pnl% | peak% | exit | ml_score]:")
        for t in recent[:10]:
            lines.append(
                f"    {t['symbol']:<10} {t['outcome_pnl_pct']:+6.1f}% | "
                f"{(t['outcome_max_gain_pct'] or 0):+6.1f}% peak | "
                f"{t['outcome_sell_reason'] or '?':<12} | score={t['ml_score']}"
            )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Parse and validate
# ---------------------------------------------------------------------------

def _parse_and_validate(raw: str) -> dict:
    text = raw
    if "```" in text:
        start = text.find("{")
        end   = text.rfind("}") + 1
        text  = text[start:end]

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error("[ai_override] Failed to parse JSON: %s\nRaw: %s", exc, raw)
        return _default_reject("JSON parse error — defaulting to no override")

    override  = bool(data.get("override", False))
    reanalyze = float(data.get("reanalyze_after_seconds", 0.0))
    reanalyze = max(0.0, min(_MAX_REANALYZE_DELAY_S, reanalyze))
    reason    = str(data.get("reason", ""))[:120]

    # Enforce mutual exclusivity — override takes priority
    if override:
        reanalyze = 0.0

    return {
        "override":                override,
        "reanalyze_after_seconds": reanalyze,
        "reason":                  reason,
    }


def _default_reject(reason: str) -> dict:
    return {
        "override":                False,
        "reanalyze_after_seconds": 0.0,
        "reason":                  reason,
    }
