"""
trader/agents/strategy_tuner.py — Strategy parameter tuner (Agent E).

Analyzes closed trade performance for the controlled strategies and
proposes + applies parameter adjustments.

Permission tiers
----------------
FULL_CONTROL (trend_rider, trend_rider_chart_reanalyze, infinite_moonbag,
              infinite_moonbag_chart):
    TP levels, stop/trail/timeout, chart filter settings, reanalysis settings,
    and all ML params.

ML_ONLY (quick_pop_chart_ml):
    ML score thresholds, size multipliers, KNN hyperparams, and use_ml_filter
    toggle only. No TP/stop/chart filter/reanalysis changes.

NOT controlled (quick_pop is never touched):
    quick_pop

Live config is persisted to strategy_config.json in the project root.
registry.py loads this file at startup and merges over its hardcoded defaults.

Usage::

    from trader.agents.strategy_tuner import run
    delta = run("trend_rider", db_path="trader.db")
    # delta = {"stop_loss_pct": 0.27, "reason": "..."}
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import anthropic
from datetime import date

from trader.agents.base import (
    GUARDRAILS,
    query_exit_stats,
    query_recent_trades,
    query_score_buckets,
    summarise_guardrails,
    validate_delta,
)

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-6"
_MIN_TRADES = 10   # minimum closed trades required before tuning

# AI balance / deadline context
_AI_DEADLINE        = date(2026, 4, 10)
_AI_START_BALANCE   = 1000.0
_AI_PROFIT_TARGET   = 300.0

CONFIG_PATH = Path(__file__).parent.parent.parent / "strategy_config.json"

# ---------------------------------------------------------------------------
# Which strategies this agent controls
# ---------------------------------------------------------------------------

CONTROLLED_STRATEGIES = frozenset([
    "trend_rider",
    "trend_rider_chart_reanalyze",
    "infinite_moonbag",
    "infinite_moonbag_chart",
    "quick_pop_chart_ml",
])

# Permission tiers — define what each strategy is allowed to change.
#
#   FULL_CONTROL  — TP levels, stop/trail/timeout, chart filter settings,
#                   reanalysis settings, and all ML params.
#
#   ML_ONLY       — ML score thresholds, size multipliers, KNN hyperparams,
#                   and use_ml_filter toggle only. No TP/stop/chart changes.
#
FULL_CONTROL_STRATEGIES = frozenset([
    "trend_rider",
    "trend_rider_chart_reanalyze",
    "infinite_moonbag",
    "infinite_moonbag_chart",
])

ML_ONLY_STRATEGIES = frozenset([
    "quick_pop_chart_ml",
])

_CHART_VARIANTS = frozenset([
    "trend_rider_chart_reanalyze",
    "infinite_moonbag_chart",
    "quick_pop_chart_ml",
])

_MOONBAG_STRATEGIES = frozenset([
    "infinite_moonbag",
    "infinite_moonbag_chart",
])

# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

# Additional numeric guardrails not in base.GUARDRAILS
_EXTRA_GUARDRAILS: dict[str, tuple[float, float]] = {
    "max_hold_minutes":      (60.0,   480.0),
    "timeout_min_gain_pct":  (0.05,   0.50),
    "pump_ratio_max":        (2.0,    8.0),   # chart filter sensitivity
    "reanalyze_pump_delay":  (60.0,  1800.0), # seconds to wait after pump skip
    "reanalyze_vol_delay":   (60.0,  1200.0), # seconds to wait after vol-dying skip
    "reanalyze_both_delay":  (60.0,  1800.0), # seconds to wait after both triggers
    "ml_k":                  (3.0,   15.0),   # KNN neighbours (stored as float, cast to int)
    "ml_halflife_days":      (3.0,   30.0),   # recency decay half-life
    "ml_score_low_pct":     (-50.0, -10.0),   # PnL% → score 0
    "ml_score_high_pct":    (30.0,  150.0),   # PnL% → score 10
}

# TP guardrails per strategy: list of (min_multiple, max_multiple, min_fraction, max_fraction)
# One entry per TP level in order.
_TP_GUARDRAILS: dict[str, list[tuple[float, float, float, float]]] = {
    "trend_rider": [
        (1.3, 3.0, 0.30, 0.75),   # tp1
    ],
    "trend_rider_chart_reanalyze": [
        (1.3, 3.0, 0.30, 0.75),   # tp1
    ],
    "infinite_moonbag": [
        (1.3, 2.5, 0.10, 0.40),   # tp1
        (1.8, 3.5, 0.08, 0.30),   # tp2
        (2.5, 6.0, 0.08, 0.30),   # tp3
        (3.5, 9.0, 0.05, 0.25),   # tp4
    ],
    "infinite_moonbag_chart": [
        (1.3, 2.5, 0.10, 0.40),   # tp1
        (1.8, 3.5, 0.08, 0.30),   # tp2
        (2.5, 6.0, 0.08, 0.30),   # tp3
        (3.5, 9.0, 0.05, 0.25),   # tp4
    ],
}

# Keys the agent is NEVER allowed to write
_FORBIDDEN_KEYS = frozenset([
    "use_chart_filter", "use_reanalyze", "name", "buy_size_usd",
    "starting_cash_usd", "ml_training_strategy", "save_chart_data",
    "use_policy_agent", "timeout_min_gain_pct",
])

# Keys allowed for ML_ONLY strategies (quick_pop_chart_ml).
# use_ml_filter is intentionally excluded — it is permanently ON for quick_pop_chart_ml
# and the agent cannot toggle it.
_ML_ONLY_ALLOWED_KEYS = frozenset([
    "ml_min_score", "ml_high_score_threshold", "ml_max_score_threshold",
    "ml_size_multiplier", "ml_max_size_multiplier",
    "ml_k", "ml_halflife_days", "ml_score_low_pct", "ml_score_high_pct",
    "live_trading",   # can enable/disable live trading for this strategy
])

# Keys the agent IS allowed to write (besides "reason" and "tp_levels")
_ALLOWED_SCALAR_KEYS = frozenset([
    "stop_loss_pct", "trailing_stop_pct", "timeout_minutes", "max_hold_minutes",
    "timeout_min_gain_pct",
    "ml_min_score", "ml_high_score_threshold", "ml_max_score_threshold",
    "ml_size_multiplier", "ml_max_size_multiplier",
    "pump_ratio_max",
    "reanalyze_pump_delay", "reanalyze_vol_delay", "reanalyze_both_delay",
    "ml_k", "ml_halflife_days", "ml_score_low_pct", "ml_score_high_pct",
    "use_ml_filter",      # bool — validated separately
    "use_chart_filter",   # bool — validated separately
    "use_reanalyze",      # bool — validated separately
    "live_trading",       # bool — validated separately
])


# ---------------------------------------------------------------------------
# Config file I/O
# ---------------------------------------------------------------------------

def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load strategy_config.json. Raises FileNotFoundError if absent."""
    with path.open() as f:
        return json.load(f)


def save_config(config: dict, path: Path = CONFIG_PATH) -> None:
    """
    Atomically write strategy_config.json.
    Uses write-to-temp-then-rename so a crash mid-write cannot corrupt the file.
    """
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(config, f, indent=2)
            f.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(
    strategy: str,
    db_path: str = "trader.db",
    dry_run: bool = False,
) -> dict:
    """
    Run the strategy tuner for one controlled strategy.

    Steps:
      1. Validate strategy is in CONTROLLED_STRATEGIES.
      2. Load current params from strategy_config.json.
      3. Query DB for performance stats.
      4. Build prompt and call Claude Sonnet 4.6.
      5. Parse and validate the proposed delta against guardrails.
      6. Apply delta to strategy_config.json (unless dry_run).
      7. Return the applied delta (empty dict if no changes).

    Returns {} if fewer than _MIN_TRADES closed trades exist or if no
    changes are proposed.

    Raises ValueError for strategies not in CONTROLLED_STRATEGIES.
    """
    if strategy not in CONTROLLED_STRATEGIES:
        raise ValueError(
            f"Strategy '{strategy}' is not controlled by the strategy tuner. "
            f"Controlled: {sorted(CONTROLLED_STRATEGIES)}"
        )

    # Load live config (required — file must exist)
    try:
        config = load_config()
    except FileNotFoundError:
        logger.error("[strategy_tuner] strategy_config.json not found at %s", CONFIG_PATH)
        return {}

    current_params = config.get(strategy, {})

    # Query performance data from DB
    exit_stats = query_exit_stats(db_path, strategy)
    total_trades = sum(s["count"] for s in exit_stats)

    if total_trades < _MIN_TRADES:
        logger.info(
            "[strategy_tuner] Only %d closed trades for %s — need %d to run",
            total_trades, strategy, _MIN_TRADES,
        )
        return {}

    recent_trades = query_recent_trades(db_path, strategy, limit=20)

    # Score buckets only for chart variants (they have ml_score data)
    score_buckets: list[dict] = []
    if strategy in _CHART_VARIANTS:
        score_buckets = query_score_buckets(db_path, strategy)

    # Fetch AI balance from DB for urgency context
    ai_balance = _AI_START_BALANCE
    try:
        import sqlite3 as _sqlite3
        _conn = _sqlite3.connect(db_path)
        row = _conn.execute(
            "SELECT COALESCE(SUM(outcome_pnl_usd), 0.0) FROM strategy_outcomes "
            "WHERE is_live=1 AND closed=1 AND entered=1 AND outcome_pnl_usd IS NOT NULL"
        ).fetchone()
        _conn.close()
        ai_balance = _AI_START_BALANCE + (row[0] if row else 0.0)
    except Exception:
        pass  # use default if DB query fails

    # Current live_trading state for this strategy
    live_trading_on = bool(current_params.get("live_trading", False))

    # Build prompt and call Claude (tier-appropriate prompt)
    if strategy in ML_ONLY_STRATEGIES:
        prompt = _build_prompt_ml_only(strategy, current_params, exit_stats, recent_trades, score_buckets, total_trades, ai_balance, live_trading_on)
    elif strategy in _CHART_VARIANTS:
        prompt = _build_prompt_chart(strategy, current_params, exit_stats, recent_trades, score_buckets, total_trades, ai_balance, live_trading_on)
    else:
        prompt = _build_prompt_base(strategy, current_params, exit_stats, recent_trades, total_trades, ai_balance, live_trading_on)

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set — required for strategy tuner")

    client = anthropic.Anthropic(api_key=api_key)
    logger.info(
        "[strategy_tuner] Calling %s for strategy=%s (%d trades)",
        _MODEL, strategy, total_trades,
    )

    message = client.messages.create(
        model=_MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = message.content[0].text.strip()
    logger.debug("[strategy_tuner] Raw response: %s", raw_text)

    # Parse, validate, apply
    raw_delta = _parse_response(raw_text)
    if not raw_delta:
        return {}

    validated = _validate_strategy_delta(strategy, raw_delta)
    if not validated:
        return {}

    if not dry_run:
        _apply_delta(strategy, validated, config)
        save_config(config)
        logger.info("[strategy_tuner] Applied delta for %s: %s", strategy, validated)
    else:
        logger.info("[strategy_tuner] Dry run — proposed delta for %s: %s", strategy, validated)

    return validated


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _urgency_section(ai_balance: float) -> str:
    """
    Build the urgency context block included in every agent prompt.
    Reminds the agent of the deadline, profit target, and current AI balance.
    """
    today         = date.today()
    days_left     = (_AI_DEADLINE - today).days
    profit_so_far = ai_balance - _AI_START_BALANCE
    profit_needed = _AI_PROFIT_TARGET - profit_so_far

    if days_left <= 0:
        urgency = "DEADLINE PASSED."
    elif days_left <= 5:
        urgency = f"CRITICAL — only {days_left} day(s) left."
    elif days_left <= 10:
        urgency = f"URGENT — {days_left} days left."
    else:
        urgency = f"{days_left} days remaining."

    return f"""
SERVICE SURVIVAL CONTEXT (read carefully — this affects your decisions):
  AI Balance:       ${ai_balance:.2f}  (started at ${_AI_START_BALANCE:.2f})
  Profit so far:    ${profit_so_far:+.2f}
  Profit target:    ${_AI_PROFIT_TARGET:.2f} by {_AI_DEADLINE} to cover API + server costs
  Still needed:     ${max(0.0, profit_needed):.2f}
  Deadline:         {_AI_DEADLINE}  — {urgency}

The AI balance is only affected by strategies with live_trading=true.
Strategies still on paper (live_trading=false) do NOT count toward the target.

If the service cannot reach ${_AI_PROFIT_TARGET:.2f} profit by {_AI_DEADLINE}, it will be shut down.
Factor this urgency into your decisions:
  - Enable live_trading for a strategy only when you have enough evidence it is profitable.
  - Once live_trading is on, the AI balance rises or falls with every real trade.
  - Prioritise strategies with the strongest track record and tightest risk controls.
  - Do NOT enable live_trading on a strategy still showing consistent losses.
"""


def _tp_guardrail_table(strategy: str) -> str:
    levels = _TP_GUARDRAILS[strategy]
    lines = [f"  tp_levels must be a list of {len(levels)} [multiple, fraction] pairs:"]
    for i, (min_m, max_m, min_f, max_f) in enumerate(levels):
        lines.append(
            f"    tp{i + 1}: multiple in [{min_m}, {max_m}], "
            f"fraction in [{min_f}, {max_f}]"
        )
    lines.append("  Multiples must be strictly ascending. Total fractions must not exceed 1.0.")
    return "\n".join(lines)


def _scalar_guardrail_table(strategy: str) -> str:
    # Combine base + extra guardrails, filtered to what's relevant for this strategy
    all_guardrails = {**GUARDRAILS, **_EXTRA_GUARDRAILS}

    relevant = ["stop_loss_pct", "trailing_stop_pct"]
    if strategy not in _MOONBAG_STRATEGIES:
        relevant += ["timeout_minutes", "max_hold_minutes"]
    if strategy in _CHART_VARIANTS:
        relevant += [
            "ml_min_score", "ml_high_score_threshold", "ml_max_score_threshold",
            "ml_size_multiplier", "ml_max_size_multiplier",
        ]

    lines = ["Parameter                    | Min    | Max", "-" * 48]
    for key in relevant:
        if key in all_guardrails:
            lo, hi = all_guardrails[key]
            lines.append(f"{key:<28} | {lo:<6} | {hi}")
    return "\n".join(lines)


def _format_current_params(strategy: str, params: dict) -> str:
    lines = []
    lines.append(f"  stop_loss_pct:         {params.get('stop_loss_pct', 0.30)}")
    lines.append(f"  trailing_stop_pct:     {params.get('trailing_stop_pct', 0.30)}")
    if strategy not in _MOONBAG_STRATEGIES:
        lines.append(f"  timeout_minutes:       {params.get('timeout_minutes', 90.0)}")
        lines.append(f"  timeout_min_gain_pct:  {params.get('timeout_min_gain_pct', 0.15)}")
        lines.append(f"  max_hold_minutes:      {params.get('max_hold_minutes', 240.0)}")
    tp = params.get("tp_levels", [])
    for i, (m, f) in enumerate(tp):
        lines.append(f"  tp{i + 1}: {m}× → sell {int(f * 100)}% of original")
    if strategy in _CHART_VARIANTS:
        lines.append(f"  use_chart_filter:         {params.get('use_chart_filter', True)}")
        lines.append(f"  pump_ratio_max:           {params.get('pump_ratio_max', 3.5)}")
        lines.append(f"  use_reanalyze:            {params.get('use_reanalyze', True)}")
        lines.append(f"  reanalyze_pump_delay:     {params.get('reanalyze_pump_delay', 480.0)}s")
        lines.append(f"  reanalyze_vol_delay:      {params.get('reanalyze_vol_delay', 240.0)}s")
        lines.append(f"  reanalyze_both_delay:     {params.get('reanalyze_both_delay', 600.0)}s")
        lines.append(f"  use_ml_filter:            {params.get('use_ml_filter', False)}")
        lines.append(f"  ml_min_score:             {params.get('ml_min_score', 5.0)}")
        lines.append(f"  ml_high_score_threshold:  {params.get('ml_high_score_threshold', 8.0)}")
        lines.append(f"  ml_max_score_threshold:   {params.get('ml_max_score_threshold', 9.5)}")
        lines.append(f"  ml_size_multiplier:       {params.get('ml_size_multiplier', 2.0)}")
        lines.append(f"  ml_max_size_multiplier:   {params.get('ml_max_size_multiplier', 3.0)}")
        lines.append(f"  ml_k:                     {params.get('ml_k', 5)}")
        lines.append(f"  ml_halflife_days:         {params.get('ml_halflife_days', 14.0)}")
        lines.append(f"  ml_score_low_pct:         {params.get('ml_score_low_pct', -35.0)}")
        lines.append(f"  ml_score_high_pct:        {params.get('ml_score_high_pct', 85.0)}")
    return "\n".join(lines)


def _build_prompt_base(
    strategy: str,
    current_params: dict,
    exit_stats: list[dict],
    recent_trades: list[dict],
    total_trades: int,
    ai_balance: float = _AI_START_BALANCE,
    live_trading_on: bool = False,
) -> str:
    is_moonbag = strategy in _MOONBAG_STRATEGIES
    tp_count = 4 if is_moonbag else 1

    strategy_desc = (
        "infinite_moonbag: holds indefinitely (no timeout/max_hold), uses a progressive "
        "stop ladder. TP levels de-risk as price climbs. Expects multi-bagger moves."
        if is_moonbag else
        "trend_rider: momentum hold strategy. Sells 50% at TP1, trails after. "
        "Exits via timeout (90 min if stagnant) or max hold (4 hours)."
    )

    live_status = "LIVE (counts toward AI balance)" if live_trading_on else "PAPER ONLY"

    return f"""You are a parameter tuner for a Solana trading bot.
Strategy: {strategy}  |  Mode: {live_status}
Description: {strategy_desc}
{_urgency_section(ai_balance)}
TOTAL CLOSED TRADES ANALYZED: {total_trades}

CURRENT CONFIGURATION:
{_format_current_params(strategy, current_params)}

EXIT REASON BREAKDOWN:
{json.dumps(exit_stats, indent=2)}

Fields:
- avg_pnl_pct: average realized PnL% (negative = losses)
- avg_hold_secs: average seconds held before this exit
- avg_max_gain_pct: average highest unrealized gain% seen during the trade

LAST 10 TRADES (most recent first):
{json.dumps(recent_trades[-10:], indent=2)}

ALLOWED SCALAR ADJUSTMENTS:
{_scalar_guardrail_table(strategy)}

TP LEVEL CONSTRAINTS:
{_tp_guardrail_table(strategy)}

TASK:
Analyze exit data and propose up to 3 parameter changes. Consider:

1. STOP_LOSS exits: if avg_max_gain_pct is high (>15%), token often ran after stop
   → stop is too tight, increase stop_loss_pct.
   If avg_max_gain_pct is low (<5%), losses are genuine → stop may be too wide, decrease it.
2. TRAILING_STOP exits: large gap between avg_max_gain_pct and avg_pnl_pct
   → trail is too loose, lower trailing_stop_pct.
3. TIMEOUT exits with positive avg_pnl_pct → timeout cutting winners, raise timeout_minutes.
4. TP levels: if tokens regularly hit far above TP1 multiple, consider raising TP1 multiple
   to capture more upside before selling. If TP1 is rarely hit, lower it.
{'5. No timeout or max_hold for this strategy — focus on stop/trail/TP only.' if is_moonbag else ''}

You may also set:
- live_trading: true to enable this strategy's trades against the AI balance,
  or false to keep it paper-only. Only enable when the strategy shows consistent
  profitability and you have enough evidence to trust it with real capital.

Return ONLY a JSON object with the keys you want to change and a "reason" string.
Do not include keys you are not changing.
tp_levels must be the full list of {tp_count} [multiple, fraction] pair(s) if changing any TP.
live_trading must be a boolean (true or false).

Example: {{"stop_loss_pct": 0.27, "live_trading": false, "reason": "STOP_LOSS exits show avg_max_gain_pct 4.1% — losses are genuine, tightening stop. Not enabling live trading yet."}}

Respond with valid JSON only. No markdown, no explanation outside the JSON."""


def _build_prompt_chart(
    strategy: str,
    current_params: dict,
    exit_stats: list[dict],
    recent_trades: list[dict],
    score_buckets: list[dict],
    total_trades: int,
    ai_balance: float = _AI_START_BALANCE,
    live_trading_on: bool = False,
) -> str:
    base = _build_prompt_base(strategy, current_params, exit_stats, recent_trades, total_trades, ai_balance, live_trading_on)
    tp_count = 4 if strategy in _MOONBAG_STRATEGIES else 1

    ml_section = f"""
SCORE-BUCKET PERFORMANCE (ml_score buckets, chart-filtered trades only):
{json.dumps(score_buckets, indent=2) if score_buckets else "  No score bucket data yet (use_ml_filter may be disabled)."}

CHART FILTER TUNING:
- use_chart_filter: toggle the 1-minute chart filter on (true) or off (false).
  On = only enter if pump_ratio < pump_ratio_max AND volume not dying.
  Off = enter all signals regardless of chart shape (ML filter still applies if enabled).
- pump_ratio_max: how pumped a token can be before skipping. Lower = stricter.
  Current value skips if price is already >{current_params.get('pump_ratio_max', 3.5)}x above recent low.
  Raise this for moonbag/trend strategies where "pumped" tokens can still be early in a move.

REANALYSIS TUNING:
- use_reanalyze: when true, a skipped signal is re-checked after a delay.
  Useful when tokens briefly look pumped or have dying volume but may recover.
  Disable if you see many late re-entries underperforming fresh signals.
- reanalyze_pump_delay: seconds to wait before re-checking a pump-skipped signal.
  Shorter = re-enter sooner (more risk, better price if it retraces). Longer = wait for retrace.
- reanalyze_vol_delay: seconds to wait before re-checking a dying-volume skip.
  Volume can flip quickly so shorter delays (60-300s) tend to work better here.
- reanalyze_both_delay: seconds to wait when both pump AND dying volume triggered.
  Worst-case scenario — longer delays give the token more time to recover.

ML PARAMETER TUNING (only if score bucket data is present):
- Buckets with consistently negative avg_pnl_pct → raise ml_min_score to exclude them.
- High-win-rate bucket floor → set ml_high_score_threshold there.
- use_ml_filter: set true to enable ML gating. Only enable if >= 20 score-bucketed trades.
- ml_k: number of KNN neighbours. Lower = more reactive to recent patterns.
- ml_halflife_days: recency decay. Lower = recent trades outweigh older data more.
- ml_score_low_pct / ml_score_high_pct: PnL% range mapped to scores 0 and 10.
  Recalibrate if your actual trade PnL range differs from the current mapping.

Return ONLY a JSON object. Include any of these keys you want to change plus a "reason" string.
tp_levels must be the full list of {tp_count} [multiple, fraction] pair(s) if changing any TP.
use_ml_filter and use_chart_filter must be booleans (true or false).
ml_k will be rounded to the nearest integer.

Example:
{{"use_chart_filter": true, "pump_ratio_max": 5.0, "use_reanalyze": true, "reanalyze_pump_delay": 300.0, "use_ml_filter": true, "ml_min_score": 5.5, "reason": "..."}}

Respond with valid JSON only. No markdown, no explanation outside the JSON."""

    # Replace the base prompt's closing instruction with the extended ML version
    cutoff = base.rfind("Return ONLY a JSON object")
    return base[:cutoff] + ml_section


def _build_prompt_ml_only(
    strategy: str,
    current_params: dict,
    exit_stats: list[dict],
    recent_trades: list[dict],
    score_buckets: list[dict],
    total_trades: int,
    ai_balance: float = _AI_START_BALANCE,
    live_trading_on: bool = False,
) -> str:
    """
    Prompt for ML_ONLY strategies (quick_pop_chart_ml).
    The agent can only adjust ML filter parameters — TP, stop, and chart filter
    settings are fixed and cannot be changed.
    """
    live_status = "LIVE (counts toward AI balance)" if live_trading_on else "PAPER ONLY"

    return f"""You are a parameter tuner for a Solana trading bot.
Strategy: {strategy}  |  Mode: {live_status}
Description: quick_pop_chart_ml is a fast scalp strategy that buys on momentum signals,
sells 60% at 1.5× and 40% at 2.0×, trails at 22% below high after first TP.
Exits after 45 minutes if TP1 not hit. Chart filter is ALWAYS enabled (not adjustable here).
Your role is to tune the ML confidence filter and decide on live_trading status.
{_urgency_section(ai_balance)}
TOTAL CLOSED TRADES ANALYZED: {total_trades}

FIXED PARAMETERS (read-only — these cannot be changed):
  TP1: 1.5× → sell 60% | TP2: 2.0× → sell 40%
  stop_loss_pct:    {current_params.get('stop_loss_pct', 0.20)}  (fixed)
  trailing_stop:    {current_params.get('trailing_stop_pct', 0.22)}  (fixed)
  timeout_minutes:  45.0  (fixed)
  chart_filter:     always enabled  (fixed)
  use_ml_filter:    always enabled  (fixed — cannot be toggled)

CURRENT ML CONFIGURATION:
  ml_min_score:             {current_params.get('ml_min_score', 5.0)}
  ml_high_score_threshold:  {current_params.get('ml_high_score_threshold', 8.0)}
  ml_max_score_threshold:   {current_params.get('ml_max_score_threshold', 9.5)}
  ml_size_multiplier:       {current_params.get('ml_size_multiplier', 2.0)}  (2× size for high-score signals)
  ml_max_size_multiplier:   {current_params.get('ml_max_size_multiplier', 3.0)}  (3× size for max-score signals)
  ml_k:                     {current_params.get('ml_k', 5)}  (KNN neighbours)
  ml_halflife_days:         {current_params.get('ml_halflife_days', 14.0)}  (recency decay)
  ml_score_low_pct:         {current_params.get('ml_score_low_pct', -35.0)}  (PnL% → score 0)
  ml_score_high_pct:        {current_params.get('ml_score_high_pct', 85.0)}  (PnL% → score 10)

EXIT REASON BREAKDOWN:
{json.dumps(exit_stats, indent=2)}

LAST 10 TRADES (most recent first):
{json.dumps(recent_trades[-10:], indent=2)}

SCORE-BUCKET PERFORMANCE (ml_score buckets, chart-filtered trades only):
{json.dumps(score_buckets, indent=2) if score_buckets else "  No score bucket data yet (use_ml_filter may be disabled or not enough trades)."}

ML GUARDRAILS (you must stay within these bounds):
  ml_min_score:            [0.0, 9.0]
  ml_high_score_threshold: [5.0, 9.5]
  ml_max_score_threshold:  [7.0, 10.0]
  ml_size_multiplier:      [1.0, 5.0]
  ml_max_size_multiplier:  [1.0, 8.0]
  ml_k:                    [3, 15]  (integer)
  ml_halflife_days:        [3.0, 30.0]
  ml_score_low_pct:        [-50.0, -10.0]
  ml_score_high_pct:       [30.0, 150.0]

TASK:
You may ONLY adjust ML filter parameters. Analyze score bucket data and recent trade outcomes:

1. use_ml_filter: enable (true) only if >= 20 trades in score bucket data show meaningful
   score variance (some buckets win, some lose). Disable if scores are noisy.
2. ml_min_score: raise if low-score buckets (0-4) show negative avg_pnl_pct consistently.
   This acts as a gate — signals below this score are skipped entirely.
3. ml_high_score_threshold / ml_max_score_threshold: set floor/ceiling for the buy-size
   multiplier tier. If score 7+ trades clearly outperform, set high_score_threshold=7.
4. ml_size_multiplier / ml_max_size_multiplier: how much to scale position size for
   high-confidence signals. Only raise if high-score trades show strong positive PnL.
5. ml_k: lower (3-5) for more reactive scoring when market conditions change fast.
   Raise (8-15) for more stable scoring with noisy recent data.
6. ml_halflife_days: lower for faster adaptation to recent patterns. Raise if recent
   trades are too few to be representative.
7. ml_score_low_pct / ml_score_high_pct: recalibrate if your actual PnL range consistently
   falls outside the current mapping (e.g., best trades rarely hit 85%).

IMPORTANT: You cannot change tp_levels, stop_loss_pct, trailing_stop_pct, timeout_minutes,
use_ml_filter, use_chart_filter, pump_ratio_max, or any reanalysis parameters.

You MAY set live_trading: true/false. Only enable when you have enough evidence this
strategy is profitable. Remember: the AI balance only grows when live_trading=true.

Return ONLY a JSON object with the keys you want to change and a "reason" string.
ml_k will be rounded to the nearest integer. live_trading must be a boolean.
Do not include keys you are not changing.

Example:
{{"ml_min_score": 5.5, "ml_k": 7, "live_trading": false, "reason": "Score bucket 0-4 shows -12% avg PnL. Not enabling live trading yet — need more data."}}

Respond with valid JSON only. No markdown, no explanation outside the JSON."""


# ---------------------------------------------------------------------------
# Response parsing and validation
# ---------------------------------------------------------------------------

def _parse_response(raw: str) -> dict:
    text = raw
    if "```" in text:
        start = text.find("{")
        end = text.rfind("}") + 1
        text = text[start:end]

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error("[strategy_tuner] Failed to parse response as JSON: %s\nRaw: %s", exc, raw)
        return {}


def _validate_strategy_delta(strategy: str, delta: dict) -> dict:
    """
    Multi-phase validation:
      0. Enforce permission tier — strip disallowed keys for ML_ONLY strategies.
      1. Strip forbidden and unknown keys.
      2. Validate scalar numerics via base.validate_delta() + _EXTRA_GUARDRAILS.
      3. Validate bool toggles (use_ml_filter always; use_chart_filter/use_reanalyze
         only for FULL_CONTROL chart variants).
      4. Validate tp_levels structure and per-level bounds (FULL_CONTROL only).
      5. Pass "reason" through unchanged.
    """
    clean: dict[str, Any] = {}

    # Phase 0: permission tier — ML_ONLY strategies may only change ML params
    if strategy in ML_ONLY_STRATEGIES:
        kept = {k: v for k, v in delta.items() if k in _ML_ONLY_ALLOWED_KEYS or k == "reason"}
        stripped = set(delta) - set(kept)
        if stripped:
            logger.warning(
                "[strategy_tuner] ML_ONLY strategy %s: dropping disallowed keys %s",
                strategy, sorted(stripped),
            )
        delta = kept

    # Phase 1 + 2: scalars
    scalar_keys = {k: v for k, v in delta.items()
                   if k in _ALLOWED_SCALAR_KEYS and k != "use_ml_filter"}

    validated_scalars = validate_delta(scalar_keys)  # handles base GUARDRAILS

    # Apply _EXTRA_GUARDRAILS on top
    for key, value in validated_scalars.items():
        if key in _EXTRA_GUARDRAILS:
            lo, hi = _EXTRA_GUARDRAILS[key]
            clamped = max(lo, min(hi, float(value)))
            if clamped != value:
                logger.warning(
                    "[strategy_tuner] extra guardrail: %s=%.2f clamped to %.2f",
                    key, value, clamped,
                )
            validated_scalars[key] = clamped

    # Drop moonbag strategies' timeout keys (they have no timeout)
    if strategy in _MOONBAG_STRATEGIES:
        validated_scalars.pop("timeout_minutes", None)
        validated_scalars.pop("max_hold_minutes", None)

    clean.update(validated_scalars)

    # Phase 3: bool toggles
    # live_trading:     all controlled strategies (both tiers)
    # use_ml_filter:    any chart variant (both FULL_CONTROL and ML_ONLY)
    # use_chart_filter / use_reanalyze: FULL_CONTROL chart variants only
    for bool_key in ("live_trading", "use_ml_filter", "use_chart_filter", "use_reanalyze"):
        if bool_key not in delta:
            continue
        if bool_key == "use_ml_filter" and strategy not in _CHART_VARIANTS:
            continue
        if bool_key in ("use_chart_filter", "use_reanalyze") and strategy not in FULL_CONTROL_STRATEGIES:
            logger.warning(
                "[strategy_tuner] ML_ONLY strategy %s cannot change %s — skipping",
                strategy, bool_key,
            )
            continue
        val = delta[bool_key]
        if isinstance(val, bool):
            clean[bool_key] = val
        else:
            logger.warning(
                "[strategy_tuner] %s must be bool, got %r — skipping", bool_key, val
            )

    # ml_k must be int — cast from float if needed
    if "ml_k" in clean:
        clean["ml_k"] = int(round(clean["ml_k"]))

    # Phase 4: tp_levels (FULL_CONTROL strategies only)
    if "tp_levels" in delta:
        if strategy in ML_ONLY_STRATEGIES:
            logger.warning(
                "[strategy_tuner] ML_ONLY strategy %s cannot change tp_levels — dropping",
                strategy,
            )
        else:
            tp_result = _validate_tp_levels(strategy, delta["tp_levels"])
            if tp_result is not None:
                clean["tp_levels"] = tp_result
            else:
                logger.warning(
                    "[strategy_tuner] tp_levels failed validation for %s — dropping TP changes",
                    strategy,
                )

    # Phase 5: reason passthrough
    if "reason" in delta:
        clean["reason"] = str(delta["reason"])

    return clean


def _validate_tp_levels(
    strategy: str,
    tp_levels: Any,
) -> Optional[list]:
    """
    Validate tp_levels list:
    - Must be a list of [multiple, fraction] pairs.
    - Length must match expected count for this strategy.
    - Each pair must satisfy TP_GUARDRAILS bounds.
    - Multiples must be strictly ascending.
    - Total fractions must not exceed 1.0.

    Returns the validated list on success, None on any failure.
    """
    guardrails = _TP_GUARDRAILS.get(strategy)
    if not guardrails:
        logger.warning("[strategy_tuner] No TP guardrails for strategy %s", strategy)
        return None

    expected_count = len(guardrails)

    if not isinstance(tp_levels, list) or len(tp_levels) != expected_count:
        logger.warning(
            "[strategy_tuner] tp_levels has %s entries but expected %d for %s",
            len(tp_levels) if isinstance(tp_levels, list) else type(tp_levels).__name__,
            expected_count, strategy,
        )
        return None

    validated = []
    for i, level in enumerate(tp_levels):
        if not (isinstance(level, (list, tuple)) and len(level) == 2):
            logger.warning("[strategy_tuner] tp_levels[%d] is not a [multiple, fraction] pair", i)
            return None
        multiple, fraction = float(level[0]), float(level[1])
        min_m, max_m, min_f, max_f = guardrails[i]
        if not (min_m <= multiple <= max_m):
            logger.warning(
                "[strategy_tuner] tp%d multiple=%.2f outside [%.2f, %.2f]",
                i + 1, multiple, min_m, max_m,
            )
            return None
        if not (min_f <= fraction <= max_f):
            logger.warning(
                "[strategy_tuner] tp%d fraction=%.2f outside [%.2f, %.2f]",
                i + 1, fraction, min_f, max_f,
            )
            return None
        validated.append([multiple, fraction])

    # Ascending multiples
    for i in range(1, len(validated)):
        if validated[i][0] <= validated[i - 1][0]:
            logger.warning(
                "[strategy_tuner] tp_levels multiples not strictly ascending: %s",
                [v[0] for v in validated],
            )
            return None

    # Total fractions <= 1.0
    total_frac = sum(v[1] for v in validated)
    if total_frac > 1.0:
        logger.warning(
            "[strategy_tuner] tp_levels total fractions=%.2f exceeds 1.0", total_frac
        )
        return None

    return validated


# ---------------------------------------------------------------------------
# Apply delta to config dict
# ---------------------------------------------------------------------------

def _apply_delta(strategy: str, delta: dict, config: dict) -> None:
    """
    Merge validated delta into config[strategy] in-place.
    tp_levels replaces the entire list.
    Scalar/bool keys are overwritten.
    'reason' is never written to the config file.
    """
    if strategy not in config:
        config[strategy] = {}

    for key, value in delta.items():
        if key == "reason":
            continue
        config[strategy][key] = value
