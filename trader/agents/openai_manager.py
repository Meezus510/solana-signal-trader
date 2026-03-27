"""
trader/agents/openai_manager.py — OpenAI-backed controller for open_ai_managed.

This agent owns exactly one strategy: `open_ai_managed`.
It can switch base family, aggressiveness mode, ML settings, filters, and exits,
then persist those changes to strategy_config.json.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from trader.agents.base import log_agent_action, query_exit_stats, query_recent_trades, query_score_buckets
from trader.agents.strategy_tuner import load_config, save_owned_config
from trader.strategies.registry import _OPEN_AI_MANAGED_BASES, _OPEN_AI_MANAGED_MODES

logger = logging.getLogger(__name__)

_STRATEGY = "open_ai_managed"
_META_PREFIX = "openai_manager_"
_MODEL = os.getenv("OPENAI_MANAGER_MODEL", "gpt-5.4-mini")

_ALLOWED_SCALARS = {
    "stop_loss_pct",
    "trailing_stop_pct",
    "timeout_minutes",
    "timeout_min_gain_pct",
    "max_hold_minutes",
    "peak_drop_exit_pct",
    "early_timeout_minutes",
    "early_timeout_max_gain_pct",
    "early_timeout_min_range_pct",
    "ml_min_score",
    "ml_high_score_threshold",
    "ml_max_score_threshold",
    "ml_size_multiplier",
    "ml_max_size_multiplier",
    "ml_k",
    "ml_halflife_days",
    "ml_score_low_pct",
    "ml_score_high_pct",
    "holder_count_max",
    "late_entry_price_chg_30m_max",
    "late_entry_pump_ratio_min",
    "buy_vol_ratio_1h_max",
    "market_cap_usd_min",
    "ml_wallet_momentum_max",
}

_ALLOWED_BOOLS = {
    "use_ml_filter",
}

_RANGES = {
    "stop_loss_pct": (0.01, 0.50),
    "trailing_stop_pct": (0.01, 0.50),
    "timeout_minutes": (5.0, 720.0),
    "timeout_min_gain_pct": (0.0, 2.0),
    "max_hold_minutes": (5.0, 1440.0),
    "peak_drop_exit_pct": (0.01, 0.80),
    "early_timeout_minutes": (1.0, 240.0),
    "early_timeout_max_gain_pct": (0.0, 1.0),
    "early_timeout_min_range_pct": (0.0, 1.0),
    "ml_min_score": (0.0, 10.0),
    "ml_high_score_threshold": (0.0, 10.0),
    "ml_max_score_threshold": (0.0, 10.0),
    "ml_size_multiplier": (1.0, 5.0),
    "ml_max_size_multiplier": (1.0, 8.0),
    "ml_k": (1.0, 25.0),
    "ml_halflife_days": (1.0, 90.0),
    "ml_score_low_pct": (-100.0, 100.0),
    "ml_score_high_pct": (1.0, 500.0),
    "holder_count_max": (1.0, 100000.0),
    "late_entry_price_chg_30m_max": (1.0, 5000.0),
    "late_entry_pump_ratio_min": (1.0, 100.0),
    "buy_vol_ratio_1h_max": (0.0, 1.0),
    "market_cap_usd_min": (0.0, 1_000_000_000.0),
    "ml_wallet_momentum_max": (0.0, 10.0),
}


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _validate_tp_levels(value: Any) -> list[list[float]] | None:
    if not isinstance(value, list) or not value:
        return None
    clean: list[list[float]] = []
    total = 0.0
    for item in value[:4]:
        if not isinstance(item, list) or len(item) != 2:
            return None
        mult = float(item[0])
        frac = float(item[1])
        mult = _clamp(mult, 1.01, 12.0)
        frac = _clamp(frac, 0.05, 1.0)
        if total + frac > 1.0:
            frac = round(1.0 - total, 2)
        if frac <= 0:
            break
        clean.append([round(mult, 4), round(frac, 4)])
        total += frac
    clean.sort(key=lambda x: x[0])
    return clean or None


def _validate_weights(value: Any) -> tuple[float, ...] | None:
    if not isinstance(value, list) or not value:
        return None
    clean = []
    for item in value[:63]:
        try:
            clean.append(round(_clamp(float(item), 0.0, 10.0), 4))
        except (TypeError, ValueError):
            return None
    return tuple(clean)


def _validate_delta(delta: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}

    base = delta.get("base_strategy")
    if base in _OPEN_AI_MANAGED_BASES:
        out["base_strategy"] = base

    mode = delta.get("mode")
    if isinstance(mode, str) and mode in {"allow_all", "lenient", "balanced", "strict", "block_all"}:
        out["mode"] = mode

    for key in _ALLOWED_BOOLS:
        if key in delta and isinstance(delta[key], bool):
            out[key] = delta[key]

    for key in _ALLOWED_SCALARS:
        if key not in delta or delta[key] is None:
            continue
        try:
            value = float(delta[key])
        except (TypeError, ValueError):
            continue
        lo, hi = _RANGES[key]
        value = _clamp(value, lo, hi)
        if key in {"ml_k", "holder_count_max"}:
            value = int(round(value))
        out[key] = value

    if "tp_levels" in delta:
        clean_tp = _validate_tp_levels(delta["tp_levels"])
        if clean_tp is not None:
            out["tp_levels"] = clean_tp

    if "ml_feature_weights" in delta:
        clean_weights = _validate_weights(delta["ml_feature_weights"])
        if clean_weights is not None:
            out["ml_feature_weights"] = clean_weights

    if "reason" in delta and isinstance(delta["reason"], str):
        out["reason"] = delta["reason"][:1000]

    return out


def _leaderboard(db_path: str) -> list[dict]:
    from scripts.backtest_open_ai_managed import backtest_named_mode

    rows = []
    for base in sorted(_OPEN_AI_MANAGED_BASES):
        for mode in ["allow_all", "lenient", "balanced", "strict", "block_all"]:
            metrics = backtest_named_mode(db_path, base, mode)
            rows.append({
                "base_strategy": base,
                "mode": mode,
                "entered": metrics["entered"],
                "block_rate": round(metrics["block_rate"], 4),
                "win_rate": round(metrics["win_rate"], 4),
                "total_pnl_usd": round(metrics["total_pnl_usd"], 2),
                "avg_pnl_per_trade_usd": round(metrics["avg_pnl_per_trade_usd"], 4),
            })
    rows.sort(key=lambda r: (r["total_pnl_usd"], r["win_rate"]), reverse=True)
    return rows[:12]


def _current_backtest(db_path: str, current: dict) -> dict:
    from scripts.backtest_open_ai_managed import backtest_config

    return backtest_config(db_path, current)


def _prompt(current_cfg: dict, current_metrics: dict, leaderboard: list[dict], exit_stats: list[dict], recent: list[dict], score_buckets: list[dict]) -> str:
    return f"""You control one paper-trading strategy: open_ai_managed.

Your job is to improve it by adjusting ONLY its own config.

You may change:
- base_strategy: quick_pop | trend_rider | safe_bet | infinite_moonbag
- mode: allow_all | lenient | balanced | strict | block_all
- tp_levels, stop_loss_pct, trailing_stop_pct, timeout_minutes, timeout_min_gain_pct, max_hold_minutes
- peak_drop_exit_pct, early_timeout_minutes, early_timeout_max_gain_pct, early_timeout_min_range_pct
- use_ml_filter, ml thresholds, ml_k, ml_halflife_days, ml score mapping
- hard filters and ml_feature_weights

Current config:
{json.dumps(current_cfg, indent=2, default=list)}

Current local backtest:
{json.dumps(current_metrics, indent=2)}

Best built-in leaderboard from local backtests:
{json.dumps(leaderboard, indent=2)}

Actual open_ai_managed exit stats:
{json.dumps(exit_stats, indent=2)}

Recent open_ai_managed trades:
{json.dumps(recent[-12:], indent=2)}

ML score buckets for open_ai_managed:
{json.dumps(score_buckets, indent=2)}

Rules:
- Prefer simple changes over wide churn.
- If the current setup is clearly worse than a leaderboard option, switch to that base/mode.
- Use block_all only as an emergency brake, not as a default.
- If you change tp_levels, return the full list.
- If you change ml_feature_weights, return the full list.
- Return JSON only.

Return an object with any subset of:
base_strategy, mode, tp_levels, stop_loss_pct, trailing_stop_pct, timeout_minutes,
timeout_min_gain_pct, max_hold_minutes, peak_drop_exit_pct,
early_timeout_minutes, early_timeout_max_gain_pct, early_timeout_min_range_pct,
use_ml_filter, ml_min_score, ml_high_score_threshold, ml_max_score_threshold,
ml_size_multiplier, ml_max_size_multiplier, ml_k, ml_halflife_days,
ml_score_low_pct, ml_score_high_pct, holder_count_max,
late_entry_price_chg_30m_max, late_entry_pump_ratio_min, buy_vol_ratio_1h_max,
market_cap_usd_min, ml_wallet_momentum_max, ml_feature_weights, reason.
"""


def run(db_path: str = "trader.db", dry_run: bool = False) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package is not installed") from exc

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set — required for open_ai_manager")

    config = load_config()
    current = dict(config.get(_STRATEGY, {}))
    current.setdefault("base_strategy", "quick_pop")
    current.setdefault("mode", "balanced")

    leaderboard = _leaderboard(db_path)
    current_metrics = _current_backtest(db_path, current)
    exit_stats = query_exit_stats(db_path, _STRATEGY)
    recent = query_recent_trades(db_path, _STRATEGY, limit=20)
    score_buckets = query_score_buckets(db_path, _STRATEGY)

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=_MODEL,
        input=_prompt(current, current_metrics, leaderboard, exit_stats, recent, score_buckets),
        text={"format": {"type": "json_object"}},
    )
    raw = response.output_text.strip()
    logger.debug("[openai_manager] Raw response: %s", raw)

    try:
        proposed = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("[openai_manager] Failed to parse JSON: %s", exc)
        return {}

    validated = _validate_delta(proposed)
    if not validated:
        return {}

    reason = validated.get("reason", "")
    before = {k: current.get(k) for k in validated if k != "reason"}

    if not dry_run:
        original_config = json.loads(json.dumps(config))
        config.setdefault(_STRATEGY, {})
        for key, value in validated.items():
            if key == "reason":
                continue
            config[_STRATEGY][key] = value
        config.setdefault("_meta", {})
        config["_meta"]["openai_manager_last_run_at"] = datetime.now(timezone.utc).isoformat()
        save_owned_config(
            original_config,
            config,
            owned_strategy=_STRATEGY,
            allowed_meta_prefixes=(_META_PREFIX,),
        )
        log_agent_action("openai_manager", _STRATEGY, validated, before)

    logger.info("[openai_manager] %s", reason or validated)
    return validated
