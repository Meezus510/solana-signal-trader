"""
Provider-agnostic manager base for AI-managed strategies.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from trader.agents.base import (
    agent_log_path,
    log_agent_action,
    query_exit_stats,
    query_recent_trades,
    query_regime_context,
    query_score_buckets,
    query_strategy_pnl_snapshots,
)
from trader.agents.strategy_tuner import load_config, save_owned_config
from trader.analysis.managed_backtest import (
    backtest_managed_config,
    compare_modes_for_base,
)
from trader.strategies.registry import _TREND_RIDER_ML_MODES, get_managed_strategy_spec

logger = logging.getLogger(__name__)

_AI_MANAGED_STRATEGIES = ("open_ai_managed", "anthropic_managed", "deepseek_managed")
_NON_AI_REFERENCE_STRATEGIES = (
    "quick_pop",
    "trend_rider",
    "safe_bet",
    "infinite_moonbag",
    "quick_pop_managed",
    "trend_rider_managed",
    "moonbag_managed",
)


def _reference_strategy_context(config: dict) -> dict[str, Any]:
    context: dict[str, Any] = {}
    for strategy in _NON_AI_REFERENCE_STRATEGIES:
        context[strategy] = {
            "current_config": dict(config.get(strategy, {})),
        }
    if "trend_rider_managed" in context:
        context["trend_rider_managed"]["available_modes"] = _TREND_RIDER_ML_MODES
    return context


@dataclass(frozen=True)
class ManagedAgentSpec:
    strategy_name: str
    agent_name: str
    meta_prefix: str
    default_model: str
    allowed_scalars: set[str]
    allowed_bools: set[str]
    ranges: dict[str, tuple[float, float]]
    max_tp_levels: int = 4
    max_feature_weights: int = 63


def _load_recent_agent_history(strategy_name: str, max_lines: int = 8) -> list[str]:
    log_path = agent_log_path(strategy_name)
    if not log_path.exists():
        return []
    lines = [line for line in log_path.read_text().splitlines() if line.strip()]
    return lines[-max_lines:]


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _validate_tp_levels(value: Any, max_levels: int) -> list[list[float]] | None:
    if not isinstance(value, list) or not value:
        return None
    clean: list[list[float]] = []
    total = 0.0
    for item in value[:max_levels]:
        if not isinstance(item, list) or len(item) != 2:
            return None
        mult = _clamp(float(item[0]), 1.01, 12.0)
        frac = _clamp(float(item[1]), 0.05, 1.0)
        if total + frac > 1.0:
            frac = round(1.0 - total, 2)
        if frac <= 0:
            break
        clean.append([round(mult, 4), round(frac, 4)])
        total += frac
    clean.sort(key=lambda x: x[0])
    return clean or None


def _validate_weights(value: Any, max_feature_weights: int) -> tuple[float, ...] | None:
    if not isinstance(value, list) or not value:
        return None
    clean = []
    for item in value[:max_feature_weights]:
        try:
            clean.append(round(_clamp(float(item), 0.0, 10.0), 4))
        except (TypeError, ValueError):
            return None
    return tuple(clean)


def validate_managed_delta(spec: ManagedAgentSpec, delta: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    managed_spec = get_managed_strategy_spec(spec.strategy_name)

    base = delta.get("base_strategy")
    if base in managed_spec.bases:
        out["base_strategy"] = base

    mode = delta.get("mode")
    if isinstance(mode, str):
        allowed_modes = set().union(*(mode_table.keys() for mode_table in managed_spec.modes.values()))
        if mode in allowed_modes:
            out["mode"] = mode

    for key in spec.allowed_bools:
        if key in delta and isinstance(delta[key], bool):
            out[key] = delta[key]

    for key in spec.allowed_scalars:
        if key not in delta or delta[key] is None:
            continue
        try:
            value = float(delta[key])
        except (TypeError, ValueError):
            continue
        lo, hi = spec.ranges[key]
        value = _clamp(value, lo, hi)
        if key in {"ml_k", "holder_count_max"}:
            value = int(round(value))
        out[key] = value

    if "tp_levels" in delta:
        clean_tp = _validate_tp_levels(delta["tp_levels"], spec.max_tp_levels)
        if clean_tp is not None:
            out["tp_levels"] = clean_tp

    if "ml_feature_weights" in delta:
        clean_weights = _validate_weights(delta["ml_feature_weights"], spec.max_feature_weights)
        if clean_weights is not None:
            out["ml_feature_weights"] = clean_weights

    if "reason" in delta and isinstance(delta["reason"], str):
        out["reason"] = delta["reason"][:1000]

    return out


def build_managed_prompt(
    spec: ManagedAgentSpec,
    current_cfg: dict,
    current_metrics: dict,
    recent_mode_comparison: list[dict],
    exit_stats: list[dict],
    recent: list[dict],
    score_buckets: list[dict],
    regime_context: dict[str, Any],
    recent_history: list[str],
    ai_peer_pnl: list[dict[str, Any]],
    reference_strategy_pnl: list[dict[str, Any]],
    reference_strategy_context: dict[str, Any],
) -> str:
    managed_spec = get_managed_strategy_spec(spec.strategy_name)
    base_list = " | ".join(sorted(managed_spec.bases))
    mode_list = " | ".join(sorted(set().union(*(mode_table.keys() for mode_table in managed_spec.modes.values()))))
    return f"""You control one paper-trading strategy: {spec.strategy_name}.

Your job is to improve it by adjusting ONLY its own config.

You may change:
- base_strategy: {base_list}
- mode: {mode_list}
- tp_levels, stop_loss_pct, trailing_stop_pct, timeout_minutes, timeout_min_gain_pct, max_hold_minutes
- peak_drop_exit_pct, early_timeout_minutes, early_timeout_max_gain_pct, early_timeout_min_range_pct
- use_ml_filter, ml thresholds, ml_k, ml_halflife_days, ml score mapping
- hard filters and ml_feature_weights

Current config:
{json.dumps(current_cfg, indent=2, default=list)}

Current local backtest:
{json.dumps(current_metrics, indent=2)}

Recent-window comparison for current base only:
{json.dumps(recent_mode_comparison, indent=2)}

Actual {spec.strategy_name} exit stats:
{json.dumps(exit_stats, indent=2)}

Recent {spec.strategy_name} trades:
{json.dumps(recent[-12:], indent=2)}

ML score buckets for {spec.strategy_name}:
{json.dumps(score_buckets, indent=2)}

Recent regime/context snapshot:
{json.dumps(regime_context, indent=2)}

Your own most recent changes:
{json.dumps(recent_history, indent=2)}

Other AI agents: pnl only
{json.dumps(ai_peer_pnl, indent=2)}

Base and non-AI managed strategy performance:
{json.dumps(reference_strategy_pnl, indent=2)}

Base and non-AI managed strategy configs/modes:
{json.dumps(reference_strategy_context, indent=2, default=list)}

Rules:
- Prefer simple changes over wide churn.
- You may see other AI agents' pnl only. Do not infer or discuss their configs, modes, or hidden strategy settings.
- Use the base and non-AI managed strategy performance for market context, not as permission to copy hidden settings.
- Use regime/context data to decide when the current mode is too strict or too loose.
- If managed_strategy_recent block_rate is extremely high while base_strategy_recent looks healthy,
  consider loosening filters or switching base_strategy.
- Prefer one decision class per run:
  either change base/mode, or tune risk/ML thresholds, but not both unless the evidence is overwhelming.
- If current config has drifted far from a canonical preset and underperforms it, snap back toward the preset instead of incremental tweaks.
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


def run_managed_agent(
    spec: ManagedAgentSpec,
    provider,
    *,
    db_path: str = "trader.db",
    dry_run: bool = False,
    model: str | None = None,
) -> dict[str, Any]:
    config = load_config()
    current = dict(config.get(spec.strategy_name, {}))
    managed_spec = get_managed_strategy_spec(spec.strategy_name)
    current.setdefault("base_strategy", managed_spec.default_base)
    current.setdefault("mode", managed_spec.default_mode)

    current_metrics = backtest_managed_config(db_path, spec.strategy_name, current)
    recent_date_from = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
    recent_mode_comparison = compare_modes_for_base(
        db_path,
        spec.strategy_name,
        current.get("base_strategy", managed_spec.default_base),
        date_from=recent_date_from,
    )[:5]
    exit_stats = query_exit_stats(db_path, spec.strategy_name)
    recent = query_recent_trades(db_path, spec.strategy_name, limit=20)
    score_buckets = query_score_buckets(db_path, spec.strategy_name)
    regime_context = query_regime_context(
        db_path,
        spec.strategy_name,
        base_strategy=current.get("base_strategy"),
    )
    recent_history = _load_recent_agent_history(spec.strategy_name)
    ai_peer_pnl = query_strategy_pnl_snapshots(
        db_path,
        [name for name in _AI_MANAGED_STRATEGIES if name != spec.strategy_name],
    )
    reference_strategy_pnl = query_strategy_pnl_snapshots(
        db_path,
        list(_NON_AI_REFERENCE_STRATEGIES),
    )
    reference_strategy_context = _reference_strategy_context(config)

    prompt = build_managed_prompt(
        spec,
        current,
        current_metrics,
        recent_mode_comparison,
        exit_stats,
        recent,
        score_buckets,
        regime_context,
        recent_history,
        ai_peer_pnl,
        reference_strategy_pnl,
        reference_strategy_context,
    )
    raw = provider.generate_json(prompt, model=model or spec.default_model).strip()
    logger.debug("[%s] Raw response: %s", spec.agent_name, raw)

    try:
        proposed = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("[%s] Failed to parse JSON: %s", spec.agent_name, exc)
        return {}

    validated = validate_managed_delta(spec, proposed)
    validated = {
        key: value
        for key, value in validated.items()
        if key == "reason" or current.get(key) != value
    }
    if not validated:
        return {}

    reason = validated.get("reason", "")
    before = {k: current.get(k) for k in validated if k != "reason"}

    if not dry_run:
        original_config = json.loads(json.dumps(config))
        config.setdefault(spec.strategy_name, {})
        for key, value in validated.items():
            if key == "reason":
                continue
            config[spec.strategy_name][key] = value
        config.setdefault("_meta", {})
        config["_meta"][f"{spec.meta_prefix}last_run_at"] = datetime.now(timezone.utc).isoformat()
        save_owned_config(
            original_config,
            config,
            owned_strategy=spec.strategy_name,
            allowed_meta_prefixes=(spec.meta_prefix,),
        )
        log_agent_action(spec.agent_name, spec.strategy_name, validated, before)

    logger.info("[%s] %s", spec.agent_name, reason or validated)
    return validated
