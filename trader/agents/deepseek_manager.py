"""
DeepSeek-backed managed-strategy adapter for `deepseek_managed`.
"""

from __future__ import annotations

import os
from typing import Any

from trader.agents.managed_agent_base import ManagedAgentSpec, run_managed_agent
from trader.agents.provider_adapters import DeepSeekJSONProvider

_SPEC = ManagedAgentSpec(
    strategy_name="deepseek_managed",
    agent_name="deepseek_manager",
    meta_prefix="deepseek_manager_",
    default_model="deepseek-chat",
    allowed_scalars={
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
    },
    allowed_bools={"use_ml_filter"},
    ranges={
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
    },
)


def run(db_path: str = "trader.db", dry_run: bool = False) -> dict[str, Any]:
    return run_managed_agent(
        _SPEC,
        DeepSeekJSONProvider(),
        db_path=db_path,
        dry_run=dry_run,
        model=os.getenv("DEEPSEEK_MANAGER_MODEL", _SPEC.default_model),
    )