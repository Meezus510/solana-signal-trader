"""
trader/strategies/registry.py — Strategy roster and runner factory.

Single source of truth for which strategies are active and how they are
configured.  Both the live entry point (run.py) and offline tools
(scripts/summary.py, scripts/backtest_chart.py) import build_runners()
from here — never from run.py.

Adding a new strategy:
    Define a StrategyConfig below and append a StrategyRunner (or subclass)
    to the list returned by build_runners().  Every strategy is fully
    isolated: its own cash, positions, and PnL.  All strategies share a
    single Birdeye price feed.

Strategy groups
---------------
Group A — no chart filter (enters every signal unconditionally):
    quick_pop, trend_rider, infinite_moonbag, safe_bet

Group B — chart filter enabled (skips late pumps and dying volume):
    quick_pop_managed (+ ML filter), trend_rider_managed (+ reanalyze), moonbag_managed
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from trader.config import Config
from trader.trading.strategy import (
    InfiniteMoonbagRunner,
    StrategyConfig,
    StrategyRunner,
    TakeProfitLevel,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# trend_rider_managed ML filter modes
# ---------------------------------------------------------------------------
# Each mode overrides ml_min_score and the three hard entry filters.
# Config values in strategy_config.json still take final precedence over modes
# (explicit config key beats the mode default), but if the key is absent the
# mode value is used instead of the hardcoded default.
#
#  lenient  — mild filtering, keeps ~30-40 % of signals. Useful for data
#             collection while reducing obvious duds.
#  balanced — moderate filtering (~15-20 % pass). Good risk/reward trade-off
#             once enough training data exists.
#  strict   — aggressive filtering (≤5 % pass). Precision-first: only enter
#             when the KNN is very confident AND hard filters all clear.
#             Use this when win rate matters more than opportunity volume.
_TREND_RIDER_ML_MODES: dict[str, dict] = {
    "lenient": {
        # Goal: miss ZERO winners, block as many losers as possible.
        # Backtest (83 closed trades, 17W / 66L):
        #   - hc_max=900  misses 4/17 winners (max winner hc = 2119) → raised to 3000
        #   - ml_min=3.5  misses 0/17 winners (all winners scored ≥5.0)
        #   - late_entry filters very loose so no moonshot is excluded
        "use_ml_filter":                 True,
        "ml_min_score":                  3.5,
        "holder_count_max":              3000,
        "late_entry_price_chg_30m_max":  350.0,
        "late_entry_pump_ratio_min":     10.0,
    },
    "balanced": {
        # Mirrors current production defaults — use this as the baseline.
        "use_ml_filter":                 True,
        "ml_min_score":                  5.0,
        "holder_count_max":              1000,
        "late_entry_price_chg_30m_max":  250.0,
        "late_entry_pump_ratio_min":     20.0,
    },
    "strict": {
        # Precision-first: only enter when ALL filters are very confident.
        # Targets ≤5% pass rate with >50% win rate based on 687-trade backtest.
        "use_ml_filter":                 True,
        "ml_min_score":                  8.0,
        "holder_count_max":              350,
        "late_entry_price_chg_30m_max":  100.0,
        "late_entry_pump_ratio_min":     8.0,
        # buy_vol_ratio_1h < 0.016 → 60% win rate at 95% block (backtest)
        "buy_vol_ratio_1h_max":          0.016,
        # market_cap_usd > $300k → 60% win rate at 95% block (backtest)
        "market_cap_usd_min":            300_000.0,
        # Tighter stop loss: fewer false-stop-outs, limits damage when wrong
        "stop_loss_pct":                 0.18,
        "trailing_stop_pct":             0.20,
    },
}

# ---------------------------------------------------------------------------
# open_ai_managed profiles
# ---------------------------------------------------------------------------
# One OpenAI-managed strategy with:
#   - its own isolated paper portfolio
#   - switchable base strategy family
#   - switchable aggressiveness mode
#   - local backtests driving config updates
#
# Supported base families:
#   quick_pop, trend_rider, safe_bet, infinite_moonbag
#
# Mode semantics:
#   allow_all — disable gating; take every signal
#   lenient   — loose filters, maximize opportunity volume
#   balanced  — default live research mode
#   strict    — precision-first
#   block_all — reject every new entry but keep managing open positions
_OPEN_AI_QP_WEIGHTS: tuple[float, ...] = (
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 8.0, 4.0, 3.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 1.0,
    0.0, 0.0, 0.0, 4.0, 0.0, 0.0,
    0.0, 1.0, 2.0, 0.0, 2.5, 0.5, 1.5, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.5, 0.0, 0.0,
)

_OPEN_AI_TR_WEIGHTS: tuple[float, ...] = (
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    3.2, 0.0, 3.2, 0.0, 2.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 3.2, 2.2,
    0.0, 2.2, 0.0,
    0.0, 0.1, 0.0, 0.0,
    2.0, 1.5,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
)

_OPEN_AI_MB_WEIGHTS: tuple[float, ...] = (
    3.0, 1.0, 3.0, 1.0, 3.0, 1.0,
    7.6, 1.78, 6.9, 5.1, 0.38, 0.28,
    0.0, 0.0, 0.0, 0.0, 0.01, 0.22,
    0.04, 0.02, 0.0,
    0.0, 0.12, 0.0, 0.0,
    1.5, 4.0,
)

_OPEN_AI_MANAGED_BASES: dict[str, dict] = {
    "quick_pop": {
        "runner": "standard",
        "buy_size_usd": 30.0,
        "timeout_minutes": 45.0,
        "timeout_min_gain_pct": 0.49,
        "max_hold_minutes": None,
        "peak_drop_exit_pct": 0.12,
        "early_timeout_minutes": 12.0,
        "early_timeout_max_gain_pct": 0.02,
        "early_timeout_min_range_pct": 0.06,
        "ml_training_strategy": "quick_pop",
        "ml_training_label": "position_peak_pnl_pct",
        "ml_use_subminute": True,
        "ml_k": 3,
        "ml_halflife_days": 7.0,
        "ml_score_low_pct": -45.0,
        "ml_score_high_pct": 300.0,
        "ml_feature_weights": _OPEN_AI_QP_WEIGHTS,
        "tp_levels": [[1.26, 0.78], [1.98, 0.22]],
        "stop_loss_pct": 0.06,
        "trailing_stop_pct": 0.07,
    },
    "trend_rider": {
        "runner": "standard",
        "buy_size_usd": 10.0,
        "timeout_minutes": 90.0,
        "timeout_min_gain_pct": 0.15,
        "max_hold_minutes": 240.0,
        "peak_drop_exit_pct": 0.18,
        "early_timeout_minutes": 25.0,
        "early_timeout_max_gain_pct": 0.03,
        "early_timeout_min_range_pct": 0.08,
        "ml_training_strategy": "trend_rider",
        "ml_training_label": "outcome_pnl_pct",
        "ml_use_subminute": False,
        "ml_k": 5,
        "ml_halflife_days": 14.0,
        "ml_score_low_pct": -35.0,
        "ml_score_high_pct": 85.0,
        "ml_feature_weights": _OPEN_AI_TR_WEIGHTS,
        "tp_levels": [[2.5, 0.50]],
        "stop_loss_pct": 0.35,
        "trailing_stop_pct": 0.28,
    },
    "safe_bet": {
        "runner": "standard",
        "buy_size_usd": 30.0,
        "timeout_minutes": 60.0,
        "timeout_min_gain_pct": 0.10,
        "max_hold_minutes": None,
        "peak_drop_exit_pct": 0.08,
        "early_timeout_minutes": 20.0,
        "early_timeout_max_gain_pct": 0.02,
        "early_timeout_min_range_pct": 0.05,
        "ml_training_strategy": "safe_bet",
        "ml_training_label": "outcome_pnl_pct",
        "ml_use_subminute": False,
        "ml_k": 5,
        "ml_halflife_days": 14.0,
        "ml_score_low_pct": -20.0,
        "ml_score_high_pct": 25.0,
        "ml_feature_weights": None,
        "tp_levels": [[1.20, 1.0]],
        "stop_loss_pct": 0.05,
        "trailing_stop_pct": 0.05,
        "use_real_exit_price": True,
    },
    "infinite_moonbag": {
        "runner": "moonbag",
        "buy_size_usd": 5.0,
        "timeout_minutes": None,
        "timeout_min_gain_pct": None,
        "max_hold_minutes": None,
        "peak_drop_exit_pct": None,
        "early_timeout_minutes": None,
        "early_timeout_max_gain_pct": None,
        "early_timeout_min_range_pct": None,
        "ml_training_strategy": "infinite_moonbag",
        "ml_training_label": "position_peak_pnl_pct",
        "ml_use_subminute": True,
        "ml_k": 3,
        "ml_halflife_days": 14.0,
        "ml_score_low_pct": 0.0,
        "ml_score_high_pct": 200.0,
        "ml_feature_weights": _OPEN_AI_MB_WEIGHTS,
        "tp_levels": [[1.5, 0.15], [2.5, 0.12], [4.0, 0.12], [6.0, 0.08]],
        "stop_loss_pct": 0.35,
        "trailing_stop_pct": 0.35,
    },
}

_OPEN_AI_MANAGED_MODES: dict[str, dict[str, dict]] = {
    "quick_pop": {
        "allow_all": {
            "use_ml_filter": False,
            "block_new_entries": False,
            "holder_count_max": None,
            "late_entry_price_chg_30m_max": None,
            "late_entry_pump_ratio_min": None,
            "buy_vol_ratio_1h_max": None,
            "market_cap_usd_min": None,
            "ml_wallet_momentum_max": None,
        },
        "lenient": {
            "use_ml_filter": True,
            "block_new_entries": False,
            "tp_levels": [[1.16, 0.60], [1.85, 0.40]],
            "stop_loss_pct": 0.06,
            "trailing_stop_pct": 0.07,
            "ml_min_score": 2.0,
            "ml_high_score_threshold": 6.0,
            "ml_max_score_threshold": 6.0,
            "ml_size_multiplier": 1.10,
            "ml_max_size_multiplier": 1.75,
            "holder_count_max": 1500,
            "late_entry_price_chg_30m_max": 180.0,
            "late_entry_pump_ratio_min": 15.0,
            "buy_vol_ratio_1h_max": 0.55,
            "market_cap_usd_min": None,
            "ml_wallet_momentum_max": None,
        },
        "balanced": {
            "use_ml_filter": True,
            "block_new_entries": False,
            "tp_levels": [[1.16, 0.60], [2.30, 0.40]],
            "stop_loss_pct": 0.08,
            "trailing_stop_pct": 0.12,
            "ml_min_score": 2.25,
            "ml_high_score_threshold": 5.0,
            "ml_max_score_threshold": 6.0,
            "ml_size_multiplier": 1.35,
            "ml_max_size_multiplier": 1.75,
            "holder_count_max": 1000,
            "late_entry_price_chg_30m_max": 250.0,
            "late_entry_pump_ratio_min": 10.0,
            "buy_vol_ratio_1h_max": 0.65,
            "market_cap_usd_min": None,
            "ml_wallet_momentum_max": 2.5,
        },
        "strict": {
            "use_ml_filter": True,
            "block_new_entries": False,
            "tp_levels": [[1.16, 0.70], [1.85, 0.30]],
            "stop_loss_pct": 0.12,
            "trailing_stop_pct": 0.07,
            "ml_min_score": 2.25,
            "ml_high_score_threshold": 4.5,
            "ml_max_score_threshold": 7.0,
            "ml_size_multiplier": 1.10,
            "ml_max_size_multiplier": 1.20,
            "holder_count_max": 1500,
            "late_entry_price_chg_30m_max": 180.0,
            "late_entry_pump_ratio_min": 20.0,
            "buy_vol_ratio_1h_max": 0.65,
            "market_cap_usd_min": 100_000.0,
            "ml_wallet_momentum_max": 2.102,
        },
        "block_all": {
            "use_ml_filter": False,
            "block_new_entries": True,
        },
    },
    "trend_rider": {
        "allow_all": {
            "use_ml_filter": False,
            "block_new_entries": False,
            "holder_count_max": None,
            "late_entry_price_chg_30m_max": None,
            "late_entry_pump_ratio_min": None,
            "buy_vol_ratio_1h_max": None,
            "market_cap_usd_min": None,
        },
        "lenient": {
            "use_ml_filter": True,
            "block_new_entries": False,
            "ml_min_score": 3.5,
            "ml_high_score_threshold": 6.5,
            "ml_max_score_threshold": 8.0,
            "ml_size_multiplier": 1.25,
            "ml_max_size_multiplier": 1.75,
            "holder_count_max": 3000,
            "late_entry_price_chg_30m_max": 350.0,
            "late_entry_pump_ratio_min": 10.0,
            "buy_vol_ratio_1h_max": None,
            "market_cap_usd_min": None,
        },
        "balanced": {
            "use_ml_filter": True,
            "block_new_entries": False,
            "ml_min_score": 5.0,
            "ml_high_score_threshold": 8.0,
            "ml_max_score_threshold": 9.5,
            "ml_size_multiplier": 1.50,
            "ml_max_size_multiplier": 2.50,
            "holder_count_max": 1000,
            "late_entry_price_chg_30m_max": 250.0,
            "late_entry_pump_ratio_min": 20.0,
            "buy_vol_ratio_1h_max": None,
            "market_cap_usd_min": None,
        },
        "strict": {
            "use_ml_filter": True,
            "block_new_entries": False,
            "ml_min_score": 8.0,
            "ml_high_score_threshold": 8.5,
            "ml_max_score_threshold": 9.5,
            "ml_size_multiplier": 1.50,
            "ml_max_size_multiplier": 2.50,
            "holder_count_max": 350,
            "late_entry_price_chg_30m_max": 100.0,
            "late_entry_pump_ratio_min": 8.0,
            "buy_vol_ratio_1h_max": 0.016,
            "market_cap_usd_min": 300_000.0,
            "stop_loss_pct": 0.18,
            "trailing_stop_pct": 0.20,
        },
        "block_all": {
            "use_ml_filter": False,
            "block_new_entries": True,
        },
    },
    "safe_bet": {
        "allow_all": {
            "use_ml_filter": False,
            "block_new_entries": False,
        },
        "lenient": {
            "use_ml_filter": False,
            "block_new_entries": False,
            "holder_count_max": 3000,
        },
        "balanced": {
            "use_ml_filter": False,
            "block_new_entries": False,
            "holder_count_max": 1500,
            "buy_vol_ratio_1h_max": 0.75,
        },
        "strict": {
            "use_ml_filter": False,
            "block_new_entries": False,
            "holder_count_max": 700,
            "buy_vol_ratio_1h_max": 0.65,
            "market_cap_usd_min": 100_000.0,
        },
        "block_all": {
            "use_ml_filter": False,
            "block_new_entries": True,
        },
    },
    "infinite_moonbag": {
        "allow_all": {
            "use_ml_filter": False,
            "block_new_entries": False,
            "holder_count_max": None,
            "late_entry_price_chg_30m_max": None,
            "late_entry_pump_ratio_min": None,
        },
        "lenient": {
            "use_ml_filter": True,
            "block_new_entries": False,
            "ml_min_score": 2.0,
            "ml_high_score_threshold": 6.0,
            "ml_max_score_threshold": 8.0,
            "ml_size_multiplier": 1.25,
            "ml_max_size_multiplier": 1.75,
            "holder_count_max": 2000,
            "late_entry_price_chg_30m_max": 350.0,
            "late_entry_pump_ratio_min": 10.0,
        },
        "balanced": {
            "use_ml_filter": True,
            "block_new_entries": False,
            "ml_min_score": 2.0,
            "ml_high_score_threshold": 6.0,
            "ml_max_score_threshold": 8.0,
            "ml_size_multiplier": 2.0,
            "ml_max_size_multiplier": 3.0,
            "holder_count_max": 1000,
            "late_entry_price_chg_30m_max": 250.0,
            "late_entry_pump_ratio_min": 20.0,
        },
        "strict": {
            "use_ml_filter": True,
            "block_new_entries": False,
            "ml_min_score": 3.0,
            "ml_high_score_threshold": 6.5,
            "ml_max_score_threshold": 8.5,
            "ml_size_multiplier": 1.5,
            "ml_max_size_multiplier": 2.0,
            "holder_count_max": 700,
            "late_entry_price_chg_30m_max": 180.0,
            "late_entry_pump_ratio_min": 10.0,
            "buy_vol_ratio_1h_max": 0.65,
        },
        "block_all": {
            "use_ml_filter": False,
            "block_new_entries": True,
        },
    },
}

_CONTROLLED = frozenset([
    "trend_rider", "trend_rider_managed",
    "infinite_moonbag", "moonbag_managed",
    "quick_pop_managed",
    "open_ai_managed",
    "safe_bet",
])

_CONFIG_PATH = Path(__file__).parent.parent.parent / "strategy_config.json"


def _load_strategy_overrides() -> dict[str, dict]:
    """
    Load live parameter overrides from strategy_config.json.

    Returns a dict mapping strategy name → param dict for the four controlled
    strategies. Returns {} silently on any failure so the bot always starts.
    """
    try:
        with _CONFIG_PATH.open() as f:
            data = json.load(f)
        return {k: v for k, v in data.items() if k in _CONTROLLED}
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        logger.warning("[registry] Could not load strategy_config.json: %s — using defaults", exc)
        return {}


def build_runners(cfg: Config, db=None) -> list[StrategyRunner]:
    """
    Instantiate and return all active strategy runners.

    quick_pop / quick_pop_managed
        Fast scalp: TP1 at 1.5× (sell 60%) → TP2 at 2.0× (sell 40%) — fully exits.
        Trail at 22% below high after TP1.
        Exit after 45 min if TP1 not yet hit (price still < 1.49×).
        quick_pop_managed also applies chart filter + ML confidence gating (score < 5
        skips, score ≥ 8 doubles size, score ≥ 9.5 triples size).

    trend_rider / trend_rider_managed
        Momentum hold: TP1 at 1.8× (sell 50% of original).
        Trail at 30% below high after TP1.
        Exit after 90 min if price < entry × 1.15. Max hold: 4 hours.
        trend_rider_managed also re-checks skipped signals after a delay.

    infinite_moonbag / moonbag_managed (v2)
        Grace period 90s: −30% floor. After grace: −20% floor.
        TP ladder: 1.8×/20%, 2.5×/15%, 4.0×/15%, 6.0×/10% of original.
        Stop ladder: 1.8×→1.00× (breakeven), 2.5×→1.65×, 4.0×→2.60×, 6.0×→4.20×.

    safe_bet (Group A, paper-only)
        Clean risk/reward benchmark: single TP at 1.2× (sell 100%), hard SL at −5%.
        Timeout 60 min if price never reaches 1.2× and gain < 10%.
        No managed variant until enough trade data is collected.
    """
    overrides = _load_strategy_overrides()

    def _o(name: str) -> dict:
        return overrides.get(name, {})

    def _tp(name: str, defaults: list[TakeProfitLevel]) -> tuple[TakeProfitLevel, ...]:
        levels = _o(name).get("tp_levels")
        if not levels:
            return tuple(defaults)
        return tuple(TakeProfitLevel(multiple=m, sell_fraction_original=f) for m, f in levels)

    # ------------------------------------------------------------------
    # Group A — baseline (no chart filter)
    # ------------------------------------------------------------------

    quick_pop_cfg = StrategyConfig(
        name="quick_pop",
        buy_size_usd=30.0,
        stop_loss_pct=0.06,
        take_profit_levels=(
            TakeProfitLevel(multiple=1.26, sell_fraction_original=0.78),
            TakeProfitLevel(multiple=1.98, sell_fraction_original=0.22),
        ),
        trailing_stop_pct=0.07,
        starting_cash_usd=cfg.starting_cash_usd,
        timeout_minutes=45.0,
        timeout_min_gain_pct=0.49,
        peak_drop_exit_pct=_o("quick_pop").get("peak_drop_exit_pct", None),
        early_timeout_minutes=_o("quick_pop").get("early_timeout_minutes", None),
        early_timeout_max_gain_pct=_o("quick_pop").get("early_timeout_max_gain_pct", None),
        early_timeout_min_range_pct=_o("quick_pop").get("early_timeout_min_range_pct", None),
        save_chart_data=True,
        recently_closed_cooldown_minutes=_o("quick_pop").get("recently_closed_cooldown_minutes", 30.0),
    )

    trend_rider_cfg = StrategyConfig(
        name="trend_rider",
        buy_size_usd=cfg.buy_size_usd,
        stop_loss_pct=_o("trend_rider").get("stop_loss_pct", 0.30),
        take_profit_levels=_tp("trend_rider", [
            TakeProfitLevel(multiple=1.8, sell_fraction_original=0.50),
        ]),
        trailing_stop_pct=_o("trend_rider").get("trailing_stop_pct", 0.30),
        starting_cash_usd=cfg.starting_cash_usd,
        timeout_minutes=_o("trend_rider").get("timeout_minutes", 90.0),
        timeout_min_gain_pct=_o("trend_rider").get("timeout_min_gain_pct", 0.15),
        max_hold_minutes=_o("trend_rider").get("max_hold_minutes", 240.0),
        peak_drop_exit_pct=_o("trend_rider").get("peak_drop_exit_pct", None),
        early_timeout_minutes=_o("trend_rider").get("early_timeout_minutes", None),
        early_timeout_max_gain_pct=_o("trend_rider").get("early_timeout_max_gain_pct", None),
        early_timeout_min_range_pct=_o("trend_rider").get("early_timeout_min_range_pct", None),
        live_trading=_o("trend_rider").get("live_trading", False),
        save_chart_data=True,
        recently_closed_cooldown_minutes=_o("trend_rider").get("recently_closed_cooldown_minutes", 30.0),
    )

    moonbag_cfg = StrategyConfig(
        name="infinite_moonbag",
        buy_size_usd=5.0,
        stop_loss_pct=_o("infinite_moonbag").get("stop_loss_pct", 0.12),
        take_profit_levels=_tp("infinite_moonbag", [
            TakeProfitLevel(multiple=2.0, sell_fraction_original=0.35),
            TakeProfitLevel(multiple=4.3, sell_fraction_original=0.12),
            TakeProfitLevel(multiple=6.8, sell_fraction_original=0.38),
            TakeProfitLevel(multiple=8.0, sell_fraction_original=0.10),
        ]),
        trailing_stop_pct=_o("infinite_moonbag").get("trailing_stop_pct", 0.30),
        starting_cash_usd=cfg.starting_cash_usd,
        save_chart_data=True,
        live_trading=_o("infinite_moonbag").get("live_trading", False),
        recently_closed_cooldown_minutes=_o("infinite_moonbag").get("recently_closed_cooldown_minutes", 30.0),
    )

    safe_bet_cfg = StrategyConfig(
        name="safe_bet",
        buy_size_usd=30.0,
        stop_loss_pct=_o("safe_bet").get("stop_loss_pct", 0.05),
        take_profit_levels=(
            TakeProfitLevel(multiple=1.2, sell_fraction_original=1.0),
        ),
        trailing_stop_pct=0.05,   # unused (100% exits at TP), required field
        starting_cash_usd=cfg.starting_cash_usd,
        timeout_minutes=_o("safe_bet").get("timeout_minutes", 60.0),
        timeout_min_gain_pct=_o("safe_bet").get("timeout_min_gain_pct", 0.10),
        peak_drop_exit_pct=_o("safe_bet").get("peak_drop_exit_pct", None),
        early_timeout_minutes=_o("safe_bet").get("early_timeout_minutes", None),
        early_timeout_max_gain_pct=_o("safe_bet").get("early_timeout_max_gain_pct", None),
        early_timeout_min_range_pct=_o("safe_bet").get("early_timeout_min_range_pct", None),
        save_chart_data=True,
        live_trading=_o("safe_bet").get("live_trading", False),
        use_real_exit_price=True,
        recently_closed_cooldown_minutes=_o("safe_bet").get("recently_closed_cooldown_minutes", 30.0),
    )

    # ------------------------------------------------------------------
    # Group B — chart-filtered mirrors
    # ------------------------------------------------------------------

    _qp_chart = _o("quick_pop_managed")
    quick_pop_chart_cfg = StrategyConfig(
        name="quick_pop_managed",
        buy_size_usd=30.0,
        stop_loss_pct=0.06,       # fixed — not agent-controlled
        take_profit_levels=(      # fixed — not agent-controlled
            TakeProfitLevel(multiple=1.26, sell_fraction_original=0.78),
            TakeProfitLevel(multiple=1.98, sell_fraction_original=0.22),
        ),
        trailing_stop_pct=0.07,   # fixed — not agent-controlled
        starting_cash_usd=cfg.starting_cash_usd,
        timeout_minutes=45.0,     # fixed — not agent-controlled
        timeout_min_gain_pct=0.49,  # fixed — not agent-controlled
        use_chart_filter=False,   # locked off for quick_pop_managed — ML filter decides instead
        save_chart_data=True,
        use_ml_filter=True,          # always on for quick_pop_managed — not agent-controlled
        use_policy_agent=_qp_chart.get("use_policy_agent", False),
        use_ai_override=_qp_chart.get("use_ai_override", False),
        use_ai_override_shadow=_qp_chart.get("use_ai_override_shadow", False),
        ml_training_strategy="quick_pop",  # train on unfiltered base outcomes
        ml_training_label="position_peak_pnl_pct",  # predict peak pump, not exit PnL
        ml_use_subminute=True,       # use Birdeye v3 15s candles for KNN — fast-scalp pump shape
        ml_min_score=_qp_chart.get("ml_min_score", 3.0),
        ml_high_score_threshold=_qp_chart.get("ml_high_score_threshold", 6.0),
        ml_max_score_threshold=_qp_chart.get("ml_max_score_threshold", 8.0),
        ml_size_multiplier=_qp_chart.get("ml_size_multiplier", 1.2),
        ml_max_size_multiplier=_qp_chart.get("ml_max_size_multiplier", 2.0),
        ml_k=int(_qp_chart.get("ml_k", 5)),
        ml_halflife_days=_qp_chart.get("ml_halflife_days", 14.0),
        ml_score_low_pct=_qp_chart.get("ml_score_low_pct", -35.0),
        ml_score_high_pct=_qp_chart.get("ml_score_high_pct", 300.0),
        # Feature weights for quick_pop KNN (63 features total).
        # Weights from AI optimizer r8_s4: net_pnl=$+320.75 (saved=$408.92, missed=$88.17),
        # 107/123 winners through (13% miss), 59/267 losers blocked. 2026-03-23.
        ml_feature_weights=(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,            # idx  0-5:  15s OHLCV
            10.0, 0.0, 0.0, 0.0, 0.0, 0.0,           # idx  6-11: 1m OHLCV
            0.0, 8.0, 4.0, 3.0, 0.0, 0.0,            # idx 12-17: pair stats + source_channel
            0.0, 0.0, 0.0,                            # idx 18-20: token metadata
            0.0, 0.0, 0.0, 0.0,                      # idx 21-24: wallet (5m) — idx 24 buy_vol_ratio_5m zeroed (hard filter)
            0.0, 1.0,                                 # idx 25-26: wallet_momentum_30m, top10_holder_pct
            0.0, 0.0, 0.0, 4.0, 0.0, 0.0,            # idx 27-32: 1s OHLCV
            0.0, 1.0, 2.0, 0.0, 2.5, 0.5, 1.5, 0.0, 0.0, 0.0,  # idx 33-42: 15s shape
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # idx 43-52: 1m shape
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.5, 0.0, 0.0,  # idx 53-62: 1s shape
        ),
        # Hard pre-filter: block signals with wallet_momentum_5m >= 2.102.
        # LOO backtest on 197 trades: blocks 9/134 losers with 0 winner misses.
        ml_wallet_momentum_max=2.102,
        # Hard entry filters (data-derived, independent of ML).
        holder_count_max=_qp_chart.get("holder_count_max", 1000),
        late_entry_price_chg_30m_max=_qp_chart.get("late_entry_price_chg_30m_max", 250.0),
        late_entry_pump_ratio_min=_qp_chart.get("late_entry_pump_ratio_min", 20.0),
        peak_drop_exit_pct=_qp_chart.get("peak_drop_exit_pct", None),
        early_timeout_minutes=_qp_chart.get("early_timeout_minutes", None),
        early_timeout_max_gain_pct=_qp_chart.get("early_timeout_max_gain_pct", None),
        early_timeout_min_range_pct=_qp_chart.get("early_timeout_min_range_pct", None),
        live_trading=_qp_chart.get("live_trading", False),
        recently_closed_cooldown_minutes=_qp_chart.get("recently_closed_cooldown_minutes", 30.0),
    )

    _tr_chart = _o("trend_rider_managed")
    # Resolve mode defaults: explicit config keys override mode, mode overrides hardcoded defaults.
    _tr_mode_name = _tr_chart.get("ml_filter_mode")
    _tr_mode = _TREND_RIDER_ML_MODES.get(_tr_mode_name, {}) if _tr_mode_name else {}
    if _tr_mode_name and _tr_mode_name not in _TREND_RIDER_ML_MODES:
        logger.warning("[registry] Unknown trend_rider ml_filter_mode %r — ignoring", _tr_mode_name)

    def _tr(key, default):
        """Resolve: explicit config > mode default > hardcoded default."""
        if key in _tr_chart:
            return _tr_chart[key]
        return _tr_mode.get(key, default)

    if _tr_mode_name:
        logger.info("[registry] trend_rider_managed ml_filter_mode=%r", _tr_mode_name)

    trend_rider_chart_cfg = StrategyConfig(
        name="trend_rider_managed",
        buy_size_usd=cfg.buy_size_usd,
        stop_loss_pct=_tr("stop_loss_pct", 0.30),
        take_profit_levels=_tp("trend_rider_managed", [
            TakeProfitLevel(multiple=1.8, sell_fraction_original=0.50),
        ]),
        trailing_stop_pct=_tr("trailing_stop_pct", 0.30),
        starting_cash_usd=cfg.starting_cash_usd,
        timeout_minutes=_tr("timeout_minutes", 90.0),
        timeout_min_gain_pct=_tr("timeout_min_gain_pct", 0.15),
        max_hold_minutes=_tr("max_hold_minutes", 240.0),
        save_chart_data=True,
        use_chart_filter=_tr("use_chart_filter", True),
        pump_ratio_max=_tr("pump_ratio_max", 3.5),
        use_reanalyze=_tr("use_reanalyze", True),
        reanalyze_pump_delay=_tr("reanalyze_pump_delay", 480.0),
        reanalyze_vol_delay=_tr("reanalyze_vol_delay", 240.0),
        reanalyze_both_delay=_tr("reanalyze_both_delay", 600.0),
        ml_training_strategy="trend_rider",
        peak_drop_exit_pct=_tr("peak_drop_exit_pct", None),
        early_timeout_minutes=_tr("early_timeout_minutes", None),
        early_timeout_max_gain_pct=_tr("early_timeout_max_gain_pct", None),
        early_timeout_min_range_pct=_tr("early_timeout_min_range_pct", None),
        use_ml_filter=_tr("use_ml_filter", False),
        ml_min_score=_tr("ml_min_score", 5.0),
        ml_high_score_threshold=_tr("ml_high_score_threshold", 8.0),
        ml_max_score_threshold=_tr("ml_max_score_threshold", 9.5),
        ml_size_multiplier=_tr("ml_size_multiplier", 2.0),
        ml_max_size_multiplier=_tr("ml_max_size_multiplier", 3.0),
        ml_k=int(_tr("ml_k", 5)),
        ml_halflife_days=_tr("ml_halflife_days", 14.0),
        ml_score_low_pct=_tr("ml_score_low_pct", -35.0),
        ml_score_high_pct=_tr("ml_score_high_pct", 85.0),
        # Feature weights for trend_rider KNN (63 features total).
        # trend_rider is ml_use_subminute=False → candles_15s=[] and candles_1s=None at inference.
        # Locked to 0: idx 0-5 (15s OHLCV), 27-42 (1s OHLCV + 15s shape), 53-62 (1s shape).
        # To update: run scripts/optimize_ml_weights.py --strategy trend_rider
        ml_feature_weights=(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,            # idx  0-5:  15s OHLCV — locked (not fetched at inference)
            3.2, 0.0, 3.2, 0.0, 2.0, 0.0,            # idx  6-11: 1m OHLCV
            0.0, 0.0, 0.0, 0.0, 3.2, 2.2,            # idx 12-17: pair stats + source_channel
            0.0, 2.2, 0.0,                            # idx 18-20: token metadata
            0.0, 0.1, 0.0, 0.0,                      # idx 21-24: wallet (5m) — idx 24 buy_vol_ratio_5m zeroed (hard filter)
            2.0, 1.5,                                 # idx 25-26: wallet_momentum_30m, top10_holder_pct
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,            # idx 27-32: 1s OHLCV — locked
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # idx 33-42: 15s shape — locked
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # idx 43-52: 1m shape
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # idx 53-62: 1s shape — locked
        ),
        # Hard entry filters (data-derived, independent of ML).
        holder_count_max=_tr("holder_count_max", 1000),
        late_entry_price_chg_30m_max=_tr("late_entry_price_chg_30m_max", 250.0),
        late_entry_pump_ratio_min=_tr("late_entry_pump_ratio_min", 20.0),
        buy_vol_ratio_1h_max=_tr("buy_vol_ratio_1h_max", None),
        market_cap_usd_min=_tr("market_cap_usd_min", None),
        live_trading=_tr("live_trading", False),
        recently_closed_cooldown_minutes=_tr("recently_closed_cooldown_minutes", 30.0),
    )

    _mb_chart = _o("moonbag_managed")
    moonbag_chart_cfg = StrategyConfig(
        name="moonbag_managed",
        buy_size_usd=5.0,
        stop_loss_pct=_mb_chart.get("stop_loss_pct", 0.12),
        take_profit_levels=_tp("moonbag_managed", [
            TakeProfitLevel(multiple=2.0, sell_fraction_original=0.35),
            TakeProfitLevel(multiple=4.3, sell_fraction_original=0.12),
            TakeProfitLevel(multiple=6.8, sell_fraction_original=0.38),
            TakeProfitLevel(multiple=8.0, sell_fraction_original=0.10),
        ]),
        trailing_stop_pct=_mb_chart.get("trailing_stop_pct", 0.30),
        starting_cash_usd=cfg.starting_cash_usd,
        save_chart_data=True,
        use_chart_filter=_mb_chart.get("use_chart_filter", True),
        pump_ratio_max=_mb_chart.get("pump_ratio_max", 3.5),
        ml_training_strategy="infinite_moonbag",
        # position_peak_pnl_pct trains on the highest price reached before sell —
        # "did the signal pump?" not "did our exit lock in profit?".
        # Uses all 64 closed rows (signal_chart_peak_pnl_pct only has 17 with
        # price_tracking_done=1, too few for reliable KNN).
        ml_training_label="position_peak_pnl_pct",
        use_ml_filter=_mb_chart.get("use_ml_filter", False),
        ml_min_score=_mb_chart.get("ml_min_score", 2.0),
        ml_high_score_threshold=_mb_chart.get("ml_high_score_threshold", 6.0),
        ml_max_score_threshold=_mb_chart.get("ml_max_score_threshold", 8.0),
        ml_size_multiplier=_mb_chart.get("ml_size_multiplier", 2.0),
        ml_max_size_multiplier=_mb_chart.get("ml_max_size_multiplier", 3.0),
        ml_k=int(_mb_chart.get("ml_k", 3)),
        ml_halflife_days=_mb_chart.get("ml_halflife_days", 14.0),
        ml_use_subminute=True,       # 15s candles already fetched; top-3 features by Cohen's d are 15s
        ml_score_low_pct=_mb_chart.get("ml_score_low_pct", 0.0),
        ml_score_high_pct=_mb_chart.get("ml_score_high_pct", 200.0),
        # Feature weights for moonbag KNN (27 features total).
        # Weights derived from Cohen's d separability analysis on 65 closed trades.
        #
        # ZEROED (0×) — confirmed useless or actively harmful:
        #   idx 0-5  (15s OHLCV)     — NOW UNLOCKED: top-3 separating features are 15s
        #                               (pump_ratio_15s d=0.36, price_slope_15s d=0.36,
        #                               volatility_15s d=0.31 — all LOSERS > WINNERS).
        #   idx 6    pump_ratio_1m   — d=0.08, LOSERS>WINNERS: near-zero signal.
        #   idx 8    price_slope_1m  — d=0.08, LOSERS>WINNERS: near-zero signal.
        #   idx 10   volatility_1m   — d=0.01: pure noise, winners/losers identical.
        #   idx 11   candle_count_1m — d=0.02: pure noise.
        #   idx 15   buy_vol_ratio_1h— d=0.02: pure noise.
        #
        # KEPT LOW (0.3×) — weak or partially wrong-direction signal:
        #   idx 7    vol_momentum_1m — d=0.14, LOSERS>WINNERS: weak, keep very low.
        #   idx 12   buy_ratio_5m    — d=0.17, LOSERS>WINNERS: 38% fallback rate + wrong dir.
        #
        # MODERATE (1-2×) — useful signals with reasonable d:
        #   idx 9    recent_momentum_1m  — d=0.50, W>L: best 1m feature.
        #   idx 13   activity_5m_norm    — d=0.16, W>L.
        #   idx 14   price_change_5m_norm— d=0.13, W>L.
        #   idx 16   liquidity_change_1h — d=0.18, W>L.
        #
        # HIGH (8×) — dominant separator:
        #   idx 17   source_channel  — d=1.04: every winner was WizzyCasino.
        #
        # TOKEN METADATA (6×/4×/2×) — no historical data yet (100% fallback = 0.5).
        #   Mathematically inert now (all-same value → zero distance contribution
        #   after z-normalisation), but weights are pre-set so they activate
        #   automatically as new signals with real market cap/liquidity data accumulate.
        ml_feature_weights=(
            3.0, 1.0, 3.0, 1.0, 3.0, 1.0,      # idx  0-5:  15s OHLCV — top-3 separators unlocked
            7.6, 1.78, 6.9, 5.1, 0.38, 0.28,   # idx  6-11: 1m OHLCV
            0.0, 0.0, 0.0, 0.0, 0.01, 0.22,    # idx 12-17: pair stats + source_channel
            0.04, 0.02, 0.0,                    # idx 18-20: token metadata
            0.0, 0.12, 0.0, 0.0,               # idx 21-24: wallet (5m) — idx 24 buy_vol_ratio_5m zeroed (hard filter)
            1.5, 4.0,                           # idx 25-26: wallet_momentum_30m, top10_holder_pct
        ),
        # Hard entry filters (data-derived, independent of ML).
        holder_count_max=_mb_chart.get("holder_count_max", 1000),
        late_entry_price_chg_30m_max=_mb_chart.get("late_entry_price_chg_30m_max", 250.0),
        late_entry_pump_ratio_min=_mb_chart.get("late_entry_pump_ratio_min", 20.0),
        live_trading=_mb_chart.get("live_trading", False),
        recently_closed_cooldown_minutes=_mb_chart.get("recently_closed_cooldown_minutes", 30.0),
    )

    _oa = _o("open_ai_managed")
    _oa_base_name = _oa.get("base_strategy", "quick_pop")
    _oa_base = _OPEN_AI_MANAGED_BASES.get(_oa_base_name, _OPEN_AI_MANAGED_BASES["quick_pop"])
    if _oa_base_name not in _OPEN_AI_MANAGED_BASES:
        logger.warning("[registry] Unknown open_ai_managed base_strategy %r — using 'quick_pop'", _oa_base_name)
        _oa_base_name = "quick_pop"
    _oa_mode_name = _oa.get("mode", "balanced")
    _oa_mode_table = _OPEN_AI_MANAGED_MODES.get(_oa_base_name, _OPEN_AI_MANAGED_MODES["quick_pop"])
    _oa_mode = _oa_mode_table.get(_oa_mode_name, _oa_mode_table["balanced"])
    if _oa_mode_name not in _oa_mode_table:
        logger.warning("[registry] Unknown open_ai_managed mode %r for %s — using 'balanced'", _oa_mode_name, _oa_base_name)
        _oa_mode_name = "balanced"
    logger.info("[registry] open_ai_managed base=%r mode=%r", _oa_base_name, _oa_mode_name)

    def _oa_v(key: str, default):
        if key in _oa:
            return _oa[key]
        if key in _oa_mode:
            return _oa_mode[key]
        if key in _oa_base:
            return _oa_base[key]
        return _oa_mode.get(key, default)

    open_ai_managed_cfg = StrategyConfig(
        name="open_ai_managed",
        buy_size_usd=_oa_v("buy_size_usd", 30.0),
        stop_loss_pct=_oa_v("stop_loss_pct", 0.08),
        take_profit_levels=_tp("open_ai_managed", [
            TakeProfitLevel(multiple=1.16, sell_fraction_original=0.60),
            TakeProfitLevel(multiple=2.30, sell_fraction_original=0.40),
        ]),
        trailing_stop_pct=_oa_v("trailing_stop_pct", 0.12),
        starting_cash_usd=cfg.starting_cash_usd,
        timeout_minutes=_oa_v("timeout_minutes", 45.0),
        timeout_min_gain_pct=_oa_v("timeout_min_gain_pct", 0.49),
        max_hold_minutes=_oa_v("max_hold_minutes", None),
        save_chart_data=True,
        use_chart_filter=False,
        use_ml_filter=_oa_v("use_ml_filter", True),
        ml_training_strategy=_oa_v("ml_training_strategy", "quick_pop"),
        ml_training_label=_oa_v("ml_training_label", "position_peak_pnl_pct"),
        ml_use_subminute=_oa_v("ml_use_subminute", True),
        ml_min_score=_oa_v("ml_min_score", 2.25),
        ml_high_score_threshold=_oa_v("ml_high_score_threshold", 5.0),
        ml_max_score_threshold=_oa_v("ml_max_score_threshold", 6.0),
        ml_size_multiplier=_oa_v("ml_size_multiplier", 1.35),
        ml_max_size_multiplier=_oa_v("ml_max_size_multiplier", 1.75),
        ml_k=int(_oa_v("ml_k", 3)),
        ml_halflife_days=_oa_v("ml_halflife_days", 7.0),
        ml_score_low_pct=_oa_v("ml_score_low_pct", -45.0),
        ml_score_high_pct=_oa_v("ml_score_high_pct", 300.0),
        ml_feature_weights=(
            tuple(_oa_v("ml_feature_weights", ()))
            if _oa_v("ml_feature_weights", None) is not None else None
        ),
        ml_wallet_momentum_max=_oa_v("ml_wallet_momentum_max", 2.5),
        holder_count_max=_oa_v("holder_count_max", 1000),
        late_entry_price_chg_30m_max=_oa_v("late_entry_price_chg_30m_max", 250.0),
        late_entry_pump_ratio_min=_oa_v("late_entry_pump_ratio_min", 10.0),
        buy_vol_ratio_1h_max=_oa_v("buy_vol_ratio_1h_max", 0.65),
        market_cap_usd_min=_oa_v("market_cap_usd_min", None),
        peak_drop_exit_pct=_oa_v("peak_drop_exit_pct", 0.12),
        early_timeout_minutes=_oa_v("early_timeout_minutes", 12.0),
        early_timeout_max_gain_pct=_oa_v("early_timeout_max_gain_pct", 0.02),
        early_timeout_min_range_pct=_oa_v("early_timeout_min_range_pct", 0.06),
        live_trading=_oa.get("live_trading", False),
        recently_closed_cooldown_minutes=_oa.get("recently_closed_cooldown_minutes", 30.0),
        use_real_exit_price=_oa_v("use_real_exit_price", False),
        block_new_entries=_oa_v("block_new_entries", False),
    )

    open_ai_runner_cls = InfiniteMoonbagRunner if _oa_base.get("runner") == "moonbag" else StrategyRunner

    return [
        StrategyRunner(cfg=quick_pop_cfg, db=db),
        StrategyRunner(cfg=trend_rider_cfg, db=db),
        # InfiniteMoonbagRunner(cfg=moonbag_cfg, db=db),      # disabled: trend_rider outperforms
        StrategyRunner(cfg=safe_bet_cfg, db=db),
        StrategyRunner(cfg=quick_pop_chart_cfg, db=db),
        open_ai_runner_cls(cfg=open_ai_managed_cfg, db=db),
        StrategyRunner(cfg=trend_rider_chart_cfg, db=db),
        # InfiniteMoonbagRunner(cfg=moonbag_chart_cfg, db=db),  # disabled: trend_rider_managed outperforms
    ]
