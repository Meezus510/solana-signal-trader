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
    quick_pop, trend_rider, infinite_moonbag

Group B — chart filter enabled (skips late pumps and dying volume):
    quick_pop_chart_ml (+ ML filter), trend_rider_chart_reanalyze (+ reanalyze), infinite_moonbag_chart
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

_CONTROLLED = frozenset([
    "trend_rider", "trend_rider_chart_reanalyze",
    "infinite_moonbag", "infinite_moonbag_chart",
    "quick_pop_chart_ml",
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

    quick_pop / quick_pop_chart_ml
        Fast scalp: TP1 at 1.5× (sell 60%) → TP2 at 2.0× (sell 40%) — fully exits.
        Trail at 22% below high after TP1.
        Exit after 45 min if TP1 not yet hit (price still < 1.49×).
        quick_pop_chart_ml also applies chart filter + ML confidence gating (score < 5
        skips, score ≥ 8 doubles size, score ≥ 9.5 triples size).

    trend_rider / trend_rider_chart_reanalyze
        Momentum hold: TP1 at 1.8× (sell 50% of original).
        Trail at 30% below high after TP1.
        Exit after 90 min if price < entry × 1.15. Max hold: 4 hours.
        trend_rider_chart_reanalyze also re-checks skipped signals after a delay.

    infinite_moonbag / infinite_moonbag_chart (v2)
        Grace period 90s: −30% floor. After grace: −22% floor.
        TP ladder: 1.8×/20%, 2.5×/15%, 4.0×/15%, 6.0×/10% of original.
        Stop ladder: 1.8×→1.35×, 2.5×→1.90×, 4.0×→2.80×, 6.0×→3.50×.
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
        stop_loss_pct=0.20,
        take_profit_levels=(
            TakeProfitLevel(multiple=1.5, sell_fraction_original=0.60),
            TakeProfitLevel(multiple=2.0, sell_fraction_original=0.40),
        ),
        trailing_stop_pct=0.22,
        starting_cash_usd=cfg.starting_cash_usd,
        timeout_minutes=45.0,
        timeout_min_gain_pct=0.49,
        save_chart_data=True,
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
        live_trading=_o("trend_rider").get("live_trading", False),
        save_chart_data=True,
    )

    moonbag_cfg = StrategyConfig(
        name="infinite_moonbag",
        buy_size_usd=5.0,
        stop_loss_pct=_o("infinite_moonbag").get("stop_loss_pct", 0.30),
        take_profit_levels=_tp("infinite_moonbag", [
            TakeProfitLevel(multiple=1.8, sell_fraction_original=0.20),
            TakeProfitLevel(multiple=2.5, sell_fraction_original=0.15),
            TakeProfitLevel(multiple=4.0, sell_fraction_original=0.15),
            TakeProfitLevel(multiple=6.0, sell_fraction_original=0.10),
        ]),
        trailing_stop_pct=_o("infinite_moonbag").get("trailing_stop_pct", 0.30),
        starting_cash_usd=cfg.starting_cash_usd,
        live_trading=_o("infinite_moonbag").get("live_trading", False),
    )

    # ------------------------------------------------------------------
    # Group B — chart-filtered mirrors
    # ------------------------------------------------------------------

    _qp_chart = _o("quick_pop_chart_ml")
    quick_pop_chart_cfg = StrategyConfig(
        name="quick_pop_chart_ml",
        buy_size_usd=30.0,
        stop_loss_pct=0.20,       # fixed — not agent-controlled
        take_profit_levels=(      # fixed — not agent-controlled
            TakeProfitLevel(multiple=1.5, sell_fraction_original=0.60),
            TakeProfitLevel(multiple=2.0, sell_fraction_original=0.40),
        ),
        trailing_stop_pct=0.22,   # fixed — not agent-controlled
        starting_cash_usd=cfg.starting_cash_usd,
        timeout_minutes=45.0,     # fixed — not agent-controlled
        timeout_min_gain_pct=0.49,  # fixed — not agent-controlled
        use_chart_filter=False,   # locked off for quick_pop_chart_ml — AI override agent decides instead
        save_chart_data=True,
        use_ml_filter=True,          # always on for quick_pop_chart_ml — not agent-controlled
        use_policy_agent=True,
        use_ai_override=_qp_chart.get("use_ai_override", False),
        use_ai_override_shadow=_qp_chart.get("use_ai_override_shadow", False),
        ml_training_strategy="quick_pop",  # train on unfiltered base outcomes
        ml_prefer_moralis=True,      # use 10s candles for KNN — fast-scalp pump shape
        ml_min_score=_qp_chart.get("ml_min_score", 5.0),
        ml_high_score_threshold=_qp_chart.get("ml_high_score_threshold", 8.0),
        ml_max_score_threshold=_qp_chart.get("ml_max_score_threshold", 9.5),
        ml_size_multiplier=_qp_chart.get("ml_size_multiplier", 2.0),
        ml_max_size_multiplier=_qp_chart.get("ml_max_size_multiplier", 3.0),
        ml_k=int(_qp_chart.get("ml_k", 5)),
        ml_halflife_days=_qp_chart.get("ml_halflife_days", 14.0),
        ml_score_low_pct=_qp_chart.get("ml_score_low_pct", -35.0),
        ml_score_high_pct=_qp_chart.get("ml_score_high_pct", 85.0),
        live_trading=_qp_chart.get("live_trading", False),
    )

    _tr_chart = _o("trend_rider_chart_reanalyze")
    trend_rider_chart_cfg = StrategyConfig(
        name="trend_rider_chart_reanalyze",
        buy_size_usd=cfg.buy_size_usd,
        stop_loss_pct=_tr_chart.get("stop_loss_pct", 0.30),
        take_profit_levels=_tp("trend_rider_chart_reanalyze", [
            TakeProfitLevel(multiple=1.8, sell_fraction_original=0.50),
        ]),
        trailing_stop_pct=_tr_chart.get("trailing_stop_pct", 0.30),
        starting_cash_usd=cfg.starting_cash_usd,
        timeout_minutes=_tr_chart.get("timeout_minutes", 90.0),
        timeout_min_gain_pct=_tr_chart.get("timeout_min_gain_pct", 0.15),
        max_hold_minutes=_tr_chart.get("max_hold_minutes", 240.0),
        use_chart_filter=_tr_chart.get("use_chart_filter", True),
        pump_ratio_max=_tr_chart.get("pump_ratio_max", 3.5),
        use_reanalyze=_tr_chart.get("use_reanalyze", True),
        reanalyze_pump_delay=_tr_chart.get("reanalyze_pump_delay", 480.0),
        reanalyze_vol_delay=_tr_chart.get("reanalyze_vol_delay", 240.0),
        reanalyze_both_delay=_tr_chart.get("reanalyze_both_delay", 600.0),
        ml_training_strategy="trend_rider",
        use_ml_filter=_tr_chart.get("use_ml_filter", False),
        ml_min_score=_tr_chart.get("ml_min_score", 5.0),
        ml_high_score_threshold=_tr_chart.get("ml_high_score_threshold", 8.0),
        ml_max_score_threshold=_tr_chart.get("ml_max_score_threshold", 9.5),
        ml_size_multiplier=_tr_chart.get("ml_size_multiplier", 2.0),
        ml_max_size_multiplier=_tr_chart.get("ml_max_size_multiplier", 3.0),
        ml_k=int(_tr_chart.get("ml_k", 5)),
        ml_halflife_days=_tr_chart.get("ml_halflife_days", 14.0),
        ml_score_low_pct=_tr_chart.get("ml_score_low_pct", -35.0),
        ml_score_high_pct=_tr_chart.get("ml_score_high_pct", 85.0),
        live_trading=_tr_chart.get("live_trading", False),
    )

    _mb_chart = _o("infinite_moonbag_chart")
    moonbag_chart_cfg = StrategyConfig(
        name="infinite_moonbag_chart",
        buy_size_usd=5.0,
        stop_loss_pct=_mb_chart.get("stop_loss_pct", 0.30),
        take_profit_levels=_tp("infinite_moonbag_chart", [
            TakeProfitLevel(multiple=1.8, sell_fraction_original=0.20),
            TakeProfitLevel(multiple=2.5, sell_fraction_original=0.15),
            TakeProfitLevel(multiple=4.0, sell_fraction_original=0.15),
            TakeProfitLevel(multiple=6.0, sell_fraction_original=0.10),
        ]),
        trailing_stop_pct=_mb_chart.get("trailing_stop_pct", 0.30),
        starting_cash_usd=cfg.starting_cash_usd,
        use_chart_filter=_mb_chart.get("use_chart_filter", True),
        pump_ratio_max=_mb_chart.get("pump_ratio_max", 3.5),
        ml_training_strategy="infinite_moonbag",
        use_ml_filter=_mb_chart.get("use_ml_filter", False),
        ml_min_score=_mb_chart.get("ml_min_score", 5.0),
        ml_high_score_threshold=_mb_chart.get("ml_high_score_threshold", 8.0),
        ml_max_score_threshold=_mb_chart.get("ml_max_score_threshold", 9.5),
        ml_size_multiplier=_mb_chart.get("ml_size_multiplier", 2.0),
        ml_max_size_multiplier=_mb_chart.get("ml_max_size_multiplier", 3.0),
        ml_k=int(_mb_chart.get("ml_k", 5)),
        ml_halflife_days=_mb_chart.get("ml_halflife_days", 14.0),
        ml_score_low_pct=_mb_chart.get("ml_score_low_pct", -35.0),
        ml_score_high_pct=_mb_chart.get("ml_score_high_pct", 85.0),
        live_trading=_mb_chart.get("live_trading", False),
    )

    return [
        StrategyRunner(cfg=quick_pop_cfg, db=db),
        StrategyRunner(cfg=trend_rider_cfg, db=db),
        InfiniteMoonbagRunner(cfg=moonbag_cfg, db=db),
        StrategyRunner(cfg=quick_pop_chart_cfg, db=db),
        StrategyRunner(cfg=trend_rider_chart_cfg, db=db),
        InfiniteMoonbagRunner(cfg=moonbag_chart_cfg, db=db),
    ]
