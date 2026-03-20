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

_CONTROLLED = frozenset([
    "trend_rider", "trend_rider_managed",
    "infinite_moonbag", "moonbag_managed",
    "quick_pop_managed",
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
        save_chart_data=True,
        live_trading=_o("infinite_moonbag").get("live_trading", False),
    )

    # ------------------------------------------------------------------
    # Group B — chart-filtered mirrors
    # ------------------------------------------------------------------

    _qp_chart = _o("quick_pop_managed")
    quick_pop_chart_cfg = StrategyConfig(
        name="quick_pop_managed",
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
        use_chart_filter=False,   # locked off for quick_pop_managed — AI override agent decides instead
        save_chart_data=True,
        use_ml_filter=True,          # always on for quick_pop_managed — not agent-controlled
        use_policy_agent=True,
        use_ai_override=_qp_chart.get("use_ai_override", False),
        use_ai_override_shadow=_qp_chart.get("use_ai_override_shadow", False),
        ml_training_strategy="quick_pop",  # train on unfiltered base outcomes
        ml_training_label="position_peak_pnl_pct",  # predict peak pump, not exit PnL
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

    _tr_chart = _o("trend_rider_managed")
    trend_rider_chart_cfg = StrategyConfig(
        name="trend_rider_managed",
        buy_size_usd=cfg.buy_size_usd,
        stop_loss_pct=_tr_chart.get("stop_loss_pct", 0.30),
        take_profit_levels=_tp("trend_rider_managed", [
            TakeProfitLevel(multiple=1.8, sell_fraction_original=0.50),
        ]),
        trailing_stop_pct=_tr_chart.get("trailing_stop_pct", 0.30),
        starting_cash_usd=cfg.starting_cash_usd,
        timeout_minutes=_tr_chart.get("timeout_minutes", 90.0),
        timeout_min_gain_pct=_tr_chart.get("timeout_min_gain_pct", 0.15),
        max_hold_minutes=_tr_chart.get("max_hold_minutes", 240.0),
        save_chart_data=True,
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

    _mb_chart = _o("moonbag_managed")
    moonbag_chart_cfg = StrategyConfig(
        name="moonbag_managed",
        buy_size_usd=5.0,
        stop_loss_pct=_mb_chart.get("stop_loss_pct", 0.30),
        take_profit_levels=_tp("moonbag_managed", [
            TakeProfitLevel(multiple=1.8, sell_fraction_original=0.20),
            TakeProfitLevel(multiple=2.5, sell_fraction_original=0.15),
            TakeProfitLevel(multiple=4.0, sell_fraction_original=0.15),
            TakeProfitLevel(multiple=6.0, sell_fraction_original=0.10),
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
        ml_score_low_pct=_mb_chart.get("ml_score_low_pct", -35.0),
        ml_score_high_pct=_mb_chart.get("ml_score_high_pct", 150.0),
        # Feature weights for moonbag KNN (21 features total).
        # Weights derived from Cohen's d separability analysis on 65 closed trades.
        #
        # ZEROED (0×) — confirmed useless or actively harmful:
        #   idx 0-5  (10s OHLCV)     — moonbag passes candles_10s=[] at scoring time;
        #                               always neutral constants, weighting them adds noise.
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
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   # idx  0-5:  10s features — zeroed
            0.0, 0.3, 0.0, 2.0, 0.0, 0.0,   # idx  6-11: 1m OHLCV (pump=0, vol_mom=0.3, slope=0, rec_mom=2, vol=0, cnt=0)
            0.3, 1.0, 1.0, 0.0, 1.0, 8.0,   # idx 12-17: pair stats + source_channel
            6.0, 4.0, 2.0,                   # idx 18-20: market_cap, liquidity, holder_count
        ),
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
