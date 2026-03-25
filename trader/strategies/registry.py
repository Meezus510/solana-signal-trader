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

_CONTROLLED = frozenset([
    "trend_rider", "trend_rider_managed",
    "infinite_moonbag", "moonbag_managed",
    "quick_pop_managed",
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
        save_chart_data=True,
        live_trading=_o("safe_bet").get("live_trading", False),
        use_real_exit_price=True,
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
            0.0, 0.0, 0.0, 3.0,                      # idx 21-24: wallet (5m)
            0.0, 1.0,                                 # idx 25-26: wallet_momentum_30m, top10_holder_pct
            0.0, 0.0, 0.0, 4.0, 0.0, 0.0,            # idx 27-32: 1s OHLCV
            0.0, 1.0, 2.0, 0.0, 2.5, 0.5, 1.5, 0.0, 0.0, 0.0,  # idx 33-42: 15s shape
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # idx 43-52: 1m shape
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.5, 0.0, 0.0,  # idx 53-62: 1s shape
        ),
        # Hard pre-filter: block signals with wallet_momentum_5m >= 2.102.
        # LOO backtest on 197 trades: blocks 9/134 losers with 0 winner misses.
        ml_wallet_momentum_max=2.102,
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
        # Feature weights for trend_rider KNN (63 features total).
        # trend_rider is ml_use_subminute=False → candles_15s=[] and candles_1s=None at inference.
        # Locked to 0: idx 0-5 (15s OHLCV), 27-42 (1s OHLCV + 15s shape), 53-62 (1s shape).
        # To update: run scripts/optimize_ml_weights.py --strategy trend_rider
        ml_feature_weights=(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,            # idx  0-5:  15s OHLCV — locked (not fetched at inference)
            3.2, 0.0, 3.2, 0.0, 2.0, 0.0,            # idx  6-11: 1m OHLCV
            0.0, 0.0, 0.0, 0.0, 3.2, 2.2,            # idx 12-17: pair stats + source_channel
            0.0, 2.2, 0.0,                            # idx 18-20: token metadata
            0.0, 0.1, 0.0, 1.8,                      # idx 21-24: wallet (5m)
            2.0, 1.5,                                 # idx 25-26: wallet_momentum_30m, top10_holder_pct
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,            # idx 27-32: 1s OHLCV — locked
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # idx 33-42: 15s shape — locked
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # idx 43-52: 1m shape
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # idx 53-62: 1s shape — locked
        ),
        live_trading=_tr_chart.get("live_trading", False),
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
            0.0, 0.12, 0.0, 6.9,               # idx 21-24: wallet (5m)
            1.5, 4.0,                           # idx 25-26: wallet_momentum_30m, top10_holder_pct
        ),
        live_trading=_mb_chart.get("live_trading", False),
    )

    return [
        StrategyRunner(cfg=quick_pop_cfg, db=db),
        StrategyRunner(cfg=trend_rider_cfg, db=db),
        InfiniteMoonbagRunner(cfg=moonbag_cfg, db=db),
        StrategyRunner(cfg=safe_bet_cfg, db=db),
        StrategyRunner(cfg=quick_pop_chart_cfg, db=db),
        StrategyRunner(cfg=trend_rider_chart_cfg, db=db),
        InfiniteMoonbagRunner(cfg=moonbag_chart_cfg, db=db),
    ]
