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
    quick_pop_chart, trend_rider_chart (+ reanalyze), infinite_moonbag_chart
"""

from __future__ import annotations

from trader.config import Config
from trader.trading.strategy import (
    InfiniteMoonbagRunner,
    StrategyConfig,
    StrategyRunner,
    TakeProfitLevel,
)


def build_runners(cfg: Config, db=None) -> list[StrategyRunner]:
    """
    Instantiate and return all active strategy runners.

    quick_pop / quick_pop_chart
        Fast scalp: TP1 at 1.5× (sell 60%) → TP2 at 2.0× (sell 40%) — fully exits.
        Trail at 22% below high after TP1.
        Exit after 45 min if TP1 not yet hit (price still < 1.49×).

    trend_rider / trend_rider_chart
        Momentum hold: TP1 at 1.8× (sell 50% of original).
        Trail at 30% below high after TP1.
        Exit after 90 min if price < entry × 1.15. Max hold: 4 hours.
        trend_rider_chart also re-checks skipped signals after a delay.

    infinite_moonbag / infinite_moonbag_chart (v2)
        Grace period 90s: −30% floor. After grace: −22% floor.
        TP ladder: 1.8×/20%, 2.5×/15%, 4.0×/15%, 6.0×/10% of original.
        Stop ladder: 1.8×→1.35×, 2.5×→1.90×, 4.0×→2.80×, 6.0×→3.50×.
    """
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
    )

    trend_rider_cfg = StrategyConfig(
        name="trend_rider",
        buy_size_usd=cfg.buy_size_usd,
        stop_loss_pct=0.30,
        take_profit_levels=(
            TakeProfitLevel(multiple=1.8, sell_fraction_original=0.50),
        ),
        trailing_stop_pct=0.30,
        starting_cash_usd=cfg.starting_cash_usd,
        timeout_minutes=90.0,
        timeout_min_gain_pct=0.15,
        max_hold_minutes=240.0,
    )

    moonbag_cfg = StrategyConfig(
        name="infinite_moonbag",
        buy_size_usd=15.0,
        stop_loss_pct=0.30,
        take_profit_levels=(
            TakeProfitLevel(multiple=1.8, sell_fraction_original=0.20),
            TakeProfitLevel(multiple=2.5, sell_fraction_original=0.15),
            TakeProfitLevel(multiple=4.0, sell_fraction_original=0.15),
            TakeProfitLevel(multiple=6.0, sell_fraction_original=0.10),
        ),
        trailing_stop_pct=0.30,
        starting_cash_usd=cfg.starting_cash_usd,
    )

    # ------------------------------------------------------------------
    # Group B — chart-filtered mirrors
    # ------------------------------------------------------------------

    quick_pop_chart_cfg = StrategyConfig(
        name="quick_pop_chart",
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
        use_chart_filter=True,
        save_chart_data=True,
    )

    trend_rider_chart_cfg = StrategyConfig(
        name="trend_rider_chart",
        buy_size_usd=cfg.buy_size_usd,
        stop_loss_pct=0.30,
        take_profit_levels=(
            TakeProfitLevel(multiple=1.8, sell_fraction_original=0.50),
        ),
        trailing_stop_pct=0.30,
        starting_cash_usd=cfg.starting_cash_usd,
        timeout_minutes=90.0,
        timeout_min_gain_pct=0.15,
        max_hold_minutes=240.0,
        use_chart_filter=True,
        use_reanalyze=True,
    )

    moonbag_chart_cfg = StrategyConfig(
        name="infinite_moonbag_chart",
        buy_size_usd=15.0,
        stop_loss_pct=0.30,
        take_profit_levels=(
            TakeProfitLevel(multiple=1.8, sell_fraction_original=0.20),
            TakeProfitLevel(multiple=2.5, sell_fraction_original=0.15),
            TakeProfitLevel(multiple=4.0, sell_fraction_original=0.15),
            TakeProfitLevel(multiple=6.0, sell_fraction_original=0.10),
        ),
        trailing_stop_pct=0.30,
        starting_cash_usd=cfg.starting_cash_usd,
        use_chart_filter=True,
    )

    return [
        StrategyRunner(cfg=quick_pop_cfg, db=db),
        StrategyRunner(cfg=trend_rider_cfg, db=db),
        InfiniteMoonbagRunner(cfg=moonbag_cfg, db=db),
        StrategyRunner(cfg=quick_pop_chart_cfg, db=db),
        StrategyRunner(cfg=trend_rider_chart_cfg, db=db),
        InfiniteMoonbagRunner(cfg=moonbag_chart_cfg, db=db),
    ]
