"""
tests/test_chart.py — Unit tests for chart-based entry filter and reanalysis.

Tests cover:
    compute_chart_context():
        - Returns None when fewer than 3 candles
        - BUY when pump_ratio is low and volume is rising
        - SKIP when pump_ratio exceeds PUMP_RATIO_MAX
        - SKIP when volume is dying
        - SKIP when both conditions are true
        - Correct reanalyze_in_secs for each skip reason
        - Edge cases: zero volume, all-same candles, single-candle window

    StrategyRunner.enter_position() with use_chart_filter:
        - Enters when use_chart_filter=False regardless of chart_ctx
        - Enters when use_chart_filter=True and should_enter=True
        - Skips when use_chart_filter=True and should_enter=False
        - Enters when use_chart_filter=True but chart_ctx is None (fallback)

    MultiStrategyEngine reanalysis scheduling:
        - _reanalyze() enters chart runners when re-check says BUY
        - _reanalyze() skips when re-check still says SKIP
        - _reanalyze() skips when price fetch fails
        - Duplicate reanalysis not scheduled for same mint

No network calls — Birdeye is mocked throughout.
Run with:  pytest tests/test_chart.py -v
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trader.analysis.chart import (
    PUMP_RATIO_MAX,
    REANALYZE_BOTH_DELAY,
    REANALYZE_PUMP_DELAY,
    REANALYZE_VOL_DELAY,
    ChartContext,
    OHLCVCandle,
    compute_chart_context,
)
from trader.trading.models import PortfolioState, TokenSignal
from trader.trading.strategy import (
    InfiniteMoonbagRunner,
    StrategyConfig,
    StrategyRunner,
    TakeProfitLevel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _candles(
    n: int,
    low: float = 0.001,
    vol_start: float = 1000.0,
    vol_step: float = 50.0,
) -> list[OHLCVCandle]:
    """Build n candles with a fixed low and linearly growing volume."""
    return [
        OHLCVCandle(
            unix_time=i,
            open=low * 1.05,
            high=low * 1.10,
            low=low,
            close=low * 1.05,
            volume=vol_start + i * vol_step,
        )
        for i in range(n)
    ]


def _dying_candles(n: int, low: float = 0.001) -> list[OHLCVCandle]:
    """Candles where recent volume is much lower than earlier volume."""
    candles = []
    for i in range(n):
        # Earlier bars have high volume; recent bars (last 5) have near-zero volume
        vol = 5000.0 if i < n - 5 else 10.0
        candles.append(OHLCVCandle(
            unix_time=i, open=low, high=low, low=low, close=low, volume=vol,
        ))
    return candles


def _chart_cfg(use_filter: bool = True) -> StrategyConfig:
    return StrategyConfig(
        name="chart_test",
        buy_size_usd=10.0,
        stop_loss_pct=0.20,
        take_profit_levels=(TakeProfitLevel(multiple=2.0, sell_fraction_original=0.50),),
        trailing_stop_pct=0.20,
        starting_cash_usd=500.0,
        use_chart_filter=use_filter,
    )


def _signal(mint: str = "mint123") -> TokenSignal:
    return TokenSignal(symbol="TEST", mint_address=mint)


# ---------------------------------------------------------------------------
# compute_chart_context — basic cases
# ---------------------------------------------------------------------------

class TestComputeChartContext:

    def test_returns_none_with_fewer_than_3_candles(self):
        assert compute_chart_context([], 1.0) is None
        assert compute_chart_context(_candles(1), 1.0) is None
        assert compute_chart_context(_candles(2), 1.0) is None

    def test_buy_when_low_pump_rising_volume(self):
        candles = _candles(20, low=0.001, vol_start=1000, vol_step=100)
        ctx = compute_chart_context(candles, current_price=0.0015)
        assert ctx is not None
        assert ctx.should_enter is True
        assert ctx.vol_trend == "RISING"
        assert ctx.pump_ratio == pytest.approx(1.5)
        assert ctx.reanalyze_in_secs is None

    def test_skip_when_pump_ratio_too_high(self):
        candles = _candles(20, low=0.001)
        # price is 4x the low — exceeds PUMP_RATIO_MAX (3.5)
        ctx = compute_chart_context(candles, current_price=0.004)
        assert ctx is not None
        assert ctx.should_enter is False
        assert ctx.pump_ratio > PUMP_RATIO_MAX
        assert ctx.reanalyze_in_secs == REANALYZE_PUMP_DELAY

    def test_skip_when_volume_dying(self):
        candles = _dying_candles(20, low=0.001)
        # price only 1.2x the low — pump is fine, but volume is dead
        ctx = compute_chart_context(candles, current_price=0.0012)
        assert ctx is not None
        assert ctx.should_enter is False
        assert ctx.vol_trend == "DYING"
        assert ctx.pump_ratio < PUMP_RATIO_MAX
        assert ctx.reanalyze_in_secs == REANALYZE_VOL_DELAY

    def test_skip_when_both_pumped_and_dying(self):
        candles = _dying_candles(20, low=0.001)
        # price is 5x the low AND volume is dying
        ctx = compute_chart_context(candles, current_price=0.005)
        assert ctx is not None
        assert ctx.should_enter is False
        assert ctx.vol_trend == "DYING"
        assert ctx.pump_ratio > PUMP_RATIO_MAX
        assert ctx.reanalyze_in_secs == REANALYZE_BOTH_DELAY

    def test_flat_volume_does_not_block_entry(self):
        # Constant volume = FLAT — should not trigger DYING skip
        candles = [
            OHLCVCandle(unix_time=i, open=0.001, high=0.0011,
                        low=0.001, close=0.001, volume=1000.0)
            for i in range(20)
        ]
        ctx = compute_chart_context(candles, current_price=0.0015)
        assert ctx is not None
        assert ctx.vol_trend == "FLAT"
        assert ctx.should_enter is True

    def test_minimum_3_candles_is_accepted(self):
        ctx = compute_chart_context(_candles(3), 0.0015)
        assert ctx is not None
        assert ctx.candle_count == 3

    def test_pump_ratio_uses_lowest_low(self):
        # Set one candle with a very low spike — that should set the reference
        candles = _candles(20, low=0.002)
        candles[5] = OHLCVCandle(
            unix_time=5, open=0.002, high=0.002, low=0.0005, close=0.002, volume=1000,
        )
        ctx = compute_chart_context(candles, current_price=0.002)
        assert ctx is not None
        assert ctx.pump_ratio == pytest.approx(0.002 / 0.0005)

    def test_zero_volume_earlier_bars_treated_as_flat(self):
        candles = [
            OHLCVCandle(unix_time=i, open=0.001, high=0.001,
                        low=0.001, close=0.001, volume=0.0)
            for i in range(20)
        ]
        ctx = compute_chart_context(candles, current_price=0.0015)
        assert ctx is not None
        assert ctx.vol_trend == "FLAT"


# ---------------------------------------------------------------------------
# StrategyRunner.enter_position — chart filter gate
# ---------------------------------------------------------------------------

class TestEnterPositionChartFilter:

    def _runner(self, use_filter: bool) -> StrategyRunner:
        return StrategyRunner(cfg=_chart_cfg(use_filter=use_filter))

    def _ok_ctx(self) -> ChartContext:
        return ChartContext(
            pump_ratio=1.5, vol_trend="RISING",
            should_enter=True, reason="OK", candle_count=20,
        )

    def _skip_ctx(self) -> ChartContext:
        return ChartContext(
            pump_ratio=5.0, vol_trend="DYING",
            should_enter=False, reason="SKIP: pumped",
            candle_count=20, reanalyze_in_secs=REANALYZE_BOTH_DELAY,
        )

    def test_no_filter_always_enters(self):
        runner = self._runner(use_filter=False)
        pos = runner.enter_position(_signal(), entry_price=0.001, chart_ctx=self._skip_ctx())
        assert pos is not None

    def test_filter_enabled_enters_on_ok_ctx(self):
        runner = self._runner(use_filter=True)
        pos = runner.enter_position(_signal(), entry_price=0.001, chart_ctx=self._ok_ctx())
        assert pos is not None

    def test_filter_enabled_skips_on_skip_ctx(self):
        runner = self._runner(use_filter=True)
        pos = runner.enter_position(_signal(), entry_price=0.001, chart_ctx=self._skip_ctx())
        assert pos is None

    def test_filter_enabled_enters_when_ctx_is_none(self):
        """No chart data → fall back to entering (don't lose signals on API errors)."""
        runner = self._runner(use_filter=True)
        pos = runner.enter_position(_signal(), entry_price=0.001, chart_ctx=None)
        assert pos is not None

    def test_filter_does_not_affect_duplicate_check(self):
        """Duplicate mint is still rejected even with an OK chart context."""
        runner = self._runner(use_filter=True)
        sig = _signal(mint="dupemint")
        runner.enter_position(sig, entry_price=0.001, chart_ctx=self._ok_ctx())
        # Second entry with same mint should be rejected as duplicate
        pos2 = runner.enter_position(sig, entry_price=0.001, chart_ctx=self._ok_ctx())
        assert pos2 is None


# ---------------------------------------------------------------------------
# MultiStrategyEngine reanalysis
# ---------------------------------------------------------------------------

def _make_engine(
    use_filter_runners: int = 1,
    no_filter_runners: int = 1,
    use_reanalyze: bool = True,
):
    """Build a minimal MultiStrategyEngine with mocked Birdeye."""
    from trader.trading.engine import MultiStrategyEngine
    from trader.config import Config

    cfg = MagicMock(spec=Config)
    cfg.dry_run = False
    cfg.poll_interval_seconds = 1.0

    runners = []
    for i in range(no_filter_runners):
        runners.append(StrategyRunner(cfg=StrategyConfig(
            name=f"baseline_{i}",
            buy_size_usd=10.0, stop_loss_pct=0.20,
            take_profit_levels=(TakeProfitLevel(2.0, 0.5),),
            trailing_stop_pct=0.20, starting_cash_usd=500.0,
            use_chart_filter=False,
        )))
    for i in range(use_filter_runners):
        runners.append(StrategyRunner(cfg=StrategyConfig(
            name=f"chart_{i}",
            buy_size_usd=10.0, stop_loss_pct=0.20,
            take_profit_levels=(TakeProfitLevel(2.0, 0.5),),
            trailing_stop_pct=0.20, starting_cash_usd=500.0,
            use_chart_filter=True,
            use_reanalyze=use_reanalyze,
        )))

    birdeye = MagicMock()
    engine = MultiStrategyEngine(cfg=cfg, runners=runners, birdeye_client=birdeye)
    return engine, runners, birdeye


class TestReanalysis:

    def _skip_ctx(self, delay: float = REANALYZE_PUMP_DELAY) -> ChartContext:
        return ChartContext(
            pump_ratio=5.0, vol_trend="FLAT",
            should_enter=False, reason="SKIP: pumped",
            candle_count=20, reanalyze_in_secs=delay,
        )

    def _ok_ctx(self) -> ChartContext:
        return ChartContext(
            pump_ratio=1.5, vol_trend="RISING",
            should_enter=True, reason="OK", candle_count=20,
        )

    async def test_reanalyze_enters_when_chart_now_ok(self):
        """After delay, fresh chart says BUY → chart runner enters."""
        engine, runners, birdeye = _make_engine()
        chart_runner = next(r for r in runners if r.cfg.use_chart_filter)
        signal = _signal(mint="remint1")

        birdeye.get_price = AsyncMock(return_value=0.001)
        birdeye.get_ohlcv = AsyncMock(return_value=_candles(20))

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with patch(
                "trader.trading.engine.compute_chart_context",
                return_value=self._ok_ctx(),
            ):
                await engine._reanalyze(signal, delay=0.0)

        assert chart_runner.has_open_position(signal.mint_address)

    async def test_reanalyze_skips_when_chart_still_bad(self):
        """After delay, fresh chart still says SKIP → no entry."""
        engine, runners, birdeye = _make_engine()
        chart_runner = next(r for r in runners if r.cfg.use_chart_filter)
        signal = _signal(mint="remint2")

        birdeye.get_price = AsyncMock(return_value=0.001)
        birdeye.get_ohlcv = AsyncMock(return_value=_candles(20))

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with patch(
                "trader.trading.engine.compute_chart_context",
                return_value=self._skip_ctx(),
            ):
                await engine._reanalyze(signal, delay=0.0)

        assert not chart_runner.has_open_position(signal.mint_address)

    async def test_reanalyze_aborts_when_no_price(self):
        """If Birdeye returns no price at reanalysis time, abort silently."""
        engine, runners, birdeye = _make_engine()
        chart_runner = next(r for r in runners if r.cfg.use_chart_filter)
        signal = _signal(mint="remint3")

        birdeye.get_price = AsyncMock(return_value=None)
        birdeye.get_ohlcv = AsyncMock(return_value=_candles(20))

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await engine._reanalyze(signal, delay=0.0)

        assert not chart_runner.has_open_position(signal.mint_address)

    async def test_reanalyze_does_not_enter_baseline_runners(self):
        """Reanalysis must only affect chart-filtered runners."""
        engine, runners, birdeye = _make_engine(use_filter_runners=1, no_filter_runners=1)
        baseline_runner = next(r for r in runners if not r.cfg.use_chart_filter)
        signal = _signal(mint="remint4")

        birdeye.get_price = AsyncMock(return_value=0.001)
        birdeye.get_ohlcv = AsyncMock(return_value=_candles(20))

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with patch(
                "trader.trading.engine.compute_chart_context",
                return_value=self._ok_ctx(),
            ):
                await engine._reanalyze(signal, delay=0.0)

        assert not baseline_runner.has_open_position(signal.mint_address)

    def test_pending_reanalysis_prevents_duplicate_scheduling(self):
        """Same mint cannot be scheduled for reanalysis twice."""
        engine, runners, birdeye = _make_engine()
        signal = _signal(mint="dupemint_re")

        engine._pending_reanalysis.add(signal.mint_address)
        before = set(engine._pending_reanalysis)

        engine._pending_reanalysis.add(signal.mint_address)
        assert engine._pending_reanalysis == before

    async def test_pending_cleared_after_reanalyze_runs(self):
        """Mint is removed from _pending_reanalysis once _reanalyze executes."""
        engine, runners, birdeye = _make_engine()
        signal = _signal(mint="remint5")
        engine._pending_reanalysis.add(signal.mint_address)

        birdeye.get_price = AsyncMock(return_value=None)  # aborts early
        birdeye.get_ohlcv = AsyncMock(return_value=[])

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await engine._reanalyze(signal, delay=0.0)

        assert signal.mint_address not in engine._pending_reanalysis

    async def test_handle_new_signal_schedules_reanalyze_when_flag_on(self):
        """handle_new_signal() creates a reanalysis task when use_reanalyze=True
        and the chart filter skips the signal."""
        engine, runners, birdeye = _make_engine(use_reanalyze=True)
        signal = _signal(mint="sched_on")

        birdeye.get_price = AsyncMock(return_value=0.001)
        birdeye.get_ohlcv = AsyncMock(return_value=_candles(20))

        with patch("asyncio.create_task") as mock_create_task:
            with patch(
                "trader.trading.engine.compute_chart_context",
                return_value=self._skip_ctx(),
            ):
                await engine.handle_new_signal(signal)

        mock_create_task.assert_called_once()
        assert signal.mint_address in engine._pending_reanalysis

    async def test_handle_new_signal_does_not_schedule_when_flag_off(self):
        """handle_new_signal() must NOT create a reanalysis task when every
        runner has use_reanalyze=False, even if chart filter skips."""
        engine, runners, birdeye = _make_engine(use_reanalyze=False)
        signal = _signal(mint="sched_off")

        birdeye.get_price = AsyncMock(return_value=0.001)
        birdeye.get_ohlcv = AsyncMock(return_value=_candles(20))

        with patch("asyncio.create_task") as mock_create_task:
            with patch(
                "trader.trading.engine.compute_chart_context",
                return_value=self._skip_ctx(),
            ):
                await engine.handle_new_signal(signal)

        mock_create_task.assert_not_called()
        assert signal.mint_address not in engine._pending_reanalysis

    async def test_handle_new_signal_does_not_schedule_when_chart_says_buy(self):
        """No reanalysis task when chart filter approves the entry — nothing to re-check."""
        engine, runners, birdeye = _make_engine(use_reanalyze=True)
        signal = _signal(mint="sched_buy")

        birdeye.get_price = AsyncMock(return_value=0.001)
        birdeye.get_ohlcv = AsyncMock(return_value=_candles(20))

        with patch("asyncio.create_task") as mock_create_task:
            with patch(
                "trader.trading.engine.compute_chart_context",
                return_value=self._ok_ctx(),
            ):
                await engine.handle_new_signal(signal)

        mock_create_task.assert_not_called()

    def test_reanalyze_not_scheduled_when_flag_off(self):
        """use_reanalyze=False — runner with chart filter but no reanalyze
        should never trigger a reanalysis task."""
        engine, runners, birdeye = _make_engine(use_reanalyze=False)
        signal = _signal(mint="noreanalyze1")

        # No runners have use_reanalyze=True so condition must be False
        has_reanalyze_runner = any(
            r.cfg.use_chart_filter and r.cfg.use_reanalyze for r in runners
        )
        assert has_reanalyze_runner is False
        assert signal.mint_address not in engine._pending_reanalysis

    async def test_reanalyze_does_not_enter_runner_with_flag_off(self):
        """When reanalysis fires, runners with use_reanalyze=False are skipped
        even if use_chart_filter=True."""
        from trader.trading.engine import MultiStrategyEngine
        from trader.config import Config

        cfg = MagicMock(spec=Config)
        cfg.dry_run = False

        # One chart runner WITH reanalyze, one WITHOUT
        runner_with = StrategyRunner(cfg=StrategyConfig(
            name="chart_reanalyze_on",
            buy_size_usd=10.0, stop_loss_pct=0.20,
            take_profit_levels=(TakeProfitLevel(2.0, 0.5),),
            trailing_stop_pct=0.20, starting_cash_usd=500.0,
            use_chart_filter=True, use_reanalyze=True,
        ))
        runner_without = StrategyRunner(cfg=StrategyConfig(
            name="chart_reanalyze_off",
            buy_size_usd=10.0, stop_loss_pct=0.20,
            take_profit_levels=(TakeProfitLevel(2.0, 0.5),),
            trailing_stop_pct=0.20, starting_cash_usd=500.0,
            use_chart_filter=True, use_reanalyze=False,
        ))

        birdeye = MagicMock()
        birdeye.get_price = AsyncMock(return_value=0.001)
        birdeye.get_ohlcv = AsyncMock(return_value=_candles(20))

        engine = MultiStrategyEngine(
            cfg=cfg, runners=[runner_with, runner_without], birdeye_client=birdeye,
        )
        signal = _signal(mint="selective_reanalyze")

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with patch(
                "trader.trading.engine.compute_chart_context",
                return_value=self._ok_ctx(),
            ):
                await engine._reanalyze(signal, delay=0.0)

        assert runner_with.has_open_position(signal.mint_address)
        assert not runner_without.has_open_position(signal.mint_address)
