"""
trader/trading/engine.py — Multi-strategy coordinator.

MultiStrategyEngine wires together N StrategyRunners and a shared Birdeye
price feed. Its only responsibilities are:

    1. handle_new_signal()   — fetch entry price once, fan buy to all runners
    2. monitor_positions()   — collect unique mints across all runners,
                               fetch prices once, distribute to each runner
    3. print_summary()       — per-strategy summaries + global aggregate

All strategy evaluation logic lives in StrategyRunner (strategy.py) so
this class stays free of trading rules and easy to extend.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from trader.analysis.chart import OHLCV_BARS, compute_chart_context
from trader.analysis.ml_scorer import (
    ChartMLScorer, ML_OHLCV_BARS, ML_OHLCV_INTERVAL,
    MORALIS_OHLCV_BARS, MORALIS_OHLCV_INTERVAL,
)
from trader.config import Config
from trader.pricing.birdeye import BirdeyePriceClient
from trader.pricing.moralis import MoralisOHLCVClient
from trader.trading.models import TokenSignal
from trader.trading.strategy import StrategyRunner

logger = logging.getLogger(__name__)
signal_log = logging.getLogger("signals")


class MultiStrategyEngine:
    """
    Coordinates N StrategyRunners with a single shared Birdeye price feed.

    Shared price polling:
        Each poll cycle collects every unique mint across all runners'
        open positions, fetches prices in one batch call, then fans each
        price out to every runner that holds that mint. A mint held by
        3 strategies still costs exactly 1 Birdeye API call per cycle.

    Independent exit logic:
        Each runner evaluates its own stop-loss / take-profit / trailing
        stop independently. One strategy closing a position does not
        affect any other strategy holding the same mint.
    """

    def __init__(
        self,
        cfg: Config,
        runners: list[StrategyRunner],
        birdeye_client: BirdeyePriceClient,
        db=None,
        moralis_client: MoralisOHLCVClient | None = None,
    ) -> None:
        self._cfg = cfg
        self._runners = runners
        self._birdeye = birdeye_client
        self._moralis = moralis_client
        self._db = db
        # Mints currently awaiting reanalysis — prevents duplicate scheduling
        self._pending_reanalysis: set[str] = set()
        # ML scorer — only instantiated when at least one runner saves chart data
        self._ml_scorer: ChartMLScorer | None = (
            ChartMLScorer(db) if db and any(r.cfg.save_chart_data for r in runners) else None
        )

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    async def handle_new_signal(self, signal: TokenSignal) -> None:
        """
        Process a new token signal across all strategies.

        Entry price is fetched once and shared — N strategies entering
        the same token costs 1 Birdeye call, not N calls.
        """
        logger.info("[SIGNAL] %s | mint=%s | channel=%s", signal.symbol, signal.mint_address, signal.source_channel)
        signal_log.info("SIGNAL     | %-10s | %-44s | ch=%-20s", signal.symbol, signal.mint_address, signal.source_channel)
        if self._db:
            self._db.log_signal("SIGNAL", symbol=signal.symbol, mint=signal.mint_address)

        if self._cfg.dry_run:
            logger.info("[DRY_RUN] %s — skipping Birdeye and entry", signal.symbol)
            return

        entry_price = await self._birdeye.get_price(signal.mint_address)
        if entry_price is None:
            logger.warning("[SKIP] No live price for %s — entry aborted", signal.symbol)
            signal_log.info("NO_PRICE   | %-10s | %-44s | ch=%-20s", signal.symbol, signal.mint_address, signal.source_channel)
            if self._db:
                self._db.log_signal("NO_PRICE", symbol=signal.symbol, mint=signal.mint_address)
            return

        logger.info(
            "[ENTRY] %s @ $%.8f — distributing to %d strategy(s)",
            signal.symbol, entry_price, len(self._runners),
        )

        # Fetch 1m Birdeye candles if any runner uses the chart filter OR saves
        # chart data (stored as candles_1m_json for future strategy use).
        candles = []
        chart_ctx = None
        needs_1m = any(r.cfg.use_chart_filter or r.cfg.save_chart_data for r in self._runners)
        if needs_1m:
            candles = await self._birdeye.get_ohlcv(signal.mint_address, bars=OHLCV_BARS)
            chart_ctx = compute_chart_context(candles, entry_price)
            if chart_ctx:
                logger.info(
                    "[CHART] %s | pump=%.1fx | vol=%s | enter=%s | %s",
                    signal.symbol, chart_ctx.pump_ratio, chart_ctx.vol_trend,
                    chart_ctx.should_enter, chart_ctx.reason,
                )
            else:
                logger.info(
                    "[CHART] %s | insufficient candle data — chart filter bypassed",
                    signal.symbol,
                )

        # ------------------------------------------------------------------
        # ML candles — try Moralis (high-res) first, fall back to Birdeye.
        # Any Moralis failure is caught silently; Birdeye is always the safety net.
        # The interval/bars are controlled by MORALIS_OHLCV_INTERVAL /
        # MORALIS_OHLCV_BARS in ml_scorer.py — change those constants to switch
        # to any supported resolution (10s, 30s, 1min, etc.).
        # ------------------------------------------------------------------
        ml_candles = []
        ml_source  = "none"
        if self._ml_scorer:
            if self._moralis:
                try:
                    ml_candles = await self._moralis.get_ohlcv(
                        signal.mint_address,
                        bars=MORALIS_OHLCV_BARS,
                        interval=MORALIS_OHLCV_INTERVAL,
                    )
                    if ml_candles:
                        ml_source = f"moralis/{MORALIS_OHLCV_INTERVAL}"
                except Exception:
                    logger.debug("[ML] Moralis OHLCV failed for %s — falling back", signal.symbol)

            if not ml_candles:
                try:
                    ml_candles = await self._birdeye.get_ohlcv(
                        signal.mint_address,
                        bars=ML_OHLCV_BARS,
                        interval=ML_OHLCV_INTERVAL,
                    )
                    if ml_candles:
                        ml_source = f"birdeye/{ML_OHLCV_INTERVAL}"
                except Exception:
                    logger.debug("[ML] Birdeye OHLCV failed for %s", signal.symbol)

        # Moralis pair stats — completely optional, never blocks scoring.
        pair_stats = None
        if self._moralis:
            try:
                pair_stats = await self._moralis.get_pair_stats(signal.mint_address)
            except Exception:
                logger.debug("[ML] Moralis pair stats failed for %s — skipping", signal.symbol)

        # ML confidence score
        ml_score: float | None = None
        if self._ml_scorer and ml_candles:
            ml_score = self._ml_scorer.score(ml_candles, chart_ctx=chart_ctx, pair_stats=pair_stats)
            if ml_score is not None:
                logger.info(
                    "[ML] %s | confidence=%.1f/10 | src=%s | pair_stats=%s",
                    signal.symbol, ml_score, ml_source, "yes" if pair_stats else "no",
                )

        for runner in self._runners:
            # ML filter — skip if score is below threshold.
            # If score is None (not enough training data yet), always allow through.
            ml_blocked = (
                runner.cfg.use_ml_filter
                and ml_score is not None
                and ml_score < runner.cfg.ml_min_score
            )
            if ml_blocked:
                logger.info(
                    "[ML_SKIP] [%-12s] %s | score=%.1f < threshold=%.1f",
                    runner.name, signal.symbol, ml_score, runner.cfg.ml_min_score,
                )
                signal_log.info(
                    "ML_SKIP    | %-10s | %-44s | score=%.1f",
                    signal.symbol, signal.mint_address, ml_score,
                )

            position = None
            if not ml_blocked:
                try:
                    position = runner.enter_position(signal, entry_price, chart_ctx)
                except Exception:
                    logger.exception(
                        "[ERROR] %s: enter_position failed for %s", runner.name, signal.symbol
                    )
                    continue

            # Save snapshot for ML-filter runners even when blocked — these become
            # labeled training examples (entered=False) once their outcome is known.
            if runner.cfg.save_chart_data and ml_candles and self._db:
                snapshot_id = self._db.save_chart_snapshot(
                    strategy=runner.name,
                    symbol=signal.symbol,
                    mint=signal.mint_address,
                    entry_price=entry_price,
                    candles=ml_candles,
                    chart_ctx=chart_ctx,
                    entered=position is not None,
                    ml_score=ml_score,
                    pair_stats=pair_stats,
                    candles_1m=candles if candles else None,
                )
                if position is not None:
                    runner.set_chart_snapshot_id(signal.mint_address, snapshot_id)

        # Schedule reanalysis if chart filter skipped this signal, at least one
        # runner has reanalysis enabled, and this mint isn't already pending.
        if (
            chart_ctx is not None
            and not chart_ctx.should_enter
            and chart_ctx.reanalyze_in_secs is not None
            and signal.mint_address not in self._pending_reanalysis
            and any(r.cfg.use_chart_filter and r.cfg.use_reanalyze for r in self._runners)
        ):
            self._pending_reanalysis.add(signal.mint_address)
            delay = chart_ctx.reanalyze_in_secs
            logger.info(
                "[REANALYZE] %s — scheduled in %.0fs (%.1f min) | reason: %s",
                signal.symbol, delay, delay / 60, chart_ctx.reason,
            )
            asyncio.create_task(self._reanalyze(signal, delay))

    # ------------------------------------------------------------------
    # Reanalysis
    # ------------------------------------------------------------------

    async def _reanalyze(self, signal: TokenSignal, delay: float) -> None:
        """
        Wait `delay` seconds, then re-fetch chart data and decide whether to
        enter the previously skipped signal.

        Only chart-filtered runners are evaluated — baseline runners already
        entered at the original signal time.  One attempt only; if the chart
        still says SKIP the signal is abandoned.
        """
        await asyncio.sleep(delay)
        self._pending_reanalysis.discard(signal.mint_address)

        logger.info("[REANALYZE] %s — re-checking chart after %.0fs", signal.symbol, delay)

        # Fresh price at reanalysis time
        entry_price = await self._birdeye.get_price(signal.mint_address)
        if entry_price is None:
            logger.warning("[REANALYZE] %s — no price available, abandoning", signal.symbol)
            return

        # Fresh OHLCV (current candles, not historical)
        candles = await self._birdeye.get_ohlcv(signal.mint_address, bars=OHLCV_BARS)
        ctx = compute_chart_context(candles, entry_price)

        if ctx is None or not ctx.should_enter:
            reason = ctx.reason if ctx else "no candle data"
            logger.info(
                "[REANALYZE] %s @ $%.8f — still SKIP: %s — abandoning",
                signal.symbol, entry_price, reason,
            )
            signal_log.info(
                "REANALYZE_SKIP | %-10s | %-44s | %s",
                signal.symbol, signal.mint_address, reason,
            )
            return

        logger.info(
            "[REANALYZE] %s @ $%.8f — now BUY: %s — entering chart runners",
            signal.symbol, entry_price, ctx.reason,
        )
        signal_log.info(
            "REANALYZE_BUY  | %-10s | %-44s | %s",
            signal.symbol, signal.mint_address, ctx.reason,
        )

        # Enter only runners with both use_chart_filter and use_reanalyze
        for runner in self._runners:
            if not (runner.cfg.use_chart_filter and runner.cfg.use_reanalyze):
                continue
            try:
                runner.enter_position(signal, entry_price, ctx)
            except Exception:
                logger.exception(
                    "[ERROR] %s: reanalysis enter_position failed for %s",
                    runner.name, signal.symbol,
                )

    # ------------------------------------------------------------------
    # Monitoring loop
    # ------------------------------------------------------------------

    async def monitor_positions(self, cycles: Optional[int] = None) -> None:
        """
        Shared price polling loop.

        Each tick:
            1. Collect unique mints across all runners' open positions.
            2. Fetch each mint's price exactly once from Birdeye.
            3. Distribute each price to every runner holding that mint.
            4. Each runner evaluates its own exit logic independently.

        A mint is tracked until ALL strategies have exited it.
        """
        if self._cfg.dry_run:
            logger.info("[DRY_RUN] Monitor loop disabled — no positions will be opened")
            return

        mode = f"{cycles} cycles" if cycles is not None else "live (∞)"
        logger.info(
            "[MONITOR] Starting | strategies=%d | mode=%s | interval=%.1fs",
            len(self._runners), mode, self._cfg.poll_interval_seconds,
        )

        tick = 0
        while cycles is None or tick < cycles:
            tick += 1

            # Map: mint → [runners that currently hold it]
            mint_to_runners: dict[str, list[StrategyRunner]] = {}
            for runner in self._runners:
                for pos in runner.get_open_positions():
                    mint_to_runners.setdefault(pos.mint_address, []).append(runner)

            if not mint_to_runners:
                logger.debug("[MONITOR] Tick %d — no open positions", tick)
                await asyncio.sleep(self._cfg.poll_interval_seconds)
                continue

            # One batch call covers every unique mint across all strategies
            prices = await self._birdeye.get_prices_batch(list(mint_to_runners.keys()))

            # Fan each price out to all runners holding that mint
            for mint, holders in mint_to_runners.items():
                price = prices.get(mint)
                if price is None:
                    logger.warning("[PRICE] No price for %s this tick", mint)
                    continue

                for runner in holders:
                    pos = runner.get_position(mint)
                    if pos is None or pos.status == "CLOSED":
                        continue

                    pos.last_price = price
                    logger.info(
                        "[PRICE] [%-12s] %-10s | $%.8f | tp=$%.8f | sl=$%.8f | trailing=%s%s",
                        runner.name, pos.symbol, price,
                        pos.take_profit_price, pos.stop_loss_price,
                        pos.trailing_active,
                        f" | trail_stop=${pos.trailing_stop_price:.8f}" if pos.trailing_stop_price else "",
                    )

                    try:
                        runner.evaluate_position(pos, price)
                    except Exception:
                        logger.exception(
                            "[ERROR] %s: evaluate_position failed for %s",
                            runner.name, pos.symbol,
                        )

            await asyncio.sleep(self._cfg.poll_interval_seconds)

        logger.info("[MONITOR] Complete after %d ticks", tick)
        self.print_summary()

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_summary(self, current_prices: dict[str, float] | None = None) -> None:
        """Emit per-strategy summaries followed by a global aggregate."""
        sep = "=" * 72
        summaries = [r.summary(current_prices) for r in self._runners]

        for s in summaries:
            logger.info(sep)
            logger.info("[SUMMARY]  Strategy: %s", s["name"])
            logger.info(sep)
            logger.info("  Starting capital   : $%.2f", s["starting_cash"])
            logger.info("  Available cash     : $%.2f", s["available_cash"])
            logger.info("  Open market value  : $%.4f", s["market_value"])
            logger.info("  Total equity       : $%.4f", s["equity"])
            logger.info("  " + "-" * 52)
            logger.info("  Positions — open   : %d", s["open_count"])
            logger.info("  Positions — closed : %d", s["closed_count"])
            logger.info("  Win rate           : %.1f%%", s["win_rate"])
            logger.info("  " + "-" * 52)
            logger.info("  Realized PnL       : $%+.4f", s["realized_pnl"])
            logger.info("  Unrealized PnL     : $%+.4f", s["unrealized_pnl"])
            logger.info("  Net PnL vs start   : $%+.4f  (%+.2f%%)", s["net_pnl"], s["net_pnl_pct"])
            logger.info(sep)

            for p in s["positions"]:
                unreal = 0.0
                if p.status == "OPEN" and p.last_price:
                    unreal = (p.last_price - p.entry_price) * p.remaining_quantity
                logger.info(
                    "  [%-6s] %-10s | entry=$%.6f | last=$%.6f | "
                    "realized=$%+.4f | unrealized=$%+.4f | trail=%s | exit=%s",
                    p.status, p.symbol,
                    p.entry_price, p.last_price or 0.0,
                    p.realized_pnl_usd, unreal,
                    p.trailing_active,
                    p.sell_reason or "—",
                )
            logger.info(sep)

        # Global aggregate (only printed when more than one strategy is running)
        if len(summaries) > 1:
            total_start   = sum(s["starting_cash"]  for s in summaries)
            total_cash    = sum(s["available_cash"]  for s in summaries)
            total_mv      = sum(s["market_value"]    for s in summaries)
            total_equity  = sum(s["equity"]          for s in summaries)
            total_realized   = sum(s["realized_pnl"]   for s in summaries)
            total_unrealized = sum(s["unrealized_pnl"]  for s in summaries)
            total_net     = total_equity - total_start
            total_net_pct = (total_net / total_start * 100) if total_start else 0.0
            total_open    = sum(s["open_count"]   for s in summaries)
            total_closed  = sum(s["closed_count"] for s in summaries)

            logger.info(sep)
            logger.info("[SUMMARY]  GLOBAL  (%d strategies)", len(self._runners))
            logger.info(sep)
            logger.info("  Total starting capital : $%.2f", total_start)
            logger.info("  Total cash             : $%.2f", total_cash)
            logger.info("  Total open value       : $%.4f", total_mv)
            logger.info("  Total equity           : $%.4f", total_equity)
            logger.info("  " + "-" * 52)
            logger.info("  Total open positions   : %d", total_open)
            logger.info("  Total closed positions : %d", total_closed)
            logger.info("  " + "-" * 52)
            logger.info("  Total realized PnL     : $%+.4f", total_realized)
            logger.info("  Total unrealized PnL   : $%+.4f", total_unrealized)
            logger.info("  Total net PnL          : $%+.4f  (%+.2f%%)", total_net, total_net_pct)
            logger.info(sep)
