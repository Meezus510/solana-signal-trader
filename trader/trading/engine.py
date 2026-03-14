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

from trader.config import Config
from trader.pricing.birdeye import BirdeyePriceClient
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
    ) -> None:
        self._cfg = cfg
        self._runners = runners
        self._birdeye = birdeye_client
        self._db = db

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    async def handle_new_signal(self, signal: TokenSignal) -> None:
        """
        Process a new token signal across all strategies.

        Entry price is fetched once and shared — N strategies entering
        the same token costs 1 Birdeye call, not N calls.
        """
        logger.info("[SIGNAL] %s | mint=%s", signal.symbol, signal.mint_address)
        signal_log.info("SIGNAL     | %-10s | %-44s", signal.symbol, signal.mint_address)
        if self._db:
            self._db.log_signal("SIGNAL", symbol=signal.symbol, mint=signal.mint_address)

        entry_price = await self._birdeye.get_price(signal.mint_address)
        if entry_price is None:
            logger.warning("[SKIP] No live price for %s — entry aborted", signal.symbol)
            signal_log.info("NO_PRICE   | %-10s | %-44s", signal.symbol, signal.mint_address)
            if self._db:
                self._db.log_signal("NO_PRICE", symbol=signal.symbol, mint=signal.mint_address)
            return

        logger.info(
            "[ENTRY] %s @ $%.8f — distributing to %d strategy(s)",
            signal.symbol, entry_price, len(self._runners),
        )
        for runner in self._runners:
            try:
                runner.enter_position(signal, entry_price)
            except Exception:
                logger.exception(
                    "[ERROR] %s: enter_position failed for %s", runner.name, signal.symbol
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
