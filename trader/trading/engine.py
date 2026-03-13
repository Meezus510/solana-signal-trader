"""
trader/trading/engine.py — Strategy orchestration.

TradingEngine is the top-level coordinator. It wires together:
    - BirdeyePriceClient  (market data)
    - PaperExchange       (order execution)
    - PortfolioManager    (position registry)

It owns the monitoring loop, the strategy evaluation logic (TP / SL /
trailing stop), and the final summary report.

All dependencies are injected so each can be independently tested or
swapped (e.g. PaperExchange → JupiterSwapExecutor for live trading).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from trader.config import Config
from trader.pricing.birdeye import BirdeyePriceClient
from trader.trading.exchange import PaperExchange
from trader.trading.models import Position, TokenSignal
from trader.trading.portfolio import PortfolioManager

logger = logging.getLogger(__name__)
trade_log = logging.getLogger("trades")
signal_log = logging.getLogger("signals")


class TradingEngine:
    """
    Orchestrates the full signal-to-close lifecycle:

        1. handle_new_signal()   — receive signal, fetch price, open position
        2. monitor_positions()   — polling loop: price → evaluate → TP/SL/trail
        3. evaluate_position()   — applies strategy rules to a single position
        4. print_summary()       — structured PnL report

    Swap points:
        - birdeye_client: replace with a WebSocket feed adapter
        - paper_exchange: replace with JupiterSwapExecutor
        - No changes needed to this class when going live
    """

    def __init__(
        self,
        cfg: Config,
        birdeye_client: BirdeyePriceClient,
        exchange: PaperExchange,
        portfolio: PortfolioManager,
        db=None,           # TradeDatabase | None
    ) -> None:
        self._cfg = cfg
        self._birdeye = birdeye_client
        self._exchange = exchange
        self._portfolio = portfolio
        self._db = db

    # ------------------------------------------------------------------
    # Signal handling / entry
    # ------------------------------------------------------------------

    async def handle_new_signal(self, signal: TokenSignal) -> None:
        """
        Process an incoming token signal and open a paper position if viable.

        Guards:
            - Rejects if an open position already exists for this mint.
            - Rejects if Birdeye cannot return a price.
            - Rejects if available cash < buy_size_usd.

        Swap point — to replace the demo signal injection with the live
        Telegram listener, see run.py: the queue-based consumer already
        calls this method with signals produced by the listener pipeline.
        """
        logger.info(
            "[SIGNAL] %s | mint=%s", signal.symbol, signal.mint_address
        )

        if self._portfolio.has_open_position(signal.mint_address):
            logger.info(
                "[SKIP] Open position already exists for %s — duplicate signal ignored",
                signal.symbol,
            )
            signal_log.info("DUPLICATE  | %-10s | %-44s", signal.symbol, signal.mint_address)
            if self._db:
                self._db.log_signal("DUPLICATE", symbol=signal.symbol, mint=signal.mint_address)
            return

        entry_price = await self._birdeye.get_price(signal.mint_address)
        if entry_price is None:
            logger.warning(
                "[SKIP] No live price available for %s — entry aborted", signal.symbol
            )
            signal_log.info("NO_PRICE   | %-10s | %-44s", signal.symbol, signal.mint_address)
            if self._db:
                self._db.log_signal("NO_PRICE", symbol=signal.symbol, mint=signal.mint_address)
            return

        logger.info("[ENTRY] %s @ $%.8f", signal.symbol, entry_price)

        position = self._exchange.buy(signal, entry_price, self._cfg.buy_size_usd)
        if position is None:
            signal_log.info("NO_CASH    | %-10s | %-44s", signal.symbol, signal.mint_address)
            if self._db:
                self._db.log_signal("NO_CASH", symbol=signal.symbol, mint=signal.mint_address)
            return  # insufficient cash — already logged by PaperExchange

        self._portfolio.add_position(position)
        logger.info(
            "[BUY] %s | qty=%.4f | tp=$%.8f | sl=$%.8f | cash=$%.2f",
            signal.symbol,
            position.initial_quantity,
            position.take_profit_price,
            position.stop_loss_price,
            self._exchange.portfolio.available_cash_usd,
        )
        trade_log.info(
            "BUY          | %-10s | %-44s | price=$%.8f | qty=%.4f | pnl=$0.00",
            position.symbol, position.mint_address, entry_price, position.initial_quantity,
        )
        if self._db:
            self._db.upsert_position(position)
            self._db.save_portfolio(self._exchange.portfolio)
            self._db.log_trade("BUY", position, entry_price, position.initial_quantity, 0.0)

    # ------------------------------------------------------------------
    # Monitoring loop
    # ------------------------------------------------------------------

    async def monitor_positions(self, cycles: Optional[int] = None) -> None:
        """
        Poll prices for all open positions and evaluate strategy on each tick.

        Args:
            cycles: Number of polling cycles. None = run indefinitely (live mode).

        Swap point — to replace REST polling with WebSocket streaming:
            Remove this method. Add BirdeyePriceClient.subscribe(mints, callback)
            and call evaluate_position() from the callback. No other changes needed.
        """
        mode = f"{cycles} cycles" if cycles is not None else "live (∞)"
        logger.info(
            "[MONITOR] Starting | mode=%s | interval=%.1fs", mode, self._cfg.poll_interval_seconds
        )

        tick = 0
        while cycles is None or tick < cycles:
            tick += 1
            open_positions = self._portfolio.get_open_positions()

            if not open_positions:
                logger.debug("[MONITOR] Tick %d — no open positions", tick)
                await asyncio.sleep(self._cfg.poll_interval_seconds)
                continue

            mints = [p.mint_address for p in open_positions]
            prices = await self._birdeye.get_prices_batch(mints)

            for pos in open_positions:
                if pos.status == "CLOSED":
                    continue

                price = prices.get(pos.mint_address)
                if price is None:
                    logger.warning("[PRICE] %s — no price this tick, skipping", pos.symbol)
                    continue

                pos.last_price = price
                logger.info(
                    "[PRICE] %-10s | $%.8f | tp=$%.8f | sl=$%.8f | trailing=%s%s",
                    pos.symbol,
                    price,
                    pos.take_profit_price,
                    pos.stop_loss_price,
                    pos.trailing_active,
                    f" | trail_stop=${pos.trailing_stop_price:.8f}" if pos.trailing_stop_price else "",
                )

                await self.evaluate_position(pos, price)

            await asyncio.sleep(self._cfg.poll_interval_seconds)

        logger.info("[MONITOR] Complete after %d ticks", tick)
        self.print_summary()

    # ------------------------------------------------------------------
    # Strategy evaluation
    # ------------------------------------------------------------------

    async def evaluate_position(
        self,
        position: Position,
        current_price: float,
    ) -> None:
        """
        Apply strategy rules to one position given the current market price.

        Phase 1 — trailing NOT active (normal monitoring):
            1. Stop loss  → sell_all()     → CLOSE (STOP_LOSS)
            2. Take profit → sell_partial() → activate trailing

        Phase 2 — trailing IS active (on remaining 50%):
            1. Update highest_price / trailing_stop_price if new high
            2. Trailing stop hit → sell_all() → CLOSE (TRAILING_STOP)

        Critical invariant:
            The position is NOT closed at the 2.5× TP event — only a
            partial sell fires. The position stays OPEN until the trailing
            stop or a stop loss fires on the remaining quantity.
        """
        if position.status == "CLOSED":
            return

        cfg = self._cfg

        if not position.trailing_active:

            # ---- Hard stop loss ----
            if current_price <= position.stop_loss_price:
                logger.info(
                    "[SL] %s stop loss | current=$%.8f | sl=$%.8f",
                    position.symbol, current_price, position.stop_loss_price,
                )
                qty = position.remaining_quantity
                self._exchange.sell_all(position, current_price, "STOP_LOSS")
                self._portfolio.close_position(position.mint_address)
                pnl = (current_price - position.entry_price) * qty
                trade_log.info(
                    "STOP_LOSS    | %-10s | %-44s | price=$%.8f | qty=%.4f | pnl=$%+.4f",
                    position.symbol, position.mint_address, current_price, qty, pnl,
                )
                if self._db:
                    self._db.upsert_position(position)
                    self._db.save_portfolio(self._exchange.portfolio)
                    self._db.log_trade("STOP_LOSS", position, current_price, qty, pnl)
                return

            # ---- Take profit (partial sell) ----
            if current_price >= position.take_profit_price:
                logger.info(
                    "[TP] %s %.1f× take profit | current=$%.8f — selling %.0f%%",
                    position.symbol,
                    cfg.take_profit_multiple,
                    current_price,
                    cfg.take_profit_sell_fraction * 100,
                )
                qty = position.remaining_quantity * cfg.take_profit_sell_fraction
                pnl = (current_price - position.entry_price) * qty
                self._exchange.sell_partial(
                    position,
                    cfg.take_profit_sell_fraction,
                    current_price,
                    "TP_2_5X",
                )
                position.trailing_active = True
                position.partial_take_profit_hit = True
                position.highest_price = current_price
                position.trailing_stop_price = current_price * (1.0 - cfg.trailing_stop_pct)
                logger.info(
                    "[TRAIL] %s activated | highest=$%.8f | trail_stop=$%.8f",
                    position.symbol, position.highest_price, position.trailing_stop_price,
                )
                trade_log.info(
                    "TP_PARTIAL   | %-10s | %-44s | price=$%.8f | qty=%.4f | pnl=$%+.4f",
                    position.symbol, position.mint_address, current_price, qty, pnl,
                )
                if self._db:
                    self._db.upsert_position(position)
                    self._db.save_portfolio(self._exchange.portfolio)
                    self._db.log_trade("TP_PARTIAL", position, current_price, qty, pnl)

        else:
            # ---- Update trailing stop on new high ----
            if current_price > position.highest_price:
                position.highest_price = current_price
                position.trailing_stop_price = current_price * (1.0 - cfg.trailing_stop_pct)
                logger.info(
                    "[TRAIL] %s new high=$%.8f | trail_stop=$%.8f",
                    position.symbol, position.highest_price, position.trailing_stop_price,
                )

            # ---- Trailing stop triggered ----
            if (
                position.trailing_stop_price is not None
                and current_price <= position.trailing_stop_price
            ):
                logger.info(
                    "[TRAIL] %s stop triggered | current=$%.8f | trail_stop=$%.8f",
                    position.symbol, current_price, position.trailing_stop_price,
                )
                qty = position.remaining_quantity
                pnl = (current_price - position.entry_price) * qty
                self._exchange.sell_all(position, current_price, "TRAILING_STOP")
                self._portfolio.close_position(position.mint_address)
                trade_log.info(
                    "TRAILING_STOP | %-10s | %-44s | price=$%.8f | qty=%.4f | pnl=$%+.4f",
                    position.symbol, position.mint_address, current_price, qty, pnl,
                )
                if self._db:
                    self._db.upsert_position(position)
                    self._db.save_portfolio(self._exchange.portfolio)
                    self._db.log_trade("TRAILING_STOP", position, current_price, qty, pnl)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        """
        Emit a structured portfolio summary to the logger.

        Reports: cash, market value, equity, realized/unrealized PnL,
        net return vs starting capital, and a per-position breakdown.
        """
        port = self._exchange.portfolio
        all_pos = self._portfolio.all_positions()
        open_pos = self._portfolio.get_open_positions()

        realized = sum(p.realized_pnl_usd for p in all_pos)
        unrealized = 0.0
        market_value = 0.0
        for p in open_pos:
            if p.last_price is not None:
                mv = p.remaining_quantity * p.last_price
                market_value += mv
                unrealized += mv - (p.remaining_quantity * p.entry_price)

        equity = port.available_cash_usd + market_value
        net_pnl = equity - port.starting_cash_usd

        sep = "=" * 68
        logger.info(sep)
        logger.info("[SUMMARY]  Portfolio Report")
        logger.info(sep)
        logger.info("  Starting capital   : $%.2f", port.starting_cash_usd)
        logger.info("  Available cash     : $%.2f", port.available_cash_usd)
        logger.info("  Open market value  : $%.4f", market_value)
        logger.info("  Total equity       : $%.4f", equity)
        logger.info("  " + "-" * 46)
        logger.info("  Positions — total  : %d", self._portfolio.total_count)
        logger.info("  Positions — open   : %d", self._portfolio.open_count)
        logger.info("  Positions — closed : %d", self._portfolio.closed_count)
        logger.info("  " + "-" * 46)
        logger.info("  Realized PnL       : $%+.4f", realized)
        logger.info("  Unrealized PnL     : $%+.4f", unrealized)
        logger.info("  Net PnL vs start   : $%+.4f  (%+.2f%%)",
                    net_pnl, (net_pnl / port.starting_cash_usd) * 100)
        logger.info(sep)

        for p in all_pos:
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
