"""
trader/trading/exchange.py — Paper (mock) trade execution.

PaperExchange simulates buy/sell execution using real Birdeye prices
but performs NO on-chain transactions. All cash accounting is handled
here so TradingEngine stays free of portfolio arithmetic.

Swap point:
    Replace PaperExchange with JupiterSwapExecutor (or any executor that
    exposes the same buy() / sell_partial() / sell_all() interface) and
    inject it into TradingEngine. No other changes required.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from trader.trading.models import PortfolioState, Position, TokenSignal

logger = logging.getLogger(__name__)

# Amount of USD injected when a paper strategy runs out of available cash.
_PAPER_RELOAD_USD = 1_000.0


class PaperExchange:
    """
    Simulates trade execution against real market prices with zero on-chain risk.

    Cash accounting:
        available_cash_usd decreases on buy, increases on every sell.
        The accounting is intentionally simple: no slippage, no fees (fee
        scaffolding is in place via total_fees_usd for future modelling).

    PnL formula (applied consistently across partial and full sells):
        pnl = (exit_price − entry_price) × quantity_sold

    cfg is duck-typed — accepts StrategyConfig as long as it exposes
    stop_loss_pct, take_profit_levels, trailing_stop_pct.
    """

    def __init__(self, portfolio: PortfolioState, cfg) -> None:
        self.portfolio = portfolio   # public — TradingEngine reads it for summary
        self._cfg = cfg

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------

    def buy(
        self,
        signal: TokenSignal,
        entry_price: float,
        usd_size: float,
    ) -> Optional[Position]:
        """
        Open a paper long position.

        Returns None and logs a warning if available cash is insufficient.
        Deducts usd_size from available_cash_usd on success.
        """
        if self.portfolio.available_cash_usd < usd_size:
            if not self._cfg.live_trading:
                # Paper strategy: reload cash so data collection is never interrupted.
                self.portfolio.available_cash_usd += _PAPER_RELOAD_USD
                logger.warning(
                    "[RELOAD] %s paper cash below trade size — injecting $%.0f "
                    "(available now $%.2f)",
                    self._cfg.name,
                    _PAPER_RELOAD_USD,
                    self.portfolio.available_cash_usd,
                )
            else:
                logger.warning(
                    "[SKIP] Insufficient cash for %s — have $%.2f, need $%.2f",
                    signal.symbol,
                    self.portfolio.available_cash_usd,
                    usd_size,
                )
                return None

        quantity = usd_size / entry_price
        self.portfolio.available_cash_usd -= usd_size
        cfg = self._cfg

        return Position(
            symbol=signal.symbol,
            mint_address=signal.mint_address,
            entry_price=entry_price,
            initial_quantity=quantity,
            remaining_quantity=quantity,
            usd_size=usd_size,
            status="OPEN",
            opened_at=datetime.now(timezone.utc),
            closed_at=None,
            highest_price=entry_price,
            highest_price_ts=datetime.now(timezone.utc),
            lowest_price=entry_price,
            lowest_price_ts=datetime.now(timezone.utc),
            take_profit_price=entry_price * cfg.take_profit_levels[0].multiple,
            stop_loss_price=entry_price * (1.0 - cfg.stop_loss_pct),
            trailing_active=False,
            trailing_stop_pct=cfg.trailing_stop_pct,
            trailing_stop_price=None,
            realized_pnl_usd=0.0,
            sell_reason=None,
            last_price=entry_price,
            total_proceeds_usd=0.0,
            total_fees_usd=0.0,
            partial_take_profit_hit=False,
        )

    # ------------------------------------------------------------------
    # Exit
    # ------------------------------------------------------------------

    def sell_partial(
        self,
        position: Position,
        fraction: float,
        exit_price: float,
        reason: str,
    ) -> None:
        """
        Sell `fraction` of the REMAINING position.

        Does NOT close the position. remaining_quantity > 0 after this call.
        Proceeds are returned to available_cash_usd immediately.
        """
        quantity_sold = position.remaining_quantity * fraction
        proceeds = quantity_sold * exit_price
        pnl = (exit_price - position.entry_price) * quantity_sold

        position.remaining_quantity -= quantity_sold
        position.total_proceeds_usd += proceeds
        position.realized_pnl_usd += pnl
        self.portfolio.available_cash_usd += proceeds

        logger.info(
            "[SELL] %s partial %.0f%% | qty=%.6f | price=$%.8f | proceeds=$%.4f | pnl=$%+.4f | reason=%s",
            position.symbol, fraction * 100, quantity_sold,
            exit_price, proceeds, pnl, reason,
        )

    def sell_all(
        self,
        position: Position,
        exit_price: float,
        reason: str,
    ) -> None:
        """
        Sell all remaining quantity and mark the position CLOSED.

        realized_pnl_usd receives the final increment; sell_reason and
        closed_at are set. Proceeds are returned to available_cash_usd.
        """
        quantity_sold = position.remaining_quantity
        proceeds = quantity_sold * exit_price
        pnl = (exit_price - position.entry_price) * quantity_sold

        position.remaining_quantity = 0.0
        position.total_proceeds_usd += proceeds
        position.realized_pnl_usd += pnl
        position.status = "CLOSED"
        position.closed_at = datetime.now(timezone.utc)
        position.sell_reason = reason
        self.portfolio.available_cash_usd += proceeds

        logger.info(
            "[SELL] %s full | qty=%.6f | price=$%.8f | proceeds=$%.4f | pnl=$%+.4f | reason=%s",
            position.symbol, quantity_sold, exit_price, proceeds, pnl, reason,
        )
        logger.info(
            "[CLOSE] %s | total_realized_pnl=$%+.4f | reason=%s",
            position.symbol, position.realized_pnl_usd, reason,
        )
