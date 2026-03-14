"""
trader/trading/strategy.py — Per-strategy configuration and isolated execution.

StrategyConfig holds the tunable parameters for one strategy instance.
StrategyRunner owns its own PortfolioManager + PaperExchange and applies
strategy rules to positions independently of all other runners.

MultiStrategyEngine (in engine.py) coordinates N StrategyRunners:
    - fetches prices once per unique mint per polling cycle
    - fans prices out to all runners that hold that mint
    - aggregates PnL for global reporting

To add a new strategy, define a StrategyConfig and pass it to run.py —
no code duplication, no new classes required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from trader.trading.exchange import PaperExchange
from trader.trading.models import Position, PortfolioState, TokenSignal
from trader.trading.portfolio import PortfolioManager

logger = logging.getLogger(__name__)
trade_log = logging.getLogger("trades")
signal_log = logging.getLogger("signals")


@dataclass(frozen=True)
class StrategyConfig:
    """
    Tunable parameters for a single strategy instance.

    Add a new strategy by instantiating another StrategyConfig with
    different values — no code changes required anywhere else.
    """
    name: str
    buy_size_usd: float
    stop_loss_pct: float
    take_profit_multiple: float
    take_profit_sell_fraction: float
    trailing_stop_pct: float
    starting_cash_usd: float


class StrategyRunner:
    """
    Manages the full lifecycle of one strategy in isolation:
        - owns its own PortfolioManager and PaperExchange
        - opens positions via enter_position()
        - evaluates and closes positions via evaluate_position()

    Price fetching is NOT done here. MultiStrategyEngine fetches each mint
    price once and fans it out to all StrategyRunners holding that mint,
    so N strategies holding the same token cost only 1 Birdeye API call.
    """

    def __init__(self, cfg: StrategyConfig, db=None) -> None:
        self._cfg = cfg
        self._db = db
        portfolio = PortfolioState(
            starting_cash_usd=cfg.starting_cash_usd,
            available_cash_usd=cfg.starting_cash_usd,
        )
        self._exchange = PaperExchange(portfolio=portfolio, cfg=cfg)
        self._portfolio = PortfolioManager()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._cfg.name

    @property
    def cfg(self) -> StrategyConfig:
        return self._cfg

    @property
    def portfolio_state(self) -> PortfolioState:
        return self._exchange.portfolio

    # ------------------------------------------------------------------
    # State restoration (called on startup from DB)
    # ------------------------------------------------------------------

    def restore_cash(self, available_cash: float, starting_cash: float) -> None:
        self._exchange.portfolio.available_cash_usd = available_cash
        self._exchange.portfolio.starting_cash_usd = starting_cash

    def restore_positions(self, positions: list[Position]) -> None:
        for pos in positions:
            self._portfolio.add_position(pos)
        if positions:
            logger.info("[%s] Restored %d open position(s)", self.name, len(positions))

    # ------------------------------------------------------------------
    # Position queries
    # ------------------------------------------------------------------

    def get_open_positions(self) -> list[Position]:
        return self._portfolio.get_open_positions()

    def get_position(self, mint_address: str) -> Optional[Position]:
        return self._portfolio.get_position(mint_address)

    def has_open_position(self, mint_address: str) -> bool:
        return self._portfolio.has_open_position(mint_address)

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------

    def enter_position(self, signal: TokenSignal, entry_price: float) -> Optional[Position]:
        """
        Attempt to open a position for this strategy at the pre-fetched price.

        Returns the opened Position, or None if skipped (duplicate / no cash).
        Price is supplied by MultiStrategyEngine to avoid duplicate API calls.
        """
        if self._portfolio.has_open_position(signal.mint_address):
            logger.info(
                "[%s] [SKIP] Already holding %s — duplicate signal ignored",
                self.name, signal.symbol,
            )
            signal_log.info(
                "DUPLICATE  | %-10s | %-44s | strategy=%s",
                signal.symbol, signal.mint_address, self.name,
            )
            if self._db:
                self._db.log_signal(
                    "DUPLICATE", symbol=signal.symbol,
                    mint=signal.mint_address, strategy=self.name,
                )
            return None

        position = self._exchange.buy(signal, entry_price, self._cfg.buy_size_usd)
        if position is None:
            signal_log.info(
                "NO_CASH    | %-10s | %-44s | strategy=%s",
                signal.symbol, signal.mint_address, self.name,
            )
            if self._db:
                self._db.log_signal(
                    "NO_CASH", symbol=signal.symbol,
                    mint=signal.mint_address, strategy=self.name,
                )
            return None

        position.strategy_name = self.name
        self._portfolio.add_position(position)

        logger.info(
            "[%s] [BUY] %s | qty=%.4f | entry=$%.8f | tp=$%.8f | sl=$%.8f | cash=$%.2f",
            self.name, signal.symbol,
            position.initial_quantity, entry_price,
            position.take_profit_price, position.stop_loss_price,
            self._exchange.portfolio.available_cash_usd,
        )
        trade_log.info(
            "BUY          | %-10s | %-44s | price=$%.8f | qty=%.4f | pnl=$0.00 | strategy=%s",
            position.symbol, position.mint_address,
            entry_price, position.initial_quantity, self.name,
        )
        if self._db:
            self._db.upsert_position(position)
            self._db.save_portfolio(self._exchange.portfolio, self.name)
            self._db.log_trade("BUY", position, entry_price, position.initial_quantity, 0.0)

        return position

    # ------------------------------------------------------------------
    # Strategy evaluation  (same logic as the original TradingEngine)
    # ------------------------------------------------------------------

    def evaluate_position(self, position: Position, current_price: float) -> None:
        """
        Apply strategy rules to one position given the current market price.

        Phase 1 — trailing NOT active:
            1. Stop loss  → sell_all()     → CLOSE (STOP_LOSS)
            2. Take profit → sell_partial() → activate trailing

        Phase 2 — trailing IS active:
            1. Update highest_price / trailing_stop_price on new high
            2. Trailing stop hit → sell_all() → CLOSE (TRAILING_STOP)
        """
        if position.status == "CLOSED":
            return

        cfg = self._cfg

        if not position.trailing_active:

            # ---- Hard stop loss ----
            if current_price <= position.stop_loss_price:
                logger.info(
                    "[%s] [SL] %s | current=$%.8f | sl=$%.8f",
                    self.name, position.symbol, current_price, position.stop_loss_price,
                )
                qty = position.remaining_quantity
                self._exchange.sell_all(position, current_price, "STOP_LOSS")
                self._portfolio.close_position(position.mint_address)
                pnl = (current_price - position.entry_price) * qty
                trade_log.info(
                    "STOP_LOSS    | %-10s | %-44s | price=$%.8f | qty=%.4f | pnl=$%+.4f | strategy=%s",
                    position.symbol, position.mint_address,
                    current_price, qty, pnl, self.name,
                )
                if self._db:
                    self._db.upsert_position(position)
                    self._db.save_portfolio(self._exchange.portfolio, self.name)
                    self._db.log_trade("STOP_LOSS", position, current_price, qty, pnl)
                return

            # ---- Take profit (partial sell) ----
            if current_price >= position.take_profit_price:
                logger.info(
                    "[%s] [TP] %s %.1f× | current=$%.8f — selling %.0f%%",
                    self.name, position.symbol, cfg.take_profit_multiple,
                    current_price, cfg.take_profit_sell_fraction * 100,
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
                    "[%s] [TRAIL] %s activated | highest=$%.8f | trail_stop=$%.8f",
                    self.name, position.symbol,
                    position.highest_price, position.trailing_stop_price,
                )
                trade_log.info(
                    "TP_PARTIAL   | %-10s | %-44s | price=$%.8f | qty=%.4f | pnl=$%+.4f | strategy=%s",
                    position.symbol, position.mint_address,
                    current_price, qty, pnl, self.name,
                )
                if self._db:
                    self._db.upsert_position(position)
                    self._db.save_portfolio(self._exchange.portfolio, self.name)
                    self._db.log_trade("TP_PARTIAL", position, current_price, qty, pnl)

        else:
            # ---- Update trailing stop on new high ----
            if current_price > position.highest_price:
                position.highest_price = current_price
                position.trailing_stop_price = current_price * (1.0 - cfg.trailing_stop_pct)
                logger.info(
                    "[%s] [TRAIL] %s new high=$%.8f | trail_stop=$%.8f",
                    self.name, position.symbol,
                    position.highest_price, position.trailing_stop_price,
                )

            # ---- Trailing stop triggered ----
            if (
                position.trailing_stop_price is not None
                and current_price <= position.trailing_stop_price
            ):
                logger.info(
                    "[%s] [TRAIL] %s stop triggered | current=$%.8f | trail_stop=$%.8f",
                    self.name, position.symbol, current_price, position.trailing_stop_price,
                )
                qty = position.remaining_quantity
                pnl = (current_price - position.entry_price) * qty
                self._exchange.sell_all(position, current_price, "TRAILING_STOP")
                self._portfolio.close_position(position.mint_address)
                trade_log.info(
                    "TRAILING_STOP | %-10s | %-44s | price=$%.8f | qty=%.4f | pnl=$%+.4f | strategy=%s",
                    position.symbol, position.mint_address,
                    current_price, qty, pnl, self.name,
                )
                if self._db:
                    self._db.upsert_position(position)
                    self._db.save_portfolio(self._exchange.portfolio, self.name)
                    self._db.log_trade("TRAILING_STOP", position, current_price, qty, pnl)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self, current_prices: dict[str, float] | None = None) -> dict:
        """Return a stats dict for this strategy — consumed by MultiStrategyEngine.print_summary()."""
        port = self._exchange.portfolio
        all_pos = self._portfolio.all_positions()
        open_pos = self._portfolio.get_open_positions()
        closed_pos = self._portfolio.get_closed_positions()

        realized = sum(p.realized_pnl_usd for p in all_pos)
        unrealized = 0.0
        market_value = 0.0
        for p in open_pos:
            last = (current_prices or {}).get(p.mint_address) or p.last_price
            if last is not None:
                mv = p.remaining_quantity * last
                market_value += mv
                unrealized += mv - (p.remaining_quantity * p.entry_price)

        equity = port.available_cash_usd + market_value
        net_pnl = equity - port.starting_cash_usd
        net_pnl_pct = (net_pnl / port.starting_cash_usd * 100) if port.starting_cash_usd else 0.0

        wins = [p for p in closed_pos if p.realized_pnl_usd > 0]
        win_rate = (len(wins) / len(closed_pos) * 100) if closed_pos else 0.0

        return {
            "name": self.name,
            "starting_cash": port.starting_cash_usd,
            "available_cash": port.available_cash_usd,
            "market_value": market_value,
            "equity": equity,
            "realized_pnl": realized,
            "unrealized_pnl": unrealized,
            "net_pnl": net_pnl,
            "net_pnl_pct": net_pnl_pct,
            "open_count": self._portfolio.open_count,
            "closed_count": self._portfolio.closed_count,
            "win_rate": win_rate,
            "positions": all_pos,
        }
