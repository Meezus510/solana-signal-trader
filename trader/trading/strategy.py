"""
trader/trading/strategy.py — Per-strategy configuration and isolated execution.

StrategyConfig holds all tunable parameters for one strategy instance,
including optional fields for TP2, timeout, and max-hold rules.

StrategyRunner owns its own PortfolioManager + PaperExchange and applies
strategy rules to positions independently of all other runners.

Adding a new strategy: define a StrategyConfig and add it to run.py.
No code duplication required.

evaluate_position() state machine
----------------------------------
Phase 1 (trailing_active = False):
    → max hold check
    → timeout check
    → stop loss check
    → TP1 check  (sell fraction of remaining, activate trailing)

Phase 2 (trailing_active = True):
    → max hold check
    → timeout check
    → TP2 check  (optional — sell fixed % of ORIGINAL qty at a higher multiple)
    → update trailing high-water mark
    → trailing stop check  (sell all remaining)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
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
    Tunable parameters for one strategy instance.

    Required fields
    ---------------
    name, buy_size_usd, stop_loss_pct, take_profit_multiple,
    take_profit_sell_fraction, trailing_stop_pct, starting_cash_usd

    Optional fields
    ---------------
    take_profit2_multiple          — price multiple for a second TP event
    take_profit2_original_fraction — fraction of INITIAL qty to sell at TP2
                                     (not fraction of remaining)
    timeout_minutes                — exit if position is older than this …
    timeout_min_gain_pct           — … AND current price < entry * (1 + this)
    max_hold_minutes               — unconditional exit after this many minutes
    """
    # Core
    name: str
    buy_size_usd: float
    stop_loss_pct: float
    take_profit_multiple: float
    take_profit_sell_fraction: float
    trailing_stop_pct: float
    starting_cash_usd: float

    # TP2 (optional — e.g. Strategy C fast-scalp second take-profit)
    take_profit2_multiple: Optional[float] = None
    take_profit2_original_fraction: Optional[float] = None

    # Timeout: exit stagnant positions early
    timeout_minutes: Optional[float] = None
    timeout_min_gain_pct: Optional[float] = None  # exit if price < entry*(1+pct)

    # Max hold: unconditional time-based exit
    max_hold_minutes: Optional[float] = None


class StrategyRunner:
    """
    Manages the full lifecycle of one strategy in isolation:
        - owns its own PortfolioManager and PaperExchange
        - opens positions via enter_position()
        - evaluates and closes positions via evaluate_position()

    Price fetching is NOT done here. MultiStrategyEngine fetches each mint
    price once and fans it out to all StrategyRunners holding that mint.
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
    # Strategy evaluation
    # ------------------------------------------------------------------

    def evaluate_position(self, position: Position, current_price: float) -> None:
        """
        Apply all strategy rules to one position at the current price.

        Evaluation order (both phases):
            1. Max hold  — unconditional time exit
            2. Timeout   — exit if stagnant beyond the configured window

        Phase 1 (trailing not yet active):
            3. Stop loss — hard floor exit
            4. TP1       — partial sell, activates trailing

        Phase 2 (trailing active):
            3. TP2       — optional second partial sell (% of ORIGINAL qty)
            4. Trail     — update high-water mark, fire trailing stop
        """
        if position.status == "CLOSED":
            return

        cfg = self._cfg
        now = datetime.now(timezone.utc)
        opened_at = position.opened_at
        if opened_at.tzinfo is None:
            opened_at = opened_at.replace(tzinfo=timezone.utc)
        age_minutes = (now - opened_at).total_seconds() / 60.0

        # ---- Max hold (unconditional exit) ----
        if cfg.max_hold_minutes is not None and age_minutes >= cfg.max_hold_minutes:
            logger.info(
                "[%s] [MAX_HOLD] %s | age=%.0fmin >= %.0fmin",
                self.name, position.symbol, age_minutes, cfg.max_hold_minutes,
            )
            self._close_position(position, current_price, "MAX_HOLD")
            return

        # ---- Timeout (position not gaining enough) ----
        if (
            cfg.timeout_minutes is not None
            and age_minutes >= cfg.timeout_minutes
            and cfg.timeout_min_gain_pct is not None
            and current_price < position.entry_price * (1.0 + cfg.timeout_min_gain_pct)
        ):
            logger.info(
                "[%s] [TIMEOUT] %s | age=%.0fmin | price=$%.8f < threshold=$%.8f",
                self.name, position.symbol, age_minutes,
                current_price,
                position.entry_price * (1.0 + cfg.timeout_min_gain_pct),
            )
            self._close_position(position, current_price, "TIMEOUT_SLOW")
            return

        # ================================================================
        # Phase 1 — TP1 not yet hit (trailing not active)
        # ================================================================
        if not position.trailing_active:

            # ---- Hard stop loss ----
            if current_price <= position.stop_loss_price:
                logger.info(
                    "[%s] [SL] %s | current=$%.8f | sl=$%.8f",
                    self.name, position.symbol, current_price, position.stop_loss_price,
                )
                self._close_position(position, current_price, "STOP_LOSS")
                return

            # ---- TP1 (partial sell, activate trailing) ----
            if current_price >= position.take_profit_price:
                qty = position.remaining_quantity * cfg.take_profit_sell_fraction
                pnl = (current_price - position.entry_price) * qty
                self._exchange.sell_partial(
                    position, cfg.take_profit_sell_fraction, current_price, "TP1"
                )
                position.trailing_active = True
                position.partial_take_profit_hit = True
                position.highest_price = current_price
                position.trailing_stop_price = current_price * (1.0 - cfg.trailing_stop_pct)

                logger.info(
                    "[%s] [TP1] %s %.1f× | sold=%.0f%% | price=$%.8f | "
                    "trail_stop=$%.8f | remaining_qty=%.6f",
                    self.name, position.symbol, cfg.take_profit_multiple,
                    cfg.take_profit_sell_fraction * 100, current_price,
                    position.trailing_stop_price, position.remaining_quantity,
                )
                trade_log.info(
                    "TP1          | %-10s | %-44s | price=$%.8f | qty=%.4f | pnl=$%+.4f | strategy=%s",
                    position.symbol, position.mint_address,
                    current_price, qty, pnl, self.name,
                )
                if self._db:
                    self._db.upsert_position(position)
                    self._db.save_portfolio(self._exchange.portfolio, self.name)
                    self._db.log_trade("TP1", position, current_price, qty, pnl)

                # Edge case: if sell fraction was 100%, position is now empty
                if position.remaining_quantity < 1e-10:
                    position.status = "CLOSED"
                    position.closed_at = now
                    position.sell_reason = "TP1_FULL"
                    self._portfolio.close_position(position.mint_address)

        # ================================================================
        # Phase 2 — TP1 fired, trailing now active
        # ================================================================
        else:

            # ---- TP2 (optional second take-profit, % of ORIGINAL qty) ----
            if (
                cfg.take_profit2_multiple is not None
                and not position.tp2_hit
                and current_price >= position.entry_price * cfg.take_profit2_multiple
            ):
                # Sell a fixed percentage of the ORIGINAL position size,
                # not the remaining size — clamped so we never oversell.
                tp2_qty = position.initial_quantity * (cfg.take_profit2_original_fraction or 0.0)
                qty_to_sell = min(tp2_qty, position.remaining_quantity)

                if qty_to_sell > 1e-12:
                    fraction = qty_to_sell / position.remaining_quantity
                    pnl = (current_price - position.entry_price) * qty_to_sell
                    self._exchange.sell_partial(position, fraction, current_price, "TP2")
                    position.tp2_hit = True

                    logger.info(
                        "[%s] [TP2] %s %.1f× | sold_qty=%.6f (%.0f%% of original) | "
                        "price=$%.8f | pnl=$%+.4f | remaining_qty=%.6f",
                        self.name, position.symbol, cfg.take_profit2_multiple,
                        qty_to_sell,
                        (cfg.take_profit2_original_fraction or 0) * 100,
                        current_price, pnl, position.remaining_quantity,
                    )
                    trade_log.info(
                        "TP2          | %-10s | %-44s | price=$%.8f | qty=%.4f | pnl=$%+.4f | strategy=%s",
                        position.symbol, position.mint_address,
                        current_price, qty_to_sell, pnl, self.name,
                    )
                    if self._db:
                        self._db.upsert_position(position)
                        self._db.save_portfolio(self._exchange.portfolio, self.name)
                        self._db.log_trade("TP2", position, current_price, qty_to_sell, pnl)

                    # If TP2 exhausted the remaining quantity, close cleanly
                    if position.remaining_quantity < 1e-10:
                        position.status = "CLOSED"
                        position.closed_at = now
                        position.sell_reason = "TP2_FULL"
                        self._portfolio.close_position(position.mint_address)
                        return

            # ---- Update trailing high-water mark ----
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
                self._close_position(position, current_price, "TRAILING_STOP")

    # ------------------------------------------------------------------
    # Internal exit helper
    # ------------------------------------------------------------------

    def _close_position(self, position: Position, price: float, reason: str) -> None:
        """Sell all remaining quantity, close the position, and persist."""
        qty = position.remaining_quantity
        pnl = (price - position.entry_price) * qty
        self._exchange.sell_all(position, price, reason)
        self._portfolio.close_position(position.mint_address)
        trade_log.info(
            "%-13s | %-10s | %-44s | price=$%.8f | qty=%.4f | pnl=$%+.4f | strategy=%s",
            reason, position.symbol, position.mint_address,
            price, qty, pnl, self.name,
        )
        if self._db:
            self._db.upsert_position(position)
            self._db.save_portfolio(self._exchange.portfolio, self.name)
            self._db.log_trade(reason, position, price, qty, pnl)

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
