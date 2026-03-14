"""
trader/trading/strategy.py — Per-strategy configuration and isolated execution.

TakeProfitLevel    — one TP milestone (price multiple + fraction of original qty)
StrategyConfig     — full tunable config for one strategy instance
StrategyRunner     — base class; handles entry, standard TP/trailing, reporting, DB
InfiniteMoonbagRunner — subclass; overrides evaluate_position() with a progressive
                        ratcheting stop ladder and no time-based exits

Order of operations per price tick (StrategyRunner):
    1. Update highest_price_seen if new high
    2. Check existing stop (from PREVIOUS tick) — trailing_stop if active, else stop_loss
    3. Check max_hold (unconditional time exit)
    4. Check timeout (exit if stagnant)
    5. Process TP milestones in ascending order — each sells % of ORIGINAL qty
    6. Update trailing stop based on latest highest_price_seen
    (No same-tick re-check so behavior is deterministic)

Order of operations per price tick (InfiniteMoonbagRunner):
    1. Update highest_price_seen if new high
    2. Check existing stop (stop_loss_price = progressive ladder from previous tick)
    3. Process TP milestones in ascending order — each sells % of ORIGINAL qty
    4. Recompute stop ladder (uses updated highest_price_seen, always monotonic)
    (No timeout. No max hold.)
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

_EPSILON = 1e-10  # treat remaining_quantity below this as zero


# ---------------------------------------------------------------------------
# Config types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TakeProfitLevel:
    """One TP milestone: fire when price >= entry * multiple, sell sell_fraction_original of ORIGINAL qty."""
    multiple: float
    sell_fraction_original: float   # fraction of position.initial_quantity, NOT remaining


@dataclass(frozen=True)
class StrategyConfig:
    """
    Tunable parameters for one strategy instance.

    take_profit_levels must be ordered ascending by multiple so that milestones
    are processed lowest-to-highest on each price tick.

    Optional fields
    ---------------
    timeout_minutes / timeout_min_gain_pct
        Exit if age > timeout_minutes AND price < entry * (1 + timeout_min_gain_pct).
    max_hold_minutes
        Unconditional exit after this many minutes regardless of price.
    """
    name: str
    buy_size_usd: float
    stop_loss_pct: float                          # initial stop: entry * (1 - pct)
    take_profit_levels: tuple[TakeProfitLevel, ...] # TP milestones, ASC by multiple
    trailing_stop_pct: float                      # trail: highest * (1 - pct) after TP1
    starting_cash_usd: float

    # Time-based exits (optional — InfiniteMoonbagRunner ignores these)
    timeout_minutes: Optional[float] = None
    timeout_min_gain_pct: Optional[float] = None
    max_hold_minutes: Optional[float] = None


# ---------------------------------------------------------------------------
# Base runner
# ---------------------------------------------------------------------------

class StrategyRunner:
    """
    Manages the full lifecycle of one strategy in complete isolation.

    Each runner owns its own PortfolioManager + PaperExchange so cash,
    positions, and PnL are never shared with other strategies.

    Price fetching lives in MultiStrategyEngine — runners only receive a
    pre-fetched price, ensuring one Birdeye call per unique mint per cycle.

    Subclass and override evaluate_position() to implement custom exit logic
    (see InfiniteMoonbagRunner below).
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
        Open a paper position at the pre-fetched price.
        Returns None if skipped (duplicate mint or insufficient cash).
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
            "[%s] [BUY] %s | qty=%.4f | entry=$%.8f | tp1=$%.8f | sl=$%.8f | cash=$%.2f",
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
        Standard evaluation used by quick_pop and trend_rider.

        All TP sells use % of ORIGINAL position quantity (not remaining).
        Trailing stop activates after TP1 and is monotonically tightened.

        Key ordering rule: check existing stop BEFORE updating the trailing
        stop, so that same-tick trailing tightening cannot cause same-tick
        unexpected exits. TPs are processed first, trailing state updated after.
        """
        if position.status == "CLOSED":
            return

        cfg = self._cfg
        now = datetime.now(timezone.utc)
        opened_at = position.opened_at
        if opened_at.tzinfo is None:
            opened_at = opened_at.replace(tzinfo=timezone.utc)
        age_minutes = (now - opened_at).total_seconds() / 60.0

        # 1. Update highest price seen
        if current_price > position.highest_price:
            position.highest_price = current_price

        # 2. Check existing stop (values from previous tick — before any update this tick)
        #    Trailing stop if active, else fixed stop_loss_price.
        current_stop = (
            position.trailing_stop_price
            if position.trailing_active and position.trailing_stop_price is not None
            else position.stop_loss_price
        )
        if current_price <= current_stop:
            reason = "TRAILING_STOP" if position.trailing_active else "STOP_LOSS"
            logger.info(
                "[%s] [%s] %s | price=$%.8f | stop=$%.8f",
                self.name, reason, position.symbol, current_price, current_stop,
            )
            self._close_position(position, current_price, reason, now)
            return

        # 3. Max hold (unconditional)
        if cfg.max_hold_minutes is not None and age_minutes >= cfg.max_hold_minutes:
            logger.info(
                "[%s] [MAX_HOLD] %s | age=%.0fmin >= %.0fmin",
                self.name, position.symbol, age_minutes, cfg.max_hold_minutes,
            )
            self._close_position(position, current_price, "MAX_HOLD", now)
            return

        # 4. Timeout (position not gaining enough)
        if (
            cfg.timeout_minutes is not None
            and age_minutes >= cfg.timeout_minutes
            and cfg.timeout_min_gain_pct is not None
            and current_price < position.entry_price * (1.0 + cfg.timeout_min_gain_pct)
        ):
            logger.info(
                "[%s] [TIMEOUT] %s | age=%.0fmin | price=$%.8f | threshold=$%.8f",
                self.name, position.symbol, age_minutes,
                current_price,
                position.entry_price * (1.0 + cfg.timeout_min_gain_pct),
            )
            self._close_position(position, current_price, "TIMEOUT_SLOW", now)
            return

        # 5. Process TP milestones in ascending order (config must be sorted ASC)
        #    Each fires when price >= entry * level.multiple and the flag is not yet set.
        #    Sells sell_fraction_original of ORIGINAL quantity (clamped to remaining).
        for i, tp_level in enumerate(cfg.take_profit_levels):
            if i > 1:
                break  # Position model supports up to 2 TP flags

            already_hit = position.partial_take_profit_hit if i == 0 else position.tp2_hit
            if already_hit:
                continue

            if current_price < position.entry_price * tp_level.multiple:
                continue  # not reached yet

            tp_qty = min(
                position.initial_quantity * tp_level.sell_fraction_original,
                position.remaining_quantity,
            )
            if tp_qty < _EPSILON:
                continue

            fraction = tp_qty / position.remaining_quantity
            pnl = (current_price - position.entry_price) * tp_qty
            label = f"TP{i + 1}"
            self._exchange.sell_partial(position, fraction, current_price, label)

            if i == 0:
                position.partial_take_profit_hit = True
                position.trailing_active = True
            else:
                position.tp2_hit = True

            logger.info(
                "[%s] [%s] %s %.1f× | sold %.0f%% of original | qty=%.6f | "
                "pnl=$%+.4f | remaining=%.6f",
                self.name, label, position.symbol, tp_level.multiple,
                tp_level.sell_fraction_original * 100,
                tp_qty, pnl, position.remaining_quantity,
            )
            trade_log.info(
                "%-13s | %-10s | %-44s | price=$%.8f | qty=%.4f | pnl=$%+.4f | strategy=%s",
                label, position.symbol, position.mint_address,
                current_price, tp_qty, pnl, self.name,
            )
            if self._db:
                self._db.upsert_position(position)
                self._db.save_portfolio(self._exchange.portfolio, self.name)
                self._db.log_trade(label, position, current_price, tp_qty, pnl)

            if position.remaining_quantity < _EPSILON:
                self._force_close(position, f"{label}_FULL", now)
                return

        # 6. Update trailing stop based on latest highest_price_seen (monotonic)
        if position.trailing_active:
            candidate = position.highest_price * (1.0 - cfg.trailing_stop_pct)
            if position.trailing_stop_price is None or candidate > position.trailing_stop_price:
                position.trailing_stop_price = candidate
                logger.debug(
                    "[%s] [TRAIL] %s | high=$%.8f | new_stop=$%.8f",
                    self.name, position.symbol,
                    position.highest_price, position.trailing_stop_price,
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _close_position(
        self,
        position: Position,
        price: float,
        reason: str,
        now: Optional[datetime] = None,
    ) -> None:
        """Sell all remaining, remove from open index, log, and persist."""
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

    def _force_close(self, position: Position, reason: str, now: datetime) -> None:
        """Mark a position closed when remaining_quantity hits zero after partial sells."""
        position.status = "CLOSED"
        position.closed_at = now
        position.sell_reason = reason
        self._portfolio.close_position(position.mint_address)
        if self._db:
            self._db.upsert_position(position)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self, current_prices: dict[str, float] | None = None) -> dict:
        """Per-strategy stats dict consumed by MultiStrategyEngine.print_summary()."""
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


# ===========================================================================
# Infinite Moonbag — progressive ratcheting stop ladder, no time exits
# ===========================================================================

class InfiniteMoonbagRunner(StrategyRunner):
    """
    Overrides evaluate_position() with a progressive monotonic stop ladder.

    Stop ladder (stop_loss_price field — always moves UP, never loosens):

        Default            : stop = entry * 0.60   (−40%)
        highest ≥ 2× entry : stop ≥ entry * 1.00   (breakeven floor)
        highest ≥ 3× entry : stop ≥ entry * 1.50   (+50% floor)
        highest ≥ 5× entry : stop ≥ highest * 0.60 (40% trail)
        highest ≥ 10× entry: stop ≥ highest * 0.70 (30% trail)
        highest ≥ 20× entry: stop ≥ highest * 0.75 (25% trail)
        highest ≥ 50× entry: stop ≥ highest * 0.80 (20% trail)

    Milestones use highest_price_seen (not current price), so the stop never
    reverts when price pulls back below a milestone.

    TP events — tiny de-risks only, bulk of position kept open:
        3×  entry: sell 5%  of original  →  partial_take_profit_hit = True
        10× entry: sell 10% of original  →  tp2_hit = True

    No timeout. No max hold. Position stays open until the stop is hit.

    Persistence: stop_loss_price and highest_price are already stored in the
    DB on every upsert, so on restart the ladder resumes correctly.
    """

    # Stop ladder definition — (price_multiple_of_entry, stop_computation)
    # Processed in order; `max()` with running stop ensures monotonicity.
    _FLOOR_MILESTONES: tuple[tuple[float, float], ...] = (
        (2.0,  1.00),   # once highest ≥ 2×: floor at breakeven
        (3.0,  1.50),   # once highest ≥ 3×: floor at +50%
    )
    _TRAIL_MILESTONES: tuple[tuple[float, float], ...] = (
        (5.0,  0.60),   # once highest ≥ 5×:  keep 60% of high (40% trail)
        (10.0, 0.70),   # once highest ≥ 10×: keep 70% of high (30% trail)
        (20.0, 0.75),   # once highest ≥ 20×: keep 75% of high (25% trail)
        (50.0, 0.80),   # once highest ≥ 50×: keep 80% of high (20% trail)
    )

    def evaluate_position(self, position: Position, current_price: float) -> None:
        if position.status == "CLOSED":
            return

        now = datetime.now(timezone.utc)
        entry = position.entry_price

        # 1. Update highest_price_seen (only moves up)
        if current_price > position.highest_price:
            position.highest_price = current_price
        highest = position.highest_price

        # 2. Check existing stop (ladder value from PREVIOUS tick — monotonic)
        if current_price <= position.stop_loss_price:
            multiple = current_price / entry if entry > 0 else 0
            logger.info(
                "[%s] [STOP_LADDER] %s | price=$%.8f (%.2f×) | stop=$%.8f (%.2f×)",
                self.name, position.symbol,
                current_price, multiple,
                position.stop_loss_price, position.stop_loss_price / entry,
            )
            self._close_position(position, current_price, "STOP_LADDER_EXIT", now)
            return

        # 3. Process TP milestones in ascending order (sell % of ORIGINAL qty)
        for i, tp_level in enumerate(self._cfg.take_profit_levels):
            if i > 1:
                break

            already_hit = position.partial_take_profit_hit if i == 0 else position.tp2_hit
            if already_hit:
                continue
            if current_price < entry * tp_level.multiple:
                continue

            tp_qty = min(
                position.initial_quantity * tp_level.sell_fraction_original,
                position.remaining_quantity,
            )
            if tp_qty < _EPSILON:
                continue

            fraction = tp_qty / position.remaining_quantity
            pnl = (current_price - entry) * tp_qty
            label = f"TP{i + 1}"
            self._exchange.sell_partial(position, fraction, current_price, label)

            if i == 0:
                position.partial_take_profit_hit = True
                position.trailing_active = True
            else:
                position.tp2_hit = True

            logger.info(
                "[%s] [%s] %s %.0f× de-risk | sold %.0f%% of original | "
                "qty=%.6f | pnl=$%+.4f | remaining=%.6f",
                self.name, label, position.symbol, tp_level.multiple,
                tp_level.sell_fraction_original * 100,
                tp_qty, pnl, position.remaining_quantity,
            )
            trade_log.info(
                "%-13s | %-10s | %-44s | price=$%.8f | qty=%.4f | pnl=$%+.4f | strategy=%s",
                label, position.symbol, position.mint_address,
                current_price, tp_qty, pnl, self.name,
            )
            if self._db:
                self._db.upsert_position(position)
                self._db.save_portfolio(self._exchange.portfolio, self.name)
                self._db.log_trade(label, position, current_price, tp_qty, pnl)

            if position.remaining_quantity < _EPSILON:
                self._force_close(position, f"{label}_FULL", now)
                return

        # 4. Recompute progressive stop ladder (uses updated highest, monotonic via max())
        #    Milestones trigger once highest_price_seen crosses the threshold —
        #    they remain active even if current price later drops below the threshold.
        new_stop = position.stop_loss_price  # seed from current (already monotonic)

        for milestone_mult, floor_mult in self._FLOOR_MILESTONES:
            if highest >= entry * milestone_mult:
                new_stop = max(new_stop, entry * floor_mult)

        for milestone_mult, keep_pct in self._TRAIL_MILESTONES:
            if highest >= entry * milestone_mult:
                new_stop = max(new_stop, highest * keep_pct)

        if new_stop != position.stop_loss_price:
            logger.info(
                "[%s] [LADDER] %s | high=%.2f× | stop raised $%.8f → $%.8f (%.2f×)",
                self.name, position.symbol,
                highest / entry,
                position.stop_loss_price, new_stop, new_stop / entry,
            )
        position.stop_loss_price = new_stop  # persisted on every DB upsert ✓

        multiple = current_price / entry if entry > 0 else 0
        logger.info(
            "[%s] [MOONBAG] %-10s | %.2f× | stop=%.2f× | high=%.2f× | tp1=%s | tp2=%s",
            self.name, position.symbol,
            multiple,
            position.stop_loss_price / entry,
            highest / entry,
            "✓" if position.partial_take_profit_hit else "—",
            "✓" if position.tp2_hit else "—",
        )
