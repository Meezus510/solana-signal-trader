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

Order of operations per price tick (InfiniteMoonbagRunner / v2):
    1. Update highest_price_seen if new high
    2. Check existing stop (stop_loss_price = progressive ladder from previous tick)
    3. Process TP milestones in ascending order — each sells % of ORIGINAL qty (4 levels)
    4. Recompute stop: grace-period baseline + floor milestones (always monotonic)
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

    # Chart-based entry filter (optional — set True for chart-enabled strategies)
    use_chart_filter: bool = False

    # Reanalysis after a chart SKIP (only meaningful when use_chart_filter=True).
    # When True, a skipped signal is re-checked after a calculated delay and
    # entered if the chart has improved.  When False, a SKIP is final.
    use_reanalyze: bool = False

    # Save chart snapshots for ML training (optional).
    # When True, every incoming signal's OHLCV candles + outcome metrics are
    # persisted to chart_snapshots.  Does not affect entry decisions.
    save_chart_data: bool = False

    # ML-based entry filter (optional — requires save_chart_data=True on the same
    # strategy so training data accumulates).
    # When True, signals with an ML confidence score below ml_min_score are skipped.
    # If the scorer returns None (fewer than MIN_SAMPLES closed examples), the signal
    # is always allowed through so the bot keeps accumulating training data.
    use_ml_filter: bool = False
    ml_min_score: float = 5.0


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
        # mint → chart_snapshot row id (populated by engine when save_chart_data=True)
        self._chart_snapshot_ids: dict[str, int] = {}

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

    def set_chart_snapshot_id(self, mint: str, snapshot_id: int) -> None:
        """Called by the engine after saving a chart snapshot for an entered position."""
        self._chart_snapshot_ids[mint] = snapshot_id

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

    def enter_position(self, signal: TokenSignal, entry_price: float, chart_ctx=None) -> Optional[Position]:
        """
        Open a paper position at the pre-fetched price.
        Returns None if skipped (duplicate mint, insufficient cash, or chart filter).

        chart_ctx — ChartContext from trader.analysis.chart, or None.
            Ignored when use_chart_filter=False (non-chart strategies always enter).
            When use_chart_filter=True and chart_ctx.should_enter is False, the
            signal is skipped and logged as CHART_SKIP.
            When use_chart_filter=True but chart_ctx is None (OHLCV fetch failed),
            the position is entered anyway — no data means no filter.
        """
        # Chart filter (only for chart-enabled strategies)
        if self._cfg.use_chart_filter and chart_ctx is not None and not chart_ctx.should_enter:
            logger.info(
                "[%s] [CHART_SKIP] %s — %s",
                self.name, signal.symbol, chart_ctx.reason,
            )
            signal_log.info(
                "CHART_SKIP | %-10s | %-44s | ch=%-20s | %s | strategy=%s",
                signal.symbol, signal.mint_address, signal.source_channel,
                chart_ctx.reason, self.name,
            )
            if self._db:
                self._db.log_signal(
                    "CHART_SKIP", symbol=signal.symbol,
                    mint=signal.mint_address, strategy=self.name,
                )
            return None

        if self._portfolio.has_open_position(signal.mint_address):
            logger.info(
                "[%s] [SKIP] Already holding %s — duplicate signal ignored",
                self.name, signal.symbol,
            )
            signal_log.info(
                "DUPLICATE  | %-10s | %-44s | ch=%-20s | strategy=%s",
                signal.symbol, signal.mint_address, signal.source_channel, self.name,
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
                "NO_CASH    | %-10s | %-44s | ch=%-20s | strategy=%s",
                signal.symbol, signal.mint_address, signal.source_channel, self.name,
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
            self._flush_chart_snapshot(position, reason)

    def _force_close(self, position: Position, reason: str, now: datetime) -> None:
        """Mark a position closed when remaining_quantity hits zero after partial sells."""
        position.status = "CLOSED"
        position.closed_at = now
        position.sell_reason = reason
        self._portfolio.close_position(position.mint_address)
        if self._db:
            self._db.upsert_position(position)
            self._flush_chart_snapshot(position, reason)

    def _flush_chart_snapshot(self, position: Position, reason: str) -> None:
        """Update chart snapshot outcome metrics when a position closes."""
        if not self._cfg.save_chart_data or not self._db:
            return
        snapshot_id = self._chart_snapshot_ids.pop(position.mint_address, None)
        if snapshot_id is None:
            return
        opened_at = position.opened_at
        if opened_at.tzinfo is None:
            opened_at = opened_at.replace(tzinfo=timezone.utc)
        closed_at = position.closed_at or datetime.now(timezone.utc)
        if closed_at.tzinfo is None:
            closed_at = closed_at.replace(tzinfo=timezone.utc)
        hold_secs = (closed_at - opened_at).total_seconds()
        max_gain_pct = (position.highest_price / position.entry_price - 1.0) * 100.0
        pnl_pct = (position.realized_pnl_usd / position.usd_size * 100.0) if position.usd_size else 0.0
        self._db.update_chart_snapshot_outcome(snapshot_id, pnl_pct, reason, hold_secs, max_gain_pct)

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
# Infinite Moonbag v2 — grace-period floor + progressive stop ladder
# ===========================================================================

class InfiniteMoonbagRunner(StrategyRunner):
    """
    infinite_moonbag_v2: grace-period catastrophic stop → normal floor → ratcheting ladder.

    Grace period (first 90s after entry):
        stop floor = entry × 0.70  (−30% catastrophic)

    After grace:
        stop floor = entry × 0.78  (−22% normal)

    Stop ladder (floor multiples of entry, triggered by highest_price_seen):
        highest ≥ 1.8× → stop ≥ entry × 1.35
        highest ≥ 2.5× → stop ≥ entry × 1.90
        highest ≥ 4.0× → stop ≥ entry × 2.80
        highest ≥ 6.0× → stop ≥ entry × 3.50

    All stops are monotonic (only ever raised, never lowered).

    TP ladder (% of ORIGINAL quantity):
        1.8× → sell 20%  (partial_take_profit_hit)
        2.5× → sell 15%  (tp2_hit)
        4.0× → sell 15%  (tp3_hit)
        6.0× → sell 10%  (tp4_hit)

    No timeout. No max hold. Position stays open until the stop is hit.
    """

    _GRACE_SECONDS = 90.0
    _GRACE_FLOOR      = 0.70  # entry × 0.70 during grace  (−30%)
    _POST_GRACE_FLOOR = 0.78  # entry × 0.78 after grace   (−22%)

    # Fixed floor milestones — (highest_multiple, stop_floor_multiple_of_entry)
    _STOP_MILESTONES: tuple[tuple[float, float], ...] = (
        (1.8, 1.35),   # highest ≥ 1.8× → stop ≥ entry × 1.35
        (2.5, 1.90),   # highest ≥ 2.5× → stop ≥ entry × 1.90
        (4.0, 2.80),   # highest ≥ 4.0× → stop ≥ entry × 2.80
        (6.0, 3.50),   # highest ≥ 6.0× → stop ≥ entry × 3.50
    )

    # Maps TP index → Position flag attribute name
    _TP_FLAG_ATTRS = (
        "partial_take_profit_hit",
        "tp2_hit",
        "tp3_hit",
        "tp4_hit",
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
            if i >= len(self._TP_FLAG_ATTRS):
                break

            flag_attr = self._TP_FLAG_ATTRS[i]
            if getattr(position, flag_attr):
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
            setattr(position, flag_attr, True)

            logger.info(
                "[%s] [%s] %s %.1f× de-risk | sold %.0f%% of original | "
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

        # 4. Recompute stop (grace period baseline + progressive floor ladder)
        opened_at = position.opened_at
        if opened_at.tzinfo is None:
            opened_at = opened_at.replace(tzinfo=timezone.utc)
        age_seconds = (now - opened_at).total_seconds()

        new_stop = position.stop_loss_price  # seed — already monotonic

        # Baseline floor: catastrophic during grace, normal after
        if age_seconds <= self._GRACE_SECONDS:
            new_stop = max(new_stop, entry * self._GRACE_FLOOR)
        else:
            new_stop = max(new_stop, entry * self._POST_GRACE_FLOOR)

        # Progressive floor milestones (trigger once highest crosses threshold)
        for milestone_mult, floor_mult in self._STOP_MILESTONES:
            if highest >= entry * milestone_mult:
                new_stop = max(new_stop, entry * floor_mult)

        if new_stop != position.stop_loss_price:
            logger.info(
                "[%s] [LADDER] %s | high=%.2f× | stop raised $%.8f → $%.8f (%.2f×)",
                self.name, position.symbol,
                highest / entry,
                position.stop_loss_price, new_stop, new_stop / entry,
            )
        position.stop_loss_price = new_stop  # persisted on every DB upsert ✓

        multiple = current_price / entry if entry > 0 else 0
        tp_flags = "".join(
            "✓" if getattr(position, a) else "—"
            for a in self._TP_FLAG_ATTRS
        )
        logger.info(
            "[%s] [MOONBAG] %-10s | %.2f× | stop=%.2f× | high=%.2f× | tp=%s",
            self.name, position.symbol,
            multiple,
            position.stop_loss_price / entry,
            highest / entry,
            tp_flags,
        )
