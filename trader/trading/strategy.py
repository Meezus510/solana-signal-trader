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

    # Cooldown after closing a position on a mint (optional).
    # If set, signals for the same mint are rejected for this many minutes after
    # the previous position closes — prevents re-entering on a second channel call
    # for a token we just stopped out of.  None = disabled (re-entry always allowed).
    recently_closed_cooldown_minutes: Optional[float] = None

    # Chart-based entry filter (optional — set True for chart-enabled strategies)
    use_chart_filter: bool = False

    # Per-runner pump ratio threshold for the chart filter.
    # Overrides the global PUMP_RATIO_MAX in chart.py so each strategy can
    # have its own sensitivity. Ignored when use_chart_filter=False.
    pump_ratio_max: float = 3.5

    # Reanalysis after a chart SKIP (only meaningful when use_chart_filter=True).
    # When True, a skipped signal is re-checked after a calculated delay and
    # entered if the chart has improved.  When False, a SKIP is final.
    use_reanalyze: bool = False

    # How long to wait (seconds) before re-checking a skipped signal.
    # Three separate delays depending on why the signal was skipped:
    #   reanalyze_pump_delay  — token already pumped too hard (pump_ratio >= max)
    #   reanalyze_vol_delay   — volume dying only
    #   reanalyze_both_delay  — both pump AND dying volume (worst case)
    reanalyze_pump_delay: float = 480.0   # 8 minutes
    reanalyze_vol_delay:  float = 240.0   # 4 minutes
    reanalyze_both_delay: float = 600.0   # 10 minutes

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

    # Hard pre-filter: block signals where wallet_momentum_5m >= this value.
    # wallet_momentum = unique_wallet_5m / unique_wallet_hist_5m; >1 means wallet
    # count is growing fast (already pumping). None = disabled.
    # LOO analysis: threshold=2.102 blocks 9/134 losers with 0 winner misses.
    ml_wallet_momentum_max: Optional[float] = None

    # Hard entry filters — applied to all strategies regardless of use_ml_filter.
    # These are data-derived absolute cuts, not soft ML weights.
    # None = disabled (filter not applied).

    # Reject signals where holder_count > this value.
    # Backtest on 617 entered signals: >1000 holders → 16% heavy-loss rate,
    # >2000 holders → 46% heavy-loss rate vs 4% baseline (<600 holders).
    holder_count_max: Optional[int] = None

    # Late-entry combined filter: reject when BOTH hold simultaneously —
    #   price_change_30m_pct > late_entry_price_chg_30m_max
    #   AND chart_ctx.pump_ratio > late_entry_pump_ratio_min
    # Catches tokens deep into a pump-dump cycle (30m already +500% AND
    # 15-bar pump ratio > 20×). Each alone is too noisy; together they are
    # the clearest late-distribution signature in the dataset.
    # If chart_ctx is unavailable the filter is skipped (fail-open).
    late_entry_price_chg_30m_max: Optional[float] = None
    late_entry_pump_ratio_min: Optional[float] = None

    # Reject signals where 1h buy-volume ratio > this value.
    # buy_vol_ratio_1h = buy_volume_1h / total_volume_1h (0–1).
    # High values mean buying pressure is already exhausted / late-stage.
    # Backtest on 687 trend_rider trades: ratio < 0.016 → 60% win rate at 95% block.
    buy_vol_ratio_1h_max: Optional[float] = None

    # Reject signals where market cap (USD) < this value.
    # Low market cap tokens are highly vulnerable to rug pulls and wash trading.
    # Backtest on 687 trend_rider trades: mc > $314k → 60% win rate at 95% block.
    market_cap_usd_min: Optional[float] = None

    # High-confidence size multiplier — when score >= ml_high_score_threshold,
    # buy size is scaled by ml_size_multiplier (e.g. 2× the normal position).
    # Only applied when use_ml_filter=True.
    ml_high_score_threshold: float = 8.0
    ml_size_multiplier: float = 2.0

    # Maximum-confidence size multiplier — when score >= ml_max_score_threshold,
    # buy size is scaled by ml_max_size_multiplier instead (e.g. 3× the normal).
    ml_max_score_threshold: float = 9.5
    ml_max_size_multiplier: float = 3.0

    # Which strategy's closed outcomes to train the ML scorer on.
    # Defaults to this strategy's own name when None.
    # Chart/ML variants should point to their base strategy so the scorer
    # trains on unbiased, unfiltered outcomes.
    # e.g. quick_pop_managed → "quick_pop"
    ml_training_strategy: Optional[str] = None

    # Which outcome column to use as the KNN training label.
    # "outcome_pnl_pct"      — final exit PnL% (default, exit-price based)
    # "position_peak_pnl_pct" — highest price reached before sell (peak-pump based)
    # quick_pop uses peak so the model learns "did this signal actually pump?"
    # rather than "did our exit happen to be profitable?"
    ml_training_label: str = "outcome_pnl_pct"

    # When True, the KNN scorer uses Birdeye v3 sub-minute candles (15s) instead
    # of Birdeye 1m.  Enable only for fast-scalp strategies (quick_pop) where
    # short-term pump shape matters most.  Trend/moonbag strategies should leave
    # this False — their longer horizon is better captured by 1m candles.
    ml_use_subminute: bool = False

    # KNN algorithm hyperparameters — tunable by the strategy agent.
    # ml_k: number of nearest neighbours (higher = smoother, slower to adapt)
    # ml_halflife_days: recency weight half-life (lower = recent trades matter more)
    # ml_score_low_pct / ml_score_high_pct: PnL% range mapped to scores 0 and 10
    ml_k: int = 5
    ml_halflife_days: float = 14.0
    ml_score_low_pct: float = -35.0
    ml_score_high_pct: float = 85.0

    # Per-feature weights for the KNN scorer (18-element list, one per feature).
    # When set, each feature value is multiplied by its weight before z-score
    # normalisation, giving higher-separability features proportionally more
    # influence on the euclidean distance.  None = uniform weights (default).
    # Feature order matches ml_scorer.py FEAT_NAMES (indices 0–17).
    ml_feature_weights: Optional[tuple[float, ...]] = None

    # Live trading flag — when True, closed trades from this strategy affect the
    # AI balance (tracked via is_live=1 in strategy_outcomes).
    # The strategy tuner agent controls this toggle; it starts False (paper only).
    live_trading: bool = False

    # Per-signal policy agent (optional — requires ANTHROPIC_API_KEY at runtime).
    # When True, Agent A is called before entering each signal and can:
    #   • block the trade entirely (allow_trade=False)
    #   • scale buy size up or down (buy_size_multiplier)
    #   • apply a downward score adjustment for degraded data quality
    # Only meaningful for chart/ML strategies where data quality varies.
    use_policy_agent: bool = False

    # AI override agent (optional — requires ANTHROPIC_API_KEY at runtime).
    # When True, signals filtered out by ML_SKIP, CHART_SKIP, or POLICY_BLK are
    # re-evaluated by Claude Haiku.  The agent sees all chart, ML, and pair-stats
    # data and can override the skip decision or schedule a delayed re-check.
    # Buys triggered by an override are tagged as AI_OVERRIDE_BUY in the logs.
    use_ai_override: bool = False

    # Shadow mode for the AI override agent.
    # When use_ai_override=False AND use_ai_override_shadow=True, the agent is
    # still called on every filtered signal (in a background task, no trade effect).
    # Decisions are recorded as SHADOW_OVERRIDE / SHADOW_REJECT / SHADOW_REANALYZE
    # in ai_override_decisions so you can evaluate agent quality before enabling it.
    use_ai_override_shadow: bool = False

    # When True, price-triggered exits (SL and TP) use a Jupiter v6 quote as the
    # execution price instead of the raw Birdeye price.  The quote reflects real
    # AMM slippage and liquidity depth — the most production-accurate simulation.
    # Requires the engine to be started with monitor_positions_ws() and an http session.
    use_real_exit_price: bool = False

    # Pre-TP1 reversal guard (optional).
    # When set, if the current price drops more than this fraction below the
    # highest price seen since entry — even BEFORE TP1 activates the trailing
    # stop — the position is closed immediately with reason PEAK_DROP.
    # e.g. 0.18 → exit if price falls 18% from its highest point.
    # Only fires while trailing_active is False (i.e., before TP1 is hit).
    # After TP1 the normal trailing_stop_pct takes over.
    peak_drop_exit_pct: Optional[float] = None

    # Early stagnation timeout (optional).
    # Exit early if the coin is truly dead — no meaningful price movement in
    # either direction after a short window.
    #
    # Fires only when ALL three conditions hold simultaneously:
    #   age >= early_timeout_minutes
    #   AND highest_price < entry * (1 + early_timeout_max_gain_pct)  — never pumped above threshold
    #   AND (highest_price - lowest_price) / entry_price < early_timeout_min_range_pct — barely oscillated
    #
    # The range check (high - low) / entry is the key volume proxy: a coin being
    # actively bought and sold will oscillate and produce a large range even if
    # the net move is small (e.g. -5%, +2%, -7%, +1% → range ≈ 9%).
    # Only a genuinely dead coin stays flat on both measures.
    #
    # Only fires while trailing_active is False (TP1 not yet hit).
    early_timeout_minutes: Optional[float] = None
    early_timeout_max_gain_pct: Optional[float] = None
    early_timeout_min_range_pct: Optional[float] = None  # default None = range check disabled

    # Operator / agent emergency gate.
    # When True, the strategy will reject all new entries but continue managing
    # any already-open positions normally.
    block_new_entries: bool = False


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
        # mint → strategy_outcomes row id (populated by engine when save_chart_data=True)
        self._outcome_ids: dict[str, int] = {}
        # mint → signal_charts row id (populated by engine when save_chart_data=True)
        self._signal_chart_ids: dict[str, int] = {}

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

    def set_outcome_id(self, mint: str, outcome_id: int) -> None:
        """Called by the engine after saving a strategy_outcomes row for an entered position."""
        self._outcome_ids[mint] = outcome_id

    def set_signal_chart_id(self, mint: str, signal_chart_id: int) -> None:
        """Called by the engine after saving a signal_charts row for an entered position."""
        self._signal_chart_ids[mint] = signal_chart_id

    def restore_cash(self, available_cash: float, starting_cash: float, total_reloads: float = 0.0) -> None:
        self._exchange.portfolio.available_cash_usd = available_cash
        self._exchange.portfolio.starting_cash_usd = starting_cash
        self._exchange.portfolio.total_reloads_usd = total_reloads

    def restore_positions(self, positions: list[Position]) -> None:
        for pos in positions:
            self._portfolio.add_position(pos)
        if positions:
            logger.info("[%s] Restored %d open position(s)", self.name, len(positions))

    def restore_closed_positions(self, positions: list[Position]) -> None:
        for pos in positions:
            self._portfolio.add_closed_position(pos)
        if positions:
            logger.info("[%s] Restored %d closed position(s)", self.name, len(positions))

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

    def enter_position(
        self,
        signal: TokenSignal,
        entry_price: float,
        chart_ctx=None,
        buy_size_override: Optional[float] = None,
    ) -> Optional[Position]:
        """
        Open a paper position at the pre-fetched price.
        Returns None if skipped (duplicate mint, insufficient cash, or chart filter).

        chart_ctx — ChartContext from trader.analysis.chart, or None.
            Ignored when use_chart_filter=False (non-chart strategies always enter).
            When use_chart_filter=True and chart_ctx.should_enter is False, the
            signal is skipped and logged as CHART_SKIP.
            When use_chart_filter=True but chart_ctx is None (OHLCV fetch failed),
            the position is entered anyway — no data means no filter.

        buy_size_override — if provided, overrides cfg.buy_size_usd for this entry
            (used by the engine to scale up on high ML confidence scores).
        """
        if self._cfg.block_new_entries:
            logger.info("[%s] [BLOCK_ALL] %s — new entries disabled by config", self.name, signal.symbol)
            signal_log.info(
                "BLOCK_ALL  | %-10s | %-44s | ch=%-20s | strategy=%s",
                signal.symbol, signal.mint_address, signal.source_channel, self.name,
            )
            if self._db:
                self._db.log_signal(
                    "BLOCK_ALL", symbol=signal.symbol,
                    mint=signal.mint_address, strategy=self.name,
                    source_channel=signal.source_channel,
                )
            return None

        # Chart filter (only for chart-enabled strategies).
        # Re-evaluated per-runner using this runner's own pump_ratio_max threshold
        # so each strategy can have independent chart filter sensitivity.
        if self._cfg.use_chart_filter and chart_ctx is not None:
            pumped = chart_ctx.pump_ratio >= self._cfg.pump_ratio_max
            vol_dead = chart_ctx.vol_trend == "DYING"
            if pumped or vol_dead:
                if pumped and vol_dead:
                    skip_reason = (
                        f"pump={chart_ctx.pump_ratio:.1f}x >= {self._cfg.pump_ratio_max}x + vol dying"
                    )
                elif pumped:
                    skip_reason = (
                        f"pump={chart_ctx.pump_ratio:.1f}x >= {self._cfg.pump_ratio_max}x"
                    )
                else:
                    skip_reason = f"vol={chart_ctx.vol_trend}"
                logger.info("[%s] [CHART_SKIP] %s — %s", self.name, signal.symbol, skip_reason)
                signal_log.info(
                    "CHART_SKIP | %-10s | %-44s | ch=%-20s | %s | strategy=%s",
                    signal.symbol, signal.mint_address, signal.source_channel,
                    skip_reason, self.name,
                )
                if self._db:
                    self._db.log_signal(
                        "CHART_SKIP", symbol=signal.symbol,
                        mint=signal.mint_address, strategy=self.name,
                        source_channel=signal.source_channel,
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
                    source_channel=signal.source_channel,
                )
            return None

        if self._cfg.recently_closed_cooldown_minutes is not None:
            cooldown_secs = self._cfg.recently_closed_cooldown_minutes * 60.0
            if self._portfolio.closed_within_seconds(signal.mint_address, cooldown_secs):
                logger.info(
                    "[%s] [SKIP] %s closed recently — cooldown %.0fm, signal ignored",
                    self.name, signal.symbol, self._cfg.recently_closed_cooldown_minutes,
                )
                signal_log.info(
                    "COOLDOWN   | %-10s | %-44s | ch=%-20s | strategy=%s",
                    signal.symbol, signal.mint_address, signal.source_channel, self.name,
                )
                if self._db:
                    self._db.log_signal(
                        "COOLDOWN", symbol=signal.symbol,
                        mint=signal.mint_address, strategy=self.name,
                        source_channel=signal.source_channel,
                    )
                return None

        buy_size = buy_size_override if buy_size_override is not None else self._cfg.buy_size_usd
        position = self._exchange.buy(signal, entry_price, buy_size)
        if position is None:
            signal_log.info(
                "NO_CASH    | %-10s | %-44s | ch=%-20s | strategy=%s",
                signal.symbol, signal.mint_address, signal.source_channel, self.name,
            )
            if self._db:
                self._db.log_signal(
                    "NO_CASH", symbol=signal.symbol,
                    mint=signal.mint_address, strategy=self.name,
                    source_channel=signal.source_channel,
                )
            return None

        position.strategy_name = self.name
        position.source_channel = signal.source_channel
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

    def evaluate_position(
        self,
        position: Position,
        current_price: float,
        exit_quote_price: Optional[float] = None,
    ) -> None:
        """
        Standard evaluation used by quick_pop and trend_rider.

        All TP sells use % of ORIGINAL position quantity (not remaining).
        Trailing stop activates after TP1 and is monotonically tightened.

        Key ordering rule: check existing stop BEFORE updating the trailing
        stop, so that same-tick trailing tightening cannot cause same-tick
        unexpected exits. TPs are processed first, trailing state updated after.

        exit_quote_price — when provided (Jupiter v6 quote), price-triggered exits
            (SL and TP) execute at this price instead of current_price, reflecting
            real AMM slippage. Time-based exits (TIMEOUT, MAX_HOLD) always use
            current_price since no trigger order would be in effect.
        """
        if position.status == "CLOSED":
            return

        cfg = self._cfg
        now = datetime.now(timezone.utc)
        opened_at = position.opened_at
        if opened_at.tzinfo is None:
            opened_at = opened_at.replace(tzinfo=timezone.utc)
        age_minutes = (now - opened_at).total_seconds() / 60.0

        # 1. Update highest/lowest price seen
        if current_price > position.highest_price:
            position.highest_price = current_price
            position.highest_price_ts = now
        if current_price < position.lowest_price:
            position.lowest_price = current_price
            position.lowest_price_ts = now

        # 2. Check existing stop (values from previous tick — before any update this tick)
        #    Trailing stop if active, else fixed stop_loss_price.
        current_stop = (
            position.trailing_stop_price
            if position.trailing_active and position.trailing_stop_price is not None
            else position.stop_loss_price
        )
        if current_price <= current_stop:
            reason = "TRAILING_STOP" if position.trailing_active else "STOP_LOSS"
            exec_price = exit_quote_price if exit_quote_price is not None else current_price
            logger.info(
                "[%s] [%s] %s | market=$%.8f | exec=$%.8f | stop=$%.8f",
                self.name, reason, position.symbol, current_price, exec_price, current_stop,
            )
            self._close_position(position, exec_price, reason, now)
            return

        # 2b. Pre-TP1 peak-drop exit — fires before trailing stop is active.
        #     Catches coins that pump then reverse without hitting TP1.
        if (
            cfg.peak_drop_exit_pct is not None
            and not position.trailing_active
            and current_price <= position.highest_price * (1.0 - cfg.peak_drop_exit_pct)
        ):
            exec_price = exit_quote_price if exit_quote_price is not None else current_price
            logger.info(
                "[%s] [PEAK_DROP] %s | market=$%.8f | exec=$%.8f | high=$%.8f | drop=%.1f%%",
                self.name, position.symbol, current_price, exec_price,
                position.highest_price,
                (1.0 - current_price / position.highest_price) * 100,
            )
            self._close_position(position, exec_price, "PEAK_DROP", now)
            return

        # 3. Max hold (unconditional — time-based, no trigger order, use market price)
        if cfg.max_hold_minutes is not None and age_minutes >= cfg.max_hold_minutes:
            logger.info(
                "[%s] [MAX_HOLD] %s | age=%.0fmin >= %.0fmin",
                self.name, position.symbol, age_minutes, cfg.max_hold_minutes,
            )
            self._close_position(position, current_price, "MAX_HOLD", now)
            return

        # 3b. Early stagnation timeout — exit sooner if coin is truly dead.
        #     Only fires before TP1 (trailing not yet active).
        #     Two guards must both pass:
        #       1. highest_price < threshold  — coin never pumped meaningfully
        #       2. price range (high-low)/entry < min_range  — coin barely oscillated
        #          (range is a volume proxy: active coins move in both directions
        #          even if net result is small; new lows/highs expand it each tick)
        if (
            cfg.early_timeout_minutes is not None
            and cfg.early_timeout_max_gain_pct is not None
            and not position.trailing_active
            and age_minutes >= cfg.early_timeout_minutes
            and position.highest_price < position.entry_price * (1.0 + cfg.early_timeout_max_gain_pct)
        ):
            price_range_pct = (position.highest_price - position.lowest_price) / position.entry_price
            if cfg.early_timeout_min_range_pct is None or price_range_pct < cfg.early_timeout_min_range_pct:
                logger.info(
                    "[%s] [EARLY_STALL] %s | age=%.0fmin | peak_gain=%.1f%% | range=%.1f%%",
                    self.name, position.symbol, age_minutes,
                    (position.highest_price / position.entry_price - 1.0) * 100,
                    price_range_pct * 100,
                )
                self._close_position(position, current_price, "EARLY_STALL", now)
                return

        # 4. Timeout (position not gaining enough — time-based, use market price)
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
        exec_price = exit_quote_price if exit_quote_price is not None else current_price
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
            pnl = (exec_price - position.entry_price) * tp_qty
            label = f"TP{i + 1}"
            self._exchange.sell_partial(position, fraction, exec_price, label)

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
                exec_price, tp_qty, pnl, self.name,
            )
            if self._db:
                self._db.upsert_position(position)
                self._db.save_portfolio(self._exchange.portfolio, self.name)
                self._db.log_trade(label, position, exec_price, tp_qty, pnl)

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
            self._flush_outcome(position, reason)

    def _force_close(self, position: Position, reason: str, now: datetime) -> None:
        """Mark a position closed when remaining_quantity hits zero after partial sells."""
        position.status = "CLOSED"
        position.closed_at = now
        position.sell_reason = reason
        self._portfolio.close_position(position.mint_address)
        if self._db:
            self._db.upsert_position(position)
            self._flush_outcome(position, reason)

    def _flush_outcome(self, position: Position, reason: str) -> None:
        """Update strategy_outcomes row with outcome metrics when a position closes."""
        if not self._db:
            return
        outcome_id = self._outcome_ids.pop(position.mint_address, None)
        if outcome_id is None:
            return
        opened_at = position.opened_at
        if opened_at.tzinfo is None:
            opened_at = opened_at.replace(tzinfo=timezone.utc)
        closed_at = position.closed_at or datetime.now(timezone.utc)
        if closed_at.tzinfo is None:
            closed_at = closed_at.replace(tzinfo=timezone.utc)
        hold_secs = (closed_at - opened_at).total_seconds()
        max_gain_pct = (position.highest_price / position.entry_price - 1.0) * 100.0
        trough_pnl_pct = (position.lowest_price / position.entry_price - 1.0) * 100.0
        pnl_pct = (position.realized_pnl_usd / position.usd_size * 100.0) if position.usd_size else 0.0
        peak_ts   = position.highest_price_ts.isoformat() if position.highest_price_ts else None
        trough_ts = position.lowest_price_ts.isoformat() if position.lowest_price_ts else None
        self._db.update_strategy_outcome(
            outcome_id, pnl_pct, reason, hold_secs, max_gain_pct,
            pnl_usd=position.realized_pnl_usd,
            position_peak_price=position.highest_price,
            position_peak_ts=peak_ts,
            position_peak_pnl_pct=max_gain_pct,
            position_trough_price=position.lowest_price,
            position_trough_ts=trough_ts,
            position_trough_pnl_pct=trough_pnl_pct,
        )

        signal_chart_id = self._signal_chart_ids.pop(position.mint_address, None)
        if signal_chart_id is not None:
            now_iso = closed_at.isoformat()
            trough_pnl_pct = (position.lowest_price / position.entry_price - 1.0) * 100.0
            hold_min = hold_secs / 60.0
            self._db.save_price_history(
                signal_chart_id=signal_chart_id,
                peak_price=position.highest_price,
                peak_price_ts=now_iso,
                peak_pnl_pct=max_gain_pct,
                trough_price=position.lowest_price,
                trough_price_ts=now_iso,
                trough_pnl_pct=trough_pnl_pct,
                snapshot_price=position.highest_price,
                snapshot_ts=now_iso,
                price_window_min=max(1, round(hold_min)),
            )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self, current_prices: dict[str, float] | None = None) -> dict:
        """Per-strategy stats dict consumed by MultiStrategyEngine.print_summary()."""
        port = self._exchange.portfolio
        open_pos = self._portfolio.get_open_positions()
        closed_pos = self._portfolio.get_closed_positions()

        unrealized = 0.0
        market_value = 0.0
        for p in open_pos:
            last = (current_prices or {}).get(p.mint_address) or p.last_price
            if last is not None:
                mv = p.remaining_quantity * last
                market_value += mv
                unrealized += mv - (p.remaining_quantity * p.entry_price)

        equity = port.available_cash_usd + market_value
        # Subtract any paper cash reloads so they don't inflate PnL.
        # Each reload injects $1000 into available_cash but represents no real gain.
        net_pnl = equity - port.starting_cash_usd - port.total_reloads_usd
        # Derive realized from equity rather than summing position records —
        # closed positions from prior sessions aren't loaded into memory, so
        # summing p.realized_pnl_usd would undercount historical PnL.
        realized = net_pnl - unrealized
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
            "positions": open_pos,
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
        stop floor = entry × 0.80  (−20% normal)

    Stop ladder (floor multiples of entry, triggered by highest_price_seen):
        highest ≥ 2.5× → stop ≥ entry × 1.65  (+65% locked)
        highest ≥ 4.0× → stop ≥ entry × 2.60  (+160% locked)
        highest ≥ 6.0× → stop ≥ entry × 4.20  (+320% locked)
        (1.8× rung removed — see _STOP_MILESTONES comment)

    All stops are monotonic (only ever raised, never lowered).

    TP ladder (% of ORIGINAL quantity):
        1.8× → sell 20%  (partial_take_profit_hit)
        2.5× → sell 15%  (tp2_hit)
        4.0× → sell 15%  (tp3_hit)
        6.0× → sell 10%  (tp4_hit)

    No timeout. No max hold. Position stays open until the stop is hit.
    """

    _GRACE_SECONDS = 90.0
    _GRACE_FLOOR      = 0.85  # entry × 0.85 during grace  (−15%)
    _POST_GRACE_FLOOR = 0.88  # entry × 0.88 after grace   (−12%)
    # Tightened from −20% → −12%: simulate_tp_sl over 381 trades found tighter
    # floor significantly improves PnL by cutting losers faster.  Grace floor kept
    # at −15% to absorb entry-tick spread noise in the first 90 seconds.

    # Fixed floor milestones — (highest_multiple, stop_floor_multiple_of_entry)
    # Data-driven update 2026-03-23 via optimize_tp_sl.py simulation (381 trades):
    #   • First rung at 1.7×: once price hits +70%, lock stop at 2.2× (+120%).
    #     Captures early pumpers aggressively — largest single PnL improvement.
    #   • Second rung at 4.0×: lock stop at 3.8× (+280%).
    #   • Third rung at 6.5×: lock stop at 5.3× (+430%) for big runners.
    #   Simulated PnL: $+441 vs $+168 production (at $5/trade, 381 trades).
    _STOP_MILESTONES: tuple[tuple[float, float], ...] = (
        (1.7, 2.20),   # highest ≥ 1.7× → stop ≥ entry × 2.20  (+120% locked)
        (4.0, 3.80),   # highest ≥ 4.0× → stop ≥ entry × 3.80  (+280% locked)
        (6.5, 5.30),   # highest ≥ 6.5× → stop ≥ entry × 5.30  (+430% locked)
    )

    # Maps TP index → Position flag attribute name
    _TP_FLAG_ATTRS = (
        "partial_take_profit_hit",
        "tp2_hit",
        "tp3_hit",
        "tp4_hit",
    )

    def evaluate_position(
        self,
        position: Position,
        current_price: float,
        exit_quote_price: Optional[float] = None,
    ) -> None:
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
            exec_price = exit_quote_price if exit_quote_price is not None else current_price
            multiple = current_price / entry if entry > 0 else 0
            logger.info(
                "[%s] [STOP_LADDER] %s | market=$%.8f (%.2f×) | exec=$%.8f | stop=$%.8f (%.2f×)",
                self.name, position.symbol,
                current_price, multiple, exec_price,
                position.stop_loss_price, position.stop_loss_price / entry,
            )
            self._close_position(position, exec_price, "STOP_LADDER_EXIT", now)
            return

        # 3. Process TP milestones in ascending order (sell % of ORIGINAL qty)
        exec_price = exit_quote_price if exit_quote_price is not None else current_price
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
            pnl = (exec_price - entry) * tp_qty
            label = f"TP{i + 1}"
            self._exchange.sell_partial(position, fraction, exec_price, label)
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
                exec_price, tp_qty, pnl, self.name,
            )
            if self._db:
                self._db.upsert_position(position)
                self._db.save_portfolio(self._exchange.portfolio, self.name)
                self._db.log_trade(label, position, exec_price, tp_qty, pnl)

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
