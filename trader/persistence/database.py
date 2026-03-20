"""
trader/persistence/database.py — SQLite persistence layer.

Provides durable storage for positions, trades, signals, and seen Telegram
message IDs so the bot can survive restarts, crashes, and redeployments
without losing state.

Schema
------
positions     — one row per (strategy, mint) pair (upserted on every state change)
trades        — append-only event log (BUY, TP_PARTIAL, STOP_LOSS, TRAILING_STOP)
signals       — append-only log of every Telegram message and its outcome
portfolio     — one row per strategy, tracks cash balance
seen_msg_ids  — set of processed Telegram message IDs (prevents re-entry on restart)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from trader.trading.models import PortfolioState, Position

logger = logging.getLogger(__name__)

_CREATE_POSITIONS = """
CREATE TABLE IF NOT EXISTS positions (
    strategy                TEXT NOT NULL DEFAULT 'default',
    mint                    TEXT NOT NULL,
    symbol                  TEXT NOT NULL,
    status                  TEXT NOT NULL,
    entry_price             REAL NOT NULL,
    initial_quantity        REAL NOT NULL,
    remaining_quantity      REAL NOT NULL,
    usd_size                REAL NOT NULL,
    highest_price           REAL NOT NULL,
    take_profit_price       REAL NOT NULL,
    stop_loss_price         REAL NOT NULL,
    trailing_active         INTEGER NOT NULL DEFAULT 0,
    trailing_stop_pct       REAL NOT NULL,
    trailing_stop_price     REAL,
    realized_pnl_usd        REAL NOT NULL DEFAULT 0,
    total_proceeds_usd      REAL NOT NULL DEFAULT 0,
    total_fees_usd          REAL NOT NULL DEFAULT 0,
    partial_take_profit_hit INTEGER NOT NULL DEFAULT 0,
    tp2_hit                 INTEGER NOT NULL DEFAULT 0,
    tp3_hit                 INTEGER NOT NULL DEFAULT 0,
    tp4_hit                 INTEGER NOT NULL DEFAULT 0,
    sell_reason             TEXT,
    last_price              REAL,
    opened_at               TEXT NOT NULL,
    closed_at               TEXT,
    source_channel          TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (strategy, mint)
)
"""

_CREATE_TRADES = """
CREATE TABLE IF NOT EXISTS trades (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             TEXT NOT NULL,
    strategy       TEXT NOT NULL DEFAULT 'default',
    event          TEXT NOT NULL,
    symbol         TEXT NOT NULL,
    mint           TEXT NOT NULL,
    price          REAL NOT NULL,
    quantity       REAL NOT NULL,
    pnl            REAL NOT NULL,
    source_channel TEXT NOT NULL DEFAULT ''
)
"""

_CREATE_SIGNALS = """
CREATE TABLE IF NOT EXISTS signals (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             TEXT NOT NULL,
    strategy       TEXT,
    msg_id         INTEGER,
    outcome        TEXT NOT NULL,
    symbol         TEXT,
    mint           TEXT,
    source_channel TEXT NOT NULL DEFAULT ''
)
"""

_CREATE_PORTFOLIO = """
CREATE TABLE IF NOT EXISTS portfolio (
    strategy           TEXT PRIMARY KEY,
    available_cash_usd REAL NOT NULL,
    starting_cash_usd  REAL NOT NULL
)
"""

_CREATE_SEEN = """
CREATE TABLE IF NOT EXISTS seen_msg_ids (
    msg_id INTEGER PRIMARY KEY
)
"""

_CREATE_CHART_SNAPSHOTS = """
CREATE TABLE IF NOT EXISTS chart_snapshots (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                   TEXT NOT NULL,       -- UTC ISO timestamp of signal
    strategy             TEXT NOT NULL,       -- e.g. "quick_pop_chart"
    symbol               TEXT NOT NULL,
    mint                 TEXT NOT NULL,
    entry_price          REAL NOT NULL,
    pump_ratio           REAL,                -- current_price / recent_low
    vol_trend            TEXT,                -- "RISING" | "FLAT" | "DYING"
    candle_count         INTEGER,
    candles_json         TEXT NOT NULL,       -- JSON array of OHLCV dicts
    entered              INTEGER NOT NULL DEFAULT 0,  -- 1 if position was opened
    ml_score             REAL,               -- KNN confidence score 0-10 at signal time
    -- outcome fields — filled in when the position closes
    outcome_pnl_pct      REAL,               -- realized_pnl / usd_size * 100
    outcome_sell_reason  TEXT,               -- STOP_LOSS | TP1 | TRAILING_STOP | …
    outcome_hold_secs    REAL,               -- seconds from entry to close
    outcome_max_gain_pct REAL,               -- (highest_price / entry_price - 1) * 100
    closed               INTEGER NOT NULL DEFAULT 0   -- 1 once outcome is written
)
"""

_CREATE_SIGNAL_CHARTS = """
CREATE TABLE IF NOT EXISTS signal_charts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    mint            TEXT NOT NULL,
    entry_price     REAL NOT NULL,
    pump_ratio      REAL,
    vol_trend       TEXT,
    candle_count    INTEGER,
    candles_json    TEXT NOT NULL,
    candles_1m_json TEXT,
    pair_stats_json TEXT,
    ml_score        REAL,
    source_channel  TEXT NOT NULL DEFAULT '',
    peak_price       REAL,
    peak_price_ts    TEXT,
    peak_pnl_pct     REAL,
    trough_price     REAL,
    trough_price_ts  TEXT,
    trough_pnl_pct   REAL,
    snapshot_price   REAL,
    snapshot_ts      TEXT,
    price_window_min    INTEGER NOT NULL DEFAULT 10,
    fetch_attempts      INTEGER NOT NULL DEFAULT 0,
    price_tracking_done   INTEGER NOT NULL DEFAULT 0,
    price_stale_count     INTEGER NOT NULL DEFAULT 0,
    price_checkpoint_price REAL
)
"""

_CREATE_AI_OVERRIDE_DECISIONS = """
CREATE TABLE IF NOT EXISTS ai_override_decisions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT NOT NULL,
    strategy        TEXT NOT NULL,
    signal_chart_id INTEGER REFERENCES signal_charts(id),
    symbol          TEXT NOT NULL,
    mint            TEXT NOT NULL,
    skip_reason     TEXT NOT NULL,
    decision        TEXT NOT NULL,
    ml_score        REAL,
    pump_ratio      REAL,
    vol_trend       TEXT,
    agent_reason    TEXT,
    reanalyze_delay REAL NOT NULL DEFAULT 0
)
"""

_CREATE_STRATEGY_OUTCOMES = """
CREATE TABLE IF NOT EXISTS strategy_outcomes (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_chart_id      INTEGER NOT NULL REFERENCES signal_charts(id),
    strategy             TEXT NOT NULL,
    entered              INTEGER NOT NULL DEFAULT 0,
    outcome_pnl_pct      REAL,
    outcome_pnl_usd      REAL,
    outcome_sell_reason  TEXT,
    outcome_hold_secs    REAL,
    outcome_max_gain_pct REAL,
    closed               INTEGER NOT NULL DEFAULT 0,
    is_live              INTEGER NOT NULL DEFAULT 0,
    source_channel       TEXT NOT NULL DEFAULT '',
    ml_score             REAL,
    is_ai_override       INTEGER NOT NULL DEFAULT 0,
    skip_reason          TEXT
)
"""


class TradeDatabase:
    """
    Thread-safe SQLite wrapper.

    Uses WAL journal mode so reads never block writes (safe for asyncio
    where coroutines may interleave DB access).
    """

    def __init__(self, path: str = "trader.db", read_only: bool = False) -> None:
        self.path = path  # exposed so agents can open read-only connections for queries
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if read_only:
            # URI mode: truly read-only — any accidental write raises OperationalError
            uri = f"file:{Path(path).resolve()}?mode=ro"
            self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False, timeout=30)
        else:
            self._conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._create_tables()
        logger.info("[DB] Opened %s%s", path, " (read-only)" if read_only else "")

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_tables(self) -> None:
        c = self._conn
        c.execute(_CREATE_POSITIONS)
        c.execute(_CREATE_TRADES)
        c.execute(_CREATE_SIGNALS)
        c.execute(_CREATE_PORTFOLIO)
        c.execute(_CREATE_SEEN)
        c.execute(_CREATE_CHART_SNAPSHOTS)
        c.execute(_CREATE_SIGNAL_CHARTS)
        c.execute(_CREATE_STRATEGY_OUTCOMES)
        c.execute(_CREATE_AI_OVERRIDE_DECISIONS)
        self._migrate(c)
        c.commit()

    def _migrate(self, c: sqlite3.Connection) -> None:
        """Add columns introduced after initial schema (safe to re-run)."""
        _LEGACY_CHANNEL = "WizzyTrades"

        for col, definition in [
            ("tp3_hit",        "INTEGER NOT NULL DEFAULT 0"),
            ("tp4_hit",        "INTEGER NOT NULL DEFAULT 0"),
            ("source_channel", "TEXT NOT NULL DEFAULT ''"),
        ]:
            try:
                c.execute(f"ALTER TABLE positions ADD COLUMN {col} {definition}")
                logger.info("[DB] Migrated: added column %s to positions", col)
                if col == "source_channel":
                    c.execute("UPDATE positions SET source_channel = ?", (_LEGACY_CHANNEL,))
                    logger.info("[DB] Backfilled positions.source_channel = '%s'", _LEGACY_CHANNEL)
            except sqlite3.OperationalError:
                pass  # column already exists

        # trades columns added after initial release
        for col, definition in [
            ("source_channel", "TEXT NOT NULL DEFAULT ''"),
        ]:
            try:
                c.execute(f"ALTER TABLE trades ADD COLUMN {col} {definition}")
                logger.info("[DB] Migrated: added column %s to trades", col)
                c.execute("UPDATE trades SET source_channel = ?", (_LEGACY_CHANNEL,))
                logger.info("[DB] Backfilled trades.source_channel = '%s'", _LEGACY_CHANNEL)
            except sqlite3.OperationalError:
                pass  # column already exists

        # signals columns added after initial release
        for col, definition in [
            ("source_channel", "TEXT NOT NULL DEFAULT ''"),
        ]:
            try:
                c.execute(f"ALTER TABLE signals ADD COLUMN {col} {definition}")
                logger.info("[DB] Migrated: added column %s to signals", col)
                c.execute("UPDATE signals SET source_channel = ?", (_LEGACY_CHANNEL,))
                logger.info("[DB] Backfilled signals.source_channel = '%s'", _LEGACY_CHANNEL)
            except sqlite3.OperationalError:
                pass  # column already exists

        # strategy_outcomes columns added after initial release
        for col, definition in [
            ("outcome_pnl_usd", "REAL"),
            ("is_live",         "INTEGER NOT NULL DEFAULT 0"),
            ("source_channel",  "TEXT NOT NULL DEFAULT ''"),
            ("ml_score",        "REAL"),
            ("is_ai_override",  "INTEGER NOT NULL DEFAULT 0"),
            ("skip_reason",     "TEXT"),
        ]:
            try:
                c.execute(f"ALTER TABLE strategy_outcomes ADD COLUMN {col} {definition}")
                logger.info("[DB] Migrated: added column %s to strategy_outcomes", col)
                if col == "source_channel":
                    c.execute("UPDATE strategy_outcomes SET source_channel = ?", (_LEGACY_CHANNEL,))
                    logger.info("[DB] Backfilled strategy_outcomes.source_channel = '%s'", _LEGACY_CHANNEL)
            except sqlite3.OperationalError:
                pass  # column already exists

        # chart_snapshots columns added after initial release
        for col, definition in [
            ("ml_score",        "REAL"),
            ("pair_stats_json", "TEXT"),
            ("candles_1m_json", "TEXT"),
        ]:
            try:
                c.execute(f"ALTER TABLE chart_snapshots ADD COLUMN {col} {definition}")
                logger.info("[DB] Migrated: added column %s to chart_snapshots", col)
            except sqlite3.OperationalError:
                pass  # column already exists

        # signal_charts columns added after initial release
        for col, definition in [
            ("source_channel",  "TEXT NOT NULL DEFAULT ''"),
            ("peak_price",      "REAL"),
            ("peak_price_ts",   "TEXT"),
            ("peak_pnl_pct",    "REAL"),
            ("trough_price",    "REAL"),
            ("trough_price_ts", "TEXT"),
            ("trough_pnl_pct",  "REAL"),
            ("snapshot_price",  "REAL"),
            ("snapshot_ts",     "TEXT"),
            ("price_window_min",    "INTEGER NOT NULL DEFAULT 10"),
            ("fetch_attempts",     "INTEGER NOT NULL DEFAULT 0"),
            ("price_tracking_done",    "INTEGER NOT NULL DEFAULT 0"),
            ("price_stale_count",      "INTEGER NOT NULL DEFAULT 0"),
            ("price_checkpoint_price", "REAL"),
        ]:
            try:
                c.execute(f"ALTER TABLE signal_charts ADD COLUMN {col} {definition}")
                logger.info("[DB] Migrated: added column %s to signal_charts", col)
                if col == "source_channel":
                    c.execute("UPDATE signal_charts SET source_channel = ?", (_LEGACY_CHANNEL,))
                    logger.info("[DB] Backfilled signal_charts.source_channel = '%s'", _LEGACY_CHANNEL)
            except sqlite3.OperationalError:
                pass  # column already exists

        # Migrate existing chart_snapshots rows into signal_charts + strategy_outcomes.
        # Deduplicates signal_charts by (mint, ts rounded to nearest 10s).
        # Safe to re-run: skips mints/ts pairs already present in signal_charts.
        try:
            count = c.execute("SELECT COUNT(*) FROM chart_snapshots").fetchone()[0]
        except sqlite3.OperationalError:
            count = 0

        if count == 0:
            return

        rows = c.execute(
            """
            SELECT id, ts, strategy, symbol, mint, entry_price,
                   pump_ratio, vol_trend, candle_count, candles_json,
                   entered, ml_score, outcome_pnl_pct, outcome_sell_reason,
                   outcome_hold_secs, outcome_max_gain_pct, closed,
                   pair_stats_json, candles_1m_json
              FROM chart_snapshots
             ORDER BY ts ASC
            """
        ).fetchall()

        # mint → rounded_ts_bucket → signal_chart_id
        sc_index: dict[tuple[str, int], int] = {}

        # Pre-load already migrated signal_charts to avoid duplicates on re-run
        existing = c.execute("SELECT id, mint, ts FROM signal_charts").fetchall()
        for sc_id, sc_mint, sc_ts_str in existing:
            try:
                sc_ts = datetime.fromisoformat(sc_ts_str)
                bucket = round(sc_ts.timestamp() / 10) * 10
                sc_index[(sc_mint, bucket)] = sc_id
            except Exception:
                pass

        migrated_sc = migrated_so = 0
        for row in rows:
            (
                _id, ts, strategy, symbol, mint, entry_price,
                pump_ratio, vol_trend, candle_count, candles_json,
                entered, ml_score, outcome_pnl_pct, outcome_sell_reason,
                outcome_hold_secs, outcome_max_gain_pct, closed,
                pair_stats_json, candles_1m_json,
            ) = row

            try:
                ts_dt = datetime.fromisoformat(ts)
            except Exception:
                continue
            bucket = round(ts_dt.timestamp() / 10) * 10
            key = (mint, bucket)

            if key not in sc_index:
                cur = c.execute(
                    """
                    INSERT INTO signal_charts
                        (ts, symbol, mint, entry_price, pump_ratio, vol_trend,
                         candle_count, candles_json, candles_1m_json, pair_stats_json, ml_score)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (ts, symbol, mint, entry_price, pump_ratio, vol_trend,
                     candle_count, candles_json, candles_1m_json, pair_stats_json, ml_score),
                )
                sc_index[key] = cur.lastrowid
                migrated_sc += 1

            sc_id = sc_index[key]
            c.execute(
                """
                INSERT INTO strategy_outcomes
                    (signal_chart_id, strategy, entered,
                     outcome_pnl_pct, outcome_sell_reason,
                     outcome_hold_secs, outcome_max_gain_pct, closed)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                (sc_id, strategy, int(entered or 0),
                 outcome_pnl_pct, outcome_sell_reason,
                 outcome_hold_secs, outcome_max_gain_pct, int(closed or 0)),
            )
            migrated_so += 1

        if migrated_sc or migrated_so:
            logger.info(
                "[DB] Migrated chart_snapshots: %d signal_charts, %d strategy_outcomes",
                migrated_sc, migrated_so,
            )

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def upsert_position(self, position: Position) -> None:
        """Insert or fully replace a position row (called on every state change)."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO positions VALUES (
                :strategy, :mint, :symbol, :status,
                :entry_price, :initial_quantity, :remaining_quantity, :usd_size,
                :highest_price, :take_profit_price, :stop_loss_price,
                :trailing_active, :trailing_stop_pct, :trailing_stop_price,
                :realized_pnl_usd, :total_proceeds_usd, :total_fees_usd,
                :partial_take_profit_hit, :tp2_hit, :tp3_hit, :tp4_hit,
                :sell_reason, :last_price,
                :opened_at, :closed_at, :source_channel
            )
            """,
            {
                "strategy": position.strategy_name,
                "mint": position.mint_address,
                "symbol": position.symbol,
                "status": position.status,
                "entry_price": position.entry_price,
                "initial_quantity": position.initial_quantity,
                "remaining_quantity": position.remaining_quantity,
                "usd_size": position.usd_size,
                "highest_price": position.highest_price,
                "take_profit_price": position.take_profit_price,
                "stop_loss_price": position.stop_loss_price,
                "trailing_active": int(position.trailing_active),
                "trailing_stop_pct": position.trailing_stop_pct,
                "trailing_stop_price": position.trailing_stop_price,
                "realized_pnl_usd": position.realized_pnl_usd,
                "total_proceeds_usd": position.total_proceeds_usd,
                "total_fees_usd": position.total_fees_usd,
                "partial_take_profit_hit": int(position.partial_take_profit_hit),
                "tp2_hit": int(position.tp2_hit),
                "tp3_hit": int(position.tp3_hit),
                "tp4_hit": int(position.tp4_hit),
                "sell_reason": position.sell_reason,
                "last_price": position.last_price,
                "opened_at": position.opened_at.isoformat(),
                "closed_at": position.closed_at.isoformat() if position.closed_at else None,
                "source_channel": position.source_channel,
            },
        )
        self._conn.commit()

    def load_open_positions(self, strategy_name: str = "default") -> list[Position]:
        """Restore all OPEN positions for a given strategy (called on startup)."""
        rows = self._conn.execute(
            """
            SELECT strategy, mint, symbol, status,
                   entry_price, initial_quantity, remaining_quantity, usd_size,
                   highest_price, take_profit_price, stop_loss_price,
                   trailing_active, trailing_stop_pct, trailing_stop_price,
                   realized_pnl_usd, total_proceeds_usd, total_fees_usd,
                   partial_take_profit_hit, tp2_hit, tp3_hit, tp4_hit,
                   sell_reason, last_price, opened_at, closed_at,
                   COALESCE(source_channel, '')
            FROM positions WHERE status = 'OPEN' AND strategy = ?
            """,
            (strategy_name,),
        ).fetchall()
        positions = [self._row_to_position(r) for r in rows]
        logger.info("[DB] Restored %d open position(s) for strategy '%s'", len(positions), strategy_name)
        return positions

    @staticmethod
    def _row_to_position(row: tuple) -> Position:
        (
            strategy, mint, symbol, status,
            entry_price, initial_quantity, remaining_quantity, usd_size,
            highest_price, take_profit_price, stop_loss_price,
            trailing_active, trailing_stop_pct, trailing_stop_price,
            realized_pnl_usd, total_proceeds_usd, total_fees_usd,
            partial_take_profit_hit, tp2_hit, tp3_hit, tp4_hit,
            sell_reason, last_price,
            opened_at, closed_at, source_channel,
        ) = row
        return Position(
            strategy_name=strategy,
            mint_address=mint,
            symbol=symbol,
            status=status,
            entry_price=entry_price,
            initial_quantity=initial_quantity,
            remaining_quantity=remaining_quantity,
            usd_size=usd_size,
            highest_price=highest_price,
            lowest_price=entry_price,  # not persisted — reset to entry on restore
            take_profit_price=take_profit_price,
            stop_loss_price=stop_loss_price,
            trailing_active=bool(trailing_active),
            trailing_stop_pct=trailing_stop_pct,
            trailing_stop_price=trailing_stop_price,
            realized_pnl_usd=realized_pnl_usd,
            total_proceeds_usd=total_proceeds_usd,
            total_fees_usd=total_fees_usd,
            partial_take_profit_hit=bool(partial_take_profit_hit),
            tp2_hit=bool(tp2_hit),
            tp3_hit=bool(tp3_hit),
            tp4_hit=bool(tp4_hit),
            sell_reason=sell_reason,
            last_price=last_price,
            opened_at=datetime.fromisoformat(opened_at),
            closed_at=datetime.fromisoformat(closed_at) if closed_at else None,
            source_channel=source_channel or "",
        )

    # ------------------------------------------------------------------
    # Portfolio cash (per strategy)
    # ------------------------------------------------------------------

    def save_portfolio(self, state: PortfolioState, strategy_name: str = "default") -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO portfolio VALUES (?, ?, ?)",
            (strategy_name, state.available_cash_usd, state.starting_cash_usd),
        )
        self._conn.commit()

    def load_portfolio(self, strategy_name: str = "default") -> Optional[tuple[float, float]]:
        """Returns (available_cash_usd, starting_cash_usd) or None if no prior session."""
        row = self._conn.execute(
            "SELECT available_cash_usd, starting_cash_usd FROM portfolio WHERE strategy = ?",
            (strategy_name,),
        ).fetchone()
        return row  # (available, starting) or None

    # ------------------------------------------------------------------
    # Trade event log
    # ------------------------------------------------------------------

    def log_trade(
        self,
        event: str,
        position: Position,
        price: float,
        quantity: float,
        pnl: float,
    ) -> None:
        self._conn.execute(
            "INSERT INTO trades (ts, strategy, event, symbol, mint, price, quantity, pnl, source_channel) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (
                datetime.now(timezone.utc).isoformat(),
                position.strategy_name,
                event,
                position.symbol,
                position.mint_address,
                price,
                quantity,
                pnl,
                position.source_channel,
            ),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Signal log
    # ------------------------------------------------------------------

    def log_signal(
        self,
        outcome: str,
        msg_id: Optional[int] = None,
        symbol: Optional[str] = None,
        mint: Optional[str] = None,
        strategy: Optional[str] = None,
        source_channel: str = "",
    ) -> None:
        self._conn.execute(
            "INSERT INTO signals (ts, strategy, msg_id, outcome, symbol, mint, source_channel) "
            "VALUES (?,?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), strategy, msg_id, outcome, symbol, mint, source_channel),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Seen message IDs (deduplication across restarts)
    # ------------------------------------------------------------------

    def load_seen_msg_ids(self) -> set[int]:
        rows = self._conn.execute("SELECT msg_id FROM seen_msg_ids").fetchall()
        ids = {r[0] for r in rows}
        logger.info("[DB] Loaded %d seen message ID(s)", len(ids))
        return ids

    def add_seen_msg_id(self, msg_id: int) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO seen_msg_ids (msg_id) VALUES (?)", (msg_id,)
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Normalised chart data (signal_charts + strategy_outcomes)
    # ------------------------------------------------------------------

    def save_signal_chart(
        self,
        symbol: str,
        mint: str,
        entry_price: float,
        candles: list,
        chart_ctx,                    # ChartContext | None
        ml_score: Optional[float] = None,
        pair_stats: Optional[dict] = None,
        candles_1m: Optional[list] = None,
        ts: Optional[str] = None,
        source_channel: str = "",
    ) -> int:
        """
        Persist OHLCV candles + signal metadata once per signal into signal_charts.

        candles      — high-res candles used for ML (e.g. 10s × 100 bars)
        candles_1m   — standard 1m Birdeye candles, saved for future strategy use
        ts           — ISO timestamp for the signal; defaults to now. Pass the
                       original entry time when backfilling so recency weighting
                       in the KNN reflects the true age of the trade.

        Returns the new signal_charts row id so callers can link strategy_outcomes to it.
        """
        def _serialise(bars: list) -> str:
            return json.dumps([
                {"t": c.unix_time, "o": c.open, "h": c.high,
                 "l": c.low, "c": c.close, "v": c.volume}
                for c in bars
            ])

        candles_json    = _serialise(candles)
        candles_1m_json = _serialise(candles_1m) if candles_1m else None
        row_ts = ts if ts else datetime.now(timezone.utc).isoformat()

        cursor = self._conn.execute(
            """
            INSERT INTO signal_charts
                (ts, symbol, mint, entry_price,
                 pump_ratio, vol_trend, candle_count, candles_json,
                 candles_1m_json, pair_stats_json, ml_score, source_channel)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                row_ts,
                symbol,
                mint,
                entry_price,
                chart_ctx.pump_ratio if chart_ctx else None,
                chart_ctx.vol_trend if chart_ctx else None,
                chart_ctx.candle_count if chart_ctx else len(candles),
                candles_json,
                candles_1m_json,
                json.dumps(pair_stats) if pair_stats else None,
                ml_score,
                source_channel,
            ),
        )
        self._conn.commit()
        return cursor.lastrowid

    def save_strategy_outcome(
        self,
        signal_chart_id: int,
        strategy: str,
        entered: bool,
        is_live: bool = False,
        source_channel: str = "",
        ml_score: float | None = None,
        is_ai_override: bool = False,
        skip_reason: str | None = None,
    ) -> int:
        """
        Insert a strategy_outcomes row linked to an existing signal_charts row.

        is_live       — when True this trade counts against the AI balance.
        ml_score      — KNN confidence score at signal time (None if scorer not ready).
        is_ai_override — True when the AI override agent forced this entry; used to
                         compute override win/loss rate independently.

        Returns the new strategy_outcomes row id so the runner can later fill in
        outcome fields when the position closes.
        """
        cursor = self._conn.execute(
            """
            INSERT INTO strategy_outcomes
                (signal_chart_id, strategy, entered, is_live, source_channel, ml_score, is_ai_override, skip_reason)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (signal_chart_id, strategy, int(entered), int(is_live),
             source_channel, ml_score, int(is_ai_override), skip_reason),
        )
        self._conn.commit()
        return cursor.lastrowid

    def save_ai_override_decision(
        self,
        strategy: str,
        signal_chart_id: int | None,
        symbol: str,
        mint: str,
        skip_reason: str,
        decision: str,
        ml_score: float | None,
        pump_ratio: float | None,
        vol_trend: str | None,
        agent_reason: str,
        reanalyze_delay: float = 0.0,
    ) -> None:
        """
        Persist one AI override agent decision for later outcome analysis.

        decision values:
            OVERRIDE          — agent said enter; trade was placed
            REJECT            — agent said skip; no trade
            REANALYZE         — agent requested delayed re-check; no trade yet
            SHADOW_OVERRIDE   — AI off, shadow mode: agent would have overridden
            SHADOW_REJECT     — AI off, shadow mode: agent would have rejected
            SHADOW_REANALYZE  — AI off, shadow mode: agent would have reanalyzed
        """
        self._conn.execute(
            """
            INSERT INTO ai_override_decisions
                (ts, strategy, signal_chart_id, symbol, mint, skip_reason,
                 decision, ml_score, pump_ratio, vol_trend, agent_reason, reanalyze_delay)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                strategy, signal_chart_id, symbol, mint, skip_reason,
                decision, ml_score, pump_ratio, vol_trend, agent_reason, reanalyze_delay,
            ),
        )
        self._conn.commit()

    def query_ai_override_stats(self, strategy: str, base_strategy: str) -> dict:
        """
        Comprehensive AI override performance breakdown across all four scenarios:

        overrides       — actual trades placed by AI override; shows if overrides profit
        rejections      — signals AI rejected; shows missed winners vs saved losses
        shadow_overrides — shadow mode: would-have-overridden vs actual base outcome
        shadow_rejects  — shadow mode: would-have-rejected vs actual base outcome

        For override trades, outcomes come from strategy_outcomes (is_ai_override=1).
        For rejected/shadow signals, outcomes come from the base strategy's outcomes
        on the same signal_chart_id (what quick_pop actually made on those signals).
        """
        c = self._conn

        # -- overrides: actual trade outcomes --------------------------------
        row = c.execute(
            """
            SELECT COUNT(*), SUM(closed),
                   SUM(CASE WHEN closed=1 AND outcome_pnl_pct > 0 THEN 1 ELSE 0 END),
                   AVG(CASE WHEN closed=1 THEN outcome_pnl_pct END),
                   AVG(CASE WHEN closed=1 THEN outcome_max_gain_pct END)
            FROM strategy_outcomes
            WHERE strategy=? AND is_ai_override=1 AND entered=1
            """,
            (strategy,),
        ).fetchone()
        total_ov, closed_ov, wins_ov, avg_pnl_ov, avg_peak_ov = row
        total_ov = total_ov or 0; closed_ov = closed_ov or 0; wins_ov = wins_ov or 0

        # -- rejections: what base strategy made on AI-rejected signals ------
        rej_rows = c.execute(
            """
            SELECT base.outcome_pnl_pct, base.outcome_max_gain_pct,
                   base.outcome_sell_reason, sc.symbol, aod.ml_score
            FROM ai_override_decisions aod
            JOIN signal_charts sc ON sc.id = aod.signal_chart_id
            JOIN strategy_outcomes base ON base.signal_chart_id = aod.signal_chart_id
                AND base.strategy = ? AND base.entered = 1
                AND base.closed = 1 AND base.outcome_pnl_pct IS NOT NULL
            WHERE aod.strategy = ? AND aod.decision = 'REJECT'
            ORDER BY aod.ts DESC
            """,
            (base_strategy, strategy),
        ).fetchall()

        rej_pnls    = [r[0] for r in rej_rows]
        rej_peaks   = [r[1] for r in rej_rows if r[1] is not None]
        rej_winners = sum(1 for p in rej_pnls if p > 0)

        # -- shadow decisions: what base strategy made on shadow calls -------
        shad_rows = c.execute(
            """
            SELECT aod.decision, base.outcome_pnl_pct, base.outcome_max_gain_pct,
                   sc.symbol, aod.ml_score, aod.skip_reason
            FROM ai_override_decisions aod
            JOIN signal_charts sc ON sc.id = aod.signal_chart_id
            JOIN strategy_outcomes base ON base.signal_chart_id = aod.signal_chart_id
                AND base.strategy = ? AND base.entered = 1
                AND base.closed = 1 AND base.outcome_pnl_pct IS NOT NULL
            WHERE aod.strategy = ? AND aod.decision IN ('SHADOW_OVERRIDE','SHADOW_REJECT')
            ORDER BY aod.ts DESC
            """,
            (base_strategy, strategy),
        ).fetchall()

        shad_override_pnls = [r[1] for r in shad_rows if r[0] == 'SHADOW_OVERRIDE']
        shad_reject_pnls   = [r[1] for r in shad_rows if r[0] == 'SHADOW_REJECT']

        def _stats(pnls: list) -> dict:
            if not pnls:
                return {"count": 0, "win_rate_pct": None, "avg_pnl_pct": None}
            wins = sum(1 for p in pnls if p > 0)
            return {
                "count":        len(pnls),
                "win_rate_pct": round(wins / len(pnls) * 100, 1),
                "avg_pnl_pct":  round(sum(pnls) / len(pnls), 2),
            }

        return {
            "overrides": {
                "total":          total_ov,
                "closed":         closed_ov,
                "wins":           wins_ov,
                "win_rate_pct":   round(wins_ov / closed_ov * 100, 1) if closed_ov else None,
                "avg_pnl_pct":    round(avg_pnl_ov, 2) if avg_pnl_ov is not None else None,
                "avg_peak_pct":   round(avg_peak_ov, 2) if avg_peak_ov is not None else None,
            },
            "rejections": {
                "total":             c.execute(
                    "SELECT COUNT(*) FROM ai_override_decisions WHERE strategy=? AND decision='REJECT'",
                    (strategy,)).fetchone()[0],
                "base_tracked":      len(rej_rows),
                "would_have_won":    rej_winners,
                "win_rate_pct":      round(rej_winners / len(rej_rows) * 100, 1) if rej_rows else None,
                "avg_base_pnl_pct":  round(sum(rej_pnls) / len(rej_pnls), 2) if rej_pnls else None,
                "avg_base_peak_pct": round(sum(rej_peaks) / len(rej_peaks), 2) if rej_peaks else None,
                "sample": [
                    {"symbol": r[3], "pnl_pct": round(r[0], 2),
                     "peak_pct": round(r[1], 2) if r[1] else None,
                     "sell_reason": r[2], "ml_score": r[4]}
                    for r in rej_rows[:10]
                ],
            },
            "shadow_overrides": _stats(shad_override_pnls),
            "shadow_rejects":   _stats(shad_reject_pnls),
        }

    def load_chart_snapshots(self, strategy: str) -> list[dict]:
        """
        Return all closed outcomes for a strategy as a list of dicts.
        JOINs signal_charts + strategy_outcomes; returns the same dict shape as
        the legacy chart_snapshots query for backward compatibility with ChartMLScorer.
        """
        rows = self._conn.execute(
            """
            SELECT sc.ts, sc.candles_json, so.outcome_pnl_pct,
                   sc.pump_ratio, sc.vol_trend, sc.pair_stats_json,
                   COALESCE(so.source_channel, sc.source_channel, '')
              FROM strategy_outcomes so
              JOIN signal_charts sc ON so.signal_chart_id = sc.id
             WHERE so.strategy = ?
               AND so.closed = 1
               AND so.outcome_pnl_pct IS NOT NULL
             ORDER BY sc.ts ASC
            """,
            (strategy,),
        ).fetchall()
        return [
            {
                "ts":              r[0],
                "candles_json":    r[1],
                "outcome_pnl_pct": r[2],
                "pump_ratio":      r[3],
                "vol_trend":       r[4],
                "pair_stats_json": r[5],
                "source_channel":  r[6],
            }
            for r in rows
        ]

    def update_strategy_outcome(
        self,
        outcome_id: int,
        pnl_pct: float,
        sell_reason: str,
        hold_secs: float,
        max_gain_pct: float,
        pnl_usd: Optional[float] = None,
    ) -> None:
        """Fill in outcome fields on a strategy_outcomes row once the position closes."""
        self._conn.execute(
            """
            UPDATE strategy_outcomes
               SET outcome_pnl_pct=?, outcome_pnl_usd=?, outcome_sell_reason=?,
                   outcome_hold_secs=?, outcome_max_gain_pct=?, closed=1
             WHERE id=?
            """,
            (pnl_pct, pnl_usd, sell_reason, hold_secs, max_gain_pct, outcome_id),
        )
        self._conn.commit()

    def save_price_history(
        self,
        signal_chart_id: int,
        peak_price: Optional[float],
        peak_price_ts: Optional[str],
        peak_pnl_pct: Optional[float],
        trough_price: Optional[float],
        trough_price_ts: Optional[str],
        trough_pnl_pct: Optional[float],
        snapshot_price: Optional[float],
        snapshot_ts: Optional[str],
        price_window_min: int = 10,
    ) -> None:
        """Write post-signal price history to a signal_charts row."""
        self._conn.execute(
            """
            UPDATE signal_charts
               SET peak_price=?, peak_price_ts=?, peak_pnl_pct=?,
                   trough_price=?, trough_price_ts=?, trough_pnl_pct=?,
                   snapshot_price=?, snapshot_ts=?, price_window_min=?
             WHERE id=?
            """,
            (
                peak_price, peak_price_ts, peak_pnl_pct,
                trough_price, trough_price_ts, trough_pnl_pct,
                snapshot_price, snapshot_ts, price_window_min,
                signal_chart_id,
            ),
        )
        self._conn.commit()

    def advance_price_watermark(
        self,
        signal_chart_id: int,
        peak_price: float,
        peak_price_ts: str,
        peak_pnl_pct: float,
        trough_price: float,
        trough_price_ts: str,
        trough_pnl_pct: float,
        snapshot_price: float,
        snapshot_ts: str,
        new_window_min: int,
        stale_count: int,
        done: bool,
        checkpoint_price: Optional[float] = None,
    ) -> None:
        """
        Update price history watermark after each subsequent 10-min fetch.
        Only replaces peak/trough if the new value is a better extreme.
        checkpoint_price: if provided, overwrites price_checkpoint_price (set at each 24h boundary).
        """
        self._conn.execute(
            """
            UPDATE signal_charts
               SET peak_price       = CASE WHEN ? > COALESCE(peak_price,   -1e18) THEN ? ELSE peak_price   END,
                   peak_price_ts    = CASE WHEN ? > COALESCE(peak_price,   -1e18) THEN ? ELSE peak_price_ts END,
                   peak_pnl_pct     = CASE WHEN ? > COALESCE(peak_price,   -1e18) THEN ? ELSE peak_pnl_pct  END,
                   trough_price     = CASE WHEN ? < COALESCE(trough_price,  1e18) THEN ? ELSE trough_price   END,
                   trough_price_ts  = CASE WHEN ? < COALESCE(trough_price,  1e18) THEN ? ELSE trough_price_ts END,
                   trough_pnl_pct   = CASE WHEN ? < COALESCE(trough_price,  1e18) THEN ? ELSE trough_pnl_pct  END,
                   snapshot_price        = ?,
                   snapshot_ts           = ?,
                   price_window_min      = ?,
                   price_stale_count     = ?,
                   price_tracking_done   = ?,
                   price_checkpoint_price = CASE WHEN ? IS NOT NULL THEN ? ELSE price_checkpoint_price END
             WHERE id = ?
            """,
            (
                # peak comparisons
                peak_price, peak_price,
                peak_price, peak_price_ts,
                peak_price, peak_pnl_pct,
                # trough comparisons
                trough_price, trough_price,
                trough_price, trough_price_ts,
                trough_price, trough_pnl_pct,
                # snapshot + watermark
                snapshot_price, snapshot_ts,
                new_window_min, stale_count, 1 if done else 0,
                # checkpoint (only written when provided)
                checkpoint_price, checkpoint_price,
                signal_chart_id,
            ),
        )
        self._conn.commit()

    def increment_fetch_attempts(self, signal_chart_id: int) -> int:
        """
        Increment fetch_attempts for a signal_charts row and return the new count.
        Used by price_history.py to track dead coins (no Birdeye candles returned).
        """
        row = self._conn.execute(
            "UPDATE signal_charts SET fetch_attempts = fetch_attempts + 1 WHERE id = ? "
            "RETURNING fetch_attempts",
            (signal_chart_id,),
        ).fetchone()
        self._conn.commit()
        return row[0] if row else 1

    def get_ai_balance(self, start_usd: float = 1000.0) -> float:
        """
        Compute current AI balance as starting capital plus sum of all closed
        live-trade PnL (outcome_pnl_usd WHERE is_live=1).

        Returns start_usd when no live trades have closed yet.
        """
        row = self._conn.execute(
            """
            SELECT COALESCE(SUM(outcome_pnl_usd), 0.0)
              FROM strategy_outcomes
             WHERE is_live = 1
               AND closed  = 1
               AND entered = 1
               AND outcome_pnl_usd IS NOT NULL
            """,
        ).fetchone()
        cumulative_pnl = row[0] if row else 0.0
        return start_usd + cumulative_pnl

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()
        logger.info("[DB] Connection closed")
