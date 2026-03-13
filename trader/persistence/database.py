"""
trader/persistence/database.py — SQLite persistence layer.

Provides durable storage for positions, trades, signals, and seen Telegram
message IDs so the bot can survive restarts, crashes, and redeployments
without losing state.

Schema
------
positions     — one row per open/closed position (upserted on every state change)
trades        — append-only event log (BUY, TP_PARTIAL, STOP_LOSS, TRAILING_STOP)
signals       — append-only log of every Telegram message and its outcome
portfolio     — single-row cash balance (updated after every trade)
seen_msg_ids  — set of processed Telegram message IDs (prevents re-entry on restart)
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from trader.trading.models import PortfolioState, Position

logger = logging.getLogger(__name__)

_CREATE_POSITIONS = """
CREATE TABLE IF NOT EXISTS positions (
    mint                    TEXT PRIMARY KEY,
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
    sell_reason             TEXT,
    last_price              REAL,
    opened_at               TEXT NOT NULL,
    closed_at               TEXT
)
"""

_CREATE_TRADES = """
CREATE TABLE IF NOT EXISTS trades (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       TEXT NOT NULL,
    event    TEXT NOT NULL,
    symbol   TEXT NOT NULL,
    mint     TEXT NOT NULL,
    price    REAL NOT NULL,
    quantity REAL NOT NULL,
    pnl      REAL NOT NULL
)
"""

_CREATE_SIGNALS = """
CREATE TABLE IF NOT EXISTS signals (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      TEXT NOT NULL,
    msg_id  INTEGER,
    outcome TEXT NOT NULL,
    symbol  TEXT,
    mint    TEXT
)
"""

_CREATE_PORTFOLIO = """
CREATE TABLE IF NOT EXISTS portfolio (
    id                 INTEGER PRIMARY KEY CHECK (id = 1),
    available_cash_usd REAL NOT NULL,
    starting_cash_usd  REAL NOT NULL
)
"""

_CREATE_SEEN = """
CREATE TABLE IF NOT EXISTS seen_msg_ids (
    msg_id INTEGER PRIMARY KEY
)
"""


class TradeDatabase:
    """
    Thread-safe SQLite wrapper.

    Uses WAL journal mode so reads never block writes (safe for asyncio
    where coroutines may interleave DB access).
    """

    def __init__(self, path: str = "trader.db") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()
        logger.info("[DB] Opened %s", path)

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
        c.commit()

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def upsert_position(self, position: Position) -> None:
        """Insert or fully replace a position row (called on every state change)."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO positions VALUES (
                :mint, :symbol, :status,
                :entry_price, :initial_quantity, :remaining_quantity, :usd_size,
                :highest_price, :take_profit_price, :stop_loss_price,
                :trailing_active, :trailing_stop_pct, :trailing_stop_price,
                :realized_pnl_usd, :total_proceeds_usd, :total_fees_usd,
                :partial_take_profit_hit, :sell_reason, :last_price,
                :opened_at, :closed_at
            )
            """,
            {
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
                "sell_reason": position.sell_reason,
                "last_price": position.last_price,
                "opened_at": position.opened_at.isoformat(),
                "closed_at": position.closed_at.isoformat() if position.closed_at else None,
            },
        )
        self._conn.commit()

    def load_open_positions(self) -> list[Position]:
        """Restore all OPEN positions from the DB (called on startup)."""
        rows = self._conn.execute(
            "SELECT * FROM positions WHERE status = 'OPEN'"
        ).fetchall()
        positions = [self._row_to_position(r) for r in rows]
        logger.info("[DB] Restored %d open position(s)", len(positions))
        return positions

    @staticmethod
    def _row_to_position(row: tuple) -> Position:
        (
            mint, symbol, status,
            entry_price, initial_quantity, remaining_quantity, usd_size,
            highest_price, take_profit_price, stop_loss_price,
            trailing_active, trailing_stop_pct, trailing_stop_price,
            realized_pnl_usd, total_proceeds_usd, total_fees_usd,
            partial_take_profit_hit, sell_reason, last_price,
            opened_at, closed_at,
        ) = row
        return Position(
            mint_address=mint,
            symbol=symbol,
            status=status,
            entry_price=entry_price,
            initial_quantity=initial_quantity,
            remaining_quantity=remaining_quantity,
            usd_size=usd_size,
            highest_price=highest_price,
            take_profit_price=take_profit_price,
            stop_loss_price=stop_loss_price,
            trailing_active=bool(trailing_active),
            trailing_stop_pct=trailing_stop_pct,
            trailing_stop_price=trailing_stop_price,
            realized_pnl_usd=realized_pnl_usd,
            total_proceeds_usd=total_proceeds_usd,
            total_fees_usd=total_fees_usd,
            partial_take_profit_hit=bool(partial_take_profit_hit),
            sell_reason=sell_reason,
            last_price=last_price,
            opened_at=datetime.fromisoformat(opened_at),
            closed_at=datetime.fromisoformat(closed_at) if closed_at else None,
        )

    # ------------------------------------------------------------------
    # Portfolio cash
    # ------------------------------------------------------------------

    def save_portfolio(self, state: PortfolioState) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO portfolio VALUES (1, ?, ?)",
            (state.available_cash_usd, state.starting_cash_usd),
        )
        self._conn.commit()

    def load_portfolio(self) -> Optional[tuple[float, float]]:
        """Returns (available_cash_usd, starting_cash_usd) or None if no prior session."""
        row = self._conn.execute(
            "SELECT available_cash_usd, starting_cash_usd FROM portfolio WHERE id = 1"
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
            "INSERT INTO trades (ts, event, symbol, mint, price, quantity, pnl) VALUES (?,?,?,?,?,?,?)",
            (
                datetime.now(timezone.utc).isoformat(),
                event,
                position.symbol,
                position.mint_address,
                price,
                quantity,
                pnl,
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
    ) -> None:
        self._conn.execute(
            "INSERT INTO signals (ts, msg_id, outcome, symbol, mint) VALUES (?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), msg_id, outcome, symbol, mint),
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
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()
        logger.info("[DB] Connection closed")
