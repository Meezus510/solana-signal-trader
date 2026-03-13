"""
trader/trading/portfolio.py — In-memory position registry.

PortfolioManager is the single source of truth for all open and closed
positions. It is intentionally free of strategy logic — it only stores
and retrieves positions. All strategy decisions live in TradingEngine.
"""

from __future__ import annotations

import logging
from typing import Optional

from trader.trading.models import Position

logger = logging.getLogger(__name__)


class PortfolioManager:
    """
    Registry of all positions keyed by mint address.

    Separation between _all (full history) and _open_by_mint (active index)
    means O(1) duplicate-entry checks while preserving closed-trade history
    for PnL reporting and later persistence.

    Duplicate-entry guard:
        If the same mint address arrives while a position is open, has_open_position()
        returns True and TradingEngine skips opening a second position. After a
        position closes, the mint is removed from the open index and re-entry is allowed.
    """

    def __init__(self) -> None:
        self._all: list[Position] = []                   # full trade history
        self._open_by_mint: dict[str, Position] = {}     # fast open-position lookup

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def add_position(self, position: Position) -> None:
        """Register a newly opened position."""
        self._all.append(position)
        self._open_by_mint[position.mint_address] = position
        logger.debug("Position added: %s | mint=%s", position.symbol, position.mint_address)

    def close_position(self, mint_address: str) -> None:
        """
        Remove a position from the open-position index.

        The position object itself is retained in _all for history and PnL reporting.
        Called by TradingEngine after PaperExchange.sell_all() closes a trade.
        """
        removed = self._open_by_mint.pop(mint_address, None)
        if removed:
            logger.debug("Position removed from open index: %s", mint_address)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_position(self, mint_address: str) -> Optional[Position]:
        """Return the currently open position for this mint, or None."""
        return self._open_by_mint.get(mint_address)

    def has_open_position(self, mint_address: str) -> bool:
        return mint_address in self._open_by_mint

    def get_open_positions(self) -> list[Position]:
        return list(self._open_by_mint.values())

    def get_closed_positions(self) -> list[Position]:
        return [p for p in self._all if p.status == "CLOSED"]

    def all_positions(self) -> list[Position]:
        return list(self._all)

    # ------------------------------------------------------------------
    # Stats (convenience — used by TradingEngine.print_summary)
    # ------------------------------------------------------------------

    @property
    def open_count(self) -> int:
        return len(self._open_by_mint)

    @property
    def closed_count(self) -> int:
        return len(self._all) - self.open_count

    @property
    def total_count(self) -> int:
        return len(self._all)
