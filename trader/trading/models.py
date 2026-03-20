"""
trader/trading/models.py — Core domain models.

Pure data containers with no business logic. Using frozen=False on Position
so TradingEngine can update strategy state (highest_price, trailing_stop_price,
etc.) in-place without constructing new objects on every price tick.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class TokenSignal:
    """
    Represents a validated first-call token signal from the listener pipeline.

    Produced by trader.listener.parser and consumed by TradingEngine.
    """
    symbol: str
    mint_address: str
    chain: str = "solana"
    detected_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    source_channel: str = ""   # Telegram channel username the signal came from


@dataclass
class Position:
    """
    Complete lifecycle state of a single paper trade.

    Fields are mutated in-place by TradingEngine.evaluate_position() and
    PaperExchange as prices move and strategy rules fire.

    PnL accounting:
        realized_pnl_usd accumulates across all sell events (partial TP
        sell + final trailing/stop sell). total_proceeds_usd tracks the
        gross USD received from all sells before netting against cost basis.
    """
    # Identity
    symbol: str
    mint_address: str

    # Entry
    entry_price: float
    initial_quantity: float       # tokens bought at open
    remaining_quantity: float     # decrements on each sell
    usd_size: float               # USD capital deployed

    # Lifecycle
    status: str                   # "OPEN" | "CLOSED"
    opened_at: datetime
    closed_at: Optional[datetime]

    # Strategy state
    highest_price: float                  # running max — drives trailing stop
    highest_price_ts: Optional[datetime]  # when highest_price was first reached
    lowest_price: float                   # running min — tracked for training data
    lowest_price_ts: Optional[datetime]   # when lowest_price was first reached
    take_profit_price: float      # entry_price × take_profit_multiple
    stop_loss_price: float        # entry_price × (1 − stop_loss_pct)
    trailing_active: bool         # True after partial TP sell fires
    trailing_stop_pct: float
    trailing_stop_price: Optional[float]  # highest_price × (1 − trailing_stop_pct)

    # Accounting
    realized_pnl_usd: float       # Σ (exit − entry) × qty across all sells
    sell_reason: Optional[str]    # STOP_LOSS | TRAILING_STOP
    last_price: Optional[float]   # most recent observed price
    total_proceeds_usd: float     # gross USD received from all sells
    total_fees_usd: float         # reserved for fee modelling (0.0 for now)

    # Flags
    partial_take_profit_hit: bool = False   # True after TP1 sell fires
    tp2_hit: bool = False                   # True after TP2 sell fires
    tp3_hit: bool = False                   # True after TP3 sell fires
    tp4_hit: bool = False                   # True after TP4 sell fires

    # Multi-strategy identity (set by StrategyRunner after buy)
    strategy_name: str = "default"

    # Signal origin (set by StrategyRunner after buy)
    source_channel: str = ""


@dataclass
class PortfolioState:
    """Tracks the aggregate mock cash balance across all trades."""
    starting_cash_usd: float
    available_cash_usd: float
