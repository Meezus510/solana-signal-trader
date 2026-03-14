"""
tests/test_strategy.py — Unit tests for PaperExchange strategy and PnL math.

Tests cover:
    - Position creation and cash deduction on buy
    - Partial TP sell PnL accumulation
    - Trailing stop full sell and position close
    - Stop loss sell
    - Insufficient cash guard
    - PortfolioManager open/close/dedup logic

No network, no Telegram, no Birdeye calls.
Run with:  pytest tests/test_strategy.py -v
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone

from trader.trading.exchange import PaperExchange
from trader.trading.models import PortfolioState, TokenSignal
from trader.trading.portfolio import PortfolioManager
from trader.trading.strategy import StrategyConfig, TakeProfitLevel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg() -> StrategyConfig:
    """Minimal StrategyConfig for testing — TP at 2.5×, SL at -35%."""
    return StrategyConfig(
        name="test",
        buy_size_usd=10.0,
        stop_loss_pct=0.35,
        take_profit_levels=(
            TakeProfitLevel(multiple=2.5, sell_fraction_original=0.50),
        ),
        trailing_stop_pct=0.35,
        starting_cash_usd=1_000.0,
    )


def _portfolio() -> PortfolioState:
    return PortfolioState(starting_cash_usd=1_000.0, available_cash_usd=1_000.0)


def _signal(symbol: str = "TEST") -> TokenSignal:
    return TokenSignal(
        symbol=symbol,
        mint_address=f"TestMint{'1' * (44 - len('TestMint'))}",
    )


# ---------------------------------------------------------------------------
# Buy
# ---------------------------------------------------------------------------

class TestBuy:
    def test_creates_position(self):
        port = _portfolio()
        ex = PaperExchange(port, _cfg())
        pos = ex.buy(_signal(), entry_price=1.0, usd_size=10.0)
        assert pos is not None
        assert pos.status == "OPEN"

    def test_quantity_is_usd_over_price(self):
        port = _portfolio()
        pos = PaperExchange(port, _cfg()).buy(_signal(), entry_price=2.0, usd_size=10.0)
        assert abs(pos.remaining_quantity - 5.0) < 1e-9

    def test_cash_deducted(self):
        port = _portfolio()
        PaperExchange(port, _cfg()).buy(_signal(), entry_price=1.0, usd_size=10.0)
        assert abs(port.available_cash_usd - 990.0) < 1e-9

    def test_tp_and_sl_prices(self):
        cfg = _cfg()
        port = _portfolio()
        pos = PaperExchange(port, cfg).buy(_signal(), entry_price=1.0, usd_size=10.0)
        assert abs(pos.take_profit_price - 2.5) < 1e-9
        assert abs(pos.stop_loss_price - 0.65) < 1e-9

    def test_insufficient_cash_returns_none(self):
        port = PortfolioState(starting_cash_usd=5.0, available_cash_usd=5.0)
        result = PaperExchange(port, _cfg()).buy(_signal(), entry_price=1.0, usd_size=10.0)
        assert result is None

    def test_insufficient_cash_does_not_deduct(self):
        port = PortfolioState(starting_cash_usd=5.0, available_cash_usd=5.0)
        PaperExchange(port, _cfg()).buy(_signal(), entry_price=1.0, usd_size=10.0)
        assert port.available_cash_usd == 5.0


# ---------------------------------------------------------------------------
# Partial TP sell
# ---------------------------------------------------------------------------

class TestSellPartial:
    def setup_method(self):
        self.port = _portfolio()
        self.ex = PaperExchange(self.port, _cfg())
        self.pos = self.ex.buy(_signal(), entry_price=1.0, usd_size=10.0)

    def test_remaining_quantity_halved(self):
        self.ex.sell_partial(self.pos, fraction=0.50, exit_price=2.5, reason="TP")
        assert abs(self.pos.remaining_quantity - 5.0) < 1e-9

    def test_pnl_correct(self):
        # (2.5 - 1.0) * 5 = 7.50
        self.ex.sell_partial(self.pos, fraction=0.50, exit_price=2.5, reason="TP")
        assert abs(self.pos.realized_pnl_usd - 7.5) < 1e-9

    def test_proceeds_returned_to_cash(self):
        # 990 (after buy) + 5 * 2.5 = 990 + 12.5 = 1002.5
        self.ex.sell_partial(self.pos, fraction=0.50, exit_price=2.5, reason="TP")
        assert abs(self.port.available_cash_usd - 1_002.5) < 1e-9

    def test_position_stays_open(self):
        self.ex.sell_partial(self.pos, fraction=0.50, exit_price=2.5, reason="TP")
        assert self.pos.status == "OPEN"


# ---------------------------------------------------------------------------
# Full sell (trailing stop / stop loss)
# ---------------------------------------------------------------------------

class TestSellAll:
    def setup_method(self):
        self.port = _portfolio()
        self.ex = PaperExchange(self.port, _cfg())
        self.pos = self.ex.buy(_signal(), entry_price=1.0, usd_size=10.0)

    def test_closes_position(self):
        self.ex.sell_all(self.pos, exit_price=1.625, reason="TRAILING_STOP")
        assert self.pos.status == "CLOSED"

    def test_remaining_quantity_zero(self):
        self.ex.sell_all(self.pos, exit_price=1.625, reason="TRAILING_STOP")
        assert self.pos.remaining_quantity == 0.0

    def test_sell_reason_recorded(self):
        self.ex.sell_all(self.pos, exit_price=0.65, reason="STOP_LOSS")
        assert self.pos.sell_reason == "STOP_LOSS"

    def test_closed_at_set(self):
        self.ex.sell_all(self.pos, exit_price=0.65, reason="STOP_LOSS")
        assert self.pos.closed_at is not None


# ---------------------------------------------------------------------------
# Full TP + trailing stop cycle (PnL accumulation)
# ---------------------------------------------------------------------------

class TestFullCycle:
    """
    Entry @ $1.00 → 10 tokens
    TP @ $2.50: sell 5 tokens → pnl = (2.50 - 1.00) * 5 = $7.50
    Trail stop @ $1.625 (2.50 * 0.65): sell 5 tokens → pnl = (1.625 - 1.00) * 5 = $3.125
    Total realized = $10.625
    """

    def test_total_realized_pnl(self):
        port = _portfolio()
        ex = PaperExchange(port, _cfg())
        pos = ex.buy(_signal(), entry_price=1.0, usd_size=10.0)

        ex.sell_partial(pos, fraction=0.50, exit_price=2.5, reason="TP_2_5X")
        ex.sell_all(pos, exit_price=1.625, reason="TRAILING_STOP")

        assert abs(pos.realized_pnl_usd - 10.625) < 1e-9

    def test_cash_after_full_cycle(self):
        port = _portfolio()
        ex = PaperExchange(port, _cfg())
        pos = ex.buy(_signal(), entry_price=1.0, usd_size=10.0)

        ex.sell_partial(pos, fraction=0.50, exit_price=2.5, reason="TP_2_5X")
        ex.sell_all(pos, exit_price=1.625, reason="TRAILING_STOP")

        # 990 + 12.5 (TP proceeds) + 8.125 (trail proceeds) = 1010.625
        assert abs(port.available_cash_usd - 1_010.625) < 1e-9

    def test_stop_loss_pnl(self):
        port = _portfolio()
        ex = PaperExchange(port, _cfg())
        pos = ex.buy(_signal(), entry_price=1.0, usd_size=10.0)

        # Hit SL at 0.65 (-35%)
        ex.sell_all(pos, exit_price=0.65, reason="STOP_LOSS")

        # (0.65 - 1.00) * 10 = -$3.50
        assert abs(pos.realized_pnl_usd - (-3.50)) < 1e-9


# ---------------------------------------------------------------------------
# PortfolioManager
# ---------------------------------------------------------------------------

class TestPortfolioManager:
    def test_add_and_retrieve(self):
        mgr = PortfolioManager()
        port = _portfolio()
        ex = PaperExchange(port, _cfg())
        sig = _signal("ALPHA")
        pos = ex.buy(sig, entry_price=1.0, usd_size=10.0)
        mgr.add_position(pos)

        assert mgr.has_open_position(pos.mint_address)
        assert mgr.get_position(pos.mint_address) is pos

    def test_close_removes_from_open_index(self):
        mgr = PortfolioManager()
        port = _portfolio()
        ex = PaperExchange(port, _cfg())
        pos = ex.buy(_signal(), entry_price=1.0, usd_size=10.0)
        mgr.add_position(pos)

        ex.sell_all(pos, exit_price=0.65, reason="STOP_LOSS")
        mgr.close_position(pos.mint_address)

        assert not mgr.has_open_position(pos.mint_address)

    def test_closed_position_preserved_in_history(self):
        mgr = PortfolioManager()
        port = _portfolio()
        ex = PaperExchange(port, _cfg())
        pos = ex.buy(_signal(), entry_price=1.0, usd_size=10.0)
        mgr.add_position(pos)
        ex.sell_all(pos, exit_price=0.65, reason="STOP_LOSS")
        mgr.close_position(pos.mint_address)

        assert pos in mgr.all_positions()
        assert pos in mgr.get_closed_positions()

    def test_counts(self):
        mgr = PortfolioManager()
        port = _portfolio()
        ex = PaperExchange(port, _cfg())

        for i in range(3):
            sig = TokenSignal(symbol=f"TOK{i}", mint_address=f"Mint{'1' * 40}{i}")
            pos = ex.buy(sig, entry_price=1.0, usd_size=10.0)
            mgr.add_position(pos)

        assert mgr.total_count == 3
        assert mgr.open_count == 3
        assert mgr.closed_count == 0
