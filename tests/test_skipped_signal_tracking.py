"""
tests/test_skipped_signal_tracking.py — Tests for skipped-signal phantom PnL tracking.

Covers:
- query_skipped_stats (base.py) — correct JOIN, aggregation, empty cases
- _format_skipped_section (strategy_tuner.py) — prompt text when data present / absent
- _BASE_STRATEGY mapping — correct base strategy per chart variant
- _get_signal_count (tune_strategies.py) — counts all rows (entered + skipped)
- tune trigger — fires on total signal count, not just closed trades

No network calls, no Anthropic API.
Run with: pytest tests/test_skipped_signal_tracking.py -v
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from trader.agents.base import query_skipped_stats
from trader.agents.strategy_tuner import (
    _BASE_STRATEGY,
    _format_skipped_section,
)


# ---------------------------------------------------------------------------
# Helpers — build an in-memory / temp-file DB with the minimal schema
# ---------------------------------------------------------------------------

def _make_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE signal_charts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT NOT NULL,
            symbol      TEXT NOT NULL,
            mint        TEXT NOT NULL,
            entry_price REAL NOT NULL,
            ml_score    REAL
        )
    """)
    conn.execute("""
        CREATE TABLE strategy_outcomes (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_chart_id      INTEGER NOT NULL,
            strategy             TEXT NOT NULL,
            entered              INTEGER NOT NULL DEFAULT 0,
            outcome_pnl_pct      REAL,
            outcome_max_gain_pct REAL,
            outcome_sell_reason  TEXT,
            closed               INTEGER NOT NULL DEFAULT 0,
            outcome_pnl_usd      REAL,
            is_live              INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.commit()
    return conn


def _insert_chart(conn, symbol="BONK", mint="abc", entry_price=0.001, ml_score=None, ts=None) -> int:
    cur = conn.execute(
        "INSERT INTO signal_charts (ts, symbol, mint, entry_price, ml_score) VALUES (?,?,?,?,?)",
        (ts or "2026-01-01T00:00:00", symbol, mint, entry_price, ml_score),
    )
    conn.commit()
    return cur.lastrowid


def _insert_outcome(conn, chart_id, strategy, entered, closed=0,
                    pnl_pct=None, max_gain_pct=None, sell_reason=None) -> int:
    cur = conn.execute(
        """INSERT INTO strategy_outcomes
               (signal_chart_id, strategy, entered, closed,
                outcome_pnl_pct, outcome_max_gain_pct, outcome_sell_reason)
           VALUES (?,?,?,?,?,?,?)""",
        (chart_id, strategy, int(entered), int(closed), pnl_pct, max_gain_pct, sell_reason),
    )
    conn.commit()
    return cur.lastrowid


# ---------------------------------------------------------------------------
# _BASE_STRATEGY mapping
# ---------------------------------------------------------------------------

class TestBaseStrategyMapping:
    def test_quick_pop_chart_ml_maps_to_quick_pop(self):
        assert _BASE_STRATEGY["quick_pop_chart_ml"] == "quick_pop"

    def test_trend_rider_chart_reanalyze_maps_to_trend_rider(self):
        assert _BASE_STRATEGY["trend_rider_chart_reanalyze"] == "trend_rider"

    def test_infinite_moonbag_chart_maps_to_infinite_moonbag(self):
        assert _BASE_STRATEGY["infinite_moonbag_chart"] == "infinite_moonbag"

    def test_all_chart_variants_have_mapping(self):
        chart_variants = {"quick_pop_chart_ml", "trend_rider_chart_reanalyze", "infinite_moonbag_chart"}
        assert chart_variants.issubset(_BASE_STRATEGY.keys())

    def test_base_strategies_not_in_mapping(self):
        for base in ("quick_pop", "trend_rider", "infinite_moonbag"):
            assert base not in _BASE_STRATEGY


# ---------------------------------------------------------------------------
# query_skipped_stats
# ---------------------------------------------------------------------------

class TestQuerySkippedStats:
    def test_no_skipped_signals_returns_zeros(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        conn = _make_db(path)
        conn.close()

        result = query_skipped_stats(path, "quick_pop_chart_ml", "quick_pop")
        assert result["total_skipped"] == 0
        assert result["base_entered"] == 0
        assert result["profitable_pct"] is None
        assert result["avg_phantom_pnl"] is None
        assert result["sample_outcomes"] == []

    def test_skipped_with_no_base_outcomes_yet(self):
        """Chart variant skipped a signal but base hasn't closed it yet."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        conn = _make_db(path)
        chart_id = _insert_chart(conn, symbol="WIF")
        _insert_outcome(conn, chart_id, "quick_pop_chart_ml", entered=False)
        # base entered but NOT closed yet
        _insert_outcome(conn, chart_id, "quick_pop", entered=True, closed=False)
        conn.close()

        result = query_skipped_stats(path, "quick_pop_chart_ml", "quick_pop")
        assert result["total_skipped"] == 1
        assert result["base_entered"] == 0  # not closed yet, excluded

    def test_profitable_phantom_trades(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        conn = _make_db(path)

        # 3 skipped signals — base strategy made money on all 3
        for pnl, gain, reason in [
            (25.0, 40.0, "TP1"),
            (15.0, 30.0, "TP2"),
            (50.0, 80.0, "TP2"),
        ]:
            cid = _insert_chart(conn)
            _insert_outcome(conn, cid, "quick_pop_chart_ml", entered=False)
            _insert_outcome(conn, cid, "quick_pop", entered=True, closed=True,
                            pnl_pct=pnl, max_gain_pct=gain, sell_reason=reason)
        conn.close()

        result = query_skipped_stats(path, "quick_pop_chart_ml", "quick_pop")
        assert result["total_skipped"] == 3
        assert result["base_entered"] == 3
        assert result["profitable_pct"] == 100.0
        assert abs(result["avg_phantom_pnl"] - 30.0) < 0.01
        assert abs(result["avg_max_gain"] - 50.0) < 0.01
        assert len(result["sample_outcomes"]) == 3

    def test_mixed_profitable_and_losing(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        conn = _make_db(path)

        # 2 profitable, 2 losing
        for pnl, gain in [(30.0, 50.0), (10.0, 20.0), (-15.0, 5.0), (-25.0, 3.0)]:
            cid = _insert_chart(conn)
            _insert_outcome(conn, cid, "quick_pop_chart_ml", entered=False)
            _insert_outcome(conn, cid, "quick_pop", entered=True, closed=True,
                            pnl_pct=pnl, max_gain_pct=gain)
        conn.close()

        result = query_skipped_stats(path, "quick_pop_chart_ml", "quick_pop")
        assert result["total_skipped"] == 4
        assert result["base_entered"] == 4
        assert result["profitable_pct"] == 50.0
        assert abs(result["avg_phantom_pnl"] - 0.0) < 0.01  # (30+10-15-25)/4

    def test_sample_outcomes_capped_at_ten(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        conn = _make_db(path)

        for i in range(15):
            cid = _insert_chart(conn, symbol=f"TK{i}", ts=f"2026-01-01T{i:02d}:00:00")
            _insert_outcome(conn, cid, "quick_pop_chart_ml", entered=False)
            _insert_outcome(conn, cid, "quick_pop", entered=True, closed=True,
                            pnl_pct=10.0, max_gain_pct=20.0)
        conn.close()

        result = query_skipped_stats(path, "quick_pop_chart_ml", "quick_pop")
        assert result["base_entered"] == 15
        assert len(result["sample_outcomes"]) == 10  # capped

    def test_does_not_count_entered_signals_as_skipped(self):
        """Signals this strategy entered should NOT appear in skipped count."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        conn = _make_db(path)

        # This signal was entered by the chart variant (not skipped)
        cid = _insert_chart(conn)
        _insert_outcome(conn, cid, "quick_pop_chart_ml", entered=True, closed=True, pnl_pct=20.0)
        _insert_outcome(conn, cid, "quick_pop", entered=True, closed=True, pnl_pct=20.0)
        conn.close()

        result = query_skipped_stats(path, "quick_pop_chart_ml", "quick_pop")
        assert result["total_skipped"] == 0
        assert result["base_entered"] == 0

    def test_trend_rider_variant_uses_trend_rider_base(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        conn = _make_db(path)

        cid = _insert_chart(conn)
        _insert_outcome(conn, cid, "trend_rider_chart_reanalyze", entered=False)
        _insert_outcome(conn, cid, "trend_rider", entered=True, closed=True,
                        pnl_pct=-10.0, max_gain_pct=5.0)
        conn.close()

        result = query_skipped_stats(path, "trend_rider_chart_reanalyze", "trend_rider")
        assert result["total_skipped"] == 1
        assert result["base_entered"] == 1
        assert result["profitable_pct"] == 0.0
        assert result["avg_phantom_pnl"] == -10.0

    def test_sample_outcomes_contain_expected_fields(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        conn = _make_db(path)

        cid = _insert_chart(conn, symbol="PEPE", ml_score=7.5)
        _insert_outcome(conn, cid, "quick_pop_chart_ml", entered=False)
        _insert_outcome(conn, cid, "quick_pop", entered=True, closed=True,
                        pnl_pct=22.5, max_gain_pct=45.0, sell_reason="TP2")
        conn.close()

        result = query_skipped_stats(path, "quick_pop_chart_ml", "quick_pop")
        outcome = result["sample_outcomes"][0]
        assert outcome["symbol"] == "PEPE"
        assert outcome["pnl_pct"] == 22.5
        assert outcome["max_gain_pct"] == 45.0
        assert outcome["sell_reason"] == "TP2"
        assert outcome["ml_score"] == 7.5


# ---------------------------------------------------------------------------
# _format_skipped_section
# ---------------------------------------------------------------------------

class TestFormatSkippedSection:
    def _no_data(self) -> dict:
        return {
            "total_skipped": 5,
            "base_entered": 0,
            "profitable_pct": None,
            "avg_phantom_pnl": None,
            "avg_max_gain": None,
            "sample_outcomes": [],
        }

    def _with_data(self, profitable_pct=60.0, avg_pnl=12.5, avg_gain=30.0) -> dict:
        return {
            "total_skipped": 10,
            "base_entered": 8,
            "profitable_pct": profitable_pct,
            "avg_phantom_pnl": avg_pnl,
            "avg_max_gain": avg_gain,
            "sample_outcomes": [
                {"symbol": "BONK", "pnl_pct": 12.5, "max_gain_pct": 30.0,
                 "sell_reason": "TP1", "ml_score": 6.0}
            ],
        }

    def test_no_data_shows_no_outcomes_message(self):
        section = _format_skipped_section(self._no_data(), "quick_pop")
        assert "No closed outcomes yet" in section
        assert "5" in section  # total_skipped

    def test_with_data_shows_profitable_pct(self):
        section = _format_skipped_section(self._with_data(profitable_pct=75.0), "quick_pop")
        assert "75.0%" in section

    def test_with_data_shows_avg_phantom_pnl(self):
        section = _format_skipped_section(self._with_data(avg_pnl=18.3), "quick_pop")
        assert "+18.30%" in section

    def test_with_data_shows_base_strategy_name(self):
        section = _format_skipped_section(self._with_data(), "trend_rider")
        assert "trend_rider" in section

    def test_with_data_shows_interpretation(self):
        section = _format_skipped_section(self._with_data(), "quick_pop")
        assert "filter" in section.lower()

    def test_negative_pnl_shown_with_sign(self):
        section = _format_skipped_section(self._with_data(avg_pnl=-8.0), "quick_pop")
        assert "-8.00%" in section

    def test_sample_outcomes_json_present(self):
        section = _format_skipped_section(self._with_data(), "quick_pop")
        assert "BONK" in section

    def test_zero_skipped_shows_zero(self):
        data = {**self._no_data(), "total_skipped": 0}
        section = _format_skipped_section(data, "quick_pop")
        assert "0" in section


# ---------------------------------------------------------------------------
# _get_signal_count (tune_strategies.py)
# ---------------------------------------------------------------------------

class TestGetSignalCount:
    """Tests for the new signal-count trigger (entered + skipped)."""

    def _import(self):
        # Deferred import so we don't break other tests if module has top-level issues
        import importlib, sys
        # Ensure scripts/ is on path
        import sys as _sys
        from pathlib import Path as _Path
        scripts_dir = str(_Path(__file__).parent.parent / "scripts")
        if scripts_dir not in _sys.path:
            _sys.path.insert(0, scripts_dir)
        import tune_strategies
        return tune_strategies._get_signal_count

    def test_empty_db_returns_zero(self):
        get_signal_count = self._import()
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        conn = _make_db(path)
        conn.close()
        assert get_signal_count(path, "quick_pop_chart_ml") == 0

    def test_counts_entered_signals(self):
        get_signal_count = self._import()
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        conn = _make_db(path)
        cid = _insert_chart(conn)
        _insert_outcome(conn, cid, "quick_pop_chart_ml", entered=True, closed=True)
        conn.close()
        assert get_signal_count(path, "quick_pop_chart_ml") == 1

    def test_counts_skipped_signals(self):
        get_signal_count = self._import()
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        conn = _make_db(path)
        for _ in range(4):
            cid = _insert_chart(conn)
            _insert_outcome(conn, cid, "quick_pop_chart_ml", entered=False)
        conn.close()
        assert get_signal_count(path, "quick_pop_chart_ml") == 4

    def test_counts_both_entered_and_skipped(self):
        get_signal_count = self._import()
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        conn = _make_db(path)
        # 3 entered, 5 skipped = 8 total
        for _ in range(3):
            cid = _insert_chart(conn)
            _insert_outcome(conn, cid, "quick_pop_chart_ml", entered=True, closed=True)
        for _ in range(5):
            cid = _insert_chart(conn)
            _insert_outcome(conn, cid, "quick_pop_chart_ml", entered=False)
        conn.close()
        assert get_signal_count(path, "quick_pop_chart_ml") == 8

    def test_does_not_count_other_strategies(self):
        get_signal_count = self._import()
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        conn = _make_db(path)
        cid = _insert_chart(conn)
        _insert_outcome(conn, cid, "trend_rider", entered=True, closed=True)
        _insert_outcome(conn, cid, "quick_pop_chart_ml", entered=False)
        conn.close()
        # trend_rider has 1, quick_pop_chart_ml has 1
        assert get_signal_count(path, "quick_pop_chart_ml") == 1
        assert get_signal_count(path, "trend_rider") == 1

    def test_missing_table_returns_zero(self):
        """Handles databases that predate strategy_outcomes table."""
        get_signal_count = self._import()
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        # Don't create strategy_outcomes table
        conn = sqlite3.connect(path)
        conn.close()
        assert get_signal_count(path, "quick_pop_chart_ml") == 0


# ---------------------------------------------------------------------------
# Tune trigger threshold logic
# ---------------------------------------------------------------------------

class TestTuneTrigger:
    def _import_should_tune(self):
        import sys
        from pathlib import Path
        scripts_dir = str(Path(__file__).parent.parent / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import tune_strategies
        return tune_strategies._should_tune

    def test_fires_when_threshold_met(self):
        _should_tune = self._import_should_tune()
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        conn = _make_db(path)
        # Insert 10 signals (mix of entered + skipped)
        for i in range(6):
            cid = _insert_chart(conn)
            _insert_outcome(conn, cid, "quick_pop_chart_ml", entered=True, closed=True)
        for i in range(4):
            cid = _insert_chart(conn)
            _insert_outcome(conn, cid, "quick_pop_chart_ml", entered=False)
        conn.close()

        meta = {"trades_at_last_tune": {"quick_pop_chart_ml": 0}}
        should, count = _should_tune("quick_pop_chart_ml", path, meta, every=10)
        assert should is True
        assert count == 10

    def test_does_not_fire_before_threshold(self):
        _should_tune = self._import_should_tune()
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        conn = _make_db(path)
        for _ in range(5):
            cid = _insert_chart(conn)
            _insert_outcome(conn, cid, "quick_pop_chart_ml", entered=False)
        conn.close()

        meta = {"trades_at_last_tune": {"quick_pop_chart_ml": 0}}
        should, count = _should_tune("quick_pop_chart_ml", path, meta, every=10)
        assert should is False
        assert count == 5

    def test_fires_based_on_delta_not_absolute(self):
        """If baseline was 8 and now 18, should fire at every=10."""
        _should_tune = self._import_should_tune()
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        conn = _make_db(path)
        for _ in range(18):
            cid = _insert_chart(conn)
            _insert_outcome(conn, cid, "quick_pop_chart_ml", entered=False)
        conn.close()

        # Baseline at 8 — so 10 new signals have come in
        meta = {"trades_at_last_tune": {"quick_pop_chart_ml": 8}}
        should, count = _should_tune("quick_pop_chart_ml", path, meta, every=10)
        assert should is True
        assert count == 18

    def test_skipped_signals_alone_trigger_tune(self):
        """Tuner should fire even if the strategy never entered a single trade."""
        _should_tune = self._import_should_tune()
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        conn = _make_db(path)
        # All skipped — chart filter blocked everything
        for _ in range(10):
            cid = _insert_chart(conn)
            _insert_outcome(conn, cid, "quick_pop_chart_ml", entered=False)
        conn.close()

        meta = {"trades_at_last_tune": {}}
        should, count = _should_tune("quick_pop_chart_ml", path, meta, every=10)
        assert should is True
