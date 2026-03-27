"""
tests/test_db_queries.py — Unit tests for DB query functions in base.py and
TradeDatabase.save_strategy_outcome().

Covers the gaps introduced when ml_score was moved from signal_charts to
strategy_outcomes (per-strategy), and save_strategy_outcome() gained an
ml_score parameter:

  - save_strategy_outcome: ml_score is written and readable
  - query_score_buckets:   reads so.ml_score (not sc.ml_score)
  - query_recent_trades:   reads so.ml_score (not sc.ml_score)
  - query_skipped_stats:   sample_outcomes ml_score comes from outcome row

No network calls, no Anthropic API.
Run with: pytest tests/test_db_queries.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from trader.agents.base import (
    query_recent_trades,
    query_score_buckets,
    query_skipped_stats,
)
from trader.analysis.chart import OHLCVCandle
from trader.persistence.database import TradeDatabase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trade_db(path: str) -> TradeDatabase:
    """Create a fresh TradeDatabase at path (runs all migrations)."""
    return TradeDatabase(path)


def _insert_signal_chart(db: TradeDatabase, symbol: str = "BONK",
                          sc_ml_score: float | None = None) -> int:
    """Insert a minimal signal_charts row. Returns the row id."""
    cursor = db._conn.execute(
        """
        INSERT INTO signal_charts (ts, symbol, mint, entry_price, candles_json, ml_score)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("2026-01-01T00:00:00", symbol, f"mint_{symbol}", 0.001, "[]", sc_ml_score),
    )
    db._conn.commit()
    return cursor.lastrowid


# ---------------------------------------------------------------------------
# save_strategy_outcome — ml_score is persisted
# ---------------------------------------------------------------------------

class TestSaveStrategyOutcome:
    def test_ml_score_written_and_readable(self, tmp_path):
        db = _make_trade_db(str(tmp_path / "trader.db"))
        sc_id = _insert_signal_chart(db)
        row_id = db.save_strategy_outcome(
            signal_chart_id=sc_id,
            strategy="quick_pop_managed",
            entered=True,
            ml_score=7.3,
        )
        row = db._conn.execute(
            "SELECT ml_score FROM strategy_outcomes WHERE id = ?", (row_id,)
        ).fetchone()
        assert row[0] == pytest.approx(7.3)

    def test_ml_score_none_stored_as_null(self, tmp_path):
        db = _make_trade_db(str(tmp_path / "trader.db"))
        sc_id = _insert_signal_chart(db)
        row_id = db.save_strategy_outcome(
            signal_chart_id=sc_id,
            strategy="quick_pop_managed",
            entered=True,
            ml_score=None,
        )
        row = db._conn.execute(
            "SELECT ml_score FROM strategy_outcomes WHERE id = ?", (row_id,)
        ).fetchone()
        assert row[0] is None

    def test_different_strategies_store_independent_scores(self, tmp_path):
        db = _make_trade_db(str(tmp_path / "trader.db"))
        sc_id = _insert_signal_chart(db)
        id_a = db.save_strategy_outcome(sc_id, "quick_pop_managed", entered=True, ml_score=6.0)
        id_b = db.save_strategy_outcome(sc_id, "trend_rider_managed", entered=True, ml_score=8.5)
        rows = {
            r[0]: r[1]
            for r in db._conn.execute(
                "SELECT id, ml_score FROM strategy_outcomes WHERE id IN (?, ?)", (id_a, id_b)
            ).fetchall()
        }
        assert rows[id_a] == pytest.approx(6.0)
        assert rows[id_b] == pytest.approx(8.5)


# ---------------------------------------------------------------------------
# query_score_buckets — uses so.ml_score, not sc.ml_score
# ---------------------------------------------------------------------------

class TestQueryScoreBuckets:
    def _setup_db(self, tmp_path) -> str:
        """
        Two trades in bucket 5 (ml_score ~5.x), one winner one loser.
        sc.ml_score is deliberately set to a different value (9.9) to
        prove the query reads from strategy_outcomes, not signal_charts.
        """
        db = _make_trade_db(str(tmp_path / "trader.db"))
        for pnl in (20.0, -10.0):
            sc_id = _insert_signal_chart(db, sc_ml_score=9.9)  # wrong score on chart row
            row_id = db.save_strategy_outcome(
                sc_id, "quick_pop_managed", entered=True, ml_score=5.5,
            )
            db._conn.execute(
                """UPDATE strategy_outcomes
                   SET closed=1, outcome_pnl_pct=?, outcome_max_gain_pct=30.0
                   WHERE id=?""",
                (pnl, row_id),
            )
        db._conn.commit()
        return str(tmp_path / "trader.db")

    def test_returns_bucket_for_outcome_ml_score(self, tmp_path):
        path = self._setup_db(tmp_path)
        buckets = query_score_buckets(path, "quick_pop_managed")
        assert len(buckets) == 1
        assert buckets[0]["bucket"] == "5.0-5.9"

    def test_ignores_signal_chart_ml_score(self, tmp_path):
        """sc.ml_score=9.9 must NOT create a bucket-9 entry."""
        path = self._setup_db(tmp_path)
        buckets = query_score_buckets(path, "quick_pop_managed")
        bucket_labels = [b["bucket"] for b in buckets]
        assert "9.0-9.9" not in bucket_labels

    def test_win_rate_computed_correctly(self, tmp_path):
        path = self._setup_db(tmp_path)
        bucket = query_score_buckets(path, "quick_pop_managed")[0]
        assert bucket["count"] == 2
        assert bucket["win_rate"] == pytest.approx(0.5)

    def test_null_outcome_ml_score_excluded(self, tmp_path):
        db = _make_trade_db(str(tmp_path / "trader.db"))
        sc_id = _insert_signal_chart(db)
        row_id = db.save_strategy_outcome(sc_id, "quick_pop_managed", entered=True, ml_score=None)
        db._conn.execute(
            "UPDATE strategy_outcomes SET closed=1, outcome_pnl_pct=10.0 WHERE id=?", (row_id,)
        )
        db._conn.commit()
        buckets = query_score_buckets(str(tmp_path / "trader.db"), "quick_pop_managed")
        assert buckets == []

    def test_not_entered_excluded(self, tmp_path):
        db = _make_trade_db(str(tmp_path / "trader.db"))
        sc_id = _insert_signal_chart(db)
        row_id = db.save_strategy_outcome(sc_id, "quick_pop_managed", entered=False, ml_score=6.0)
        db._conn.execute(
            "UPDATE strategy_outcomes SET closed=1, outcome_pnl_pct=15.0 WHERE id=?", (row_id,)
        )
        db._conn.commit()
        assert query_score_buckets(str(tmp_path / "trader.db"), "quick_pop_managed") == []

    def test_different_strategies_bucketed_independently(self, tmp_path):
        db = _make_trade_db(str(tmp_path / "trader.db"))
        sc_id = _insert_signal_chart(db)
        for strategy, score in [("quick_pop_managed", 5.0), ("trend_rider_managed", 7.0)]:
            row_id = db.save_strategy_outcome(sc_id, strategy, entered=True, ml_score=score)
            db._conn.execute(
                "UPDATE strategy_outcomes SET closed=1, outcome_pnl_pct=10.0 WHERE id=?", (row_id,)
            )
        db._conn.commit()
        path = str(tmp_path / "trader.db")
        qp_buckets = query_score_buckets(path, "quick_pop_managed")
        tr_buckets  = query_score_buckets(path, "trend_rider_managed")
        assert qp_buckets[0]["bucket"] == "5.0-5.9"
        assert tr_buckets[0]["bucket"]  == "7.0-7.9"


# ---------------------------------------------------------------------------
# query_recent_trades — uses so.ml_score
# ---------------------------------------------------------------------------

class TestQueryRecentTrades:
    def test_ml_score_from_outcome_row(self, tmp_path):
        db = _make_trade_db(str(tmp_path / "trader.db"))
        sc_id = _insert_signal_chart(db, sc_ml_score=9.9)  # different from outcome score
        row_id = db.save_strategy_outcome(
            sc_id, "quick_pop_managed", entered=True, ml_score=6.2,
        )
        db._conn.execute(
            "UPDATE strategy_outcomes SET closed=1, outcome_pnl_pct=12.0 WHERE id=?", (row_id,)
        )
        db._conn.commit()
        trades = query_recent_trades(str(tmp_path / "trader.db"), "quick_pop_managed")
        assert len(trades) == 1
        assert trades[0]["ml_score"] == pytest.approx(6.2)

    def test_null_outcome_ml_score_returned_as_none(self, tmp_path):
        db = _make_trade_db(str(tmp_path / "trader.db"))
        sc_id = _insert_signal_chart(db, sc_ml_score=8.0)
        row_id = db.save_strategy_outcome(sc_id, "quick_pop_managed", entered=True, ml_score=None)
        db._conn.execute(
            "UPDATE strategy_outcomes SET closed=1, outcome_pnl_pct=5.0 WHERE id=?", (row_id,)
        )
        db._conn.commit()
        trades = query_recent_trades(str(tmp_path / "trader.db"), "quick_pop_managed")
        assert trades[0]["ml_score"] is None

    def test_scored_only_filters_null_ml_score(self, tmp_path):
        db = _make_trade_db(str(tmp_path / "trader.db"))
        for score in (5.0, None):
            sc_id = _insert_signal_chart(db)
            row_id = db.save_strategy_outcome(
                sc_id, "quick_pop_managed", entered=True, ml_score=score,
            )
            db._conn.execute(
                "UPDATE strategy_outcomes SET closed=1, outcome_pnl_pct=10.0 WHERE id=?", (row_id,)
            )
        db._conn.commit()
        trades = query_recent_trades(
            str(tmp_path / "trader.db"), "quick_pop_managed", scored_only=True
        )
        assert len(trades) == 1
        assert trades[0]["ml_score"] == pytest.approx(5.0)

    def test_limit_respected(self, tmp_path):
        db = _make_trade_db(str(tmp_path / "trader.db"))
        for i in range(5):
            sc_id = _insert_signal_chart(db)
            row_id = db.save_strategy_outcome(sc_id, "quick_pop_managed", entered=True, ml_score=float(i))
            db._conn.execute(
                "UPDATE strategy_outcomes SET closed=1, outcome_pnl_pct=10.0 WHERE id=?", (row_id,)
            )
        db._conn.commit()
        trades = query_recent_trades(str(tmp_path / "trader.db"), "quick_pop_managed", limit=3)
        assert len(trades) == 3


# ---------------------------------------------------------------------------
# query_skipped_stats — sample_outcomes ml_score from base outcome row
# ---------------------------------------------------------------------------

class TestQuerySkippedStatsMlScore:
    def test_sample_outcome_ml_score_from_outcome_row(self, tmp_path):
        """
        The chart variant skips a signal. The base strategy enters it and
        has ml_score=7.5 on its outcome row. query_skipped_stats should
        report that score in sample_outcomes — NOT from signal_charts.
        """
        db = _make_trade_db(str(tmp_path / "trader.db"))
        sc_id = _insert_signal_chart(db, symbol="PEPE", sc_ml_score=9.9)  # decoy on chart row

        # Chart variant skips it (no ml_score on this row — it was blocked)
        db.save_strategy_outcome(sc_id, "quick_pop_managed", entered=False, ml_score=None)

        # Base strategy enters and has its own score
        base_id = db.save_strategy_outcome(sc_id, "quick_pop", entered=True, ml_score=7.5)
        db._conn.execute(
            """UPDATE strategy_outcomes
               SET closed=1, outcome_pnl_pct=22.5, outcome_max_gain_pct=45.0,
                   outcome_sell_reason='TP2'
               WHERE id=?""",
            (base_id,),
        )
        db._conn.commit()

        result = query_skipped_stats(str(tmp_path / "trader.db"), "quick_pop_managed", "quick_pop")
        assert result["base_entered"] == 1
        outcome = result["sample_outcomes"][0]
        assert outcome["symbol"] == "PEPE"
        assert outcome["ml_score"] == pytest.approx(7.5)   # from outcome row, not sc.ml_score=9.9

    def test_sample_outcome_ml_score_none_when_not_scored(self, tmp_path):
        db = _make_trade_db(str(tmp_path / "trader.db"))
        sc_id = _insert_signal_chart(db)
        db.save_strategy_outcome(sc_id, "quick_pop_managed", entered=False)
        base_id = db.save_strategy_outcome(sc_id, "quick_pop", entered=True, ml_score=None)
        db._conn.execute(
            "UPDATE strategy_outcomes SET closed=1, outcome_pnl_pct=5.0 WHERE id=?", (base_id,)
        )
        db._conn.commit()
        result = query_skipped_stats(str(tmp_path / "trader.db"), "quick_pop_managed", "quick_pop")
        assert result["sample_outcomes"][0]["ml_score"] is None


# ---------------------------------------------------------------------------
# save_signal_chart — candles_1s_json persistence
# ---------------------------------------------------------------------------

def _make_candle(unix_time: int, price: float = 1.0) -> OHLCVCandle:
    return OHLCVCandle(unix_time=unix_time, open=price, high=price*1.1,
                       low=price*0.9, close=price, volume=100.0)


class TestSaveSignalChartCandles1s:
    def test_candles_1s_saved_and_readable(self, tmp_path):
        db = TradeDatabase(str(tmp_path / "trader.db"))
        candles_1s = [_make_candle(1_000_000 + i) for i in range(5)]
        row_id = db.save_signal_chart(
            symbol="TEST", mint="mint_test", entry_price=0.001,
            candles=[_make_candle(999_999)],
            chart_ctx=None,
            candles_1s=candles_1s,
        )
        raw = db._conn.execute(
            "SELECT candles_1s_json FROM signal_charts WHERE id=?", (row_id,)
        ).fetchone()[0]
        assert raw is not None
        parsed = json.loads(raw)
        assert len(parsed) == 5
        assert parsed[0]["t"] == 1_000_000

    def test_candles_1s_fields_serialised_correctly(self, tmp_path):
        db = TradeDatabase(str(tmp_path / "trader.db"))
        c = _make_candle(7777, price=2.5)
        row_id = db.save_signal_chart(
            symbol="TEST", mint="mint_test", entry_price=0.001,
            candles=[_make_candle(7776)], chart_ctx=None, candles_1s=[c],
        )
        raw = json.loads(db._conn.execute(
            "SELECT candles_1s_json FROM signal_charts WHERE id=?", (row_id,)
        ).fetchone()[0])
        item = raw[0]
        assert item["t"] == 7777
        assert item["o"] == pytest.approx(2.5)
        assert item["h"] == pytest.approx(2.75)
        assert item["l"] == pytest.approx(2.25)
        assert item["c"] == pytest.approx(2.5)
        assert item["v"] == pytest.approx(100.0)

    def test_candles_1s_null_when_not_provided(self, tmp_path):
        db = TradeDatabase(str(tmp_path / "trader.db"))
        row_id = db.save_signal_chart(
            symbol="TEST", mint="mint_test", entry_price=0.001,
            candles=[_make_candle(1)], chart_ctx=None,
        )
        raw = db._conn.execute(
            "SELECT candles_1s_json FROM signal_charts WHERE id=?", (row_id,)
        ).fetchone()[0]
        assert raw is None

    def test_candles_1s_null_when_empty_list(self, tmp_path):
        db = TradeDatabase(str(tmp_path / "trader.db"))
        row_id = db.save_signal_chart(
            symbol="TEST", mint="mint_test", entry_price=0.001,
            candles=[_make_candle(1)], chart_ctx=None, candles_1s=[],
        )
        raw = db._conn.execute(
            "SELECT candles_1s_json FROM signal_charts WHERE id=?", (row_id,)
        ).fetchone()[0]
        assert raw is None

    def test_candles_1s_column_exists_after_migration(self, tmp_path):
        """Opening a fresh DB should have candles_1s_json in the schema."""
        db = TradeDatabase(str(tmp_path / "trader.db"))
        cols = [r[1] for r in db._conn.execute(
            "PRAGMA table_info(signal_charts)"
        ).fetchall()]
        assert "candles_1s_json" in cols

    def test_existing_rows_have_null_candles_1s(self, tmp_path):
        """Rows inserted before the migration should read back as NULL."""
        db = TradeDatabase(str(tmp_path / "trader.db"))
        # Insert without candles_1s_json (simulates pre-migration row)
        cursor = db._conn.execute(
            "INSERT INTO signal_charts (ts, symbol, mint, entry_price, candles_json)"
            " VALUES (?, ?, ?, ?, ?)",
            ("2025-01-01T00:00:00", "OLD", "mint_old", 0.001, "[]"),
        )
        db._conn.commit()
        raw = db._conn.execute(
            "SELECT candles_1s_json FROM signal_charts WHERE id=?",
            (cursor.lastrowid,)
        ).fetchone()[0]
        assert raw is None
