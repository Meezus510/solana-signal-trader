"""
tests/test_ai_override.py — Tests for the AI override agent feature.

Covers:
- TradeDatabase.save_ai_override_decision — all fields persisted correctly
- TradeDatabase.save_strategy_outcome    — is_ai_override flag stored/read
- TradeDatabase.query_ai_override_stats  — override wins, rejection counterfactuals, shadow
- ai_override.summarize_candles          — dict/object candles, empty, 8-bar cap
- ai_override.log_override_decision      — file format, SHADOW prefix, non-fatal errors
- ai_override._parse_and_validate        — mutual exclusivity, clamp, bad JSON fallback
- StrategyConfig                         — use_ai_override / use_ai_override_shadow defaults

No network calls, no Anthropic API.
Run with: pytest tests/test_ai_override.py -v
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from trader.agents.ai_override import (
    _parse_and_validate,
    log_override_decision,
    summarize_candles,
)
from trader.persistence.database import TradeDatabase
from trader.trading.strategy import StrategyConfig, TakeProfitLevel


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_db(tmp_path) -> TradeDatabase:
    return TradeDatabase(str(tmp_path / "trader.db"))


def _insert_chart(db: TradeDatabase, symbol: str = "BONK", ts: str = "2026-01-01T00:00:00") -> int:
    cur = db._conn.execute(
        "INSERT INTO signal_charts (ts, symbol, mint, entry_price, candles_json) VALUES (?,?,?,?,?)",
        (ts, symbol, f"mint_{symbol}", 0.001, "[]"),
    )
    db._conn.commit()
    return cur.lastrowid


def _close_outcome(db: TradeDatabase, outcome_id: int, pnl_pct: float, max_gain: float = 20.0) -> None:
    db._conn.execute(
        "UPDATE strategy_outcomes SET closed=1, outcome_pnl_pct=?, outcome_max_gain_pct=? WHERE id=?",
        (pnl_pct, max_gain, outcome_id),
    )
    db._conn.commit()


# ---------------------------------------------------------------------------
# save_ai_override_decision
# ---------------------------------------------------------------------------

class TestSaveAiOverrideDecision:
    def test_decision_persisted_and_readable(self, tmp_path):
        db = _make_db(tmp_path)
        chart_id = _insert_chart(db)

        db.save_ai_override_decision(
            strategy="quick_pop_managed",
            signal_chart_id=chart_id,
            symbol="BONK",
            mint="abc123",
            skip_reason="ML_SKIP",
            decision="OVERRIDE",
            ml_score=1.8,
            pump_ratio=2.1,
            vol_trend="RISING",
            agent_reason="Strong buy pressure",
            reanalyze_delay=0.0,
        )

        row = db._conn.execute(
            "SELECT strategy, symbol, mint, skip_reason, decision, ml_score, "
            "pump_ratio, vol_trend, agent_reason, reanalyze_delay "
            "FROM ai_override_decisions WHERE signal_chart_id = ?",
            (chart_id,),
        ).fetchone()

        assert row[0] == "quick_pop_managed"
        assert row[1] == "BONK"
        assert row[2] == "abc123"
        assert row[3] == "ML_SKIP"
        assert row[4] == "OVERRIDE"
        assert row[5] == pytest.approx(1.8)
        assert row[6] == pytest.approx(2.1)
        assert row[7] == "RISING"
        assert row[8] == "Strong buy pressure"
        assert row[9] == pytest.approx(0.0)

    def test_null_signal_chart_id_allowed(self, tmp_path):
        db = _make_db(tmp_path)
        db.save_ai_override_decision(
            strategy="quick_pop_managed",
            signal_chart_id=None,
            symbol="WIF",
            mint="xyz",
            skip_reason="CHART_SKIP",
            decision="REJECT",
            ml_score=None,
            pump_ratio=3.8,
            vol_trend="DYING",
            agent_reason="Volume dying",
        )
        row = db._conn.execute("SELECT signal_chart_id FROM ai_override_decisions").fetchone()
        assert row[0] is None

    @pytest.mark.parametrize("decision", [
        "OVERRIDE", "REJECT", "REANALYZE",
        "SHADOW_OVERRIDE", "SHADOW_REJECT", "SHADOW_REANALYZE",
    ])
    def test_all_decision_types_accepted(self, tmp_path, decision):
        db = _make_db(tmp_path)
        db.save_ai_override_decision(
            strategy="quick_pop_managed",
            signal_chart_id=None,
            symbol="TKN",
            mint="m",
            skip_reason="ML_SKIP",
            decision=decision,
            ml_score=None,
            pump_ratio=None,
            vol_trend=None,
            agent_reason="test",
        )
        row = db._conn.execute("SELECT decision FROM ai_override_decisions").fetchone()
        assert row[0] == decision

    def test_reanalyze_delay_stored(self, tmp_path):
        db = _make_db(tmp_path)
        db.save_ai_override_decision(
            strategy="quick_pop_managed", signal_chart_id=None,
            symbol="X", mint="m", skip_reason="ML_SKIP", decision="REANALYZE",
            ml_score=None, pump_ratio=None, vol_trend=None, agent_reason="check again",
            reanalyze_delay=120.0,
        )
        row = db._conn.execute("SELECT reanalyze_delay FROM ai_override_decisions").fetchone()
        assert row[0] == pytest.approx(120.0)


# ---------------------------------------------------------------------------
# save_strategy_outcome — is_ai_override flag
# ---------------------------------------------------------------------------

class TestSaveStrategyOutcomeAiOverride:
    def test_is_ai_override_true_stored_as_1(self, tmp_path):
        db = _make_db(tmp_path)
        sc_id = _insert_chart(db)
        row_id = db.save_strategy_outcome(
            signal_chart_id=sc_id, strategy="quick_pop_managed",
            entered=True, is_ai_override=True,
        )
        row = db._conn.execute(
            "SELECT is_ai_override FROM strategy_outcomes WHERE id=?", (row_id,)
        ).fetchone()
        assert row[0] == 1

    def test_is_ai_override_false_stored_as_0(self, tmp_path):
        db = _make_db(tmp_path)
        sc_id = _insert_chart(db)
        row_id = db.save_strategy_outcome(
            signal_chart_id=sc_id, strategy="quick_pop_managed",
            entered=True, is_ai_override=False,
        )
        row = db._conn.execute(
            "SELECT is_ai_override FROM strategy_outcomes WHERE id=?", (row_id,)
        ).fetchone()
        assert row[0] == 0

    def test_is_ai_override_defaults_to_false(self, tmp_path):
        """Existing callers that don't pass the param get 0."""
        db = _make_db(tmp_path)
        sc_id = _insert_chart(db)
        row_id = db.save_strategy_outcome(
            signal_chart_id=sc_id, strategy="quick_pop_managed", entered=True,
        )
        row = db._conn.execute(
            "SELECT is_ai_override FROM strategy_outcomes WHERE id=?", (row_id,)
        ).fetchone()
        assert row[0] == 0

    def test_ai_override_flag_independent_per_row(self, tmp_path):
        db = _make_db(tmp_path)
        sc_id = _insert_chart(db)
        ov_id   = db.save_strategy_outcome(sc_id, "quick_pop_managed", entered=True,  is_ai_override=True)
        norm_id = db.save_strategy_outcome(sc_id, "quick_pop",           entered=True,  is_ai_override=False)
        flags = {
            r[0]: r[1]
            for r in db._conn.execute(
                "SELECT id, is_ai_override FROM strategy_outcomes WHERE id IN (?,?)",
                (ov_id, norm_id),
            ).fetchall()
        }
        assert flags[ov_id]   == 1
        assert flags[norm_id] == 0


# ---------------------------------------------------------------------------
# query_ai_override_stats
# ---------------------------------------------------------------------------

class TestQueryAiOverrideStats:
    def _setup(self, tmp_path):
        db = _make_db(tmp_path)
        return db, str(tmp_path / "trader.db")

    def test_empty_db_returns_zero_overrides(self, tmp_path):
        db, path = self._setup(tmp_path)
        stats = db.query_ai_override_stats("quick_pop_managed", "quick_pop")
        assert stats["overrides"]["total"] == 0
        assert stats["overrides"]["closed"] == 0
        assert stats["overrides"]["win_rate_pct"] is None
        assert stats["rejections"]["total"] == 0

    def test_override_wins_and_losses_counted(self, tmp_path):
        db, path = self._setup(tmp_path)

        # 2 override trades: one win, one loss
        for pnl in (30.0, -15.0):
            sc_id = _insert_chart(db)
            db.save_ai_override_decision(
                strategy="quick_pop_managed", signal_chart_id=sc_id,
                symbol="TK", mint="m", skip_reason="ML_SKIP", decision="OVERRIDE",
                ml_score=1.8, pump_ratio=2.1, vol_trend="RISING", agent_reason="test",
            )
            ov_id = db.save_strategy_outcome(
                sc_id, "quick_pop_managed", entered=True, is_ai_override=True,
            )
            _close_outcome(db, ov_id, pnl)

        stats = db.query_ai_override_stats("quick_pop_managed", "quick_pop")
        ov = stats["overrides"]
        assert ov["total"]        == 2
        assert ov["closed"]       == 2
        assert ov["wins"]         == 1
        assert ov["win_rate_pct"] == pytest.approx(50.0)
        assert ov["avg_pnl_pct"]  == pytest.approx(7.5)  # (30 - 15) / 2

    def test_open_overrides_not_counted_in_closed(self, tmp_path):
        db, path = self._setup(tmp_path)
        sc_id = _insert_chart(db)
        db.save_ai_override_decision(
            strategy="quick_pop_managed", signal_chart_id=sc_id,
            symbol="TK", mint="m", skip_reason="ML_SKIP", decision="OVERRIDE",
            ml_score=2.0, pump_ratio=2.0, vol_trend="RISING", agent_reason="x",
        )
        # Trade entered but NOT closed yet
        db.save_strategy_outcome(sc_id, "quick_pop_managed", entered=True, is_ai_override=True)

        stats = db.query_ai_override_stats("quick_pop_managed", "quick_pop")
        assert stats["overrides"]["total"]  == 1
        assert stats["overrides"]["closed"] == 0
        assert stats["overrides"]["win_rate_pct"] is None

    def test_rejections_counterfactual_with_base_winner(self, tmp_path):
        db, path = self._setup(tmp_path)

        # AI rejected — base strategy entered and won
        sc_id = _insert_chart(db)
        db.save_ai_override_decision(
            strategy="quick_pop_managed", signal_chart_id=sc_id,
            symbol="BONK", mint="m", skip_reason="ML_SKIP", decision="REJECT",
            ml_score=0.9, pump_ratio=1.5, vol_trend="DYING", agent_reason="too weak",
        )
        db.save_strategy_outcome(sc_id, "quick_pop_managed", entered=False)
        base_id = db.save_strategy_outcome(sc_id, "quick_pop", entered=True)
        _close_outcome(db, base_id, pnl_pct=45.0, max_gain=80.0)

        stats = db.query_ai_override_stats("quick_pop_managed", "quick_pop")
        rej = stats["rejections"]
        assert rej["total"]            == 1
        assert rej["base_tracked"]     == 1
        assert rej["would_have_won"]   == 1
        assert rej["win_rate_pct"]     == pytest.approx(100.0)
        assert rej["avg_base_pnl_pct"] == pytest.approx(45.0)
        assert len(rej["sample"])      == 1
        assert rej["sample"][0]["symbol"]  == "BONK"
        assert rej["sample"][0]["pnl_pct"] == pytest.approx(45.0)

    def test_rejections_counterfactual_with_base_loser(self, tmp_path):
        db, path = self._setup(tmp_path)

        sc_id = _insert_chart(db)
        db.save_ai_override_decision(
            strategy="quick_pop_managed", signal_chart_id=sc_id,
            symbol="WIF", mint="m", skip_reason="CHART_SKIP", decision="REJECT",
            ml_score=None, pump_ratio=3.5, vol_trend="DYING", agent_reason="pump too big",
        )
        db.save_strategy_outcome(sc_id, "quick_pop_managed", entered=False)
        base_id = db.save_strategy_outcome(sc_id, "quick_pop", entered=True)
        _close_outcome(db, base_id, pnl_pct=-18.0, max_gain=3.0)

        stats = db.query_ai_override_stats("quick_pop_managed", "quick_pop")
        rej = stats["rejections"]
        assert rej["would_have_won"]   == 0
        assert rej["win_rate_pct"]     == pytest.approx(0.0)
        assert rej["avg_base_pnl_pct"] == pytest.approx(-18.0)

    def test_shadow_override_tracked_separately(self, tmp_path):
        db, path = self._setup(tmp_path)

        # Shadow: agent would have overridden; base strategy won
        sc_id = _insert_chart(db)
        db.save_ai_override_decision(
            strategy="quick_pop_managed", signal_chart_id=sc_id,
            symbol="PEPE", mint="m", skip_reason="ML_SKIP", decision="SHADOW_OVERRIDE",
            ml_score=1.5, pump_ratio=2.0, vol_trend="RISING", agent_reason="strong",
        )
        db.save_strategy_outcome(sc_id, "quick_pop_managed", entered=False)
        base_id = db.save_strategy_outcome(sc_id, "quick_pop", entered=True)
        _close_outcome(db, base_id, pnl_pct=25.0)

        stats = db.query_ai_override_stats("quick_pop_managed", "quick_pop")
        sh = stats["shadow_overrides"]
        assert sh["count"]        == 1
        assert sh["win_rate_pct"] == pytest.approx(100.0)
        assert sh["avg_pnl_pct"]  == pytest.approx(25.0)

    def test_shadow_reject_tracked_separately(self, tmp_path):
        db, path = self._setup(tmp_path)

        sc_id = _insert_chart(db)
        db.save_ai_override_decision(
            strategy="quick_pop_managed", signal_chart_id=sc_id,
            symbol="DOGE", mint="m", skip_reason="CHART_SKIP", decision="SHADOW_REJECT",
            ml_score=None, pump_ratio=4.0, vol_trend="DYING", agent_reason="too pumped",
        )
        db.save_strategy_outcome(sc_id, "quick_pop_managed", entered=False)
        base_id = db.save_strategy_outcome(sc_id, "quick_pop", entered=True)
        _close_outcome(db, base_id, pnl_pct=-20.0)

        stats = db.query_ai_override_stats("quick_pop_managed", "quick_pop")
        sr = stats["shadow_rejects"]
        assert sr["count"]        == 1
        assert sr["win_rate_pct"] == pytest.approx(0.0)   # -20% is a loss
        assert sr["avg_pnl_pct"]  == pytest.approx(-20.0)

    def test_rejection_sample_capped_at_ten(self, tmp_path):
        db, path = self._setup(tmp_path)
        for i in range(15):
            sc_id = _insert_chart(db, symbol=f"TK{i}", ts=f"2026-01-01T{i:02d}:00:00")
            db.save_ai_override_decision(
                strategy="quick_pop_managed", signal_chart_id=sc_id,
                symbol=f"TK{i}", mint=f"m{i}", skip_reason="ML_SKIP", decision="REJECT",
                ml_score=0.5, pump_ratio=1.0, vol_trend="FLAT", agent_reason="weak",
            )
            db.save_strategy_outcome(sc_id, "quick_pop_managed", entered=False)
            base_id = db.save_strategy_outcome(sc_id, "quick_pop", entered=True)
            _close_outcome(db, base_id, pnl_pct=10.0)

        stats = db.query_ai_override_stats("quick_pop_managed", "quick_pop")
        assert len(stats["rejections"]["sample"]) == 10


# ---------------------------------------------------------------------------
# summarize_candles
# ---------------------------------------------------------------------------

class TestSummarizeCandles:
    def test_empty_list_returns_empty_dict(self):
        assert summarize_candles([]) == {}

    def test_dict_candles_extracted(self):
        candles = [{"open": 0.001, "close": 0.002, "volume": 500.0}]
        result = summarize_candles(candles)
        assert result["recent"][0] == {"o": 0.001, "c": 0.002, "v": 500.0}

    def test_object_candles_extracted(self):
        c = SimpleNamespace(open=0.003, close=0.004, volume=200.0)
        result = summarize_candles([c])
        assert result["recent"][0]["o"] == pytest.approx(0.003)
        assert result["recent"][0]["c"] == pytest.approx(0.004)
        assert result["recent"][0]["v"] == pytest.approx(200.0)

    def test_capped_at_eight_bars(self):
        candles = [{"open": float(i), "close": float(i), "volume": 1.0} for i in range(12)]
        result = summarize_candles(candles)
        assert len(result["recent"]) == 8

    def test_returns_last_eight_not_first(self):
        candles = [{"open": float(i), "close": float(i), "volume": 1.0} for i in range(10)]
        result = summarize_candles(candles)
        # Last 8 = indices 2..9 → opens 2.0 through 9.0
        assert result["recent"][0]["o"] == pytest.approx(2.0)
        assert result["recent"][-1]["o"] == pytest.approx(9.0)

    def test_none_values_coerced_to_zero(self):
        candles = [{"open": None, "close": None, "volume": None}]
        result = summarize_candles(candles)
        bar = result["recent"][0]
        assert bar["o"] == pytest.approx(0.0)
        assert bar["c"] == pytest.approx(0.0)
        assert bar["v"] == pytest.approx(0.0)

    def test_single_bad_candle_skipped(self):
        candles = [{"open": "bad"}, {"open": 0.001, "close": 0.002, "volume": 10.0}]
        result = summarize_candles(candles)
        # First candle fails float() conversion — only second should appear
        assert len(result["recent"]) == 1
        assert result["recent"][0]["o"] == pytest.approx(0.001)


# ---------------------------------------------------------------------------
# log_override_decision
# ---------------------------------------------------------------------------

class TestLogOverrideDecision:
    def _decision(self, override=False, reanalyze=0.0, reason="test reason"):
        return {"override": override, "reanalyze_after_seconds": reanalyze, "reason": reason}

    def _ctx(self, ml_score=1.8, pump_ratio=2.1, vol_trend="RISING"):
        return {"ml_score": ml_score, "pump_ratio": pump_ratio, "vol_trend": vol_trend}

    def test_creates_log_file(self, tmp_path):
        with patch.dict(os.environ, {"AGENT_LOG_DIR": str(tmp_path)}):
            log_override_decision("quick_pop_managed", "BONK", "ML_SKIP",
                                  self._decision(override=True), self._ctx())
        log_file = tmp_path / "ai_override_quick_pop_managed.log"
        assert log_file.exists()

    def test_override_action_label(self, tmp_path):
        with patch.dict(os.environ, {"AGENT_LOG_DIR": str(tmp_path)}):
            log_override_decision("quick_pop_managed", "BONK", "ML_SKIP",
                                  self._decision(override=True, reason="good signal"), self._ctx())
        content = (tmp_path / "ai_override_quick_pop_managed.log").read_text()
        assert "OVERRIDE" in content
        assert "good signal" in content

    def test_reject_action_label(self, tmp_path):
        with patch.dict(os.environ, {"AGENT_LOG_DIR": str(tmp_path)}):
            log_override_decision("quick_pop_managed", "WIF", "CHART_SKIP",
                                  self._decision(override=False, reanalyze=0.0, reason="weak"),
                                  self._ctx())
        content = (tmp_path / "ai_override_quick_pop_managed.log").read_text()
        assert "REJECT" in content

    def test_reanalyze_action_label_and_delay(self, tmp_path):
        with patch.dict(os.environ, {"AGENT_LOG_DIR": str(tmp_path)}):
            log_override_decision("quick_pop_managed", "PEPE", "ML_SKIP",
                                  self._decision(override=False, reanalyze=120.0, reason="unclear"),
                                  self._ctx())
        content = (tmp_path / "ai_override_quick_pop_managed.log").read_text()
        assert "REANALYZE" in content
        assert "120" in content

    def test_shadow_prefix_added(self, tmp_path):
        with patch.dict(os.environ, {"AGENT_LOG_DIR": str(tmp_path)}):
            log_override_decision("quick_pop_managed", "BONK", "ML_SKIP",
                                  self._decision(override=True), self._ctx(), shadow=True)
        content = (tmp_path / "ai_override_quick_pop_managed.log").read_text()
        assert "SHADOW_OVERRIDE" in content

    def test_symbol_and_skip_reason_in_line(self, tmp_path):
        with patch.dict(os.environ, {"AGENT_LOG_DIR": str(tmp_path)}):
            log_override_decision("quick_pop_managed", "DOGE123", "POLICY_BLK",
                                  self._decision(), self._ctx())
        content = (tmp_path / "ai_override_quick_pop_managed.log").read_text()
        assert "DOGE123" in content
        assert "POLICY_BLK" in content

    def test_ml_score_none_shown_as_none(self, tmp_path):
        with patch.dict(os.environ, {"AGENT_LOG_DIR": str(tmp_path)}):
            log_override_decision("quick_pop_managed", "TKN", "ML_SKIP",
                                  self._decision(), self._ctx(ml_score=None))
        content = (tmp_path / "ai_override_quick_pop_managed.log").read_text()
        assert "None" in content

    def test_non_fatal_on_bad_dir(self):
        """Should not raise even if log dir can't be written."""
        with patch.dict(os.environ, {"AGENT_LOG_DIR": "/nonexistent_root_dir/subdir"}):
            # Should complete without raising
            log_override_decision("quick_pop_managed", "TKN", "ML_SKIP",
                                  self._decision(), self._ctx())

    def test_appends_multiple_lines(self, tmp_path):
        with patch.dict(os.environ, {"AGENT_LOG_DIR": str(tmp_path)}):
            for symbol in ("BONK", "WIF", "PEPE"):
                log_override_decision("quick_pop_managed", symbol, "ML_SKIP",
                                      self._decision(), self._ctx())
        lines = (tmp_path / "ai_override_quick_pop_managed.log").read_text().splitlines()
        assert len(lines) == 3


# ---------------------------------------------------------------------------
# _parse_and_validate
# ---------------------------------------------------------------------------

class TestParseAndValidate:
    def test_override_true_clears_reanalyze(self):
        import json
        raw = json.dumps({"override": True, "reanalyze_after_seconds": 120.0, "reason": "test"})
        result = _parse_and_validate(raw)
        assert result["override"] is True
        assert result["reanalyze_after_seconds"] == pytest.approx(0.0)

    def test_reanalyze_capped_at_600(self):
        import json
        raw = json.dumps({"override": False, "reanalyze_after_seconds": 9999.0, "reason": "x"})
        result = _parse_and_validate(raw)
        assert result["reanalyze_after_seconds"] == pytest.approx(600.0)

    def test_reanalyze_floored_at_zero(self):
        import json
        raw = json.dumps({"override": False, "reanalyze_after_seconds": -50.0, "reason": "x"})
        result = _parse_and_validate(raw)
        assert result["reanalyze_after_seconds"] == pytest.approx(0.0)

    def test_reason_truncated_at_120_chars(self):
        import json
        long_reason = "x" * 200
        raw = json.dumps({"override": False, "reanalyze_after_seconds": 0.0, "reason": long_reason})
        result = _parse_and_validate(raw)
        assert len(result["reason"]) == 120

    def test_bad_json_returns_default_reject(self):
        result = _parse_and_validate("not valid json at all {{{")
        assert result["override"] is False
        assert result["reanalyze_after_seconds"] == pytest.approx(0.0)
        assert isinstance(result["reason"], str)

    def test_all_keys_present_in_result(self):
        import json
        raw = json.dumps({"override": False, "reanalyze_after_seconds": 0.0, "reason": "ok"})
        result = _parse_and_validate(raw)
        assert set(result.keys()) == {"override", "reanalyze_after_seconds", "reason"}

    def test_markdown_json_block_parsed(self):
        """Model sometimes wraps JSON in markdown code fences."""
        import json
        inner = json.dumps({"override": True, "reanalyze_after_seconds": 0, "reason": "yes"})
        raw = f"```json\n{inner}\n```"
        result = _parse_and_validate(raw)
        assert result["override"] is True

    def test_reject_decision_no_reanalyze(self):
        import json
        raw = json.dumps({"override": False, "reanalyze_after_seconds": 0.0, "reason": "weak"})
        result = _parse_and_validate(raw)
        assert result["override"] is False
        assert result["reanalyze_after_seconds"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# StrategyConfig defaults
# ---------------------------------------------------------------------------

class TestStrategyConfigDefaults:
    def _minimal_cfg(self, **overrides) -> StrategyConfig:
        defaults = dict(
            name="test",
            buy_size_usd=30.0,
            stop_loss_pct=0.20,
            take_profit_levels=(TakeProfitLevel(multiple=1.5, sell_fraction_original=1.0),),
            trailing_stop_pct=0.22,
            starting_cash_usd=1000.0,
        )
        defaults.update(overrides)
        return StrategyConfig(**defaults)

    def test_use_ai_override_defaults_false(self):
        cfg = self._minimal_cfg()
        assert cfg.use_ai_override is False

    def test_use_ai_override_shadow_defaults_false(self):
        cfg = self._minimal_cfg()
        assert cfg.use_ai_override_shadow is False

    def test_use_ai_override_can_be_set_true(self):
        cfg = self._minimal_cfg(use_ai_override=True)
        assert cfg.use_ai_override is True

    def test_use_ai_override_shadow_can_be_set_true(self):
        cfg = self._minimal_cfg(use_ai_override_shadow=True)
        assert cfg.use_ai_override_shadow is True

    def test_both_can_coexist(self):
        """override=True and shadow=True is valid config (shadow ignored when override active)."""
        cfg = self._minimal_cfg(use_ai_override=True, use_ai_override_shadow=True)
        assert cfg.use_ai_override is True
        assert cfg.use_ai_override_shadow is True
