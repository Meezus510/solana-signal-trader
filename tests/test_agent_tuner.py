"""
tests/test_agent_tuner.py — Unit tests for the strategy tuner agent.

Covers:
- Permission tier enforcement (_validate_strategy_delta)
- TP level validation (bounds, ascending multiples, total fraction)
- _parse_response (valid JSON, markdown-wrapped, malformed)
- load_config / save_config (round-trip, atomic write)
- _apply_delta (merges delta into config dict in-place)
- registry._load_strategy_overrides (graceful fallback on missing/corrupt file)

No network calls, no Anthropic API, no real DB.
Run with: pytest tests/test_agent_tuner.py -v
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path

from trader.agents.strategy_tuner import (
    CONTROLLED_STRATEGIES,
    FULL_CONTROL_STRATEGIES,
    ML_ONLY_STRATEGIES,
    _ML_ONLY_ALLOWED_KEYS,
    _apply_delta,
    _parse_response,
    _validate_strategy_delta,
    load_config,
    save_config,
)


# ---------------------------------------------------------------------------
# Permission tier constants
# ---------------------------------------------------------------------------

class TestPermissionTiers:
    def test_controlled_is_union_of_tiers(self):
        assert CONTROLLED_STRATEGIES == FULL_CONTROL_STRATEGIES | ML_ONLY_STRATEGIES

    def test_tiers_are_disjoint(self):
        assert FULL_CONTROL_STRATEGIES.isdisjoint(ML_ONLY_STRATEGIES)

    def test_quick_pop_base_not_controlled(self):
        assert "quick_pop" not in CONTROLLED_STRATEGIES

    def test_quick_pop_chart_ml_is_ml_only(self):
        assert "quick_pop_chart_ml" in ML_ONLY_STRATEGIES

    def test_trend_rider_is_full_control(self):
        assert "trend_rider" in FULL_CONTROL_STRATEGIES
        assert "trend_rider" not in ML_ONLY_STRATEGIES

    def test_ml_only_allowed_keys_contains_ml_params(self):
        for key in ("use_ml_filter", "ml_min_score", "ml_k", "ml_halflife_days",
                    "ml_score_low_pct", "ml_score_high_pct",
                    "ml_high_score_threshold", "ml_max_score_threshold",
                    "ml_size_multiplier", "ml_max_size_multiplier"):
            assert key in _ML_ONLY_ALLOWED_KEYS

    def test_ml_only_allowed_keys_excludes_trade_params(self):
        for key in ("stop_loss_pct", "trailing_stop_pct", "tp_levels",
                    "use_chart_filter", "use_reanalyze", "pump_ratio_max"):
            assert key not in _ML_ONLY_ALLOWED_KEYS


# ---------------------------------------------------------------------------
# _validate_strategy_delta — ML_ONLY tier
# ---------------------------------------------------------------------------

class TestValidateDeltaMLOnly:
    """quick_pop_chart_ml can only change ML params."""

    def test_strips_stop_loss(self):
        result = _validate_strategy_delta("quick_pop_chart_ml", {"stop_loss_pct": 0.15, "reason": "x"})
        assert "stop_loss_pct" not in result

    def test_strips_trailing_stop(self):
        result = _validate_strategy_delta("quick_pop_chart_ml", {"trailing_stop_pct": 0.20, "reason": "x"})
        assert "trailing_stop_pct" not in result

    def test_strips_tp_levels(self):
        result = _validate_strategy_delta("quick_pop_chart_ml", {"tp_levels": [[1.5, 0.6]], "reason": "x"})
        assert "tp_levels" not in result

    def test_strips_use_chart_filter(self):
        result = _validate_strategy_delta("quick_pop_chart_ml", {"use_chart_filter": False, "reason": "x"})
        assert "use_chart_filter" not in result

    def test_strips_use_reanalyze(self):
        result = _validate_strategy_delta("quick_pop_chart_ml", {"use_reanalyze": True, "reason": "x"})
        assert "use_reanalyze" not in result

    def test_strips_pump_ratio_max(self):
        result = _validate_strategy_delta("quick_pop_chart_ml", {"pump_ratio_max": 4.0, "reason": "x"})
        assert "pump_ratio_max" not in result

    def test_keeps_use_ml_filter(self):
        result = _validate_strategy_delta("quick_pop_chart_ml", {"use_ml_filter": True, "reason": "x"})
        assert result.get("use_ml_filter") is True

    def test_keeps_ml_min_score(self):
        result = _validate_strategy_delta("quick_pop_chart_ml", {"ml_min_score": 6.0, "reason": "x"})
        assert result.get("ml_min_score") == 6.0

    def test_keeps_ml_k(self):
        result = _validate_strategy_delta("quick_pop_chart_ml", {"ml_k": 7, "reason": "x"})
        assert result.get("ml_k") == 7

    def test_keeps_ml_halflife(self):
        result = _validate_strategy_delta("quick_pop_chart_ml", {"ml_halflife_days": 10.0, "reason": "x"})
        assert result.get("ml_halflife_days") == 10.0

    def test_keeps_ml_score_low_high(self):
        result = _validate_strategy_delta("quick_pop_chart_ml", {
            "ml_score_low_pct": -40.0, "ml_score_high_pct": 100.0, "reason": "x",
        })
        assert result.get("ml_score_low_pct") == -40.0
        assert result.get("ml_score_high_pct") == 100.0

    def test_passes_reason_through(self):
        result = _validate_strategy_delta("quick_pop_chart_ml", {"use_ml_filter": True, "reason": "test reason"})
        assert result.get("reason") == "test reason"

    def test_empty_delta_returns_empty(self):
        result = _validate_strategy_delta("quick_pop_chart_ml", {})
        assert result == {}


# ---------------------------------------------------------------------------
# _validate_strategy_delta — FULL_CONTROL tier
# ---------------------------------------------------------------------------

class TestValidateDeltaFullControl:
    """trend_rider / infinite_moonbag have full control."""

    def test_allows_stop_loss(self):
        result = _validate_strategy_delta("trend_rider", {"stop_loss_pct": 0.25})
        assert "stop_loss_pct" in result

    def test_allows_trailing_stop(self):
        result = _validate_strategy_delta("trend_rider", {"trailing_stop_pct": 0.20})
        assert "trailing_stop_pct" in result

    def test_allows_use_ml_filter(self):
        result = _validate_strategy_delta("trend_rider_chart_reanalyze", {"use_ml_filter": True})
        assert result.get("use_ml_filter") is True

    def test_allows_use_chart_filter(self):
        result = _validate_strategy_delta("trend_rider_chart_reanalyze", {"use_chart_filter": False})
        assert result.get("use_chart_filter") is False

    def test_allows_use_reanalyze(self):
        result = _validate_strategy_delta("trend_rider_chart_reanalyze", {"use_reanalyze": True})
        assert result.get("use_reanalyze") is True

    def test_allows_pump_ratio_max(self):
        result = _validate_strategy_delta("trend_rider_chart_reanalyze", {"pump_ratio_max": 5.0})
        assert result.get("pump_ratio_max") == 5.0

    def test_allows_tp_levels_trend_rider(self):
        result = _validate_strategy_delta("trend_rider", {"tp_levels": [[1.8, 0.50]]})
        assert result.get("tp_levels") == [[1.8, 0.50]]

    def test_clamps_stop_loss_to_guardrail_max(self):
        # GUARDRAILS stop_loss_pct max is 0.35
        result = _validate_strategy_delta("trend_rider", {"stop_loss_pct": 0.80})
        assert result.get("stop_loss_pct") == pytest.approx(0.35)

    def test_clamps_stop_loss_to_guardrail_min(self):
        result = _validate_strategy_delta("trend_rider", {"stop_loss_pct": 0.01})
        assert result.get("stop_loss_pct") == pytest.approx(0.10)

    def test_clamps_pump_ratio_to_extra_guardrail_max(self):
        # _EXTRA_GUARDRAILS pump_ratio_max max is 8.0
        result = _validate_strategy_delta("trend_rider_chart_reanalyze", {"pump_ratio_max": 20.0})
        assert result.get("pump_ratio_max") == pytest.approx(8.0)

    def test_moonbag_timeout_stripped(self):
        # InfiniteMoonbag has no timeout
        result = _validate_strategy_delta("infinite_moonbag", {"timeout_minutes": 60.0})
        assert "timeout_minutes" not in result

    def test_ml_k_cast_to_int(self):
        result = _validate_strategy_delta("trend_rider_chart_reanalyze", {"ml_k": 6.7})
        assert result.get("ml_k") == 7
        assert isinstance(result.get("ml_k"), int)

    def test_bool_must_be_bool(self):
        # String "true" is not a valid bool
        result = _validate_strategy_delta("trend_rider_chart_reanalyze", {"use_ml_filter": "true"})
        assert "use_ml_filter" not in result


# ---------------------------------------------------------------------------
# _validate_tp_levels
# ---------------------------------------------------------------------------

class TestValidateTpLevels:
    def test_valid_trend_rider_tp(self):
        result = _validate_strategy_delta("trend_rider", {"tp_levels": [[1.8, 0.50]]})
        assert result.get("tp_levels") == [[1.8, 0.50]]

    def test_tp_multiple_below_min_rejected(self):
        # trend_rider tp1 min_multiple=1.3
        result = _validate_strategy_delta("trend_rider", {"tp_levels": [[1.0, 0.50]]})
        assert "tp_levels" not in result

    def test_tp_multiple_above_max_rejected(self):
        # trend_rider tp1 max_multiple=3.0
        result = _validate_strategy_delta("trend_rider", {"tp_levels": [[5.0, 0.50]]})
        assert "tp_levels" not in result

    def test_tp_fraction_below_min_rejected(self):
        # trend_rider tp1 min_fraction=0.30
        result = _validate_strategy_delta("trend_rider", {"tp_levels": [[1.8, 0.10]]})
        assert "tp_levels" not in result

    def test_tp_fraction_above_max_rejected(self):
        # trend_rider tp1 max_fraction=0.75
        result = _validate_strategy_delta("trend_rider", {"tp_levels": [[1.8, 0.99]]})
        assert "tp_levels" not in result

    def test_tp_total_fraction_above_1_rejected(self):
        result = _validate_strategy_delta("infinite_moonbag", {
            "tp_levels": [[1.5, 0.40], [2.0, 0.30], [3.0, 0.25], [5.0, 0.20]]
        })
        assert "tp_levels" not in result

    def test_tp_non_ascending_multiples_rejected(self):
        result = _validate_strategy_delta("infinite_moonbag", {
            "tp_levels": [[2.0, 0.15], [1.5, 0.10], [3.0, 0.10], [5.0, 0.08]]
        })
        assert "tp_levels" not in result

    def test_tp_wrong_count_rejected(self):
        # trend_rider expects exactly 1 level
        result = _validate_strategy_delta("trend_rider", {
            "tp_levels": [[1.5, 0.40], [2.5, 0.35]]
        })
        assert "tp_levels" not in result

    def test_valid_moonbag_four_levels(self):
        levels = [[1.5, 0.15], [2.0, 0.10], [3.0, 0.10], [5.0, 0.08]]
        result = _validate_strategy_delta("infinite_moonbag", {"tp_levels": levels})
        assert result.get("tp_levels") == levels

    def test_tp_not_list_rejected(self):
        result = _validate_strategy_delta("trend_rider", {"tp_levels": "not_a_list"})
        assert "tp_levels" not in result


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------

class TestParseResponse:
    def test_valid_json(self):
        raw = '{"stop_loss_pct": 0.25, "reason": "test"}'
        result = _parse_response(raw)
        assert result == {"stop_loss_pct": 0.25, "reason": "test"}

    def test_markdown_wrapped_json(self):
        raw = '```json\n{"ml_k": 7}\n```'
        result = _parse_response(raw)
        assert result == {"ml_k": 7}

    def test_markdown_no_lang_hint(self):
        raw = '```\n{"use_ml_filter": true}\n```'
        result = _parse_response(raw)
        assert result == {"use_ml_filter": True}

    def test_invalid_json_returns_empty(self):
        result = _parse_response("this is not json at all")
        assert result == {}

    def test_empty_string_returns_empty(self):
        result = _parse_response("")
        assert result == {}

    def test_partial_json_returns_empty(self):
        result = _parse_response('{"stop_loss_pct": ')
        assert result == {}


# ---------------------------------------------------------------------------
# load_config / save_config
# ---------------------------------------------------------------------------

class TestConfigIO:
    def test_round_trip(self, tmp_path):
        cfg_path = tmp_path / "strategy_config.json"
        data = {"_meta": {"version": 1}, "trend_rider": {"stop_loss_pct": 0.25}}
        save_config(data, path=cfg_path)
        loaded = load_config(path=cfg_path)
        assert loaded == data

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(path=tmp_path / "nonexistent.json")

    def test_file_is_valid_json(self, tmp_path):
        cfg_path = tmp_path / "strategy_config.json"
        data = {"x": 1}
        save_config(data, path=cfg_path)
        raw = cfg_path.read_text()
        parsed = json.loads(raw)
        assert parsed == data

    def test_save_overwrites(self, tmp_path):
        cfg_path = tmp_path / "strategy_config.json"
        save_config({"a": 1}, path=cfg_path)
        save_config({"b": 2}, path=cfg_path)
        loaded = load_config(path=cfg_path)
        assert loaded == {"b": 2}
        assert "a" not in loaded

    def test_no_leftover_tmp_files(self, tmp_path):
        cfg_path = tmp_path / "strategy_config.json"
        save_config({"x": 1}, path=cfg_path)
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == []


# ---------------------------------------------------------------------------
# _apply_delta
# ---------------------------------------------------------------------------

class TestApplyDelta:
    def test_scalar_overwritten(self):
        config = {"trend_rider": {"stop_loss_pct": 0.30}}
        _apply_delta("trend_rider", {"stop_loss_pct": 0.25}, config)
        assert config["trend_rider"]["stop_loss_pct"] == 0.25

    def test_tp_levels_replaced(self):
        config = {"trend_rider": {"tp_levels": [[1.8, 0.50]]}}
        _apply_delta("trend_rider", {"tp_levels": [[2.0, 0.50]]}, config)
        assert config["trend_rider"]["tp_levels"] == [[2.0, 0.50]]

    def test_reason_not_written_to_config(self):
        config = {"trend_rider": {}}
        _apply_delta("trend_rider", {"stop_loss_pct": 0.25, "reason": "test"}, config)
        assert "reason" not in config["trend_rider"]

    def test_creates_strategy_key_if_missing(self):
        config = {}
        _apply_delta("trend_rider", {"stop_loss_pct": 0.25}, config)
        assert "trend_rider" in config
        assert config["trend_rider"]["stop_loss_pct"] == 0.25

    def test_unrelated_keys_preserved(self):
        config = {"trend_rider": {"stop_loss_pct": 0.30, "trailing_stop_pct": 0.25}}
        _apply_delta("trend_rider", {"stop_loss_pct": 0.20}, config)
        assert config["trend_rider"]["trailing_stop_pct"] == 0.25

    def test_bool_written_correctly(self):
        config = {"trend_rider_chart_reanalyze": {"use_ml_filter": False}}
        _apply_delta("trend_rider_chart_reanalyze", {"use_ml_filter": True}, config)
        assert config["trend_rider_chart_reanalyze"]["use_ml_filter"] is True


# ---------------------------------------------------------------------------
# registry._load_strategy_overrides — graceful fallbacks
# ---------------------------------------------------------------------------

class TestLoadStrategyOverrides:
    def test_missing_file_returns_empty(self, tmp_path, monkeypatch):
        import trader.strategies.registry as reg
        monkeypatch.setattr(reg, "_CONFIG_PATH", tmp_path / "nonexistent.json")
        result = reg._load_strategy_overrides()
        assert result == {}

    def test_corrupt_json_returns_empty(self, tmp_path, monkeypatch):
        import trader.strategies.registry as reg
        bad = tmp_path / "strategy_config.json"
        bad.write_text("{ this is not valid json }")
        monkeypatch.setattr(reg, "_CONFIG_PATH", bad)
        result = reg._load_strategy_overrides()
        assert result == {}

    def test_only_controlled_strategies_returned(self, tmp_path, monkeypatch):
        import trader.strategies.registry as reg
        cfg = tmp_path / "strategy_config.json"
        data = {
            "trend_rider": {"stop_loss_pct": 0.20},
            "quick_pop": {"stop_loss_pct": 0.15},   # not in _CONTROLLED
            "_meta": {"version": 1},
        }
        cfg.write_text(json.dumps(data))
        monkeypatch.setattr(reg, "_CONFIG_PATH", cfg)
        result = reg._load_strategy_overrides()
        assert "trend_rider" in result
        assert "quick_pop" not in result
        assert "_meta" not in result

    def test_quick_pop_chart_ml_included(self, tmp_path, monkeypatch):
        import trader.strategies.registry as reg
        cfg = tmp_path / "strategy_config.json"
        data = {"quick_pop_chart_ml": {"use_ml_filter": True, "ml_k": 7}}
        cfg.write_text(json.dumps(data))
        monkeypatch.setattr(reg, "_CONFIG_PATH", cfg)
        result = reg._load_strategy_overrides()
        assert "quick_pop_chart_ml" in result
        assert result["quick_pop_chart_ml"]["ml_k"] == 7
