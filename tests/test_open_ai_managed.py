from trader.config import Config
from trader.agents.strategy_tuner import assert_owned_config_changes
from trader.strategies.registry import (
    _OPEN_AI_MANAGED_BASES,
    _OPEN_AI_MANAGED_MODES,
    build_runners,
)
from trader.trading.strategy import InfiniteMoonbagRunner


def _cfg() -> Config:
    return Config(
        birdeye_api_key="test",
        tg_api_id=1,
        tg_api_hash="test",
        channel_usernames=("test_channel",),
    )


def test_open_ai_managed_mode_table_is_complete():
    assert set(_OPEN_AI_MANAGED_BASES) == {
        "quick_pop", "trend_rider", "safe_bet", "infinite_moonbag"
    }
    assert set(_OPEN_AI_MANAGED_MODES["quick_pop"]) == {
        "allow_all", "lenient", "balanced", "strict", "block_all"
    }


def test_build_runners_includes_open_ai_managed():
    runners = build_runners(_cfg(), db=None)
    by_name = {runner.name: runner for runner in runners}

    assert "open_ai_managed" in by_name

    cfg = by_name["open_ai_managed"].cfg
    assert cfg.ml_training_strategy == "quick_pop"
    assert cfg.use_ml_filter is True
    assert cfg.use_chart_filter is False


def test_open_ai_managed_defaults_to_standard_runner():
    runners = build_runners(_cfg(), db=None)
    runner = next(r for r in runners if r.name == "open_ai_managed")
    assert not isinstance(runner, InfiniteMoonbagRunner)


def test_openai_manager_ownership_guard_allows_only_its_strategy():
    before = {
        "_meta": {"version": 1},
        "open_ai_managed": {"mode": "balanced"},
        "deep_seek_managed": {"mode": "strict"},
    }
    after = {
        "_meta": {"version": 1, "openai_manager_last_run_at": "2026-03-27T01:00:00+00:00"},
        "open_ai_managed": {"mode": "lenient"},
        "deep_seek_managed": {"mode": "strict"},
    }
    assert_owned_config_changes(
        before,
        after,
        owned_strategy="open_ai_managed",
        allowed_meta_prefixes=("openai_manager_",),
    )
