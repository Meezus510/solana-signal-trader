from trader.agents.managed_agent_base import ManagedAgentSpec, validate_managed_delta
from trader.analysis.managed_backtest import resolve_managed_config
from trader.strategies.registry import MANAGED_STRATEGY_SPECS, resolve_managed_strategy_config


_SPEC = ManagedAgentSpec(
    strategy_name="open_ai_managed",
    agent_name="openai_manager",
    meta_prefix="openai_manager_",
    default_model="gpt-5.4-mini",
    allowed_scalars={
        "ml_k",
        "stop_loss_pct",
        "holder_count_max",
    },
    allowed_bools={"use_ml_filter"},
    ranges={
        "ml_k": (1.0, 25.0),
        "stop_loss_pct": (0.01, 0.50),
        "holder_count_max": (1.0, 100000.0),
    },
)


def test_managed_strategy_spec_registry_contains_openai():
    assert "open_ai_managed" in MANAGED_STRATEGY_SPECS
    assert "anthropic_managed" in MANAGED_STRATEGY_SPECS


def test_resolve_managed_strategy_config_falls_back_to_defaults():
    base, mode, resolved = resolve_managed_strategy_config(
        "open_ai_managed",
        {"base_strategy": "not_real", "mode": "nope"},
    )
    assert base == "quick_pop"
    assert mode == "balanced"
    assert resolved["base_strategy"] == "quick_pop"
    assert resolved["mode"] == "balanced"


def test_resolve_managed_config_backtest_wrapper_matches_registry():
    base, resolved = resolve_managed_config("open_ai_managed", {"mode": "strict"})
    assert base == "quick_pop"
    assert resolved["mode"] == "strict"


def test_validate_managed_delta_allows_owned_base_and_mode():
    result = validate_managed_delta(
        _SPEC,
        {"base_strategy": "trend_rider", "mode": "strict", "use_ml_filter": True},
    )
    assert result["base_strategy"] == "trend_rider"
    assert result["mode"] == "strict"
    assert result["use_ml_filter"] is True


def test_validate_managed_delta_clamps_and_casts():
    result = validate_managed_delta(
        _SPEC,
        {"ml_k": 99, "stop_loss_pct": -1, "holder_count_max": 1234.6},
    )
    assert result["ml_k"] == 25
    assert result["stop_loss_pct"] == 0.01
    assert result["holder_count_max"] == 1235
