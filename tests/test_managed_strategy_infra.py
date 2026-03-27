import sqlite3

from trader.agents.base import query_regime_context
from trader.agents.managed_agent_base import ManagedAgentSpec, build_managed_prompt, validate_managed_delta
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


def test_build_managed_prompt_includes_regime_context():
    prompt = build_managed_prompt(
        _SPEC,
        {"base_strategy": "quick_pop", "mode": "balanced"},
        {"total_pnl_usd": 12.3},
        [{"base_strategy": "quick_pop", "mode": "lenient"}],
        [{"sell_reason": "TP1", "count": 2}],
        [{"symbol": "ABC"}],
        [{"bucket": "5.0-5.9", "count": 1}],
        {"managed_strategy_recent": {"block_rate": 0.9}},
    )
    assert "Recent regime/context snapshot" in prompt
    assert "\"block_rate\": 0.9" in prompt


def test_query_regime_context_summarises_recent_state(tmp_path):
    db_path = tmp_path / "regime.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE signal_charts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            symbol TEXT NOT NULL,
            mint TEXT NOT NULL,
            entry_price REAL NOT NULL,
            pump_ratio REAL,
            price_change_30m_pct REAL,
            unique_wallet_5m INTEGER,
            market_cap_usd REAL,
            liquidity_usd REAL,
            source_channel TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE strategy_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_chart_id INTEGER NOT NULL,
            strategy TEXT NOT NULL,
            entered INTEGER NOT NULL DEFAULT 0,
            outcome_pnl_pct REAL,
            outcome_max_gain_pct REAL,
            closed INTEGER NOT NULL DEFAULT 0,
            source_channel TEXT NOT NULL DEFAULT '',
            skip_reason TEXT
        );
        """
    )
    conn.execute(
        "INSERT INTO signal_charts (id, ts, symbol, mint, entry_price, pump_ratio, price_change_30m_pct, unique_wallet_5m, market_cap_usd, liquidity_usd, source_channel) VALUES (1, '2026-03-27T00:00:00+00:00', 'AAA', 'mint1', 1.0, 3.0, 120.0, 25, 150000.0, 40000.0, 'WizzyTrades')"
    )
    conn.execute(
        "INSERT INTO signal_charts (id, ts, symbol, mint, entry_price, pump_ratio, price_change_30m_pct, unique_wallet_5m, market_cap_usd, liquidity_usd, source_channel) VALUES (2, '2026-03-27T00:05:00+00:00', 'BBB', 'mint2', 1.0, 1.8, 40.0, 10, 80000.0, 15000.0, 'WizzyCasino')"
    )
    conn.execute(
        "INSERT INTO strategy_outcomes (signal_chart_id, strategy, entered, outcome_pnl_pct, outcome_max_gain_pct, closed, source_channel, skip_reason) VALUES (1, 'open_ai_managed', 1, 12.0, 20.0, 1, 'WizzyTrades', NULL)"
    )
    conn.execute(
        "INSERT INTO strategy_outcomes (signal_chart_id, strategy, entered, outcome_pnl_pct, outcome_max_gain_pct, closed, source_channel, skip_reason) VALUES (2, 'open_ai_managed', 0, NULL, NULL, 0, 'WizzyCasino', 'ML_SKIP')"
    )
    conn.execute(
        "INSERT INTO strategy_outcomes (signal_chart_id, strategy, entered, outcome_pnl_pct, outcome_max_gain_pct, closed, source_channel, skip_reason) VALUES (1, 'quick_pop', 1, 15.0, 22.0, 1, 'WizzyTrades', NULL)"
    )
    conn.commit()
    conn.close()

    result = query_regime_context(str(db_path), "open_ai_managed", base_strategy="quick_pop", lookback_signals=10)
    assert result["managed_strategy_recent"]["signals"] == 2
    assert result["managed_strategy_recent"]["blocked"] == 1
    assert result["managed_strategy_recent"]["skip_reasons"]["ML_SKIP"] == 1
    assert result["base_strategy_recent"]["signals"] == 1
    assert result["market_recent"]["signals"] == 2
