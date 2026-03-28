import sqlite3

from trader.agents.base import query_regime_context, query_strategy_pnl_snapshots
from trader.agents.managed_agent_base import ManagedAgentSpec, build_managed_prompt, validate_managed_delta
from trader.analysis.managed_backtest import _load_rows, resolve_managed_config
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
        [{"mode": "lenient", "total_pnl_usd": 10.0}],
        [{"sell_reason": "TP1", "count": 2}],
        [{"symbol": "ABC"}],
        [{"bucket": "5.0-5.9", "count": 1}],
        {"managed_strategy_recent": {"block_rate": 0.9}},
        ["2026-03-28T00:00:00Z | openai_manager | open_ai_managed | mode: balanced → lenient"],
        [{"strategy": "anthropic_managed", "total_pnl_usd": -10.0, "daily_pnl_usd": -2.0, "closed_trades": 3}],
        [{"strategy": "quick_pop", "total_pnl_usd": 25.0, "daily_pnl_usd": 3.0, "closed_trades": 5}],
        {"quick_pop": {"current_config": {"peak_drop_exit_pct": 0.12}}},
    )
    assert "Recent regime/context snapshot" in prompt
    assert "\"block_rate\": 0.9" in prompt
    assert "Recent-window comparison for current base only" in prompt
    assert "Your own most recent changes" in prompt
    assert "Other AI agents: pnl only" in prompt
    assert "Base and non-AI managed strategy performance" in prompt
    assert "Base and non-AI managed strategy configs/modes" in prompt


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


def test_load_rows_respects_date_range(tmp_path):
    db_path = tmp_path / "managed_backtest.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE signal_charts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            symbol TEXT NOT NULL,
            mint TEXT NOT NULL,
            entry_price REAL NOT NULL,
            candles_json TEXT NOT NULL,
            candles_1m_json TEXT,
            candles_1s_json TEXT,
            pair_stats_json TEXT,
            source_channel TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE strategy_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_chart_id INTEGER NOT NULL,
            strategy TEXT NOT NULL,
            entered INTEGER NOT NULL DEFAULT 0,
            outcome_pnl_pct REAL,
            position_peak_pnl_pct REAL,
            position_peak_ts TEXT,
            position_trough_pnl_pct REAL,
            position_trough_ts TEXT,
            closed INTEGER NOT NULL DEFAULT 0,
            source_channel TEXT NOT NULL DEFAULT ''
        );
        """
    )
    candles = '[{"o":1,"c":1.1,"l":0.9,"v":10},{"o":1.1,"c":1.2,"l":1.0,"v":12},{"o":1.2,"c":1.3,"l":1.1,"v":11}]'
    pair_stats = '{"buy_volume_1h": 10, "total_volume_1h": 20}'
    conn.execute(
        "INSERT INTO signal_charts (id, ts, symbol, mint, entry_price, candles_json, candles_1m_json, candles_1s_json, pair_stats_json, source_channel) VALUES (1, '2026-03-25T00:00:00+00:00', 'A', 'm1', 1.0, ?, ?, '[]', ?, 'WizzyTrades')",
        (candles, candles, pair_stats),
    )
    conn.execute(
        "INSERT INTO signal_charts (id, ts, symbol, mint, entry_price, candles_json, candles_1m_json, candles_1s_json, pair_stats_json, source_channel) VALUES (2, '2026-03-28T00:00:00+00:00', 'B', 'm2', 1.0, ?, ?, '[]', ?, 'WizzyCasino')",
        (candles, candles, pair_stats),
    )
    conn.execute(
        "INSERT INTO strategy_outcomes (signal_chart_id, strategy, entered, outcome_pnl_pct, position_peak_pnl_pct, position_peak_ts, position_trough_pnl_pct, position_trough_ts, closed, source_channel) VALUES (1, 'quick_pop', 1, 10.0, 20.0, '2026-03-25T00:05:00+00:00', -5.0, '2026-03-25T00:02:00+00:00', 1, 'WizzyTrades')"
    )
    conn.execute(
        "INSERT INTO strategy_outcomes (signal_chart_id, strategy, entered, outcome_pnl_pct, position_peak_pnl_pct, position_peak_ts, position_trough_pnl_pct, position_trough_ts, closed, source_channel) VALUES (2, 'quick_pop', 1, 12.0, 25.0, '2026-03-28T00:05:00+00:00', -4.0, '2026-03-28T00:02:00+00:00', 1, 'WizzyCasino')"
    )
    conn.commit()
    conn.close()

    all_rows = _load_rows(str(db_path), "quick_pop")
    recent_rows = _load_rows(str(db_path), "quick_pop", date_from="2026-03-27T00:00:00+00:00")
    old_rows = _load_rows(str(db_path), "quick_pop", date_to="2026-03-26T00:00:00+00:00")

    assert len(all_rows) == 2
    assert len(recent_rows) == 1
    assert recent_rows[0]["ts"] == "2026-03-28T00:00:00+00:00"
    assert len(old_rows) == 1
    assert old_rows[0]["ts"] == "2026-03-25T00:00:00+00:00"


def test_query_strategy_pnl_snapshots_returns_compact_pnl_only(tmp_path):
    db_path = tmp_path / "pnl.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE signal_charts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            symbol TEXT NOT NULL,
            mint TEXT NOT NULL,
            entry_price REAL NOT NULL
        );
        CREATE TABLE strategy_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_chart_id INTEGER NOT NULL,
            strategy TEXT NOT NULL,
            entered INTEGER NOT NULL DEFAULT 0,
            outcome_pnl_usd REAL,
            closed INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    conn.execute("INSERT INTO signal_charts (id, ts, symbol, mint, entry_price) VALUES (1, '2026-03-28T00:00:00+00:00', 'A', 'm1', 1.0)")
    conn.execute("INSERT INTO signal_charts (id, ts, symbol, mint, entry_price) VALUES (2, '2026-03-27T00:00:00+00:00', 'B', 'm2', 1.0)")
    conn.execute("INSERT INTO strategy_outcomes (signal_chart_id, strategy, entered, outcome_pnl_usd, closed) VALUES (1, 'open_ai_managed', 1, 5.5, 1)")
    conn.execute("INSERT INTO strategy_outcomes (signal_chart_id, strategy, entered, outcome_pnl_usd, closed) VALUES (2, 'open_ai_managed', 1, -2.0, 1)")
    conn.commit()
    conn.close()

    rows = query_strategy_pnl_snapshots(str(db_path), ["open_ai_managed", "anthropic_managed"])
    assert rows[0]["strategy"] == "open_ai_managed"
    assert rows[0]["total_pnl_usd"] == 3.5
    assert "mode" not in rows[0]
    assert rows[1]["strategy"] == "anthropic_managed"
    assert rows[1]["closed_trades"] == 0
