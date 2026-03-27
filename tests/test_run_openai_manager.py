from datetime import datetime, timezone

import scripts.run_openai_manager as loop


def test_should_trigger_on_bootstrap_signal_volume(monkeypatch):
    monkeypatch.setattr(loop, "_closed_count", lambda db_path: 3)
    monkeypatch.setattr(loop, "_signal_counts", lambda db_path: (48, 45))
    monkeypatch.setattr(loop, "_last_run_meta", lambda: (0, 0, 0, None))

    trigger, status = loop.should_trigger(
        "trader.db",
        every_closed=12,
        min_hours=3.0,
        min_closed_hours=10,
        every_signals=40,
        every_blocked=30,
        blocked_rate=0.85,
        heartbeat_signals=20,
        bootstrap_signals=30,
    )

    assert trigger is True
    assert "bootstrap_signals" in status["trigger_reasons"]
    assert "signal_flow" in status["trigger_reasons"]
    assert "high_block_rate" in status["trigger_reasons"]
    assert status["new_signals"] == 48
    assert status["new_blocked"] == 45


def test_should_not_trigger_on_small_sample(monkeypatch):
    last_run = datetime(2026, 3, 27, 8, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(loop, "_closed_count", lambda db_path: 3)
    monkeypatch.setattr(loop, "_signal_counts", lambda db_path: (10, 8))
    monkeypatch.setattr(loop, "_last_run_meta", lambda: (0, 0, 0, last_run))

    trigger, status = loop.should_trigger(
        "trader.db",
        every_closed=12,
        min_hours=3.0,
        min_closed_hours=10,
        every_signals=40,
        every_blocked=30,
        blocked_rate=0.85,
        heartbeat_signals=20,
        bootstrap_signals=30,
    )

    assert trigger is False
    assert status["trigger_reasons"] == []
