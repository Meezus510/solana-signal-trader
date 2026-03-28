"""
trader/agents/base.py — Shared infrastructure for all trading agents.

Provides:
    GUARDRAILS        — hardcoded (min, max) bands for every tunable parameter.
    validate_delta()  — clamps an agent's proposed delta to guardrail bands.
    query_score_buckets()   — score-bucket win/loss/avg-pnl breakdown from chart_snapshots.
    query_exit_stats()      — sell-reason breakdown (count, avg pnl, avg hold, avg max gain).
    query_recent_trades()   — last N closed positions for context.
    query_regime_context()  — recent market/strategy/channel state for mode selection.
    query_strategy_pnl_snapshots() — compact total/daily pnl snapshots by strategy.

The guardrail bands are the single source of truth for what the agents are
allowed to change. Tighten or widen them here — never in a prompt.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent action log — append-only audit trail of every config change.
# ---------------------------------------------------------------------------

_LOG_DIR = Path(os.getenv("AGENT_LOG_DIR", "logs"))


def agent_log_path(strategy: str) -> Path:
    """Return the per-strategy agent action log path."""
    return _LOG_DIR / f"agent_actions_{strategy}.log"


def log_agent_action(
    agent: str,
    strategy: str,
    delta: dict[str, Any],
    before: dict[str, Any],
) -> None:
    """
    Append one line per changed key to logs/agent_actions_{strategy}.log.

    Format:
        2026-03-17T22:01:05Z | strategy_tuner | quick_pop_managed | ml_min_score: 5.0 → 6.0 | reason: ...
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    reason = delta.get("reason", "")
    changes = []
    for k, v in delta.items():
        if k == "reason":
            continue
        if before.get(k) == v:
            continue
        changes.append(f"{k}: {before.get(k, '?')} → {v}")
    if not changes:
        return

    log_path = agent_log_path(strategy)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as f:
            for change in changes:
                f.write(f"{ts} | {agent} | {strategy} | {change} | reason: {reason}\n")
    except OSError as exc:
        logger.warning("[agent] Could not write agent action log: %s", exc)

# ---------------------------------------------------------------------------
# Guardrail bands — (min_allowed, max_allowed) for each tunable parameter.
# Agents propose values; Python clamps them to these ranges before applying.
# ---------------------------------------------------------------------------

GUARDRAILS: dict[str, tuple[float, float]] = {
    # ML score thresholds
    "ml_min_score":             (2.0, 7.0),   # floor for taking any trade
    "ml_high_score_threshold":  (6.0, 9.0),   # score at which size doubles
    "ml_max_score_threshold":   (8.0, 9.9),   # score at which size triples

    # ML size multipliers
    "ml_size_multiplier":       (1.2, 3.0),   # 2× band (high confidence)
    "ml_max_size_multiplier":   (2.0, 5.0),   # 3× band (max confidence)

    # Exit parameters
    "stop_loss_pct":            (0.10, 0.35), # initial stop % below entry
    "trailing_stop_pct":        (0.10, 0.40), # trail % below highest
    "timeout_minutes":          (20.0, 120.0),# stagnation timeout
}


def validate_delta(delta: dict[str, Any]) -> dict[str, Any]:
    """
    Clamp every numeric key in delta to its guardrail band.

    Unknown keys (e.g. "reason", "analysis") are passed through unchanged
    so agents can include explanatory text without it being stripped.

    Raises ValueError if the delta contains a key whose guardrail band
    would require clamping to a nonsensical value (shouldn't happen with
    a well-prompted agent, but acts as a final safety net).
    """
    result: dict[str, Any] = {}
    for key, value in delta.items():
        if key not in GUARDRAILS:
            result[key] = value  # non-numeric annotation — pass through
            continue

        lo, hi = GUARDRAILS[key]
        if not isinstance(value, (int, float)):
            logger.warning("[agent] guardrail: %s has non-numeric value %r — skipping", key, value)
            continue

        clamped = max(lo, min(hi, float(value)))
        if clamped != value:
            logger.warning(
                "[agent] guardrail: %s=%.4f clamped to %.4f (band [%.4f, %.4f])",
                key, value, clamped, lo, hi,
            )
        result[key] = clamped

    return result


# ---------------------------------------------------------------------------
# DB queries — read-only, return plain dicts for prompt serialisation.
# ---------------------------------------------------------------------------

def query_score_buckets(db_path: str, strategy: str = "quick_pop_managed") -> list[dict]:
    """
    Break strategy_outcomes into 1-point ml_score buckets and return win rate +
    avg PnL per bucket.

    Only uses closed outcomes that were actually entered (entered=1) and
    have a valid ml_score and outcome_pnl_pct.

    Returns a list of dicts (sorted ascending by bucket floor):
        [{"bucket": "5.0-5.9", "count": 12, "win_rate": 0.42,
          "avg_pnl_pct": -3.1, "avg_max_gain_pct": 18.4}, ...]
    """
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT
                CAST(so.ml_score AS INTEGER) AS bucket_floor,
                COUNT(*)                     AS cnt,
                SUM(CASE WHEN so.outcome_pnl_pct > 0 THEN 1 ELSE 0 END) AS wins,
                AVG(so.outcome_pnl_pct)      AS avg_pnl,
                AVG(so.outcome_max_gain_pct) AS avg_max_gain
            FROM strategy_outcomes so
            JOIN signal_charts sc ON so.signal_chart_id = sc.id
            WHERE so.strategy = ?
              AND so.closed = 1
              AND so.entered = 1
              AND so.ml_score IS NOT NULL
              AND so.outcome_pnl_pct IS NOT NULL
            GROUP BY bucket_floor
            ORDER BY bucket_floor ASC
            """,
            (strategy,),
        ).fetchall()
    finally:
        conn.close()

    buckets = []
    for floor, cnt, wins, avg_pnl, avg_max_gain in rows:
        buckets.append({
            "bucket":        f"{floor}.0-{floor}.9",
            "count":         cnt,
            "win_rate":      round(wins / cnt, 3) if cnt else 0.0,
            "avg_pnl_pct":   round(avg_pnl or 0.0, 2),
            "avg_max_gain_pct": round(avg_max_gain or 0.0, 2),
        })
    return buckets


def query_exit_stats(db_path: str, strategy: str = "quick_pop_managed") -> list[dict]:
    """
    Per-sell-reason breakdown from strategy_outcomes.

    Returns:
        [{"sell_reason": "STOP_LOSS", "count": 34, "avg_pnl_pct": -18.2,
          "avg_hold_secs": 420, "avg_max_gain_pct": 8.1}, ...]
    """
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT
                so.outcome_sell_reason,
                COUNT(*)                    AS cnt,
                AVG(so.outcome_pnl_pct)     AS avg_pnl,
                AVG(so.outcome_hold_secs)   AS avg_hold,
                AVG(so.outcome_max_gain_pct) AS avg_max_gain
            FROM strategy_outcomes so
            JOIN signal_charts sc ON so.signal_chart_id = sc.id
            WHERE so.strategy = ?
              AND so.closed = 1
              AND so.entered = 1
              AND so.outcome_pnl_pct IS NOT NULL
            GROUP BY so.outcome_sell_reason
            ORDER BY cnt DESC
            """,
            (strategy,),
        ).fetchall()
    finally:
        conn.close()

    return [
        {
            "sell_reason":       reason or "UNKNOWN",
            "count":             cnt,
            "avg_pnl_pct":       round(avg_pnl or 0.0, 2),
            "avg_hold_secs":     round(avg_hold or 0.0, 0),
            "avg_max_gain_pct":  round(avg_max_gain or 0.0, 2),
        }
        for reason, cnt, avg_pnl, avg_hold, avg_max_gain in rows
    ]


def query_recent_trades(
    db_path: str,
    strategy: str = "quick_pop_managed",
    limit: int = 30,
    scored_only: bool = False,
) -> list[dict]:
    """
    Most recent closed positions for a strategy.

    scored_only — when True, only returns trades that have a non-null ml_score.
                  Use this for ML strategies so the agent only reasons about
                  trades the scorer actually evaluated, not pre-ML backfill data.

    Returns:
        [{"symbol": "BONK", "ml_score": 7.2, "outcome_pnl_pct": 12.5,
          "outcome_sell_reason": "TP2", "outcome_hold_secs": 830}, ...]
    """
    score_filter = "AND so.ml_score IS NOT NULL" if scored_only else ""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            f"""
            SELECT sc.symbol, so.ml_score, so.outcome_pnl_pct, so.outcome_sell_reason,
                   so.outcome_hold_secs, so.outcome_max_gain_pct
            FROM strategy_outcomes so
            JOIN signal_charts sc ON so.signal_chart_id = sc.id
            WHERE so.strategy = ?
              AND so.closed = 1
              AND so.entered = 1
              AND so.outcome_pnl_pct IS NOT NULL
              {score_filter}
            ORDER BY sc.ts DESC
            LIMIT ?
            """,
            (strategy, limit),
        ).fetchall()
    finally:
        conn.close()

    return [
        {
            "symbol":               r[0],
            "ml_score":             r[1],
            "outcome_pnl_pct":      round(r[2] or 0.0, 2),
            "outcome_sell_reason":  r[3],
            "outcome_hold_secs":    r[4],
            "outcome_max_gain_pct": round(r[5] or 0.0, 2),
        }
        for r in rows
    ]


def query_regime_context(
    db_path: str,
    strategy: str,
    *,
    base_strategy: str | None = None,
    lookback_signals: int = 50,
) -> dict[str, Any]:
    """
    Summarise recent strategy/base/channel conditions for mode switching.
    """
    conn = sqlite3.connect(db_path)
    try:
        def _recent_strategy_rows(target: str) -> list[tuple]:
            return conn.execute(
                """
                SELECT
                    so.entered,
                    so.closed,
                    so.outcome_pnl_pct,
                    so.outcome_max_gain_pct,
                    so.skip_reason,
                    COALESCE(so.source_channel, sc.source_channel, 'UNKNOWN')
                FROM strategy_outcomes so
                JOIN signal_charts sc ON sc.id = so.signal_chart_id
                WHERE so.strategy = ?
                ORDER BY sc.ts DESC
                LIMIT ?
                """,
                (target, lookback_signals),
            ).fetchall()

        def _summarise(rows: list[tuple]) -> dict[str, Any]:
            total = len(rows)
            entered_rows = [r for r in rows if r[0] == 1]
            closed_rows = [r for r in entered_rows if r[1] == 1 and r[2] is not None]
            skipped_rows = [r for r in rows if r[0] == 0]
            wins = sum(1 for r in closed_rows if (r[2] or 0.0) > 0)
            avg_pnl = sum((r[2] or 0.0) for r in closed_rows) / len(closed_rows) if closed_rows else 0.0
            avg_peak = sum((r[3] or 0.0) for r in closed_rows) / len(closed_rows) if closed_rows else 0.0
            skip_reasons: dict[str, int] = {}
            channel_counts: dict[str, int] = {}
            for _, _, _, _, skip_reason, source_channel in rows:
                channel_counts[source_channel] = channel_counts.get(source_channel, 0) + 1
                if skip_reason:
                    skip_reasons[skip_reason] = skip_reasons.get(skip_reason, 0) + 1
            top_channels = [
                {"channel": channel, "count": count}
                for channel, count in sorted(channel_counts.items(), key=lambda kv: kv[1], reverse=True)[:3]
            ]
            return {
                "signals": total,
                "entered": len(entered_rows),
                "blocked": len(skipped_rows),
                "block_rate": round(len(skipped_rows) / total, 4) if total else 0.0,
                "closed": len(closed_rows),
                "win_rate": round(wins / len(closed_rows), 4) if closed_rows else 0.0,
                "avg_pnl_pct": round(avg_pnl, 2),
                "avg_peak_gain_pct": round(avg_peak, 2),
                "skip_reasons": skip_reasons,
                "top_channels": top_channels,
            }

        strategy_rows = _recent_strategy_rows(strategy)
        base_rows = _recent_strategy_rows(base_strategy) if base_strategy else []

        market_row = conn.execute(
            """
            SELECT
                COUNT(*) AS signals,
                AVG(sc.pump_ratio) AS avg_pump_ratio,
                AVG(sc.price_change_30m_pct) AS avg_price_change_30m_pct,
                AVG(sc.unique_wallet_5m) AS avg_unique_wallet_5m,
                AVG(sc.market_cap_usd) AS avg_market_cap_usd,
                AVG(sc.liquidity_usd) AS avg_liquidity_usd
            FROM (
                SELECT id
                FROM signal_charts
                ORDER BY ts DESC
                LIMIT ?
            ) recent
            JOIN signal_charts sc ON sc.id = recent.id
            """,
            (lookback_signals,),
        ).fetchone()

        channel_rows = conn.execute(
            """
            SELECT
                COALESCE(so.source_channel, sc.source_channel, 'UNKNOWN') AS channel,
                COUNT(*) AS signals,
                SUM(CASE WHEN so.entered = 1 AND so.closed = 1 THEN 1 ELSE 0 END) AS closed,
                SUM(CASE WHEN so.entered = 1 AND so.closed = 1 AND so.outcome_pnl_pct > 0 THEN 1 ELSE 0 END) AS wins,
                AVG(CASE WHEN so.entered = 1 AND so.closed = 1 THEN so.outcome_pnl_pct END) AS avg_pnl_pct
            FROM strategy_outcomes so
            JOIN signal_charts sc ON sc.id = so.signal_chart_id
            WHERE so.strategy = ?
            GROUP BY 1
            ORDER BY signals DESC
            LIMIT 5
            """,
            (base_strategy or strategy,),
        ).fetchall()
    finally:
        conn.close()

    return {
        "lookback_signals": lookback_signals,
        "managed_strategy_recent": _summarise(strategy_rows),
        "base_strategy_recent": _summarise(base_rows) if base_rows else None,
        "market_recent": {
            "signals": int(market_row[0] or 0) if market_row else 0,
            "avg_pump_ratio": round(market_row[1] or 0.0, 3) if market_row else 0.0,
            "avg_price_change_30m_pct": round(market_row[2] or 0.0, 2) if market_row else 0.0,
            "avg_unique_wallet_5m": round(market_row[3] or 0.0, 2) if market_row else 0.0,
            "avg_market_cap_usd": round(market_row[4] or 0.0, 2) if market_row else 0.0,
            "avg_liquidity_usd": round(market_row[5] or 0.0, 2) if market_row else 0.0,
        },
        "base_channel_recent": [
            {
                "channel": channel,
                "signals": signals,
                "closed": closed,
                "win_rate": round((wins / closed), 4) if closed else 0.0,
                "avg_pnl_pct": round(avg_pnl_pct or 0.0, 2),
            }
            for channel, signals, closed, wins, avg_pnl_pct in channel_rows
        ],
    }


def query_strategy_pnl_snapshots(
    db_path: str,
    strategies: list[str],
) -> list[dict[str, Any]]:
    """
    Return compact pnl summaries for named strategies.

    daily_pnl_usd is based on signal date (signal_charts.ts) for closed, entered rows
    whose signal timestamp falls on the current local calendar day.
    """
    if not strategies:
        return []

    placeholders = ",".join("?" for _ in strategies)
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            f"""
            SELECT
                so.strategy,
                SUM(CASE WHEN so.entered = 1 AND so.closed = 1 THEN COALESCE(so.outcome_pnl_usd, 0.0) ELSE 0.0 END) AS total_pnl_usd,
                SUM(
                    CASE
                        WHEN so.entered = 1
                         AND so.closed = 1
                         AND date(sc.ts, 'localtime') = date('now', 'localtime')
                        THEN COALESCE(so.outcome_pnl_usd, 0.0)
                        ELSE 0.0
                    END
                ) AS daily_pnl_usd,
                SUM(CASE WHEN so.entered = 1 AND so.closed = 1 THEN 1 ELSE 0 END) AS closed_trades
            FROM strategy_outcomes so
            JOIN signal_charts sc ON sc.id = so.signal_chart_id
            WHERE so.strategy IN ({placeholders})
            GROUP BY so.strategy
            ORDER BY so.strategy
            """,
            strategies,
        ).fetchall()
    finally:
        conn.close()

    by_strategy = {
        strategy: {
            "strategy": strategy,
            "total_pnl_usd": round(total_pnl or 0.0, 2),
            "daily_pnl_usd": round(daily_pnl or 0.0, 2),
            "closed_trades": int(closed or 0),
        }
        for strategy, total_pnl, daily_pnl, closed in rows
    }
    return [
        by_strategy.get(
            strategy,
            {
                "strategy": strategy,
                "total_pnl_usd": 0.0,
                "daily_pnl_usd": 0.0,
                "closed_trades": 0,
            },
        )
        for strategy in strategies
    ]


def summarise_guardrails() -> str:
    """Human-readable guardrail table for including in agent prompts."""
    lines = ["Parameter                   | Min    | Max"]
    lines.append("-" * 46)
    for param, (lo, hi) in GUARDRAILS.items():
        lines.append(f"{param:<28} | {lo:<6} | {hi}")
    return "\n".join(lines)


def query_skipped_stats(db_path: str, strategy: str, base_strategy: str) -> dict:
    """
    For chart-filtered strategies, compute the outcome of signals they SKIPPED
    by looking at what the base strategy made on those same signal_chart_ids.

    Returns a dict with:
        total_skipped   — how many signals this strategy did not enter
        base_entered    — of those, how many the base strategy entered and closed
        profitable_pct  — % of those that would have been profitable (or None)
        avg_phantom_pnl — avg PnL% the base strategy made on them (or None)
        avg_max_gain    — avg peak gain% the base strategy saw (or None)
        sample_outcomes — last 10 phantom trades [{symbol, pnl_pct, max_gain_pct, sell_reason, ml_score}]
    """
    conn = sqlite3.connect(db_path)
    try:
        total_skipped = conn.execute(
            "SELECT COUNT(*) FROM strategy_outcomes WHERE strategy=? AND entered=0",
            (strategy,),
        ).fetchone()[0]

        rows = conn.execute(
            """
            SELECT base.outcome_pnl_pct, base.outcome_max_gain_pct,
                   base.outcome_sell_reason, sc.symbol, base.ml_score
              FROM strategy_outcomes skipped
              JOIN signal_charts sc   ON skipped.signal_chart_id = sc.id
              JOIN strategy_outcomes base ON base.signal_chart_id = skipped.signal_chart_id
             WHERE skipped.strategy = ?
               AND skipped.entered = 0
               AND base.strategy = ?
               AND base.entered = 1
               AND base.closed = 1
               AND base.outcome_pnl_pct IS NOT NULL
             ORDER BY sc.ts DESC
            """,
            (strategy, base_strategy),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return {
            "total_skipped": total_skipped,
            "base_entered": 0,
            "profitable_pct": None,
            "avg_phantom_pnl": None,
            "avg_max_gain": None,
            "sample_outcomes": [],
        }

    pnl_pcts  = [r[0] for r in rows]
    max_gains = [r[1] for r in rows if r[1] is not None]
    profitable = sum(1 for p in pnl_pcts if p > 0)

    return {
        "total_skipped":   total_skipped,
        "base_entered":    len(rows),
        "profitable_pct":  round(profitable / len(rows) * 100, 1),
        "avg_phantom_pnl": round(sum(pnl_pcts) / len(pnl_pcts), 2),
        "avg_max_gain":    round(sum(max_gains) / len(max_gains), 2) if max_gains else None,
        "sample_outcomes": [
            {
                "symbol":       r[3],
                "pnl_pct":      round(r[0], 2),
                "max_gain_pct": round(r[1], 2) if r[1] is not None else None,
                "sell_reason":  r[2],
                "ml_score":     r[4],
            }
            for r in rows[:10]
        ],
    }
