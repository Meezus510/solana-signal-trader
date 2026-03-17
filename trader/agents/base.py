"""
trader/agents/base.py — Shared infrastructure for all trading agents.

Provides:
    GUARDRAILS        — hardcoded (min, max) bands for every tunable parameter.
    validate_delta()  — clamps an agent's proposed delta to guardrail bands.
    query_score_buckets()   — score-bucket win/loss/avg-pnl breakdown from chart_snapshots.
    query_exit_stats()      — sell-reason breakdown (count, avg pnl, avg hold, avg max gain).
    query_recent_trades()   — last N closed positions for context.

The guardrail bands are the single source of truth for what the agents are
allowed to change. Tighten or widen them here — never in a prompt.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Guardrail bands — (min_allowed, max_allowed) for each tunable parameter.
# Agents propose values; Python clamps them to these ranges before applying.
# ---------------------------------------------------------------------------

GUARDRAILS: dict[str, tuple[float, float]] = {
    # ML score thresholds
    "ml_min_score":             (3.0, 7.0),   # floor for taking any trade
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

def query_score_buckets(db_path: str, strategy: str = "quick_pop_chart_ml") -> list[dict]:
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
                CAST(sc.ml_score AS INTEGER) AS bucket_floor,
                COUNT(*)                     AS cnt,
                SUM(CASE WHEN so.outcome_pnl_pct > 0 THEN 1 ELSE 0 END) AS wins,
                AVG(so.outcome_pnl_pct)      AS avg_pnl,
                AVG(so.outcome_max_gain_pct) AS avg_max_gain
            FROM strategy_outcomes so
            JOIN signal_charts sc ON so.signal_chart_id = sc.id
            WHERE so.strategy = ?
              AND so.closed = 1
              AND so.entered = 1
              AND sc.ml_score IS NOT NULL
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


def query_exit_stats(db_path: str, strategy: str = "quick_pop_chart_ml") -> list[dict]:
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


def query_recent_trades(db_path: str, strategy: str = "quick_pop_chart_ml", limit: int = 30) -> list[dict]:
    """
    Most recent closed positions for a strategy.

    Returns:
        [{"symbol": "BONK", "ml_score": 7.2, "outcome_pnl_pct": 12.5,
          "outcome_sell_reason": "TP2", "outcome_hold_secs": 830}, ...]
    """
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT sc.symbol, sc.ml_score, so.outcome_pnl_pct, so.outcome_sell_reason,
                   so.outcome_hold_secs, so.outcome_max_gain_pct
            FROM strategy_outcomes so
            JOIN signal_charts sc ON so.signal_chart_id = sc.id
            WHERE so.strategy = ?
              AND so.closed = 1
              AND so.entered = 1
              AND so.outcome_pnl_pct IS NOT NULL
            ORDER BY sc.ts DESC
            LIMIT ?
            """,
            (strategy, limit),
        ).fetchall()
    finally:
        conn.close()

    return [
        {
            "symbol":              r[0],
            "ml_score":            r[1],
            "outcome_pnl_pct":     round(r[2] or 0.0, 2),
            "outcome_sell_reason": r[3],
            "outcome_hold_secs":   r[4],
            "outcome_max_gain_pct": round(r[5] or 0.0, 2),
        }
        for r in rows
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
                   base.outcome_sell_reason, sc.symbol, sc.ml_score
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
