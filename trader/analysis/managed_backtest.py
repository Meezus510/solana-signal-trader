"""
Generic walk-forward backtest helpers for AI-managed strategies.
"""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone

from scripts.optimize_tp_sl import simulate_trade, simulate_trade_qp
from trader.analysis.ml_scorer import extract_features, euclidean
from trader.strategies.registry import get_managed_strategy_spec, resolve_managed_strategy_config


def load_managed_config(strategy_name: str, config_path: str = "strategy_config.json") -> dict:
    with open(config_path) as f:
        data = json.load(f)
    return data.get(strategy_name, {})


def resolve_managed_config(strategy_name: str, raw_cfg: dict) -> tuple[str, dict]:
    base, _mode, resolved = resolve_managed_strategy_config(strategy_name, raw_cfg)
    return base, resolved


def _load_rows(db_path: str, base_strategy: str) -> list[dict]:
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT
            sc.ts,
            so.outcome_pnl_pct,
            so.position_peak_pnl_pct,
            so.position_peak_ts,
            so.position_trough_pnl_pct,
            so.position_trough_ts,
            sc.candles_json,
            sc.candles_1m_json,
            sc.candles_1s_json,
            sc.pair_stats_json,
            COALESCE(so.source_channel, sc.source_channel, '')
        FROM strategy_outcomes so
        JOIN signal_charts sc ON sc.id = so.signal_chart_id
        WHERE so.strategy = ?
          AND so.entered = 1
          AND so.closed = 1
          AND so.position_peak_pnl_pct IS NOT NULL
          AND so.position_peak_ts IS NOT NULL
          AND so.position_trough_pnl_pct IS NOT NULL
          AND so.position_trough_ts IS NOT NULL
        ORDER BY sc.ts ASC
        """,
        (base_strategy,),
    ).fetchall()
    conn.close()

    out = []
    for row in rows:
        ts, outcome_pnl_pct, peak_pnl_pct, peak_ts, trough_pnl_pct, trough_ts, c15, c1m, c1s, ps, ch = row
        pair_stats = json.loads(ps) if ps else {}
        candles_15s = json.loads(c15) if c15 else []
        candles_1m = json.loads(c1m) if c1m else []
        candles_1s = json.loads(c1s) if c1s else None

        lows = [float(c["l"]) for c in candles_1m if c.get("l") is not None]
        close = float(candles_1m[-1]["c"]) if candles_1m and candles_1m[-1].get("c") is not None else None
        buy_vol_1h = pair_stats.get("buy_volume_1h", 0.0) or 0.0
        total_vol_1h = pair_stats.get("total_volume_1h", 0.0) or 0.0
        uw5 = pair_stats.get("unique_wallet_5m")
        uwh5 = pair_stats.get("unique_wallet_hist_5m")

        out.append({
            "ts": ts,
            "outcome_pnl_pct": outcome_pnl_pct or 0.0,
            "peak_pnl_pct": peak_pnl_pct,
            "peak_ts": peak_ts,
            "trough_pnl_pct": trough_pnl_pct,
            "trough_ts": trough_ts,
            "candles_15s": candles_15s,
            "candles_1m": candles_1m,
            "candles_1s": candles_1s,
            "pair_stats": pair_stats,
            "source_channel": ch or "",
            "holder_count": pair_stats.get("holder_count"),
            "price_change_30m_pct": pair_stats.get("price_change_30m_pct"),
            "buy_vol_ratio_1h": buy_vol_1h / (total_vol_1h + 1e-9) if total_vol_1h > 0 else None,
            "market_cap_usd": pair_stats.get("market_cap_usd"),
            "wallet_momentum_5m": (
                min((uw5 or 0) / max(uwh5 or 0, 1), 5.0)
                if uw5 is not None else None
            ),
            "pump_ratio_1m": close / min(lows) if lows and close and min(lows) > 0 else None,
        })
    return out


def _compute_scores(rows: list[dict], cfg: dict) -> None:
    weights = cfg.get("ml_feature_weights")
    if not cfg.get("use_ml_filter", False) or weights is None:
        for row in rows:
            row["ml_score"] = None
        return

    weights = list(weights)
    feats = [
        extract_features(
            r["candles_15s"],
            r["candles_1m"],
            r["candles_1s"],
            r["pair_stats"],
            r["source_channel"],
        )
        for r in rows
    ]
    score_low = cfg.get("ml_score_low_pct", -35.0)
    score_high = cfg.get("ml_score_high_pct", 85.0)
    score_range = score_high - score_low
    halflife = cfg.get("ml_halflife_days", 14.0)
    k = int(cfg.get("ml_k", 5))

    timestamps = []
    for row in rows:
        ts = datetime.fromisoformat(row["ts"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        timestamps.append(ts)

    min_train = max(k * 3, max(5, len(rows) // 7))
    label_key = "peak_pnl_pct" if cfg.get("ml_training_label") == "position_peak_pnl_pct" else "outcome_pnl_pct"

    for i, row in enumerate(rows):
        qf = feats[i]
        if qf is None:
            row["ml_score"] = None
            continue

        train_feats = []
        train_labels = []
        recency_weights = []
        for j, other in enumerate(rows):
            if j == i or feats[j] is None or timestamps[j] >= timestamps[i]:
                continue
            age_days = (timestamps[i] - timestamps[j]).total_seconds() / 86400.0
            train_feats.append([
                feats[j][idx] * (weights[idx] if idx < len(weights) else 1.0)
                for idx in range(len(feats[j]))
            ])
            train_labels.append(other[label_key])
            recency_weights.append(math.exp(-age_days / halflife))

        if len(train_feats) < min_train:
            row["ml_score"] = None
            continue

        weighted_qf = [
            qf[idx] * (weights[idx] if idx < len(weights) else 1.0)
            for idx in range(len(qf))
        ]
        n_train = len(train_feats)
        n_feat = len(weighted_qf)
        means = [sum(train_feats[r][f] for r in range(n_train)) / n_train for f in range(n_feat)]
        stds = [
            math.sqrt(sum((train_feats[r][f] - means[f]) ** 2 for r in range(n_train)) / n_train) or 1e-9
            for f in range(n_feat)
        ]
        norm_q = [(weighted_qf[f] - means[f]) / stds[f] for f in range(n_feat)]
        norm_train = [
            [(train_feats[r][f] - means[f]) / stds[f] for f in range(n_feat)]
            for r in range(n_train)
        ]
        dists = [euclidean(norm_q, tf) for tf in norm_train]
        top_idx = sorted(range(n_train), key=lambda idx: dists[idx])[:k]
        pred = sum(recency_weights[idx] * train_labels[idx] for idx in top_idx) / (
            sum(recency_weights[idx] for idx in top_idx) or 1e-9
        )
        row["ml_score"] = max(0.0, min(10.0, (pred - score_low) / score_range * 10.0))


def _blocked_by_filters(row: dict, cfg: dict) -> bool:
    if cfg.get("block_new_entries"):
        return True
    if cfg.get("ml_wallet_momentum_max") is not None:
        wm = row["wallet_momentum_5m"]
        if wm is not None and wm >= cfg["ml_wallet_momentum_max"]:
            return True
    if cfg.get("holder_count_max") is not None:
        hc = row["holder_count"]
        if hc is not None and hc > cfg["holder_count_max"]:
            return True
    if cfg.get("late_entry_price_chg_30m_max") is not None and cfg.get("late_entry_pump_ratio_min") is not None:
        pc30 = row["price_change_30m_pct"]
        pump = row["pump_ratio_1m"]
        if pc30 is not None and pump is not None and pc30 > cfg["late_entry_price_chg_30m_max"] and pump > cfg["late_entry_pump_ratio_min"]:
            return True
    if cfg.get("buy_vol_ratio_1h_max") is not None:
        bvr = row["buy_vol_ratio_1h"]
        if bvr is not None and bvr > cfg["buy_vol_ratio_1h_max"]:
            return True
    if cfg.get("market_cap_usd_min") is not None:
        mcap = row["market_cap_usd"]
        if mcap is not None and mcap < cfg["market_cap_usd_min"]:
            return True
    return False


def _size_multiplier(score: float | None, cfg: dict) -> float:
    if score is None:
        return 1.0
    if score >= cfg.get("ml_max_score_threshold", 9.5):
        return cfg.get("ml_max_size_multiplier", 1.0)
    if score >= cfg.get("ml_high_score_threshold", 8.0):
        return cfg.get("ml_size_multiplier", 1.0)
    return 1.0


def _simulate_one(row: dict, cfg: dict, base_strategy: str) -> float:
    if base_strategy == "infinite_moonbag":
        sim_cfg = {
            "initial_floor_pct": cfg["stop_loss_pct"] * 100.0,
            "stop_milestones": [],
            "tp_levels": cfg["tp_levels"],
        }
        return simulate_trade(row, sim_cfg, buy_size=cfg["buy_size_usd"])

    sim_cfg = {
        "stop_loss_pct": cfg["stop_loss_pct"],
        "trailing_stop_pct": cfg["trailing_stop_pct"],
        "tp_levels": cfg["tp_levels"],
    }
    return simulate_trade_qp(row, sim_cfg, buy_size=cfg["buy_size_usd"])


def backtest_managed_config(db_path: str, strategy_name: str, cfg: dict) -> dict:
    base_strategy, resolved = resolve_managed_config(strategy_name, cfg)
    rows = _load_rows(db_path, base_strategy)
    _compute_scores(rows, resolved)

    total_pnl = 0.0
    entered = 0
    wins = 0
    blocked = 0
    scored = 0

    for row in rows:
        score = row["ml_score"]
        if _blocked_by_filters(row, resolved):
            blocked += 1
            continue
        if resolved.get("use_ml_filter") and score is not None and score < resolved.get("ml_min_score", 0.0):
            blocked += 1
            continue
        pnl = _simulate_one(row, resolved, base_strategy) * _size_multiplier(score, resolved)
        total_pnl += pnl
        entered += 1
        wins += pnl > 0
        scored += score is not None

    n_rows = len(rows)
    return {
        "strategy": strategy_name,
        "base_strategy": base_strategy,
        "mode": resolved.get("mode", "custom"),
        "total_rows": n_rows,
        "entered": entered,
        "blocked": blocked,
        "block_rate": blocked / n_rows if n_rows else 0.0,
        "win_rate": wins / entered if entered else 0.0,
        "total_pnl_usd": total_pnl,
        "avg_pnl_per_trade_usd": total_pnl / entered if entered else 0.0,
        "scored_entries": scored,
    }


def backtest_managed_mode(db_path: str, strategy_name: str, base_strategy: str, mode: str) -> dict:
    return backtest_managed_config(db_path, strategy_name, {"base_strategy": base_strategy, "mode": mode})


def leaderboard_for_managed_strategy(db_path: str, strategy_name: str) -> list[dict]:
    spec = get_managed_strategy_spec(strategy_name)
    rows = []
    for base in sorted(spec.bases):
        for mode in spec.modes.get(base, {}).keys():
            metrics = backtest_managed_mode(db_path, strategy_name, base, mode)
            rows.append({
                "base_strategy": base,
                "mode": mode,
                "entered": metrics["entered"],
                "block_rate": round(metrics["block_rate"], 4),
                "win_rate": round(metrics["win_rate"], 4),
                "total_pnl_usd": round(metrics["total_pnl_usd"], 2),
                "avg_pnl_per_trade_usd": round(metrics["avg_pnl_per_trade_usd"], 4),
            })
    rows.sort(key=lambda r: (r["total_pnl_usd"], r["win_rate"]), reverse=True)
    return rows[:12]
