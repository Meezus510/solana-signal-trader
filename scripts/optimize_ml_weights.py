#!/usr/bin/env python3
"""
scripts/optimize_ml_weights.py — AI-driven ML weight optimizer for KNN strategies.

Uses Claude to iteratively suggest feature weight configurations, evaluates each
via temporal LOO cross-validation, and converges on the best config.

How it works:
  1. Loads feature separability (Cohen's d) from the DB.
  2. Sends separability + current weights + LOO results to Claude.
  3. Claude suggests up to N_SUGGESTIONS new weight vectors to try.
  4. Evaluates each via LOO and scores by: winners_through + losers_blocked.
  5. Feeds all results back to Claude for the next round.
  6. Repeats for N_ROUNDS, then prints the best configuration found.

Usage:
    python scripts/optimize_ml_weights.py [--strategy quick_pop|moonbag] [--rounds 4] [--suggestions 8] [--min-score 1.5]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))
from trader.analysis.ml_scorer import extract_features, zscore_normalize, euclidean

DB_PATH = "trader.db"

# Per-strategy KNN hyperparameters (must match registry.py)
STRATEGY_CONFIGS = {
    "quick_pop": {
        "db_strategy":    "quick_pop",
        "registry_name":  "quick_pop_managed",
        "knn_k":          3,
        "knn_hl":         7.0,
        "knn_sl":         -45.0,
        "knn_sh":         300.0,
        # position_peak_pnl_pct works well: quick_pop losers dump hard (peak ≈ -30%)
        # so the KNN predicts negative values → low scores → blocked.
        "label_key":           "position_peak_pnl_pct",
        "sep_threshold":       49.0,   # Cohen's d: TP1 proxy (peak > 49% ≈ 1.5× target)
        # idx 25-26: unlocked — recent records (75/305 wallet_30m, 53/305 top10)
        # have real values; fallbacks (1.0 neutral, 0.5 neutral) are reasonable
        # so partial coverage produces valid z-score variance.
        # idx 27-32 (1s OHLCV): no closed training data yet — start at 0, let optimizer explore.
        # idx 33-42 (15s shape): all 236 examples have 15s candles → start at 1.0.
        # idx 43-52 (1m shape): all 236 examples have 1m candles → start at 1.0.
        # idx 53-62 (1s shape): no closed 1s training data yet — start at 0.
        "locked_zero_indices": [],
        "seed_weights": [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,            # idx  0-5:  15s OHLCV
            10.0, 0.0, 0.0, 0.0, 0.0, 0.0,           # idx  6-11: 1m OHLCV
            0.0, 8.0, 4.0, 3.0, 0.0, 0.0,            # idx 12-17: pair stats + source_channel
            0.0, 0.0, 0.0,                            # idx 18-20: token metadata
            0.0, 0.0, 0.0, 3.0,                      # idx 21-24: wallet (5m)
            0.0, 1.0,                                 # idx 25-26: wallet_momentum_30m, top10_holder_pct
            0.0, 0.0, 0.0, 4.0, 0.0, 0.0,            # idx 27-32: 1s OHLCV
            0.0, 1.0, 2.0, 0.0, 2.5, 0.5, 1.5, 0.0, 0.0, 0.0,  # idx 33-42: 15s shape
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # idx 43-52: 1m shape
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.5, 0.0, 0.0,  # idx 53-62: 1s shape
        ],
        # [score_threshold, size_multiplier] — applied to through-trades at inference
        "seed_multiplier_tiers": [[5.0, 1.5], [7.5, 2.5]],
    },
    "trend_rider": {
        "db_strategy":         "trend_rider",
        "registry_name":       "trend_rider_managed",
        "knn_k":               5,
        "knn_hl":              14.0,
        "knn_sl":              -35.0,
        "knn_sh":              85.0,
        # outcome_pnl_pct: trend_rider trades have clear +/- outcomes (no long hold drift).
        "label_key":           "outcome_pnl_pct",
        "sep_threshold":       0.0,    # any positive final outcome = winner
        # trend_rider is ml_use_subminute=False → candles_15s=[] and candles_1s=None at inference.
        # Same locking pattern as moonbag: lock 15s/1s features that are always neutral at inference.
        # UNLOCKED: idx 6-24 (1m OHLCV + pair stats + wallet 5m), 43-52 (1m shape).
        "locked_zero_indices": list(range(6)) + [25, 26] + list(range(27, 43)) + list(range(53, 63)),
        "seed_weights": [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,          # idx  0-5:  15s OHLCV — LOCKED
            3.2, 0.0, 3.2, 0.0, 2.0, 0.0,          # idx  6-11: 1m OHLCV
            0.0, 0.0, 0.0, 0.0, 3.2, 2.2,          # idx 12-17: pair stats + source_channel
            0.0, 2.2, 0.0,                          # idx 18-20: token metadata
            0.0, 0.1, 0.0, 1.8,                    # idx 21-24: wallet (5m)
            0.0, 0.0,                               # idx 25-26: LOCKED
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,          # idx 27-32: 1s OHLCV — LOCKED
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # idx 33-42: 15s shape — LOCKED
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # idx 43-52: 1m shape
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # idx 53-62: 1s shape — LOCKED
        ],
        "seed_multiplier_tiers": [[5.0, 1.5], [7.5, 2.5]],
    },
    "moonbag": {
        "db_strategy":    "infinite_moonbag",
        "registry_name":  "moonbag",
        "knn_k":          3,
        "knn_hl":         14.0,
        "knn_sl":         0.0,
        "knn_sh":         200.0,
        # position_peak_pnl_pct: measures entry quality independent of exit timing.
        # Winners peak at ~159% avg, losers at ~23% avg — strong separation.
        # outcome_pnl_pct conflates entry quality with trailing-stop/TP mechanics;
        # peak_pnl purely answers "did this token pump?" which is what we want to predict.
        # Score range [0, 200]: losers cluster near 0-30, winners near 50-200+.
        "label_key":      "position_peak_pnl_pct",
        "sep_threshold":  50.0,   # Cohen's d: peak > 50% = strong pump = winner
        # LOCKED features — always neutral at inference, weighting them fits training noise:
        #   idx 25-26: wallet 30m    — no historical data yet
        #   idx 27-32: 1s OHLCV     — candles_1s=None at inference (moonbag never fetches 1s)
        #   idx 53-62: 1s shape     — candles_1s=None at inference → always zeros
        # UNLOCKED: idx 0-5 (15s OHLCV) — ml_use_subminute=True; top-3 Cohen's d features
        # UNLOCKED: idx 33-42 (15s shape) — 15s candles now fetched at inference
        # UNLOCKED: idx 43-52 (1m shape) — moonbag fetches 1m at inference; training data exists.
        "locked_zero_indices": [25, 26] + list(range(27, 33)) + list(range(53, 63)),
        "seed_weights": [
            3.0,  1.0,  3.0,  1.0,  3.0,  1.0,          # idx  0-5:  15s OHLCV — UNLOCKED
            7.6,  1.78, 6.9,  5.1,  0.38, 0.28,         # idx  6-11: 1m OHLCV
            0.0,  0.0,  0.0,  0.0,  0.01, 0.22,         # idx 12-17: pair stats + source_channel
            0.04, 0.02, 0.0,                             # idx 18-20: token metadata
            0.0,  0.12, 0.0,  6.9,                      # idx 21-24: wallet (5m)
            0.0,  0.0,                                   # idx 25-26: LOCKED
            0.0,  0.0,  0.0,  0.0,  0.0,  0.0,          # idx 27-32: 1s OHLCV — LOCKED
            1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  # idx 33-42: 15s shape — UNLOCKED
            1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  # idx 43-52: 1m shape
            0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  # idx 53-62: 1s shape — LOCKED
        ],
        "seed_multiplier_tiers": [[5.0, 1.5], [7.5, 2.5]],
    },
}

FEAT_NAMES = [
    "pump_ratio_15s", "vol_momentum_15s", "price_slope_15s",
    "recent_momentum_15s", "volatility_15s", "candle_count_15s",       # idx  0-5
    "pump_ratio_1m", "vol_momentum_1m", "price_slope_1m",
    "recent_momentum_1m", "volatility_1m", "candle_count_1m",          # idx  6-11
    "buy_ratio_5m", "activity_5m_norm", "price_change_5m_norm",
    "buy_vol_ratio_1h", "liquidity_change_1h", "source_channel",       # idx 12-17
    "market_cap_norm", "liquidity_usd_norm", "holder_count_norm",      # idx 18-20
    "unique_wallet_5m_norm", "wallet_momentum_5m", "price_change_30m_norm",
    "buy_vol_ratio_5m",                                                 # idx 21-24
    "wallet_momentum_30m", "top10_holder_pct",                         # idx 25-26
    "pump_ratio_1s", "vol_momentum_1s", "price_slope_1s",
    "recent_momentum_1s", "volatility_1s", "candle_count_1s",         # idx 27-32
    "shape_15s_0", "shape_15s_1", "shape_15s_2", "shape_15s_3", "shape_15s_4",
    "shape_15s_5", "shape_15s_6", "shape_15s_7", "shape_15s_8", "shape_15s_9",  # idx 33-42
    "shape_1m_0", "shape_1m_1", "shape_1m_2", "shape_1m_3", "shape_1m_4",
    "shape_1m_5", "shape_1m_6", "shape_1m_7", "shape_1m_8", "shape_1m_9",      # idx 43-52
    "shape_1s_0", "shape_1s_1", "shape_1s_2", "shape_1s_3", "shape_1s_4",
    "shape_1s_5", "shape_1s_6", "shape_1s_7", "shape_1s_8", "shape_1s_9",      # idx 53-62
]

N_FEAT = len(FEAT_NAMES)

# KNN hyperparameter defaults (overridden per strategy in main)
KNN_K   = 3
KNN_HL  = 7.0
KNN_SL  = -45.0
KNN_SH  = 300.0


# ---------------------------------------------------------------------------
# Data loading (same as debug_quick_pop_ml.py)
# ---------------------------------------------------------------------------

def load_data(db_path: str, db_strategy: str = "quick_pop") -> list[dict]:
    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT so.outcome_pnl_pct, so.outcome_max_gain_pct, so.position_peak_pnl_pct,
               so.outcome_pnl_usd,
               sc.candles_json, sc.candles_1m_json, sc.candles_1s_json, sc.pair_stats_json,
               sc.entry_price, sc.ts, sc.source_channel
          FROM strategy_outcomes so
          JOIN signal_charts sc ON sc.id = so.signal_chart_id
         WHERE so.strategy     = ?
           AND so.closed       = 1
           AND so.entered      = 1
           AND so.outcome_pnl_pct IS NOT NULL
           AND so.position_peak_pnl_pct IS NOT NULL
         ORDER BY sc.ts
    """, (db_strategy,)).fetchall()
    conn.close()

    records = []
    for row in rows:
        outcome_pnl, max_gain, peak_pnl, pnl_usd, c15s_j, c1m_j, c1s_j, ps_j, entry, ts, ch = row
        records.append({
            "outcome_pnl_pct":       outcome_pnl,
            "outcome_max_gain_pct":  max_gain,
            "position_peak_pnl_pct": peak_pnl,
            "outcome_pnl_usd":       pnl_usd or 0.0,
            "candles_15s":   json.loads(c15s_j) if c15s_j else [],
            "candles_1m":    json.loads(c1m_j)  if c1m_j  else [],
            "candles_1s":    json.loads(c1s_j)  if c1s_j  else None,
            "pair_stats":    json.loads(ps_j)   if ps_j   else {},
            "entry_price":   entry,
            "ts":            ts,
            "source_channel": ch or "",
        })
    return records


def precompute_features(records: list[dict]) -> list[list[float] | None]:
    out = []
    for r in records:
        try:
            f = extract_features(
                candles_15s=r["candles_15s"],
                candles_1m=r["candles_1m"],
                candles_1s=r["candles_1s"],
                pair_stats=r["pair_stats"],
                source_channel=r["source_channel"],
            )
            out.append(f)
        except Exception:
            out.append(None)
    return out


# ---------------------------------------------------------------------------
# LOO evaluation
# ---------------------------------------------------------------------------

def _size_for_score(score: float, tiers: list[list[float]]) -> float:
    """Return size multiplier for a given score using tiered config.

    tiers: list of [threshold, multiplier] sorted by threshold ascending.
    Applies the highest tier whose threshold is exceeded; defaults to 1.0.
    """
    mult = 1.0
    for threshold, multiplier in tiers:
        if score >= threshold:
            mult = multiplier
    return mult


def loo_evaluate(
    records: list[dict],
    all_feats: list[list[float] | None],
    weights: list[float],
    min_score: float,
    k: int = KNN_K,
    halflife: float = KNN_HL,
    score_low: float = KNN_SL,
    score_high: float = KNN_SH,
    label_key: str = "position_peak_pnl_pct",
    multiplier_tiers: list[list[float]] | None = None,
) -> tuple[int, int, int, int, float, float, float, float, list[float]]:
    """
    Temporal leave-one-out KNN evaluation.
    Returns (winners_through, losers_blocked, total_winners, total_losers,
             pnl_saved_usd, pnl_missed_usd,
             through_pnl_base, through_pnl_boosted,
             passing_scores).

    pnl_saved_usd      — sum of |outcome_pnl_usd| for losers correctly blocked.
    pnl_missed_usd     — sum of outcome_pnl_usd for winners incorrectly blocked.
    through_pnl_base   — sum of outcome_pnl_usd for all through-trades (1× sizing).
    through_pnl_boosted— same, with tiered size multipliers applied per-score.
    passing_scores     — raw KNN score for every trade that was NOT blocked.
    """
    if multiplier_tiers is None:
        multiplier_tiers = []
    n = len(records)
    now = datetime.now(timezone.utc)
    pnl_range = score_high - score_low

    timestamps = []
    for r in records:
        ts = datetime.fromisoformat(r["ts"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        timestamps.append(ts)

    winners_through = losers_blocked = 0
    pnl_saved = pnl_missed = 0.0
    through_pnl_base = through_pnl_boosted = 0.0
    passing_scores: list[float] = []
    total_winners = sum(1 for r in records if r["outcome_pnl_pct"] > 0)
    total_losers  = n - total_winners

    for i in range(n):
        qf = all_feats[i]
        is_winner  = records[i]["outcome_pnl_pct"] > 0
        trade_pnl  = records[i]["outcome_pnl_usd"]
        if qf is None:
            if is_winner:
                winners_through += 1
            continue

        wq = [qf[j] * (weights[j] if j < len(weights) else 1.0) for j in range(N_FEAT)]

        train_feats, train_labels, recency_ws = [], [], []
        for j in range(n):
            if j == i or all_feats[j] is None or timestamps[j] >= timestamps[i]:
                continue
            tf = all_feats[j]
            wf = [tf[k2] * (weights[k2] if k2 < len(weights) else 1.0) for k2 in range(N_FEAT)]
            age = (timestamps[i] - timestamps[j]).total_seconds() / 86400.0
            train_feats.append(wf)
            train_labels.append(records[j][label_key])
            recency_ws.append(math.exp(-age / halflife))

        # Need enough training data for KNN to be meaningful.
        # min_train = max(k*3, 15% of dataset) so early records don't skew results.
        min_train = max(k * 3, max(5, n // 7))
        if len(train_feats) < min_train:
            if is_winner:
                winners_through += 1
            continue

        # z-normalise
        n_train = len(train_feats)
        nf = len(train_feats[0])
        means = [sum(train_feats[r2][fi] for r2 in range(n_train)) / n_train for fi in range(nf)]
        stds  = [
            math.sqrt(sum((train_feats[r2][fi] - means[fi]) ** 2 for r2 in range(n_train)) / n_train) or 1e-9
            for fi in range(nf)
        ]
        norm_train = [[(train_feats[r2][fi] - means[fi]) / stds[fi] for fi in range(nf)] for r2 in range(n_train)]
        norm_q     = [(wq[fi] - means[fi]) / stds[fi] for fi in range(nf)]

        dists = [euclidean(norm_q, norm_train[j]) for j in range(n_train)]
        top_k = sorted(range(n_train), key=lambda j: dists[j])[:k]

        num   = sum(recency_ws[j] * train_labels[j] for j in top_k)
        denom = sum(recency_ws[j] for j in top_k) or 1e-9
        pred  = num / denom
        score = max(0.0, min(10.0, (pred - score_low) / pnl_range * 10.0))

        blocked = score < min_score
        if blocked:
            if is_winner:
                pnl_missed += trade_pnl      # money left on the table
            else:
                pnl_saved  += abs(trade_pnl) # loss avoided
                losers_blocked += 1
        else:
            if is_winner:
                winners_through += 1
            passing_scores.append(score)
            size = _size_for_score(score, multiplier_tiers)
            through_pnl_base    += trade_pnl
            through_pnl_boosted += trade_pnl * size

    return (winners_through, losers_blocked, total_winners, total_losers,
            pnl_saved, pnl_missed,
            through_pnl_base, through_pnl_boosted,
            passing_scores)


# ---------------------------------------------------------------------------
# Feature separability (Cohen's d for TP1-proxy label)
# ---------------------------------------------------------------------------

def compute_separability(
    records: list[dict],
    all_feats: list[list[float] | None],
    threshold: float = 49.0,
    label_key: str = "position_peak_pnl_pct",
) -> list[tuple[int, str, float, float, float, float, float]]:
    tp1    = [i for i, r in enumerate(records) if r[label_key] > threshold and all_feats[i]]
    others = [i for i, r in enumerate(records) if r[label_key] <= threshold and all_feats[i]]
    sep = []
    for fi, name in enumerate(FEAT_NAMES):
        wv = [all_feats[i][fi] for i in tp1]
        lv = [all_feats[i][fi] for i in others]
        if not wv or not lv:
            sep.append((fi, name, 0.0, 0.0, 0.0, 0.0, 0.0))
            continue
        wm = sum(wv) / len(wv)
        lm = sum(lv) / len(lv)
        ws = math.sqrt(sum((x - wm) ** 2 for x in wv) / len(wv)) or 1e-9
        ls = math.sqrt(sum((x - lm) ** 2 for x in lv) / len(lv)) or 1e-9
        pooled = math.sqrt((ws ** 2 + ls ** 2) / 2)
        d = abs(wm - lm) / pooled
        sep.append((fi, name, wm, ws, lm, ls, d))
    return sorted(sep, key=lambda x: -x[6])


# ---------------------------------------------------------------------------
# AI optimizer
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_WEIGHTS_ONLY = """You are an expert ML feature weight optimizer for a cryptocurrency trading KNN classifier.

The model predicts whether a token signal will pump past TP1 (1.5×) within 45 minutes.
Features are z-score normalized before KNN distance computation.
Weights amplify features before normalization — a weight of 0 removes the feature entirely.

You will receive:
- Feature separability (Cohen's d) for each feature
- Current best weight configuration and its LOO performance
- History of all tested configurations and their results

Your goal: suggest weight vectors that maximize `through_pnl` (actual dollar PnL of trades
that pass the filter). This is the ONLY metric that matters — it represents real trading profit.
Do NOT optimize for blocking count, saved PnL, or any other metric.
A config that blocks everything scores $0 through_pnl and is worthless.
Look for weight combinations where the trades that DO pass are net profitable.

Rules:
- Return ONLY a JSON array of weight vectors, no explanation
- Each weight vector is a list of 63 floats (one per feature, order matches FEAT_NAMES)
- Valid weight range: 0.0 to 10.0
- DIVERSITY RULE: at least HALF your suggestions must differ significantly from the
  current best — change 5+ features by >2.0, or zero out entirely different feature groups
- Suggest diverse configurations — explore the space, don't just tweak the best
- Features with d < 0.1 are noise — keep them at 0.0 or very low (≤0.3)
- wallet_momentum_5m (idx 22) is LOSERS > WINNERS — still use a positive weight,
  KNN will naturally learn the inverse relationship from training labels
- Return exactly the number of suggestions requested"""

SYSTEM_PROMPT_WITH_TIERS = """You are an expert optimizer for a cryptocurrency trading KNN classifier with tiered buy sizing.

The model scores incoming signals 0–10. Scores below min_score are blocked entirely.
For trades that pass, a tiered multiplier table determines position size:
  - Each tier is [score_threshold, size_multiplier]
  - Tiers are sorted ascending; the highest threshold exceeded wins
  - Score < lowest threshold → 1× (base size)
  - Example: [[5.0, 1.5], [7.5, 2.5]] means score 5–7.5 → 1.5×, score ≥7.5 → 2.5×

You will receive:
- Feature separability (Cohen's d) for each feature
- Results history: each config has weights + multiplier_tiers + performance metrics

PRIMARY metric to maximize: `through_pnl` (actual dollar PnL of trades that pass the filter).
  This is the ONLY metric that maps to real trading profit. Do NOT optimize for blocking count
  or saved PnL — a config that blocks everything is worthless.
SECONDARY metric (tiers): `boost_gain` = through_pnl_boosted - through_pnl_base
  (extra PnL from sizing up high-confidence trades — optimise this via tier thresholds/multipliers)
CRITICAL for tiers: each config in the history shows `passing_score_pcts` — the actual
  distribution of scores for trades that passed the filter. Set tier thresholds WITHIN
  this range (between p25 and p90), not above it. Tiers above p90 will never trigger.
HARD CONSTRAINT: winner_miss_rate <= given limit (configs violating this are rejected)
TERTIARY: `combined` (winners_through + losers_blocked) when both pnl metrics are similar

Rules:
- Return ONLY a JSON array of config objects, no explanation
- Each config object has exactly two keys:
    "weights": list of 63 floats (0.0–10.0), one per feature
    "multiplier_tiers": list of [threshold, multiplier] pairs (1–4 tiers)
- threshold range: min_score to 10.0 (must exceed min_score to matter)
- multiplier range: 1.0 to 5.0; tiers must be sorted ascending by threshold
- DIVERSITY RULE: at least HALF your suggestions must differ significantly from the
  current best — change 5+ features by >2.0, or zero out entirely different feature groups,
  or use completely different tier structures
- Suggest diverse configs — vary both weights AND tier structures
- Features with d < 0.1 are noise — keep at 0.0 or very low (≤0.3)
- wallet_momentum_5m (idx 22) is LOSERS > WINNERS — still use a positive weight,
  KNN will naturally learn the inverse relationship from training labels
- Return exactly the number of suggestions requested"""


def ask_claude_for_configs(
    client: anthropic.Anthropic,
    separability: list[tuple],
    results_history: list[dict],
    n_suggestions: int,
    min_score: float,
    label_key: str = "position_peak_pnl_pct",
    sep_threshold: float = 49.0,
    locked_zero_indices: list[int] | None = None,
    max_miss_rate: float = 0.20,
    optimize_tiers: bool = False,
) -> list[dict]:
    """Returns list of {"weights": [...], "multiplier_tiers": [[t, m], ...]} dicts.
    When optimize_tiers=False, all configs get multiplier_tiers=[] (1× for every trade).
    """
    locked_zero_indices = locked_zero_indices or []
    sep_lines = "\n".join(
        f"  [{fi:2d}] {name:<25} d={d:.3f}  {'W>L' if wm > lm else 'L>W'}"
        for fi, name, wm, ws, lm, ls, d in separability
    )

    locked_note = ""
    if locked_zero_indices:
        locked_note = (
            f"\nLOCKED ZERO FEATURES (must be 0.0 in weights): indices {locked_zero_indices}\n"
        )

    # Cap history: send top 12 by through_pnl + last 4 (most recent) to keep prompt lean.
    # Sending all configs leads to degraded Claude suggestions after many rounds.
    sorted_by_pnl = sorted(results_history, key=lambda r: -r["through_pnl_boosted"])
    recent = results_history[-4:] if len(results_history) > 4 else []
    seen_labels = set()
    capped_history = []
    for r in sorted_by_pnl[:12]:
        capped_history.append(r)
        seen_labels.add(r["label"])
    for r in recent:
        if r["label"] not in seen_labels:
            capped_history.append(r)
            seen_labels.add(r["label"])

    history_lines = []
    for i, r in enumerate(capped_history):
        wt, lb, tw, tl = r["winners_through"], r["losers_blocked"], r["total_winners"], r["total_losers"]
        miss_pct = (tw - wt) / tw * 100 if tw else 0
        sp = r.get("score_pcts", {})
        line = (
            f"  Config {i+1}: through_pnl=${r['through_pnl_base']:+.2f}  "
            f"winners={wt}/{tw} (miss={miss_pct:.0f}%)  blocked={lb}/{tl}  "
            f"passing_score_pcts=(p25={sp.get('p25',0):.2f} p50={sp.get('p50',0):.2f} "
            f"p75={sp.get('p75',0):.2f} p90={sp.get('p90',0):.2f} max={sp.get('max',0):.2f})  "
            f"weights={r['weights']}"
        )
        if optimize_tiers:
            line += (f"  boost_gain=${r['boost_gain_usd']:+.2f}  "
                     f"multiplier_tiers={json.dumps(r['multiplier_tiers'])}")
        history_lines.append(line)

    if optimize_tiers:
        system_prompt = SYSTEM_PROMPT_WITH_TIERS
        suggest_line  = (f"Suggest {n_suggestions} new configs. "
                         f"Return only a JSON array of objects with \"weights\" and \"multiplier_tiers\".")
    else:
        system_prompt = SYSTEM_PROMPT_WEIGHTS_ONLY
        suggest_line  = f"Suggest {n_suggestions} new weight vectors to try. Return only a JSON array."

    ref = capped_history[0] if capped_history else results_history[0]
    user_msg = f"""Dataset: {ref['total_winners'] + ref['total_losers']} trades,
{ref['total_winners']} winners, {ref['total_losers']} losers.
min_score threshold: {min_score}
KNN training label: {label_key}
KNN: k={KNN_K}, halflife={KNN_HL}d, score_range=[{KNN_SL}, {KNN_SH}]
HARD CONSTRAINT: winner_miss_rate must be <= {max_miss_rate*100:.0f}% (configs violating this are rejected){locked_note}
FEATURE NAMES (index order):
{chr(10).join(f'  [{i}] {n}' for i, n in enumerate(FEAT_NAMES))}

FEATURE SEPARABILITY (winners: {label_key} > {sep_threshold}):
{sep_lines}

RESULTS HISTORY (all configs tested so far):
{chr(10).join(history_lines)}

{suggest_line}"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=8192,
        system=system_prompt,
        messages=[{"role": "user", "content": user_msg}],
    )

    text = response.content[0].text.strip()
    start = text.find("[")
    end   = text.rfind("]") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON array in Claude response: {text[:200]}")
    raw = text[start:end]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        # Surface stop_reason to diagnose future truncations
        stop_reason = getattr(response, "stop_reason", "unknown")
        raise ValueError(
            f"JSON parse error ({e}) — stop_reason={stop_reason} — "
            f"response tail: ...{raw[-200:]}"
        ) from e

    def _clamp_weights(w: list) -> list[float]:
        clamped = [max(0.0, min(10.0, float(x))) for x in w]
        for idx in locked_zero_indices:
            clamped[idx] = 0.0
        return clamped

    def _clamp_tiers(tiers: list) -> list[list[float]]:
        valid_tiers = []
        for tier in tiers:
            if isinstance(tier, list) and len(tier) == 2:
                t = max(float(min_score), min(10.0, float(tier[0])))
                m = max(1.0, min(5.0, float(tier[1])))
                valid_tiers.append([t, m])
        valid_tiers.sort(key=lambda x: x[0])
        return valid_tiers

    valid = []
    if optimize_tiers:
        for obj in parsed:
            if not isinstance(obj, dict):
                continue
            w = obj.get("weights")
            tiers = obj.get("multiplier_tiers")
            if not isinstance(w, list) or len(w) != N_FEAT or not isinstance(tiers, list):
                continue
            valid.append({"weights": _clamp_weights(w), "multiplier_tiers": _clamp_tiers(tiers)})
    else:
        for w in parsed:
            if isinstance(w, list) and len(w) == N_FEAT:
                valid.append({"weights": _clamp_weights(w), "multiplier_tiers": []})

    return valid[:n_suggestions]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="AI-driven ML weight optimizer")
    parser.add_argument("--strategy",    type=str,   default="quick_pop",
                        choices=list(STRATEGY_CONFIGS.keys()),
                        help="Strategy to optimize (default: quick_pop)")
    parser.add_argument("--rounds",      type=int,   default=4,   help="Optimization rounds")
    parser.add_argument("--suggestions", type=int,   default=8,   help="Suggestions per round")
    parser.add_argument("--min-score",     type=float, default=None,
                        help="KNN min_score threshold (default: 1.5 normal, 8.0 strict)")
    parser.add_argument("--max-miss-rate",    type=float, default=None,
                        help="Max winner miss rate (default: 0.25 normal, 0.85 strict)")
    parser.add_argument("--random-per-round", type=int,   default=3,
                        help="Random configs injected each round for exploration (default 3)")
    parser.add_argument("--optimize-tiers", action="store_true",
                        help="Also let AI suggest tiered buy multipliers per score level (default: all trades 1×)")
    parser.add_argument("--strict", action="store_true",
                        help="Strict mode: target ≤5%% pass rate and >50%% win rate. "
                             "Sets min-score=8.0, max-miss-rate=0.85, ranks by win_rate first.")
    parser.add_argument("--lenient", action="store_true",
                        help="Lenient mode: miss ZERO winners, block as many losers as possible. "
                             "Sets min-score=3.5, max-miss-rate=0.02.")
    parser.add_argument("--db",            type=str,   default=DB_PATH)
    args = parser.parse_args()

    if args.strict and args.lenient:
        parser.error("--strict and --lenient are mutually exclusive")

    # Apply mode defaults (can still be overridden by explicit flags)
    if args.min_score is None:
        if args.strict:
            args.min_score = 8.0
        elif args.lenient:
            args.min_score = 3.5
        else:
            args.min_score = 1.5
    if args.max_miss_rate is None:
        if args.strict:
            args.max_miss_rate = 0.85
        elif args.lenient:
            args.max_miss_rate = 0.02
        else:
            args.max_miss_rate = 0.25

    cfg = STRATEGY_CONFIGS[args.strategy]
    global KNN_K, KNN_HL, KNN_SL, KNN_SH
    KNN_K  = cfg["knn_k"]
    KNN_HL = cfg["knn_hl"]
    KNN_SL = cfg["knn_sl"]
    KNN_SH = cfg["knn_sh"]

    label_key           = cfg["label_key"]
    sep_threshold       = cfg["sep_threshold"]
    locked_zero_indices = cfg["locked_zero_indices"]
    max_miss_rate       = args.max_miss_rate

    print(f"Strategy: {args.strategy} (db: {cfg['db_strategy']}, hl={KNN_HL}d, score=[{KNN_SL},{KNN_SH}], label={label_key})")
    if locked_zero_indices:
        print(f"Locked-zero features: idx {locked_zero_indices} (not available at inference)")
    print(f"Max miss rate: {max_miss_rate*100:.0f}%")
    print(f"Loading data from {args.db}...")
    records   = load_data(args.db, cfg["db_strategy"])
    all_feats = precompute_features(records)
    valid     = [(i, f) for i, f in enumerate(all_feats) if f is not None]
    n         = len(records)
    n_winners = sum(1 for r in records if r["outcome_pnl_pct"] > 0)
    n_valid   = len(valid)
    print(f"{n} trades | {n_winners} winners | {n - n_winners} losers | {n_valid} with valid features\n")

    sep = compute_separability(records, all_feats, sep_threshold, label_key)

    # Seed config: production weights; tiers only if --optimize-tiers was passed
    seed_weights = list(cfg["seed_weights"])
    seed_tiers   = cfg.get("seed_multiplier_tiers", []) if args.optimize_tiers else []
    for idx in locked_zero_indices:
        seed_weights[idx] = 0.0

    results_history: list[dict] = []
    _seen_weight_hashes: set[tuple] = set()

    def _weight_hash(w: list[float]) -> tuple:
        """Round to 1dp so near-identical configs are deduplicated."""
        return tuple(round(x, 1) for x in w)

    def generate_random_config() -> dict:
        """Random weight vector + optional random tiers for broad space exploration."""
        w = []
        for i in range(N_FEAT):
            if i in locked_zero_indices:
                w.append(0.0)
            elif random.random() < 0.45:   # ~45% of features zeroed out
                w.append(0.0)
            else:
                w.append(round(random.uniform(0.5, 10.0), 2))
        tiers = []
        if args.optimize_tiers:
            n_tiers = random.randint(1, 3)
            # sample thresholds within the passing score range (seed p25–p90 or fallback)
            sp = results_history[0].get("score_pcts", {}) if results_history else {}
            lo = sp.get("p25", args.min_score + 0.5)
            hi = sp.get("p90", 5.0)
            thresholds = sorted(random.uniform(lo, hi) for _ in range(n_tiers))
            for j, t in enumerate(thresholds):
                mult = round(random.uniform(1.2, 3.5), 2)
                tiers.append([round(t, 2), mult])
        return {"weights": w, "multiplier_tiers": tiers}

    def _percentile(scores: list[float], p: float) -> float:
        if not scores:
            return 0.0
        s = sorted(scores)
        idx = (len(s) - 1) * p / 100.0
        lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
        return s[lo] + (s[hi] - s[lo]) * (idx - lo)

    def evaluate_and_record(weights: list[float], tiers: list[list[float]], label: str) -> dict | None:
        h = _weight_hash(weights)
        if h in _seen_weight_hashes:
            print(f"  [{label:<18}] (duplicate — skipped)")
            return None
        _seen_weight_hashes.add(h)
        (wt, lb, tw, tl, pnl_saved, pnl_missed,
         through_base, through_boosted,
         passing_scores) = loo_evaluate(
            records, all_feats, weights, args.min_score,
            k=KNN_K, halflife=KNN_HL, score_low=KNN_SL, score_high=KNN_SH,
            label_key=label_key,
            multiplier_tiers=tiers,
        )
        boost_gain   = through_boosted - through_base
        combined_pnl = through_boosted + pnl_saved
        net_pnl      = pnl_saved - pnl_missed
        score_pcts   = {
            "p25": _percentile(passing_scores, 25),
            "p50": _percentile(passing_scores, 50),
            "p75": _percentile(passing_scores, 75),
            "p90": _percentile(passing_scores, 90),
            "max": max(passing_scores) if passing_scores else 0.0,
        }
        result = {
            "label":              label,
            "weights":            weights,
            "multiplier_tiers":   tiers,
            "winners_through":    wt,
            "losers_blocked":     lb,
            "total_winners":      tw,
            "total_losers":       tl,
            "combined":           wt + lb,
            "pnl_saved_usd":      pnl_saved,
            "pnl_missed_usd":     pnl_missed,
            "net_pnl_usd":        net_pnl,
            "through_pnl_base":   through_base,
            "through_pnl_boosted":through_boosted,
            "boost_gain_usd":     boost_gain,
            "combined_pnl_usd":   combined_pnl,
            "score_pcts":         score_pcts,
        }
        miss_rate    = (tw - wt) / tw * 100 if tw else 0
        trades_through = wt + (tl - lb)   # winners + losers that passed
        win_rate_pct   = wt / trades_through * 100 if trades_through else 0
        pass_rate_pct  = trades_through / (tw + tl) * 100 if (tw + tl) else 0
        result["win_rate_pct"]   = win_rate_pct
        result["pass_rate_pct"]  = pass_rate_pct
        result["trades_through"] = trades_through
        tiers_str = json.dumps(tiers) if tiers else "[]"
        win_flag = " >50%!" if win_rate_pct > 50 else ""
        print(
            f"  [{label:<18}] through=${through_base:>+7.2f}  net=${net_pnl:>+7.2f}  "
            f"winners={wt}/{tw} (miss={miss_rate:.0f}%)  blocked={lb}/{tl}  "
            f"win_rate={win_rate_pct:.0f}%{win_flag}  pass_rate={pass_rate_pct:.1f}%  "
            f"saved=${pnl_saved:>+6.2f}  missed=${pnl_missed:>+6.2f}  "
            f"boost=${boost_gain:>+6.2f}  "
            f"scores(p50={score_pcts['p50']:.1f} p75={score_pcts['p75']:.1f} p90={score_pcts['p90']:.1f} max={score_pcts['max']:.1f})  "
            f"tiers={tiers_str}"
        )
        return result

    mode_label = "weights + tiers" if args.optimize_tiers else "weights only (all trades 1×)"
    print(f"Optimization mode: {mode_label}")
    print("Round 0: Evaluating seed (production weights)...")
    r0 = evaluate_and_record(seed_weights, seed_tiers, "production")
    if r0:
        results_history.append(r0)

    client = anthropic.Anthropic()

    for round_num in range(1, args.rounds + 1):
        round_desc = "weights + tiers" if args.optimize_tiers else "weight vectors"
        print(f"\nRound {round_num}: Asking Claude for {args.suggestions} {round_desc}...")
        try:
            suggestions = ask_claude_for_configs(
                client, sep, results_history, args.suggestions, args.min_score,
                label_key=label_key, sep_threshold=sep_threshold,
                locked_zero_indices=locked_zero_indices, max_miss_rate=max_miss_rate,
                optimize_tiers=args.optimize_tiers,
            )
        except Exception as exc:
            print(f"  Claude error: {exc} — skipping round")
            continue

        print(f"  Received {len(suggestions)} configs. Evaluating...")
        for idx, cfg_suggestion in enumerate(suggestions):
            label = f"r{round_num}_s{idx+1}"
            result = evaluate_and_record(
                cfg_suggestion["weights"], cfg_suggestion["multiplier_tiers"], label
            )
            if result:
                results_history.append(result)

        if args.random_per_round > 0:
            print(f"  Injecting {args.random_per_round} random configs for exploration...")
            for idx in range(args.random_per_round):
                rand_cfg = generate_random_config()
                label = f"r{round_num}_rand{idx+1}"
                result = evaluate_and_record(rand_cfg["weights"], rand_cfg["multiplier_tiers"], label)
                if result:
                    results_history.append(result)

    if args.strict:
        # Strict mode: rank by win_rate > 50% first, then through_pnl.
        # A config with 60% win rate and $+5 beats one with 40% win rate and $+20.
        results_history.sort(
            key=lambda r: (1 if r["win_rate_pct"] > 50 else 0,
                           r["through_pnl_boosted"], r["net_pnl_usd"]),
            reverse=True,
        )
        sort_note = "win_rate>50% first, then through_pnl"
    elif args.lenient:
        # Lenient mode: fewest winner misses first, then highest through_pnl.
        # Goal: let ALL winners through while blocking as many losers as possible.
        results_history.sort(
            key=lambda r: (-(r["total_winners"] - r["winners_through"]),
                           r["through_pnl_boosted"], r["losers_blocked"]),
            reverse=False,
        )
        # Reverse=False because we want the smallest miss count first, but
        # through_pnl and losers_blocked should be maximized — negate them inline.
        results_history.sort(
            key=lambda r: (r["total_winners"] - r["winners_through"],
                           -r["through_pnl_boosted"], -r["losers_blocked"]),
        )
        sort_note = "fewest winner misses first, then through_pnl"
    else:
        # Normal mode: sort by through_pnl (actual trading PnL) — the only metric that
        # maps to real profit. net_pnl_usd secondary, boost_gain tertiary when tiers active.
        results_history.sort(
            key=lambda r: (r["through_pnl_boosted"], r["net_pnl_usd"], r["combined"]),
            reverse=True,
        )
        sort_note = "through_pnl = actual trading PnL"

    print("\n" + "=" * 80)
    print(f"  OPTIMIZATION RESULTS — TOP 10  (sorted by {sort_note})")
    print("=" * 80)
    print(f"  {'Label':<20} {'ThroughPnL':>11} {'NetPnL':>8} {'WinRate':>8} {'PassRate':>9} {'Saved':>7} {'Missed':>7} {'Boost':>7} {'ScoreP50':>8} {'Miss%':>6}  Tiers")
    print("  " + "-" * 110)
    for r in results_history[:10]:
        miss_pct = (r["total_winners"] - r["winners_through"]) / r["total_winners"] * 100
        flag = " !" if miss_pct / 100 > max_miss_rate else ""
        sp = r.get("score_pcts", {})
        win_flag = "*" if r.get("win_rate_pct", 0) > 50 else " "
        print(
            f"  {r['label']:<20} ${r['through_pnl_boosted']:>+9.2f}  "
            f"${r['net_pnl_usd']:>+6.2f}  "
            f"{r.get('win_rate_pct', 0):>7.0f}%{win_flag}  "
            f"{r.get('pass_rate_pct', 0):>8.1f}%  "
            f"${r['pnl_saved_usd']:>5.2f}  ${r['pnl_missed_usd']:>5.2f}  "
            f"${r['boost_gain_usd']:>+5.2f}  "
            f"{sp.get('p50', 0):>8.2f}  "
            f"{miss_pct:>5.0f}%{flag}  "
            f"{json.dumps(r['multiplier_tiers'])}"
        )

    # Hard-filter: only consider configs within the miss rate constraint.
    eligible = [r for r in results_history
                if (r["total_winners"] - r["winners_through"]) / r["total_winners"] <= max_miss_rate]
    if not eligible:
        print(f"\n  [WARN] No config met the {max_miss_rate*100:.0f}% miss rate constraint.")
        print(f"         Showing lowest-miss config instead — consider more training data.\n")
        eligible = sorted(results_history,
                          key=lambda r: (r["total_winners"] - r["winners_through"]) / r["total_winners"])

    best = eligible[0]
    sp = best.get("score_pcts", {})
    print("\n" + "=" * 80)
    print("  BEST CONFIGURATION")
    print("=" * 80)
    miss_pct = (best["total_winners"] - best["winners_through"]) / best["total_winners"] * 100
    print(f"  Label:             {best['label']}")
    _mode_label = "STRICT" if args.strict else ("LENIENT" if args.lenient else "normal")
    print(f"  Mode:              {_mode_label}")
    print(f"  Net PnL:           ${best['net_pnl_usd']:+.2f}  "
          f"(saved ${best['pnl_saved_usd']:.2f} — missed ${best['pnl_missed_usd']:.2f})")
    print(f"  Boost gain:        ${best['boost_gain_usd']:+.2f}  "
          f"(through base ${best['through_pnl_base']:+.2f} → boosted ${best['through_pnl_boosted']:+.2f})")
    print(f"  Combined score:    {best['combined']}")
    print(f"  Win rate:          {best.get('win_rate_pct', 0):.0f}%  "
          f"({best['winners_through']} winners / {best.get('trades_through', '?')} passed)")
    print(f"  Pass rate:         {best.get('pass_rate_pct', 0):.1f}%  "
          f"({best.get('trades_through', '?')} / {best['total_winners'] + best['total_losers']} trades)")
    print(f"  Winners through:   {best['winners_through']}/{best['total_winners']} ({miss_pct:.0f}% miss)")
    print(f"  Losers blocked:    {best['losers_blocked']}/{best['total_losers']}")
    print(f"  Passing score pcts: p25={sp.get('p25',0):.2f}  p50={sp.get('p50',0):.2f}  "
          f"p75={sp.get('p75',0):.2f}  p90={sp.get('p90',0):.2f}  max={sp.get('max',0):.2f}")
    print()
    if args.optimize_tiers:
        print(f"  multiplier_tiers = {json.dumps(best['multiplier_tiers'])}")
        print()
    print("  ml_feature_weights=(")
    w = best["weights"]
    print(f"      {w[0]}, {w[1]}, {w[2]}, {w[3]}, {w[4]}, {w[5]},   # idx  0-5:  15s OHLCV")
    print(f"      {w[6]}, {w[7]}, {w[8]}, {w[9]}, {w[10]}, {w[11]},  # idx  6-11: 1m OHLCV")
    print(f"      {w[12]}, {w[13]}, {w[14]}, {w[15]}, {w[16]}, {w[17]},  # idx 12-17: pair stats + source_channel")
    print(f"      {w[18]}, {w[19]}, {w[20]},  # idx 18-20: token metadata")
    print(f"      {w[21]}, {w[22]}, {w[23]}, {w[24]},  # idx 21-24: wallet (5m)")
    print(f"      {w[25]}, {w[26]},  # idx 25-26: wallet_momentum_30m, top10_holder_pct")
    if len(w) > 27:
        print(f"      {w[27]}, {w[28]}, {w[29]}, {w[30]}, {w[31]}, {w[32]},  # idx 27-32: 1s OHLCV")
        print(f"      {w[33]}, {w[34]}, {w[35]}, {w[36]}, {w[37]}, {w[38]}, {w[39]}, {w[40]}, {w[41]}, {w[42]},  # idx 33-42: 15s shape")
        print(f"      {w[43]}, {w[44]}, {w[45]}, {w[46]}, {w[47]}, {w[48]}, {w[49]}, {w[50]}, {w[51]}, {w[52]},  # idx 43-52: 1m shape")
        print(f"      {w[53]}, {w[54]}, {w[55]}, {w[56]}, {w[57]}, {w[58]}, {w[59]}, {w[60]}, {w[61]}, {w[62]},  # idx 53-62: 1s shape")
    print("  )")
    print()
    print(f"To apply: copy weights + multiplier_tiers above into registry.py for {cfg['registry_name']}.")


if __name__ == "__main__":
    main()
