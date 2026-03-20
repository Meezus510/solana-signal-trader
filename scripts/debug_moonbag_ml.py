#!/usr/bin/env python3
"""
scripts/debug_moonbag_ml.py — Deep investigation of ML separability for infinite_moonbag.

Phase 1: Feature distribution analysis — which features differ between winners/losers?
Phase 2: Rule-based thresholds — single-feature and combo rules via LOO-CV
Phase 3: Weighted KNN with best features emphasized
Phase 4: Summary + recommendation
"""
from __future__ import annotations

import json
import math
import sqlite3
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from trader.analysis.ml_scorer import extract_features, zscore_normalize, euclidean

DB_PATH = "trader.db"

FEAT_NAMES = [
    "pump_ratio_10s", "vol_momentum_10s", "price_slope_10s",
    "recent_momentum_10s", "volatility_10s", "candle_count_10s",
    "pump_ratio_1m", "vol_momentum_1m", "price_slope_1m",
    "recent_momentum_1m", "volatility_1m", "candle_count_1m",
    "buy_ratio_5m", "activity_5m_norm", "price_change_5m_norm",
    "buy_vol_ratio_1h", "liquidity_change_1h", "source_channel",
]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(db_path: str) -> list[dict]:
    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT so.outcome_pnl_pct, so.outcome_max_gain_pct, so.position_peak_pnl_pct,
               sc.candles_json, sc.candles_1m_json, sc.pair_stats_json,
               sc.source_channel, sc.ts, sc.pump_ratio as raw_pump
        FROM strategy_outcomes so
        JOIN signal_charts sc ON sc.id = so.signal_chart_id
        WHERE so.strategy = 'infinite_moonbag' AND so.closed = 1 AND so.entered = 1
        ORDER BY so.id
    """).fetchall()
    conn.close()
    records = []
    for r in rows:
        pnl, max_gain, peak_pnl, c10s_raw, c1m_raw, ps_raw, ch, ts, raw_pump = r
        c10s = json.loads(c10s_raw) if c10s_raw else []
        c1m  = json.loads(c1m_raw)  if c1m_raw and c1m_raw != "null" else None
        ps   = json.loads(ps_raw)   if ps_raw  and ps_raw  != "null" else None
        records.append({
            "outcome_pnl_pct":       pnl or 0.0,
            "outcome_max_gain_pct":  max_gain or 0.0,
            "position_peak_pnl_pct": peak_pnl or 0.0,
            "candles_10s":  c10s,
            "candles_1m":   c1m,
            "pair_stats":   ps,
            "source_channel": ch or "",
            "ts": ts,
            "raw_pump": raw_pump or 0.0,
        })
    return records


def precompute_features(records: list[dict]) -> list[list[float] | None]:
    return [
        extract_features(
            r["candles_10s"], candles_1m=r["candles_1m"],
            pair_stats=r["pair_stats"], source_channel=r["source_channel"],
        )
        for r in records
    ]


# ---------------------------------------------------------------------------
# Phase 1: Feature separability
# ---------------------------------------------------------------------------

def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0

def _std(vals):
    if len(vals) < 2: return 0.0
    m = _mean(vals)
    return math.sqrt(sum((v - m)**2 for v in vals) / len(vals))

def feature_separability(records, all_feats):
    print("=" * 80)
    print("PHASE 1: FEATURE SEPARABILITY (winner mean vs loser mean)")
    print("=" * 80)

    winner_feats = [all_feats[i] for i, r in enumerate(records)
                    if r["outcome_pnl_pct"] > 0 and all_feats[i] is not None]
    loser_feats  = [all_feats[i] for i, r in enumerate(records)
                    if r["outcome_pnl_pct"] <= 0 and all_feats[i] is not None]

    print(f"  Winners with features: {len(winner_feats)} | Losers: {len(loser_feats)}\n")
    print(f"  {'Feature':<25} {'W_mean':>9} {'W_std':>7} {'L_mean':>9} {'L_std':>7} {'Sep':>7}")
    print("  " + "-" * 70)

    separability = []
    for fi, name in enumerate(FEAT_NAMES):
        w_vals = [f[fi] for f in winner_feats]
        l_vals = [f[fi] for f in loser_feats]
        w_mean, w_std = _mean(w_vals), _std(w_vals)
        l_mean, l_std = _mean(l_vals), _std(l_vals)
        # Cohen's d
        pooled_std = math.sqrt((w_std**2 + l_std**2) / 2) if (w_std + l_std) > 0 else 1.0
        cohens_d = abs(w_mean - l_mean) / (pooled_std + 1e-9)
        separability.append((fi, name, w_mean, w_std, l_mean, l_std, cohens_d))
        print(f"  {name:<25} {w_mean:>9.3f} {w_std:>7.3f} {l_mean:>9.3f} {l_std:>7.3f} {cohens_d:>7.3f}")

    print()
    separability.sort(key=lambda x: x[6], reverse=True)
    print("  Top features by separability (Cohen's d):")
    for fi, name, wm, ws, lm, ls, d in separability[:6]:
        direction = "WINNERS > LOSERS" if wm > lm else "LOSERS > WINNERS"
        print(f"    [{fi:2d}] {name:<28} d={d:.3f}  ({direction})")

    return separability


# ---------------------------------------------------------------------------
# Phase 2: Rule-based filters (single feature thresholds)
# ---------------------------------------------------------------------------

def loo_rule(records, all_feats, feat_idx, threshold, direction="above"):
    """
    LOO rule: pass if feat[feat_idx] <direction> threshold.
    direction: 'above' means pass if feat >= threshold (higher = better)
               'below' means pass if feat <= threshold (lower = better)
    """
    n = len(records)
    winners_through = losers_blocked = 0
    winners_total = sum(1 for r in records if r["outcome_pnl_pct"] > 0)
    losers_total = n - winners_total

    for i in range(n):
        f = all_feats[i]
        is_winner = records[i]["outcome_pnl_pct"] > 0
        if f is None:
            if is_winner: winners_through += 1
            continue
        val = f[feat_idx]
        passed = (val >= threshold) if direction == "above" else (val <= threshold)
        if is_winner and passed:
            winners_through += 1
        if not is_winner and not passed:
            losers_blocked += 1

    return winners_through, losers_blocked, winners_total, losers_total


def rule_search(records, all_feats, separability):
    print("=" * 80)
    print("PHASE 2: SINGLE-FEATURE RULE SEARCH")
    print("=" * 80)

    # For each of the top-8 features, sweep thresholds
    top_feats = separability[:8]
    best_rules = []

    for fi, name, wm, ws, lm, ls, d in top_feats:
        # Get all values for this feature
        vals = sorted(set(f[fi] for f in all_feats if f is not None))
        direction = "above" if wm > lm else "below"

        best_wt = best_lb = 0
        best_thresh = None

        for threshold in vals:
            wt, lb, wn, ln = loo_rule(records, all_feats, fi, threshold, direction)
            # Prioritise: all winners through, then maximize losers blocked
            if wt > best_wt or (wt == best_wt and lb > best_lb):
                best_wt, best_lb = wt, lb
                best_thresh = threshold

        if best_thresh is not None:
            wt, lb, wn, ln = loo_rule(records, all_feats, fi, best_thresh, direction)
            combined = wt + lb
            best_rules.append((combined, wt, lb, fi, name, best_thresh, direction, wn, ln))

    best_rules.sort(key=lambda x: (x[1], x[2]), reverse=True)
    print(f"\n  {'Feature':<28} {'Dir':<6} {'Thresh':>9} {'W':>5} {'L_blk':>6} {'Comb':>5}")
    print("  " + "-" * 65)
    for combined, wt, lb, fi, name, thr, dir_, wn, ln in best_rules[:10]:
        print(f"  {name:<28} {dir_:<6} {thr:>9.3f} {wt:>3}/{wn}  {lb:>3}/{ln}  {combined:>5}")

    return best_rules


# ---------------------------------------------------------------------------
# Phase 3: Combined two-rule filters
# ---------------------------------------------------------------------------

def loo_two_rules(records, all_feats, fi1, thr1, dir1, fi2, thr2, dir2):
    n = len(records)
    winners_through = losers_blocked = 0
    winners_total = sum(1 for r in records if r["outcome_pnl_pct"] > 0)
    losers_total = n - winners_total

    for i in range(n):
        f = all_feats[i]
        is_winner = records[i]["outcome_pnl_pct"] > 0
        if f is None:
            if is_winner: winners_through += 1
            continue
        p1 = (f[fi1] >= thr1) if dir1 == "above" else (f[fi1] <= thr1)
        p2 = (f[fi2] >= thr2) if dir2 == "above" else (f[fi2] <= thr2)
        passed = p1 and p2
        if is_winner and passed:
            winners_through += 1
        if not is_winner and not passed:
            losers_blocked += 1

    return winners_through, losers_blocked, winners_total, losers_total


def two_rule_search(records, all_feats, separability):
    print("=" * 80)
    print("PHASE 3: TWO-RULE COMBO SEARCH")
    print("=" * 80)

    top_feats = separability[:6]
    best_combos = []

    # Candidate thresholds per feature
    def candidate_thresholds(fi, wm, lm):
        vals = sorted(set(f[fi] for f in all_feats if f is not None))
        # Use percentile-based candidates to keep search manageable
        n = len(vals)
        step = max(1, n // 10)
        return vals[::step] + [vals[-1]]

    for i, (fi1, nm1, wm1, ws1, lm1, ls1, d1) in enumerate(top_feats):
        dir1 = "above" if wm1 > lm1 else "below"
        thrs1 = candidate_thresholds(fi1, wm1, lm1)
        for fi2, nm2, wm2, ws2, lm2, ls2, d2 in top_feats[i+1:]:
            dir2 = "above" if wm2 > lm2 else "below"
            thrs2 = candidate_thresholds(fi2, wm2, lm2)
            for t1 in thrs1:
                for t2 in thrs2:
                    wt, lb, wn, ln = loo_two_rules(
                        records, all_feats, fi1, t1, dir1, fi2, t2, dir2)
                    combined = wt + lb
                    best_combos.append((combined, wt, lb, fi1, nm1, t1, dir1, fi2, nm2, t2, dir2, wn, ln))

    best_combos.sort(key=lambda x: (x[1], x[2]), reverse=True)
    print(f"\n  {'Feat1':<25} {'Thr1':>8} {'Feat2':<25} {'Thr2':>8} {'W':>5} {'L_blk':>6} {'Comb':>5}")
    print("  " + "-" * 90)
    seen = set()
    shown = 0
    for row in best_combos:
        combined, wt, lb, fi1, nm1, t1, dir1, fi2, nm2, t2, dir2, wn, ln = row
        key = (fi1, round(t1, 3), fi2, round(t2, 3))
        if key in seen: continue
        seen.add(key)
        print(f"  {nm1:<25} {t1:>8.3f}  {nm2:<25} {t2:>8.3f}  {wt:>3}/{wn}  {lb:>3}/{ln}  {combined:>5}")
        shown += 1
        if shown >= 10: break

    return best_combos


# ---------------------------------------------------------------------------
# Phase 4: Weighted KNN (best features only, higher weight on top separators)
# ---------------------------------------------------------------------------

def weighted_knn_loo(records, all_feats, feat_weights, label_key, k,
                     halflife_days, score_low, score_high, min_score):
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
    winners_total = sum(1 for r in records if r["outcome_pnl_pct"] > 0)
    losers_total = n - winners_total

    for i in range(n):
        qf = all_feats[i]
        is_winner = records[i]["outcome_pnl_pct"] > 0
        if qf is None:
            if is_winner: winners_through += 1
            continue

        # Apply weights
        wq = [qf[j] * feat_weights[j] for j in range(len(qf))]

        train_feats, train_labels, recency_ws = [], [], []
        for j in range(n):
            if j == i or all_feats[j] is None: continue
            wf = [all_feats[j][k2] * feat_weights[k2] for k2 in range(len(all_feats[j]))]
            age = (now - timestamps[j]).total_seconds() / 86400.0
            train_feats.append(wf)
            train_labels.append(records[j][label_key])
            recency_ws.append(math.exp(-age / halflife_days))

        if len(train_feats) < 5:
            if is_winner: winners_through += 1
            continue

        norm_q, norm_train = zscore_normalize(wq, train_feats)

        candidates = []
        for j, feat in enumerate(norm_train):
            dist = euclidean(norm_q, feat)
            sim_w = 1.0 / (dist + 1e-6)
            w = sim_w * recency_ws[j]
            candidates.append((dist, w, train_labels[j]))

        candidates.sort(key=lambda x: x[0])
        neighbours = candidates[:k]
        total_w = sum(w for _, w, _ in neighbours)
        if total_w == 0:
            ml_score = 5.0
        else:
            avg_label = sum(w * lbl for _, w, lbl in neighbours) / total_w
            raw = (avg_label - score_low) / pnl_range * 10.0
            ml_score = max(0.0, min(10.0, raw))

        passed = ml_score >= min_score
        if is_winner and passed: winners_through += 1
        if not is_winner and not passed: losers_blocked += 1

    return winners_through, losers_blocked, winners_total, losers_total


def weighted_knn_search(records, all_feats, separability):
    print("=" * 80)
    print("PHASE 4: WEIGHTED KNN (emphasize top-separating features)")
    print("=" * 80)

    # Build two weight vectors: uniform and top-4 boosted
    uniform_weights = [1.0] * 18
    boosted_weights = [1.0] * 18
    for fi, name, wm, ws, lm, ls, d in separability[:4]:
        boosted_weights[fi] = 3.0  # 3x weight on top separators

    # Only features that matter (zero out weakest)
    selective_weights = [0.2] * 18
    for fi, name, wm, ws, lm, ls, d in separability[:6]:
        selective_weights[fi] = 3.0

    weight_configs = [
        ("uniform",    uniform_weights),
        ("top4_boost", boosted_weights),
        ("selective",  selective_weights),
    ]

    best = []
    for wname, weights in weight_configs:
        for label_key in ["outcome_max_gain_pct", "outcome_pnl_pct", "position_peak_pnl_pct"]:
            for k in [3, 5, 7]:
                for hl in [7.0, 14.0, 30.0]:
                    for sl, sh in [(-35, 150), (-20, 300), (-35, 300)]:
                        for ms in [1.0, 1.5, 2.0, 2.5, 3.0]:
                            wt, lb, wn, ln = weighted_knn_loo(
                                records, all_feats, weights, label_key,
                                k, hl, sl, sh, ms)
                            combined = wt + lb
                            best.append((combined, wt, lb, wname, label_key, k, hl, sl, sh, ms, wn, ln))

    best.sort(key=lambda x: (x[1], x[2]), reverse=True)
    print(f"\n  {'Weights':<12} {'Label':<20} {'K':>3} {'HL':>5} {'SL':>5} {'SH':>5} {'MinSc':>6} {'W':>5} {'L_blk':>6} {'Comb':>5}")
    print("  " + "-" * 90)
    shown = 0
    seen_top = set()
    for row in best:
        combined, wt, lb, wname, lk, k, hl, sl, sh, ms, wn, ln = row
        key = (wt, lb)
        if key in seen_top and shown > 5: continue
        seen_top.add(key)
        lk_short = lk.replace("outcome_", "").replace("_pct", "").replace("position_", "")
        print(f"  {wname:<12} {lk_short:<20} {k:>3} {hl:>5.0f} {sl:>5.0f} {sh:>5.0f} {ms:>6.1f} {wt:>3}/{wn}  {lb:>3}/{ln}  {combined:>5}")
        shown += 1
        if shown >= 15: break

    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    records = load_data(DB_PATH)
    all_feats = precompute_features(records)
    n = len(records)
    winners_total = sum(1 for r in records if r["outcome_pnl_pct"] > 0)
    losers_total  = n - winners_total
    print(f"\n{n} trades | {winners_total} winners | {losers_total} losers\n")

    # Phase 1
    separability = feature_separability(records, all_feats)
    print()

    # Phase 2
    best_rules = rule_search(records, all_feats, separability)
    print()

    # Phase 3
    best_combos = two_rule_search(records, all_feats, separability)
    print()

    # Phase 4
    best_knn = weighted_knn_search(records, all_feats, separability)
    print()

    # Final summary
    print("=" * 80)
    print("SUMMARY & RECOMMENDATION")
    print("=" * 80)

    # Best single rule
    br = best_rules[0] if best_rules else None
    if br:
        combined, wt, lb, fi, name, thr, dir_, wn, ln = br
        print(f"\nBest single rule:  {name} {'>=' if dir_=='above' else '<='} {thr:.3f}")
        print(f"  Winners through: {wt}/{wn}  |  Losers blocked: {lb}/{ln}  |  Combined: {combined}")

    # Best combo rule
    bc = None
    seen = set()
    for row in best_combos:
        combined, wt, lb, fi1, nm1, t1, dir1, fi2, nm2, t2, dir2, wn, ln = row
        key = (fi1, round(t1, 3), fi2, round(t2, 3))
        if key not in seen:
            bc = row
            break
    if bc:
        combined, wt, lb, fi1, nm1, t1, dir1, fi2, nm2, t2, dir2, wn, ln = bc
        print(f"\nBest 2-rule combo: {nm1} {'>=' if dir1=='above' else '<='} {t1:.3f}")
        print(f"                   AND {nm2} {'>=' if dir2=='above' else '<='} {t2:.3f}")
        print(f"  Winners through: {wt}/{wn}  |  Losers blocked: {lb}/{ln}  |  Combined: {combined}")

    # Best weighted KNN
    bk = best_knn[0] if best_knn else None
    if bk:
        combined, wt, lb, wname, lk, k, hl, sl, sh, ms, wn, ln = bk
        print(f"\nBest weighted KNN: weights={wname}, label={lk}, K={k}, HL={hl}, "
              f"score=[{sl},{sh}], min={ms}")
        print(f"  Winners through: {wt}/{wn}  |  Losers blocked: {lb}/{ln}  |  Combined: {combined}")


if __name__ == "__main__":
    main()
