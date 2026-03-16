"""
scripts/test_ml_mock.py — Test the KNN ML scorer with synthetic 10s candle data.

Generates realistic WIN and LOSS patterns, runs leave-one-out evaluation,
then scores a fresh query signal to verify the model produces sensible results.

WIN  pattern: flat baseline → sharp pump → high buy pressure in pair stats
LOSS pattern: flat/declining → no pump   → sellers dominating pair stats

Usage:
    source venv/bin/activate
    python scripts/test_ml_mock.py
"""

from __future__ import annotations

import math
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trader.analysis.ml_scorer import (
    K, MIN_SAMPLES,
    extract_features, zscore_normalize, euclidean,
    _SCORE_LOW_PCT, _SCORE_HIGH_PCT,
)

SEP = "-" * 72
random.seed(42)


# ---------------------------------------------------------------------------
# Mock candle generators (10s bars)
# ---------------------------------------------------------------------------

def make_candles_win(n: int = 80, pump_multiplier: float = None) -> list[dict]:
    """
    Simulate a quick_pop WIN: ~60 bars of accumulation then a sharp pump.
    pump_multiplier: how much the price peaks (default: random 1.4–3.0x)
    """
    if pump_multiplier is None:
        pump_multiplier = random.uniform(1.4, 3.0)

    base_price = random.uniform(0.00005, 0.0002)
    bars = []
    pump_start = int(n * 0.65)

    for i in range(n):
        if i < pump_start:
            # Accumulation: flat with small noise
            drift    = 1.0 + random.uniform(-0.005, 0.008)
            volume   = random.uniform(100, 500)
        else:
            # Pump: price rising sharply
            progress = (i - pump_start) / (n - pump_start)
            drift    = 1.0 + random.uniform(0.02, 0.08) * (1 + progress)
            volume   = random.uniform(2000, 8000) * (1 + progress * 2)

        open_  = base_price
        close_ = open_ * drift
        high_  = close_ * random.uniform(1.0, 1.03)
        low_   = open_  * random.uniform(0.97, 1.0)
        bars.append({"t": i * 10, "o": open_, "h": high_, "l": low_, "c": close_, "v": volume})
        base_price = close_

    return bars


def make_candles_loss(n: int = 80) -> list[dict]:
    """
    Simulate a quick_pop LOSS: flat or declining price, no real pump, volume fading.
    """
    base_price = random.uniform(0.00005, 0.0002)
    # Small initial spike then fades (false pump)
    spike_bars = random.randint(3, 8)
    bars = []

    for i in range(n):
        if i < spike_bars:
            drift  = 1.0 + random.uniform(0.01, 0.03)
            volume = random.uniform(800, 2000)
        else:
            drift  = 1.0 + random.uniform(-0.015, 0.005)
            volume = random.uniform(50, 400) * max(0.3, 1 - i / n)

        open_  = base_price
        close_ = open_ * drift
        high_  = close_ * random.uniform(1.0, 1.01)
        low_   = open_  * random.uniform(0.99, 1.0)
        bars.append({"t": i * 10, "o": open_, "h": high_, "l": low_, "c": close_, "v": volume})
        base_price = close_

    return bars


# ---------------------------------------------------------------------------
# Mock pair stats generators
# ---------------------------------------------------------------------------

def make_pair_stats_win() -> dict:
    """Strong buy pressure — buyers dominating at signal time."""
    buys_5m  = random.randint(60, 200)
    sells_5m = random.randint(5, 30)
    total_vol_1h = random.uniform(50000, 400000)
    buy_vol_1h   = total_vol_1h * random.uniform(0.60, 0.85)
    return {
        "buys_5m":                 buys_5m,
        "sells_5m":                sells_5m,
        "buy_volume_1h":           buy_vol_1h,
        "total_volume_1h":         total_vol_1h,
        "price_change_5m_pct":     random.uniform(15.0, 80.0),
        "liquidity_change_1h_pct": random.uniform(-2.0, 5.0),
    }


def make_pair_stats_loss() -> dict:
    """Sell pressure dominant — sellers dumping at signal time."""
    buys_5m  = random.randint(5, 40)
    sells_5m = random.randint(50, 300)
    total_vol_1h = random.uniform(10000, 100000)
    buy_vol_1h   = total_vol_1h * random.uniform(0.10, 0.35)
    return {
        "buys_5m":                 buys_5m,
        "sells_5m":                sells_5m,
        "buy_volume_1h":           buy_vol_1h,
        "total_volume_1h":         total_vol_1h,
        "price_change_5m_pct":     random.uniform(-40.0, 5.0),
        "liquidity_change_1h_pct": random.uniform(-10.0, 1.0),
    }


# ---------------------------------------------------------------------------
# Build training set
# ---------------------------------------------------------------------------

def build_dataset(n_wins: int = 15, n_losses: int = 15) -> list[dict]:
    records = []

    for _ in range(n_wins):
        pnl = random.uniform(35.0, 85.0)
        candles = make_candles_win()
        ps      = make_pair_stats_win()
        feat    = extract_features(candles, pair_stats=ps)
        records.append({"outcome": "WIN",  "pnl_pct": pnl,  "feat": feat,
                         "candles": candles, "pair_stats": ps})

    for _ in range(n_losses):
        pnl = random.uniform(-35.0, -18.0)
        candles = make_candles_loss()
        ps      = make_pair_stats_loss()
        feat    = extract_features(candles, pair_stats=ps)
        records.append({"outcome": "LOSS", "pnl_pct": pnl, "feat": feat,
                         "candles": candles, "pair_stats": ps})

    return records


# ---------------------------------------------------------------------------
# KNN scorer (mirrors test_ml_scorer.py)
# ---------------------------------------------------------------------------

def knn_score(
    query_feat: list[float],
    training_feats: list[list[float]],
    training_pnl: list[float],
    k: int = K,
) -> float | None:
    if len(training_feats) < MIN_SAMPLES:
        return None
    norm_q, norm_t = zscore_normalize(query_feat, training_feats)
    candidates = sorted(
        [(euclidean(norm_q, f), 1.0 / (euclidean(norm_q, f) + 1e-6), training_pnl[i])
         for i, f in enumerate(norm_t)],
        key=lambda x: x[0],
    )
    neighbours = candidates[:k]
    total_w = sum(w for _, w, _ in neighbours)
    if total_w == 0:
        return 5.0
    avg_pnl   = sum(w * p for _, w, p in neighbours) / total_w
    raw_score = (avg_pnl - _SCORE_LOW_PCT) / (_SCORE_HIGH_PCT - _SCORE_LOW_PCT) * 10.0
    return max(0.0, min(10.0, raw_score))


# ---------------------------------------------------------------------------
# Leave-one-out evaluation
# ---------------------------------------------------------------------------

def run_loo(records: list[dict], threshold: float = 5.0) -> None:
    print(SEP)
    print(f"  LEAVE-ONE-OUT  (K={K}, threshold={threshold:.0f}, n={len(records)})")
    print(SEP)
    print(f"  {'Outcome':<8}  {'PnL%':>8}  {'Score':>6}  Prediction")
    print(f"  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*14}")

    correct = scored = tp = fp = tn = fn = 0

    for i, rec in enumerate(records):
        train_feats = [r["feat"]    for j, r in enumerate(records) if j != i]
        train_pnl   = [r["pnl_pct"] for j, r in enumerate(records) if j != i]

        score = knn_score(rec["feat"], train_feats, train_pnl)
        if score is None:
            print(f"  {rec['outcome']:<8}  {rec['pnl_pct']:>+7.1f}%  {'—':>6}  not enough data")
            continue

        scored   += 1
        is_win    = rec["pnl_pct"] > 0
        pred_win  = score >= threshold
        ok        = is_win == pred_win
        correct  += ok

        if is_win and pred_win:      tp += 1
        elif not is_win and pred_win: fp += 1
        elif is_win and not pred_win: fn += 1
        else:                         tn += 1

        label = ("BUY ✓" if (pred_win and ok) else
                 "BUY ✗" if pred_win else
                 "SKIP ✓" if ok else "SKIP ✗")
        print(f"  {rec['outcome']:<8}  {rec['pnl_pct']:>+7.1f}%  {score:>6.2f}  {label}")

    print(SEP)
    if scored:
        acc  = correct / scored * 100
        prec = tp / (tp + fp) * 100 if (tp + fp) else 0
        rec_ = tp / (tp + fn) * 100 if (tp + fn) else 0
        print(f"  Accuracy  : {correct}/{scored}  ({acc:.0f}%)")
        print(f"  Precision : {tp}/{tp+fp}  ({prec:.0f}%)")
        print(f"  Recall    : {tp}/{tp+fn}  ({rec_:.0f}%)")
    print(SEP)


# ---------------------------------------------------------------------------
# Query test — score a fresh signal against the full training set
# ---------------------------------------------------------------------------

def test_query(records: list[dict]) -> None:
    print()
    print(SEP)
    print("  QUERY TEST — score a fresh signal against all training data")
    print(SEP)

    train_feats = [r["feat"]    for r in records]
    train_pnl   = [r["pnl_pct"] for r in records]

    scenarios = [
        ("Strong WIN signal",  make_candles_win(pump_multiplier=2.5), make_pair_stats_win()),
        ("Weak WIN signal",    make_candles_win(pump_multiplier=1.5), make_pair_stats_win()),
        ("LOSS signal",        make_candles_loss(),                   make_pair_stats_loss()),
        ("No pair stats (fallback)", make_candles_win(pump_multiplier=2.0), None),
    ]

    for label, candles, ps in scenarios:
        feat  = extract_features(candles, pair_stats=ps)
        score = knn_score(feat, train_feats, train_pnl)
        pred  = "BUY" if (score or 0) >= 5.0 else "SKIP"
        ps_label = "with pair_stats" if ps else "no pair_stats (neutral fallback)"
        print(f"  {label:<32}  score={score:>5.2f}  → {pred}  [{ps_label}]")

    print(SEP)


# ---------------------------------------------------------------------------
# Feature importance — show mean feature values for wins vs losses
# ---------------------------------------------------------------------------

def show_feature_importance(records: list[dict]) -> None:
    labels = [
        "pump_ratio", "vol_momentum", "price_slope", "recent_momentum",
        "volatility", "candle_count_norm", "pump_ratio_1m", "vol_trend_1m",
        "buy_ratio_5m", "activity_5m", "price_chg_5m", "buy_vol_ratio_1h", "liq_chg_1h",
    ]
    wins   = [r for r in records if r["pnl_pct"] > 0]
    losses = [r for r in records if r["pnl_pct"] <= 0]

    print()
    print(SEP)
    print("  FEATURE MEANS — wins vs losses")
    print(SEP)
    print(f"  {'Feature':<22}  {'Wins':>8}  {'Losses':>8}  {'Δ':>8}")
    print(f"  {'-'*22}  {'-'*8}  {'-'*8}  {'-'*8}")

    for i, label in enumerate(labels):
        win_mean  = sum(r["feat"][i] for r in wins)  / len(wins)
        loss_mean = sum(r["feat"][i] for r in losses) / len(losses)
        delta     = win_mean - loss_mean
        marker    = " ◀" if abs(delta) > 0.3 else ""
        print(f"  {label:<22}  {win_mean:>8.3f}  {loss_mean:>8.3f}  {delta:>+8.3f}{marker}")

    print(SEP)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(SEP)
    print("  ML MOCK DATA TEST — 10s candles + pair stats")
    print(SEP)

    # Test 1: 10s candles WITH pair stats (full feature vector)
    print("\n[1/2]  Full feature vector (10s OHLCV + pair stats)  — 13 features\n")
    records_full = build_dataset(n_wins=15, n_losses=15)
    show_feature_importance(records_full)
    run_loo(records_full)
    test_query(records_full)

    # Test 2: same candles WITHOUT pair stats (OHLCV only, features 9-13 neutral)
    print("\n[2/2]  OHLCV only (pair stats disabled)  — 13 features, 9-13 neutral\n")
    records_nostat = []
    for r in records_full:
        feat_no_ps = extract_features(r["candles"], pair_stats=None)
        records_nostat.append({**r, "feat": feat_no_ps})
    run_loo(records_nostat)
    test_query(records_nostat)
