"""
tests/test_ml_scorer.py — Unit tests for ChartMLScorer and supporting functions.

Covers:
- extract_features: 18-element output, neutrals when optional inputs absent,
  None when both candle sources have fewer than 3 candles
- zscore_normalize: zero-mean / unit-variance over training set
- euclidean: correct distance
- ChartMLScorer: None when too few snapshots, score in [0, 10],
  custom score_low_pct / score_high_pct mapping, configurable k / halflife

No network calls, no DB, no Anthropic API.
Run with: pytest tests/test_ml_scorer.py -v
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from typing import Optional

import pytest

from trader.analysis.ml_scorer import (
    MIN_SAMPLES,
    ChartMLScorer,
    euclidean,
    extract_features,
    zscore_normalize,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candles(n: int, price: float = 1.0, vol: float = 1000.0) -> list[dict]:
    """Generate n simple candles with flat price and constant volume."""
    return [
        {"t": i * 60, "o": price, "h": price * 1.01, "l": price * 0.99,
         "c": price, "v": vol}
        for i in range(n)
    ]


def _make_snapshot(pnl_pct: float, days_ago: float = 1.0) -> dict:
    """Return a minimal snapshot dict compatible with ChartMLScorer.score()."""
    candles = _make_candles(10)
    ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    return {
        "candles_json":    json.dumps(candles),
        "candles_1m_json": json.dumps(_make_candles(8)),
        "pair_stats_json": None,
        "pump_ratio":      None,
        "vol_trend":       None,
        "outcome_pnl_pct": pnl_pct,
        "ts":              ts,
    }


class MockDB:
    def __init__(self, snapshots: list[dict]):
        self._snapshots = snapshots

    def load_chart_snapshots(self, strategy: str, label_column: str = "outcome_pnl_pct") -> list[dict]:
        return self._snapshots


# ---------------------------------------------------------------------------
# extract_features
# ---------------------------------------------------------------------------

class TestExtractFeatures:
    def test_returns_27_features_with_both_sources(self):
        feat = extract_features(_make_candles(10), candles_1m=_make_candles(8))
        assert feat is not None
        assert len(feat) == 27

    def test_returns_27_features_with_15s_only(self):
        # candles_1m absent → features 7-12 are neutral but vector is still 27
        feat = extract_features(_make_candles(10))
        assert feat is not None
        assert len(feat) == 27

    def test_returns_27_features_with_1m_only(self):
        # candles_15s too short → features 1-6 are neutral but vector is still 27
        feat = extract_features([], candles_1m=_make_candles(10))
        assert feat is not None
        assert len(feat) == 27

    def test_returns_none_when_both_sources_too_short(self):
        assert extract_features(_make_candles(2)) is None
        assert extract_features([]) is None
        assert extract_features(_make_candles(2), candles_1m=_make_candles(2)) is None

    def test_pump_ratio_15s_flat_price_is_one(self):
        feat = extract_features(_make_candles(10, price=2.0))
        # features 1-6 from 15s candles; pump_ratio = feat[0]
        assert feat[0] >= 1.0

    def test_pump_ratio_1m_flat_price_is_one(self):
        feat = extract_features([], candles_1m=_make_candles(10, price=2.0))
        # features 7-12 from 1m candles; pump_ratio_1m = feat[6]
        assert feat[6] >= 1.0

    def test_15s_neutral_when_short(self):
        # candles_15s has <3 → features 1-6 should be neutral [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        feat = extract_features([], candles_1m=_make_candles(10))
        assert feat[0] == pytest.approx(1.0)   # neutral pump_ratio
        assert feat[1] == pytest.approx(1.0)   # neutral vol_momentum
        assert feat[2] == pytest.approx(0.0)   # neutral price_slope
        assert feat[5] == pytest.approx(0.0)   # neutral candle_count_norm

    def test_1m_neutral_when_absent(self):
        # candles_1m absent → features 7-12 should be neutral
        feat = extract_features(_make_candles(10))
        assert feat[6]  == pytest.approx(1.0)   # neutral pump_ratio_1m
        assert feat[7]  == pytest.approx(1.0)   # neutral vol_momentum_1m
        assert feat[8]  == pytest.approx(0.0)   # neutral price_slope_1m
        assert feat[11] == pytest.approx(0.0)   # neutral candle_count_norm_1m

    def test_pair_stats_buy_ratio_balanced_when_absent(self):
        feat = extract_features(_make_candles(10))
        assert feat[12] == pytest.approx(0.5)   # neutral buy_ratio_5m (index 12)

    def test_pair_stats_buy_ratio_all_buys(self):
        feat = extract_features(_make_candles(10), pair_stats={"buys_5m": 100, "sells_5m": 0})
        assert feat[12] == pytest.approx(1.0, abs=0.01)

    def test_pair_stats_buy_vol_neutral_when_absent(self):
        feat = extract_features(_make_candles(10))
        assert feat[15] == pytest.approx(0.5)   # neutral buy_vol_ratio_1h (index 15)

    def test_price_slope_positive_for_rising_10s_candles(self):
        candles = [
            {"t": i, "o": 1.0 + i * 0.01, "h": 1.1 + i * 0.01,
             "l": 0.9 + i * 0.01, "c": 1.0 + i * 0.01, "v": 100.0}
            for i in range(10)
        ]
        feat = extract_features(candles)
        assert feat[2] > 0   # price_slope_15s

    def test_price_slope_positive_for_rising_1m_candles(self):
        candles = [
            {"t": i * 60, "o": 1.0 + i * 0.01, "h": 1.1 + i * 0.01,
             "l": 0.9 + i * 0.01, "c": 1.0 + i * 0.01, "v": 100.0}
            for i in range(10)
        ]
        feat = extract_features([], candles_1m=candles)
        assert feat[8] > 0   # price_slope_1m

    def test_candle_count_norm_15s_scales_with_n(self):
        feat10 = extract_features(_make_candles(10))
        feat20 = extract_features(_make_candles(20))
        assert feat20[5] > feat10[5]   # candle_count_15s

    def test_candle_count_norm_1m_scales_with_n(self):
        feat10 = extract_features([], candles_1m=_make_candles(10))
        feat20 = extract_features([], candles_1m=_make_candles(20))
        assert feat20[11] > feat10[11]  # candle_count_1m

    # ------------------------------------------------------------------
    # Features 25-26: wallet_momentum_30m, top10_holder_pct
    # ------------------------------------------------------------------

    def test_wallet_momentum_30m_fallback_when_absent(self):
        # No pair_stats → neutral fallback 1.0
        feat = extract_features(_make_candles(10))
        assert feat[25] == pytest.approx(1.0)

    def test_top10_holder_pct_fallback_when_absent(self):
        # No pair_stats → neutral fallback 0.5
        feat = extract_features(_make_candles(10))
        assert feat[26] == pytest.approx(0.5)

    def test_wallet_momentum_30m_growing(self):
        # uw30=100 > uwh30=50 → ratio 2.0
        ps = {"unique_wallet_30m": 100, "unique_wallet_hist_30m": 50}
        feat = extract_features(_make_candles(10), pair_stats=ps)
        assert feat[25] == pytest.approx(2.0)

    def test_wallet_momentum_30m_shrinking(self):
        # uw30=20, uwh30=100 → ratio 0.2
        ps = {"unique_wallet_30m": 20, "unique_wallet_hist_30m": 100}
        feat = extract_features(_make_candles(10), pair_stats=ps)
        assert feat[25] == pytest.approx(0.2)

    def test_wallet_momentum_30m_capped_at_five(self):
        # ratio > 5 → capped at 5.0
        ps = {"unique_wallet_30m": 600, "unique_wallet_hist_30m": 1}
        feat = extract_features(_make_candles(10), pair_stats=ps)
        assert feat[25] == pytest.approx(5.0)

    def test_wallet_momentum_30m_zero_history_no_div_error(self):
        # uwh30=0 → denominator clamped to 1
        ps = {"unique_wallet_30m": 50, "unique_wallet_hist_30m": 0}
        feat = extract_features(_make_candles(10), pair_stats=ps)
        assert feat[25] == pytest.approx(min(50.0, 5.0))

    def test_top10_holder_pct_exact(self):
        ps = {"top10_concentration": 0.65}
        feat = extract_features(_make_candles(10), pair_stats=ps)
        assert feat[26] == pytest.approx(0.65)

    def test_top10_holder_pct_clamped_above_one(self):
        ps = {"top10_concentration": 1.5}
        feat = extract_features(_make_candles(10), pair_stats=ps)
        assert feat[26] == pytest.approx(1.0)

    def test_top10_holder_pct_clamped_below_zero(self):
        ps = {"top10_concentration": -0.1}
        feat = extract_features(_make_candles(10), pair_stats=ps)
        assert feat[26] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# zscore_normalize
# ---------------------------------------------------------------------------

class TestZscoreNormalize:
    def test_normalized_training_has_zero_mean(self):
        training = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        query = [3.0, 4.0]
        _, norm_training = zscore_normalize(query, training)
        for feat_idx in range(2):
            vals = [row[feat_idx] for row in norm_training]
            mean = sum(vals) / len(vals)
            assert abs(mean) < 1e-10

    def test_query_at_mean_normalizes_to_zero(self):
        training = [[1.0], [3.0], [5.0]]  # mean=3
        query = [3.0]
        norm_q, _ = zscore_normalize(query, training)
        assert abs(norm_q[0]) < 1e-10

    def test_constant_feature_uses_std_one(self):
        # If all training values are identical, std=0 → replaced with 1.0
        training = [[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]]
        query = [5.0, 2.0]
        norm_q, norm_t = zscore_normalize(query, training)
        # Feature 0 is constant → (5-5)/1 = 0 for all
        assert all(abs(row[0]) < 1e-10 for row in norm_t)
        assert abs(norm_q[0]) < 1e-10


# ---------------------------------------------------------------------------
# euclidean
# ---------------------------------------------------------------------------

class TestEuclidean:
    def test_same_point_is_zero(self):
        assert euclidean([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(0.0)

    def test_known_distance(self):
        # 3-4-5 right triangle
        assert euclidean([0.0, 0.0], [3.0, 4.0]) == pytest.approx(5.0)

    def test_one_dimension(self):
        assert euclidean([2.0], [5.0]) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# ChartMLScorer
# ---------------------------------------------------------------------------

class TestChartMLScorer:
    def test_returns_none_when_too_few_snapshots(self):
        db = MockDB([_make_snapshot(10.0)] * (MIN_SAMPLES - 1))
        scorer = ChartMLScorer(db)
        result = scorer.score(_make_candles(10))
        assert result is None

    def test_score_in_range_with_enough_snapshots(self):
        db = MockDB([_make_snapshot(20.0)] * MIN_SAMPLES)
        scorer = ChartMLScorer(db)
        result = scorer.score(_make_candles(10))
        assert result is not None
        assert 0.0 <= result <= 10.0

    def test_returns_none_when_candles_too_short(self):
        db = MockDB([_make_snapshot(20.0)] * MIN_SAMPLES)
        scorer = ChartMLScorer(db)
        result = scorer.score(_make_candles(2))  # fewer than 3
        assert result is None

    def test_custom_score_low_high_stored(self):
        db = MockDB([])
        scorer = ChartMLScorer(db, score_low_pct=-50.0, score_high_pct=100.0)
        assert scorer._score_low_pct == -50.0
        assert scorer._score_high_pct == 100.0

    def test_score_maps_low_pnl_to_zero(self):
        """All neighbours return pnl = score_low_pct → score should be 0."""
        low = -35.0
        db = MockDB([_make_snapshot(low)] * MIN_SAMPLES)
        scorer = ChartMLScorer(db, score_low_pct=low, score_high_pct=85.0)
        result = scorer.score(_make_candles(10))
        assert result is not None
        assert result == pytest.approx(0.0, abs=0.01)

    def test_score_maps_high_pnl_to_ten(self):
        """All neighbours return pnl = score_high_pct → score should be 10."""
        high = 85.0
        db = MockDB([_make_snapshot(high)] * MIN_SAMPLES)
        scorer = ChartMLScorer(db, score_low_pct=-35.0, score_high_pct=high)
        result = scorer.score(_make_candles(10))
        assert result is not None
        assert result == pytest.approx(10.0, abs=0.01)

    def test_score_clamped_at_zero_below_low(self):
        """PnL worse than score_low_pct → clamped to 0."""
        db = MockDB([_make_snapshot(-100.0)] * MIN_SAMPLES)
        scorer = ChartMLScorer(db, score_low_pct=-35.0, score_high_pct=85.0)
        result = scorer.score(_make_candles(10))
        assert result == pytest.approx(0.0, abs=0.01)

    def test_score_clamped_at_ten_above_high(self):
        """PnL better than score_high_pct → clamped to 10."""
        db = MockDB([_make_snapshot(500.0)] * MIN_SAMPLES)
        scorer = ChartMLScorer(db, score_low_pct=-35.0, score_high_pct=85.0)
        result = scorer.score(_make_candles(10))
        assert result == pytest.approx(10.0, abs=0.01)

    def test_midpoint_pnl_scores_near_five(self):
        """PnL at midpoint of [low, high] → score ≈ 5."""
        low, high = -35.0, 85.0
        mid = (low + high) / 2  # 25.0
        db = MockDB([_make_snapshot(mid)] * MIN_SAMPLES)
        scorer = ChartMLScorer(db, score_low_pct=low, score_high_pct=high)
        result = scorer.score(_make_candles(10))
        assert result is not None
        assert result == pytest.approx(5.0, abs=0.5)

    def test_custom_k_stored(self):
        db = MockDB([])
        scorer = ChartMLScorer(db, k=3)
        assert scorer._k == 3

    def test_custom_halflife_stored(self):
        db = MockDB([])
        scorer = ChartMLScorer(db, recency_halflife_days=7.0)
        assert scorer._halflife == 7.0

    def test_custom_score_range_shifts_score(self):
        """
        Tighter score range → same PnL produces a higher score.
        With pnl=25%:
          default  [−35, 85]  → score = (25+35)/120 * 10 ≈ 5.0
          tight    [0,   50]  → score = (25-0)/50  * 10 ≈ 5.0 (midpoint)
          off-centre [0, 30]  → score = (25-0)/30  * 10 ≈ 8.3
        """
        db_tight = MockDB([_make_snapshot(25.0)] * MIN_SAMPLES)
        scorer_tight = ChartMLScorer(db_tight, score_low_pct=0.0, score_high_pct=30.0)
        result_tight = scorer_tight.score(_make_candles(10))

        db_default = MockDB([_make_snapshot(25.0)] * MIN_SAMPLES)
        scorer_default = ChartMLScorer(db_default, score_low_pct=-35.0, score_high_pct=85.0)
        result_default = scorer_default.score(_make_candles(10))

        assert result_tight is not None
        assert result_default is not None
        assert result_tight > result_default
