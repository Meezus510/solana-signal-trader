"""
trader/analysis/ml_scorer.py — KNN-based chart confidence scorer.

Compares incoming OHLCV candles against historical quick_pop_chart snapshots
stored in the database and returns a confidence score from 0 to 10.

    0–3   chart resembles historical losers
    4–6   mixed / uncertain
    7–10  chart resembles historical winners

Algorithm
---------
Recency-weighted K-Nearest Neighbours (no external dependencies).

1. Extract a 6-dimensional feature vector from the incoming candles.
2. Load all closed snapshots for the strategy from the database.
3. Extract the same feature vector from each stored snapshot.
4. Z-score normalise everything using training-set statistics.
5. For each of the K nearest neighbours, compute a combined weight:
       w = similarity_weight × recency_weight
   where:
       similarity_weight = 1 / (euclidean_distance + ε)
       recency_weight    = exp(−age_days / HALFLIFE_DAYS)
6. Compute the weighted average PnL% of the K neighbours.
7. Map that average to [0, 10]:
       score = clamp((weighted_pnl_pct + 20) / 40 × 10, 0, 10)
   i.e. −20 % PnL → 0 | 0 % PnL → 5 | +20 % PnL → 10.

Returns None when fewer than MIN_SAMPLES closed snapshots exist.

Features (one vector per chart window)
---------------------------------------
1. pump_ratio        — current_price / recent_low
2. vol_momentum      — avg volume last 5 bars / avg volume earlier bars
3. price_slope_pct   — (close[-1] / open[0] − 1) × 100
4. recent_momentum   — (close[-1] / close[-6] − 1) × 100 (last 5 bars)
5. price_volatility  — std-dev of bar-to-bar % returns
6. candle_count_norm — candle_count / 20  (proxy for token age)
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

MIN_SAMPLES: int = 5        # refuse to score until this many closed examples exist
K: int = 5                  # nearest neighbours to use
HALFLIFE_DAYS: float = 14.0 # recency weight half-life (older data counts less)

# Candle resolution used for ML snapshots — separate from the 1-minute chart filter.
# 15-second bars at 40 bars = 10 minutes of high-resolution entry data.
ML_OHLCV_BARS: int = 40
ML_OHLCV_INTERVAL: str = "15s"

# Score mapping: PnL% range covered by [0, 10]
_SCORE_LOW_PCT:  float = -20.0   # maps to score 0
_SCORE_HIGH_PCT: float =  20.0   # maps to score 10

# Encoding for the 1-minute vol_trend label → numeric feature
_VOL_TREND_ENC: dict[str, float] = {"RISING": 1.0, "FLAT": 0.5, "DYING": 0.0}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(
    candles_data: list[dict],
    pump_ratio_1m: Optional[float] = None,
    vol_trend_1m: Optional[str] = None,
) -> Optional[list[float]]:
    """
    Extract an 8-element feature vector from a list of candle dicts.

    Features 1-6 come from the 15-second candles.
    Features 7-8 come from the 1-minute chart filter context and add a
    broader time-scale perspective that the 15s window alone can't see:

        7. pump_ratio_1m  — current_price / 20-minute recent low (1m candles).
                            Falls back to the 15s pump_ratio when unavailable.
        8. vol_trend_1m   — RISING=1.0 | FLAT=0.5 | DYING=0.0 (1m candles).
                            Falls back to 0.5 (neutral) when unavailable.

    Accepts dicts with keys {t, o, h, l, c, v} (the format stored in
    candles_json) or any dict with those same keys.

    Returns None if there are fewer than 3 candles.
    """
    if len(candles_data) < 3:
        return None

    opens   = [c["o"] for c in candles_data]
    closes  = [c["c"] for c in candles_data]
    lows    = [c["l"] for c in candles_data]
    volumes = [c["v"] for c in candles_data]

    # 1. Pump ratio (15s)
    recent_low = min(lows)
    pump_ratio = closes[-1] / recent_low if recent_low > 0 else 1.0

    # 2. Volume momentum (recent 5 bars vs earlier, 15s)
    vol_window = min(5, len(volumes) - 1)
    recent_vols  = volumes[-vol_window:]
    earlier_vols = volumes[:-vol_window] if len(volumes) > vol_window else volumes
    avg_recent  = sum(recent_vols)  / len(recent_vols)
    avg_earlier = sum(earlier_vols) / len(earlier_vols)
    vol_momentum = avg_recent / avg_earlier if avg_earlier > 0 else 1.0

    # 3. Price slope over the full window (15s)
    price_slope = (closes[-1] / opens[0] - 1.0) * 100.0 if opens[0] > 0 else 0.0

    # 4. Recent momentum — last 5 bars (15s)
    lookback = min(5, len(closes) - 1)
    base = closes[-1 - lookback]
    recent_momentum = (closes[-1] / base - 1.0) * 100.0 if base > 0 else 0.0

    # 5. Price volatility — std-dev of bar returns (15s)
    returns = []
    for i in range(1, len(closes)):
        if closes[i - 1] > 0:
            returns.append((closes[i] / closes[i - 1] - 1.0) * 100.0)
    if len(returns) >= 2:
        mean_ret = sum(returns) / len(returns)
        volatility = math.sqrt(
            sum((r - mean_ret) ** 2 for r in returns) / len(returns)
        )
    else:
        volatility = 0.0

    # 6. Candle count normalised to [0, 1] (15s)
    candle_count_norm = len(candles_data) / 20.0

    # 7. Pump ratio from 1-minute chart filter (broader 20-min window)
    #    Falls back to the 15s pump_ratio — no new info but keeps vector length stable.
    f_pump_1m = pump_ratio_1m if pump_ratio_1m is not None else pump_ratio

    # 8. Vol trend from 1-minute chart filter, encoded as float
    #    Falls back to 0.5 (FLAT / neutral) when unavailable.
    f_vol_trend_1m = _VOL_TREND_ENC.get(vol_trend_1m, 0.5) if vol_trend_1m else 0.5

    return [
        pump_ratio, vol_momentum, price_slope, recent_momentum,
        volatility, candle_count_norm,
        f_pump_1m, f_vol_trend_1m,
    ]


# ---------------------------------------------------------------------------
# Normalisation helpers (pure functions so tests can call them directly)
# ---------------------------------------------------------------------------

def zscore_normalize(
    query: list[float],
    training: list[list[float]],
) -> tuple[list[float], list[list[float]]]:
    """Z-score normalise query and all training vectors using training stats."""
    n_feat   = len(query)
    n_samp   = len(training)
    means, stds = [], []

    for i in range(n_feat):
        vals  = [training[j][i] for j in range(n_samp)]
        mean  = sum(vals) / n_samp
        var   = sum((v - mean) ** 2 for v in vals) / n_samp
        std   = math.sqrt(var) if var > 0 else 1.0
        means.append(mean)
        stds.append(std)

    norm_query    = [(query[i] - means[i]) / stds[i] for i in range(n_feat)]
    norm_training = [
        [(training[j][i] - means[i]) / stds[i] for i in range(n_feat)]
        for j in range(n_samp)
    ]
    return norm_query, norm_training


def euclidean(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class ChartMLScorer:
    """
    Recency-weighted KNN chart confidence scorer.

    Usage::

        scorer = ChartMLScorer(db)
        score = scorer.score(candles)   # float 0–10 or None
    """

    def __init__(
        self,
        db,
        strategy: str = "quick_pop_chart",
        k: int = K,
        recency_halflife_days: float = HALFLIFE_DAYS,
    ) -> None:
        self._db = db
        self._strategy = strategy
        self._k = k
        self._halflife = recency_halflife_days

    def score(self, candles: list, chart_ctx=None) -> Optional[float]:
        """
        Score incoming candles against historical closed snapshots.

        candles    — list of OHLCVCandle objects (trader.analysis.chart) or
                     dicts {t, o, h, l, c, v}.
        chart_ctx  — ChartContext from the 1-minute chart filter (optional).
                     When provided, features 7 and 8 (pump_ratio_1m, vol_trend_1m)
                     are included, giving the model a broader time-scale view.

        Returns a float in [0.0, 10.0] or None if there are fewer than
        MIN_SAMPLES closed snapshots to learn from.
        """
        snapshots = self._db.load_chart_snapshots(self._strategy)
        if len(snapshots) < MIN_SAMPLES:
            logger.debug(
                "[ML] Not enough training data (%d/%d closed snapshots) — skipping score",
                len(snapshots), MIN_SAMPLES,
            )
            return None

        # Normalise incoming candles to dicts
        if candles and hasattr(candles[0], "unix_time"):
            candle_dicts = [
                {"t": c.unix_time, "o": c.open, "h": c.high,
                 "l": c.low,       "c": c.close, "v": c.volume}
                for c in candles
            ]
        else:
            candle_dicts = candles

        query_feat = extract_features(
            candle_dicts,
            pump_ratio_1m=chart_ctx.pump_ratio if chart_ctx else None,
            vol_trend_1m=chart_ctx.vol_trend if chart_ctx else None,
        )
        if query_feat is None:
            return None

        now = datetime.now(timezone.utc)
        training_feats, training_pnl, recency_weights = [], [], []

        for snap in snapshots:
            snap_candles = json.loads(snap["candles_json"])
            # Pass stored 1-minute features so training vectors match query shape
            feat = extract_features(
                snap_candles,
                pump_ratio_1m=snap.get("pump_ratio"),
                vol_trend_1m=snap.get("vol_trend"),
            )
            if feat is None:
                continue

            snap_ts = datetime.fromisoformat(snap["ts"])
            if snap_ts.tzinfo is None:
                snap_ts = snap_ts.replace(tzinfo=timezone.utc)
            age_days = (now - snap_ts).total_seconds() / 86400.0

            training_feats.append(feat)
            training_pnl.append(snap["outcome_pnl_pct"])
            recency_weights.append(math.exp(-age_days / self._halflife))

        if len(training_feats) < MIN_SAMPLES:
            return None

        # Normalise all feature vectors
        norm_query, norm_training = zscore_normalize(query_feat, training_feats)

        # Compute combined weights and sort by distance
        candidates = []
        for i, feat in enumerate(norm_training):
            dist = euclidean(norm_query, feat)
            sim_w = 1.0 / (dist + 1e-6)
            w = sim_w * recency_weights[i]
            candidates.append((dist, w, training_pnl[i]))

        candidates.sort(key=lambda x: x[0])
        neighbours = candidates[: self._k]

        # Weighted average PnL across neighbours
        total_w     = sum(w   for _, w, _   in neighbours)
        weighted_pnl = sum(w * p for _, w, p in neighbours)

        if total_w == 0:
            return 5.0

        avg_pnl = weighted_pnl / total_w

        # Map to [0, 10]
        pnl_range = _SCORE_HIGH_PCT - _SCORE_LOW_PCT
        raw_score = (avg_pnl - _SCORE_LOW_PCT) / pnl_range * 10.0
        final_score = max(0.0, min(10.0, raw_score))

        logger.info(
            "[ML] score=%.1f | avg_neighbour_pnl=%.1f%% | k=%d/%d snapshots",
            final_score, avg_pnl, len(neighbours), len(snapshots),
        )
        return final_score
