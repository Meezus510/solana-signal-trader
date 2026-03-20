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

1. Extract an 18-dimensional feature vector from the incoming candles.
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
       score = clamp((weighted_pnl_pct − (−35)) / 120 × 10, 0, 10)
   i.e. −35 % PnL → 0 | +25 % PnL → 5 | +85 % PnL → 10.
   PnL% is price-based (weighted avg exit / entry − 1) × 100, so it is
   independent of position size across different capital-risk regimes.

Returns None when fewer than MIN_SAMPLES closed snapshots exist.

Candle resolution — dual
------------------------
Two candle sources are combined in a single feature vector:
  • Moralis 10s  — 100 bars ≈ 17 min. Captures fine-grained pump shape.
                   Used as features 1–6. Falls back to neutral zeros when
                   Moralis is unavailable (API failure, new token with no
                   pair address yet, etc.).
  • Birdeye 1m   — 40 bars = 40 min. Broader momentum context; always
                   available. Used as features 7–12.

Features (one vector per chart window)
---------------------------------------
OHLCV features from Moralis 10s candles (neutral fallback when unavailable):
1. pump_ratio_10s     — close[-1] / min(low) over the 10s window
2. vol_momentum_10s   — avg volume last 5 bars / avg volume earlier bars
3. price_slope_10s    — (close[-1] / open[0] − 1) × 100
4. recent_momentum_10s— (close[-1] / close[-6] − 1) × 100 (last 5 bars)
5. volatility_10s     — std-dev of bar-to-bar % returns
6. candle_count_10s   — candle_count / 20  (proxy for token age / Moralis coverage)

OHLCV features from Birdeye 1m candles (same 6 computations, different timescale):
7. pump_ratio_1m
8. vol_momentum_1m
9. price_slope_1m
10. recent_momentum_1m
11. volatility_1m
12. candle_count_1m

Pair stats features (from Moralis live pair stats — fallback to neutral when unavailable):
13. buy_ratio_5m          — buys / (buys + sells) in last 5 min  [0–1, fallback 0.5]
14. activity_5m_norm      — (buys + sells) / 20, capped at 3.0   [0–3, fallback 0.0]
15. price_change_5m_norm  — price % change last 5 min / 50, clamped [-1, 1]  [fallback 0.0]
16. buy_vol_ratio_1h      — buy USD vol / total USD vol last hour  [0–1, fallback 0.5]
17. liquidity_change_1h   — liquidity % change last hour / 20, clamped [-1, 1]  [fallback 0.0]

Source:
18. source_channel        — WizzyTrades=1.0, WizzyCasino=2.0, other=0.0

Token metadata features (from Birdeye token overview — fallback 0.5 when unavailable):
Log-scale normalisation used because market cap and liquidity span multiple orders of magnitude.
19. market_cap_norm       — log10(market_cap_usd) / 8, clamped [0, 1]  [fallback 0.5]
                            $10K→0.50, $100K→0.63, $1M→0.75, $10M→0.88, $100M→1.0
20. liquidity_usd_norm    — log10(liquidity_usd) / 7, clamped [0, 1]   [fallback 0.5]
                            $1K→0.43, $10K→0.57, $100K→0.71, $1M→0.86, $10M→1.0
21. holder_count_norm     — log10(holder_count) / 5, clamped [0, 1]    [fallback 0.5]
                            10→0.20, 100→0.40, 1K→0.60, 10K→0.80, 100K→1.0
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

# Candle resolution used for ML snapshots via Birdeye (1m is finest available).
# 40 bars × 1m = 40-minute window.
ML_OHLCV_BARS: int = 40
ML_OHLCV_INTERVAL: str = "1m"

# Higher-resolution ML candles via Moralis (supports sub-minute intervals).
# 100 bars × 10s ≈ 17-minute window at 6× the detail of 1m bars.
# Used when MORALIS_API_KEY is set — captures pump shape with much finer granularity.
MORALIS_OHLCV_BARS: int = 100
MORALIS_OHLCV_INTERVAL: str = "10s"

# Score mapping: PnL% range covered by [0, 10].
# Uses price-based return (weighted avg exit price / entry price - 1) × 100,
# which is position-size-independent.
# Range chosen from observed quick_pop data: losses -20% to -35%, wins +35% to +85%.
_SCORE_LOW_PCT:  float = -35.0   # maps to score 0
_SCORE_HIGH_PCT: float =  85.0   # maps to score 10

# Encoding for source channel → numeric feature (0.0 = unknown/other)
_CHANNEL_ENC: dict[str, float] = {
    "WizzyTrades": 1.0,
    "WizzyCasino": 2.0,
}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _compute_ohlcv_features(candles: list[dict]) -> list[float]:
    """
    Compute 6 OHLCV-derived features from a candle list.

    Returns neutral values [1.0, 1.0, 0.0, 0.0, 0.0, 0.0] when fewer than
    3 candles are available, so the caller can still produce a valid vector.
    """
    if not candles or len(candles) < 3:
        return [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]

    opens   = [c["o"] for c in candles]
    closes  = [c["c"] for c in candles]
    lows    = [c["l"] for c in candles]
    volumes = [c["v"] for c in candles]

    recent_low = min(lows)
    pump_ratio = closes[-1] / recent_low if recent_low > 0 else 1.0

    vol_window   = min(5, len(volumes) - 1)
    recent_vols  = volumes[-vol_window:]
    earlier_vols = volumes[:-vol_window] if len(volumes) > vol_window else volumes
    avg_recent   = sum(recent_vols)  / len(recent_vols)
    avg_earlier  = sum(earlier_vols) / len(earlier_vols)
    vol_momentum = avg_recent / avg_earlier if avg_earlier > 0 else 1.0

    price_slope = (closes[-1] / opens[0] - 1.0) * 100.0 if opens[0] > 0 else 0.0

    lookback        = min(5, len(closes) - 1)
    base            = closes[-1 - lookback]
    recent_momentum = (closes[-1] / base - 1.0) * 100.0 if base > 0 else 0.0

    returns = []
    for i in range(1, len(closes)):
        if closes[i - 1] > 0:
            returns.append((closes[i] / closes[i - 1] - 1.0) * 100.0)
    if len(returns) >= 2:
        mean_ret   = sum(returns) / len(returns)
        volatility = math.sqrt(sum((r - mean_ret) ** 2 for r in returns) / len(returns))
    else:
        volatility = 0.0

    candle_count_norm = len(candles) / 20.0

    return [pump_ratio, vol_momentum, price_slope, recent_momentum, volatility, candle_count_norm]


def extract_features(
    candles_10s: list[dict],
    candles_1m: Optional[list[dict]] = None,
    pair_stats: Optional[dict] = None,
    source_channel: Optional[str] = None,
) -> Optional[list[float]]:
    """
    Extract an 18-element feature vector from dual-resolution candle data.

    Features 1-6:  OHLCV from 10s Moralis candles (neutral when unavailable).
    Features 7-12: OHLCV from 1m Birdeye candles  (neutral when unavailable).
    Features 13-17: Moralis live pair stats (neutral when unavailable).
    Feature 18: source channel encoded as float.

    Returns None only when both candles_10s and candles_1m have fewer than 3
    candles (nothing meaningful to score against).

    pair_stats dict keys (all optional, each falls back independently):
        buys_5m, sells_5m, buy_volume_1h, total_volume_1h,
        price_change_5m_pct, liquidity_change_1h_pct
    """
    has_10s = bool(candles_10s) and len(candles_10s) >= 3
    has_1m  = bool(candles_1m)  and len(candles_1m)  >= 3
    if not has_10s and not has_1m:
        return None

    feats_10s = _compute_ohlcv_features(candles_10s or [])
    feats_1m  = _compute_ohlcv_features(candles_1m  or [])

    # ------------------------------------------------------------------
    # Features 13-17: Moralis live pair stats
    # All fall back to neutral values when pair_stats is None.
    # ------------------------------------------------------------------
    ps = pair_stats or {}

    buys_5m  = ps.get("buys_5m",  0) or 0
    sells_5m = ps.get("sells_5m", 0) or 0
    f_buy_ratio_5m     = buys_5m / (buys_5m + sells_5m + 1e-9) if (buys_5m + sells_5m) > 0 else 0.5
    f_activity_5m      = min((buys_5m + sells_5m) / 20.0, 3.0)

    price_5m           = ps.get("price_change_5m_pct", 0.0) or 0.0
    f_price_change_5m  = max(-1.0, min(1.0, price_5m / 50.0))

    buy_vol_1h         = ps.get("buy_volume_1h",   0.0) or 0.0
    total_vol_1h       = ps.get("total_volume_1h", 0.0) or 0.0
    f_buy_vol_ratio_1h = buy_vol_1h / (total_vol_1h + 1e-9) if total_vol_1h > 0 else 0.5

    liq_1h                 = ps.get("liquidity_change_1h_pct", 0.0) or 0.0
    f_liquidity_change_1h  = max(-1.0, min(1.0, liq_1h / 20.0))

    # Feature 18: source channel
    f_channel = _CHANNEL_ENC.get(source_channel or "", 0.0)

    # ------------------------------------------------------------------
    # Features 19-21: Birdeye token metadata (log-scale normalised).
    # Falls back to 0.5 (neutral midpoint) when not available so all
    # historical rows without metadata still produce a valid vector.
    # ------------------------------------------------------------------
    mc  = ps.get("market_cap_usd")
    liq = ps.get("liquidity_usd")
    hld = ps.get("holder_count")

    f_market_cap_norm  = max(0.0, min(1.0, math.log10(max(mc,  1.0)) / 8.0)) if mc  else 0.5
    f_liquidity_norm   = max(0.0, min(1.0, math.log10(max(liq, 1.0)) / 7.0)) if liq else 0.5
    f_holder_norm      = max(0.0, min(1.0, math.log10(max(hld, 1.0)) / 5.0)) if hld else 0.5

    return (
        feats_10s
        + feats_1m
        + [
            f_buy_ratio_5m, f_activity_5m, f_price_change_5m,
            f_buy_vol_ratio_1h, f_liquidity_change_1h,
            f_channel,
            f_market_cap_norm, f_liquidity_norm, f_holder_norm,
        ]
    )


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
        score_low_pct: float = _SCORE_LOW_PCT,
        score_high_pct: float = _SCORE_HIGH_PCT,
        training_label: str = "outcome_pnl_pct",
        feature_weights: Optional[list[float]] = None,
    ) -> None:
        self._db = db
        self._strategy = strategy
        self._k = k
        self._halflife = recency_halflife_days
        self._score_low_pct = score_low_pct
        self._score_high_pct = score_high_pct
        self._training_label = training_label
        # Per-feature weights (18-element). Applied before z-normalisation so
        # high-separability features have proportionally more distance influence.
        self._feature_weights: list[float] = list(feature_weights) if feature_weights else []

    def score(
        self,
        candles_10s: list,
        candles_1m: Optional[list] = None,
        chart_ctx=None,       # retained for call-site compatibility; no longer used
        pair_stats=None,
        source_channel: str = "",
    ) -> Optional[float]:
        """
        Score incoming candles against historical closed snapshots.

        candles_10s — Moralis 10s candles (list of OHLCVCandle or dicts {t,o,h,l,c,v}).
                      Features 1-6. Pass [] / None when Moralis was unavailable;
                      those features fall back to neutral.
        candles_1m  — Birdeye 1m candles (same format). Features 7-12.
                      Pass [] / None when unavailable.
        pair_stats  — dict from MoralisOHLCVClient.get_pair_stats() (optional).
        chart_ctx   — no longer used; kept so existing call sites don't break.

        Returns a float in [0.0, 10.0] or None if fewer than MIN_SAMPLES
        closed snapshots exist or if all candle inputs are too short.
        """
        snapshots = self._db.load_chart_snapshots(self._strategy, label_column=self._training_label)
        if len(snapshots) < MIN_SAMPLES:
            logger.debug(
                "[ML] Not enough training data (%d/%d closed snapshots) — skipping score",
                len(snapshots), MIN_SAMPLES,
            )
            return None

        def _to_dicts(candles: list) -> list[dict]:
            if candles and hasattr(candles[0], "unix_time"):
                return [
                    {"t": c.unix_time, "o": c.open, "h": c.high,
                     "l": c.low,       "c": c.close, "v": c.volume}
                    for c in candles
                ]
            return candles or []

        query_feat = extract_features(
            _to_dicts(candles_10s),
            candles_1m=_to_dicts(candles_1m) if candles_1m else None,
            pair_stats=pair_stats,
            source_channel=source_channel,
        )
        if query_feat is None:
            return None

        now = datetime.now(timezone.utc)
        training_feats, training_pnl, recency_weights = [], [], []

        for snap in snapshots:
            snap_candles_10s = json.loads(snap["candles_json"])
            snap_candles_1m_raw = snap.get("candles_1m_json")
            snap_candles_1m = json.loads(snap_candles_1m_raw) if snap_candles_1m_raw else None
            snap_pair_stats = json.loads(snap["pair_stats_json"]) \
                if snap.get("pair_stats_json") else None
            feat = extract_features(
                snap_candles_10s,
                candles_1m=snap_candles_1m,
                pair_stats=snap_pair_stats,
                source_channel=snap.get("source_channel", ""),
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

        # Apply per-feature weights before normalisation (boosts high-separability
        # features' influence on euclidean distance without changing the scale).
        if self._feature_weights:
            n_w = len(self._feature_weights)
            query_feat    = [query_feat[i] * (self._feature_weights[i] if i < n_w else 1.0)
                             for i in range(len(query_feat))]
            training_feats = [[f[i] * (self._feature_weights[i] if i < n_w else 1.0)
                               for i in range(len(f))]
                              for f in training_feats]

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

        # Map to [0, 10] using per-instance score range (defaults to global constants)
        pnl_range = self._score_high_pct - self._score_low_pct
        raw_score = (avg_pnl - self._score_low_pct) / pnl_range * 10.0
        final_score = max(0.0, min(10.0, raw_score))

        logger.info(
            "[ML] score=%.1f | avg_neighbour_pnl=%.1f%% | k=%d/%d snapshots",
            final_score, avg_pnl, len(neighbours), len(snapshots),
        )
        return final_score
