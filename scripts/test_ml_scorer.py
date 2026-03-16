"""
scripts/test_ml_scorer.py — Offline leave-one-out evaluation of the KNN ML scorer.

Uses the known March 15 quick_pop trades as a labelled dataset.
For each trade it:
  1. Fetches 40×15s + 20×1m candles at entry time from Birdeye (cached to
     scripts/candle_cache.json so re-runs are instant and free).
  2. Builds a training set from all OTHER trades in the dataset.
  3. Runs the KNN scorer to get a 0–10 confidence score.
  4. Compares the prediction to the actual outcome.

Usage:
    source venv/bin/activate
    python scripts/test_ml_scorer.py           # full LOO evaluation
    python scripts/test_ml_scorer.py --fetch   # force-refresh candle cache
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone

import aiohttp
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trader.analysis.chart import OHLCV_BARS, compute_chart_context
from trader.analysis.ml_scorer import (
    K, MIN_SAMPLES, extract_features, zscore_normalize, euclidean,
    ML_OHLCV_BARS, ML_OHLCV_INTERVAL,
    _SCORE_LOW_PCT, _SCORE_HIGH_PCT,
)
from trader.config import Config
from trader.pricing.birdeye import BirdeyePriceClient

CACHE_PATH = os.path.join(os.path.dirname(__file__), "candle_cache.json")
SEP = "-" * 80

# ---------------------------------------------------------------------------
# Known trades from 2026-03-15 quick_pop session
# ---------------------------------------------------------------------------
# entry_ts  — ISO UTC timestamp of the BUY event
# pnl_pct   — realized_pnl_usd / usd_size * 100  (all positions were $30)
# outcome   — human label

TRADES = [
    {
        "symbol":   "NETAINYAHU",
        "mint":     "BBkUQdTDdVySDKXT4TngbmbpC2ktwBeymMSdn41ppump",
        # BUY not in log — estimated ~3 min before TP1 at 18:13:34
        "entry_ts": "2026-03-15T18:10:00+00:00",
        "entry_ts_note": "(estimated — BUY not in log)",
        "pnl_pct":  11.3036 / 30 * 100,   # +37.7%
        "outcome":  "WIN  (TP1)",
    },
    {
        "symbol":   "MVM",
        "mint":     "9yKLqa49XBvfdGtvVXa3GDYPbQvLZCMKYneof6xmpump",
        "entry_ts": "2026-03-15T18:27:57+00:00",
        "pnl_pct":  -6.3539 / 30 * 100,   # -21.2%
        "outcome":  "LOSS (STOP_LOSS)",
    },
    {
        "symbol":   "ELONGATE",
        "mint":     "3aGbWBEpCfDxMSb5KfpSvWrBo3tCCbMiZP9x8PpKpump",
        "entry_ts": "2026-03-15T19:06:04+00:00",
        "pnl_pct":  -6.5404 / 30 * 100,   # -21.8%
        "outcome":  "LOSS (STOP_LOSS)",
    },
    {
        "symbol":   "BAGWORKOOR",
        "mint":     "FDQ77aHDgV6ozbv1b4WM5oXuHGV1cMnjSXpxvgSzpump",
        "entry_ts": "2026-03-15T19:10:08+00:00",
        "pnl_pct":  (9.2196 + 2.8354) / 30 * 100,  # +40.2%
        "outcome":  "WIN  (TP1 + TRAILING_STOP)",
    },
    {
        "symbol":   "BTC",
        "mint":     "KWmej3HSuuLgaoWWdELniXGhA3gKzLxkf7FRj7xpump",
        "entry_ts": "2026-03-15T19:31:18+00:00",
        "pnl_pct":  -6.3811 / 30 * 100,   # -21.3%
        "outcome":  "LOSS (STOP_LOSS)",
    },
    {
        "symbol":   "PEEP",
        "mint":     "8HeSKdX9XkJB9PBZiXhFuTYaWbfn3u6sftyPbAcxpump",
        "entry_ts": "2026-03-15T19:39:29+00:00",
        "pnl_pct":  -8.4896 / 30 * 100,   # -28.3%
        "outcome":  "LOSS (STOP_LOSS)",
    },
    {
        "symbol":   "MEFAI",
        "mint":     "7gcoey4EXJcZ8u3iGYhgTBrh3JuhLWzV4Gs1zNaPtu3U",
        "entry_ts": "2026-03-15T20:18:17+00:00",
        "pnl_pct":  4.5249 / 30 * 100,    # +15.1%
        "outcome":  "WIN  (TIMEOUT_SLOW)",
    },
    {
        "symbol":   "PIXEL",
        "mint":     "C2hH5X3GGSo4UtFV4evV2r56aMBy2m3KjCeW3wNipump",
        "entry_ts": "2026-03-15T20:26:16+00:00",
        "pnl_pct":  -6.0356 / 30 * 100,   # -20.1%
        "outcome":  "LOSS (STOP_LOSS)",
    },
    {
        "symbol":   "SILKROAD",
        "mint":     "25PaVmYrnFBSJikKU1kKNS4Zjo4NyiMXr9DnhNT9pump",
        "entry_ts": "2026-03-15T20:29:17+00:00",
        "pnl_pct":  -7.2428 / 30 * 100,   # -24.1%
        "outcome":  "LOSS (STOP_LOSS)",
    },
]


# ---------------------------------------------------------------------------
# Candle fetching + caching
# ---------------------------------------------------------------------------

def load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict) -> None:
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


async def fetch_all_candles(
    trades: list[dict],
    birdeye: BirdeyePriceClient,
    cache: dict,
    force_refresh: bool,
    delay: float = 1.0,
) -> None:
    """Fetch ML candles (40 × 1m) + chart-filter candles (20 × 1m) for every trade."""
    for trade in trades:
        key_ml = f"{trade['mint']}|ml"
        key_1m = f"{trade['mint']}|1m"
        entry_unix = int(datetime.fromisoformat(trade["entry_ts"]).timestamp())

        if not force_refresh and key_ml in cache and key_1m in cache:
            print(f"  [CACHE] {trade['symbol']}")
            continue

        print(f"  [FETCH] {trade['symbol']} @ {trade['entry_ts'][:19]} UTC ...")

        # ML candles: 40 × 1m = 40-minute window
        ml_candles = await birdeye.get_ohlcv(
            trade["mint"],
            bars=ML_OHLCV_BARS,
            interval=ML_OHLCV_INTERVAL,
            time_to=entry_unix,
        )
        # Chart-filter candles: 20 × 1m (shorter window)
        chart_candles = await birdeye.get_ohlcv(
            trade["mint"],
            bars=OHLCV_BARS,
            interval="1m",
            time_to=entry_unix,
        )

        cache[key_ml] = [
            {"t": c.unix_time, "o": c.open, "h": c.high,
             "l": c.low,       "c": c.close, "v": c.volume}
            for c in ml_candles
        ]
        cache[key_1m] = [
            {"t": c.unix_time, "o": c.open, "h": c.high,
             "l": c.low,       "c": c.close, "v": c.volume}
            for c in chart_candles
        ]

        bars_ml = len(ml_candles)
        bars_1m = len(chart_candles)
        if bars_ml == 0:
            print(f"    [WARN] no ML candles — token may be delisted or too new")
        else:
            print(f"    OK — {bars_ml} × {ML_OHLCV_INTERVAL} ML bars, {bars_1m} × 1m chart bars")

        await asyncio.sleep(delay)

    save_cache(cache)


# ---------------------------------------------------------------------------
# Leave-one-out evaluation
# ---------------------------------------------------------------------------

def knn_score_raw(
    query_feat: list[float],
    training_feats: list[list[float]],
    training_pnl: list[float],
    k: int = K,
) -> float | None:
    """Pure KNN score (no recency weighting — all data from the same day)."""
    if len(training_feats) < MIN_SAMPLES:
        return None

    norm_query, norm_training = zscore_normalize(query_feat, training_feats)

    candidates = []
    for i, feat in enumerate(norm_training):
        dist = euclidean(norm_query, feat)
        w = 1.0 / (dist + 1e-6)
        candidates.append((dist, w, training_pnl[i]))

    candidates.sort(key=lambda x: x[0])
    neighbours = candidates[:k]

    total_w      = sum(w   for _, w, _   in neighbours)
    weighted_pnl = sum(w * p for _, w, p in neighbours)
    if total_w == 0:
        return 5.0

    avg_pnl   = weighted_pnl / total_w
    pnl_range = _SCORE_HIGH_PCT - _SCORE_LOW_PCT
    raw_score = (avg_pnl - _SCORE_LOW_PCT) / pnl_range * 10.0
    return max(0.0, min(10.0, raw_score))


def run_loo_evaluation(trades: list[dict], cache: dict) -> None:
    """
    Leave-one-out cross-validation.
    For each trade: train on all others, score the held-out trade, compare to truth.
    Falls back to 1m candles when 15s data has expired (Birdeye keeps 15s for ~4h only).
    """
    # 1. Extract features for every trade
    records = []
    for trade in trades:
        key_ml = f"{trade['mint']}|ml"
        key_1m = f"{trade['mint']}|1m"

        ml_candles    = cache.get(key_ml, [])
        chart_candles = cache.get(key_1m, [])

        if not ml_candles:
            print(f"  [SKIP] {trade['symbol']} — no candle data in cache")
            continue

        # Derive 1-minute chart context (pump_ratio, vol_trend)
        pump_ratio_1m = None
        vol_trend_1m  = None
        if chart_candles:
            # Reconstruct OHLCVCandle objects for compute_chart_context
            from trader.analysis.chart import OHLCVCandle
            candle_objs = [
                OHLCVCandle(
                    unix_time=c["t"], open=c["o"], high=c["h"],
                    low=c["l"], close=c["c"], volume=c["v"],
                )
                for c in chart_candles
            ]
            # Use last close as a proxy entry price
            entry_price = candle_objs[-1].close if candle_objs else 0.0
            ctx = compute_chart_context(candle_objs, entry_price)
            if ctx:
                pump_ratio_1m = ctx.pump_ratio
                vol_trend_1m  = ctx.vol_trend

        feat = extract_features(
            ml_candles,
            pump_ratio_1m=pump_ratio_1m,
            vol_trend_1m=vol_trend_1m,
        )
        if feat is None:
            print(f"  [SKIP] {trade['symbol']} — feature extraction failed (< 3 candles)")
            continue

        records.append({
            "symbol":   trade["symbol"],
            "outcome":  trade["outcome"],
            "pnl_pct":  trade["pnl_pct"],
            "feat":     feat,
            "pump_1m":  pump_ratio_1m,
            "vol_1m":   vol_trend_1m,
            "note":     trade.get("entry_ts_note", ""),
        })

    print()
    print(SEP)
    print("  LEAVE-ONE-OUT EVALUATION  (training K={}, min_samples={})".format(K, MIN_SAMPLES))
    print(SEP)
    print(f"  {'Symbol':<12} {'Actual PnL%':>11}  {'Score':>5}  {'Label':<28}  {'Prediction'}")
    print(f"  {'-'*12} {'-'*11}  {'-'*5}  {'-'*28}  {'-'*14}")

    correct = 0
    scored  = 0
    threshold = 5.0  # score ≥ 5 → predicted win, < 5 → predicted skip

    for i, rec in enumerate(records):
        train_feats = [r["feat"]    for j, r in enumerate(records) if j != i]
        train_pnl   = [r["pnl_pct"] for j, r in enumerate(records) if j != i]

        score = knn_score_raw(rec["feat"], train_feats, train_pnl)
        if score is None:
            tag = "NOT ENOUGH DATA"
            pred_label = "—"
        else:
            scored += 1
            is_win      = rec["pnl_pct"] > 0
            pred_win    = score >= threshold
            correct    += 1 if is_win == pred_win else 0
            pred_label  = "BUY" if pred_win else "SKIP"
            ok = "✓" if is_win == pred_win else "✗"
            tag = f"{ok}  ({pred_label})"

        score_str = f"{score:.1f}" if score is not None else "—"
        note = f"  {rec['note']}" if rec["note"] else ""
        print(
            f"  {rec['symbol']:<12} {rec['pnl_pct']:>+10.1f}%  {score_str:>5}  "
            f"{rec['outcome']:<28}  {tag}{note}"
        )

    print(SEP)
    if scored:
        accuracy = correct / scored * 100
        print(f"  Accuracy: {correct}/{scored}  ({accuracy:.0f}%)")
        print(f"  Threshold: score >= {threshold:.0f} → BUY, else SKIP")
    print(SEP)
    print()
    print("  NOTE: Leave-one-out with 9 trades is a rough signal only.")
    print("  The real scorer will improve significantly once the backfill")
    print("  seeds 50+ historical snapshots from the full trade history.")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def run(args: argparse.Namespace, cfg: Config) -> None:
    cache = load_cache()

    print(SEP)
    print("  FETCHING / LOADING CANDLES")
    print(SEP)
    async with aiohttp.ClientSession() as session:
        birdeye = BirdeyePriceClient(cfg, session)
        await fetch_all_candles(TRADES, birdeye, cache, args.fetch, delay=args.delay)

    run_loo_evaluation(TRADES, cache)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leave-one-out evaluation of the KNN ML scorer on March 15 trades."
    )
    parser.add_argument(
        "--fetch", action="store_true",
        help="Force-refresh all candles (ignore cache)",
    )
    parser.add_argument(
        "--delay", type=float, default=1.0, metavar="SECS",
        help="Delay between Birdeye calls when fetching (default: 1.0)",
    )
    args = parser.parse_args()

    try:
        cfg = Config.load()
    except ValueError as exc:
        print(f"[ERROR] Config: {exc}", file=sys.stderr)
        sys.exit(1)

    asyncio.run(run(args, cfg))


if __name__ == "__main__":
    main()
