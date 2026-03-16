"""
scripts/test_ml_scorer.py — Offline leave-one-out evaluation of the KNN ML scorer.

Reads completed quick_pop positions from a trades.log file, fetches OHLCV
candles at each entry time from Birdeye (cached to scripts/candle_cache.json
so re-runs are instant), and runs leave-one-out cross-validation.

Usage:
    source venv/bin/activate
    python scripts/test_ml_scorer.py --log trades.log
    python scripts/test_ml_scorer.py --log trades.log --fetch       # force-refresh cache
    python scripts/test_ml_scorer.py --log trades.log --limit 20    # first 20 trades only
    python scripts/test_ml_scorer.py --log trades.log --moralis     # 10s candles via Moralis
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone

import aiohttp
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trader.analysis.chart import OHLCV_BARS, OHLCVCandle, compute_chart_context
from trader.analysis.ml_scorer import (
    K, MIN_SAMPLES, extract_features, zscore_normalize, euclidean,
    ML_OHLCV_BARS, ML_OHLCV_INTERVAL,
    MORALIS_OHLCV_BARS, MORALIS_OHLCV_INTERVAL,
    _SCORE_LOW_PCT, _SCORE_HIGH_PCT,
)
from trader.config import Config
from trader.pricing.birdeye import BirdeyePriceClient
from trader.pricing.moralis import MoralisOHLCVClient

CACHE_PATH = os.path.join(os.path.dirname(__file__), "candle_cache.json")
SEP = "-" * 84

CLOSE_EVENTS = {"TP1", "TP2", "TP3", "TP4", "STOP_LOSS", "TRAILING_STOP", "TIMEOUT_SLOW"}

# ---------------------------------------------------------------------------
# Log parser
# ---------------------------------------------------------------------------

_LOG_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"   # timestamp
    r"\s*\|\s*(\S+)"                              # event
    r"\s*\|\s*(\S+)"                              # symbol
    r"\s*\|\s*(\S+)"                              # mint
    r".*?price=\$([0-9.]+)"                       # price
    r".*?qty=([0-9.]+)"                            # qty
    r".*?pnl=\$([+\-]?[0-9.]+)"                   # pnl (sign optional on BUY lines)
    r"(?:.*?strategy=(\S+))?"                      # strategy (optional)
)


def parse_log(path: str, strategy: str = "quick_pop") -> list[dict]:
    """
    Parse a trades.log file and return completed positions as a list of dicts.

    Handles both log formats:
      - New (server): has  | strategy=quick_pop  at the end
      - Old (local):  no strategy field — all lines are included

    Positions where only a BUY exists (still open or close not in log) are skipped.
    Multi-close positions (TP1 + TRAILING_STOP, TP1 + TP2, etc.) have pnl summed.

    pnl_pct is computed as a price-based return:
        (qty-weighted avg exit price / entry price - 1) × 100
    This is independent of position size, making it comparable across $10 and $30
    position regimes.
    """
    buys: dict[str, dict]                        = {}   # mint -> buy data
    closes: dict[str, list[tuple[float, float]]] = defaultdict(list)  # mint -> [(price, qty)]

    with open(path) as f:
        for line in f:
            m = _LOG_RE.search(line)
            if not m:
                continue
            ts, event, symbol, mint, price, qty, pnl, strat = m.groups()
            # If log has strategy field, filter by it; otherwise include all lines
            if strat is not None and strat != strategy:
                continue

            if event == "BUY":
                if mint not in buys:   # first BUY wins (no re-entry on same mint)
                    buys[mint] = {
                        "symbol":   symbol,
                        "mint":     mint,
                        "entry_ts": ts.replace(" ", "T") + "+00:00",
                        "entry_price": float(price),
                        "initial_qty": float(qty),
                    }
            elif event in CLOSE_EVENTS:
                closes[mint].append((float(price), float(qty)))

    result = []
    for mint, buy in buys.items():
        if mint not in closes:
            continue   # still open or close not in this log file

        close_events = closes[mint]
        total_qty = sum(q for _, q in close_events if q > 0)
        if total_qty > 0:
            weighted_exit = sum(p * q for p, q in close_events if q > 0) / total_qty
            pnl_pct = (weighted_exit / buy["entry_price"] - 1.0) * 100.0
        else:
            pnl_pct = 0.0

        usd_size  = buy["entry_price"] * buy["initial_qty"]

        result.append({
            "symbol":    buy["symbol"],
            "mint":      mint,
            "entry_ts":  buy["entry_ts"],
            "pnl_pct":   pnl_pct,
            "usd_size":  usd_size,
            "outcome":   "WIN" if pnl_pct > 0 else "LOSS",
        })

    return sorted(result, key=lambda x: x["entry_ts"])


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
    moralis: "MoralisOHLCVClient | None" = None,
    delay: float = 1.0,
) -> None:
    """Fetch ML candles + 20×1m chart candles for every trade.

    ML candles source:
      - Moralis (10s × 100 bars) when --moralis is set, cache key {mint}|ml_10s
      - Birdeye  (1m  × 40  bars) otherwise,            cache key {mint}|ml
    Chart (1m) candles always come from Birdeye.
    """
    ml_cache_key = lambda mint: f"{mint}|ml_10s" if moralis else f"{mint}|ml"

    needs_fetch = [
        t for t in trades
        if force_refresh
        or ml_cache_key(t["mint"]) not in cache
        or f"{t['mint']}|1m" not in cache
    ]

    if not needs_fetch:
        print(f"  All {len(trades)} trade(s) loaded from cache.")
        return

    source = f"Moralis {MORALIS_OHLCV_BARS}×{MORALIS_OHLCV_INTERVAL}" if moralis \
             else f"Birdeye {ML_OHLCV_BARS}×{ML_OHLCV_INTERVAL}"
    print(f"  Fetching {len(needs_fetch)} trade(s) [{source}] "
          f"(cached: {len(trades)-len(needs_fetch)}) ...")

    for i, trade in enumerate(needs_fetch, 1):
        entry_unix = int(datetime.fromisoformat(trade["entry_ts"]).timestamp())
        symbol     = trade["symbol"]

        if moralis:
            ml_candles = await moralis.get_ohlcv(
                trade["mint"], bars=MORALIS_OHLCV_BARS, interval=MORALIS_OHLCV_INTERVAL,
                time_to=entry_unix,
            )
        else:
            ml_candles = await birdeye.get_ohlcv(
                trade["mint"], bars=ML_OHLCV_BARS, interval=ML_OHLCV_INTERVAL,
                time_to=entry_unix,
            )

        chart_candles = await birdeye.get_ohlcv(
            trade["mint"], bars=OHLCV_BARS, interval="1m",
            time_to=entry_unix,
        )

        cache[ml_cache_key(trade["mint"])] = [
            {"t": c.unix_time, "o": c.open, "h": c.high,
             "l": c.low,       "c": c.close, "v": c.volume}
            for c in ml_candles
        ]
        cache[f"{trade['mint']}|1m"] = [
            {"t": c.unix_time, "o": c.open, "h": c.high,
             "l": c.low,       "c": c.close, "v": c.volume}
            for c in chart_candles
        ]

        bars_ml = len(ml_candles)
        status  = f"{bars_ml} bars" if bars_ml else "NO DATA"
        print(f"  [{i:>3}/{len(needs_fetch)}] {symbol:<14} {trade['entry_ts'][:16]}  {status}")

        await asyncio.sleep(delay)

    save_cache(cache)
    print()


# ---------------------------------------------------------------------------
# Leave-one-out evaluation
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


def run_loo_evaluation(
    trades: list[dict],
    cache: dict,
    threshold: float = 5.0,
    use_moralis: bool = False,
) -> None:
    ml_key = lambda mint: f"{mint}|ml_10s" if use_moralis else f"{mint}|ml"

    # Build feature records
    records = []
    skipped = 0
    for trade in trades:
        ml_candles    = cache.get(ml_key(trade["mint"]), [])
        chart_candles = cache.get(f"{trade['mint']}|1m", [])

        if not ml_candles:
            skipped += 1
            continue

        pump_ratio_1m = vol_trend_1m = None
        if chart_candles:
            objs = [OHLCVCandle(unix_time=c["t"], open=c["o"], high=c["h"],
                                low=c["l"], close=c["c"], volume=c["v"])
                    for c in chart_candles]
            ctx = compute_chart_context(objs, objs[-1].close if objs else 0.0)
            if ctx:
                pump_ratio_1m = ctx.pump_ratio
                vol_trend_1m  = ctx.vol_trend

        feat = extract_features(ml_candles, pump_ratio_1m=pump_ratio_1m,
                                 vol_trend_1m=vol_trend_1m)
        if feat is None:
            skipped += 1
            continue

        records.append({**trade, "feat": feat})

    print(SEP)
    print(f"  LEAVE-ONE-OUT EVALUATION  (K={K}, min_samples={MIN_SAMPLES}, threshold={threshold:.0f})")
    print(f"  Trades: {len(records)} scorable, {skipped} skipped (no candle data)")
    print(SEP)
    print(f"  {'Symbol':<14} {'PnL%':>8}  {'Score':>5}  {'Outcome':<10}  Prediction")
    print(f"  {'-'*14} {'-'*8}  {'-'*5}  {'-'*10}  {'-'*14}")

    correct = scored = true_pos = false_pos = true_neg = false_neg = 0

    for i, rec in enumerate(records):
        train_feats = [r["feat"]    for j, r in enumerate(records) if j != i]
        train_pnl   = [r["pnl_pct"] for j, r in enumerate(records) if j != i]

        score = knn_score(rec["feat"], train_feats, train_pnl)

        if score is None:
            print(f"  {rec['symbol']:<14} {rec['pnl_pct']:>+7.1f}%  {'—':>5}  {rec['outcome']:<10}  not enough data")
            continue

        scored   += 1
        is_win    = rec["pnl_pct"] > 0
        pred_win  = score >= threshold
        ok        = is_win == pred_win
        correct  += ok

        if is_win and pred_win:   true_pos  += 1
        elif not is_win and pred_win: false_pos += 1
        elif is_win and not pred_win: false_neg += 1
        else:                         true_neg  += 1

        pred_label = "BUY ✓" if (pred_win and ok) else "BUY ✗" if pred_win else "SKIP ✓" if ok else "SKIP ✗"
        print(
            f"  {rec['symbol']:<14} {rec['pnl_pct']:>+7.1f}%  {score:>5.1f}  {rec['outcome']:<10}  {pred_label}"
        )

    print(SEP)
    if scored:
        accuracy  = correct / scored * 100
        precision = true_pos / (true_pos + false_pos) * 100 if (true_pos + false_pos) else 0
        recall    = true_pos / (true_pos + false_neg) * 100 if (true_pos + false_neg) else 0
        wins      = sum(1 for r in records if r["pnl_pct"] > 0)
        print(f"  Accuracy  : {correct}/{scored}  ({accuracy:.0f}%)")
        print(f"  Precision : {true_pos}/{true_pos+false_pos}  ({precision:.0f}%)  — of predicted BUYs, how many won")
        print(f"  Recall    : {true_pos}/{true_pos+false_neg}  ({recall:.0f}%)  — of actual wins, how many were caught")
        print(f"  Base rate : {wins}/{len(records)} trades were wins  ({wins/len(records)*100:.0f}%)")
        print(f"  Threshold : score >= {threshold:.0f} → BUY, else SKIP")
    print(SEP)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def run(args: argparse.Namespace, cfg: Config) -> None:
    print(SEP)
    print(f"  PARSING  {args.log}")
    print(SEP)

    trades = parse_log(args.log, strategy=args.strategy)
    if not trades:
        print(f"[ERROR] No completed {args.strategy} positions found in {args.log}")
        return

    if args.min_usd:
        before = len(trades)
        trades = [t for t in trades if t["usd_size"] >= args.min_usd]
        print(f"  Filtered to usd_size >= ${args.min_usd:.0f}: {len(trades)}/{before} trades kept")

    if args.limit:
        trades = trades[: args.limit]

    if args.tail:
        trades = trades[-args.tail :]

    wins   = sum(1 for t in trades if t["pnl_pct"] > 0)
    losses = len(trades) - wins
    print(f"  {len(trades)} completed positions  ({wins} wins / {losses} losses)")
    print()

    cache = load_cache()

    print(SEP)
    print("  FETCHING / LOADING CANDLES")
    print(SEP)
    async with aiohttp.ClientSession() as session:
        birdeye  = BirdeyePriceClient(cfg, session)
        moralis  = MoralisOHLCVClient(cfg, session) if args.moralis else None
        if args.moralis and not cfg.moralis_api_key:
            print("[ERROR] --moralis requires MORALIS_API_KEY in .env")
            return
        await fetch_all_candles(
            trades, birdeye, cache, args.fetch,
            moralis=moralis, delay=args.delay,
        )

    run_loo_evaluation(trades, cache, threshold=args.threshold, use_moralis=args.moralis)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leave-one-out ML scorer evaluation from a trades.log file."
    )
    parser.add_argument("--log", required=True, metavar="FILE",
                        help="Path to trades.log")
    parser.add_argument("--strategy", default="quick_pop",
                        help="Strategy to evaluate (default: quick_pop)")
    parser.add_argument("--fetch", action="store_true",
                        help="Force-refresh all candles (ignore cache)")
    parser.add_argument("--limit", type=int, default=0, metavar="N",
                        help="Only process first N trades (0 = all)")
    parser.add_argument("--tail", type=int, default=0, metavar="N",
                        help="Only process last N trades — use for recent data where 10s candles are still live")
    parser.add_argument("--min-usd", type=float, default=0.0, metavar="USD",
                        help="Only include positions with usd_size >= this value (default: 0 = all)")
    parser.add_argument("--threshold", type=float, default=5.0, metavar="SCORE",
                        help="Score threshold for BUY prediction (default: 5.0)")
    parser.add_argument("--delay", type=float, default=1.0, metavar="SECS",
                        help="Delay between API calls (default: 1.0)")
    parser.add_argument("--moralis", action="store_true",
                        help="Use Moralis 10s candles for ML features instead of Birdeye 1m "
                             "(requires MORALIS_API_KEY in .env)")
    args = parser.parse_args()

    if not os.path.exists(args.log):
        print(f"[ERROR] Log file not found: {args.log}", file=sys.stderr)
        sys.exit(1)

    try:
        cfg = Config.load()
    except ValueError as exc:
        print(f"[ERROR] Config: {exc}", file=sys.stderr)
        sys.exit(1)

    asyncio.run(run(args, cfg))


if __name__ == "__main__":
    main()
