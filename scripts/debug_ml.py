#!/usr/bin/env python3
"""
scripts/debug_ml.py — Diagnose why ML scoring is returning None.

Checks:
  1. How many quick_pop training examples exist in the DB
  2. Candle quality of those training rows (empty candles = scorer returns None)
  3. Whether signal_charts rows have ml_score populated
  4. Whether Moralis/Birdeye candle fetching works for a recent mint
  5. Whether the scorer can produce a score given the current training data

Usage:
    python scripts/debug_ml.py               # DB checks only (no API calls)
    python scripts/debug_ml.py --live MINT   # also test candle fetch + scoring for a mint
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

DB_PATH = os.getenv("DB_PATH", "trader.db")
SEP = "=" * 60


# ---------------------------------------------------------------------------
# 1. DB diagnostics
# ---------------------------------------------------------------------------

def check_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)

    # Training data available to the scorer
    training = conn.execute(
        """
        SELECT COUNT(*), SUM(CASE WHEN so.outcome_pnl_pct IS NOT NULL THEN 1 ELSE 0 END)
          FROM strategy_outcomes so
          JOIN signal_charts sc ON so.signal_chart_id = sc.id
         WHERE so.strategy = 'quick_pop'
           AND so.closed = 1
           AND so.entered = 1
        """
    ).fetchone()
    total_training, with_pnl = training
    print(f"\n{SEP}")
    print("  TRAINING DATA (quick_pop closed outcomes)")
    print(SEP)
    print(f"  Total closed quick_pop rows : {total_training}")
    print(f"  Rows with outcome_pnl_pct   : {with_pnl}")
    print(f"  MIN_SAMPLES required        : 5")
    if (with_pnl or 0) >= 5:
        print(f"  ✓ Enough training data rows exist")
    else:
        print(f"  ✗ Not enough training data — need {5 - (with_pnl or 0)} more closed trades")

    # Candle quality of the training rows — this is what actually causes None scores
    training_rows = conn.execute(
        """
        SELECT sc.candle_count, LENGTH(sc.candles_json) as clen, sc.candles_json
          FROM strategy_outcomes so
          JOIN signal_charts sc ON so.signal_chart_id = sc.id
         WHERE so.strategy = 'quick_pop'
           AND so.closed = 1 AND so.entered = 1
           AND so.outcome_pnl_pct IS NOT NULL
        """
    ).fetchall()
    empty   = sum(1 for _, _, cj in training_rows if not cj or cj in ('[]', 'null', '') or len(cj) < 10)
    usable  = sum(1 for cnt, _, _ in training_rows if (cnt or 0) >= 3)
    print(f"\n{SEP}")
    print("  TRAINING DATA CANDLE QUALITY (what scorer loads per signal)")
    print(SEP)
    print(f"  Total training rows         : {len(training_rows)}")
    print(f"  Rows with empty candles_json: {empty}")
    print(f"  Rows with >= 3 candles      : {usable}  ← scorer needs >= 5 of these")
    if usable >= 5:
        print(f"  ✓ Enough usable training rows — scorer should be producing scores")
        print(f"    If scores are still None, run --live to test the candle fetch")
    else:
        print(f"  ✗ Only {usable} usable training rows — this is why scorer returns None")
        print(f"    Fix: python scripts/backfill_snapshots.py")

    # signal_charts ml_score population
    sc_stats = conn.execute(
        """
        SELECT COUNT(*),
               SUM(CASE WHEN ml_score IS NOT NULL THEN 1 ELSE 0 END),
               SUM(CASE WHEN candles_json IS NOT NULL AND candles_json != '[]' THEN 1 ELSE 0 END)
          FROM signal_charts
        """
    ).fetchone()
    total_sc, with_score, with_candles = sc_stats
    print(f"\n{SEP}")
    print("  SIGNAL_CHARTS")
    print(SEP)
    print(f"  Total signal_chart rows     : {total_sc}")
    print(f"  Rows with ml_score          : {with_score}")
    print(f"  Rows with candles           : {with_candles}")
    if (with_score or 0) == 0:
        print(f"  ✗ No ml_scores stored — scorer is returning None for every signal")
    else:
        print(f"  ✓ Some signals have been scored ({with_score}/{total_sc})")

    # quick_pop_chart_ml strategy_outcomes
    qp_stats = conn.execute(
        """
        SELECT COUNT(*),
               SUM(CASE WHEN entered = 1 THEN 1 ELSE 0 END),
               SUM(CASE WHEN entered = 0 THEN 1 ELSE 0 END)
          FROM strategy_outcomes
         WHERE strategy = 'quick_pop_chart_ml'
        """
    ).fetchone()
    total_qp, entered_qp, skipped_qp = qp_stats
    print(f"\n{SEP}")
    print("  QUICK_POP_CHART_ML OUTCOMES")
    print(SEP)
    print(f"  Total signals seen          : {total_qp}")
    print(f"  Entered                     : {entered_qp}")
    print(f"  Skipped (filtered)          : {skipped_qp}")

    # Most recent signal_charts to spot candle issues
    recent = conn.execute(
        """
        SELECT ts, symbol, mint, ml_score,
               candle_count,
               LENGTH(candles_json) as candles_len
          FROM signal_charts
         ORDER BY ts DESC
         LIMIT 5
        """
    ).fetchall()
    print(f"\n{SEP}")
    print("  5 MOST RECENT SIGNALS")
    print(SEP)
    for ts, sym, mint, score, count, clen in recent:
        score_str = f"{score:.1f}" if score is not None else "None"
        print(f"  {ts[:16]}  {sym:<10}  ml_score={score_str:<6}  candles={count}  candles_json_len={clen}")

    conn.close()


# ---------------------------------------------------------------------------
# 2. Live candle + scorer test
# ---------------------------------------------------------------------------

async def test_live_scoring(mint: str) -> None:
    from trader.config import Config
    from trader.persistence.database import TradeDatabase
    from trader.pricing.birdeye import BirdeyePriceClient
    from trader.analysis.ml_scorer import (
        ChartMLScorer, ML_OHLCV_BARS, ML_OHLCV_INTERVAL,
        MORALIS_OHLCV_BARS, MORALIS_OHLCV_INTERVAL,
    )

    try:
        cfg = Config.load()
    except ValueError as exc:
        print(f"[ERROR] Config: {exc}")
        return

    import aiohttp
    async with aiohttp.ClientSession() as session:
        birdeye = BirdeyePriceClient(cfg, session)

        print(f"\n{SEP}")
        print(f"  LIVE CANDLE FETCH  mint={mint[:12]}...")
        print(SEP)

        # Birdeye 1m candles
        candles = await birdeye.get_ohlcv(mint, bars=ML_OHLCV_BARS, interval=ML_OHLCV_INTERVAL)
        print(f"  Birdeye 1m candles returned : {len(candles)}")
        if not candles:
            print("  ✗ No candles — scorer will return None (no input data)")
        else:
            print(f"  ✓ Got candles  (first close={candles[0].close:.8f}, last close={candles[-1].close:.8f})")

        # Moralis (optional)
        moralis_candles = []
        moralis_key = os.getenv("MORALIS_API_KEY", "").strip()
        if moralis_key:
            try:
                from trader.pricing.moralis import MoralisOHLCVClient
                moralis = MoralisOHLCVClient(moralis_key, session)
                moralis_candles = await moralis.get_ohlcv(
                    mint, bars=MORALIS_OHLCV_BARS, interval=MORALIS_OHLCV_INTERVAL
                )
                print(f"  Moralis 10s candles returned: {len(moralis_candles)}")
            except Exception as exc:
                print(f"  Moralis fetch failed: {exc}")
        else:
            print("  Moralis: MORALIS_API_KEY not set — using Birdeye only")

        active_candles = moralis_candles if moralis_candles else candles
        if not active_candles:
            print("  ✗ No candles from any source — cannot score")
            return

        print(f"\n{SEP}")
        print("  SCORER TEST")
        print(SEP)

        db = TradeDatabase(path=DB_PATH)
        scorer = ChartMLScorer(db, strategy="quick_pop")
        score = scorer.score(active_candles, chart_ctx=None, pair_stats=None)
        db.close()

        if score is None:
            print("  ✗ Scorer returned None")
            print("    Most likely cause: training rows have empty candles_json")
            print("    Fix: python scripts/backfill_snapshots.py")
        else:
            print(f"  ✓ Score = {score:.2f} / 10")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Debug ML scoring pipeline")
    parser.add_argument(
        "--live", metavar="MINT",
        help="Also fetch live candles and run the scorer for this mint address"
    )
    args = parser.parse_args()

    if not os.path.exists(DB_PATH):
        print(f"[ERROR] Database not found: {DB_PATH}")
        sys.exit(1)

    check_db(DB_PATH)

    if args.live:
        asyncio.run(test_live_scoring(args.live))
    else:
        print(f"\n  Tip: add --live <MINT> to also test candle fetching + live scoring")

    print()


if __name__ == "__main__":
    main()
