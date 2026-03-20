#!/usr/bin/env python3
"""
scripts/check_quick_pop_ml.py — Inspect quick_pop_managed signal history.

Checks:
  1. Recent signals seen by quick_pop_managed — entered vs skipped
  2. Candle source for each signal (moralis-10s vs birdeye-1m)
  3. Whether ml_score is populated and what values look like
  4. Training data quality (quick_pop closed outcomes used by KNN)

Usage:
    python scripts/check_quick_pop_ml.py
    python scripts/check_quick_pop_ml.py --limit 50   # show more signals
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

DB_PATH = os.getenv("DB_PATH", "trader.db")
SEP = "=" * 72


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=30, metavar="N",
                        help="Number of recent signals to show (default: 30)")
    args = parser.parse_args()

    if not os.path.exists(DB_PATH):
        print(f"[ERROR] Database not found: {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)

    # ------------------------------------------------------------------
    # 1. Recent quick_pop_managed signals
    # ------------------------------------------------------------------
    rows = conn.execute(
        """
        SELECT sc.ts, sc.symbol, sc.ml_score,
               LENGTH(sc.candles_json)                                    AS clen,
               sc.candle_count,
               ROUND(LENGTH(sc.candles_json) * 1.0 / MAX(sc.candle_count, 1)) AS bpc,
               so.entered, so.outcome_pnl_pct
          FROM strategy_outcomes so
          JOIN signal_charts sc ON so.signal_chart_id = sc.id
         WHERE so.strategy = 'quick_pop_managed'
         ORDER BY sc.ts DESC
         LIMIT ?
        """,
        (args.limit,),
    ).fetchall()

    print(f"\n{SEP}")
    print(f"  QUICK_POP_CHART_ML — last {args.limit} signals")
    print(SEP)

    if not rows:
        print("  No signals found for quick_pop_managed.")
    else:
        scored = sum(1 for r in rows if r[2] is not None)
        moralis = sum(1 for r in rows if r[5] and r[5] > 400)
        entered = sum(1 for r in rows if r[6] == 1)

        print(f"  {'ts':16}  {'symbol':10}  {'src':11}  {'ml_score':8}  {'entered':7}  pnl%")
        print(f"  {'-'*16}  {'-'*10}  {'-'*11}  {'-'*8}  {'-'*7}  ----")

        for ts, symbol, ml_score, clen, candle_count, bpc, entered_flag, pnl in rows:
            src = "moralis-10s" if bpc and bpc > 400 else "birdeye-1m "
            score_str = f"{ml_score:.2f}" if ml_score is not None else "None"
            ent_str = "entered" if entered_flag == 1 else "skipped"
            pnl_str = f"{pnl:+.1f}%" if pnl is not None else "open"
            print(f"  {str(ts)[:16]}  {str(symbol):<10}  {src}  {score_str:<8}  {ent_str:<7}  {pnl_str}")

        print()
        print(f"  Signals shown   : {len(rows)}")
        print(f"  With ml_score   : {scored}/{len(rows)}"
              + (" ✓" if scored > 0 else " ✗ scorer not running or returning None"))
        print(f"  Moralis-10s src : {moralis}/{len(rows)}"
              + (" ✓" if moralis > 0 else " ✗ Moralis candles not being saved"))
        print(f"  Entered         : {entered}/{len(rows)}")

    # ------------------------------------------------------------------
    # 2. quick_pop_chart entered buys (the base strategy)
    # ------------------------------------------------------------------
    buy_rows = conn.execute(
        """
        SELECT sc.ts, sc.symbol, so.outcome_pnl_pct, so.outcome_sell_reason,
               so.outcome_hold_secs, so.closed
          FROM strategy_outcomes so
          JOIN signal_charts sc ON so.signal_chart_id = sc.id
         WHERE so.strategy = 'quick_pop_chart'
           AND so.entered  = 1
         ORDER BY sc.ts DESC
         LIMIT ?
        """,
        (args.limit,),
    ).fetchall()

    print(f"\n{SEP}")
    print(f"  QUICK_POP_CHART — last {args.limit} entered buys (base strategy / training source)")
    print(SEP)

    if not buy_rows:
        print("  No entered buys found for quick_pop_chart.")
    else:
        closed_count = sum(1 for r in buy_rows if r[5] == 1)
        wins = sum(1 for r in buy_rows if r[2] is not None and r[2] > 0)
        losses = sum(1 for r in buy_rows if r[2] is not None and r[2] <= 0)
        pnls = [r[2] for r in buy_rows if r[2] is not None]
        avg_pnl = sum(pnls) / len(pnls) if pnls else None

        print(f"  {'ts':16}  {'symbol':12}  {'pnl%':7}  {'reason':14}  {'hold':8}")
        print(f"  {'-'*16}  {'-'*12}  {'-'*7}  {'-'*14}  {'-'*8}")
        for ts, symbol, pnl, reason, hold_secs, closed in buy_rows:
            pnl_str  = f"{pnl:+.1f}%" if pnl is not None else "open"
            hold_str = f"{hold_secs/60:.0f}m"  if hold_secs is not None else "-"
            reason_str = reason or ("open" if not closed else "-")
            print(f"  {str(ts)[:16]}  {str(symbol):<12}  {pnl_str:<7}  {reason_str:<14}  {hold_str}")

        print()
        print(f"  Total shown     : {len(buy_rows)}  ({closed_count} closed)")
        if avg_pnl is not None:
            print(f"  Win/Loss        : {wins}W / {losses}L  avg={avg_pnl:+.1f}%")

    # ------------------------------------------------------------------
    # 3. Overall outcome summary
    # ------------------------------------------------------------------
    summary = conn.execute(
        """
        SELECT
          COUNT(*)                                                       AS total,
          SUM(CASE WHEN so.entered = 1 THEN 1 ELSE 0 END)              AS entered,
          SUM(CASE WHEN so.entered = 0 THEN 1 ELSE 0 END)              AS skipped,
          SUM(CASE WHEN sc.ml_score IS NOT NULL THEN 1 ELSE 0 END)     AS scored,
          SUM(CASE WHEN sc.ml_score IS NULL THEN 1 ELSE 0 END)         AS unscored,
          MIN(sc.ts),
          MAX(sc.ts)
        FROM strategy_outcomes so
        JOIN signal_charts sc ON so.signal_chart_id = sc.id
        WHERE so.strategy = 'quick_pop_managed'
        """
    ).fetchone()

    total, entered_t, skipped_t, scored_t, unscored_t, first_ts, last_ts = summary

    print(f"\n{SEP}")
    print("  OVERALL SUMMARY (all time)")
    print(SEP)
    print(f"  Total signals   : {total}")
    print(f"  Entered         : {entered_t}")
    print(f"  Skipped         : {skipped_t}")
    print(f"  With ml_score   : {scored_t}  ({'%.0f' % (scored_t/total*100) if total else 0}%)")
    print(f"  Without score   : {unscored_t}")
    if first_ts:
        print(f"  First signal    : {str(first_ts)[:16]}")
        print(f"  Last signal     : {str(last_ts)[:16]}")

    # ------------------------------------------------------------------
    # 3. ml_score distribution (for scored signals)
    # ------------------------------------------------------------------
    score_dist = conn.execute(
        """
        SELECT
          SUM(CASE WHEN sc.ml_score < 3  THEN 1 ELSE 0 END) AS low,
          SUM(CASE WHEN sc.ml_score >= 3 AND sc.ml_score < 5  THEN 1 ELSE 0 END) AS below_thresh,
          SUM(CASE WHEN sc.ml_score >= 5 AND sc.ml_score < 8  THEN 1 ELSE 0 END) AS pass,
          SUM(CASE WHEN sc.ml_score >= 8 AND sc.ml_score < 9.5 THEN 1 ELSE 0 END) AS high,
          SUM(CASE WHEN sc.ml_score >= 9.5 THEN 1 ELSE 0 END) AS max_tier,
          MIN(sc.ml_score), MAX(sc.ml_score), AVG(sc.ml_score)
        FROM strategy_outcomes so
        JOIN signal_charts sc ON so.signal_chart_id = sc.id
        WHERE so.strategy = 'quick_pop_managed'
          AND sc.ml_score IS NOT NULL
        """
    ).fetchone()

    low, below, passing, high, max_tier, mn, mx, avg = score_dist
    if scored_t and scored_t > 0:
        print(f"\n{SEP}")
        print("  ML SCORE DISTRIBUTION (scored signals only)")
        print(SEP)
        print(f"  < 3.0  (low confidence)     : {low}")
        print(f"  3–5    (below threshold)     : {below}  ← skipped by ML filter")
        print(f"  5–8    (pass, normal size)   : {passing}")
        print(f"  8–9.5  (high, 2× size)       : {high}")
        print(f"  ≥ 9.5  (max, 3× size)        : {max_tier}")
        print(f"  Range  : {mn:.2f} – {mx:.2f}  avg={avg:.2f}")

    # ------------------------------------------------------------------
    # 4. Training data (quick_pop closed outcomes)
    # ------------------------------------------------------------------
    training = conn.execute(
        """
        SELECT COUNT(*),
               SUM(CASE WHEN sc.candles_json IS NOT NULL
                         AND LENGTH(sc.candles_json) > 10 THEN 1 ELSE 0 END),
               SUM(CASE WHEN ROUND(LENGTH(sc.candles_json)*1.0/MAX(sc.candle_count,1)) > 400
                         THEN 1 ELSE 0 END),
               MIN(sc.ts), MAX(sc.ts)
          FROM strategy_outcomes so
          JOIN signal_charts sc ON so.signal_chart_id = sc.id
         WHERE so.strategy = 'quick_pop'
           AND so.closed = 1
           AND so.outcome_pnl_pct IS NOT NULL
        """
    ).fetchone()

    tr_total, tr_with_candles, tr_moralis, tr_first, tr_last = training

    print(f"\n{SEP}")
    print("  TRAINING DATA (quick_pop closed outcomes used by KNN)")
    print(SEP)
    print(f"  Total rows      : {tr_total}  (need >= 5)")
    print(f"  With candles    : {tr_with_candles}")
    print(f"  Moralis-10s     : {tr_moralis}  (backfill is birdeye-1m; grows as live trades close)")
    if tr_first:
        print(f"  Date range      : {str(tr_first)[:16]} → {str(tr_last)[:16]}")
        ts_span = (tr_last or "")[:16]
        ts_start = (tr_first or "")[:16]
        if ts_start[:10] == ts_span[:10]:
            print(f"  ⚠ All training rows have the same date — likely backfill timestamp bug.")
            print(f"    Fix: re-run  python scripts/backfill_snapshots.py --force")
            print(f"    (needs --force flag to overwrite existing rows with correct ts)")

    # ------------------------------------------------------------------
    # 5. Training PnL distribution — shows why scores are high/low
    # ------------------------------------------------------------------
    pnl_dist = conn.execute(
        """
        SELECT
          COUNT(*)                                                          AS total,
          SUM(CASE WHEN so.outcome_pnl_pct > 0  THEN 1 ELSE 0 END)       AS wins,
          SUM(CASE WHEN so.outcome_pnl_pct <= 0 THEN 1 ELSE 0 END)       AS losses,
          ROUND(AVG(so.outcome_pnl_pct), 1)                               AS avg_pnl,
          ROUND(MIN(so.outcome_pnl_pct), 1)                               AS worst,
          ROUND(MAX(so.outcome_pnl_pct), 1)                               AS best,
          SUM(CASE WHEN so.outcome_pnl_pct > 20  THEN 1 ELSE 0 END)      AS big_wins,
          SUM(CASE WHEN so.outcome_pnl_pct < -15 THEN 1 ELSE 0 END)      AS big_losses
        FROM strategy_outcomes so
        WHERE so.strategy = 'quick_pop'
          AND so.closed = 1
          AND so.outcome_pnl_pct IS NOT NULL
        """
    ).fetchone()

    total_tr, wins_tr, losses_tr, avg_pnl, worst, best, big_wins, big_losses = pnl_dist

    if total_tr:
        win_rate = wins_tr / total_tr * 100
        # Score at avg_pnl using default quick_pop range (-35 to 100)
        score_at_avg = max(0.0, min(10.0, (avg_pnl + 35) / 135 * 10)) if avg_pnl else 0.0
        print(f"\n{SEP}")
        print("  TRAINING PnL DISTRIBUTION (explains why scores are where they are)")
        print(SEP)
        print(f"  Win rate        : {win_rate:.1f}%  ({wins_tr} wins / {losses_tr} losses)")
        print(f"  Avg PnL         : {avg_pnl:+.1f}%  → KNN avg score ≈ {score_at_avg:.1f}/10")
        print(f"  Range           : {worst:+.1f}% to {best:+.1f}%")
        print(f"  Big wins (>20%) : {big_wins}")
        print(f"  Big losses(<-15%): {big_losses}")
        if avg_pnl < 0:
            print(f"\n  ⚠ Training data is net-negative — KNN predicts losses for most signals.")
            print(f"    Scores will cluster below 5.0 until the underlying strategy improves")
            print(f"    or the training set accumulates more winning trades.")
            threshold_pnl = -35 + (5.0 / 10) * 135  # pnl that maps to score=5
            print(f"    Score ≥ 5.0 requires signals resembling training rows with PnL ≥ {threshold_pnl:+.1f}%")

    conn.close()
    print()


if __name__ == "__main__":
    main()
