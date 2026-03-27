#!/usr/bin/env python3
"""
scripts/optimize_filters.py — AI-driven hard entry filter optimizer.

Tests combinations of holder_count_max, late_entry_price_chg_30m_max,
late_entry_pump_ratio_min, buy_vol_ratio_1h_max, and market_cap_usd_min
against historical entered trades to find the filter settings that maximise
net PnL impact (losses avoided minus profits foregone).

How it works:
  1. Loads all entered+closed trades from trader.db with pair_stats + 1m candles.
  2. Computes per-trade filter inputs: holder_count, price_change_30m_pct, pump_ratio,
     buy_vol_ratio_1h, market_cap_usd.
  3. For each filter config candidate, simulates which trades would have been blocked.
  4. Computes metrics: net_pnl_delta, losses_avoided, profits_foregone,
     heavy_losses_blocked, moonshots_missed, block_rate, win_rate_passed.
  5. Uses Claude to iteratively suggest better filter configs based on results.
  6. Prints the top configs found across all rounds.

Modes:
  --mode balanced (default): maximize passed_pnl with block_rate < 35%
  --mode strict:  maximize win_rate_passed (>50% target), high block_rate OK
  --mode lenient: zero moonshot misses, maximize passed_pnl

Usage:
    python scripts/optimize_filters.py [--strategy quick_pop|trend_rider|moonbag|all]
                                       [--rounds 5] [--suggestions 10]
                                       [--random-per-round 8] [--mode balanced|strict|lenient]
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

DB_PATH = "trader.db"

# Current production filter settings — used as baseline
CURRENT_FILTERS = {
    "holder_count_max":             1000,
    "late_entry_price_chg_30m_max": 250.0,
    "late_entry_pump_ratio_min":    20.0,
    "buy_vol_ratio_1h_max":         None,   # disabled by default
    "market_cap_usd_min":           None,   # disabled by default
}

STRATEGY_CONFIGS = {
    "quick_pop": {
        "db_strategy":        "quick_pop",
        "registry_name":      "quick_pop_managed",
        "buy_size":           30.0,
        "heavy_loss_threshold": -40.0,   # outcome_pnl_pct below this = heavy loss
        "moonshot_threshold":   200.0,   # outcome_pnl_pct above this = moonshot
    },
    "trend_rider": {
        "db_strategy":        "trend_rider",
        "registry_name":      "trend_rider_managed",
        "buy_size":           10.0,
        "heavy_loss_threshold": -40.0,
        "moonshot_threshold":   200.0,
    },
    "moonbag": {
        "db_strategy":        "infinite_moonbag",
        "registry_name":      "moonbag_managed",
        "buy_size":           5.0,
        "heavy_loss_threshold": -40.0,
        "moonshot_threshold":   200.0,
    },
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _pump_ratio_from_candles(candles_1m: list[dict]) -> float | None:
    """Compute pump_ratio = close_of_last_candle / lowest_low across all candles."""
    if not candles_1m:
        return None
    try:
        lows  = [float(c["l"]) for c in candles_1m if c.get("l") is not None]
        close = float(candles_1m[-1]["c"])
        if not lows or close <= 0:
            return None
        lowest = min(lows)
        if lowest <= 0:
            return None
        return close / lowest
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None


def load_trades(db_path: str, db_strategy: str) -> list[dict]:
    """Load all entered+closed trades with filter inputs and PnL."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT
            so.outcome_pnl_pct,
            so.outcome_pnl_usd,
            sc.pair_stats_json,
            sc.candles_1m_json,
            sc.ts AS signal_ts
        FROM strategy_outcomes so
        JOIN signal_charts sc ON sc.id = so.signal_chart_id
        WHERE so.strategy = ?
          AND so.closed   = 1
          AND so.entered  = 1
          AND so.outcome_pnl_pct IS NOT NULL
        ORDER BY sc.ts
    """, (db_strategy,)).fetchall()
    conn.close()

    trades = []
    for row in rows:
        pnl_pct, pnl_usd, ps_j, c1m_j, signal_ts = row

        pair_stats = json.loads(ps_j)  if ps_j  else {}
        candles_1m = json.loads(c1m_j) if c1m_j else []

        bv  = pair_stats.get("buy_volume_1h",   0.0) or 0.0
        tv  = pair_stats.get("total_volume_1h", 0.0) or 0.0
        bvr = bv / (tv + 1e-9) if tv > 0 else None

        trades.append({
            "outcome_pnl_pct":    pnl_pct or 0.0,
            "outcome_pnl_usd":    pnl_usd or 0.0,
            "holder_count":       pair_stats.get("holder_count"),
            "price_chg_30m":      pair_stats.get("price_change_30m_pct"),
            "pump_ratio":         _pump_ratio_from_candles(candles_1m),
            "buy_vol_ratio_1h":   bvr,
            "market_cap_usd":     pair_stats.get("market_cap_usd"),
            "signal_ts":          signal_ts,
        })
    return trades


# ---------------------------------------------------------------------------
# Filter application
# ---------------------------------------------------------------------------

def would_block(trade: dict, cfg: dict) -> bool:
    """Return True if this filter config would block the trade at entry."""
    hc_max   = cfg.get("holder_count_max")
    pc30_max = cfg.get("late_entry_price_chg_30m_max")
    pr_min   = cfg.get("late_entry_pump_ratio_min")
    bvr_max  = cfg.get("buy_vol_ratio_1h_max")
    mc_min   = cfg.get("market_cap_usd_min")

    # Holder count filter
    if hc_max is not None and trade["holder_count"] is not None:
        if trade["holder_count"] > hc_max:
            return True

    # Late-entry combined filter — BOTH conditions must exceed thresholds
    if pc30_max is not None and pr_min is not None:
        pc30 = trade["price_chg_30m"]
        pr   = trade["pump_ratio"]
        if (pc30 is not None and pr is not None
                and pc30 > pc30_max and pr > pr_min):
            return True

    # Buy volume ratio filter — high ratio = selling pressure / wash trading
    if bvr_max is not None and trade["buy_vol_ratio_1h"] is not None:
        if trade["buy_vol_ratio_1h"] > bvr_max:
            return True

    # Market cap minimum — tiny mc tokens are extremely high risk
    if mc_min is not None and trade["market_cap_usd"] is not None:
        if trade["market_cap_usd"] < mc_min:
            return True

    return False


# ---------------------------------------------------------------------------
# Filter evaluation
# ---------------------------------------------------------------------------

def evaluate_filters(trades: list[dict], cfg: dict,
                     heavy_loss_thr: float, moonshot_thr: float) -> dict:
    """Evaluate a filter config across all trades and return metrics."""
    blocked_pnl      = 0.0
    passed_pnl       = 0.0
    blocked_count    = 0
    passed_count     = 0
    heavy_blocked    = 0   # heavy losers correctly blocked
    moonshots_missed = 0   # moonshots incorrectly blocked

    passed_winners = 0
    for t in trades:
        blocked = would_block(t, cfg)
        pnl_usd = t["outcome_pnl_usd"]
        pnl_pct = t["outcome_pnl_pct"]

        if blocked:
            blocked_pnl += pnl_usd
            blocked_count += 1
            if pnl_pct <= heavy_loss_thr:
                heavy_blocked += 1
            if pnl_pct >= moonshot_thr:
                moonshots_missed += 1
        else:
            passed_pnl += pnl_usd
            passed_count += 1
            if pnl_pct > 0:
                passed_winners += 1

    n = len(trades)
    # net_pnl_delta: positive means we avoided more losses than profits foregone.
    # = -(sum of PnL we blocked). If we blocked losses (negative PnL), delta is positive.
    net_pnl_delta    = -blocked_pnl
    losses_avoided   = abs(blocked_pnl) if blocked_pnl < 0 else 0.0
    profits_foregone = blocked_pnl      if blocked_pnl > 0 else 0.0
    win_rate_passed  = passed_winners / passed_count if passed_count else 0.0

    return {
        "n_trades":         n,
        "blocked_count":    blocked_count,
        "passed_count":     passed_count,
        "block_rate":       blocked_count / n if n else 0.0,
        "net_pnl_delta":    net_pnl_delta,
        "losses_avoided":   losses_avoided,
        "profits_foregone": profits_foregone,
        "heavy_blocked":    heavy_blocked,
        "moonshots_missed": moonshots_missed,
        "passed_pnl":       passed_pnl,
        "win_rate_passed":  win_rate_passed,
        "passed_winners":   passed_winners,
    }


# ---------------------------------------------------------------------------
# Config validation / random generation
# ---------------------------------------------------------------------------

def validate_filters(cfg: dict) -> dict | None:
    """Clamp and validate a filter config. Returns None if invalid."""
    try:
        hc   = cfg.get("holder_count_max")
        pc30 = cfg.get("late_entry_price_chg_30m_max")
        pr   = cfg.get("late_entry_pump_ratio_min")
        bvr  = cfg.get("buy_vol_ratio_1h_max")
        mc   = cfg.get("market_cap_usd_min")

        if hc is not None:
            hc = max(100, min(10_000, int(round(hc))))

        if pc30 is not None:
            pc30 = max(50.0, min(2000.0, float(pc30)))

        if pr is not None:
            pr = max(2.0, min(100.0, float(pr)))

        if bvr is not None:
            bvr = max(0.01, min(1.0, float(bvr)))

        if mc is not None:
            mc = max(1_000.0, min(5_000_000.0, float(mc)))

        # late_entry filter: both params must be set together, or both null
        if (pc30 is None) != (pr is None):
            return None

        return {
            "holder_count_max":             hc,
            "late_entry_price_chg_30m_max": pc30,
            "late_entry_pump_ratio_min":    pr,
            "buy_vol_ratio_1h_max":         bvr,
            "market_cap_usd_min":           mc,
        }
    except Exception:
        return None


def random_filters(data_coverage: dict | None = None) -> dict:
    """Generate a random filter config within reasonable bounds.

    data_coverage: dict of {field: fraction of trades with non-null data}.
    Filters with <10% data coverage are disabled more often (70% chance of None)
    to avoid overfitting to sparse data.
    """
    cov = data_coverage or {}

    def use_filter(field: str, base_prob: float = 0.75) -> bool:
        c = cov.get(field, 1.0)
        if c < 0.10:
            return random.random() > 0.70   # only 30% chance if data very sparse
        elif c < 0.30:
            return random.random() > 0.40
        return random.random() > (1 - base_prob)

    use_holder     = use_filter("holder_count")
    use_late_entry = use_filter("price_chg_30m")
    use_bvr        = use_filter("buy_vol_ratio_1h")
    use_mc         = use_filter("market_cap_usd")

    hc   = int(round(random.uniform(200, 3000))) if use_holder     else None
    pc30 = round(random.uniform(60, 800),  1)    if use_late_entry else None
    pr   = round(random.uniform(4, 60),    1)    if use_late_entry else None
    bvr  = round(random.uniform(0.02, 0.80), 3)  if use_bvr        else None
    mc   = round(random.uniform(5_000, 500_000)) if use_mc         else None

    return validate_filters({
        "holder_count_max":             hc,
        "late_entry_price_chg_30m_max": pc30,
        "late_entry_pump_ratio_min":    pr,
        "buy_vol_ratio_1h_max":         bvr,
        "market_cap_usd_min":           mc,
    }) or CURRENT_FILTERS


# ---------------------------------------------------------------------------
# Claude suggestion
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_BALANCED = """You are an expert optimizer for cryptocurrency trading entry filters.

The bot uses these hard entry filters (all optional — set to null to disable):

1. holder_count_max (int|null): Block tokens with holder count > this. Range: 100–10000.
   Rationale: very high counts may indicate Sybil/wash-trading pump-and-dump setups.

2. Late-entry combined filter (BOTH params required together, or BOTH null):
   - late_entry_price_chg_30m_max (float): max allowed 30m price change (%).
   - late_entry_pump_ratio_min (float): min pump_ratio (close / lowest_low from 1m candles).
   Blocked ONLY when BOTH thresholds exceeded simultaneously. Range: pc30 50–2000, pr 2–100.
   Rationale: tokens that pumped hard AND show high pump_ratio are likely late-entry chasing.

3. buy_vol_ratio_1h_max (float|null): Block when 1h buy_volume / total_volume > this. Range: 0.01–1.0.
   Rationale: very high buy ratio can signal coordinated wash trading / bot manipulation.

4. market_cap_usd_min (float|null): Block when market cap < this (USD). Range: 1000–5000000.
   Rationale: extremely low market caps have very high rug/dump risk.

DATA COVERAGE NOTE: you will be told what fraction of trades have each filter input.
Only suggest filters for fields with >20% coverage — otherwise results are statistically unreliable.

PRIMARY metric: passed_pnl = total PnL (USD) of trades that pass the filter.
  Maximise this. Blocking losers increases it; blocking winners decreases it.
  Blocking everything → passed_pnl = $0 (worst). The baseline "no_filters" shows max possible.

SECONDARY metrics:
  - net_pnl_delta (positive = net benefit from blocking)
  - heavy_blocked (want high), moonshots_missed (want low)
  - block_rate: keep < 35% — aggressive filters kill opportunity flow
  - win_rate_passed: fraction of passed trades that are winners

Rules:
- Return ONLY a JSON array of config objects, no explanation.
- Each config must have EXACTLY these five keys:
    "holder_count_max" (int or null),
    "late_entry_price_chg_30m_max" (float or null),
    "late_entry_pump_ratio_min" (float or null),
    "buy_vol_ratio_1h_max" (float or null),
    "market_cap_usd_min" (float or null)
- If late_entry_price_chg_30m_max is null, late_entry_pump_ratio_min must also be null.
- DIVERSITY RULE: at least half your suggestions must differ significantly from the current best.
- Return exactly the number of suggestions requested."""

SYSTEM_PROMPT_STRICT = """You are an expert optimizer for cryptocurrency trading entry filters in STRICT mode.

STRICT MODE GOAL: Maximize win_rate_passed (fraction of passed trades that are winners).
Target >50% win rate. High block_rate is ACCEPTABLE — even desirable.
A config that blocks 90% of trades but has 60% win rate beats one with 20% block rate and 25% win rate.

The bot uses these hard entry filters (all optional — set to null to disable):

1. holder_count_max (int|null): Block tokens with holder count > this. Range: 100–10000.
2. Late-entry combined filter (BOTH params required together, or BOTH null):
   - late_entry_price_chg_30m_max (float|null): max allowed 30m price change (%). Range: 50–2000.
   - late_entry_pump_ratio_min (float|null): min pump_ratio (close/lowest_low). Range: 2–100.
3. buy_vol_ratio_1h_max (float|null): Block when 1h buy_volume / total_volume > this. Range: 0.01–1.0.
4. market_cap_usd_min (float|null): Block when market cap < this (USD). Range: 1000–5000000.

DATA COVERAGE NOTE: only suggest filters for fields with >20% data coverage (shown in history).

PRIMARY metric: win_rate_passed (fraction of passed trades that are winners). MAXIMIZE THIS.
SECONDARY metric: passed_pnl (avoid blocking all winners — passed_pnl must be > $0).
TERTIARY: block_rate (higher is better in strict mode — precision over volume).

Rules:
- Return ONLY a JSON array of config objects, no explanation.
- Each config must have EXACTLY these five keys:
    "holder_count_max", "late_entry_price_chg_30m_max", "late_entry_pump_ratio_min",
    "buy_vol_ratio_1h_max", "market_cap_usd_min"
- If late_entry_price_chg_30m_max is null, late_entry_pump_ratio_min must also be null.
- DIVERSITY RULE: at least half your suggestions must differ significantly from the current best.
- Return exactly the number of suggestions requested."""

SYSTEM_PROMPT_LENIENT = """You are an expert optimizer for cryptocurrency trading entry filters in LENIENT mode.

LENIENT MODE GOAL: Let ALL winners through (especially moonshots). Miss ZERO winners.
Block as many losers as possible, but NEVER at the cost of missing a winner.

The bot uses these hard entry filters (all optional — set to null to disable):

1. holder_count_max (int|null): Block tokens with holder count > this. Range: 100–10000.
2. Late-entry combined filter (BOTH params required together, or BOTH null):
   - late_entry_price_chg_30m_max (float|null): max allowed 30m price change (%). Range: 50–2000.
   - late_entry_pump_ratio_min (float|null): min pump_ratio (close/lowest_low). Range: 2–100.
3. buy_vol_ratio_1h_max (float|null): Block when 1h buy_volume / total_volume > this. Range: 0.01–1.0.
4. market_cap_usd_min (float|null): Block when market cap < this (USD). Range: 1000–5000000.

DATA COVERAGE NOTE: only suggest filters for fields with >20% data coverage (shown in history).

PRIMARY metric: moonshots_missed — this MUST be 0. Configs that miss moonshots are worthless.
SECONDARY metric: heavy_blocked (want high — block as many big losers as possible).
TERTIARY: block_rate (want high — more filtering is better as long as moonshots=0).

STRATEGY: use very loose thresholds. A filter that blocks 5% of losers with 0 moonshots missed
beats one that blocks 50% of losers but misses 1 moonshot.

Rules:
- Return ONLY a JSON array of config objects, no explanation.
- Each config must have EXACTLY these five keys:
    "holder_count_max", "late_entry_price_chg_30m_max", "late_entry_pump_ratio_min",
    "buy_vol_ratio_1h_max", "market_cap_usd_min"
- If late_entry_price_chg_30m_max is null, late_entry_pump_ratio_min must also be null.
- DIVERSITY RULE: at least half your suggestions must differ significantly from the current best.
- Return exactly the number of suggestions requested."""


def ask_claude(client: anthropic.Anthropic, results_history: list[dict],
               n_suggestions: int, strategy_name: str,
               mode: str = "balanced",
               data_coverage: dict | None = None) -> list[dict]:
    """Ask Claude for n_suggestions new filter configs."""
    cov = data_coverage or {}

    # Cap history: top 12 by primary metric + last 4 recent configs
    if mode == "strict":
        primary_key = "win_rate_passed"
    elif mode == "lenient":
        primary_key = "moonshots_missed"   # sort ascending for lenient (want 0)
    else:
        primary_key = "passed_pnl"

    if mode == "lenient":
        sorted_hist = sorted(results_history, key=lambda r: (r["moonshots_missed"], -r["heavy_blocked"]))
    else:
        sorted_hist = sorted(results_history, key=lambda r: -r[primary_key])

    seen = set()
    capped = []
    for r in sorted_hist[:12]:
        capped.append(r)
        seen.add(r["label"])
    for r in results_history[-4:]:
        if r["label"] not in seen:
            capped.append(r)
            seen.add(r["label"])

    history_lines = []
    for i, r in enumerate(capped):
        cfg = r["config"]
        history_lines.append(
            f"  Config {i+1}: passed_pnl=${r['passed_pnl']:+.2f}  "
            f"win_rate={r['win_rate_passed']*100:.1f}%  "
            f"net_delta=${r['net_pnl_delta']:+.2f}  "
            f"heavy_blocked={r['heavy_blocked']}  "
            f"moonshots_missed={r['moonshots_missed']}  "
            f"block_rate={r['block_rate']*100:.1f}%  "
            f"hc_max={cfg['holder_count_max']}  "
            f"pc30_max={cfg['late_entry_price_chg_30m_max']}  "
            f"pr_min={cfg['late_entry_pump_ratio_min']}  "
            f"bvr_max={cfg.get('buy_vol_ratio_1h_max')}  "
            f"mc_min={cfg.get('market_cap_usd_min')}"
        )

    cov_lines = "\n".join(
        f"  {field}: {frac*100:.0f}% of trades have data"
        for field, frac in sorted(cov.items())
    )

    system_prompts = {
        "balanced": SYSTEM_PROMPT_BALANCED,
        "strict":   SYSTEM_PROMPT_STRICT,
        "lenient":  SYSTEM_PROMPT_LENIENT,
    }
    system_prompt = system_prompts.get(mode, SYSTEM_PROMPT_BALANCED)

    goal_line = {
        "balanced": "Maximise passed_pnl while keeping block_rate < 35%.",
        "strict":   "Maximise win_rate_passed (>50% target). High block_rate is fine.",
        "lenient":  "Achieve moonshots_missed=0, then maximise heavy_blocked.",
    }[mode]

    user_msg = (
        f"Strategy: {strategy_name}. Mode: {mode.upper()}. "
        f"Dataset: {results_history[0]['n_trades']} historical entered trades.\n\n"
        f"DATA COVERAGE (fraction of trades with non-null filter input):\n{cov_lines}\n\n"
        f"RESULTS HISTORY (showing best configs for this mode):\n"
        + "\n".join(history_lines)
        + f"\n\n{goal_line} "
        f"Return {n_suggestions} new filter configs as a JSON array."
    )

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": user_msg}],
    )

    text  = response.content[0].text.strip()
    start = text.find("[")
    end   = text.rfind("]") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON array in response: {text[:200]}")
    parsed = json.loads(text[start:end])

    valid = []
    for obj in parsed:
        if not isinstance(obj, dict):
            continue
        v = validate_filters(obj)
        if v:
            valid.append(v)
    return valid[:n_suggestions]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize hard entry filters via backtest")
    parser.add_argument("--strategy", default="all",
                        choices=list(STRATEGY_CONFIGS) + ["all"])
    parser.add_argument("--rounds",           type=int, default=5)
    parser.add_argument("--suggestions",      type=int, default=10)
    parser.add_argument("--random-per-round", type=int, default=8)
    parser.add_argument("--mode", default="balanced",
                        choices=["balanced", "strict", "lenient"],
                        help="Optimization goal: balanced=max passed_pnl, "
                             "strict=max win_rate, lenient=zero moonshot misses")
    args = parser.parse_args()

    strategies = list(STRATEGY_CONFIGS) if args.strategy == "all" else [args.strategy]

    client = anthropic.Anthropic()

    for strategy_name in strategies:
        meta         = STRATEGY_CONFIGS[strategy_name]
        heavy_thr    = meta["heavy_loss_threshold"]
        moonshot_thr = meta["moonshot_threshold"]

        print(f"\n{'='*72}")
        print(f"Strategy: {strategy_name}  (db_strategy={meta['db_strategy']})")
        print(f"{'='*72}")

        trades = load_trades(DB_PATH, meta["db_strategy"])
        if not trades:
            print(f"  No closed entered trades found.")
            continue

        n = len(trades)
        n_hc  = sum(1 for t in trades if t["holder_count"]     is not None)
        n_pc  = sum(1 for t in trades if t["price_chg_30m"]    is not None)
        n_pr  = sum(1 for t in trades if t["pump_ratio"]       is not None)
        n_bvr = sum(1 for t in trades if t["buy_vol_ratio_1h"] is not None)
        n_mc  = sum(1 for t in trades if t["market_cap_usd"]   is not None)
        data_coverage = {
            "holder_count":     n_hc  / n,
            "price_chg_30m":    n_pc  / n,
            "pump_ratio":       n_pr  / n,
            "buy_vol_ratio_1h": n_bvr / n,
            "market_cap_usd":   n_mc  / n,
        }
        print(f"  {n} trades | mode={args.mode}")
        print(f"  Data coverage: hc={n_hc}/{n} ({n_hc/n*100:.0f}%)  "
              f"pc30={n_pc}/{n} ({n_pc/n*100:.0f}%)  pr={n_pr}/{n} ({n_pr/n*100:.0f}%)  "
              f"bvr={n_bvr}/{n} ({n_bvr/n*100:.0f}%)  mc={n_mc}/{n} ({n_mc/n*100:.0f}%)")
        if any(v < 0.10 for v in data_coverage.values()):
            sparse = [k for k, v in data_coverage.items() if v < 0.10]
            print(f"  [WARN] Sparse data (<10%): {sparse} — filters on these may overfit noise")
        print(f"  heavy_loss_threshold={heavy_thr}%  moonshot_threshold={moonshot_thr}%\n")

        results_history: list[dict] = []

        def record(cfg: dict, label: str) -> dict:
            metrics = evaluate_filters(trades, cfg, heavy_thr, moonshot_thr)
            result  = {"label": label, "config": cfg, **metrics}
            results_history.append(result)
            hc  = cfg["holder_count_max"]
            pc  = cfg["late_entry_price_chg_30m_max"]
            pr  = cfg["late_entry_pump_ratio_min"]
            win_flag = " >50%!" if result['win_rate_passed'] > 0.50 else ""
            print(
                f"  [{label:13s}] passed=${result['passed_pnl']:+7.2f}  "
                f"net=${result['net_pnl_delta']:+7.2f}  "
                f"win={result['win_rate_passed']*100:4.1f}%{win_flag}  "
                f"avd=${result['losses_avoided']:6.2f}  "
                f"frg=${result['profits_foregone']:6.2f}  "
                f"hvy={result['heavy_blocked']:3d}  "
                f"mss={result['moonshots_missed']:2d}  "
                f"blk={result['block_rate']*100:4.1f}%  "
                f"hc={hc}  pc30={pc}  pr={pr}"
            )
            return result

        # Baseline: no filters
        record({"holder_count_max": None, "late_entry_price_chg_30m_max": None,
                "late_entry_pump_ratio_min": None}, "no_filters")
        # Baseline: current production filters
        record(CURRENT_FILTERS, "production")

        for round_num in range(1, args.rounds + 1):
            print(f"\n  --- Round {round_num}/{args.rounds} ---")

            # Random exploration (coverage-aware)
            for i in range(args.random_per_round):
                record(random_filters(data_coverage), f"rnd{round_num}_{i+1}")

            # AI-guided suggestions
            try:
                suggestions = ask_claude(
                    client, results_history, args.suggestions, strategy_name,
                    mode=args.mode, data_coverage=data_coverage,
                )
                for i, cfg in enumerate(suggestions):
                    record(cfg, f"ai{round_num}_{i+1}")
            except Exception as e:
                print(f"  [Claude error: {e}]")

        # Final summary — sort order depends on mode
        if args.mode == "strict":
            top10 = sorted(results_history,
                           key=lambda r: (-r["win_rate_passed"], -r["passed_pnl"]))[:10]
            sort_desc = "win_rate_passed desc"
        elif args.mode == "lenient":
            top10 = sorted(results_history,
                           key=lambda r: (r["moonshots_missed"], -r["heavy_blocked"], -r["passed_pnl"]))[:10]
            sort_desc = "moonshots_missed asc, heavy_blocked desc"
        else:
            top10 = sorted(results_history, key=lambda r: -r["passed_pnl"])[:10]
            sort_desc = "passed_pnl desc"

        print(f"\n  === Top 10 filter configs for {strategy_name} [mode={args.mode}, sorted by {sort_desc}] ===")
        for rank, r in enumerate(top10, 1):
            cfg = r["config"]
            win_flag = "*" if r['win_rate_passed'] > 0.50 else " "
            print(
                f"  #{rank:2d} [{r['label']:13s}] passed=${r['passed_pnl']:+7.2f}  "
                f"net=${r['net_pnl_delta']:+7.2f}  "
                f"win={r['win_rate_passed']*100:4.1f}%{win_flag}  "
                f"avd=${r['losses_avoided']:6.2f}  frg=${r['profits_foregone']:6.2f}  "
                f"hvy={r['heavy_blocked']:3d}  mss={r['moonshots_missed']:2d}  "
                f"blk={r['block_rate']*100:4.1f}%  "
                f"hc={cfg['holder_count_max']}  "
                f"pc30={cfg['late_entry_price_chg_30m_max']}  "
                f"pr={cfg['late_entry_pump_ratio_min']}  "
                f"bvr={cfg.get('buy_vol_ratio_1h_max')}  "
                f"mc={cfg.get('market_cap_usd_min')}"
            )
        if top10:
            best = top10[0]
            print(f"\n  Best config for {strategy_name} [mode={args.mode}]:")
            print(f"  {json.dumps(best['config'], indent=4)}")


if __name__ == "__main__":
    main()
