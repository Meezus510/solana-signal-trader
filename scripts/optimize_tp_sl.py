#!/usr/bin/env python3
"""
scripts/optimize_tp_sl.py — AI-driven stop-ladder / take-profit optimizer for moonbag.

Uses Claude to iteratively suggest stop-ladder + TP configurations, evaluates each
via simulation on historical closed trades, and converges on the most profitable setup.

How the simulation works:
  Each closed trade has a recorded peak price (highest reached) and trough price
  (lowest reached), with timestamps for each.  From these two points we infer the
  approximate price path:

    peak_first (peak_ts <= trough_ts):
        Price rose to peak first, then fell to trough.
        → TPs fire on the way up.
        → Stop ladder fires at its current floor on the way down.
        → If trough never reached the floor, exit at trough (e.g. timeout).

    trough_first (trough_ts < peak_ts):
        Price fell to trough first, then recovered to peak.
        → Initial floor fires if trough is below it → stop out immediately.
        → If survived: TPs fire on recovery.
        → Remaining position exits at the stop floor for the peak reached.

  This is an approximation: we do not have tick-level data, so intrabar retracements
  are not modelled.  The simulation is conservative for trough-first trades (assumes
  the remaining position exits at the stop floor after the peak, not above it).

Usage:
    python scripts/optimize_tp_sl.py [--strategy moonbag] [--rounds 6]
                                     [--suggestions 10] [--random-per-round 8]
                                     [--buy-size 5.0]
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

DB_PATH  = "trader.db"
BUY_SIZE = 5.0   # USD position size (override with --buy-size)

# Current production config — used as baseline seed
PRODUCTION_CONFIG = {
    "initial_floor_pct":  20.0,                        # post-grace floor (%)
    "stop_milestones":    [[2.5, 1.65], [4.0, 2.60], [6.0, 4.20]],  # (highest_mult, stop_floor_mult)
    "tp_levels":          [[1.8, 0.20], [2.5, 0.15], [4.0, 0.15], [6.0, 0.10]],  # (mult, frac_of_original)
}

# quick_pop uses trailing stop + fixed TPs (no stop ladder)
QP_PRODUCTION_CONFIG = {
    "stop_loss_pct":     0.06,   # initial hard stop below entry
    "trailing_stop_pct": 0.07,   # trail below highest price reached (activates after TP1)
    "tp_levels":         [[1.26, 0.78], [1.98, 0.22]],  # (mult, frac_of_original)
}

STRATEGY_CONFIGS = {
    "moonbag": {
        "db_strategy":    "infinite_moonbag",
        "registry_name":  "moonbag_managed",
        "config_type":    "moonbag",
        "production_cfg": PRODUCTION_CONFIG,
        "default_buy_size": 5.0,
    },
    "quick_pop": {
        "db_strategy":    "quick_pop",
        "registry_name":  "quick_pop_managed",
        "config_type":    "quick_pop",
        "production_cfg": QP_PRODUCTION_CONFIG,
        "default_buy_size": 30.0,
    },
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_trades(db_path: str, db_strategy: str) -> list[dict]:
    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT
            so.outcome_pnl_pct,
            so.outcome_pnl_usd,
            so.outcome_sell_reason,
            so.outcome_hold_secs,
            so.position_peak_pnl_pct,
            so.position_peak_ts,
            so.position_trough_pnl_pct,
            so.position_trough_ts,
            sc.ts AS signal_ts
        FROM strategy_outcomes so
        JOIN signal_charts sc ON sc.id = so.signal_chart_id
        WHERE so.strategy     = ?
          AND so.closed       = 1
          AND so.entered      = 1
          AND so.position_peak_pnl_pct   IS NOT NULL
          AND so.position_trough_pnl_pct IS NOT NULL
          AND so.position_peak_ts        IS NOT NULL
          AND so.position_trough_ts      IS NOT NULL
        ORDER BY sc.ts
    """, (db_strategy,)).fetchall()
    conn.close()

    trades = []
    for row in rows:
        (out_pnl, out_usd, sell_reason, hold_secs,
         peak_pct, peak_ts, trough_pct, trough_ts, signal_ts) = row
        trades.append({
            "outcome_pnl_pct":    out_pnl or 0.0,
            "outcome_pnl_usd":    out_usd or 0.0,
            "outcome_sell_reason":sell_reason or "",
            "outcome_hold_secs":  hold_secs or 0.0,
            "peak_pnl_pct":       peak_pct,
            "peak_ts":            peak_ts,
            "trough_pnl_pct":     trough_pct,
            "trough_ts":          trough_ts,
            "signal_ts":          signal_ts,
        })
    return trades


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def validate_config(cfg: dict) -> dict | None:
    """Clamp and validate a config.  Returns None if fatally invalid."""
    try:
        floor = max(5.0, min(50.0, float(cfg["initial_floor_pct"])))

        milestones = []
        for m in cfg.get("stop_milestones", []):
            if not isinstance(m, list) or len(m) != 2:
                continue
            mult  = max(1.1, min(20.0, float(m[0])))
            floor_mult = max(1 - floor / 100 + 0.01, min(20.0, float(m[1])))
            milestones.append([mult, floor_mult])
        milestones.sort(key=lambda x: x[0])
        # Enforce monotonicity: higher milestones must lock in higher floors
        clean_milestones = []
        running_floor = 0.0
        for mult, lock in milestones:
            if lock > running_floor:
                clean_milestones.append([mult, lock])
                running_floor = lock

        tp_levels = []
        total_frac = 0.0
        for t in cfg.get("tp_levels", []):
            if not isinstance(t, list) or len(t) != 2:
                continue
            mult = max(1.01, min(30.0, float(t[0])))
            frac = max(0.05, min(0.50, float(t[1])))
            if total_frac + frac > 1.0:
                frac = round(1.0 - total_frac, 2)
            if frac <= 0:
                break
            tp_levels.append([mult, frac])
            total_frac += frac
        tp_levels.sort(key=lambda x: x[0])

        if not tp_levels:
            return None

        return {
            "initial_floor_pct": floor,
            "stop_milestones":   clean_milestones,
            "tp_levels":         tp_levels,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Trade simulation
# ---------------------------------------------------------------------------

def _get_stop_floor(highest_mult: float, initial_floor_mult: float,
                    milestones: list[list[float]]) -> float:
    """Return the current stop floor (as a multiple of entry) for a given highest_mult."""
    floor = initial_floor_mult
    for milestone_mult, lock_mult in milestones:
        if highest_mult >= milestone_mult:
            floor = max(floor, lock_mult)
    return floor


def simulate_trade(trade: dict, config: dict, buy_size: float = 5.0) -> float:
    """
    Simulate one trade under `config`.  Returns simulated PnL in USD.

    peak_first  → price rose to peak_pct first, then fell to trough_pct.
    trough_first → price fell to trough_pct first, then recovered to peak_pct.
    """
    peak_mult   = 1 + trade["peak_pnl_pct"]   / 100
    trough_mult = 1 + trade["trough_pnl_pct"] / 100

    # Determine ordering
    peak_first = trade["peak_ts"] <= trade["trough_ts"]

    floor_pct        = config["initial_floor_pct"]
    initial_floor_mult = 1 - floor_pct / 100          # e.g. 0.80 for 20%
    milestones       = config["stop_milestones"]       # [[mult, floor_mult], ...]
    tp_levels        = sorted(config["tp_levels"], key=lambda x: x[0])  # [[mult, frac], ...]

    remaining = 1.0    # fraction of original position still held
    realized  = 0.0    # USD profit/loss (relative to buy_size entry)

    def apply_tps(up_to_mult: float) -> None:
        nonlocal remaining, realized
        for tp_mult, tp_frac in tp_levels:
            if up_to_mult < tp_mult:
                break
            if remaining <= 0:
                break
            sell_frac = min(tp_frac, remaining)
            realized += sell_frac * buy_size * (tp_mult - 1)
            remaining -= sell_frac

    if peak_first:
        # 1. TPs fire on the way up to peak
        apply_tps(peak_mult)

        if remaining > 0:
            # 2. Stop floor at the peak reached
            stop_floor = _get_stop_floor(peak_mult, initial_floor_mult, milestones)
            if trough_mult <= stop_floor:
                # Stop fires at floor
                realized += remaining * buy_size * (stop_floor - 1)
            else:
                # Price didn't drop to floor — exit at trough (timeout / continued hold)
                realized += remaining * buy_size * (trough_mult - 1)
    else:
        # trough_first: price fell to trough first
        if trough_mult <= initial_floor_mult:
            # Initial floor fires immediately
            realized = remaining * buy_size * (initial_floor_mult - 1)
        else:
            # Survived the dip; price recovers to peak
            apply_tps(peak_mult)

            if remaining > 0:
                # After recovery to peak, stop floor is set by peak milestone.
                # Conservative assumption: remaining exits at stop floor.
                stop_floor = _get_stop_floor(peak_mult, initial_floor_mult, milestones)
                realized += remaining * buy_size * (stop_floor - 1)

    return realized


# ---------------------------------------------------------------------------
# Config evaluation
# ---------------------------------------------------------------------------

def evaluate_config(trades: list[dict], config: dict, buy_size: float) -> dict:
    """Evaluate a config across all trades.  Returns metrics dict."""
    total_pnl    = 0.0
    wins         = 0
    tp_hits      = [0] * len(config["tp_levels"])
    stop_fires   = 0

    for trade in trades:
        pnl = simulate_trade(trade, config, buy_size)
        total_pnl += pnl
        if pnl > 0:
            wins += 1
        # Count TP hits
        peak_mult = 1 + trade["peak_pnl_pct"] / 100
        for i, (tp_mult, _) in enumerate(sorted(config["tp_levels"], key=lambda x: x[0])):
            if peak_mult >= tp_mult:
                tp_hits[i] += 1
        # Count stop fires (trough hits floor)
        trough_mult = 1 + trade["trough_pnl_pct"] / 100
        floor_mult  = 1 - config["initial_floor_pct"] / 100
        stop_floor  = _get_stop_floor(
            1 + trade["peak_pnl_pct"] / 100, floor_mult, config["stop_milestones"]
        )
        if trough_mult <= stop_floor:
            stop_fires += 1

    n = len(trades)
    return {
        "total_pnl_usd":   total_pnl,
        "win_rate":        wins / n if n else 0.0,
        "avg_pnl_usd":     total_pnl / n if n else 0.0,
        "stop_fire_rate":  stop_fires / n if n else 0.0,
        "tp_hit_counts":   tp_hits,
        "n_trades":        n,
    }


# ---------------------------------------------------------------------------
# Random config generation
# ---------------------------------------------------------------------------

def random_config() -> dict:
    """Generate a random TP/SL config within reasonable bounds."""
    floor = round(random.uniform(10.0, 35.0), 1)

    # Random stop milestones (1–3 rungs)
    n_milestones = random.randint(1, 3)
    milestones = []
    last_mult  = 1.5
    last_lock  = 1.0
    for _ in range(n_milestones):
        mult = round(random.uniform(last_mult + 0.3, last_mult + 3.0), 1)
        lock = round(random.uniform(last_lock + 0.1, last_lock + 2.0), 2)
        milestones.append([mult, lock])
        last_mult = mult
        last_lock = lock

    # Random TP levels (2–5 rungs)
    n_tp = random.randint(2, 5)
    tp_levels   = []
    total_frac  = 0.0
    last_tp_mult = 1.1
    for i in range(n_tp):
        mult = round(random.uniform(last_tp_mult + 0.2, last_tp_mult + 3.0), 1)
        max_frac = min(0.40, 1.0 - total_frac - 0.05 * (n_tp - i - 1))
        if max_frac <= 0.05:
            break
        frac = round(random.uniform(0.05, max_frac), 2)
        tp_levels.append([mult, frac])
        total_frac  += frac
        last_tp_mult = mult

    cfg = {
        "initial_floor_pct": floor,
        "stop_milestones":   milestones,
        "tp_levels":         tp_levels,
    }
    return validate_config(cfg) or PRODUCTION_CONFIG


# ---------------------------------------------------------------------------
# quick_pop: validate / simulate / random
# ---------------------------------------------------------------------------

def validate_config_qp(cfg: dict) -> dict | None:
    """Validate a quick_pop config. Returns None if fatally invalid."""
    try:
        sl  = max(0.05, min(0.40, float(cfg["stop_loss_pct"])))
        trl = max(0.05, min(0.45, float(cfg["trailing_stop_pct"])))

        tp_levels = []
        total_frac = 0.0
        for t in cfg.get("tp_levels", []):
            if not isinstance(t, list) or len(t) != 2:
                continue
            mult = max(1.01, min(10.0, float(t[0])))
            frac = max(0.05, min(0.95, float(t[1])))
            if total_frac + frac > 1.0:
                frac = round(1.0 - total_frac, 2)
            if frac <= 0:
                break
            tp_levels.append([mult, frac])
            total_frac += frac
        tp_levels.sort(key=lambda x: x[0])

        if not tp_levels:
            return None

        return {
            "stop_loss_pct":     sl,
            "trailing_stop_pct": trl,
            "tp_levels":         tp_levels,
        }
    except Exception:
        return None


def simulate_trade_qp(trade: dict, config: dict, buy_size: float = 30.0) -> float:
    """
    Simulate one quick_pop trade under `config`.  Returns simulated PnL in USD.

    Matches live strategy.py behaviour exactly:
      - Hard stop (stop_loss_pct) is always active.
      - Trailing stop only activates after TP1 fires, then trails highest * (1 - trail_pct).
      - Max 2 TP levels (position model only has partial_take_profit_hit + tp2_hit flags).

    peak_first  → price rose to peak first, then fell to trough.
    trough_first → price fell to trough first, then recovered to peak.
    """
    peak_mult   = 1 + trade["peak_pnl_pct"]   / 100
    trough_mult = 1 + trade["trough_pnl_pct"] / 100
    peak_first  = trade["peak_ts"] <= trade["trough_ts"]

    sl_pct    = config["stop_loss_pct"]
    trail_pct = config["trailing_stop_pct"]
    tp_levels = sorted(config["tp_levels"], key=lambda x: x[0])[:2]  # max 2 TPs

    hard_floor = 1 - sl_pct   # e.g. 0.80 for 20% SL

    remaining  = 1.0
    realized   = 0.0
    tp1_fired  = False

    def apply_tps(up_to_mult: float) -> None:
        nonlocal remaining, realized, tp1_fired
        for i, (tp_mult, tp_frac) in enumerate(tp_levels):
            if up_to_mult < tp_mult or remaining <= 0:
                break
            sell_frac = min(tp_frac, remaining)
            realized += sell_frac * buy_size * (tp_mult - 1)
            remaining -= sell_frac
            if i == 0:
                tp1_fired = True

    def current_stop(highest_mult: float) -> float:
        """After TP1: trail from highest. Before TP1: hard floor only."""
        if tp1_fired:
            return max(hard_floor, highest_mult * (1 - trail_pct))
        return hard_floor

    if peak_first:
        # 1. TPs fire on the way up to peak
        apply_tps(peak_mult)

        if remaining > 0:
            # 2. Stop check on the way down from peak
            stop = current_stop(peak_mult)
            if trough_mult <= stop:
                realized += remaining * buy_size * (stop - 1)
            else:
                # Didn't reach stop — exit at trough (timeout / continued hold)
                realized += remaining * buy_size * (trough_mult - 1)
    else:
        # trough_first: price fell before peak
        if trough_mult <= hard_floor:
            # Hard stop fires immediately (no TP1 yet → trailing not active)
            realized = buy_size * (hard_floor - 1)
        else:
            # Survived dip; price recovers to peak
            apply_tps(peak_mult)

            if remaining > 0:
                # After peak, stop is trailing (if TP1 fired) or hard floor
                stop = current_stop(peak_mult)
                realized += remaining * buy_size * (stop - 1)

    return realized


def evaluate_config_qp(trades: list[dict], config: dict, buy_size: float) -> dict:
    """Evaluate a quick_pop config across all trades."""
    total_pnl  = 0.0
    wins       = 0
    tp_hits    = [0] * len(config["tp_levels"])
    stop_fires = 0

    for trade in trades:
        pnl = simulate_trade_qp(trade, config, buy_size)
        total_pnl += pnl
        if pnl > 0:
            wins += 1
        peak_mult   = 1 + trade["peak_pnl_pct"]   / 100
        trough_mult = 1 + trade["trough_pnl_pct"] / 100
        for i, (tp_mult, _) in enumerate(sorted(config["tp_levels"], key=lambda x: x[0])):
            if peak_mult >= tp_mult:
                tp_hits[i] += 1
        floor_mult = max(1 - config["stop_loss_pct"],
                         peak_mult * (1 - config["trailing_stop_pct"]))
        if trough_mult <= floor_mult:
            stop_fires += 1

    n = len(trades)
    return {
        "total_pnl_usd":  total_pnl,
        "win_rate":       wins / n if n else 0.0,
        "avg_pnl_usd":    total_pnl / n if n else 0.0,
        "stop_fire_rate": stop_fires / n if n else 0.0,
        "tp_hit_counts":  tp_hits,
        "n_trades":       n,
    }


def random_config_qp() -> dict:
    """Generate a random quick_pop TP/SL config within reasonable bounds."""
    sl  = round(random.uniform(0.10, 0.35), 2)
    trl = round(random.uniform(0.10, 0.38), 2)

    n_tp = random.randint(1, 2)
    tp_levels  = []
    total_frac = 0.0
    last_mult  = 1.1
    for i in range(n_tp):
        mult = round(random.uniform(last_mult + 0.1, last_mult + 1.5), 1)
        max_frac = min(0.90, 1.0 - total_frac - 0.05 * (n_tp - i - 1))
        if max_frac <= 0.05:
            break
        frac = round(random.uniform(0.10, max_frac), 2)
        tp_levels.append([mult, frac])
        total_frac += frac
        last_mult = mult

    cfg = {
        "stop_loss_pct":     sl,
        "trailing_stop_pct": trl,
        "tp_levels":         tp_levels,
    }
    return validate_config_qp(cfg) or QP_PRODUCTION_CONFIG


# ---------------------------------------------------------------------------
# Claude suggestion
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_MOONBAG = """You are an expert optimizer for a cryptocurrency moonbag trading strategy.

The strategy buys a token and holds it using a stop ladder and partial take-profits:
  - initial_floor_pct: if price falls below (entry × (1 - floor/100)), close the position.
  - stop_milestones: once the highest price reaches a multiple, the stop floor is raised
    to lock in gains. Format: [[highest_mult, stop_floor_mult], ...] sorted ascending.
    Example: [[2.5, 1.65], [4.0, 2.60]] means if price ever hits 2.5×, stop floor moves
    to 1.65× entry (+65% locked). Milestones must have monotonically increasing floors.
  - tp_levels: partial sells at fixed multiples. Format: [[multiple, fraction_of_original], ...]
    Fractions are of the ORIGINAL position, not remaining. Must sum to ≤ 1.0.

Simulation model:
  - Each trade has a recorded peak price and trough price (lowest).
  - If peak came first: TPs fire on way up, stop fires on way down.
  - If trough came first: initial floor may fire immediately; otherwise price recovers to peak
    and TPs fire, then remaining exits at stop floor for the peak reached.

PRIMARY metric: total_pnl_usd (sum of simulated PnL across all trades at $BUY_SIZE entry).
SECONDARY: win_rate (fraction of trades with positive simulated PnL).

Rules:
- Return ONLY a JSON array of config objects, no explanation.
- Each config has exactly three keys: "initial_floor_pct", "stop_milestones", "tp_levels".
- initial_floor_pct range: 5.0 to 40.0
- stop_milestones: 0–4 rungs, each [highest_mult, floor_mult], sorted ascending by highest_mult.
  floor_mult must be > 1.0 (locking in gains). Rungs must have increasing floor_mult values.
- tp_levels: 1–5 rungs, each [multiple, fraction_of_original], sorted ascending by multiple.
  Fractions sum to ≤ 1.0. Multiple > 1.0.
- DIVERSITY RULE: at least half your suggestions must differ significantly from current best
  (change floor by >5%, add/remove a milestone, change TP multiples by >0.5).
- Return exactly the number of suggestions requested."""

SYSTEM_PROMPT_QUICK_POP = """You are an expert optimizer for a cryptocurrency quick-pop scalp strategy.

The strategy buys a token and uses a trailing stop + partial take-profits:
  - stop_loss_pct: hard stop. If price falls below entry × (1 - stop_loss_pct), close position.
    Range: 0.05 to 0.40 (e.g. 0.20 = 20% stop below entry).
  - trailing_stop_pct: once TPs fire, remaining position trails this % below the highest price seen.
    The effective stop = max(initial_hard_stop, highest × (1 - trailing_stop_pct)).
    Range: 0.05 to 0.45.
  - tp_levels: partial sells at fixed multiples. Format: [[multiple, fraction_of_original], ...]
    Fractions are of the ORIGINAL position. Must sum to ≤ 1.0. Multiple > 1.0.

Simulation model (approximation from peak/trough data):
  - If peak came first: TPs fire on way up to peak; trailing stop fires if trough ≤ trailing floor.
  - If trough came first: hard stop fires if trough ≤ initial floor; otherwise price recovers to
    peak, TPs fire, remaining exits at trailing stop from peak.

Context: quick_pop is a short-term scalp (45-min timeout). Most winners pump 1.5–3× quickly.
Most losers dump immediately. Typical peak_pnl ≈ +20–200% for winners, trough ≈ -15–30% for losers.

PRIMARY metric: total_pnl_usd (sum of simulated PnL across all trades at $30 buy size).
SECONDARY: win_rate.

Rules:
- Return ONLY a JSON array of config objects, no explanation.
- Each config has exactly three keys: "stop_loss_pct", "trailing_stop_pct", "tp_levels".
- stop_loss_pct: float 0.05–0.40
- trailing_stop_pct: float 0.05–0.45
- tp_levels: 1–2 rungs, each [multiple, fraction_of_original], sorted ascending.
  Fractions sum to ≤ 1.0. Multiple > 1.0. MAX 2 levels (position model limit).
- DIVERSITY RULE: at least half your suggestions must differ significantly from the current best.
- Return exactly the number of suggestions requested."""


def ask_claude(client: anthropic.Anthropic, results_history: list[dict],
               n_suggestions: int, buy_size: float, config_type: str = "moonbag") -> list[dict]:
    """Ask Claude for n_suggestions new configs.  Returns validated config list."""
    history_lines = []
    for i, r in enumerate(results_history):
        cfg = r["config"]
        if config_type == "quick_pop":
            history_lines.append(
                f"  Config {i+1}: total_pnl=${r['total_pnl_usd']:+.2f}  "
                f"win_rate={r['win_rate']*100:.0f}%  "
                f"avg_pnl=${r['avg_pnl_usd']:+.3f}  "
                f"stop_fire_rate={r['stop_fire_rate']*100:.0f}%  "
                f"tp_hits={r['tp_hit_counts']}  "
                f"stop_loss={cfg['stop_loss_pct']}  "
                f"trailing_stop={cfg['trailing_stop_pct']}  "
                f"tp_levels={json.dumps(cfg['tp_levels'])}"
            )
        else:
            history_lines.append(
                f"  Config {i+1}: total_pnl=${r['total_pnl_usd']:+.2f}  "
                f"win_rate={r['win_rate']*100:.0f}%  "
                f"avg_pnl=${r['avg_pnl_usd']:+.3f}  "
                f"stop_fire_rate={r['stop_fire_rate']*100:.0f}%  "
                f"tp_hits={r['tp_hit_counts']}  "
                f"floor={cfg['initial_floor_pct']}%  "
                f"milestones={json.dumps(cfg['stop_milestones'])}  "
                f"tp_levels={json.dumps(cfg['tp_levels'])}"
            )

    system_prompt = SYSTEM_PROMPT_QUICK_POP if config_type == "quick_pop" else SYSTEM_PROMPT_MOONBAG
    user_msg = (
        f"Dataset: {results_history[0]['n_trades']} trades.  "
        f"Buy size: ${buy_size:.2f} per trade.\n\n"
        f"RESULTS HISTORY (all configs tested so far):\n"
        + "\n".join(history_lines)
        + f"\n\nSuggest {n_suggestions} new configs to maximise total_pnl_usd. "
        f"Return only a JSON array."
    )

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_msg}],
    )

    text  = response.content[0].text.strip()
    start = text.find("[")
    end   = text.rfind("]") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON array in response: {text[:200]}")
    parsed = json.loads(text[start:end])

    validate_fn = validate_config_qp if config_type == "quick_pop" else validate_config
    valid = []
    for obj in parsed:
        if not isinstance(obj, dict):
            continue
        v = validate_fn(obj)
        if v:
            valid.append(v)
    return valid[:n_suggestions]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize TP/SL via simulation")
    parser.add_argument("--strategy",         default="moonbag",
                        choices=list(STRATEGY_CONFIGS))
    parser.add_argument("--rounds",       type=int,   default=6)
    parser.add_argument("--suggestions",  type=int,   default=10)
    parser.add_argument("--random-per-round", type=int, default=8)
    parser.add_argument("--buy-size",     type=float, default=None)
    args = parser.parse_args()

    cfg_meta    = STRATEGY_CONFIGS[args.strategy]
    config_type = cfg_meta["config_type"]
    buy_size    = args.buy_size if args.buy_size is not None else cfg_meta["default_buy_size"]
    prod_cfg    = cfg_meta["production_cfg"]

    # Dispatch to strategy-specific helpers
    if config_type == "quick_pop":
        eval_fn   = evaluate_config_qp
        rand_fn   = random_config_qp
    else:
        eval_fn   = evaluate_config
        rand_fn   = random_config

    print(f"Loading trades for {cfg_meta['db_strategy']}...")
    trades = load_trades(DB_PATH, cfg_meta["db_strategy"])
    if not trades:
        print("No closed trades found.")
        sys.exit(1)

    n_total   = len(trades)
    n_winners = sum(1 for t in trades if t["outcome_pnl_pct"] > 0)
    peak_first_count = sum(1 for t in trades if t["peak_ts"] <= t["trough_ts"])
    print(f"  {n_total} trades | {n_winners} actual winners | "
          f"{peak_first_count} peak-first | {n_total - peak_first_count} trough-first")
    print(f"  Buy size: ${buy_size:.2f}")

    results_history: list[dict] = []

    def record(config: dict, label: str) -> dict:
        metrics = eval_fn(trades, config, buy_size)
        result = {"label": label, "config": config, **metrics}
        tp_hits_str = "/".join(str(x) for x in metrics["tp_hit_counts"])
        if config_type == "quick_pop":
            cfg_str = (f"sl={config['stop_loss_pct']}  "
                       f"trail={config['trailing_stop_pct']}  "
                       f"tp={json.dumps(config['tp_levels'])}")
        else:
            cfg_str = (f"floor={config['initial_floor_pct']:.0f}%  "
                       f"milestones={json.dumps(config['stop_milestones'])}  "
                       f"tp={json.dumps(config['tp_levels'])}")
        print(
            f"  [{label:<18}]  pnl=${metrics['total_pnl_usd']:>+8.2f}  "
            f"win={metrics['win_rate']*100:>4.0f}%  "
            f"stop_fire={metrics['stop_fire_rate']*100:>4.0f}%  "
            f"tp_hits={tp_hits_str}  {cfg_str}"
        )
        results_history.append(result)
        return result

    # --- Seed: production config ---
    print("\nRound 0: Evaluating production config...")
    record(prod_cfg, "production")

    client = anthropic.Anthropic()

    for round_num in range(1, args.rounds + 1):
        print(f"\nRound {round_num}: Asking Claude for {args.suggestions} configs...")
        try:
            suggestions = ask_claude(client, results_history, args.suggestions,
                                     buy_size, config_type=config_type)
        except Exception as exc:
            print(f"  Claude error: {exc} — skipping round")
            continue

        print(f"  Received {len(suggestions)} configs. Evaluating...")
        for i, cfg in enumerate(suggestions):
            record(cfg, f"r{round_num}_s{i+1}")

        if args.random_per_round > 0:
            print(f"  Injecting {args.random_per_round} random configs...")
            for i in range(args.random_per_round):
                record(rand_fn(), f"r{round_num}_rand{i+1}")

    # --- Results ---
    results_history.sort(key=lambda r: (r["total_pnl_usd"], r["win_rate"]), reverse=True)

    print("\n" + "=" * 90)
    print("  OPTIMIZATION RESULTS — TOP 10  (sorted by total simulated PnL)")
    print("=" * 90)
    if config_type == "quick_pop":
        print(f"  {'Label':<20} {'TotalPnL':>10} {'WinRate':>8} {'StopFire':>9} {'TP hits':>12}  SL    Trail  TPs")
    else:
        print(f"  {'Label':<20} {'TotalPnL':>10} {'WinRate':>8} {'StopFire':>9} {'TP hits':>20}  Floor")
    print("  " + "-" * 90)
    for r in results_history[:10]:
        tp_str = "/".join(str(x) for x in r["tp_hit_counts"])
        if config_type == "quick_pop":
            print(
                f"  {r['label']:<20} ${r['total_pnl_usd']:>+8.2f}  "
                f"{r['win_rate']*100:>6.1f}%  "
                f"{r['stop_fire_rate']*100:>7.1f}%  "
                f"{tp_str:>12}  "
                f"{r['config']['stop_loss_pct']:.2f}  "
                f"{r['config']['trailing_stop_pct']:.2f}  "
                f"{json.dumps(r['config']['tp_levels'])}"
            )
        else:
            print(
                f"  {r['label']:<20} ${r['total_pnl_usd']:>+8.2f}  "
                f"{r['win_rate']*100:>6.1f}%  "
                f"{r['stop_fire_rate']*100:>7.1f}%  "
                f"{tp_str:>20}  "
                f"{r['config']['initial_floor_pct']:.0f}%"
            )

    best = results_history[0]
    bc   = best["config"]
    prod = next(r for r in results_history if r["label"] == "production")

    print("\n" + "=" * 90)
    print("  BEST CONFIGURATION")
    print("=" * 90)
    print(f"  Label:           {best['label']}")
    print(f"  Total PnL:       ${best['total_pnl_usd']:+.2f}  "
          f"(avg ${best['avg_pnl_usd']:+.4f}/trade, win rate {best['win_rate']*100:.1f}%)")
    print(f"  Stop fire rate:  {best['stop_fire_rate']*100:.1f}%")
    print(f"  TP hits:         {best['tp_hit_counts']}")
    print()
    if config_type == "quick_pop":
        print(f"  stop_loss_pct      = {bc['stop_loss_pct']}")
        print(f"  trailing_stop_pct  = {bc['trailing_stop_pct']}")
        print(f"  tp_levels          = {json.dumps(bc['tp_levels'])}")
        print()
        print("To apply: update registry.py quick_pop / quick_pop_managed StrategyConfig.")
    else:
        print(f"  initial_floor_pct  = {bc['initial_floor_pct']}")
        print(f"  stop_milestones    = {json.dumps(bc['stop_milestones'])}")
        print(f"  tp_levels          = {json.dumps(bc['tp_levels'])}")
        print()
        print("To apply: update registry.py moonbag_managed config and strategy.py _STOP_MILESTONES.")
    print(f"  Production config: ${prod['total_pnl_usd']:+.2f}")
    print(f"  Best config:       ${best['total_pnl_usd']:+.2f}  "
          f"(delta ${best['total_pnl_usd'] - prod['total_pnl_usd']:+.2f})")


if __name__ == "__main__":
    main()
