import sqlite3, json

conn = sqlite3.connect("trader.db")
rows = conn.execute("""
    SELECT so.outcome_pnl_pct, so.outcome_pnl_usd, sc.pair_stats_json, sc.symbol
    FROM strategy_outcomes so
    JOIN signal_charts sc ON sc.id = so.signal_chart_id
    WHERE so.strategy = 'quick_pop_managed'
      AND so.closed = 1 AND so.entered = 1
      AND sc.ts >= datetime('now', '-7 days')
      AND sc.pair_stats_json IS NOT NULL
      AND so.outcome_pnl_usd IS NOT NULL
      AND so.outcome_pnl_pct IS NOT NULL
""").fetchall()
conn.close()

def tof(v):
    if v is None: return None
    try: return float(v)
    except: return None

trades = []
for pct, usd, ps_j, sym in rows:
    pct = tof(pct); usd = tof(usd)
    if pct is None or usd is None: continue
    ps  = json.loads(ps_j)
    liq = tof(ps.get("liquidity_usd"))
    trades.append({
        "pct": pct, "usd": usd, "sym": sym,
        "winner": usd > 0,
        "top10": tof(ps.get("top10_concentration")),
        "liq":   liq,
        "age":   tof(ps.get("security_token_age_hours")),
    })

def f_liq0(t):    return t["liq"] is not None and t["liq"] == 0
def f_top10(t):   return t["top10"] is not None and t["top10"] > 0.4
def f_no_data(t): return t["top10"] is None and t["age"] is None
def f_combo(t):   return f_liq0(t) or f_top10(t)
def f_all(t):     return f_liq0(t) or f_top10(t) or f_no_data(t)

filters = [
    ("liq == 0",                       f_liq0),
    ("top10 > 0.4",                    f_top10),
    ("no top10 AND no age",            f_no_data),
    ("liq==0 OR top10>0.4",            f_combo),
    ("liq==0 OR top10>0.4 OR no_data", f_all),
]

total_w    = sum(1 for t in trades if t["winner"])
total_wpnl = sum(t["usd"] for t in trades if t["winner"])
bt_total   = sum(1 for t in trades if t["pct"] < -30)
bt_pnl_tot = sum(t["usd"] for t in trades if t["pct"] < -30)

print(f"Total: {len(trades)}  winners: {total_w} (${total_wpnl:+.2f})  blow-throughs: {bt_total} (${bt_pnl_tot:+.2f})\n")
print(f"{'Filter':<42} {'W blocked':>10} {'W$ lost':>10} {'BT blocked':>11} {'BT$ saved':>10}  {'Net':>8}")
print("-" * 97)
for name, fn in filters:
    flagged   = [t for t in trades if fn(t)]
    w_blocked = [t for t in flagged if t["winner"]]
    bt_saved  = [t for t in flagged if t["pct"] < -30]
    w_pnl        = sum(t["usd"] for t in w_blocked)
    bt_saved_pnl = abs(sum(t["usd"] for t in bt_saved))
    net = bt_saved_pnl + w_pnl
    print(f"  {name:<40} {len(w_blocked):>4}/{total_w:<4}  ${w_pnl:>+8.2f}  "
          f"{len(bt_saved):>4}/{bt_total:<4}  ${bt_saved_pnl:>8.2f}  ${net:>+7.2f}")

print()
print("Winners blocked by top10 > 0.4:")
for t in sorted([t for t in trades if t["winner"] and f_top10(t)], key=lambda x: -x["usd"]):
    print(f"  {t['sym']:<12} +${t['usd']:>7.2f}  top10={t['top10']:.3f}  liq=${t['liq'] or 0:,.0f}")

print()
print("Winners blocked by liq == 0:")
for t in sorted([t for t in trades if t["winner"] and f_liq0(t)], key=lambda x: -x["usd"]):
    print(f"  {t['sym']:<12} +${t['usd']:>7.2f}  top10={round(t['top10'],3) if t['top10'] else 'N/A'}")
