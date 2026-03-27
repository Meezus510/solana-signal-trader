#!/usr/bin/env python3
import sqlite3, json, sys

conn = sqlite3.connect('trader.db')
# Get actual outcomes for anthropic_managed
rows = conn.execute('''
    SELECT so.outcome_pnl_usd, so.position_peak_pnl_pct, so.position_trough_pnl_pct,
           so.position_peak_ts, so.position_trough_ts
    FROM strategy_outcomes so
    WHERE so.strategy = 'anthropic_managed' 
      AND so.closed = 1 
      AND so.entered = 1
''').fetchall()
conn.close()

print(f"Found {len(rows)} closed anthropic_managed trades")
total_actual = sum(r[0] for r in rows if r[0] is not None)
print(f"Total actual PnL: ${total_actual:.2f}")
print("\nFirst 5 trades:")
for i, row in enumerate(rows[:5]):
    print(f"  {i+1}: Outcome: ${row[0]:.2f}, Peak: {row[1]:.1f}%, Trough: {row[2]:.1f}%")

# Now let's see what the backtest is simulating
print("\n--- Backtest simulation ---")
# Load the config
with open('strategy_config.json', 'r') as f:
    cfg_data = json.load(f)
    
anthropic_cfg = cfg_data.get('anthropic_managed', {})
print(f"Config: {json.dumps(anthropic_cfg, indent=2)}")

# Quick check: how many quick_pop base trades are there?
conn = sqlite3.connect('trader.db')
base_trades = conn.execute('''
    SELECT COUNT(*) as total, SUM(outcome_pnl_usd) as total_pnl
    FROM strategy_outcomes 
    WHERE strategy = 'quick_pop' AND closed = 1 AND entered = 1
''').fetchone()
conn.close()
print(f"\nquick_pop base trades: {base_trades[0]} total, PnL: ${base_trades[1]:.2f}")