# Solana Signal Trader

Algorithmic paper trading bot for Solana tokens. Ingests token call signals from a Telegram channel, fetches real-time prices via Birdeye, and runs **six parallel strategies** with independent cash, positions, and PnL tracking.

Three strategies run a **chart filter** (OHLCV-based pump/volume check) and three enter every signal unconditionally — enabling direct A/B comparison of filter performance.

Currently running in **paper mode** for strategy validation. On-chain execution via Jupiter is wired and ready to activate.

---

## Signal Flow

```
Telegram Channel
      │
      ▼
TelegramListener              trader/listener/client.py
  parse_message()             trader/listener/parser.py
      │
      │  asyncio.Queue[TokenSignal]
      ▼
MultiStrategyEngine           trader/trading/engine.py
  handle_new_signal()
      │
      ├─→ BirdeyePriceClient           trader/pricing/birdeye.py
      │     get_price()        ← entry price (fetched once, shared)
      │     get_ohlcv()        ← 20 × 1-min candles (fetched once if any chart strategy active)
      │     get_prices_batch() ← monitoring loop (unique mints only)
      │
      ├─→ [Group A] StrategyRunner × 3   ← enter every signal
      │
      └─→ [Group B] StrategyRunner × 3   ← chart-filtered; may schedule reanalysis
```

---

## Strategies

Six strategies run in parallel, each with its own cash, positions, and PnL:

| Name | Buy | TP Ladder | Trail | Timeout | Chart Filter | Reanalyze |
|------|-----|-----------|-------|---------|:------------:|:---------:|
| `quick_pop` | $30 | 1.5× (60%) → 2.0× (40%) | 22% | 45 min | — | — |
| `trend_rider` | cfg | 1.8× (50%) | 30% | 90 min | — | — |
| `infinite_moonbag` | $15 | 1.8/2.5/4.0/6.0× | 30% | — | — | — |
| `quick_pop_chart` | $30 | 1.5× (60%) → 2.0× (40%) | 22% | 45 min | ✓ | — |
| `trend_rider_chart` | cfg | 1.8× (50%) | 30% | 90 min | ✓ | ✓ |
| `infinite_moonbag_chart` | $15 | 1.8/2.5/4.0/6.0× | 30% | — | ✓ | — |

**Chart filter logic** (20 × 1-min OHLCV candles):
- `pump_ratio = current_price / lowest_low` — skip if ≥ 3.5×
- `vol_trend` — skip if volume is DYING (recent 5 bars avg < 60% of prior bars avg)
- **Reanalysis**: if skipped but chart looks recoverable, re-checks after a calculated delay (pump → 8 min, vol → 4 min, both → 10 min). Only `trend_rider_chart` has this enabled.

---

## Project Structure

```
.
├── run.py                          # entry point (live / demo mode)
├── pyproject.toml                  # packaging & dependencies
├── Makefile                        # common dev tasks
│
├── trader/                         # core trading package
│   ├── config.py                   # Config dataclass — loaded from .env
│   │
│   ├── analysis/
│   │   └── chart.py                # OHLCV chart filter — compute_chart_context()
│   │
│   ├── listener/
│   │   ├── client.py               # TelegramListener → asyncio.Queue
│   │   └── parser.py               # parse_message() — mint + symbol extraction
│   │
│   ├── persistence/
│   │   └── database.py             # SQLite — open positions, portfolio, signal log
│   │
│   ├── pricing/
│   │   └── birdeye.py              # BirdeyePriceClient — price + OHLCV
│   │
│   ├── strategies/
│   │   └── registry.py             # build_runners() — single source of truth for all strategies
│   │
│   ├── trading/
│   │   ├── models.py               # TokenSignal, Position, PortfolioState
│   │   ├── strategy.py             # StrategyRunner, InfiniteMoonbagRunner, StrategyConfig
│   │   └── engine.py               # MultiStrategyEngine — orchestrates all runners
│   │
│   └── utils/
│       └── logging.py              # structured logging setup
│
├── services/                       # on-chain execution (live trading)
│   ├── rest_client.py              # Jupiter / Solana RPC REST client
│   └── webhook_server.py           # Quart webhook — market swaps + limit orders
│
├── scripts/                        # offline tools
│   ├── summary.py                  # print PnL snapshot from trader.db
│   ├── backtest_chart.py           # replay chart filter against trades.log
│   ├── analyze_channel.py          # inspect Telegram channel metadata
│   └── generate_session_string.py  # create Telethon session string
│
└── tests/
    ├── test_parser.py              # parser unit tests (pure, no network)
    ├── test_strategy.py            # strategy math, exchange, portfolio
    └── test_chart.py               # chart filter + reanalysis (25 tests)
```

---

## Setup

### 1. Prerequisites

- Python 3.11+
- Telegram account with API credentials from [my.telegram.org/apps](https://my.telegram.org/apps)
- [Birdeye API key](https://birdeye.so/)
- [Jupiter API key](https://jup.ag/) *(live execution only)*
- Funded Solana wallet *(live execution only)*

### 2. Install

```bash
git clone <repo>
cd <repo>

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -e ".[dev]"
```

### 3. Configure

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Birdeye
BIRDEYE_API_KEY=your_key_here

# Telegram
TG_API_ID=12345678
TG_API_HASH=abcdef...
TG_CHANNEL=channel_username

# Live execution (when ready)
JUPITER_API_KEY=your_key_here
WALLET_ID=your_solana_wallet_address
PRIVATE_KEY=your_base58_private_key
SOLANA_RPC_URL=https://api.mainnet-beta.solana.com
```

### 4. First-time Telegram authentication

On first run, Telethon prompts for your phone number and OTP and saves a session file:

```bash
python run.py
# Enter phone number and OTP when prompted
# Session saved to trader/listener/tg_session.session
```

Or generate a session string directly:

```bash
python scripts/generate_session_string.py
```

### 5. Run

```bash
# Live mode — Telegram listener + all 6 strategies
python run.py
make run

# Demo mode — inject sample tokens, no Telegram required
python run.py --demo
make demo

# Tests
pytest tests/ -v
make test

# Print PnL summary from DB (safe while bot is running)
python scripts/summary.py
make summary

# Backtest chart filter against trades.log
python scripts/backtest_chart.py
make backtest
```

---

## Adding or Tuning Strategies

Edit `trader/strategies/registry.py`. Each `StrategyConfig` is fully isolated:

```python
my_cfg = StrategyConfig(
    name="my_strategy",
    buy_size_usd=25.0,
    stop_loss_pct=0.25,
    take_profit_levels=(
        TakeProfitLevel(multiple=2.0, sell_fraction_original=0.50),
    ),
    trailing_stop_pct=0.25,
    starting_cash_usd=cfg.starting_cash_usd,
    use_chart_filter=True,   # enable OHLCV filter
    use_reanalyze=True,      # re-check skipped signals after delay
)
# Append to the list returned by build_runners()
```

All strategies share a single Birdeye price feed. OHLCV is fetched at most once per signal regardless of how many chart strategies are active.

---

## Going Live

Swap `PaperExchange` for `JupiterSwapExecutor` in `trader/trading/strategy.py` and start the webhook server:

```bash
python services/webhook_server.py
```

The webhook server handles on-chain execution independently and accepts signals via `POST /webhook`.

---

## Extending

| Goal | Where to change |
|------|----------------|
| Add / tune a strategy | `trader/strategies/registry.py` |
| Tune chart filter thresholds | `trader/analysis/chart.py` — `PUMP_RATIO_MAX`, `VOL_WINDOW`, `VOL_DYING_THRESHOLD` |
| Add a new signal source | New listener class alongside `TelegramListener` |
| WebSocket price feed | Add `subscribe()` to `BirdeyePriceClient`; remove polling loop in `engine.py` |
| Enable live on-chain execution | Replace `PaperExchange` with `JupiterSwapExecutor` in `trader/trading/strategy.py` |
| Fee / slippage modelling | Add to `PaperExchange.sell_*`; use `total_fees_usd` field on `PortfolioState` |
