# Solana Signal Trader

Algorithmic paper trading bot for Solana tokens. Ingests token call signals from a Telegram channel, fetches real-time prices via Birdeye, and runs **six parallel strategies** with independent cash, positions, and PnL tracking.

Three strategies run a **chart filter** (OHLCV-based pump/volume check) and three enter every signal unconditionally — enabling direct A/B comparison of filter performance.

The chart-filtered `quick_pop_chart_ml` strategy also applies a **KNN machine learning filter** that scores each signal 0–10 based on historical outcomes, skips low-confidence signals, and scales up position size for high-confidence ones.

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
      │     get_ohlcv()        ← 1m candles (chart filter + snapshot storage)
      │     get_prices_batch() ← monitoring loop (unique mints only)
      │
      ├─→ MoralisOHLCVClient           trader/pricing/moralis.py  [optional]
      │     get_ohlcv()        ← 10s × 100 candles (ML scoring)
      │     get_pair_stats()   ← buy/sell counts, volume, price momentum
      │
      ├─→ ChartMLScorer                trader/analysis/ml_scorer.py
      │     score()            ← KNN confidence score 0–10
      │
      ├─→ [Group A] StrategyRunner × 3   ← enter every signal
      │
      └─→ [Group B] StrategyRunner × 3   ← chart-filtered; may apply ML gate or reanalysis
```

---

## Strategies

Six strategies run in parallel, each with its own cash, positions, and PnL:

| Name | Buy | TP Ladder | Trail | Timeout | Chart | Reanalyze | ML Filter |
|------|-----|-----------|-------|---------|:-----:|:---------:|:---------:|
| `quick_pop` | $30 | 1.5× (60%) → 2.0× (40%) | 22% | 45 min | — | — | — |
| `trend_rider` | cfg | 1.8× (50%) | 30% | 90 min | — | — | — |
| `infinite_moonbag` | $15 | 1.8/2.5/4.0/6.0× | 30% | — | — | — | — |
| `quick_pop_chart_ml` | $30 | 1.5× (60%) → 2.0× (40%) | 22% | 45 min | ✓ | — | ✓ |
| `trend_rider_chart_reanalyze` | cfg | 1.8× (50%) | 30% | 90 min | ✓ | ✓ | — |
| `infinite_moonbag_chart` | $15 | 1.8/2.5/4.0/6.0× | 30% | — | ✓ | — | — |

**Chart filter** (1m × 40 OHLCV candles from Birdeye):
- `pump_ratio = current_price / lowest_low` — skip if ≥ 3.5×
- `vol_trend` — skip if volume is DYING (recent 5 bars avg < 60% of prior bars avg)
- **Reanalysis**: if skipped but chart looks recoverable, re-checks after a calculated delay (pump → 8 min, vol → 4 min, both → 10 min). Only `trend_rider_chart_reanalyze` has this enabled.

**ML filter** (`quick_pop_chart_ml` only):
- Scores each signal 0–10 using KNN over historical `quick_pop_chart_ml` trade outcomes
- Uses a 13-feature vector: 8 OHLCV-derived features + 5 Moralis pair stats (buy/sell pressure, volume, price momentum)
- Falls back to 1m Birdeye candles when 10s Moralis data is unavailable — training and inference stay consistent
- Requires `MIN_SAMPLES=5` closed trades before activating; allows all signals through until then

| ML Score | Action | Buy size |
|----------|--------|----------|
| `None` (< 5 samples) | Enter | $30 (1×) |
| < 5.0 | Skip | — |
| 5.0 – 7.9 | Enter | $30 (1×) |
| 8.0 – 9.4 | Enter | $60 (2×) |
| ≥ 9.5 | Enter | $90 (3×) |

Skipped signals are still saved as training data (labeled `entered=False`) so the model learns from rejected signals too.

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
│   │   ├── chart.py                # OHLCV chart filter — compute_chart_context()
│   │   └── ml_scorer.py            # KNN ML scorer — ChartMLScorer, extract_features()
│   │
│   ├── listener/
│   │   ├── client.py               # TelegramListener → asyncio.Queue
│   │   └── parser.py               # parse_message() — mint + symbol extraction
│   │
│   ├── persistence/
│   │   └── database.py             # SQLite — positions, portfolio, chart snapshots
│   │
│   ├── pricing/
│   │   ├── birdeye.py              # BirdeyePriceClient — price + OHLCV
│   │   └── moralis.py              # MoralisOHLCVClient — 10s candles + pair stats
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
│   ├── test_ml_scorer.py           # LOO evaluation of ML scorer against trades.log
│   ├── test_ml_mock.py             # ML scorer test with synthetic 10s candle data
│   ├── test_apis.py                # sanity-check Birdeye + Moralis API calls
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
- [Moralis API key](https://moralis.io/) *(optional — enables 10s candles and pair stats for ML)*
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

# Moralis (optional — enables 10s candles + pair stats for ML scoring)
MORALIS_API_KEY=your_key_here

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

# Evaluate ML scorer against historical trades.log
python scripts/test_ml_scorer.py

# Test ML scorer with synthetic data (no DB or API required)
python scripts/test_ml_mock.py
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
    use_chart_filter=True,        # enable OHLCV chart filter
    use_reanalyze=True,           # re-check skipped signals after delay
    save_chart_data=True,         # persist OHLCV snapshots for ML training
    use_ml_filter=True,           # gate entries on KNN confidence score
    ml_min_score=5.0,             # skip if score < this
    ml_high_score_threshold=8.0,  # 2× size if score >= this
    ml_max_score_threshold=9.5,   # 3× size if score >= this
)
# Append to the list returned by build_runners()
```

Strategy naming convention: `{base}_chart[_reanalyze][_ml]`

All strategies share a single Birdeye price feed. OHLCV and Moralis data are fetched at most once per signal regardless of how many strategies are active.

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
| Tune ML score thresholds / multipliers | `trader/strategies/registry.py` — per-strategy `ml_min_score`, `ml_high_score_threshold`, etc. |
| Change ML candle resolution | `trader/analysis/ml_scorer.py` — `MORALIS_OHLCV_INTERVAL`, `MORALIS_OHLCV_BARS` |
| Add a new signal source | New listener class alongside `TelegramListener` |
| WebSocket price feed | Add `subscribe()` to `BirdeyePriceClient`; remove polling loop in `engine.py` |
| Enable live on-chain execution | Replace `PaperExchange` with `JupiterSwapExecutor` in `trader/trading/strategy.py` |
| Fee / slippage modelling | Add to `PaperExchange.sell_*`; use `total_fees_usd` field on `PortfolioState` |
