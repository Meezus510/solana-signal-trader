# Solana Signal Trader

A two-component Solana trading system built in Python:

- **Paper trader** — ingests token signals from a Telegram channel via a user account (Telethon), fetches real-time prices from Birdeye, and runs a TP / trailing-stop strategy against a mock portfolio with zero on-chain risk.
- **Live execution webhook** — a Quart HTTP server that receives buy/sell signals and executes market swaps + limit orders on-chain via Jupiter.

---

## Architecture

```
Telegram Channel
      │
      ▼
TelegramListener              trader/listener/client.py
  parse_message()             trader/listener/parser.py
      │
      │  asyncio.Queue[TokenSignal]      ← decouples I/O from execution
      ▼
TradingEngine                 trader/trading/engine.py
  handle_new_signal()
      │
      ├─→ BirdeyePriceClient            trader/pricing/birdeye.py
      │     get_price()        ← entry price (single token)
      │     get_prices_batch() ← monitoring (all open positions)
      │
      ├─→ PaperExchange                 trader/trading/exchange.py
      │     buy() / sell_partial() / sell_all()
      │     [swap → JupiterSwapExecutor for live trading]
      │
      └─→ PortfolioManager              trader/trading/portfolio.py
            in-memory position registry

Live execution (separate process):
POST /webhook ──→ solana_api/my_bot_script.py
                    perform_swap()        ← Jupiter market swap (entry)
                    create_limit_order()  ← Jupiter limit orders (TP exits)
```

---

## Paper Trading Strategy

| Event | Rule | Action |
|-------|------|--------|
| Entry | Signal received, price available, cash sufficient | Buy $10 at live price |
| Stop loss | Price ≤ entry × 0.65 | Sell 100% → CLOSE |
| Take profit | Price ≥ entry × 2.5 | Sell 50% → activate trailing stop |
| Trailing — new high | Price > highest seen | Update trailing stop level |
| Trailing stop | Price ≤ highest × 0.65 | Sell remaining 100% → CLOSE |

PnL accumulates correctly across the partial TP sell and the final close:

```
pnl_per_sell = (exit_price − entry_price) × quantity_sold
```

---

## Project Structure

```
.
├── run.py                      # entry point (live / demo mode)
├── pyproject.toml              # packaging & dev dependencies
├── Makefile                    # common dev tasks
├── .env.example                # environment variable template
│
├── trader/                     # paper trading package
│   ├── config.py               # centralised, immutable Config dataclass
│   │
│   ├── listener/
│   │   ├── client.py           # TelegramListener — channel → asyncio.Queue
│   │   └── parser.py           # parse_message(), is_update_message(), etc.
│   │
│   ├── pricing/
│   │   └── birdeye.py          # BirdeyePriceClient — single + batch REST
│   │
│   ├── trading/
│   │   ├── models.py           # TokenSignal, Position, PortfolioState
│   │   ├── portfolio.py        # PortfolioManager
│   │   ├── exchange.py         # PaperExchange  ← swap for live executor here
│   │   └── engine.py           # TradingEngine — strategy orchestration
│   │
│   └── utils/
│       └── logging.py          # structured logging setup
│
├── solana_api/                 # live execution package
│   ├── RestClientHelper.py     # Jupiter / Solana RPC sync REST client
│   └── my_bot_script.py        # Quart webhook server — market swaps + limit orders
│
└── tests/
    ├── test_parser.py          # parser unit tests (pure, no network)
    └── test_strategy.py        # PnL math, exchange, portfolio tests
```

---

## Setup

### 1. Prerequisites

- Python 3.11+
- A Telegram account with API credentials from [my.telegram.org/apps](https://my.telegram.org/apps)
- A [Birdeye API key](https://birdeye.so/)
- A [Jupiter API key](https://jup.ag/) *(live execution only)*
- A funded Solana wallet *(live execution only)*

### 2. Install

```bash
git clone <repo>
cd <repo>

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -e ".[dev]"           # installs runtime + dev dependencies
```

### 3. Configure

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# Birdeye
BIRDEYE_API_KEY=your_key_here

# Telegram
TG_API_ID=12345678
TG_API_HASH=abcdef...
TG_CHANNEL=channel_username

# Live execution (optional)
JUPITER_API_KEY=your_key_here
WALLET_ID=your_solana_wallet_address
PRIVATE_KEY=your_base58_private_key
SOLANA_RPC_URL=https://api.mainnet-beta.solana.com
```

### 4. First-time Telegram authentication

On first run, Telethon will prompt for your phone number and OTP and save a session file:

```bash
python run.py
# Enter phone number and OTP when prompted
# trader/listener/tg_session.session is saved for future runs
```

### 5. Run

```bash
# Paper trading — listens to Telegram and simulates trades with real prices
python run.py

# Demo mode — injects sample tokens, no Telegram required
python run.py --demo

# Live execution webhook (separate process)
python solana_api/my_bot_script.py

# Tests
pytest tests/ -v
```

---

## Extending

| Goal | Where to change |
|------|----------------|
| Live on-chain execution | Replace `PaperExchange` with `JupiterSwapExecutor` in `run.py` |
| WebSocket price feed | Add `subscribe()` to `BirdeyePriceClient`; remove polling loop in `engine.py` |
| Additional TP ladders | Extend `evaluate_position()` in `engine.py` |
| Fee / slippage modelling | Add to `PaperExchange.sell_*` methods; use `total_fees_usd` field |
| Persistent trade log | Add a `TradeRepository` and inject into `TradingEngine` |
| Multiple signal sources | Add new listener classes alongside `TelegramListener` |
| Custom limit order ladders | Edit the `create_limit_order` calls in `handle_buy()` |
