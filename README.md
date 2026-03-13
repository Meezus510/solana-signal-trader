# Solana Signal Trader

Algorithmic trading bot for Solana tokens. Ingests token signals from a Telegram channel, fetches real-time prices from Birdeye, and executes a take-profit / trailing-stop strategy via Jupiter.

Currently running in **paper mode** for strategy validation — live on-chain execution is wired and ready to activate by swapping `PaperExchange` for `JupiterSwapExecutor`.

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
      ├─→ PaperExchange / JupiterSwapExecutor    trader/trading/exchange.py
      │     buy() / sell_partial() / sell_all()
      │
      └─→ PortfolioManager              trader/trading/portfolio.py
            in-memory position registry

Live execution (webhook server):
POST /webhook ──→ solana_api/my_bot_script.py
                    perform_swap()        ← Jupiter market swap (entry)
                    create_limit_order()  ← Jupiter limit orders (TP exits)
```

---

## Strategy

| Event | Rule | Action |
|-------|------|--------|
| Entry | Signal received, price available, cash sufficient | Buy $10 at live price |
| Stop loss | Price ≤ entry × 0.65 | Sell 100% → CLOSE |
| Take profit | Price ≥ entry × 2.5 | Sell 50% → activate trailing stop |
| Trailing — new high | Price > highest seen | Update trailing stop level |
| Trailing stop | Price ≤ highest × 0.65 | Sell remaining 100% → CLOSE |

PnL is tracked per sell event and accumulated across partial and full closes:

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
├── trader/                     # core trading package
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
│   │   ├── exchange.py         # PaperExchange  ← swap for JupiterSwapExecutor here
│   │   └── engine.py           # TradingEngine — strategy orchestration
│   │
│   └── utils/
│       └── logging.py          # structured logging setup
│
├── solana_api/                 # on-chain execution package
│   ├── RestClientHelper.py     # Jupiter / Solana RPC sync REST client
│   └── my_bot_script.py        # Quart webhook server — market swaps + limit orders
│
└── tests/
    ├── test_parser.py          # parser unit tests (pure, no network)
    └── test_strategy.py        # strategy math, exchange, portfolio tests
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

On first run, Telethon will prompt for your phone number and OTP and save a session file:

```bash
python run.py
# Enter phone number and OTP when prompted
# trader/listener/tg_session.session is saved for future runs
```

### 5. Run

```bash
# Live mode — listens to Telegram and executes strategy
python run.py

# Demo mode — injects sample tokens, no Telegram required
python run.py --demo

# On-chain execution webhook (separate process, live trading)
python solana_api/my_bot_script.py

# Tests
pytest tests/ -v
```

---

## Going Live

The system is designed to transition from paper to live trading with a single swap in `run.py`:

```python
# Current — paper mode
exchange = PaperExchange(portfolio=portfolio, cfg=cfg)

# Live — replace with:
exchange = JupiterSwapExecutor(cfg=cfg)
```

The `solana_api/my_bot_script.py` webhook server handles on-chain execution independently and can receive signals from any source via POST `/webhook`.

---

## Extending

| Goal | Where to change |
|------|----------------|
| Enable live on-chain execution | Replace `PaperExchange` with `JupiterSwapExecutor` in `run.py` |
| WebSocket price feed | Add `subscribe()` to `BirdeyePriceClient`; remove polling loop in `engine.py` |
| Tune TP / SL levels | Edit constants in `trader/config.py` |
| Additional TP ladders | Extend `evaluate_position()` in `engine.py` |
| Custom limit order exits | Edit `create_limit_order` calls in `handle_buy()` in `my_bot_script.py` |
| Fee / slippage modelling | Add to `PaperExchange.sell_*`; use `total_fees_usd` field |
| Persistent trade log | Add a `TradeRepository` and inject into `TradingEngine` |
| Multiple signal sources | Add new listener classes alongside `TelegramListener` |
