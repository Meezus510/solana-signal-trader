"""
run.py — Entry point for the Solana Trader system.

Usage:
    python run.py          # live mode: Telegram listener + paper trading engine
    python run.py --demo   # demo mode: inject sample signals without Telegram

Live signal flow:
    Telegram channel post
        → TelegramListener._on_new_message()
        → trader.listener.parser.parse_message()
        → asyncio.Queue[TokenSignal]
        → TradingEngine.handle_new_signal()
        → BirdeyePriceClient (entry price)
        → PaperExchange.buy()
        → monitor_positions() polling loop
        → evaluate_position() → TP / SL / trailing stop
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

import aiohttp
from dotenv import load_dotenv

from trader.config import Config
from trader.listener.client import TelegramListener
from trader.pricing.birdeye import BirdeyePriceClient
from trader.trading.engine import TradingEngine
from trader.trading.exchange import PaperExchange
from trader.trading.models import PortfolioState, TokenSignal
from trader.trading.portfolio import PortfolioManager
from trader.utils.logging import configure_logging

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

load_dotenv()   # reads .env in the project root
configure_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Live mode
# ---------------------------------------------------------------------------

async def run_live(cfg: Config) -> None:
    """
    Connect to Telegram, listen for signals, and trade them with real prices.

    All three coroutines run concurrently via asyncio.gather:
        1. TelegramListener  — produces signals into the queue
        2. consume_signals() — feeds signals to the trading engine
        3. monitor_positions() — polls prices and fires strategy rules
    """
    signal_queue: asyncio.Queue[TokenSignal] = asyncio.Queue()

    listener = TelegramListener(cfg=cfg, signal_queue=signal_queue)
    await listener.start()

    portfolio = PortfolioState(
        starting_cash_usd=cfg.starting_cash_usd,
        available_cash_usd=cfg.starting_cash_usd,
    )

    async with aiohttp.ClientSession() as http:
        birdeye = BirdeyePriceClient(cfg=cfg, session=http)
        exchange = PaperExchange(portfolio=portfolio, cfg=cfg)
        mgr = PortfolioManager()
        engine = TradingEngine(
            cfg=cfg,
            birdeye_client=birdeye,
            exchange=exchange,
            portfolio=mgr,
        )

        async def consume_signals() -> None:
            while True:
                signal = await signal_queue.get()
                logger.info("[DEQUEUE] %s | mint=%s", signal.symbol, signal.mint_address)
                try:
                    await engine.handle_new_signal(signal)
                except Exception:
                    logger.exception("[ERROR] handle_new_signal failed for %s", signal.symbol)
                finally:
                    signal_queue.task_done()

        logger.info("=" * 60)
        logger.info("  Paper trading session started (live mode)")
        logger.info("  Cash      : $%.2f", cfg.starting_cash_usd)
        logger.info("  Buy size  : $%.2f", cfg.buy_size_usd)
        logger.info("  TP        : %.1f×  |  SL : %.0f%%  |  Trail : %.0f%%",
                    cfg.take_profit_multiple,
                    cfg.stop_loss_pct * 100,
                    cfg.trailing_stop_pct * 100)
        logger.info("  Press Ctrl+C to stop and print final summary.")
        logger.info("=" * 60)

        try:
            await asyncio.gather(
                consume_signals(),
                engine.monitor_positions(cycles=None),   # runs indefinitely
                listener.run_until_disconnected(),
            )
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            logger.info("Shutting down...")
            engine.print_summary()
            await listener.disconnect()


# ---------------------------------------------------------------------------
# Demo mode
# ---------------------------------------------------------------------------

async def run_demo(cfg: Config) -> None:
    """
    Inject a handful of known Solana tokens and monitor them.

    No Telegram connection required — useful for verifying the Birdeye
    integration and strategy logic without needing a live channel post.

    Replace or extend the demo_signals list with mints from your listener.
    """
    # Known liquid Solana tokens — Birdeye prices these reliably
    # NOTE: verify these mints are still canonical before use in production
    demo_signals = [
        TokenSignal(symbol="SOL",  mint_address="So11111111111111111111111111111111111111112"),
        TokenSignal(symbol="JUP",  mint_address="JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN"),
        TokenSignal(symbol="BONK", mint_address="DezXAZ8z7PnrnRJjz3wXBoRgixCa6xpB263pB263pB26"),
    ]

    portfolio = PortfolioState(
        starting_cash_usd=cfg.starting_cash_usd,
        available_cash_usd=cfg.starting_cash_usd,
    )

    async with aiohttp.ClientSession() as http:
        birdeye = BirdeyePriceClient(cfg=cfg, session=http)
        exchange = PaperExchange(portfolio=portfolio, cfg=cfg)
        mgr = PortfolioManager()
        engine = TradingEngine(cfg=cfg, birdeye_client=birdeye, exchange=exchange, portfolio=mgr)

        logger.info("=" * 60)
        logger.info("  Paper trading session started (DEMO mode)")
        logger.info("  Signals   : %d", len(demo_signals))
        logger.info("  Cycles    : %d @ %.1fs", cfg.demo_cycles, cfg.poll_interval_seconds)
        logger.info("=" * 60)

        async def inject() -> None:
            for i, signal in enumerate(demo_signals):
                await engine.handle_new_signal(signal)
                if i < len(demo_signals) - 1:
                    await asyncio.sleep(3)

        await asyncio.gather(
            inject(),
            engine.monitor_positions(cycles=cfg.demo_cycles),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Solana Trader — Solana paper trading bot")
    p.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode with sample tokens (no Telegram required)",
    )
    return p.parse_args()


async def main() -> None:
    args = parse_args()

    try:
        cfg = Config.load()
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)

    if args.demo:
        await run_demo(cfg)
    else:
        await run_live(cfg)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
