"""
run.py — Entry point for the Solana Trader system.

Usage:
    python run.py          # live mode: Telegram listener + multi-strategy engine
    python run.py --demo   # demo mode: inject sample signals without Telegram

Strategy configuration:
    Edit trader/strategies/registry.py to add, remove, or tune strategies.
    Each strategy is fully isolated: its own cash, positions, and PnL.
    All strategies share a single Birdeye price feed.

Live signal flow:
    Telegram channel post
        → TelegramListener._on_new_message()
        → trader.listener.parser.parse_message()
        → asyncio.Queue[TokenSignal]
        → MultiStrategyEngine.handle_new_signal()
            → BirdeyePriceClient.get_price()  (once, shared)
            → StrategyRunner.enter_position()  (once per strategy)
        → MultiStrategyEngine.monitor_positions()
            → BirdeyePriceClient.get_prices_batch()  (unique mints only)
            → StrategyRunner.evaluate_position()  (per strategy, independent)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

import aiohttp
from dotenv import load_dotenv

from trader.config import Config
from trader.listener.client import TelegramListener
from trader.persistence.database import TradeDatabase
from trader.pricing.birdeye import BirdeyePriceClient
from trader.pricing.moralis import MoralisOHLCVClient
from trader.strategies.registry import build_runners
from trader.trading.engine import MultiStrategyEngine
from trader.trading.models import TokenSignal
from trader.utils.logging import configure_logging

load_dotenv()
configure_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Live mode
# ---------------------------------------------------------------------------

async def run_live(cfg: Config) -> None:
    """
    Connect to Telegram, listen for signals, and trade them across all strategies.

    All coroutines run concurrently via asyncio.gather:
        1. TelegramListener  — produces signals into the queue
        2. consume_signals() — feeds signals to the engine
        3. monitor_positions() — polls prices and fires strategy rules
    """
    signal_queue: asyncio.Queue[TokenSignal] = asyncio.Queue()
    db = TradeDatabase(path=os.getenv("DB_PATH", "trader.db"))

    runners = build_runners(cfg, db=db)
    for runner in runners:
        saved = db.load_portfolio(runner.name)
        if saved:
            available_cash, starting_cash = saved
            runner.restore_cash(available_cash, starting_cash)
            logger.info("[RESTORE] Strategy '%s' | cash=$%.2f", runner.name, available_cash)
        runner.restore_positions(db.load_open_positions(runner.name))

    listener = TelegramListener(cfg=cfg, signal_queue=signal_queue, db=db)
    await listener.start()

    async with aiohttp.ClientSession() as http:
        birdeye  = BirdeyePriceClient(cfg=cfg, session=http)
        moralis  = MoralisOHLCVClient(cfg=cfg, session=http) if cfg.moralis_api_key else None
        if moralis:
            logger.info("[MORALIS] High-res OHLCV enabled (10s candles + pair stats)")
        engine = MultiStrategyEngine(
            cfg=cfg, runners=runners, birdeye_client=birdeye, db=db, moralis_client=moralis,
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

        names = ", ".join(r.name for r in runners)
        logger.info("=" * 60)
        logger.info("  Paper trading session started (live mode)")
        logger.info("  Strategies : %s", names)
        logger.info("  Buy size   : $%.2f / strategy", cfg.buy_size_usd)
        logger.info("  Press Ctrl+C to stop and print final summary.")
        logger.info("=" * 60)

        try:
            await asyncio.gather(
                consume_signals(),
                engine.monitor_positions(cycles=None),
                listener.run_until_disconnected(),
            )
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            logger.info("Shutting down...")
            engine.print_summary()
            for runner in runners:
                db.save_portfolio(runner.portfolio_state, runner.name)
            db.close()
            await listener.disconnect()


# ---------------------------------------------------------------------------
# Demo mode
# ---------------------------------------------------------------------------

async def run_demo(cfg: Config) -> None:
    """
    Inject a handful of known Solana tokens across all configured strategies.

    No Telegram connection required — useful for verifying the Birdeye
    integration and strategy logic without needing a live channel post.
    """
    demo_signals = [
        TokenSignal(symbol="SOL",  mint_address="So11111111111111111111111111111111111111112"),
        TokenSignal(symbol="JUP",  mint_address="JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN"),
        TokenSignal(symbol="BONK", mint_address="DezXAZ8z7PnrnRJjz3wXBoRgixCa6xpB263pB263pB26"),
    ]

    runners = build_runners(cfg)

    async with aiohttp.ClientSession() as http:
        birdeye = BirdeyePriceClient(cfg=cfg, session=http)
        engine = MultiStrategyEngine(cfg=cfg, runners=runners, birdeye_client=birdeye)

        names = ", ".join(r.name for r in runners)
        logger.info("=" * 60)
        logger.info("  Paper trading session started (DEMO mode)")
        logger.info("  Strategies : %s", names)
        logger.info("  Signals    : %d", len(demo_signals))
        logger.info("  Cycles     : %d @ %.1fs", cfg.demo_cycles, cfg.poll_interval_seconds)
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
    p = argparse.ArgumentParser(description="Solana Trader — multi-strategy paper trading bot")
    p.add_argument("--demo", action="store_true", help="Run in demo mode (no Telegram required)")
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
