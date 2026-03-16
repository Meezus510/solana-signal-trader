"""
run.py — Entry point for the Solana Trader system.

Usage:
    python run.py          # live mode: Telegram listener + multi-strategy engine
    python run.py --demo   # demo mode: inject sample signals without Telegram

Adding a new strategy:
    Define a StrategyConfig in the STRATEGIES list below. The bot will
    independently manage cash, positions, and PnL for each strategy.
    All strategies share a single Birdeye price feed — one API call per
    unique mint per polling cycle, regardless of how many strategies hold it.

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
from trader.trading.engine import MultiStrategyEngine
from trader.trading.models import TokenSignal
from trader.trading.strategy import (
    InfiniteMoonbagRunner,
    StrategyConfig,
    StrategyRunner,
    TakeProfitLevel,
)
from trader.utils.logging import configure_logging

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

load_dotenv()   # reads .env in the project root
configure_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy definitions
# ---------------------------------------------------------------------------
# Add entries here to run additional strategies in parallel.
# Each strategy is fully isolated: its own cash, positions, and PnL.
# All strategies share the same Birdeye price feed.

def build_runners(cfg: Config, db=None) -> list[StrategyRunner]:
    """
    Active strategy roster — six runners in two groups, one shared Birdeye feed.

    Group A (no chart filter) — enters every signal unconditionally:
        quick_pop, trend_rider, infinite_moonbag

    Group B (chart filter enabled) — skips entry when the token has already
    pumped >3.5× from its recent low or volume is dying:
        quick_pop_chart, trend_rider_chart, infinite_moonbag_chart

    All six strategies share the same Birdeye price feed (one API call per
    unique mint per poll cycle). The OHLCV fetch for chart analysis is also
    performed once per signal and shared across the three chart runners.

    quick_pop / quick_pop_chart
        Fast scalp: TP1 at 1.5× (sell 60%) → TP2 at 2.0× (sell 40%) — fully exits.
        Trail at 22% below high after TP1.
        Exit after 45 min if TP1 not yet hit (price still < 1.49×).

    trend_rider / trend_rider_chart
        Momentum hold: TP1 at 1.8× (sell 50% of original)
        Trail at 30% below high after TP1.
        Exit after 90 min if price < entry × 1.15. Max hold: 4 hours.

    infinite_moonbag / infinite_moonbag_chart (v2)
        Grace period 90s: −30% floor. After grace: −22% floor.
        TP ladder: 1.8×/20%, 2.5×/15%, 4.0×/15%, 6.0×/10% of original.
        Stop ladder: 1.8×→1.35×, 2.5×→1.90×, 4.0×→2.80×, 6.0×→3.50×.
    """
    quick_pop_cfg = StrategyConfig(
        name="quick_pop",
        buy_size_usd=30.0,
        stop_loss_pct=0.20,
        take_profit_levels=(
            TakeProfitLevel(multiple=1.5, sell_fraction_original=0.60),
            TakeProfitLevel(multiple=2.0, sell_fraction_original=0.40),
        ),
        trailing_stop_pct=0.22,
        starting_cash_usd=cfg.starting_cash_usd,
        timeout_minutes=45.0,
        timeout_min_gain_pct=0.49,  # exit if price still below 1.49× (TP1 not hit)
    )

    trend_rider_cfg = StrategyConfig(
        name="trend_rider",
        buy_size_usd=cfg.buy_size_usd,
        stop_loss_pct=0.30,
        take_profit_levels=(
            TakeProfitLevel(multiple=1.8, sell_fraction_original=0.50),
        ),
        trailing_stop_pct=0.30,
        starting_cash_usd=cfg.starting_cash_usd,
        timeout_minutes=90.0,
        timeout_min_gain_pct=0.15,
        max_hold_minutes=240.0,
    )

    moonbag_cfg = StrategyConfig(
        name="infinite_moonbag",
        buy_size_usd=15.0,
        stop_loss_pct=0.30,   # initial stop at entry × 0.70 (−30% grace floor)
        take_profit_levels=(
            TakeProfitLevel(multiple=1.8, sell_fraction_original=0.20),
            TakeProfitLevel(multiple=2.5, sell_fraction_original=0.15),
            TakeProfitLevel(multiple=4.0, sell_fraction_original=0.15),
            TakeProfitLevel(multiple=6.0, sell_fraction_original=0.10),
        ),
        trailing_stop_pct=0.30,   # not used — ladder overrides stop_loss_price
        starting_cash_usd=cfg.starting_cash_usd,
        # No timeout or max_hold — InfiniteMoonbagRunner ignores them
    )

    # --- Chart-filtered mirrors (identical params, use_chart_filter=True) ---

    quick_pop_chart_cfg = StrategyConfig(
        name="quick_pop_chart",
        buy_size_usd=30.0,
        stop_loss_pct=0.20,
        take_profit_levels=(
            TakeProfitLevel(multiple=1.5, sell_fraction_original=0.60),
            TakeProfitLevel(multiple=2.0, sell_fraction_original=0.40),
        ),
        trailing_stop_pct=0.22,
        starting_cash_usd=cfg.starting_cash_usd,
        timeout_minutes=45.0,
        timeout_min_gain_pct=0.49,
        use_chart_filter=True,
    )

    trend_rider_chart_cfg = StrategyConfig(
        name="trend_rider_chart",
        buy_size_usd=cfg.buy_size_usd,
        stop_loss_pct=0.30,
        take_profit_levels=(
            TakeProfitLevel(multiple=1.8, sell_fraction_original=0.50),
        ),
        trailing_stop_pct=0.30,
        starting_cash_usd=cfg.starting_cash_usd,
        timeout_minutes=90.0,
        timeout_min_gain_pct=0.15,
        max_hold_minutes=240.0,
        use_chart_filter=True,
    )

    moonbag_chart_cfg = StrategyConfig(
        name="infinite_moonbag_chart",
        buy_size_usd=15.0,
        stop_loss_pct=0.30,
        take_profit_levels=(
            TakeProfitLevel(multiple=1.8, sell_fraction_original=0.20),
            TakeProfitLevel(multiple=2.5, sell_fraction_original=0.15),
            TakeProfitLevel(multiple=4.0, sell_fraction_original=0.15),
            TakeProfitLevel(multiple=6.0, sell_fraction_original=0.10),
        ),
        trailing_stop_pct=0.30,
        starting_cash_usd=cfg.starting_cash_usd,
        use_chart_filter=True,
    )

    return [
        # Group A — no chart filter (enters every signal)
        StrategyRunner(cfg=quick_pop_cfg, db=db),
        StrategyRunner(cfg=trend_rider_cfg, db=db),
        InfiniteMoonbagRunner(cfg=moonbag_cfg, db=db),
        # Group B — chart filter enabled (skips late/dead entries)
        StrategyRunner(cfg=quick_pop_chart_cfg, db=db),
        StrategyRunner(cfg=trend_rider_chart_cfg, db=db),
        InfiniteMoonbagRunner(cfg=moonbag_chart_cfg, db=db),
    ]


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

    # Build and restore each strategy runner
    runners = build_runners(cfg, db=db)
    for runner in runners:
        saved = db.load_portfolio(runner.name)
        if saved:
            available_cash, starting_cash = saved
            runner.restore_cash(available_cash, starting_cash)
            logger.info(
                "[RESTORE] Strategy '%s' | cash=$%.2f", runner.name, available_cash
            )
        runner.restore_positions(db.load_open_positions(runner.name))

    listener = TelegramListener(cfg=cfg, signal_queue=signal_queue, db=db)
    await listener.start()

    async with aiohttp.ClientSession() as http:
        birdeye = BirdeyePriceClient(cfg=cfg, session=http)
        engine = MultiStrategyEngine(
            cfg=cfg,
            runners=runners,
            birdeye_client=birdeye,
            db=db,
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
