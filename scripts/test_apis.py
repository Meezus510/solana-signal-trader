"""
scripts/test_apis.py — Quick sanity check for Birdeye and Moralis OHLCV APIs.

Uses NETAINYAHU (a known token from trades.log) to verify both APIs
return candle data correctly.

Usage:
    source venv/bin/activate
    python scripts/test_apis.py
"""

import asyncio
import os
import sys

import aiohttp
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trader.config import Config
from trader.pricing.birdeye import BirdeyePriceClient
from trader.pricing.moralis import MoralisOHLCVClient

# WIF (dogwifhat) — major Solana token, always has live candle data
TEST_MINT   = "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm"
TEST_TIME   = None   # None = now
SEP = "-" * 60


async def main() -> None:
    try:
        cfg = Config.load()
    except ValueError as exc:
        print(f"[ERROR] Config: {exc}")
        sys.exit(1)

    async with aiohttp.ClientSession() as session:
        # ------------------------------------------------------------------
        # 1. Birdeye — 40 × 1m candles
        # ------------------------------------------------------------------
        print(SEP)
        print("  BIRDEYE  40 × 1m  candles")
        print(SEP)
        import time as _time
        t = TEST_TIME or int(_time.time())

        birdeye = BirdeyePriceClient(cfg, session)
        candles = await birdeye.get_ohlcv(TEST_MINT, bars=40, interval="1m", time_to=t)
        if candles:
            first, last = candles[0], candles[-1]
            print(f"  OK — {len(candles)} bars returned")
            print(f"  First: t={first.unix_time}  o={first.open:.8f}  c={first.close:.8f}")
            print(f"  Last:  t={last.unix_time}  o={last.open:.8f}  c={last.close:.8f}")
        else:
            print("  FAIL — no candles returned")

        print()

        # ------------------------------------------------------------------
        # 2. Jupiter pair resolution
        # ------------------------------------------------------------------
        print(SEP)
        print("  JUPITER  pair address resolution")
        print(SEP)
        moralis = MoralisOHLCVClient(cfg, session)
        pair = await moralis._resolve_pair_address(TEST_MINT)
        if pair:
            print(f"  OK — pair address: {pair}")
        else:
            print("  FAIL — could not resolve pair address")

        print()

        # ------------------------------------------------------------------
        # 3. Moralis — 100 × 10s candles
        # ------------------------------------------------------------------
        print(SEP)
        print("  MORALIS  100 × 10s  candles")
        print(SEP)
        if not cfg.moralis_api_key:
            print("  SKIP — MORALIS_API_KEY not set in .env")
        elif not pair:
            print("  SKIP — pair resolution failed above")
        else:
            candles = await moralis.get_ohlcv(TEST_MINT, bars=100, interval="10s", time_to=t)
            if candles:
                first, last = candles[0], candles[-1]
                print(f"  OK — {len(candles)} bars returned")
                print(f"  First: t={first.unix_time}  o={first.open:.8f}  c={first.close:.8f}")
                print(f"  Last:  t={last.unix_time}  o={last.open:.8f}  c={last.close:.8f}")
            else:
                print("  FAIL — no candles returned")

        print(SEP)


if __name__ == "__main__":
    asyncio.run(main())
