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

# WIF (dogwifhat) — major Solana token, always has live candle data
TEST_MINT        = "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm"  # WIF — stable, 15s works
TEST_MINT_PUMP   = "5RxYqEnX6bAFJ8Eh4TJSXhv6khrzc47d1kAc4jTDpump"   # recent pump — 1s works
TEST_TIME        = None   # None = now
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
        # 2. Birdeye v3 — 100 × 15s candles
        # ------------------------------------------------------------------
        print(SEP)
        print("  BIRDEYE v3  100 × 15s  candles")
        print(SEP)
        candles_15s = await birdeye.get_ohlcv_v3(TEST_MINT, bars=100, interval="15s")
        if candles_15s:
            first, last = candles_15s[0], candles_15s[-1]
            print(f"  OK — {len(candles_15s)} bars returned")
            print(f"  First: t={first.unix_time}  o={first.open:.8f}  c={first.close:.8f}")
            print(f"  Last:  t={last.unix_time}  o={last.open:.8f}  c={last.close:.8f}")
        else:
            print("  FAIL — no candles returned")

        print()

        # ------------------------------------------------------------------
        # 3. Birdeye v3 — 60 × 1s candles (pump token — WIF has no 1s data)
        # ------------------------------------------------------------------
        print(SEP)
        print("  BIRDEYE v3  60 × 1s  candles  (pump token)")
        print(SEP)
        candles_1s = await birdeye.get_ohlcv_v3(TEST_MINT_PUMP, bars=60, interval="1s")
        if candles_1s:
            first, last = candles_1s[0], candles_1s[-1]
            print(f"  OK — {len(candles_1s)} bars returned")
            print(f"  First: t={first.unix_time}  o={first.open:.8f}  c={first.close:.8f}")
            print(f"  Last:  t={last.unix_time}  o={last.open:.8f}  c={last.close:.8f}")
        else:
            print("  FAIL — no candles returned")

        print()

        # ------------------------------------------------------------------
        # 4. Birdeye — token overview (incl. new 30m wallet fields)
        # ------------------------------------------------------------------
        print(SEP)
        print("  BIRDEYE  token_overview  (incl. 30m wallet + supply)")
        print(SEP)
        overview = await birdeye.get_token_overview(TEST_MINT)
        if overview:
            print(f"  OK — fields returned:")
            for k, v in overview.items():
                print(f"    {k:<28} {v}")
        else:
            print("  FAIL — no overview returned")

        print()

        # ------------------------------------------------------------------
        # 5. Birdeye — token security
        # ------------------------------------------------------------------
        print(SEP)
        print("  BIRDEYE  token_security")
        print(SEP)
        security = await birdeye.get_token_security(TEST_MINT)
        if security:
            print("  OK — fields returned:")
            for k, v in security.items():
                print(f"    {k:<35} {v}")
        else:
            print("  FAIL — no security data returned")

        print(SEP)


if __name__ == "__main__":
    asyncio.run(main())
