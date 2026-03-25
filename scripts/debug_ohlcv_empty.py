#!/usr/bin/env python3
"""
scripts/debug_ohlcv_empty.py — Diagnose OHLCV_EMPTY log entries.

For each (mint, pair) from the fallback log, this script:
  1. Checks what Moralis /token/mainnet/{mint}/pairs returns — verifies the
     pair address and whether Moralis knows the token.
  2. Tries the OHLCV endpoint with a wide window (last 3h of 1min candles) to
     see if ANY data exists for the pair at all.
  3. Retries the original 10s window to see if data is available now (hours later).
  4. Checks whether the pair address in the log matches Moralis's top pair.

Usage:
    python scripts/debug_ohlcv_empty.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

API_KEY  = os.environ["MORALIS_API_KEY"]
BASE_URL = os.getenv("MORALIS_BASE_URL", "https://solana-gateway.moralis.io")
HEADERS  = {"X-Api-Key": API_KEY}

# Cases from the fallback log
CASES = [
    {
        "mint":      "8XpWfbccbiucDKLX2Ak7XKUYZ9pBM6eUDHVkTC1Epump",
        "pair":      "8Ua2d718WnVQ4PZJXHffd8cNZiSkxeeP2ELyR37eyzKK",
        "from_unix": 1774213826,
        "to_unix":   1774214826,
    },
    {
        "mint":      "3KufQnJEhMfTdtNGE7bRGST7i3DHPC53GNpTeXZtbonk",
        "pair":      "7rToK3LY3NPYinyd4hnRYeHimRcAByxiZcxHeJ4ENt6a",
        "from_unix": 1774216667,
        "to_unix":   1774217667,
    },
    {
        "mint":      "4L7GyYmQh859rvXtzZz7Smbz3F3hrCZDTeNonFQ5pPDR",
        "pair":      "31SEAzEGSsU4kkFp1woPFiaKGig91is1MSG2VkP3rCVB",
        "from_unix": 1774221226,
        "to_unix":   1774222226,
    },
]


def ts(unix: int) -> str:
    return datetime.fromtimestamp(unix, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


async def check_pairs_endpoint(session: aiohttp.ClientSession, mint: str) -> dict:
    """Call /token/mainnet/{mint}/pairs and return summary."""
    url = f"{BASE_URL}/token/mainnet/{mint}/pairs"
    async with session.get(url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=15)) as r:
        status = r.status
        body   = await r.json() if status == 200 else await r.text()
    if status != 200:
        return {"status": status, "error": body}
    pairs = body.get("pairs") or []
    summary = []
    for p in pairs[:5]:
        summary.append({
            "pairAddress": p.get("pairAddress"),
            "exchange":    p.get("exchangeName"),
            "inactive":    p.get("inactivePair"),
            "liquidityUsd": p.get("liquidityUsd"),
            "quoteToken":  p.get("quoteToken"),
        })
    return {"status": status, "pair_count": len(pairs), "top_pairs": summary}


async def check_ohlcv(
    session: aiohttp.ClientSession,
    pair: str,
    from_unix: int,
    to_unix: int,
    timeframe: str = "10s",
    label: str = "",
) -> dict:
    """Call the OHLCV endpoint and return a summary."""
    url = f"{BASE_URL}/token/mainnet/pairs/{pair}/ohlcv"
    params = {
        "fromDate":  str(from_unix),
        "toDate":    str(to_unix),
        "timeframe": timeframe,
        "currency":  "usd",
        "limit":     "100",
    }
    async with session.get(url, headers=HEADERS, params=params, timeout=aiohttp.ClientTimeout(total=15)) as r:
        status  = r.status
        raw     = await r.text()
    if status != 200:
        return {"label": label, "status": status, "error": raw[:200]}
    data    = json.loads(raw)
    results = data.get("result") or []
    return {
        "label":        label,
        "status":       status,
        "candle_count": len(results),
        "tokenAddress": data.get("tokenAddress"),
        "window":       f"{ts(from_unix)} → {ts(to_unix)}",
        "timeframe":    timeframe,
        "first_candle": results[0].get("timestamp") if results else None,
        "last_candle":  results[-1].get("timestamp") if results else None,
    }


async def debug_case(session: aiohttp.ClientSession, case: dict) -> None:
    mint      = case["mint"]
    pair      = case["pair"]
    from_unix = case["from_unix"]
    to_unix   = case["to_unix"]
    now       = int(time.time())

    print("=" * 72)
    print(f"MINT:  {mint}")
    print(f"PAIR:  {pair}")
    print(f"ORIGINAL WINDOW: {ts(from_unix)} → {ts(to_unix)}  ({to_unix - from_unix}s)")
    print()

    # 1. What does Moralis /pairs say for this mint?
    pairs_info = await check_pairs_endpoint(session, mint)
    print(f"[1] /token/mainnet/{mint[:12]}…/pairs → status={pairs_info['status']}")
    if "error" in pairs_info:
        print(f"    ERROR: {pairs_info['error']}")
    else:
        print(f"    total pairs: {pairs_info['pair_count']}")
        for p in pairs_info["top_pairs"]:
            match = " ← MATCH" if p["pairAddress"] == pair else ""
            print(f"    {p['pairAddress']}  exchange={p['exchange']}  "
                  f"liq=${p['liquidityUsd'] or 0:.0f}  inactive={p['inactive']}{match}")
        if not any(p["pairAddress"] == pair for p in pairs_info["top_pairs"]):
            if pairs_info["pair_count"] > 5:
                print(f"    (logged pair NOT in top 5 — Moralis returned {pairs_info['pair_count']} pairs total)")
            elif pairs_info["pair_count"] == 0:
                print(f"    WARNING: Moralis has NO pairs for this mint now")
            else:
                print(f"    WARNING: logged pair address NOT found in Moralis pairs list")
    print()

    # 2. Retry the exact original 10s window
    r_orig = await check_ohlcv(session, pair, from_unix, to_unix, "10s", "original 10s window (now)")
    print(f"[2] OHLCV retry (original 10s window, queried now):")
    print(f"    candles={r_orig['candle_count']}  tokenAddress={r_orig.get('tokenAddress')}  status={r_orig['status']}")
    print()

    # 3. Wide 1min window: last 3h of 1m candles
    wide_to   = now
    wide_from = now - 180 * 60  # 3h
    r_wide_1m = await check_ohlcv(session, pair, wide_from, wide_to, "1min", "wide 3h 1min window")
    print(f"[3] OHLCV wide window (3h, 1min timeframe):")
    print(f"    candles={r_wide_1m['candle_count']}  tokenAddress={r_wide_1m.get('tokenAddress')}  status={r_wide_1m['status']}")
    if r_wide_1m["candle_count"] > 0:
        print(f"    first={r_wide_1m['first_candle']}  last={r_wide_1m['last_candle']}")
    print()

    # 4. 1min candles around the original signal window (to see if ANY data exists there)
    r_orig_1m = await check_ohlcv(session, pair, from_unix - 1800, to_unix + 1800, "1min", "original window ±30min 1min")
    print(f"[4] OHLCV around original window ±30min (1min timeframe):")
    print(f"    candles={r_orig_1m['candle_count']}  tokenAddress={r_orig_1m.get('tokenAddress')}  status={r_orig_1m['status']}")
    if r_orig_1m["candle_count"] > 0:
        print(f"    first={r_orig_1m['first_candle']}  last={r_orig_1m['last_candle']}")
    print()


async def main() -> None:
    async with aiohttp.ClientSession() as session:
        for case in CASES:
            await debug_case(session, case)
            await asyncio.sleep(0.5)  # be polite to the API


if __name__ == "__main__":
    asyncio.run(main())
