"""
trader/pricing/moralis.py — Moralis sub-minute OHLCV client.

Fetches high-resolution (10s) OHLCV candles for Solana tokens via the
Moralis Solana Gateway API.  Used by the ML scorer to get finer-grained
entry-pattern data than Birdeye's 1-minute minimum allows.

Moralis requires a *pair address* (liquidity pool) rather than a token
mint.  This client resolves mint → pair address once via the Jupiter
quote API, then caches the result in memory for the session lifetime.

Endpoint:
    GET /token/mainnet/pairs/{pair_address}/ohlcv
    ?fromDate=<unix>&toDate=<unix>&timeframe=10s&currency=usd&limit=100

Pair resolution:
    GET https://api.jup.ag/swap/v1/quote
        ?inputMint=<token>&outputMint=<WSOL>&amount=100000&slippageBps=50
    → routePlan[0].swapInfo.ammKey  (primary Raydium/Orca pool)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiohttp

from trader.analysis.chart import OHLCVCandle
from trader.config import Config

logger = logging.getLogger(__name__)

_FALLBACK_LOG_PATH = Path(os.getenv("MORALIS_FALLBACK_LOG", "logs/moralis_fallback.log"))

# Wrapped SOL mint — used as the output side of the Jupiter quote to
# find the token's primary SOL trading pool.
_WSOL_MINT = "So11111111111111111111111111111111111111112"

# Jupiter quote endpoint for pair-address resolution.
_JUPITER_QUOTE_URL = "https://api.jup.ag/swap/v1/quote"

# Seconds per bar for each Moralis timeframe string.
_INTERVAL_SECONDS: dict[str, int] = {
    "1s":   1,
    "10s":  10,
    "30s":  30,
    "1min": 60,
    "5min": 300,
    "10min": 600,
    "30min": 1800,
    "1h":   3600,
    "4h":   14400,
    "12h":  43200,
    "1d":   86400,
}


@dataclass
class MoralisStats:
    """
    In-memory counters for Moralis fallback events.
    Also writes one structured line per event to logs/moralis_fallback.log.

    Log format:
        2026-03-18T06:02:00Z | NO_PAIRS_INDEXED | <mint>
        2026-03-18T06:02:00Z | HTTP_ERROR | <mint> | status=404
        2026-03-18T06:02:00Z | JUPITER_FAILED | <mint>
        2026-03-18T06:02:00Z | OHLCV_EMPTY | <mint>
    """
    no_pairs_indexed: int = 0
    http_error: int = 0
    jupiter_failed: int = 0
    ohlcv_empty: int = 0

    @property
    def total_fallbacks(self) -> int:
        return self.no_pairs_indexed + self.http_error + self.jupiter_failed + self.ohlcv_empty

    def record(self, reason: str, mint: str, detail: str = "") -> None:
        """Increment counter and append one line to the fallback log."""
        if reason == "NO_PAIRS_INDEXED":
            self.no_pairs_indexed += 1
        elif reason == "HTTP_ERROR":
            self.http_error += 1
        elif reason == "JUPITER_FAILED":
            self.jupiter_failed += 1
        elif reason == "OHLCV_EMPTY":
            self.ohlcv_empty += 1

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        line = f"{ts} | {reason} | {mint}"
        if detail:
            line += f" | {detail}"
        try:
            _FALLBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with _FALLBACK_LOG_PATH.open("a") as f:
                f.write(line + "\n")
        except OSError as exc:
            logger.warning("[Moralis] Could not write fallback log: %s", exc)

    def log_summary(self) -> None:
        """Log a one-line counter summary — call on shutdown or periodically."""
        logger.info(
            "[Moralis] fallback summary — no_pairs=%d http_error=%d jupiter_failed=%d ohlcv_empty=%d total=%d",
            self.no_pairs_indexed, self.http_error, self.jupiter_failed,
            self.ohlcv_empty, self.total_fallbacks,
        )


class MoralisOHLCVClient:
    """
    Async OHLCV client backed by the Moralis Solana Gateway API.

    Provides 10-second candle resolution (vs Birdeye's 1-minute minimum),
    which captures the precise pump shape of quick_pop signals.

    Session-level state:
        _pair_cache  — mint → pair_address, populated on first use.
                       Pair addresses are stable for the lifetime of a pool,
                       so a single lookup per mint per session is sufficient.
    """

    def __init__(self, cfg: Config, session: aiohttp.ClientSession) -> None:
        self._cfg = cfg
        self._session = session
        self._pair_cache: dict[str, str] = {}
        self._jupiter_api_key: str = os.getenv("JUPITER_API_KEY", "").strip()
        self.stats = MoralisStats()

    # ------------------------------------------------------------------
    # Pair address resolution
    # ------------------------------------------------------------------

    async def _resolve_pair_address(self, mint: str) -> Optional[str]:
        """
        Resolve a token mint to its most liquid SOL pair address.

        Primary:  Moralis /token/mainnet/{mint}/pairs — returns Moralis-native
                  pair addresses guaranteed to work with the OHLCV endpoint.
                  Picks the active SOL/WSOL pair with highest USD liquidity.

        Fallback: Jupiter quote API — used when Moralis has no pairs listed
                  (e.g. very new token not yet indexed). Extracts ammKey from
                  the first route hop.

        Returns None if neither source can find a pair.
        """
        if mint in self._pair_cache:
            return self._pair_cache[mint]

        pair_address = await self._resolve_via_moralis(mint)
        if not pair_address:
            pair_address = await self._resolve_via_jupiter(mint)

        if pair_address:
            self._pair_cache[mint] = pair_address
            logger.debug("[Moralis] Resolved %s → pair %s", mint, pair_address)

        return pair_address

    async def _resolve_via_moralis(self, mint: str) -> Optional[str]:
        """Pick the highest-liquidity active SOL pair from Moralis pairs endpoint."""
        url = f"{self._cfg.moralis_base_url}/token/mainnet/{mint}/pairs"
        headers = {"X-Api-Key": self._cfg.moralis_api_key}
        try:
            async with self._session.get(
                url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    logger.warning("[Moralis] pairs HTTP %d for %s — cannot resolve pair", resp.status, mint)
                    self.stats.record("HTTP_ERROR", mint, f"status={resp.status}")
                    return None
                data  = await resp.json()
                pairs = data.get("pairs") or []

                # Keep active SOL pairs, sort by liquidity descending
                sol_pairs = [
                    p for p in pairs
                    if not p.get("inactivePair")
                    and p.get("quoteToken") == _WSOL_MINT
                ]
                if not sol_pairs:
                    # fall back to any active pair
                    sol_pairs = [p for p in pairs if not p.get("inactivePair")]

                if not sol_pairs:
                    logger.warning("[Moralis] no active pairs found for %s — token may not be indexed yet", mint)
                    self.stats.record("NO_PAIRS_INDEXED", mint)
                    return None

                sol_pairs.sort(key=lambda p: p.get("liquidityUsd") or 0, reverse=True)
                return sol_pairs[0].get("pairAddress")

        except Exception as exc:
            logger.warning("[Moralis] pair resolution error for %s: %s", mint, exc)
            return None

    async def _resolve_via_jupiter(self, mint: str) -> Optional[str]:
        """Fallback: extract pool address from Jupiter quote routePlan."""
        params = {
            "inputMint":   mint,
            "outputMint":  _WSOL_MINT,
            "amount":      "100000",
            "slippageBps": "50",
        }
        headers: dict[str, str] = {}
        if self._jupiter_api_key:
            headers["x-api-key"] = self._jupiter_api_key

        try:
            async with self._session.get(
                _JUPITER_QUOTE_URL,
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    logger.warning("[Moralis] Jupiter fallback HTTP %d for %s", resp.status, mint)
                    return None
                data       = await resp.json()
                route_plan = data.get("routePlan") or []
                if not route_plan:
                    logger.warning("[Moralis] Jupiter returned no route for %s — pair unresolvable, will use Birdeye", mint)
                    self.stats.record("JUPITER_FAILED", mint, "no_route")
                    return None
                return (route_plan[0].get("swapInfo") or {}).get("ammKey")

        except Exception as exc:
            logger.warning("[Moralis] Jupiter fallback error for %s: %s", mint, exc)
            return None

    # ------------------------------------------------------------------
    # Pair stats
    # ------------------------------------------------------------------

    async def get_pair_stats(self, mint: str) -> Optional[dict]:
        """
        Fetch live pair statistics for a token from Moralis.

        Returns a flat dict with the fields used as ML features:
            buys_5m, sells_5m            — trade counts in last 5 minutes
            buy_volume_1h, total_volume_1h — USD volume in last hour
            price_change_5m_pct          — % price change last 5 minutes
            liquidity_change_1h_pct      — % liquidity change last hour

        Returns None on any error so callers fall back to neutral features.
        """
        if not self._cfg.moralis_api_key:
            return None

        pair_address = await self._resolve_pair_address(mint)
        if not pair_address:
            return None

        url = f"{self._cfg.moralis_base_url}/token/mainnet/pairs/{pair_address}/stats"
        headers = {"X-Api-Key": self._cfg.moralis_api_key}

        try:
            async with self._session.get(
                url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    logger.warning("[Moralis] pair stats HTTP %d for %s", resp.status, mint)
                    return None

                data = await resp.json()
                buys  = data.get("buys")  or {}
                sells = data.get("sells") or {}
                total_vol = data.get("totalVolume") or {}
                buy_vol   = data.get("buyVolume")   or {}
                price_chg = data.get("pricePercentChange") or {}
                liq_chg   = data.get("liquidityPercentChange") or {}

                return {
                    "buys_5m":                buys.get("5min", 0),
                    "sells_5m":               sells.get("5min", 0),
                    "buy_volume_1h":          buy_vol.get("1h", 0.0),
                    "total_volume_1h":        total_vol.get("1h", 0.0),
                    "price_change_5m_pct":    price_chg.get("5min", 0.0),
                    "liquidity_change_1h_pct": liq_chg.get("1h", 0.0),
                }

        except asyncio.TimeoutError:
            logger.warning("[Moralis] pair stats timeout for %s", mint)
        except aiohttp.ClientError as exc:
            logger.warning("[Moralis] pair stats error for %s: %s", mint, exc)
        except Exception as exc:
            logger.error("[Moralis] pair stats unexpected error for %s: %s", mint, exc)

        return None

    # ------------------------------------------------------------------
    # OHLCV fetch
    # ------------------------------------------------------------------

    async def get_ohlcv(
        self,
        mint: str,
        bars: int = 100,
        interval: str = "10s",
        time_to: Optional[int] = None,
    ) -> list[OHLCVCandle]:
        """
        Fetch the last `bars` OHLCV candles at `interval` resolution.

        Returns an empty list on any error so callers fall back gracefully.

        interval — Moralis timeframe string: "10s", "30s", "1min", etc.
        time_to  — Unix timestamp upper bound (defaults to now).
        """
        if not self._cfg.moralis_api_key:
            logger.warning("[Moralis] MORALIS_API_KEY not set — skipping")
            return []

        if time_to is None:
            time_to = int(time.time())

        pair_address = await self._resolve_pair_address(mint)
        if not pair_address:
            # stats already recorded by _resolve_via_moralis / _resolve_via_jupiter
            logger.warning("[Moralis] %s — pair unresolvable via Moralis+Jupiter, falling back to Birdeye", mint)
            return []

        secs      = _INTERVAL_SECONDS.get(interval, 10)
        time_from = time_to - bars * secs

        url = f"{self._cfg.moralis_base_url}/token/mainnet/pairs/{pair_address}/ohlcv"
        params = {
            "fromDate":  str(time_from),
            "toDate":    str(time_to),
            "timeframe": interval,
            "currency":  "usd",
            "limit":     str(bars),
        }
        headers = {"X-Api-Key": self._cfg.moralis_api_key}

        try:
            async with self._session.get(
                url,
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status == 401:
                    logger.error(
                        "[Moralis] 401 Unauthorized for %s — check MORALIS_API_KEY", mint
                    )
                    return []
                if resp.status == 429:
                    logger.warning("[Moralis] 429 rate limit for %s", mint)
                    return []
                if resp.status != 200:
                    logger.warning("[Moralis] HTTP %d for %s", resp.status, mint)
                    return []

                data  = await resp.json()
                items = data.get("result") or []

                candles: list[OHLCVCandle] = []
                for item in items:
                    ts = item.get("timestamp")
                    if isinstance(ts, str):
                        # ISO-8601 string, e.g. "2026-03-15T18:10:00.000Z"
                        unix_time = int(
                            datetime.fromisoformat(
                                ts.replace("Z", "+00:00")
                            ).timestamp()
                        )
                    else:
                        unix_time = int(ts)

                    candles.append(OHLCVCandle(
                        unix_time=unix_time,
                        open=float(item.get("open",   0)),
                        high=float(item.get("high",   0)),
                        low=float(item.get("low",    0)),
                        close=float(item.get("close",  0)),
                        volume=float(item.get("volume", 0)),
                    ))

                # Sort ascending and keep only the most recent `bars`
                candles.sort(key=lambda c: c.unix_time)
                if not candles:
                    logger.warning("[Moralis] %s — pair found (%s) but OHLCV returned 0 candles", mint, pair_address)
                    self.stats.record("OHLCV_EMPTY", mint, f"pair={pair_address}")
                else:
                    logger.debug("[Moralis] %s — %d × %s candles", mint, len(candles), interval)
                return candles[-bars:]

        except asyncio.TimeoutError:
            logger.warning("[Moralis] Timeout for %s", mint)
            self.stats.record("HTTP_ERROR", mint, "timeout")
        except aiohttp.ClientError as exc:
            logger.warning("[Moralis] HTTP error for %s: %s", mint, exc)
            self.stats.record("HTTP_ERROR", mint, str(exc))
        except Exception as exc:
            logger.error("[Moralis] Unexpected error for %s: %s", mint, exc)
            self.stats.record("HTTP_ERROR", mint, str(exc))

        return []
