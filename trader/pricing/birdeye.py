"""
trader/pricing/birdeye.py — Birdeye REST price client.

Provides real-time Solana token prices via Birdeye's public API.
Two access patterns are supported:

    get_price(mint)              — single token, used for position entry
    get_prices_batch(mints)      — multi-token, used for position monitoring
                                   (automatically disabled for the session if
                                   the endpoint returns 401 Unauthorized)

Rate-limit behaviour:
    - 401 Unauthorized: treated as permanent — batch endpoint is disabled for
      the session and individual fetches stop retrying immediately. No spam.
    - 429 Too Many Requests: backs off with increasing wait times (10s, 30s)
      and retries at most once. If still rate-limited, returns None and lets
      the next poll cycle try again.

Swap point:
    To move from REST polling to WebSocket streaming, add a subscribe()
    method here and remove the polling loop in TradingEngine.monitor_positions().
    evaluate_position() needs no changes regardless of the price delivery mechanism.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import aiohttp

from trader.analysis.chart import OHLCVCandle
from trader.config import Config

logger = logging.getLogger(__name__)

# How long to wait after a 429 before retrying (seconds)
_RATE_LIMIT_BACKOFF = [10.0, 30.0]


class BirdeyePriceClient:
    """
    Async HTTP client for the Birdeye token price API.

    The aiohttp.ClientSession is injected so it can be shared across the
    application (one TCP connection pool for the process lifetime) and
    closed cleanly at shutdown.

    Session-level state:
        _batch_disabled  — set True on first 401 from the batch endpoint.
                           Avoids retrying an endpoint that will never work
                           with the current API key tier.
    """

    def __init__(self, cfg: Config, session: aiohttp.ClientSession) -> None:
        self._cfg = cfg
        self._session = session
        self._batch_disabled: bool = False  # latched True on batch 401

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "X-API-KEY": self._cfg.birdeye_api_key,
            "x-chain": "solana",
            "accept": "application/json",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_price(
        self,
        mint_address: str,
        retries: int = 1,
    ) -> Optional[float]:
        """
        Fetch the current USD price for a single Solana token mint.

        Error handling:
            401 — API key unauthorised: returns None immediately, no retry.
            429 — Rate limited: backs off and retries once with a long wait.
            Other 4xx/5xx — logs and retries up to `retries` times.

        Endpoint: GET /defi/price?address=<mint>
        """
        url = f"{self._cfg.birdeye_base_url}/defi/price"
        params = {"address": mint_address}

        for attempt in range(retries + 1):
            try:
                async with self._session.get(
                    url,
                    headers=self._headers(),
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=self._cfg.request_timeout_seconds),
                ) as resp:

                    if resp.status == 401:
                        logger.error(
                            "[ERROR] Birdeye 401 Unauthorized for %s — "
                            "check BIRDEYE_API_KEY in .env", mint_address,
                        )
                        return None  # auth won't change, no point retrying

                    if resp.status == 429:
                        wait = _RATE_LIMIT_BACKOFF[min(attempt, len(_RATE_LIMIT_BACKOFF) - 1)]
                        logger.warning(
                            "[ERROR] Birdeye 429 rate limit for %s — "
                            "backing off %.0fs (attempt %d/%d)",
                            mint_address, wait, attempt + 1, retries + 1,
                        )
                        if attempt < retries:
                            await asyncio.sleep(wait)
                            continue
                        return None

                    if resp.status != 200:
                        logger.warning(
                            "[ERROR] Birdeye single-price HTTP %d for %s",
                            resp.status, mint_address,
                        )
                        if attempt < retries:
                            await asyncio.sleep(2.0)
                        continue

                    data = await resp.json()
                    value = (data.get("data") or {}).get("value")
                    if value is None:
                        logger.warning(
                            "[ERROR] No price value in response for %s", mint_address
                        )
                        return None
                    return float(value)

            except asyncio.TimeoutError:
                logger.warning(
                    "[ERROR] Timeout for %s (attempt %d/%d)",
                    mint_address, attempt + 1, retries + 1,
                )
            except aiohttp.ClientError as exc:
                logger.warning(
                    "[ERROR] HTTP error for %s: %s (attempt %d/%d)",
                    mint_address, exc, attempt + 1, retries + 1,
                )
            except Exception as exc:
                logger.error("[ERROR] Unexpected error for %s: %s", mint_address, exc)
                return None

            if attempt < retries:
                await asyncio.sleep(2.0)

        logger.warning("[ERROR] Giving up on price for %s this cycle", mint_address)
        return None

    async def get_prices_batch(
        self,
        mint_addresses: list[str],
    ) -> dict[str, Optional[float]]:
        """
        Fetch prices for multiple mints in a single Birdeye multi-price request.

        If the batch endpoint returns 401, it is disabled permanently for this
        session (_batch_disabled = True) and subsequent calls fall through to
        individual fetches without attempting the batch endpoint again.

        If the batch endpoint returns 429 or any other error, this cycle is
        skipped (returns None for all mints) rather than immediately firing
        individual requests that would compound the rate-limit problem.

        Endpoint: GET /defi/multi_price?list_address=mint1,mint2,...
        """
        if not mint_addresses:
            return {}

        if self._batch_disabled:
            return await self._individual_with_rate_limit(mint_addresses)

        url = f"{self._cfg.birdeye_base_url}/defi/multi_price"
        params = {"list_address": ",".join(mint_addresses)}

        try:
            async with self._session.get(
                url,
                headers=self._headers(),
                params=params,
                timeout=aiohttp.ClientTimeout(total=self._cfg.request_timeout_seconds),
            ) as resp:

                if resp.status == 401:
                    logger.error(
                        "[ERROR] Birdeye batch endpoint 401 Unauthorized — "
                        "disabling batch for this session. "
                        "Your API key may not have access to /defi/multi_price. "
                        "Switching to individual price fetches."
                    )
                    self._batch_disabled = True
                    return await self._individual_with_rate_limit(mint_addresses)

                if resp.status == 429:
                    logger.warning(
                        "[ERROR] Birdeye batch 429 rate limit — "
                        "skipping this poll cycle to avoid compounding rate limits."
                    )
                    return {m: None for m in mint_addresses}

                if resp.status != 200:
                    logger.warning(
                        "[ERROR] Birdeye batch HTTP %d — skipping cycle", resp.status
                    )
                    return {m: None for m in mint_addresses}

                data = await resp.json()
                raw: dict = data.get("data") or {}
                result: dict[str, Optional[float]] = {}

                for mint in mint_addresses:
                    entry = raw.get(mint)
                    if entry and entry.get("value") is not None:
                        result[mint] = float(entry["value"])
                    else:
                        result[mint] = None
                        logger.warning("[ERROR] No batch price for %s", mint)

                return result

        except asyncio.TimeoutError:
            logger.warning("[ERROR] Timeout on batch fetch — skipping cycle")
        except aiohttp.ClientError as exc:
            logger.warning("[ERROR] HTTP batch error: %s — skipping cycle", exc)
        except Exception as exc:
            logger.error("[ERROR] Unexpected batch error: %s — skipping cycle", exc)

        return {m: None for m in mint_addresses}

    # Seconds per bar for each supported interval type.
    # Birdeye's OHLCV endpoint minimum granularity is 1m — sub-minute
    # intervals (1s, 5s, 15s, 30s) return 400 "type invalid format".
    _INTERVAL_SECONDS: dict[str, int] = {
        "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
        "1h": 3600, "2h": 7200, "4h": 14400,
        "6h": 21600, "8h": 28800, "12h": 43200,
        "1d": 86400,
    }

    async def get_ohlcv(
        self,
        mint_address: str,
        bars: int = 20,
        interval: str = "1m",
        time_to: Optional[int] = None,
    ) -> list[OHLCVCandle]:
        """
        Fetch the last `bars` OHLCV candles for a token at the given interval.

        Returns an empty list on any error so callers can safely fall back to
        entering without chart data rather than blocking the signal.

        time_to — optional Unix timestamp upper bound (defaults to now).
                   Pass a historical timestamp to fetch candles as they looked
                   at a specific point in time (used for backtesting).

        Endpoint: GET /defi/ohlcv?address=<mint>&type=<interval>&time_from=<ts>&time_to=<ts>
        """
        if time_to is None:
            time_to = int(time.time())
        secs = self._INTERVAL_SECONDS.get(interval, 60)
        # fetch a slightly wider window to guarantee we have `bars` full candles
        time_from = time_to - (bars + 5) * secs
        url = f"{self._cfg.birdeye_base_url}/defi/ohlcv"
        params = {
            "address": mint_address,
            "type": interval,
            "time_from": time_from,
            "time_to": time_to,
        }

        try:
            async with self._session.get(
                url,
                headers=self._headers(),
                params=params,
                timeout=aiohttp.ClientTimeout(total=self._cfg.request_timeout_seconds),
            ) as resp:
                if resp.status == 401:
                    logger.warning("[OHLCV] 401 Unauthorized for %s — chart filter disabled this signal", mint_address)
                    return []
                if resp.status == 429:
                    logger.warning("[OHLCV] 429 rate limit for %s — skipping chart filter", mint_address)
                    return []
                if resp.status != 200:
                    logger.warning("[OHLCV] HTTP %d for %s — skipping chart filter", resp.status, mint_address)
                    return []

                data = await resp.json()
                items = (data.get("data") or {}).get("items") or []
                candles = [
                    OHLCVCandle(
                        unix_time=int(item.get("unixTime", 0)),
                        open=float(item.get("o", 0)),
                        high=float(item.get("h", 0)),
                        low=float(item.get("l", 0)),
                        close=float(item.get("c", 0)),
                        volume=float(item.get("v", 0)),
                    )
                    for item in items
                    if item.get("l") is not None
                ]
                logger.debug("[OHLCV] %s — %d candles fetched", mint_address, len(candles))
                return candles[-bars:]  # keep only the most recent `bars`

        except asyncio.TimeoutError:
            logger.warning("[OHLCV] Timeout for %s — skipping chart filter", mint_address)
        except aiohttp.ClientError as exc:
            logger.warning("[OHLCV] HTTP error for %s: %s — skipping chart filter", mint_address, exc)
        except Exception as exc:
            logger.error("[OHLCV] Unexpected error for %s: %s", mint_address, exc)

        return []

    async def get_token_overview(
        self,
        mint_address: str,
    ) -> Optional[dict]:
        """
        Fetch token metadata from Birdeye's token overview endpoint.

        Returns a flat dict with fields merged into pair_stats for ML features:
            market_cap_usd   — USD market cap at signal time (None if unavailable)
            liquidity_usd    — total USD liquidity across all pools
            holder_count     — number of unique holders

        Returns None on any error so callers fall back to neutral ML features.

        Endpoint: GET /defi/token_overview?address=<mint>
        """
        url = f"{self._cfg.birdeye_base_url}/defi/token_overview"
        params = {"address": mint_address}
        try:
            async with self._session.get(
                url,
                headers=self._headers(),
                params=params,
                timeout=aiohttp.ClientTimeout(total=self._cfg.request_timeout_seconds),
            ) as resp:
                if resp.status == 401:
                    logger.warning("[TokenOverview] 401 Unauthorized for %s — skipping metadata", mint_address)
                    return None
                if resp.status == 429:
                    logger.warning("[TokenOverview] 429 rate limit for %s — skipping metadata", mint_address)
                    return None
                if resp.status != 200:
                    logger.warning("[TokenOverview] HTTP %d for %s — skipping metadata", resp.status, mint_address)
                    return None

                data = (await resp.json()).get("data") or {}
                mc  = data.get("marketCap") or data.get("realMc")
                liq = data.get("liquidity")
                hld = data.get("holder")

                if mc is None and liq is None and hld is None:
                    logger.debug("[TokenOverview] no usable metadata for %s", mint_address)
                    return None

                return {
                    "market_cap_usd": float(mc)  if mc  is not None else None,
                    "liquidity_usd":  float(liq) if liq is not None else None,
                    "holder_count":   int(hld)   if hld is not None else None,
                }

        except asyncio.TimeoutError:
            logger.warning("[TokenOverview] timeout for %s", mint_address)
        except aiohttp.ClientError as exc:
            logger.warning("[TokenOverview] HTTP error for %s: %s", mint_address, exc)
        except Exception as exc:
            logger.error("[TokenOverview] unexpected error for %s: %s", mint_address, exc)

        return None

    async def _individual_with_rate_limit(
        self,
        mint_addresses: list[str],
    ) -> dict[str, Optional[float]]:
        """
        Fetch prices one-by-one with a small delay between each request
        to stay well within per-minute rate limits.

        Used when batch is disabled (e.g. API key tier doesn't support it).
        With a 15s poll interval and a gap between individual requests,
        even 10 open positions use only ~40 req/min.
        """
        result: dict[str, Optional[float]] = {}
        for mint in mint_addresses:
            result[mint] = await self.get_price(mint, retries=0)
            # Small inter-request gap — prevents bursting the rate limit
            # when many positions are open
            if len(mint_addresses) > 1:
                await asyncio.sleep(1.0)
        return result
