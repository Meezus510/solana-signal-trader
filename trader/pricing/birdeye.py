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
import json
import logging
import time
from typing import Callable, Coroutine, Optional

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
    # Birdeye's v1 OHLCV endpoint minimum granularity is 1m.
    # Sub-minute candles are available via the v3 endpoint (see get_ohlcv_v3).
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

    # Seconds per bar for sub-minute intervals supported by the v3 OHLCV endpoint.
    # Data retention: 1s = 2 weeks, 15s/30s = 3 months.
    _V3_INTERVAL_SECONDS: dict[str, int] = {"1s": 1, "15s": 15, "30s": 30}

    async def get_ohlcv_v3(
        self,
        mint_address: str,
        bars: int = 100,
        interval: str = "15s",
        time_to: Optional[int] = None,
    ) -> list[OHLCVCandle]:
        """
        Fetch the last `bars` sub-minute OHLCV candles via Birdeye's v3 endpoint.

        Replaces Moralis for sub-minute ML feature extraction (features 1-6).
        Supported intervals: "1s" (2wk retention), "15s" (3mo retention),
        "30s" (3mo retention).

        Endpoint: GET /defi/v3/ohlcv
        Response items have unix_time (not unixTime), o, h, l, c, v fields.

        Returns an empty list on any error so callers can safely fall back to
        neutral ML features rather than blocking the signal.
        """
        if time_to is None:
            time_to = int(time.time())
        secs = self._V3_INTERVAL_SECONDS.get(interval, 15)
        time_from = time_to - (bars + 5) * secs
        url = f"{self._cfg.birdeye_base_url}/defi/v3/ohlcv"
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
                    logger.warning("[OHLCVv3] 401 Unauthorized for %s — sub-minute features will be neutral", mint_address)
                    return []
                if resp.status == 429:
                    logger.warning("[OHLCVv3] 429 rate limit for %s — sub-minute features will be neutral", mint_address)
                    return []
                if resp.status != 200:
                    logger.warning("[OHLCVv3] HTTP %d for %s — sub-minute features will be neutral", resp.status, mint_address)
                    return []

                data = await resp.json()
                items = (data.get("data") or {}).get("items") or []
                candles = [
                    OHLCVCandle(
                        unix_time=int(item.get("unix_time", 0)),
                        open=float(item.get("o", 0)),
                        high=float(item.get("h", 0)),
                        low=float(item.get("l", 0)),
                        close=float(item.get("c", 0)),
                        volume=float(item.get("v", 0)),
                    )
                    for item in items
                    if item.get("l") is not None
                ]
                logger.debug("[OHLCVv3] %s — %d candles fetched (%s)", mint_address, len(candles), interval)
                return candles[-bars:]

        except asyncio.TimeoutError:
            logger.warning("[OHLCVv3] Timeout for %s — sub-minute features will be neutral", mint_address)
        except aiohttp.ClientError as exc:
            logger.warning("[OHLCVv3] HTTP error for %s: %s — sub-minute features will be neutral", mint_address, exc)
        except Exception as exc:
            logger.error("[OHLCVv3] Unexpected error for %s: %s", mint_address, exc)

        return []

    async def get_token_overview(
        self,
        mint_address: str,
    ) -> Optional[dict]:
        """
        Fetch token metadata from Birdeye's token overview endpoint.

        Returns a flat dict with fields merged into pair_stats for ML features:
            market_cap_usd        — USD market cap at signal time (None if unavailable)
            liquidity_usd         — total USD liquidity across all pools
            holder_count          — number of unique holders
            unique_wallet_5m      — distinct wallets that traded in last 5 min
            unique_wallet_hist_5m — distinct wallets that traded 5–10 min ago
            unique_wallet_30m     — distinct wallets that traded in last 30 min
            unique_wallet_hist_30m— distinct wallets that traded 30–60 min ago
            price_change_30m_pct  — price % change over last 30 min
            buy_volume_usd_5m     — buy-side USD volume in last 5 min
            sell_volume_usd_5m    — sell-side USD volume in last 5 min

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
                mc    = data.get("marketCap") or data.get("realMc")
                liq   = data.get("liquidity")
                hld   = data.get("holder")
                uw5   = data.get("uniqueWallet5m")
                uwh5  = data.get("uniqueWalletHistory5m")
                uw30  = data.get("uniqueWallet30m")
                uwh30 = data.get("uniqueWalletHistory30m")
                pc30  = data.get("priceChange30mPercent")
                vb5   = data.get("vBuy5mUSD")
                vs5   = data.get("vSell5mUSD")
                if mc is None and liq is None and hld is None and uw5 is None:
                    logger.debug("[TokenOverview] no usable metadata for %s", mint_address)
                    return None

                return {
                    "market_cap_usd":         float(mc)    if mc    is not None else None,
                    "liquidity_usd":          float(liq)   if liq   is not None else None,
                    "holder_count":           int(hld)     if hld   is not None else None,
                    "unique_wallet_5m":       int(uw5)     if uw5   is not None else None,
                    "unique_wallet_hist_5m":  int(uwh5)    if uwh5  is not None else None,
                    "unique_wallet_30m":      int(uw30)    if uw30  is not None else None,
                    "unique_wallet_hist_30m": int(uwh30)   if uwh30 is not None else None,
                    "price_change_30m_pct":   float(pc30)  if pc30  is not None else None,
                    "buy_volume_usd_5m":      float(vb5)   if vb5   is not None else None,
                    "sell_volume_usd_5m":     float(vs5)   if vs5   is not None else None,
                }

        except asyncio.TimeoutError:
            logger.warning("[TokenOverview] timeout for %s", mint_address)
        except aiohttp.ClientError as exc:
            logger.warning("[TokenOverview] HTTP error for %s: %s", mint_address, exc)
        except Exception as exc:
            logger.error("[TokenOverview] unexpected error for %s: %s", mint_address, exc)

        return None

    async def get_token_security(
        self,
        mint_address: str,
    ) -> Optional[dict]:
        """
        Fetch token security data from Birdeye's security endpoint.

        Returns a flat dict merged into pair_stats for persistence and future ML use.
        All fields are prefixed with 'security_' except top10_concentration which
        is already an active ML feature (idx 26).

        Fields returned:
            top10_concentration       — fraction [0–1] of supply held by top 10 wallets (ML feature)
            security_freezeable       — True if freeze authority exists (rug vector)
            security_transfer_fee     — True if transfer tax is enabled (honeypot signal)
            security_owner_pct        — fraction of supply held by current owner (None if renounced)
            security_creator_pct      — fraction of supply still held by creator
            security_mutable_metadata — True if token metadata can still be changed
            security_jup_strict       — True if token is on Jupiter's strict verified list
            security_is_token2022     — True if Token-2022 standard (supports hidden fees)
            security_non_transferable — True if tokens cannot be transferred (honeypot)
            security_pre_market_holders — number of wallets that held tokens pre-launch
            security_token_age_hours  — age of token in hours at signal time

        Returns None on any error so callers fall back gracefully.

        Endpoint: GET /defi/token_security?address=<mint>
        """
        url = f"{self._cfg.birdeye_base_url}/defi/token_security"
        params = {"address": mint_address}
        try:
            async with self._session.get(
                url,
                headers=self._headers(),
                params=params,
                timeout=aiohttp.ClientTimeout(total=self._cfg.request_timeout_seconds),
            ) as resp:
                if resp.status == 401:
                    logger.warning("[TokenSecurity] 401 Unauthorized for %s — skipping", mint_address)
                    return None
                if resp.status == 429:
                    logger.warning("[TokenSecurity] 429 rate limit for %s — skipping", mint_address)
                    return None
                if resp.status != 200:
                    logger.warning("[TokenSecurity] HTTP %d for %s — skipping", resp.status, mint_address)
                    return None

                data = (await resp.json()).get("data") or {}

                top10 = data.get("top10HolderPercent")
                owner_pct = data.get("ownerPercentage")
                creator_pct = data.get("creatorPercentage")
                creation_time = data.get("creationTime")
                pre_market = data.get("preMarketHolder") or []

                age_hours: Optional[float] = None
                if creation_time:
                    age_hours = (time.time() - float(creation_time)) / 3600.0

                return {
                    "top10_concentration":          max(0.0, min(1.0, float(top10))) if top10 is not None else None,
                    "security_freezeable":          bool(data.get("freezeable"))      if data.get("freezeable") is not None else None,
                    "security_transfer_fee":        bool(data.get("transferFeeEnable")) if data.get("transferFeeEnable") is not None else None,
                    "security_owner_pct":           float(owner_pct)                  if owner_pct is not None else None,
                    "security_creator_pct":         float(creator_pct)               if creator_pct is not None else None,
                    "security_mutable_metadata":    bool(data.get("mutableMetadata")) if data.get("mutableMetadata") is not None else None,
                    "security_jup_strict":          bool(data.get("jupStrictList"))   if data.get("jupStrictList") is not None else None,
                    "security_is_token2022":        bool(data.get("isToken2022"))     if data.get("isToken2022") is not None else None,
                    "security_non_transferable":    bool(data.get("nonTransferable")) if data.get("nonTransferable") is not None else None,
                    "security_pre_market_holders":  len(pre_market),
                    "security_token_age_hours":     round(age_hours, 1)              if age_hours is not None else None,
                }

        except asyncio.TimeoutError:
            logger.warning("[TokenSecurity] timeout for %s", mint_address)
        except aiohttp.ClientError as exc:
            logger.warning("[TokenSecurity] HTTP error for %s: %s", mint_address, exc)
        except Exception as exc:
            logger.error("[TokenSecurity] unexpected error for %s: %s", mint_address, exc)

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


# ---------------------------------------------------------------------------
# WebSocket client — real-time price streaming
# ---------------------------------------------------------------------------

PriceCallback = Callable[[str, float], Coroutine]


class BirdeyeWebSocketClient:
    """
    Birdeye WebSocket client for real-time token price streaming.

    Replaces the REST polling loop in TradingEngine.monitor_positions_ws():
        client = BirdeyeWebSocketClient(api_key)
        asyncio.create_task(client.run())          # start connection loop
        await client.subscribe(mint, callback)     # callback(mint, price) coroutine
        await client.unsubscribe(mint)             # when position closes

    Auto-reconnects on disconnect and re-subscribes all active mints.
    Each price update is dispatched as a new asyncio task so slow callbacks
    (e.g. Jupiter quote fetch) never block the WebSocket reader.
    """

    _WS_URL = "wss://public-api.birdeye.so/socket"
    _RECONNECT_DELAY = 5.0

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._callbacks: dict[str, PriceCallback] = {}
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None

    async def subscribe(self, mint: str, callback: PriceCallback) -> None:
        """Subscribe to price updates for a mint. Safe to call before run()."""
        self._callbacks[mint] = callback
        if self._ws and not self._ws.closed:
            await self._send_subscribe(mint)

    async def unsubscribe(self, mint: str) -> None:
        """Stop receiving price updates for a mint."""
        self._callbacks.pop(mint, None)
        if self._ws and not self._ws.closed:
            await self._ws.send_json({"type": "UNSUBSCRIBE_PRICE", "data": {"address": mint}})

    async def run(self) -> None:
        """Maintain WebSocket connection. Run as an asyncio task."""
        headers = {
            "X-API-KEY": self._api_key,
            "x-chain": "solana",
        }
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    async with session.ws_connect(self._WS_URL, headers=headers, heartbeat=30) as ws:
                        self._ws = ws
                        logger.info("[WS] Birdeye WebSocket connected (%d subscriptions)", len(self._callbacks))
                        for mint in list(self._callbacks):
                            await self._send_subscribe(mint)
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                await self._dispatch(msg.data)
                            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                logger.warning("[WS] Birdeye WebSocket %s", msg.type.name)
                                break
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.error("[WS] Birdeye WebSocket error: %s — reconnecting in %.0fs", exc, self._RECONNECT_DELAY)
                    await asyncio.sleep(self._RECONNECT_DELAY)

    async def _send_subscribe(self, mint: str) -> None:
        await self._ws.send_json({
            "type": "SUBSCRIBE_PRICE",
            "data": {"chartType": "1s", "address": mint, "currency": "usd"},
        })

    async def _dispatch(self, raw: str) -> None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return
        if data.get("type") != "PRICE_DATA":
            return
        payload = data.get("data") or {}
        mint: Optional[str] = payload.get("address")
        value = payload.get("value")
        if mint and value is not None and mint in self._callbacks:
            asyncio.create_task(self._callbacks[mint](mint, float(value)))
