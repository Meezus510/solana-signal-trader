"""
trader/pricing/jupiter.py — Jupiter v6 quote client.

Provides an async quote for selling a token via Jupiter's swap aggregator.
Used by the monitoring loop to get a production-accurate exit price that
reflects real AMM slippage and on-chain liquidity depth.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

_JUPITER_QUOTE_HOST = "https://quote-api.jup.ag"
_USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
_SOLANA_RPC_URL = "https://api.mainnet-beta.solana.com"
_DEFAULT_SLIPPAGE_BPS = 300   # 3% — reasonable for meme coins


async def jupiter_quote_exit_price(
    session: aiohttp.ClientSession,
    input_mint: str,
    quantity: float,
    token_decimals: int = 6,
    slippage_bps: int = _DEFAULT_SLIPPAGE_BPS,
) -> Optional[float]:
    """
    Query Jupiter v6 for the effective USD exit price when selling `quantity` tokens.

    Returns USD-per-token (i.e. the simulated fill price including AMM slippage),
    or None if the quote fails. Used as the source-of-truth exit price for
    strategies with use_real_exit_price=True.
    """
    amount = int(quantity * 10 ** token_decimals)
    if amount <= 0:
        return None
    api_key = os.getenv("JUPITER_API_KEY", "")
    url = (
        f"{_JUPITER_QUOTE_HOST}/v6/quote"
        f"?inputMint={input_mint}&outputMint={_USDC_MINT}"
        f"&amount={amount}&slippageBps={slippage_bps}"
    )
    headers = {"x-api-key": api_key} if api_key else {}
    try:
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if resp.status != 200:
                logger.warning("[JUPITER] Quote HTTP %d for %s", resp.status, input_mint)
                return None
            data = await resp.json()
            out_amount = data.get("outAmount")
            if out_amount is None:
                return None
            usdc_received = int(out_amount) / 1_000_000  # USDC has 6 decimals
            return usdc_received / quantity
    except Exception as exc:
        logger.warning("[JUPITER] Quote failed for %s: %s", input_mint, exc)
        return None


async def fetch_token_decimals(
    session: aiohttp.ClientSession,
    mint_address: str,
    default: int = 6,
) -> int:
    """
    Fetch SPL token decimals from Solana RPC getTokenSupply.

    Returns `default` on any failure so callers never block on bad RPC responses.
    Results should be cached by the caller — this makes one RPC call per invocation.
    """
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getTokenSupply",
        "params": [mint_address],
    }
    try:
        async with session.post(
            _SOLANA_RPC_URL,
            json=body,
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            if resp.status != 200:
                logger.warning("[DECIMALS] RPC HTTP %d for %s", resp.status, mint_address)
                return default
            data = await resp.json()
            decimals = (data.get("result") or {}).get("value", {}).get("decimals")
            if decimals is None:
                return default
            return int(decimals)
    except Exception as exc:
        logger.warning("[DECIMALS] Failed to fetch decimals for %s: %s", mint_address, exc)
        return default
