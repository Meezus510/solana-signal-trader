import logging
import os
import requests
import time
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

class RestClient():

    def __init__(self, timeout=10, retry=3):
        self.timeout = timeout
        self.retry = retry
        self.session = requests.Session()

    def _make_request(self, method, url, **kwargs):
        for attempt in range(self.retry):
            try:
                response = self.session.request(method, url, timeout=self.timeout, **kwargs)

                if response.status_code >= 500:
                    raise Exception(f"internal server error:{url} - {response.status_code} - {response.json()}")

                return response

            except Exception as e:
                if attempt < self.retry - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise e

    def get(self, url, **kwargs):
        response = self._make_request("GET", url, **kwargs)

        if response.status_code >= 400:
            raise Exception(f"API Error:{url} - {response.status_code} - {response.json()}")

        return response

    def post(self, url, **kwargs):
        response = self._make_request("POST", url, **kwargs)

        if response.status_code >= 400:
            raise Exception(f"API Error:{url} - {response.status_code} - {response.json()}")
        
        return response

restClient = RestClient()

JUPITER_API_KEY = os.getenv("JUPITER_API_KEY", "")
JUPITER_HOST = "https://api.jup.ag"
JUPITER_HOST_QUOTE = "https://quote-api.jup.ag"
JUPITER_HOST_LIMIT = "https://api.jup.ag"
SOLANA_RPC_URL = "https://api.mainnet-beta.solana.com"
COIN_GECKO_HOST = "https://api.coingecko.com"

def jupiter_quote_api(input_mint, output_mint, amount, slippage):
    url = f"{JUPITER_HOST_QUOTE}/v6/quote?inputMint={input_mint}&outputMint={output_mint}&amount={amount}&slippageBps={slippage}"
    headers = {"x-api-key": JUPITER_API_KEY}

    return restClient.get(url, headers=headers)


_USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
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

    The output mint is USDC so the quote reflects real on-chain liquidity.
    """
    amount = int(quantity * 10 ** token_decimals)
    if amount <= 0:
        return None
    url = (
        f"{JUPITER_HOST_QUOTE}/v6/quote"
        f"?inputMint={input_mint}&outputMint={_USDC_MINT}"
        f"&amount={amount}&slippageBps={slippage_bps}"
    )
    headers = {"x-api-key": JUPITER_API_KEY} if JUPITER_API_KEY else {}
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

def jupiter_swap_api(quote_response, wallet_id, prioritization_fee_lamports):
    url = f"{JUPITER_HOST}/v6/swap"
    headers = {"Authorization": f"Bearer {JUPITER_API_KEY}"}
    request_body = {
        "quoteResponse": quote_response,
        "userPublicKey": wallet_id,
        "wrapAndUnwrapSol": True,
        "dynamicComputeUnitLimit": True,
        "dynamicSlippage": True,
        "prioritizationFeeLamports": prioritization_fee_lamports
    }

    return restClient.post(url, headers=headers, json=request_body)

def jupiter_limit_api(input_mint, output_mint, wallet_id, making_amount, taking_amount, expired_at):
    url = f"{JUPITER_HOST_LIMIT}/limit/v2/createOrder"
    headers = {"Authorization": f"Bearer {JUPITER_API_KEY}"}
    request_body = {
        "inputMint": input_mint,
        "outputMint": output_mint,
        "maker": wallet_id,
        "payer": wallet_id,
        "params": {
            "makingAmount": making_amount,
            "takingAmount": taking_amount,
            "expiredAt": expired_at
        },
        "computeUnitPrice": "auto"
    }

    return restClient.post(url, headers=headers, json=request_body)

def solana_rpc_api(method, params):
    request_body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params
    }

    return restClient.post(SOLANA_RPC_URL, json=request_body)

def solana_get_balance(wallet_id):
    params = [wallet_id]

    return solana_rpc_api("getBalance", params)

def solana_get_token_accounts(wallet_id, mint_address):
    params = [
        wallet_id,
        { "mint": mint_address },
        { "encoding": "jsonParsed" }
    ]
    
    return solana_rpc_api("getTokenAccountsByOwner", params)

def solana_get_token_supply(mint_address):
    params = [mint_address]

    return solana_rpc_api("getTokenSupply", params)

def solana_get_prioritization_fees():
    params = []

    return solana_rpc_api("getRecentPrioritizationFees", params)

def solana_send_raw_transaction(transaction):
    params = [transaction]

    return solana_rpc_api("sendTransaction", params)

def solana_get_signature_statuses(signature):
    params = [[signature]]

    return solana_rpc_api("getSignatureStatuses", params)

async def fetch_token_decimals_async(
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
            SOLANA_RPC_URL,
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


def coin_gecko_price(ids="solana",vs_currencies="usd"):
    url = f"{COIN_GECKO_HOST}/api/v3/simple/price?ids={ids}&vs_currencies={vs_currencies}"

    return restClient.get(url)

