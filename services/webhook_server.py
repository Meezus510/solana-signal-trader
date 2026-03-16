"""
solana_api/my_bot_script.py

Quart webhook server for Solana token trading via Jupiter.
Receives buy/sell signals via POST /webhook and executes:
  - Market swap  (perform_swap)         for position entry
  - Limit orders (create_limit_order)   for take-profit exits

Environment variables (see .env.example):
  JUPITER_API_KEY, WALLET_ID, PRIVATE_KEY,
  SOLANA_RPC_URL, CMC_API_KEY, SWAP_AMOUNT, RETRY_COUNT
"""

from __future__ import annotations

import asyncio
import base58
import base64
import json
import logging
import os
import re
import time

import requests
from dotenv import load_dotenv
from quart import Quart, request, jsonify
from quart_cors import cors

from solana.rpc.api import Client
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Processed
from solana.rpc.types import TxOpts
from solders.instruction import AccountMeta, Instruction
from solders.keypair import Keypair
from solders import message
from solders.pubkey import Pubkey
from solders.signature import Signature
from solders.transaction import VersionedTransaction

from my_bot_environment import restart_server

# ---------------------------------------------------------------------------
# App & CORS
# ---------------------------------------------------------------------------

load_dotenv()

CHROME_EXTENSION_ORIGIN = "chrome-extension://beamfeaeoojdfafafeclgeijihdcbaed"
app = Quart(__name__)
app = cors(app, allow_origin=[CHROME_EXTENSION_ORIGIN])

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

file_logger = logging.getLogger("file_logger")
file_logger.setLevel(logging.INFO)
_fh = logging.FileHandler("file_log.log")
_fh.setLevel(logging.INFO)
_fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
file_logger.addHandler(_fh)

transaction_logger = logging.getLogger("transaction_logger")
transaction_logger.setLevel(logging.INFO)
_th = logging.FileHandler("transaction_log.log")
_th.setLevel(logging.INFO)
_th.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - Transaction: %(message)s"))
transaction_logger.addHandler(_th)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

JUPITER_API_KEY = os.getenv("JUPITER_API_KEY", "")
WALLET_ID = os.getenv("WALLET_ID")
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
SOLANA_RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
CMC_API_KEY = os.getenv("CMC_API_KEY", "")

try:
    SWAP_AMOUNT = float(os.getenv("SWAP_AMOUNT", "0.1"))
except ValueError:
    logger.warning("Invalid SWAP_AMOUNT in env — defaulting to 0.1")
    SWAP_AMOUNT = 0.1

try:
    RETRY_COUNT = int(os.getenv("RETRY_COUNT", "1"))
except ValueError:
    logger.warning("Invalid RETRY_COUNT in env — defaulting to 1")
    RETRY_COUNT = 1

# ---------------------------------------------------------------------------
# Solana program IDs & constants
# ---------------------------------------------------------------------------

SOL_TICKER = "So11111111111111111111111111111111111111112"
TOKEN_DECIMALS = 10**9  # lamports per SOL

TOKEN_PROGRAM_ID = Pubkey(base58.b58decode("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"))
SYS_PROGRAM_ID = Pubkey(base58.b58decode("11111111111111111111111111111111"))
SYSVAR_RENT_PUBKEY = Pubkey(base58.b58decode("SysvarRent111111111111111111111111111111111"))
ASSOCIATED_TOKEN_ADDRESS = Pubkey.from_string("D2z2ZVyKRnG5e2CCHvtoTRW6oKVyA3KnQZWrCLYjqwoA")

# ---------------------------------------------------------------------------
# Solana client & keypair (guarded against missing env vars)
# ---------------------------------------------------------------------------

client = Client(SOLANA_RPC_URL)

if WALLET_ID:
    payer = Pubkey.from_string(WALLET_ID)
    owner = payer
else:
    payer = owner = None
    logger.warning("WALLET_ID not set — payer/owner will be None until configured")

if PRIVATE_KEY:
    PRIVATE_KEY_PAIR = Keypair.from_bytes(base58.b58decode(PRIVATE_KEY))
else:
    PRIVATE_KEY_PAIR = None
    logger.warning("PRIVATE_KEY not set — transaction signing will fail until configured")

# ---------------------------------------------------------------------------
# Mutable runtime state
# ---------------------------------------------------------------------------

DESIRED_CRYPTO_TICKER: str | None = None
action_lock = asyncio.Lock()
buy_task: asyncio.Task | None = None
sell_task: asyncio.Task | None = None

# ---------------------------------------------------------------------------
# Price & balance helpers
# ---------------------------------------------------------------------------

def get_crypto_count(swap_quote: dict, token_decimals: int) -> dict:
    """Return SOL and non-SOL amounts (native units + lamports) from a Jupiter quote."""
    input_mint = swap_quote.get("inputMint")
    output_mint = swap_quote.get("outputMint")
    in_amount = int(swap_quote.get("inAmount", 0))
    out_amount = int(swap_quote.get("outAmount", 0))

    result = {"sol": None, "sol_lamports": None, "non_sol": None, "non_sol_lamports": None}

    if input_mint == SOL_TICKER:
        result["sol"] = in_amount / 10**9
        result["sol_lamports"] = in_amount
    elif output_mint == SOL_TICKER:
        result["sol"] = out_amount / 10**9
        result["sol_lamports"] = out_amount

    if input_mint != SOL_TICKER:
        result["non_sol"] = in_amount / 10**token_decimals
        result["non_sol_lamports"] = in_amount
    if output_mint != SOL_TICKER:
        result["non_sol"] = out_amount / 10**token_decimals
        result["non_sol_lamports"] = out_amount

    return result


def get_sol_amount(swap_quote: dict) -> float:
    """Return the SOL side of a swap quote (in SOL, not lamports)."""
    input_mint = swap_quote.get("inputMint")
    output_mint = swap_quote.get("outputMint")
    in_amount = int(swap_quote.get("inAmount", 0))
    out_amount = int(swap_quote.get("outAmount", 0))

    if input_mint == SOL_TICKER:
        return in_amount / 10**9
    elif output_mint == SOL_TICKER:
        return out_amount / 10**9
    return 0.0


def get_usd_price(symbol: str) -> float | None:
    """Fetch USD price from CoinGecko by CoinGecko coin ID (e.g. 'solana')."""
    params = {"ids": symbol, "vs_currencies": "usd"}
    response = requests.get("https://api.coingecko.com/api/v3/simple/price", params=params)
    if response.status_code == 200:
        return response.json()[symbol]["usd"]
    logger.error("CoinGecko price fetch failed: %s", response.text)
    return None


def get_native_sol_balance(wallet_id: str) -> float:
    """Return native SOL balance (in SOL) for the given wallet via JSON-RPC."""
    logger.info("Retrieving native SOL balance for wallet %s", wallet_id)
    payload = {"jsonrpc": "2.0", "id": 1, "method": "getBalance", "params": [wallet_id]}
    try:
        response = requests.post(SOLANA_RPC_URL, json=payload)
        lamports = response.json().get("result", {}).get("value", 0)
        sol = lamports / 1e9
        logger.info("SOL balance: %.6f SOL", sol)
        return sol
    except Exception as exc:
        logger.error("Error retrieving SOL balance: %s", exc)
        return 0.0


def get_wallet_balance(wallet_id: str, token_mint_address: str) -> float:
    """Return SPL token balance (UI amount) for the given wallet and mint."""
    logger.info("Retrieving balance for wallet %s, token %s", wallet_id, token_mint_address)
    payload = {
        "id": 1,
        "jsonrpc": "2.0",
        "method": "getTokenAccountsByOwner",
        "params": [wallet_id, {"mint": token_mint_address}, {"encoding": "jsonParsed"}],
    }
    try:
        response = requests.post(SOLANA_RPC_URL, json=payload)
        data = response.json()
        accounts = data["result"]["value"]
        if accounts:
            amount = accounts[0]["account"]["data"]["parsed"]["info"]["tokenAmount"]["uiAmount"]
            logger.info("Balance for %s: %s", token_mint_address, amount)
            return amount
        logger.warning("No token account found for wallet %s, mint %s", wallet_id, token_mint_address)
        return 0.0
    except Exception as exc:
        logger.error("Error retrieving token balance: %s", exc)
        return 0.0


def get_prioritization_fee_lamports(action: str = "buy") -> int:
    """
    Fetch recent prioritization fees from the RPC and return a clamped value
    in the range [500_000, 1_000_000] lamports.
    """
    payload = {"jsonrpc": "2.0", "method": "getRecentPrioritizationFees", "params": [], "id": 1}
    try:
        response = requests.post(SOLANA_RPC_URL, json=payload)
        logger.info("Prioritization fees response: %s", response.status_code)
        if response.status_code == 200:
            data = response.json()
            fees = [entry["prioritizationFee"] for entry in data.get("result", [])]
            logger.info("Recent fees: %s", fees)
            average = sum(fees) / len(fees) if fees else 0
            fee = max(500_000, min(1_000_000, int(average)))
            logger.info("prioritizationFeeLamports: %d", fee)
            return fee
        logger.error("Prioritization fee request failed: %s", response.status_code)
    except Exception as exc:
        logger.error("Error fetching prioritization fees: %s", exc)
    return 500_000


def get_token_decimals(token_mint: str) -> int | None:
    """Return the decimal precision for a Solana SPL token."""
    payload = {"jsonrpc": "2.0", "id": 1, "method": "getTokenSupply", "params": [token_mint]}
    try:
        response = requests.post(SOLANA_RPC_URL, json=payload)
        if response.status_code == 200:
            decimals = response.json()["result"]["value"]["decimals"]
            logger.info("Decimals for %s: %d", token_mint, decimals)
            return decimals
        logger.error("Failed to fetch decimals for %s (HTTP %s)", token_mint, response.status_code)
    except Exception as exc:
        logger.error("Error fetching token decimals for %s: %s", token_mint, exc)
    return None


def get_swap_quote_response(
    input_mint: str,
    in_amount: float,
    output_mint: str,
    slippage_percentage: float,
    decimals: int,
    max_retries: int = 3,
) -> dict | int:
    """Request a swap quote from Jupiter v6. Returns the quote dict or 0 on failure."""
    if in_amount <= 0:
        logger.error("Invalid in_amount: must be greater than 0")
        return 0

    lamports = int(in_amount * (10**decimals))
    slippage_bps = int(100 * slippage_percentage)
    url = (
        f"https://quote-api.jup.ag/v6/quote"
        f"?inputMint={input_mint}&outputMint={output_mint}"
        f"&amount={lamports}&slippageBps={slippage_bps}"
    )
    logger.info("Swap quote URL: %s", url)

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                quote = response.json()
                logger.info("Swap quote response: %s", quote)
                return quote
            logger.error("Quote fetch failed (HTTP %s), attempt %d/%d", response.status_code, attempt, max_retries)
        except Exception as exc:
            logger.error("Error fetching swap quote (attempt %d/%d): %s", attempt, max_retries, exc)

    logger.error("All %d attempts to fetch swap quote failed", max_retries)
    return 0

# ---------------------------------------------------------------------------
# Trade execution
# ---------------------------------------------------------------------------

async def perform_swap(
    action: str,
    slippage_percentage: float,
    attempt: int,
    percentage: float | None = None,
) -> dict:
    """Execute a market swap via Jupiter v6 swap API."""
    logger.info("Performing swap: action=%s", action)

    order_size = "full"

    if action == "buy":
        amount_to_swap = percentage
        order_size = f"{amount_to_swap:.2f} SOL"
        input_mint = SOL_TICKER
        output_mint = DESIRED_CRYPTO_TICKER
        logger.info("Buy amount: %s", order_size)

    elif action == "sell":
        crypto_balance = get_wallet_balance(WALLET_ID, DESIRED_CRYPTO_TICKER)
        if crypto_balance == 0:
            logger.error("%s balance is 0 — cannot sell", DESIRED_CRYPTO_TICKER)
            return {"error": f"Insufficient {DESIRED_CRYPTO_TICKER} balance"}
        if percentage is not None:
            amount_to_swap = crypto_balance * percentage
            if percentage < 1.00:
                order_size = "partial"
        else:
            amount_to_swap = crypto_balance
        logger.info("Sell amount: %s %s", amount_to_swap, DESIRED_CRYPTO_TICKER)
        input_mint = DESIRED_CRYPTO_TICKER
        output_mint = SOL_TICKER

    else:
        logger.error("Invalid action: %s", action)
        return {"error": "Invalid action specified"}

    action_label = action.ljust(6)
    file_logger.info(
        "Action: %s | mint: %s | Before Transaction | Attempt: %d | Order Size: %s",
        action_label, DESIRED_CRYPTO_TICKER, attempt, order_size,
    )

    decimals = get_token_decimals(input_mint)
    quote_response = get_swap_quote_response(input_mint, amount_to_swap, output_mint, slippage_percentage, decimals)

    if not quote_response:
        logger.error("Invalid quote response — aborting swap")
        return {"status": "error", "message": "Invalid quote response"}

    logger.info("Quote response received: %s", quote_response)

    payload = json.dumps({
        "quoteResponse": quote_response,
        "userPublicKey": WALLET_ID,
        "wrapAndUnwrapSol": True,
        "dynamicComputeUnitLimit": True,
        "dynamicSlippage": True,
        "prioritizationFeeLamports": 1_000_000,
    })
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {JUPITER_API_KEY}",
    }
    url = "https://quote-api.jup.ag/v6/swap"

    try:
        logger.info("Sending swap request to %s", url)
        response = requests.post(url, headers=headers, data=payload)
        logger.info("Swap HTTP response: %s", response)

        if response.status_code != 200:
            logger.error("Swap failed (HTTP %s): %s", response.status_code, response.text)
            return {"status": "error", "message": "Swap failed", "details": response.json()}

        swap_transaction = response.json().get("swapTransaction")
        logger.info("Swap transaction received: %s", swap_transaction)
        txn_response = await sign_and_send_transaction(swap_transaction)

        if txn_response.get("success"):
            logger.info("Transaction successful: %s", txn_response["transaction_id"])
            file_logger.info("Action: %s | mint: %s | After Transaction", action_label, DESIRED_CRYPTO_TICKER)

            sol_amount = get_sol_amount(quote_response)
            fee = 0.001
            txn_amount_sol = sol_amount + fee if action == "buy" else sol_amount - fee
            logger.info("sol_amount: %.6f", sol_amount)
            transaction_logger.info(
                "Action: %s | mint: %s | txn_amount_sol: %.5f | percent: %s",
                action_label, DESIRED_CRYPTO_TICKER, txn_amount_sol, percentage,
            )
            return {"status": "success", "transaction_id": txn_response["transaction_id"]}

        logger.error("Transaction failed: %s", txn_response.get("error"))
        return {
            "status": "error",
            "message": txn_response.get("error"),
            "details": txn_response.get("details"),
        }

    except Exception as exc:
        logger.error("Error during swap request: %s", exc)
        return {"status": "error", "message": "Swap request failed", "details": str(exc)}


async def create_limit_order(
    action: str,
    attempt: int,
    increase_percent: float,
    profit_percent: float,
) -> dict | None:
    """Place a Jupiter limit order for take-profit or stop-loss exits."""
    logger.info("Creating limit order: action=%s", action)

    EXPIRATION = int(time.time()) + 30 * 24 * 60 * 60  # 30 days

    order_size = "full"
    price_in_sol: float | None = None
    usd_non_sol_price: float | None = None
    crypto_balance: float | None = None

    if action == "buy":
        amount_to_swap = 0.05
        input_mint = SOL_TICKER
        output_mint = DESIRED_CRYPTO_TICKER

    elif action == "sell":
        crypto_balance = get_wallet_balance(WALLET_ID, DESIRED_CRYPTO_TICKER)
        if crypto_balance == 0:
            logger.error("%s balance is 0 — cannot sell", DESIRED_CRYPTO_TICKER)
            return {"error": f"Insufficient {DESIRED_CRYPTO_TICKER} balance"}
        amount_to_swap = crypto_balance * profit_percent / 100
        input_mint = DESIRED_CRYPTO_TICKER
        output_mint = SOL_TICKER

    else:
        logger.error("Invalid action: %s", action)
        return {"error": "Invalid action specified"}

    action_label = action.ljust(6)
    file_logger.info(
        "Action: %s | mint: %s | Before Transaction | Attempt: %d | Order Size: %s",
        action_label, DESIRED_CRYPTO_TICKER, attempt, order_size,
    )

    input_token_decimals = get_token_decimals(input_mint)
    output_token_decimals = 9  # SOL always has 9 decimals

    quote_response = get_swap_quote_response(input_mint, amount_to_swap, output_mint, 0, input_token_decimals)
    crypto_count = get_crypto_count(quote_response, input_token_decimals)
    usd_sol_price = get_usd_price("solana")

    if crypto_count.get("sol") and crypto_count.get("non_sol"):
        price_in_sol = crypto_count["sol"] / crypto_count["non_sol"]
        usd_non_sol_price = (usd_sol_price * crypto_count["sol"]) / crypto_count["non_sol"]
        logger.info("Token price: %.8f SOL / %.8f USD", price_in_sol, usd_non_sol_price)
    else:
        logger.warning("Missing SOL or non-SOL amount in quote — price unavailable")

    making_amount = crypto_count["non_sol_lamports"]
    taking_amount = int(crypto_count["sol_lamports"] * (1 + increase_percent / 100))

    logger.info("input_token_decimals: %s", input_token_decimals)
    logger.info("output_token_decimals: %s", output_token_decimals)
    logger.info("quote_response: %s", quote_response)
    logger.info("price_in_sol: %s", price_in_sol)
    logger.info("usd_non_sol_price: %s", usd_non_sol_price)
    logger.info("crypto_balance: %s", crypto_balance)
    logger.info("making_amount: %s", making_amount)
    logger.info("taking_amount: %s", taking_amount)

    payload = {
        "inputMint": str(input_mint),
        "outputMint": str(output_mint),
        "maker": str(WALLET_ID),
        "payer": str(WALLET_ID),
        "params": {
            "makingAmount": str(making_amount),
            "takingAmount": str(taking_amount),
            "expiredAt": str(EXPIRATION),
        },
        "computeUnitPrice": "auto",
    }
    logger.info("Limit order payload: %s", json.dumps(payload, indent=2))

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {JUPITER_API_KEY}",
    }
    url = "https://api.jup.ag/limit/v2/createOrder"

    try:
        response = requests.post(url, headers=headers, json=payload)
        logger.info("Limit order response (HTTP %s): %s", response.status_code, response.text)

        if response.status_code != 200:
            logger.error("Error creating limit order: %s — %s", response.status_code, response.text)
            return None

        result = response.json()
        logger.info("Limit order result: %s", json.dumps(result, indent=2))
        tx = result["tx"]
        txn_response = await sign_and_send_transaction(tx)

        if txn_response.get("success"):
            logger.info("Transaction successful: %s", txn_response["transaction_id"])
            file_logger.info("Action: %s | mint: %s | After Transaction", action_label, DESIRED_CRYPTO_TICKER)

            sol_amount = get_sol_amount(quote_response)
            fee = 0.0001
            txn_amount_sol = sol_amount + fee if action == "buy" else sol_amount - fee
            logger.info("sol_amount: %.6f", sol_amount)
            transaction_logger.info(
                "Action: %s | mint: %s | txn_amount_sol: %.5f | percent: %s",
                action_label, DESIRED_CRYPTO_TICKER, txn_amount_sol, profit_percent,
            )
            return {"status": "success", "transaction_id": txn_response["transaction_id"]}

        logger.error("Transaction failed: %s", txn_response.get("error"))
        return {
            "status": "error",
            "message": txn_response.get("error"),
            "details": txn_response.get("details"),
        }

    except requests.exceptions.RequestException as exc:
        logger.error("Request error creating limit order: %s", exc)
        return None


def create_associated_token_account(payer: Pubkey, owner: Pubkey, mint: Pubkey) -> Instruction:
    """Build an associated token account (ATA) creation instruction."""
    logger.info("Building ATA creation instruction")
    instruction = Instruction(
        program_id=TOKEN_PROGRAM_ID,
        accounts=[
            AccountMeta(pubkey=payer, is_signer=True, is_writable=True),
            AccountMeta(pubkey=ASSOCIATED_TOKEN_ADDRESS, is_signer=False, is_writable=True),
            AccountMeta(pubkey=owner, is_signer=False, is_writable=False),
            AccountMeta(pubkey=mint, is_signer=False, is_writable=False),
            AccountMeta(pubkey=SYS_PROGRAM_ID, is_signer=False, is_writable=False),
            AccountMeta(pubkey=TOKEN_PROGRAM_ID, is_signer=False, is_writable=False),
            AccountMeta(pubkey=SYSVAR_RENT_PUBKEY, is_signer=False, is_writable=False),
        ],
        data=bytes([]),
    )
    return instruction


async def sign_and_send_transaction(swap_transaction: str) -> dict:
    """Decode, sign, and broadcast a base64-encoded versioned transaction."""
    try:
        logger.info("Signing transaction: %s", swap_transaction)

        if not re.fullmatch(r"[A-Za-z0-9+/=]+", swap_transaction):
            raise ValueError("Invalid base64 string")

        try:
            raw_transaction = VersionedTransaction.from_bytes(base64.b64decode(swap_transaction))
        except Exception as exc:
            logger.error("Transaction decode error: %s", exc)
            return {"error": "Decoding error", "details": str(exc)}

        logger.info("Decoded transaction: %s", raw_transaction)

        signature = PRIVATE_KEY_PAIR.sign_message(message.to_bytes_versioned(raw_transaction.message))
        logger.info("Signature: %s", signature)

        signed_txn = VersionedTransaction.populate(raw_transaction.message, [signature])
        logger.info("Signed transaction: %s", signed_txn)

        opts = TxOpts(skip_preflight=False, preflight_commitment=Processed)

        transaction_id = None
        for attempt in range(1, 4):
            try:
                response = client.send_raw_transaction(txn=bytes(signed_txn), opts=opts)
                transaction_id = str(response.value) if hasattr(response, "value") else None
                if transaction_id:
                    logger.info("Transaction sent: https://explorer.solana.com/tx/%s", transaction_id)
                    break
                logger.warning("Attempt %d: no transaction ID in response", attempt)
            except Exception as exc:
                logger.error("Attempt %d error: %s", attempt, exc)

            if attempt < 3:
                wait = 2 ** (attempt - 1)
                logger.info("Retrying in %ds...", wait)
                time.sleep(wait)

        if not transaction_id:
            return {"error": "Transaction failed after multiple attempts"}

        connection = AsyncClient(SOLANA_RPC_URL)
        try:
            status = await get_transaction_status(connection, transaction_id)
        finally:
            await connection.close()

        if status.get("status") == "finalized":
            logger.info("Transaction %s finalized", transaction_id)
            return {"success": True, "transaction_id": transaction_id}

        if status.get("status") == "Error":
            logger.error("Transaction %s returned an error status", transaction_id)
            return {"error": "Transaction status contains error"}

        return {"error": "Transaction did not reach expected status", "details": status}

    except Exception as exc:
        logger.error("Error during transaction signing/sending: %s", exc)
        return {"error": "Transaction signing and sending failed", "details": str(exc)}


async def get_transaction_status(
    connection: AsyncClient,
    txid: str,
    polling_interval: int = 5,
    timeout: int = 300,
) -> dict:
    """Poll for transaction confirmation status until finalized or timeout."""
    elapsed = 0
    while elapsed < timeout:
        try:
            sig = Signature.from_string(txid)
            logger.info("Polling status for %s", txid)
            response = await connection.get_signature_statuses([sig])
            logger.info("Status response: %s", response)

            if response and response.value:
                status_info = response.value[0]
                if status_info is not None:
                    confirmation = str(status_info.confirmation_status)
                    error = str(status_info.err)
                    logger.info("Transaction %s — status: %s, err: %s", txid, confirmation, error)

                    if error != "None":
                        logger.error("Transaction %s has an error", txid)
                        return {"status": "Error", "txid": txid}

                    if confirmation == "TransactionConfirmationStatus.Finalized":
                        logger.info("Transaction %s finalized", txid)
                        return {"status": "finalized", "txid": txid}
                else:
                    logger.warning("Transaction %s not yet visible", txid)
            else:
                logger.warning("No status found for transaction %s", txid)

        except Exception as exc:
            logger.error("Error fetching status for %s: %s", txid, exc)

        await asyncio.sleep(polling_interval)
        elapsed += polling_interval

    logger.error("Timeout waiting for transaction %s to finalize", txid)
    return {"status": "timeout", "txid": txid}

# ---------------------------------------------------------------------------
# Webhook route & handlers
# ---------------------------------------------------------------------------

@app.route("/webhook", methods=["POST"])
async def webhook_listener():
    global buy_task, sell_task, DESIRED_CRYPTO_TICKER

    data = await request.get_json()
    action = data.get("action")
    mint = data.get("mint")
    percent = data.get("percent")
    action_label = action.ljust(6)

    logger.info("Webhook received — action: %s | mint: %s | percent: %s", action_label, mint, percent)

    async with action_lock:
        DESIRED_CRYPTO_TICKER = mint
        file_logger.info(
            "Action: %s | mint: %s | Received Alert | Percent: %s",
            action_label, mint, percent,
        )
        try:
            if action == "buy":
                buy_task = asyncio.create_task(handle_buy(action, percent))
                await buy_task
            elif action == "sell":
                sell_task = asyncio.create_task(handle_sell(action, percent))
                await sell_task
            else:
                logger.warning("Unknown action: %s", action)
                return jsonify({"status": "error", "message": "Invalid action"})

        except Exception as exc:
            msg = f"Exception during {action} for {mint}: {exc}"
            logger.error(msg)
            restart_server()
            return jsonify({"status": "error", "message": msg})
        else:
            result = {"status": "success", "message": f"{action} completed successfully"}

    logger.info("Action %s complete: %s", action, result)
    return jsonify(result)


async def handle_buy(action: str, percent: float) -> dict:
    """Execute a market buy then place limit sell orders for take-profit."""
    slippage = 3.0
    attempt = 1
    success = False

    while attempt <= RETRY_COUNT and not success:
        logger.info("Buy attempt %d — slippage: %.1f%%", attempt, slippage)

        await perform_swap(action, slippage, attempt, percent)

        # Place take-profit limit sell orders after entry
        result = await create_limit_order("sell", attempt, 200, 33)   # 2× TP, sell 33%
        result = await create_limit_order("sell", attempt, 500, 50)   # 5× TP, sell 50%

        if result and result.get("status") == "success":
            success = True
            logger.info("Buy + limit orders succeeded")
        else:
            logger.warning("Buy attempt %d failed — increasing slippage to %.1f%%", attempt, slippage)
            slippage += 0.5
            attempt += 1

    if not success:
        logger.error("Buy failed after %d attempt(s)", RETRY_COUNT)
        return {"status": "error", "message": "Buy failed after maximum retries"}
    return {"status": "success", "message": "Buy succeeded"}


async def handle_sell(action: str, percent: float) -> dict:
    """Execute a limit sell order."""
    slippage = 1.0
    attempt = 1
    success = False

    while attempt <= RETRY_COUNT and not success:
        logger.info("Sell attempt %d — slippage: %.1f%%", attempt, slippage)

        result = await create_limit_order(action, attempt, 10, 50)

        if result and "error" in result and "Insufficient" in result["error"]:
            logger.error("Sell skipped — insufficient balance: %s", result["error"])
            file_logger.info(
                "Action: %-6s | mint: %s | Transaction Skipped: Insufficient Balance",
                action, DESIRED_CRYPTO_TICKER,
            )
            break

        if result and result.get("status") == "success":
            success = True
            logger.info("Sell succeeded")
        else:
            logger.warning("Sell attempt %d failed — increasing slippage to %.1f%%", attempt, slippage)
            slippage += 0.5
            attempt += 1

    if not success:
        logger.error("Sell failed after %d attempt(s)", RETRY_COUNT)
        return {"status": "error", "message": "Sell failed after maximum retries"}
    return {"status": "success", "message": "Sell succeeded"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
