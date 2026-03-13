import os
import requests
import time

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
    url = f"{JUPITER_HOST_LIMIT}//swap/v1/quote?inputMint={input_mint}&outputMint={output_mint}&amount={amount}&slippageBps={slippage}"
    headers = {"x-api-key": JUPITER_API_KEY}
    
    return restClient.get(url, headers=headers)

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

def coin_gecko_price(ids="solana",vs_currencies="usd"):
    url = f"{COIN_GECKO_HOST}/api/v3/simple/price?ids={ids}&vs_currencies={vs_currencies}"

    return restClient.get(url)

