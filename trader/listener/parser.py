"""
trader/listener/parser.py — Telegram message parsing pipeline.

Stateless, pure functions only. No I/O, no logging side-effects during normal
operation. Every public function is independently unit-testable.

Pipeline:
    raw text
        → extract_urls()          strip trailing punctuation from Markdown links
        → is_update_message()     reject "hit 5X / hovering at 2X" update posts
        → is_solana_call()        accept/reject by chain heuristics
        → _extract_gmgn_mint()    priority-1 mint extraction
        → _extract_dexscreener_pair()   priority-2 pair address
        → extract_symbol_hint()   $TOKEN → "TOKEN"
        → parse_message()         assembles and returns the result dict
"""

from __future__ import annotations

import logging
import re
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Solana Base58 alphabet — excludes 0, O, I, l
_BASE58_RE = re.compile(
    r"^[123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz]+$"
)
_MINT_MIN_LEN = 32
_MINT_MAX_LEN = 50

# Non-Solana chain keywords — any of these disqualifies a message
_REJECT_CHAINS = frozenset({
    "bsc", "eth", "ethereum", "base", "tron", "avax", "arbitrum"
})

# ---------------------------------------------------------------------------
# URL extraction
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
_URL_TRAILING_JUNK = re.compile(r"[)\].,!?]+$")


def extract_urls(text: str) -> list[str]:
    """
    Return all http/https URLs in *text*, in order of appearance.

    Trailing punctuation (e.g. ')' from Markdown [label](url) syntax,
    trailing '.' or ',') is stripped from each match to avoid capturing
    non-URL characters that follow immediately.
    """
    return [_URL_TRAILING_JUNK.sub("", u) for u in _URL_RE.findall(text)]


# ---------------------------------------------------------------------------
# Update message detection
# ---------------------------------------------------------------------------

_UPDATE_PATTERNS = re.compile(
    r"""
    \b\d+\.?\d*[xX]\b          # standalone multiplier:  2X · 5X · 2.5x
    | \bhit\s+\d+[xX]\b        # "hit 5X"
    | \bmoon\s*mode\b           # "moon mode"
    | \bhover(?:ing)?\s+around\b  # "hovering around"
    | \bup\s+\d+%               # "up 300%"
    | \b\d+x\s+from\b           # "10x from entry"
    """,
    re.IGNORECASE | re.VERBOSE,
)


def is_update_message(text: str) -> bool:
    """
    Return True if the message is a performance update on an existing call
    rather than a new first-call entry signal.

    Update messages reference multipliers ("2X", "hit 5X"), "moon mode", etc.
    These should be ignored by the trading bot.
    """
    return bool(_UPDATE_PATTERNS.search(text))


# ---------------------------------------------------------------------------
# Solana chain detection
# ---------------------------------------------------------------------------

def is_solana_call(text: str, urls: list[str]) -> bool:
    """
    Return True when the message is identifiable as a Solana token call.

    Accept if ANY of:
        - gmgn.ai/sol/token/... URL is present
        - dexscreener.com/solana/... URL is present
        - the word "solana" appears in the text

    Reject if ANY non-Solana chain keyword appears as a whole word.
    """
    lower = text.lower()

    for chain in _REJECT_CHAINS:
        if re.search(rf"\b{re.escape(chain)}\b", lower):
            return False

    if re.search(r"\bsolana\b", lower):
        return True

    for url in urls:
        url_lower = url.lower()
        if "gmgn.ai/sol/token/" in url_lower:
            return True
        parsed = urlparse(url_lower)
        if parsed.netloc in ("dexscreener.com", "www.dexscreener.com"):
            if parsed.path.startswith("/solana/"):
                return True

    return False


# ---------------------------------------------------------------------------
# Mint / pair address extraction
# ---------------------------------------------------------------------------

def _extract_gmgn_mint(urls: list[str]) -> Optional[str]:
    """
    Extract the token mint address from a GMGN Solana URL.

    URL form:  https://gmgn.ai/sol/token/<prefix>_<mint>
    If no underscore prefix: https://gmgn.ai/sol/token/<mint>

    Rule: take the final path segment; if it contains '_', the mint is
    everything after the LAST underscore.
    """
    for url in urls:
        parsed = urlparse(url)
        if parsed.netloc.lower() not in ("gmgn.ai", "www.gmgn.ai"):
            continue
        if "/sol/token/" not in parsed.path.lower():
            continue

        segment = parsed.path.rstrip("/").split("/")[-1]
        if not segment:
            continue

        mint = segment.rsplit("_", 1)[-1] if "_" in segment else segment
        if mint:
            return mint

    return None


def _extract_dexscreener_pair(urls: list[str]) -> Optional[str]:
    """
    Extract the liquidity pool address from a DexScreener Solana URL.

    URL form:  https://dexscreener.com/solana/<pair_address>

    NOTE: This is a pool/pair address, NOT the token mint. A token can have
    multiple pools. Use the mint address (from GMGN) for token identification.
    """
    for url in urls:
        parsed = urlparse(url)
        if parsed.netloc.lower() not in ("dexscreener.com", "www.dexscreener.com"):
            continue
        if not parsed.path.lower().startswith("/solana/"):
            continue

        pair = parsed.path.rstrip("/").split("/")[-1]
        if pair and pair.lower() != "solana":
            return pair

    return None


# ---------------------------------------------------------------------------
# Mint validation
# ---------------------------------------------------------------------------

def is_valid_solana_mint(mint: str) -> bool:
    """
    Validate a Solana mint address using local heuristics (no RPC call).

    Rules:
        - Non-empty
        - Length 32–50 characters
        - Only Base58 characters (no 0, O, I, l)
        - Pump.fun-style mints ending in 'pump' are valid if they satisfy
          the charset and length constraints
    """
    if not mint:
        return False
    if not (_MINT_MIN_LEN <= len(mint) <= _MINT_MAX_LEN):
        return False
    return bool(_BASE58_RE.match(mint))


# ---------------------------------------------------------------------------
# Symbol hint extraction
# ---------------------------------------------------------------------------

_SYMBOL_RE = re.compile(r"\$([A-Za-z][A-Za-z0-9]{0,19})")


def extract_symbol_hint(text: str) -> Optional[str]:
    """
    Return the first $TICKER found in *text*, normalised to uppercase.

    Examples: '$Bread' → 'BREAD',  '$BONK' → 'BONK'
    """
    match = _SYMBOL_RE.search(text)
    return match.group(1).upper() if match else None


# ---------------------------------------------------------------------------
# Top-level parse function
# ---------------------------------------------------------------------------

def parse_message(text: str) -> dict:
    """
    Parse a raw Telegram message and return a structured signal dict.

    This is the only public entry point needed by the listener pipeline.
    It is pure (no I/O) and fully unit-testable.

    Returns:
        {
            "is_first_call": bool,
            "is_solana":     bool,
            "mint_address":  str | None,   # GMGN mint (highest priority)
            "pair_address":  str | None,   # DexScreener pool (informational)
            "symbol_hint":   str | None,   # e.g. "BREAD"
            "urls":          list[str],
            "raw_text":      str,
        }
    """
    urls = extract_urls(text)
    update = is_update_message(text)
    solana = is_solana_call(text, urls)

    mint_address: Optional[str] = None
    pair_address: Optional[str] = None

    if solana and not update:
        candidate = _extract_gmgn_mint(urls)
        if candidate and is_valid_solana_mint(candidate):
            mint_address = candidate
        elif candidate:
            logger.warning("GMGN candidate '%s' failed mint validation", candidate)

        pair_address = _extract_dexscreener_pair(urls)

    return {
        "is_first_call": not update,
        "is_solana": solana,
        "mint_address": mint_address,
        "pair_address": pair_address,
        "symbol_hint": extract_symbol_hint(text),
        "urls": urls,
        "raw_text": text,
    }
