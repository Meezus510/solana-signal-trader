"""
tests/test_parser.py — Unit tests for the message parsing pipeline.

All tests are pure: no network, no filesystem, no Telegram connection.
Run with:  pytest tests/test_parser.py -v
"""

import pytest

from trader.listener.parser import (
    extract_symbol_hint,
    extract_urls,
    is_solana_call,
    is_update_message,
    is_valid_solana_mint,
    parse_message,
)

# ---------------------------------------------------------------------------
# Fixtures — real message shapes from the channel
# ---------------------------------------------------------------------------

FIRST_CALL_FULL = (
    "some action on $Bread, worth monitoring, price action in equilibrium, "
    "and blowing past ATHs right now volume's warming up nicely. mcap's sitting around 100K,\n\n"
    "https://x.com/i/communities/2032304492880834932\n\n"
    "https://dexscreener.com/solana/6nikn2kqczlvfrmjlyfwzzs4frawfpevrqwgefncycma\n\n"
    "Trade it fast with: [**Banana**](https://t.me/BananaGunSniper_bot?start=snp_x) "
    "| [**GmGn**](https://gmgn.ai/sol/token/ref_9QMwJiVFrnCzuqCNvNy2fFUHUmNX5hvAavENz6z9pump)"
)

UPDATE_2X = "$DOOMSDAY hovering around 2X, ~193K mcap, volume heating up, sending hard"
UPDATE_5X = "$ROFL hit 5X, moon mode, mcap at 502K, buyers stacking volume"
BSC_MSG   = "$TOKEN launching on BSC!\nhttps://dexscreener.com/bsc/0xABCDEF1234"

KNOWN_MINT = "9QMwJiVFrnCzuqCNvNy2fFUHUmNX5hvAavENz6z9pump"
KNOWN_PAIR = "6nikn2kqczlvfrmjlyfwzzs4frawfpevrqwgefncycma"


# ---------------------------------------------------------------------------
# extract_urls
# ---------------------------------------------------------------------------

class TestExtractUrls:
    def test_plain_urls(self):
        text = "see https://dexscreener.com/solana/abc123 and https://x.com/foo"
        urls = extract_urls(text)
        assert len(urls) == 2
        assert all(u.startswith("http") for u in urls)

    def test_strips_trailing_paren(self):
        text = "[label](https://gmgn.ai/sol/token/ref_9QMwJiVFrnCzuqCNvNy2fFUHUmNX5hvAavENz6z9pump)"
        urls = extract_urls(text)
        assert not any(u.endswith(")") for u in urls)

    def test_strips_trailing_period(self):
        text = "check https://example.com."
        urls = extract_urls(text)
        assert urls == ["https://example.com"]

    def test_no_urls(self):
        assert extract_urls("no links here") == []


# ---------------------------------------------------------------------------
# is_update_message
# ---------------------------------------------------------------------------

class TestIsUpdateMessage:
    @pytest.mark.parametrize("text", [
        UPDATE_2X,
        UPDATE_5X,
        "$WIF 10x from launch, insane",
        "$BONK up 300% from entry",
    ])
    def test_detects_updates(self, text: str):
        assert is_update_message(text) is True

    @pytest.mark.parametrize("text", [
        FIRST_CALL_FULL,
        "$NEWTOKEN just launched, mcap around 80K",
        "chart signals triggered light entries",
    ])
    def test_does_not_flag_first_calls(self, text: str):
        assert is_update_message(text) is False


# ---------------------------------------------------------------------------
# is_solana_call
# ---------------------------------------------------------------------------

class TestIsSolanaCall:
    def test_accepts_gmgn_url(self):
        urls = ["https://gmgn.ai/sol/token/ref_abc123"]
        assert is_solana_call("some text", urls) is True

    def test_accepts_dexscreener_solana_url(self):
        urls = ["https://dexscreener.com/solana/abc123"]
        assert is_solana_call("some text", urls) is True

    def test_accepts_solana_keyword(self):
        assert is_solana_call("gem launching on Solana today", []) is True

    def test_rejects_bsc(self):
        assert is_solana_call("launching on BSC", []) is False

    def test_rejects_ethereum(self):
        assert is_solana_call("ERC-20 token on Ethereum", []) is False

    def test_rejects_base(self):
        assert is_solana_call("trading on Base chain", []) is False

    def test_rejects_when_no_signals(self):
        assert is_solana_call("just some random text", []) is False


# ---------------------------------------------------------------------------
# is_valid_solana_mint
# ---------------------------------------------------------------------------

class TestIsValidSolanaMint:
    @pytest.mark.parametrize("mint", [
        KNOWN_MINT,                                          # pump.fun style
        "So11111111111111111111111111111111111111112",        # wrapped SOL
        "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN",     # JUP
    ])
    def test_valid_mints(self, mint: str):
        assert is_valid_solana_mint(mint) is True

    @pytest.mark.parametrize("mint,reason", [
        ("", "empty string"),
        ("short", "too short"),
        ("0xABCDEF1234" * 3, "contains 0 and x"),
        ("OOOOIIII" * 5, "contains O and I"),
        ("a" * 60, "too long"),
    ])
    def test_invalid_mints(self, mint: str, reason: str):
        assert is_valid_solana_mint(mint) is False, f"Expected invalid: {reason}"


# ---------------------------------------------------------------------------
# extract_symbol_hint
# ---------------------------------------------------------------------------

class TestExtractSymbolHint:
    def test_normalises_to_uppercase(self):
        assert extract_symbol_hint("some action on $Bread today") == "BREAD"

    def test_all_caps(self):
        assert extract_symbol_hint("$BONK is pumping") == "BONK"

    def test_first_occurrence(self):
        assert extract_symbol_hint("$WIF or $BONK?") == "WIF"

    def test_no_symbol(self):
        assert extract_symbol_hint("no ticker here") is None


# ---------------------------------------------------------------------------
# parse_message (integration)
# ---------------------------------------------------------------------------

class TestParseMessage:
    def test_full_first_call(self):
        result = parse_message(FIRST_CALL_FULL)
        assert result["is_first_call"] is True
        assert result["is_solana"] is True
        assert result["mint_address"] == KNOWN_MINT
        assert result["pair_address"] == KNOWN_PAIR
        assert result["symbol_hint"] == "BREAD"

    def test_update_message_ignored(self):
        result = parse_message(UPDATE_2X)
        assert result["is_first_call"] is False
        assert result["mint_address"] is None   # no extraction on update posts

    def test_bsc_rejected(self):
        result = parse_message(BSC_MSG)
        assert result["is_solana"] is False
        assert result["mint_address"] is None

    def test_dexscreener_only_no_mint(self):
        text = (
            "$BONK launching on Solana\n"
            "https://dexscreener.com/solana/AbCdEfGhIjKmNpQrStUvWxYz1234567890abcdefghij"
        )
        result = parse_message(text)
        assert result["is_solana"] is True
        assert result["mint_address"] is None     # no GMGN URL → no mint
        assert result["pair_address"] is not None # DexScreener pair is set

    def test_gmgn_without_prefix(self):
        text = (
            "Solana gem\n"
            f"https://gmgn.ai/sol/token/{KNOWN_MINT}"
        )
        result = parse_message(text)
        assert result["mint_address"] == KNOWN_MINT

    def test_no_trailing_paren_in_urls(self):
        text = (
            "$DOOMSDAY hovering around 2X\n"
            "[GmGn](https://gmgn.ai/sol/token/ref_9QMwJiVFrnCzuqCNvNy2fFUHUmNX5hvAavENz6z9pump)"
        )
        result = parse_message(text)
        for url in result["urls"]:
            assert not url.endswith(")"), f"URL has trailing ')': {url}"
