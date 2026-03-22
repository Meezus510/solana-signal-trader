"""
tests/test_birdeye.py — Unit tests for BirdeyePriceClient.get_token_security()
and the new 30m wallet fields in get_token_overview().

No network calls — aiohttp session is mocked throughout.
Run with: pytest tests/test_birdeye.py -v
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from trader.pricing.birdeye import BirdeyePriceClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(base_url: str = "https://api.birdeye.test") -> MagicMock:
    cfg = MagicMock()
    cfg.birdeye_api_key = "test-key"
    cfg.birdeye_base_url = base_url
    cfg.request_timeout_seconds = 10
    return cfg


def _mock_response(status: int = 200, body: dict | None = None) -> MagicMock:
    """Return a mock aiohttp response context manager."""
    resp = AsyncMock()
    resp.status = status
    resp.json = AsyncMock(return_value=body or {})
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=resp)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _make_client(response: MagicMock) -> BirdeyePriceClient:
    session = MagicMock()
    session.get = MagicMock(return_value=response)
    return BirdeyePriceClient(_make_cfg(), session)


_FULL_SECURITY_BODY = {
    "data": {
        "top10HolderPercent":   0.45,
        "freezeable":           None,
        "transferFeeEnable":    None,
        "ownerPercentage":      None,
        "creatorPercentage":    0.02,
        "mutableMetadata":      False,
        "jupStrictList":        True,
        "isToken2022":          False,
        "nonTransferable":      None,
        "preMarketHolder":      [],
        "creationTime":         int(time.time()) - 3600,  # 1 hour ago
    }
}

_FULL_OVERVIEW_BODY = {
    "data": {
        "marketCap":               1_000_000,
        "liquidity":               50_000,
        "holder":                  500,
        "uniqueWallet5m":          20,
        "uniqueWalletHistory5m":   10,
        "uniqueWallet30m":         80,
        "uniqueWalletHistory30m":  60,
        "priceChange30mPercent":   5.0,
        "vBuy5mUSD":               1000.0,
        "vSell5mUSD":              400.0,
    }
}


# ---------------------------------------------------------------------------
# get_token_security — happy path
# ---------------------------------------------------------------------------

class TestGetTokenSecurity:
    @pytest.mark.asyncio
    async def test_returns_dict_on_200(self):
        client = _make_client(_mock_response(200, _FULL_SECURITY_BODY))
        result = await client.get_token_security("MINT")
        assert result is not None
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_top10_concentration_parsed(self):
        client = _make_client(_mock_response(200, _FULL_SECURITY_BODY))
        result = await client.get_token_security("MINT")
        assert result["top10_concentration"] == pytest.approx(0.45)

    @pytest.mark.asyncio
    async def test_top10_concentration_clamped_above_one(self):
        body = {"data": {**_FULL_SECURITY_BODY["data"], "top10HolderPercent": 1.5}}
        client = _make_client(_mock_response(200, body))
        result = await client.get_token_security("MINT")
        assert result["top10_concentration"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_top10_concentration_clamped_below_zero(self):
        body = {"data": {**_FULL_SECURITY_BODY["data"], "top10HolderPercent": -0.1}}
        client = _make_client(_mock_response(200, body))
        result = await client.get_token_security("MINT")
        assert result["top10_concentration"] == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_null_fields_returned_as_none(self):
        client = _make_client(_mock_response(200, _FULL_SECURITY_BODY))
        result = await client.get_token_security("MINT")
        assert result["security_freezeable"] is None
        assert result["security_transfer_fee"] is None
        assert result["security_owner_pct"] is None
        assert result["security_non_transferable"] is None

    @pytest.mark.asyncio
    async def test_bool_fields_parsed(self):
        client = _make_client(_mock_response(200, _FULL_SECURITY_BODY))
        result = await client.get_token_security("MINT")
        assert result["security_mutable_metadata"] is False
        assert result["security_jup_strict"] is True
        assert result["security_is_token2022"] is False

    @pytest.mark.asyncio
    async def test_creator_pct_parsed(self):
        client = _make_client(_mock_response(200, _FULL_SECURITY_BODY))
        result = await client.get_token_security("MINT")
        assert result["security_creator_pct"] == pytest.approx(0.02)

    @pytest.mark.asyncio
    async def test_pre_market_holders_empty_list(self):
        client = _make_client(_mock_response(200, _FULL_SECURITY_BODY))
        result = await client.get_token_security("MINT")
        assert result["security_pre_market_holders"] == 0

    @pytest.mark.asyncio
    async def test_pre_market_holders_counted(self):
        body = {"data": {**_FULL_SECURITY_BODY["data"], "preMarketHolder": ["w1", "w2", "w3"]}}
        client = _make_client(_mock_response(200, body))
        result = await client.get_token_security("MINT")
        assert result["security_pre_market_holders"] == 3

    @pytest.mark.asyncio
    async def test_token_age_hours_positive(self):
        client = _make_client(_mock_response(200, _FULL_SECURITY_BODY))
        result = await client.get_token_security("MINT")
        # created 1 hour ago → age ~1h
        assert result["security_token_age_hours"] == pytest.approx(1.0, abs=0.1)

    @pytest.mark.asyncio
    async def test_token_age_none_when_no_creation_time(self):
        body = {"data": {**_FULL_SECURITY_BODY["data"], "creationTime": None}}
        client = _make_client(_mock_response(200, body))
        result = await client.get_token_security("MINT")
        assert result["security_token_age_hours"] is None

    @pytest.mark.asyncio
    async def test_freezeable_true_when_set(self):
        body = {"data": {**_FULL_SECURITY_BODY["data"], "freezeable": True}}
        client = _make_client(_mock_response(200, body))
        result = await client.get_token_security("MINT")
        assert result["security_freezeable"] is True

    @pytest.mark.asyncio
    async def test_transfer_fee_true_when_set(self):
        body = {"data": {**_FULL_SECURITY_BODY["data"], "transferFeeEnable": True}}
        client = _make_client(_mock_response(200, body))
        result = await client.get_token_security("MINT")
        assert result["security_transfer_fee"] is True

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_returns_none_on_401(self):
        client = _make_client(_mock_response(401))
        assert await client.get_token_security("MINT") is None

    @pytest.mark.asyncio
    async def test_returns_none_on_429(self):
        client = _make_client(_mock_response(429))
        assert await client.get_token_security("MINT") is None

    @pytest.mark.asyncio
    async def test_returns_none_on_404(self):
        client = _make_client(_mock_response(404))
        assert await client.get_token_security("MINT") is None

    @pytest.mark.asyncio
    async def test_returns_none_on_empty_data(self):
        client = _make_client(_mock_response(200, {"data": {}}))
        # top10HolderPercent is None → top10_concentration is None, but dict still returned
        result = await client.get_token_security("MINT")
        assert result is not None
        assert result["top10_concentration"] is None

    @pytest.mark.asyncio
    async def test_returns_none_on_exception(self):
        session = MagicMock()
        session.get = MagicMock(side_effect=Exception("network failure"))
        client = BirdeyePriceClient(_make_cfg(), session)
        assert await client.get_token_security("MINT") is None


# ---------------------------------------------------------------------------
# get_token_overview — 30m wallet fields
# ---------------------------------------------------------------------------

class TestGetTokenOverview30mFields:
    @pytest.mark.asyncio
    async def test_unique_wallet_30m_parsed(self):
        client = _make_client(_mock_response(200, _FULL_OVERVIEW_BODY))
        result = await client.get_token_overview("MINT")
        assert result is not None
        assert result["unique_wallet_30m"] == 80

    @pytest.mark.asyncio
    async def test_unique_wallet_hist_30m_parsed(self):
        client = _make_client(_mock_response(200, _FULL_OVERVIEW_BODY))
        result = await client.get_token_overview("MINT")
        assert result["unique_wallet_hist_30m"] == 60

    @pytest.mark.asyncio
    async def test_30m_wallet_fields_none_when_absent(self):
        body = {"data": {**_FULL_OVERVIEW_BODY["data"]}}
        del body["data"]["uniqueWallet30m"]
        del body["data"]["uniqueWalletHistory30m"]
        client = _make_client(_mock_response(200, body))
        result = await client.get_token_overview("MINT")
        assert result["unique_wallet_30m"] is None
        assert result["unique_wallet_hist_30m"] is None

    @pytest.mark.asyncio
    async def test_existing_5m_fields_still_present(self):
        client = _make_client(_mock_response(200, _FULL_OVERVIEW_BODY))
        result = await client.get_token_overview("MINT")
        assert result["unique_wallet_5m"] == 20
        assert result["unique_wallet_hist_5m"] == 10
        assert result["buy_volume_usd_5m"] == pytest.approx(1000.0)
        assert result["sell_volume_usd_5m"] == pytest.approx(400.0)
