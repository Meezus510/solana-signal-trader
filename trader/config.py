"""
trader/config.py — Centralised, immutable configuration.

All tuneable parameters live here so a single file change (or .env edit)
is all that is needed to adjust the bot's behaviour. Strategy constants,
API credentials, and operational settings are co-located for transparency.

Usage:
    from trader.config import Config
    cfg = Config.load()          # reads env vars, raises early on missing values
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """
    Immutable configuration object.

    frozen=True prevents accidental mutation during a live session and makes
    the object safely hashable and usable as a dict key.

    Instantiate via Config.load() — never construct directly in production.
    """

    # ------------------------------------------------------------------
    # Birdeye market data
    # ------------------------------------------------------------------
    birdeye_api_key: str
    birdeye_base_url: str = "https://public-api.birdeye.so"

    # ------------------------------------------------------------------
    # Portfolio
    # ------------------------------------------------------------------
    starting_cash_usd: float = 1_000.0
    buy_size_usd: float = 10.0

    # ------------------------------------------------------------------
    # Strategy
    # ------------------------------------------------------------------
    stop_loss_pct: float = 0.35              # hard stop at -35% below entry
    take_profit_multiple: float = 2.5        # first TP fires at 2.5× entry
    take_profit_sell_fraction: float = 0.50  # sell 50% of position at TP
    trailing_stop_pct: float = 0.35          # 35% trailing stop on remainder

    # ------------------------------------------------------------------
    # Operational
    # ------------------------------------------------------------------
    poll_interval_seconds: float = 1.0
    request_timeout_seconds: int = 10
    max_concurrent_price_requests: int = 10
    demo_cycles: int = 60

    # ------------------------------------------------------------------
    # Telegram
    # ------------------------------------------------------------------
    tg_api_id: int = 0
    tg_api_hash: str = ""
    channel_username: str = ""
    session_file: str = "listener/tg_session"  # path to .session file

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def load(cls) -> "Config":
        """
        Build a Config from environment variables.

        Fails fast with a descriptive ValueError if any required variable is
        absent, so misconfiguration is caught at startup rather than mid-session.
        """
        birdeye_api_key = os.getenv("BIRDEYE_API_KEY", "").strip()
        if not birdeye_api_key:
            raise ValueError(
                "BIRDEYE_API_KEY is required. Set it in your .env file."
            )

        tg_api_id_raw = os.getenv("TG_API_ID", "").strip()
        if not tg_api_id_raw:
            raise ValueError("TG_API_ID is required. Set it in your .env file.")

        tg_api_hash = os.getenv("TG_API_HASH", "").strip()
        if not tg_api_hash:
            raise ValueError("TG_API_HASH is required. Set it in your .env file.")

        try:
            tg_api_id = int(tg_api_id_raw)
        except ValueError:
            raise ValueError(
                f"TG_API_ID must be an integer, got: {tg_api_id_raw!r}"
            )

        channel = os.getenv("TG_CHANNEL", "").strip()
        if not channel:
            raise ValueError("TG_CHANNEL is required. Set it in your .env file.")

        return cls(
            birdeye_api_key=birdeye_api_key,
            tg_api_id=tg_api_id,
            tg_api_hash=tg_api_hash,
            channel_username=channel,
        )
