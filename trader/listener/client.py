"""
trader/listener/client.py — Telegram channel listener.

Connects to Telegram as a user account (Telethon) and converts new channel
posts into TokenSignal objects, which are pushed into an asyncio.Queue for
consumption by the trading engine.

Separation of concerns:
    - This module owns: Telegram connection, deduplication, text → signal routing
    - trader.listener.parser owns: all message text parsing logic
    - TradingEngine owns: price fetch, position management, strategy
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from telethon import TelegramClient, events
from telethon.sessions import StringSession
from telethon.tl.types import Message

from trader.config import Config
from trader.listener.parser import parse_message
from trader.trading.models import TokenSignal

logger = logging.getLogger(__name__)
signal_log = logging.getLogger("signals")


class TelegramListener:
    """
    Listens to a configured Telegram channel as a user account and emits
    TokenSignal objects for valid first-call Solana entries.

    The signal_queue decouples message receipt from trade execution so the
    Telegram event loop is never blocked by price fetches or order logic.
    """

    def __init__(
        self,
        cfg: Config,
        signal_queue: asyncio.Queue[TokenSignal],
        db=None,           # TradeDatabase | None — optional to avoid circular import
    ) -> None:
        self._cfg = cfg
        self._queue = signal_queue
        self._db = db
        self._seen_ids: set[int] = set()
        self._client: Optional[TelegramClient] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """
        Authenticate, verify channel access, and begin listening.

        Raises on authentication failure or if the channel cannot be resolved.
        """
        session = (
            StringSession(self._cfg.tg_session_string)
            if self._cfg.tg_session_string
            else self._cfg.session_file
        )
        self._client = TelegramClient(
            session,
            self._cfg.tg_api_id,
            self._cfg.tg_api_hash,
        )

        self._client.add_event_handler(
            self._on_new_message,
            events.NewMessage(chats=self._cfg.channel_username),
        )

        await self._client.start(phone=self._cfg.tg_phone or None)
        me = await self._client.get_me()
        logger.info("[TG] Authenticated as: %s", me.username or me.first_name)

        if self._db:
            self._seen_ids = self._db.load_seen_msg_ids()

        try:
            entity = await self._client.get_entity(self._cfg.channel_username)
            logger.info("[TG] Monitoring channel id=%s", entity.id)
        except Exception:
            logger.exception(
                "[TG] Could not resolve channel '%s'", self._cfg.channel_username
            )
            raise

        await self._catch_up(entity)

    async def _catch_up(self, entity, limit: int = 50) -> None:
        """
        Fetch the most recent `limit` messages and process any that arrived
        after the last message we saw in a previous session.

        High-water mark = max(seen_msg_ids).  Only messages with a higher ID
        are traded; everything else is just marked seen.

        Fresh database (no seen IDs): skips catch-up entirely and starts
        listening for new signals from this point forward.
        """
        if not self._seen_ids:
            logger.info("[TG] Fresh database — skipping catch-up, listening for new signals.")
            return

        last_seen_id: int = max(self._seen_ids)
        logger.info(
            "[TG] Catching up on last %d messages (last seen id=%d)...",
            limit, last_seen_id,
        )
        caught = skipped_old = 0
        async for msg in self._client.iter_messages(entity, limit=limit):
            self._seen_ids.add(msg.id)
            if self._db:
                self._db.add_seen_msg_id(msg.id)

            if msg.id <= last_seen_id:
                skipped_old += 1
                continue

            if not (msg.text or "").strip():
                continue

            logger.info("[CATCHUP] Processing missed message id=%d", msg.id)
            class _FakeEvent:
                message = msg
            await self._on_new_message(_FakeEvent())
            caught += 1

        logger.info(
            "[TG] Catch-up complete — %d processed, %d skipped (already seen)",
            caught, skipped_old,
        )

    async def run_until_disconnected(self) -> None:
        """Block until the Telegram connection drops."""
        if self._client:
            await self._client.run_until_disconnected()

    async def disconnect(self) -> None:
        if self._client:
            await self._client.disconnect()

    # ------------------------------------------------------------------
    # Event handler
    # ------------------------------------------------------------------

    async def _on_new_message(self, event: events.NewMessage.Event) -> None:
        """
        Handle a new channel post.

        Steps:
            1. Deduplicate by message ID.
            2. Parse text through the parser pipeline.
            3. Reject non-first-call or non-Solana messages.
            4. Reject messages with no extractable mint address.
            5. Push a TokenSignal onto the queue.
        """
        msg: Message = event.message

        if msg.id in self._seen_ids:
            return
        self._seen_ids.add(msg.id)
        if self._db:
            self._db.add_seen_msg_id(msg.id)

        raw_text: str = msg.text or ""
        if not raw_text.strip():
            return

        try:
            result = parse_message(raw_text)
        except Exception:
            logger.exception("[TG] Parse error on message id=%d", msg.id)
            return

        if not result["is_first_call"]:
            logger.info("[TG] id=%d — update post, skipping", msg.id)
            signal_log.info("UPDATE     | %-10s | %-44s | msg_id=%d", "-", "-", msg.id)
            if self._db:
                self._db.log_signal("UPDATE", msg_id=msg.id)
            return

        if not result["is_solana"]:
            logger.info("[TG] id=%d — not a Solana call, skipping", msg.id)
            signal_log.info("NOT_SOLANA | %-10s | %-44s | msg_id=%d", "-", "-", msg.id)
            if self._db:
                self._db.log_signal("NOT_SOLANA", msg_id=msg.id)
            return

        mint: Optional[str] = result["mint_address"]
        symbol: str = result["symbol_hint"] or "UNKNOWN"

        if not mint:
            logger.info("[TG] id=%d — no mint address found, skipping", msg.id)
            signal_log.info("NO_MINT    | %-10s | %-44s | msg_id=%d", symbol, "-", msg.id)
            if self._db:
                self._db.log_signal("NO_MINT", msg_id=msg.id, symbol=symbol)
            return

        signal = TokenSignal(
            symbol=symbol,
            mint_address=mint,
            detected_at=datetime.now(timezone.utc),
        )

        logger.info("[QUEUE] %s | mint=%s | queue_depth=%d", symbol, mint, self._queue.qsize() + 1)
        signal_log.info("QUEUED     | %-10s | %-44s | msg_id=%d", symbol, mint, msg.id)
        if self._db:
            self._db.log_signal("QUEUED", msg_id=msg.id, symbol=symbol, mint=mint)

        await self._queue.put(signal)
