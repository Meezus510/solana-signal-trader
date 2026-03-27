"""
dump_channel.py — Print raw message text + entity URLs for a channel.

Usage:
    python scripts/dump_channel.py <channel> [--limit 15]

Useful for inspecting message format before writing a parser.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.sessions import StringSession

load_dotenv()


def _require(key: str) -> str:
    val = os.getenv(key, "").strip()
    if not val:
        print(f"[ERROR] {key} is required in your .env file.", file=sys.stderr)
        sys.exit(1)
    return val


async def dump(channel: str, limit: int) -> None:
    session_str = os.getenv("TG_SESSION_STRING", "").strip()
    session = StringSession(session_str) if session_str else "scripts/analyze_session"
    client = TelegramClient(session, int(_require("TG_API_ID")), _require("TG_API_HASH"))

    await client.start(phone=os.getenv("TG_PHONE") or None)
    me = await client.get_me()
    print(f"[TG] Authenticated as: {me.username or me.first_name}")

    entity = await client.get_entity(channel)
    msgs = []
    async for msg in client.iter_messages(entity, limit=limit):
        if (msg.text or "").strip():
            msgs.append(msg)
    await client.disconnect()

    print(f"\nFetched {len(msgs)} messages from @{channel}\n")
    for m in reversed(msgs):
        entity_urls = [e.url for e in (m.entities or []) if hasattr(e, "url") and e.url]
        print(f"{'='*70}")
        print(f"MSG {m.id} | {m.date}")
        print(m.text)
        if entity_urls:
            print(f"[ENTITY URLS] {entity_urls}")
    print(f"{'='*70}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("channel")
    parser.add_argument("--limit", type=int, default=15)
    args = parser.parse_args()
    asyncio.run(dump(args.channel, args.limit))


if __name__ == "__main__":
    main()
