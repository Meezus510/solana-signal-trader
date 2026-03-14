"""
generate_session_string.py — One-time script to produce a TG_SESSION_STRING.

Run locally:
    python generate_session_string.py

It will prompt for your phone number + SMS code, then print a session string.
Paste that string into DigitalOcean App Platform as the TG_SESSION_STRING env var.
You never need to run this again unless you revoke the session.
"""

import asyncio
from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.sessions import StringSession
from trader.config import Config

load_dotenv()


async def main() -> None:
    cfg = Config.load()
    async with TelegramClient(StringSession(), cfg.tg_api_id, cfg.tg_api_hash) as client:
        await client.start(phone=cfg.tg_phone or None)
        print("\n" + "=" * 60)
        print("TG_SESSION_STRING (add this to DigitalOcean env vars):")
        print("=" * 60)
        print(client.session.save())
        print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
