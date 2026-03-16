"""
analyze_channel.py — Telegram channel signal quality analyzer.

Fetches the last N posts from a Telegram channel and uses Claude to evaluate
signal quality, detect fake/inflated market cap claims, and assess whether
the channel is worth adding to the trading bot's monitoring list.

Usage:
    python analyze_channel.py <channel_username> [--limit 25]

Examples:
    python analyze_channel.py AlphaStrikeSol
    python analyze_channel.py WizzyTrades --limit 30

Requires ANTHROPIC_API_KEY in your .env file.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from typing import Optional

import anthropic
from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.sessions import StringSession

load_dotenv()

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _require(key: str) -> str:
    val = os.getenv(key, "").strip()
    if not val:
        print(f"[ERROR] {key} is required in your .env file.", file=sys.stderr)
        sys.exit(1)
    return val


# ---------------------------------------------------------------------------
# Telegram fetch
# ---------------------------------------------------------------------------

async def fetch_messages(channel: str, limit: int) -> list[dict]:
    """Connect to Telegram and pull the last `limit` messages from `channel`."""
    tg_api_id    = int(_require("TG_API_ID"))
    tg_api_hash  = _require("TG_API_HASH")
    tg_phone     = os.getenv("TG_PHONE", "").strip() or None
    session_str  = os.getenv("TG_SESSION_STRING", "").strip()
    session_file = "trader/listener/tg_session"

    session = StringSession(session_str) if session_str else session_file
    client  = TelegramClient(session, tg_api_id, tg_api_hash)

    print(f"[TG] Connecting...")
    await client.start(phone=tg_phone)
    me = await client.get_me()
    print(f"[TG] Authenticated as: {me.username or me.first_name}")

    try:
        entity = await client.get_entity(channel)
    except Exception as exc:
        print(f"[ERROR] Could not resolve channel '{channel}': {exc}", file=sys.stderr)
        await client.disconnect()
        sys.exit(1)

    print(f"[TG] Fetching last {limit} messages from '{channel}'...")
    messages = []
    async for msg in client.iter_messages(entity, limit=limit):
        if not (msg.text or "").strip():
            continue

        # Extract embedded URLs from message entities
        entity_urls = [
            ent.url
            for ent in (msg.entities or [])
            if hasattr(ent, "url") and ent.url
        ]

        messages.append({
            "id":          msg.id,
            "date":        msg.date.strftime("%Y-%m-%d %H:%M:%S UTC") if msg.date else "",
            "text":        msg.text.strip(),
            "entity_urls": entity_urls,
            "edited":      bool(msg.edit_date),
            "edit_date":   msg.edit_date.strftime("%Y-%m-%d %H:%M:%S UTC") if msg.edit_date else None,
        })

    await client.disconnect()
    messages.reverse()   # chronological order
    print(f"[TG] Fetched {len(messages)} non-empty messages.\n")
    return messages


# ---------------------------------------------------------------------------
# Claude analysis
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert cryptocurrency signal channel auditor. Your job is to analyze \
Telegram channel messages and provide an honest, detailed quality assessment to \
help decide whether the channel is trustworthy enough to use as an automated \
trading signal source.

Focus on:
1. **Signal vs Update ratio** — What fraction of posts are fresh token calls vs \
   price-update follow-ups? A channel that mostly posts updates is less useful.
2. **Market cap accuracy** — When a post claims a token is at a certain MC \
   (e.g. "27K MC"), then a later update shows a large profit (e.g. "+252% from \
   signal entry"), the actual MC at signal time was much higher than claimed. \
   Flag this as inflated/misleading MC reporting.
3. **Missing originals** — If you see update posts that reference a signal ("from \
   signal entry", "Updated:", profit %%) but you do NOT see the corresponding \
   original signal post in the data, the channel may selectively delete bad calls.
4. **Mint address presence** — Good signal channels include the contract/mint \
   address so bots can act on them. Flag channels that omit mint addresses.
5. **Pump.fun concentration** — Heavy use of pump.fun tokens is a red flag; they \
   are extremely high risk and often manipulated.
6. **Overall verdict**: RECOMMENDED, BORDERLINE, or DO NOT USE — with a brief \
   rationale and any conditions for use.

Be specific. Quote message IDs and text snippets as evidence.
"""


def analyze_with_claude(channel: str, messages: list[dict]) -> None:
    """Send messages to Claude and stream the analysis."""
    api_key = _require("ANTHROPIC_API_KEY")
    client  = anthropic.Anthropic(api_key=api_key)

    # Format messages for Claude
    formatted = []
    for m in messages:
        urls = f"\n  URLs: {', '.join(m['entity_urls'])}" if m["entity_urls"] else ""
        edited = f"\n  [EDITED at {m['edit_date']}]" if m["edited"] else ""
        formatted.append(
            f"--- Message {m['id']} | {m['date']}{edited} ---\n{m['text']}{urls}"
        )

    messages_block = "\n\n".join(formatted)
    user_content = (
        f"Channel: @{channel}\n"
        f"Total messages analyzed: {len(messages)}\n\n"
        f"=== MESSAGES (oldest → newest) ===\n\n"
        f"{messages_block}\n\n"
        f"=== END OF MESSAGES ===\n\n"
        f"Please provide your full channel quality assessment."
    )

    print("=" * 72)
    print(f"  CHANNEL ANALYSIS: @{channel}")
    print("=" * 72)
    print()

    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)

    print()
    print()
    print("=" * 72)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a Telegram signal channel with Claude.")
    parser.add_argument("channel", help="Telegram channel username (without @)")
    parser.add_argument("--limit", type=int, default=25, help="Number of recent messages to fetch (default: 25)")
    args = parser.parse_args()

    messages = asyncio.run(fetch_messages(args.channel, args.limit))
    if not messages:
        print("[WARN] No messages found — nothing to analyze.")
        sys.exit(0)

    analyze_with_claude(args.channel, messages)


if __name__ == "__main__":
    main()
