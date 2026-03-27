#!/usr/bin/env python3
"""Test DeepSeek API connection"""

import os
import sys
from openai import OpenAI

# Load environment
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    print("❌ DEEPSEEK_API_KEY not found")
    sys.exit(1)

print(f"✅ API key found (length: {len(api_key)})")
print(f"   Key: sk-...{api_key[-8:]}")

# Try to create a simple request
try:
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )
    
    # Simple test request
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'API test successful' in one word."}
        ],
        max_tokens=10
    )
    
    print(f"✅ API connection successful!")
    print(f"   Response: {response.choices[0].message.content}")
    
except Exception as e:
    print(f"❌ API error: {type(e).__name__}")
    print(f"   {str(e)}")
    sys.exit(1)