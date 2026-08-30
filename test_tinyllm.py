#!/usr/bin/env python3
"""Smoke-test for TinyLLM proxy with verbose output."""
import json
import os
import urllib.error
import urllib.request

BASE = os.environ.get("TINYLLM_BASE", "http://172.29.0.1:4100")
TOKEN = os.environ.get("TINYLLM_API_KEY", "")

def req(method, path, body=None):
    data = json.dumps(body).encode() if body else None
    r = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {TOKEN}",
        },
        method=method,
    )
    try:
        resp = urllib.request.urlopen(r)
        return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())

def main():
    if not TOKEN:
        raise SystemExit("Set TINYLLM_API_KEY before running the smoke test")

    # 1. Health
    status, data = req("GET", "/health/liveliness")
    print(f"HEALTH: {status} {data}")
    assert status == 200 and data.get("status") == "ok"

    # 2. Models
    status, data = req("GET", "/v1/models")
    print(f"MODELS: {status} {data}")
    assert status == 200 and len(data.get("data", [])) > 0

    # 3. Chat non-streaming with 100 max_tokens
    status, data = req("POST", "/v1/chat/completions", {
        "model": "coding-auto",
        "messages": [{"role": "user", "content": "say hi in one word"}],
        "max_tokens": 100,
    })
    print(f"CHAT:   {status}")
    if status == 200:
        print(f"  full response: {json.dumps(data, indent=2)}")
        msg = data["choices"][0]["message"]
        print(f"  content: {msg.get('content', '')!r}")
        print(f"  reasoning: {msg.get('reasoning_content', 'N/A')!r}")
        print(f"  finish_reason: {data['choices'][0]['finish_reason']}")
    else:
        print(f"  error: {data}")


if __name__ == "__main__":
    main()
