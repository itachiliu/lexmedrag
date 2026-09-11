"""Minimal DeepSeek chat client (stdlib only).

The API key is read from the DEEPSEEK_API_KEY environment variable or from
the project's .env file. It is never printed or logged.
"""

from __future__ import annotations

import json
import os
import urllib.request

API_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-chat"


def _load_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if key:
        return key.strip()
    env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("DEEPSEEK_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("DEEPSEEK_API_KEY not found in env or .env")


def chat(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 800,
    timeout: int = 120,
) -> str:
    """Send one chat completion request and return assistant content."""
    key = _load_key()
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + key,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def chat_json(prompt: str, retries: int = 1, **kwargs) -> dict:
    """Ask the model to reply with a single JSON object and parse it."""
    kwargs.setdefault("max_tokens", 1600)
    for attempt in range(retries + 1):
        try:
            text = chat(prompt, **kwargs)
            text = text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                text = text.rsplit("```", 1)[0]
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end < 0:
                raise ValueError("model did not return a JSON object")
            return json.loads(text[start : end + 1])
        except (json.JSONDecodeError, ValueError):
            if attempt == retries:
                raise
    raise RuntimeError("unreachable")
