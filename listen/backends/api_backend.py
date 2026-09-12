"""OpenAI-compatible chat-completions backend. Black-box only.

Works with OpenAI, Anthropic-via-proxy, OpenRouter, or a local vLLM/TGI server
(`base_url=http://localhost:8000/v1`). API key is read from `api_key_env`.
"""
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

from .base import Backend, GenResult


class APIBackend(Backend):
    is_whitebox = False

    def __init__(self, model: str, base_url: Optional[str] = None,
                 api_key_env: str = "OPENAI_API_KEY", name: Optional[str] = None,
                 max_retries: int = 5, extra_body: Optional[Dict] = None):
        from openai import OpenAI
        key = os.environ.get(api_key_env, "EMPTY")
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.model = model
        self.name = name or f"api:{model}"
        self.max_retries = max_retries
        self.extra_body = extra_body or {}

    def generate(self, system, turns, spans=None, max_new_tokens=512, temperature=0.0) -> GenResult:
        msgs = ([{"role": "system", "content": system}] if system else []) + list(turns)
        delay = 1.0
        for attempt in range(self.max_retries):
            try:
                r = self.client.chat.completions.create(
                    model=self.model, messages=msgs, max_tokens=max_new_tokens,
                    temperature=temperature, **self.extra_body)
                text = r.choices[0].message.content or ""
                u = getattr(r, "usage", None)
                return GenResult(text=text, prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
                                 completion_tokens=getattr(u, "completion_tokens", 0) or 0)
            except Exception as e:  # noqa: BLE001
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 30)
        raise RuntimeError("unreachable")
