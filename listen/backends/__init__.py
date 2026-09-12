from __future__ import annotations

from typing import Any, Dict

from .base import Backend, Capture, GenResult, Span


def build_backend(cfg: Dict[str, Any], **overrides) -> Backend:
    """cfg example:
      {type: whitebox, model: Qwen/Qwen3-8B, dtype: bfloat16,
       chat_template_kwargs: {enable_thinking: false}}
      {type: hf_blackbox, model: Qwen/Qwen3-8B}
      {type: api, model: gpt-4o-mini, base_url: null, api_key_env: OPENAI_API_KEY}
      {type: mock}   (policy injected by the runner)
    """
    cfg = {**cfg, **overrides}
    t = cfg["type"]
    if t in ("whitebox", "hf_blackbox"):
        from .hf_backend import HFBackend
        gen_be = build_backend(cfg["gen_backend"]) if cfg.get("gen_backend") else None
        return HFBackend(gen_backend=gen_be, model_name=cfg["model"], capture=(t == "whitebox"),
                         dtype=cfg.get("dtype", "bfloat16"), device_map=cfg.get("device_map", "auto"),
                         chat_template_kwargs=cfg.get("chat_template_kwargs"),
                         components=tuple(cfg.get("components", ("resid", "attn", "mlp"))),
                         poolings=tuple(cfg.get("poolings", ("mean", "last"))),
                         layers=cfg.get("layers"), trust_remote_code=cfg.get("trust_remote_code", False),
                         tiny_random=cfg.get("tiny_random", False), name=cfg.get("name"))
    if t == "api":
        from .api_backend import APIBackend
        return APIBackend(model=cfg["model"], base_url=cfg.get("base_url"),
                          api_key_env=cfg.get("api_key_env", "OPENAI_API_KEY"), name=cfg.get("name"),
                          extra_body=cfg.get("extra_body"))
    if t == "mock":
        from .mock_backend import MockBackend
        return MockBackend(policy=cfg["policy"], fake_capture=cfg.get("fake_capture", False),
                           capture_signal=cfg.get("capture_signal"), seed=cfg.get("seed", 0),
                           name=cfg.get("name", "mock"))
    raise ValueError(f"unknown backend type {t}")


__all__ = ["Backend", "Capture", "GenResult", "Span", "build_backend"]
