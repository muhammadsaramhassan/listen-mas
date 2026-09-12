"""Scripted backend for tests. A policy callable receives (system, turns) and
returns the assistant text. Optionally emits fake captures so the capture
store / orchestrator path can be exercised without a GPU."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from .base import Backend, Capture, GenResult, Span


class MockBackend(Backend):
    def __init__(self, policy: Callable[[str, List[Dict[str, str]]], str],
                 fake_capture: bool = False, n_layers: int = 4, d_model: int = 16,
                 capture_signal: Optional[Callable[[Span], float]] = None, seed: int = 0,
                 name: str = "mock"):
        self.policy = policy
        self.is_whitebox = fake_capture
        self.n_layers = n_layers
        self.d = d_model
        self.signal = capture_signal
        self.rng = np.random.default_rng(seed)
        self.name = name

    def _fake(self, sp: Span) -> Capture:
        shift = self.signal(sp) if self.signal else 0.0
        arrays = {}
        for comp, L in (("resid", self.n_layers + 1), ("attn", self.n_layers), ("mlp", self.n_layers)):
            for pool in ("mean", "last"):
                a = self.rng.normal(0, 1, size=(L, self.d)).astype(np.float16)
                a[:, 0] += shift          # signal lives in dimension 0
                arrays[f"{comp}_{pool}"] = a
        return Capture(msg_id=sp.msg_id, arrays=arrays, token_span=(0, 1), n_prompt_tokens=1, meta=dict(sp.meta))

    def generate(self, system, turns, spans=None, max_new_tokens=512, temperature=0.0) -> GenResult:
        text = self.policy(system, turns)
        caps = [self._fake(sp) for sp in (spans or [])] if self.is_whitebox else []
        return GenResult(text=text, captures=caps)

    def capture_only(self, system, turns, spans):
        return [self._fake(sp) for sp in spans]

    def option_logprobs(self, system, turns, options):
        return {o: float(np.log(1.0 / len(options))) for o in options}
