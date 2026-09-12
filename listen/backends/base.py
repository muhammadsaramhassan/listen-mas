from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Span:
    """A piece of text inside the prompt whose activations we want.

    msg_id:  identifier (message uuid)
    text:    exact text as it appears in the prompt (the full [MSG]...[/MSG] block)
    """
    msg_id: str
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Capture:
    """Pooled activations for one span.

    arrays: dict name -> np.ndarray, e.g.
        resid_mean [L+1, D], resid_last [L+1, D],
        attn_mean  [L, D],   attn_last  [L, D],
        mlp_mean   [L, D],   mlp_last   [L, D]
    token_span: (start, end) token indices in the prompt
    """
    msg_id: str
    arrays: Dict[str, Any]
    token_span: Tuple[int, int]
    n_prompt_tokens: int
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenResult:
    text: str
    captures: List[Capture] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: Dict[str, Any] = field(default_factory=dict)


class Backend:
    """Common interface. `turns` is a list of {"role": ..., "content": ...}
    excluding the system message, which is passed separately."""

    name: str = "base"
    is_whitebox: bool = False

    def generate(self, system: str, turns: List[Dict[str, str]],
                 spans: Optional[List[Span]] = None,
                 max_new_tokens: int = 512, temperature: float = 0.0) -> GenResult:
        raise NotImplementedError

    def capture_only(self, system: str, turns: List[Dict[str, str]],
                     spans: List[Span]) -> List[Capture]:
        raise NotImplementedError("this backend does not expose internals")

    def option_logprobs(self, system: str, turns: List[Dict[str, str]],
                        options: List[str]) -> Dict[str, float]:
        raise NotImplementedError("this backend does not expose internals")
