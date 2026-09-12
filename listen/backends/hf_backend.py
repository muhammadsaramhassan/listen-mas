"""HuggingFace transformers backend.

Two modes behind one class:
  * capture=True  -> white-box reader: hooks every decoder layer, pools
                     activations over requested message spans.
  * capture=False -> black-box proxy: same weights (shared in memory via the
                     registry), no hooks, no activation output. Used to stand in
                     for API agents without spending API credits.

Captured components per layer l (0..L-1):
  resid  : hidden_states[l+1]  (residual stream after layer l; index 0 = embeddings)
  attn   : output of layer.self_attn (attention block contribution)
  mlp    : output of layer.mlp       (MLP block contribution)
Pooled two ways over the span: mean over tokens, last token.
Stored as float16 numpy arrays.
"""
from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .base import Backend, Capture, GenResult, Span

log = logging.getLogger(__name__)

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class HFModelRegistry:
    """Load each (model_name, dtype) once and share across agents."""
    _lock = threading.Lock()
    _models: Dict[Tuple[str, str], Tuple[object, object]] = {}

    @classmethod
    def get(cls, model_name: str, dtype: str = "bfloat16", device_map: str = "auto",
            trust_remote_code: bool = False, tiny_random: bool = False):
        key = (model_name, dtype)
        with cls._lock:
            if key in cls._models:
                return cls._models[key]
            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
            tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            if tiny_random:
                # test-only: random weights with the architecture of model_name but tiny
                cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
                for k, v in dict(hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                                 num_attention_heads=4, num_key_value_heads=2, head_dim=16).items():
                    if hasattr(cfg, k):
                        setattr(cfg, k, v)
                try:
                    model = AutoModelForCausalLM.from_config(cfg, dtype=_DTYPES[dtype])
                except TypeError:
                    model = AutoModelForCausalLM.from_config(cfg, torch_dtype=_DTYPES[dtype])
            else:
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name, dtype=_DTYPES[dtype], device_map=device_map,
                        trust_remote_code=trust_remote_code)
                except TypeError:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name, torch_dtype=_DTYPES[dtype], device_map=device_map,
                        trust_remote_code=trust_remote_code)
            model.eval()
            cls._models[key] = (model, tok)
            return model, tok


def _find_layers(model) -> List[torch.nn.Module]:
    for attr in ("model", "transformer", "language_model"):
        m = getattr(model, attr, None)
        if m is not None and hasattr(m, "layers"):
            return list(m.layers)
        if m is not None and hasattr(m, "h"):
            return list(m.h)
    if hasattr(model, "layers"):
        return list(model.layers)
    raise RuntimeError("could not locate decoder layers on model")


class HFBackend(Backend):
    def __init__(self, model_name: str, capture: bool = False, dtype: str = "bfloat16",
                 device_map: str = "auto", chat_template_kwargs: Optional[Dict] = None,
                 components: Tuple[str, ...] = ("resid", "attn", "mlp"),
                 poolings: Tuple[str, ...] = ("mean", "last"),
                 layers: Optional[List[int]] = None, trust_remote_code: bool = False,
                 tiny_random: bool = False, name: Optional[str] = None,
                 gen_backend: Optional[Backend] = None):
        self.model, self.tok = HFModelRegistry.get(model_name, dtype, device_map,
                                                   trust_remote_code, tiny_random)
        self.model_name = model_name
        self.is_whitebox = capture
        self.name = name or ("hf_whitebox" if capture else "hf_blackbox")
        self.ctk = chat_template_kwargs or {}
        self.components = tuple(components)
        self.poolings = tuple(poolings)
        self.layer_filter = layers
        self._buf: Dict[str, List[torch.Tensor]] = {"attn": [], "mlp": []}
        self._hooks_active = False
        self._handles = []
        # optional: delegate text generation (e.g. to a vLLM server serving the same weights);
        # activation capture still happens here on the HF copy.
        self.gen_backend = gen_backend
        if capture:
            self._install_hooks()

    # ------------------------------------------------------------------
    @property
    def device(self):
        return next(self.model.parameters()).device

    def _install_hooks(self):
        layers = _find_layers(self.model)
        for li, layer in enumerate(layers):
            if "attn" in self.components and hasattr(layer, "self_attn"):
                self._handles.append(layer.self_attn.register_forward_hook(self._mk_hook("attn", li)))
            if "mlp" in self.components and hasattr(layer, "mlp"):
                self._handles.append(layer.mlp.register_forward_hook(self._mk_hook("mlp", li)))
        self.n_layers = len(layers)

    def _mk_hook(self, comp: str, li: int):
        def hook(module, inputs, output):
            if not self._hooks_active:
                return
            out = output[0] if isinstance(output, tuple) else output
            if out.dim() == 3 and out.shape[1] > 1:      # prefill only
                self._buf[comp].append(out.detach())
        return hook

    # ------------------------------------------------------------------
    def _render(self, system: str, turns: List[Dict[str, str]], add_generation_prompt=True) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + list(turns)
        return self.tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=add_generation_prompt, **self.ctk)

    def _encode(self, prompt: str):
        enc = self.tok(prompt, return_tensors="pt", return_offsets_mapping=True,
                       add_special_tokens=False)
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(self.device) for k, v in enc.items()}
        return enc, offsets

    @staticmethod
    def _char_to_token_span(offsets, c0: int, c1: int) -> Tuple[int, int]:
        toks = [i for i, (a, b) in enumerate(offsets) if b > c0 and a < c1 and b > a]
        if not toks:
            raise ValueError("span maps to no tokens")
        return toks[0], toks[-1] + 1

    def _locate_spans(self, prompt: str, offsets, spans: List[Span]) -> List[Tuple[Span, Tuple[int, int]]]:
        out = []
        cursor = 0
        for sp in spans:
            c0 = prompt.find(sp.text, cursor)
            if c0 < 0:
                c0 = prompt.find(sp.text)      # fall back to global search
                if c0 < 0:
                    log.warning("span %s not found verbatim in prompt; skipping", sp.msg_id)
                    continue
            c1 = c0 + len(sp.text)
            cursor = c1
            out.append((sp, self._char_to_token_span(offsets, c0, c1)))
        return out

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _forward_capture(self, enc) -> Tuple[List[torch.Tensor], Dict[str, List[torch.Tensor]]]:
        self._buf = {"attn": [], "mlp": []}
        self._hooks_active = True
        try:
            out = self.model(**enc, output_hidden_states=True, use_cache=False)
        finally:
            self._hooks_active = False
        hs = [h[0] for h in out.hidden_states]           # (L+1) x [T, D]
        comps = {k: [t[0] for t in v] for k, v in self._buf.items()}
        self._buf = {"attn": [], "mlp": []}
        return hs, comps

    def _pool(self, hs, comps, tspan: Tuple[int, int]) -> Dict[str, np.ndarray]:
        s, e = tspan
        arrays: Dict[str, np.ndarray] = {}

        def _sel(stack: torch.Tensor):
            if self.layer_filter is not None:
                idx = torch.tensor(self.layer_filter, device=stack.device)
                stack = stack.index_select(0, idx)
            return stack

        if "resid" in self.components:
            R = torch.stack(hs, 0)[:, s:e, :]            # [L+1, n, D]
            R = _sel(R)
            if "mean" in self.poolings:
                arrays["resid_mean"] = R.float().mean(1).to(torch.float16).cpu().numpy()
            if "last" in self.poolings:
                arrays["resid_last"] = R[:, -1, :].to(torch.float16).cpu().numpy()
        for comp in ("attn", "mlp"):
            if comp in self.components and comps.get(comp):
                C = torch.stack(comps[comp], 0)[:, s:e, :]  # [L, n, D]
                C = _sel(C)
                if "mean" in self.poolings:
                    arrays[f"{comp}_mean"] = C.float().mean(1).to(torch.float16).cpu().numpy()
                if "last" in self.poolings:
                    arrays[f"{comp}_last"] = C[:, -1, :].to(torch.float16).cpu().numpy()
        return arrays

    # ------------------------------------------------------------------
    def capture_only(self, system, turns, spans: List[Span]) -> List[Capture]:
        if not self.is_whitebox:
            raise RuntimeError("capture_only called on a black-box backend")
        prompt = self._render(system, turns)
        enc, offsets = self._encode(prompt)
        located = self._locate_spans(prompt, offsets, spans)
        if not located:
            return []
        hs, comps = self._forward_capture(enc)
        caps = []
        for sp, tspan in located:
            caps.append(Capture(msg_id=sp.msg_id, arrays=self._pool(hs, comps, tspan),
                                token_span=tspan, n_prompt_tokens=enc["input_ids"].shape[1],
                                meta=dict(sp.meta)))
        del hs, comps
        return caps

    @torch.no_grad()
    def generate(self, system, turns, spans=None, max_new_tokens=512, temperature=0.0) -> GenResult:
        prompt = self._render(system, turns)
        enc, offsets = self._encode(prompt)
        captures: List[Capture] = []
        if self.is_whitebox and spans:
            located = self._locate_spans(prompt, offsets, spans)
            if located:
                hs, comps = self._forward_capture(enc)
                for sp, tspan in located:
                    captures.append(Capture(msg_id=sp.msg_id, arrays=self._pool(hs, comps, tspan),
                                            token_span=tspan, n_prompt_tokens=enc["input_ids"].shape[1],
                                            meta=dict(sp.meta)))
                del hs, comps
        if self.gen_backend is not None:
            g = self.gen_backend.generate(system, turns, None, max_new_tokens, temperature)
            return GenResult(text=g.text, captures=captures, prompt_tokens=enc["input_ids"].shape[1],
                             completion_tokens=g.completion_tokens)
        gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id)
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature)
        else:
            gen_kwargs.update(do_sample=False)
        self._hooks_active = False
        out = self.model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], **gen_kwargs)
        new = out[0, enc["input_ids"].shape[1]:]
        text = self.tok.decode(new, skip_special_tokens=True)
        return GenResult(text=text, captures=captures, prompt_tokens=enc["input_ids"].shape[1],
                         completion_tokens=int(new.shape[0]))

    @torch.no_grad()
    def option_logprobs(self, system, turns, options: List[str]) -> Dict[str, float]:
        """Log-prob of each option's first token at the next position (no generation)."""
        if not self.is_whitebox:
            raise RuntimeError("option_logprobs requires a white-box backend")
        prompt = self._render(system, turns)
        enc, _ = self._encode(prompt)
        logits = self.model(**enc, use_cache=False).logits[0, -1].float()
        lp = torch.log_softmax(logits, -1)
        res = {}
        for opt in options:
            cands = set()
            for v in (opt, " " + opt):
                ids = self.tok(v, add_special_tokens=False)["input_ids"]
                if ids:
                    cands.add(ids[0])
            res[opt] = max(float(lp[i]) for i in cands) if cands else float("-inf")
        return res
