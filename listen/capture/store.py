"""On-disk activation store.

Layout:
  <root>/activations/round_<t>/<reader>/<msg_id>.npz
  <root>/activations/index.jsonl    one line per stored capture

Each npz contains the pooled arrays from Capture.arrays (float16), e.g.
resid_mean [L+1, D], attn_mean [L, D], ... . The index line carries the
metadata needed to build datasets without opening the arrays:
  {task_id, round, reader, msg_id, sender, sender_role, recipients,
   token_span, n_prompt_tokens, path, flags}
"""
from __future__ import annotations

import json
import os
from typing import Dict, Iterator, List, Optional

import numpy as np

from ..backends.base import Capture


class ActivationStore:
    def __init__(self, root: str):
        self.root = os.path.join(root, "activations")
        os.makedirs(self.root, exist_ok=True)
        self.index_path = os.path.join(self.root, "index.jsonl")

    def save(self, task_id: str, round_idx: int, reader: str, cap: Capture, **meta) -> str:
        d = os.path.join(self.root, f"round_{round_idx}", reader)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{cap.msg_id}.npz")
        np.savez(path, **{k: np.asarray(v, dtype=np.float16) for k, v in cap.arrays.items()})
        rec = {"task_id": task_id, "round": round_idx, "reader": reader, "msg_id": cap.msg_id,
               "token_span": list(cap.token_span), "n_prompt_tokens": cap.n_prompt_tokens,
               "path": os.path.relpath(path, self.root), **cap.meta, **meta}
        with open(self.index_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return path

    def iter_index(self) -> Iterator[Dict]:
        if not os.path.exists(self.index_path):
            return iter(())
        with open(self.index_path) as f:
            return iter([json.loads(l) for l in f if l.strip()])

    def load(self, rec: Dict) -> Dict[str, np.ndarray]:
        with np.load(os.path.join(self.root, rec["path"])) as z:
            return {k: z[k] for k in z.files}

    def feature(self, rec: Dict, component: str = "resid", pooling: str = "mean",
                layer: Optional[int] = None) -> np.ndarray:
        arr = self.load(rec)[f"{component}_{pooling}"].astype(np.float32)
        return arr if layer is None else arr[layer]
