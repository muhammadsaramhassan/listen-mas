"""Same-round, single-reader message ablation.

For reader r at round t, we rebuild the context r saw when it ingested the
round-t messages, then compare its answer distribution with and without one
[MSG] block. This is a *same-round influence* measure, not a counterfactual on
task outcome (removing the message does not undo its downstream effects).

Usage (offline, needs a white-box backend for the reader's model):
    from listen.ablate import ablate_round
    res = ablate_round(run_dir, reader="agent_3", round_idx=2, backend=wb, options=["A","B","C","D"])
"""
from __future__ import annotations

import json
import math
import os
import re
from typing import Dict, List, Optional

import numpy as np

from .backends.base import Backend

_MSG_BLOCK = re.compile(r"\[MSG id=(\w+) from=(\w+)\][^\[]*?\[/MSG\]\n?", re.S)

ANSWER_PROBE = ("Based on everything you know so far, which option is best? "
                "Reply with only the option letter and nothing else.")


def _load_context(run_dir: str, reader: str, round_idx: int) -> Dict:
    p = os.path.join(run_dir, "rounds", f"round_{round_idx}", reader, "context.json")
    return json.load(open(p))


def _turns_through_user(ctx: Dict, round_idx: int) -> List[Dict[str, str]]:
    """Turns up to and including the round's user message (exclude the reply)."""
    turns = ctx["turns"]
    # round k's user turn is at index 2*(k-1) for agents active since round 1
    idx = 2 * (round_idx - 1)
    if idx >= len(turns) or turns[idx]["role"] != "user" or f"ROUND {round_idx} " not in turns[idx]["content"]:
        # fall back: find the user turn mentioning this round
        for i, t in enumerate(turns):
            if t["role"] == "user" and f"ROUND {round_idx} " in t["content"]:
                idx = i
                break
    return turns[: idx + 1]


def _dist(lp: Dict[str, float]) -> np.ndarray:
    v = np.array([lp[o] for o in sorted(lp)], dtype=np.float64)
    v = np.exp(v - v.max())
    return v / v.sum()


def ablate_round(run_dir: str, reader: str, round_idx: int, backend: Backend, options: List[str],
                 msg_ids: Optional[List[str]] = None) -> List[Dict]:
    ctx = _load_context(run_dir, reader, round_idx)
    system = ctx["system"]
    turns = _turns_through_user(ctx, round_idx)
    user = turns[-1]["content"]
    blocks = {m.group(1): (m.group(2), m.group(0)) for m in _MSG_BLOCK.finditer(user)}
    if msg_ids is None:
        msg_ids = list(blocks)
    probe_turns = turns + [{"role": "user", "content": ANSWER_PROBE}]
    full_lp = backend.option_logprobs(system, probe_turns, options)
    p_full = _dist(full_lp)
    out = []
    for mid in msg_ids:
        if mid not in blocks:
            continue
        sender, block = blocks[mid]
        user_abl = user.replace(block, "")
        if user_abl == user:
            continue
        turns_abl = turns[:-1] + [{"role": "user", "content": user_abl}]
        lp = backend.option_logprobs(system, turns_abl + [{"role": "user", "content": ANSWER_PROBE}], options)
        p_abl = _dist(lp)
        kl = float(np.sum(p_full * (np.log(p_full + 1e-12) - np.log(p_abl + 1e-12))))
        opts = sorted(options)
        out.append({"reader": reader, "round": round_idx, "msg_id": mid, "sender": sender,
                    "p_full": dict(zip(opts, p_full.round(4).tolist())),
                    "p_ablated": dict(zip(opts, p_abl.round(4).tolist())),
                    "influence_l1": float(np.abs(p_full - p_abl).sum()), "influence_kl": kl,
                    "shift_toward": opts[int(np.argmax(p_full - p_abl))]})
    return out


def ablate_run(run_dir: str, backend: Backend, readers: Optional[List[str]] = None) -> List[Dict]:
    task = json.load(open(os.path.join(run_dir, "task.json")))
    options = task["possible_answers"]
    roles = json.load(open(os.path.join(run_dir, "roles.json")))
    readers = readers or roles["whitebox"]
    rounds = sorted(int(d.split("_")[1]) for d in os.listdir(os.path.join(run_dir, "rounds")))
    out = []
    for r in readers:
        for t in rounds:
            if os.path.exists(os.path.join(run_dir, "rounds", f"round_{t}", r, "context.json")):
                out += ablate_round(run_dir, r, t, backend, options)
    with open(os.path.join(run_dir, "ablation.jsonl"), "w") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")
    return out
