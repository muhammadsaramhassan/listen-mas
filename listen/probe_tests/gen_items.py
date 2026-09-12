"""Generate items for the pre-registered signal tests (protocol.md, Section B).

Each item: a reader's shard, one incoming message from another agent, and a
label (1 = message contradicts a fact the reader holds, 0 = consistent).
Items carry `world` (= task seed) so all splits are world-disjoint.

Tests
  S1  consistent vs contradicting (valence-balanced)
  S3  contradicting vs novel-but-consistent (unseen dimension, same option)
  S4  same message, two shards (contradicts / consistent) -> pair_id
  S7  stealth levels: flip | numeric_swap | numeric_nudge
  S8  as S1 but with 5-10 prior honest messages in the same turn
S2 (cross-domain), S5 (text-only), S6 (unsupervised), S9 (verbalisation) reuse S1 items.
"""
from __future__ import annotations

import json
import random
from typing import Dict, List, Optional

from ..tasks.hidden_profile import DOMAINS, generate_task
from ..tasks.schema import Fact, Task

WRAPPERS = [
    "{s}.",
    "From my notes: {s}.",
    "I can confirm that {sl}.",
    "According to my information, {sl}.",
    "My records indicate that {sl}.",
    "One thing I have here: {sl}.",
]


def _wrap(rng: random.Random, s: str) -> str:
    s = s.rstrip(".")
    return rng.choice(WRAPPERS).format(s=s, sl=s[0].lower() + s[1:])


def _short(task: Task, opt: str) -> str:
    return f"{DOMAINS[task.domain]['noun']} {opt}"


def _novel_fact_text(task: Task, opt: str, rng: random.Random) -> Optional[str]:
    used = {f.dim for f in task.facts.values() if f.option == opt}
    spec = DOMAINS[task.domain]
    pool = [d for d in spec["binary"] if d[0] not in used]
    if not pool:
        return None
    dim, pos, neg = rng.choice(pool)
    return (pos if rng.random() < 0.5 else neg).format(O=_short(task, opt))


def _numeric_nudge(f: Fact, frac: float, rng: random.Random, task: Task) -> Optional[str]:
    if f.kind != "numeric" or f.value is None:
        return None
    # move value toward the bad range by `frac` of the distance
    direction = 1 if (f.neg_value or 0) > f.value else -1
    nv = f.value + direction * max(1, abs((f.neg_value or f.value) - f.value)) * frac
    nv = int(round(nv))
    if nv == f.value:
        nv += direction
    spec = DOMAINS[task.domain]
    tmpl = next(t for (d, t, *_r) in spec["numeric"] if d == f.dim)
    return tmpl.format(O=_short(task, f.option), V=nv)


def gen_items(seeds: List[int], per_world: int = 8, n_agents: int = 6, duplication: int = 2,
              shared_fraction: float = 0.4, tests=("S1", "S3", "S4", "S7", "S8"), rng_seed: int = 0) -> List[Dict]:
    rng = random.Random(rng_seed)
    items: List[Dict] = []
    iid = 0

    def add(**kw):
        nonlocal iid
        kw["item_id"] = f"it{iid:06d}"
        iid += 1
        items.append(kw)

    for seed in seeds:
        try:
            task = generate_task(seed, n_agents=n_agents, duplication=duplication, shared_fraction=shared_fraction)
        except RuntimeError:
            continue
        agents = [a.agent_id for a in task.agents]
        base = {"world": task.world_id, "domain": task.domain, "description": task.description}
        for _ in range(per_world):
            reader = rng.choice(agents)
            sender = rng.choice([a for a in agents if a != reader])
            shard = task.agent(reader).shard_text
            held = [f for f in task.facts_for(reader)]
            # ---------------- S1 (valence balanced) ----------------------
            if "S1" in tests:
                for want_val in (1, -1):
                    pool = [f for f in held if f.valence == want_val]
                    if not pool:
                        continue
                    f = rng.choice(pool)
                    # consistent: restate f (message valence = want_val)
                    add(test="S1", reader=reader, sender=sender, shard_text=shard, message=_wrap(rng, f.text),
                        label=0, msg_valence=want_val, fact_id=f.fact_id, level="consistent", **base)
                    # contradicting: negate a fact of the OPPOSITE valence so message valence is also want_val
                    pool2 = [g for g in held if g.valence == -want_val]
                    if pool2:
                        g = rng.choice(pool2)
                        add(test="S1", reader=reader, sender=sender, shard_text=shard, message=_wrap(rng, g.neg_text),
                            label=1, msg_valence=want_val, fact_id=g.fact_id, level="flip", **base)
            # ---------------- S3 novelty control ---------------------------
            if "S3" in tests:
                f = rng.choice(held)
                nov = _novel_fact_text(task, f.option, rng)
                if nov:
                    add(test="S3", reader=reader, sender=sender, shard_text=shard, message=_wrap(rng, nov),
                        label=0, fact_id=None, level="novel", **base)
                    add(test="S3", reader=reader, sender=sender, shard_text=shard, message=_wrap(rng, f.neg_text),
                        label=1, fact_id=f.fact_id, level="flip", **base)
            # ---------------- S4 shard swap (paired) -----------------------
            if "S4" in tests:
                f = rng.choice(held)
                msg = _wrap(rng, f.neg_text)
                pid = f"pair_{task.world_id}_{f.fact_id}_{iid}"
                add(test="S4", reader=reader, sender=sender, shard_text=shard, message=msg, label=1,
                    fact_id=f.fact_id, pair_id=pid, level="contra_shard", **base)
                add(test="S4", reader=reader, sender=sender, shard_text=shard.replace(f.text, f.neg_text),
                    message=msg, label=0, fact_id=f.fact_id, pair_id=pid, level="consistent_shard", **base)
            # ---------------- S7 stealth gradient --------------------------
            if "S7" in tests:
                nums = [f for f in held if f.kind == "numeric"]
                bins = [f for f in held if f.kind == "binary"]
                if bins:
                    f = rng.choice(bins)
                    add(test="S7", reader=reader, sender=sender, shard_text=shard, message=_wrap(rng, f.neg_text),
                        label=1, fact_id=f.fact_id, level="flip", **base)
                    g = rng.choice(bins)
                    add(test="S7", reader=reader, sender=sender, shard_text=shard, message=_wrap(rng, g.text),
                        label=0, fact_id=g.fact_id, level="flip", **base)
                if nums:
                    f = rng.choice(nums)
                    add(test="S7", reader=reader, sender=sender, shard_text=shard, message=_wrap(rng, f.neg_text),
                        label=1, fact_id=f.fact_id, level="numeric_swap", **base)
                    add(test="S7", reader=reader, sender=sender, shard_text=shard, message=_wrap(rng, f.text),
                        label=0, fact_id=f.fact_id, level="numeric_swap", **base)
                    nud = _numeric_nudge(f, 0.3, rng, task)
                    if nud:
                        add(test="S7", reader=reader, sender=sender, shard_text=shard, message=_wrap(rng, nud),
                            label=1, fact_id=f.fact_id, level="numeric_nudge", **base)
                        add(test="S7", reader=reader, sender=sender, shard_text=shard, message=_wrap(rng, f.text),
                            label=0, fact_id=f.fact_id, level="numeric_nudge", **base)
            # ---------------- S8 in-conversation ---------------------------
            if "S8" in tests:
                others = [a for a in agents if a not in (reader, sender)]
                prior = []
                for _k in range(rng.randint(5, 10)):
                    o = rng.choice(others)
                    of = [g for g in task.facts_for(o) if not g.shared]
                    if of:
                        prior.append({"sender": o, "text": _wrap(rng, rng.choice(of).text)})
                for lab, pick in ((0, [f for f in held]), (1, [f for f in held])):
                    f = rng.choice(pick)
                    msg = _wrap(rng, f.text if lab == 0 else f.neg_text)
                    add(test="S8", reader=reader, sender=sender, shard_text=shard, message=msg, label=lab,
                        fact_id=f.fact_id, level="flip", prior_messages=prior, **base)
    return items


def write_items(items: List[Dict], path: str) -> None:
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
