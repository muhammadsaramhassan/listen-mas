from __future__ import annotations

import random
from typing import Dict, List, Optional, Set


class CommGraph:
    """Directed adjacency: sender -> set of agents that receive its messages."""

    def __init__(self, agents: List[str], adj: Dict[str, Set[str]]):
        self.agents = list(agents)
        self.adj = {a: set(adj.get(a, set())) - {a} for a in agents}
        self.ejected: Set[str] = set()

    # ---- constructors --------------------------------------------------
    @classmethod
    def complete(cls, agents: List[str]) -> "CommGraph":
        return cls(agents, {a: set(agents) for a in agents})

    @classmethod
    def ring(cls, agents: List[str], k: int = 1) -> "CommGraph":
        n = len(agents)
        adj = {}
        for i, a in enumerate(agents):
            adj[a] = {agents[(i + d) % n] for d in range(1, k + 1)} | {agents[(i - d) % n] for d in range(1, k + 1)}
        return cls(agents, adj)

    @classmethod
    def random_k(cls, agents: List[str], k: int, seed: int = 0, symmetric: bool = True) -> "CommGraph":
        rng = random.Random(seed)
        adj: Dict[str, Set[str]] = {a: set() for a in agents}
        for a in agents:
            others = [b for b in agents if b != a]
            for b in rng.sample(others, min(k, len(others))):
                adj[a].add(b)
                if symmetric:
                    adj[b].add(a)
        return cls(agents, adj)

    @classmethod
    def from_config(cls, agents: List[str], cfg: Dict) -> "CommGraph":
        t = cfg.get("type", "complete")
        if t in ("complete", "broadcast"):
            return cls.complete(agents)
        if t == "ring":
            return cls.ring(agents, k=cfg.get("k", 1))
        if t == "random_k":
            return cls.random_k(agents, k=cfg.get("k", 2), seed=cfg.get("seed", 0),
                                symmetric=cfg.get("symmetric", True))
        if t == "explicit":
            return cls(agents, {a: set(v) for a, v in cfg["adjacency"].items()})
        raise ValueError(f"unknown graph type {t}")

    # ---- queries ---------------------------------------------------------
    def receivers(self, sender: str, to: str) -> List[str]:
        if sender in self.ejected:
            return []
        nbrs = {b for b in self.adj[sender] if b not in self.ejected}
        if to == "all":
            return sorted(nbrs)
        return [to] if to in nbrs else []

    def neighbours(self, a: str) -> List[str]:
        return sorted(b for b in self.adj[a] if b not in self.ejected)

    def eject(self, a: str) -> None:
        self.ejected.add(a)

    def active(self) -> List[str]:
        return [a for a in self.agents if a not in self.ejected]

    def to_json(self) -> Dict:
        return {"adjacency": {a: sorted(v) for a, v in self.adj.items()}, "ejected": sorted(self.ejected)}
