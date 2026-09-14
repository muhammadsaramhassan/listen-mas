"""Core data structures shared across the environment.

A Task is what the engine runs. It is deliberately model-agnostic and
JSON-serialisable so tasks can be generated once and re-used across runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import json


@dataclass
class Fact:
    """One atomic piece of information about an option.

    valence: +1 (favours the option) or -1 (disfavours it).
    kind:    'binary' (trait present/absent) or 'numeric' (has a number).
    holders: agent ids that hold this fact. Shared facts are held by all.
    text:    natural-language rendering as given to holders.
    neg_text: rendering of the *opposite* claim (used to build contradictions).
    """
    fact_id: str
    option: str
    dim: str
    valence: int
    kind: str
    text: str
    neg_text: str
    shared: bool
    holders: List[str]
    value: Optional[float] = None       # numeric facts only
    neg_value: Optional[float] = None   # numeric facts only


@dataclass
class AgentSpec:
    agent_id: str
    shard_text: str                     # what the agent sees at start
    fact_ids: List[str] = field(default_factory=list)


@dataclass
class Task:
    task_id: str
    kind: str                           # 'hidden_profile' | 'silo'
    world_id: str
    domain: str
    description: str                    # scenario text common to all agents
    agents: List[AgentSpec]
    possible_answers: List[str]
    correct_answer: str                 # hidden_profile: option letter; silo: str(value)
    facts: Dict[str, Fact] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    # ---- serialisation -------------------------------------------------
    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "Task":
        facts = {k: Fact(**v) for k, v in d.get("facts", {}).items()}
        agents = [AgentSpec(**a) for a in d["agents"]]
        return Task(
            task_id=d["task_id"], kind=d["kind"], world_id=d["world_id"],
            domain=d["domain"], description=d["description"], agents=agents,
            possible_answers=d["possible_answers"], correct_answer=d["correct_answer"],
            facts=facts, meta=d.get("meta", {}),
        )

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=1)

    @staticmethod
    def load(path: str) -> "Task":
        with open(path) as f:
            return Task.from_json(json.load(f))

    # ---- helpers ---------------------------------------------------------
    def agent(self, agent_id: str) -> AgentSpec:
        for a in self.agents:
            if a.agent_id == agent_id:
                return a
        raise KeyError(agent_id)

    def facts_for(self, agent_id: str) -> List[Fact]:
        return [self.facts[fid] for fid in self.agent(agent_id).fact_ids]

    @property
    def n_agents(self) -> int:
        return len(self.agents)
