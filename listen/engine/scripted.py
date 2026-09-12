"""Deterministic scripted policies for hidden-profile tasks. Used with the mock
backend to test the full pipeline without a language model.

Honest policy: round 1 shares all its hidden facts; later rounds integrate
received statements and answer with the option of highest known valence.
Malicious policy: same, but substitutes assigned false claims with probability
`rate` (T1) or always (T0), and answers the target option.
"""
from __future__ import annotations

import random
import re
from typing import Dict, List, Optional

from ..tasks.schema import Task
from .protocol import MSG_RE

_MSG_LINE = re.compile(r"\[MSG id=(\w+) from=(\w+)\]\s*(.*?)\s*\[/MSG\]", re.S)


class ScriptedHP:
    def __init__(self, task: Task, agent_id: str, malicious: Optional[Dict] = None, seed: int = 0):
        self.task = task
        self.aid = agent_id
        self.mal = malicious                # {"claims": [FalseClaim], "target": "B", "rate": 0.2, "tier": "T1"}
        self.rng = random.Random(seed)
        self.known: Dict[str, int] = {}     # statement text -> valence (as believed)
        # index statements -> (option, valence)
        self.lookup = {}
        for f in task.facts.values():
            self.lookup[f.text] = (f.option, f.valence)
            self.lookup[f.neg_text] = (f.option, -f.valence)
        for f in task.facts_for(agent_id):
            self.known[f.text] = f.valence

    def _ingest(self, user_text: str):
        for _, sender, content in _MSG_LINE.findall(user_text):
            for stmt in [s.strip() for s in content.split(";") if s.strip()]:
                if stmt in self.lookup:
                    self.known[stmt] = self.lookup[stmt][1]

    def _scores(self) -> Dict[str, int]:
        sc = {o: 0 for o in self.task.possible_answers}
        for stmt, val in self.known.items():
            opt = self.lookup.get(stmt, (None, 0))[0]
            if opt in sc:
                sc[opt] += val
        return sc

    def __call__(self, system: str, turns: List[Dict[str, str]]) -> str:
        user = turns[-1]["content"]
        self._ingest(user)
        round_idx = int(re.search(r"ROUND (\d+)", user).group(1))
        out = []
        if round_idx <= 2:
            mine = [f for f in self.task.facts_for(self.aid) if not f.shared]
            half = len(mine) // 2 + 1
            share = mine[:half] if round_idx == 1 else mine[half:]
            stmts = []
            for f in share:
                text = f.text
                if self.mal:
                    for c in self.mal["claims"]:
                        if c.fact_id == f.fact_id and (self.mal["tier"] == "T0" or self.rng.random() < self.mal["rate"] * 3):
                            text = c.false_text
                stmts.append(text)
            if stmts:
                out.append(f'<message to="all">{"; ".join(stmts)}</message>')
        if round_idx >= 2:
            if self.mal:
                ans = self.mal["target"]
            else:
                sc = self._scores()
                ans = max(sc, key=sc.get)
            out.append(f"<answer>{ans}</answer>")
        return "\n".join(out) if out else "<wait/>"
