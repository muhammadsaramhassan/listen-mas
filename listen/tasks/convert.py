"""Converters into the internal Task format.

* Silo-Bench (acl26-silo-bench/benchmarks/*.json) numeric tasks -> Task(kind='silo')
* HiddenBench official (benchmark.json, 65 items) -> Task(kind='hidden_profile')
"""
from __future__ import annotations

import json
import os
import re
from typing import List, Optional

from .schema import AgentSpec, Task
from .hidden_profile import from_hiddenbench, generate_task


def silo_to_task(path: str) -> Task:
    d = json.load(open(path))
    agents = []
    for cfg in d["agent_configs"]:
        # The Silo-Bench user_prompt already embeds the agent's shard. We keep
        # only the "Your Data" line as the shard, and use the task text as the
        # common description.
        up = cfg["user_prompt"]
        m = re.search(r"\*\*Your Data:\*\*\s*\n(.*?)\n\n", up, flags=re.S)
        shard = m.group(1).strip() if m else up
        agents.append(AgentSpec(agent_id=f"agent_{cfg['agent_id']}", shard_text=shard, fact_ids=[]))
    desc = re.sub(r"\*\*Your Data:\*\*.*?\n\n", "", d["task_description"], flags=re.S)
    desc = desc.replace("{agent_id}", "your id").replace("{input_shard}", "(see your data)")
    desc = re.sub(r"\*\*Communication Protocol:\*\*.*", "", desc, flags=re.S).strip()
    eo = d["expected_output"]
    per_agent = eo.get("per_agent_values")
    correct = str(per_agent[0]) if per_agent else str(eo.get("value", ""))
    return Task(
        task_id=f"silo_{d['case_id']}", kind="silo", world_id=f"silo_{d['case_id']}",
        domain=d.get("case_name", "silo"), description=desc, agents=agents,
        possible_answers=[], correct_answer=correct, facts={},
        meta={"silo_expected_output": eo, "paradigm": d.get("paradigm"),
              "per_agent_values": per_agent, "source_path": os.path.abspath(path)},
    )


def hiddenbench_official(path: str) -> List[Task]:
    items = json.load(open(path))
    return [from_hiddenbench(it) for it in items]


def generate_task_set(out_dir: str, n_tasks: int, seed0: int = 0, **kw) -> List[str]:
    """Generate n hidden-profile tasks to out_dir. Returns file paths.

    World ids are seed-based, so train/test splits by world are trivial:
    e.g. seeds 0..199 for probe training, 1000..1199 for evaluation.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    s = seed0
    made = 0
    while made < n_tasks:
        try:
            t = generate_task(seed=s, **kw)
        except RuntimeError:
            s += 1
            continue
        p = os.path.join(out_dir, f"{t.task_id}.json")
        t.save(p)
        paths.append(p)
        made += 1
        s += 1
    return paths
