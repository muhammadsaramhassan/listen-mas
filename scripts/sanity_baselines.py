#!/usr/bin/env python3
"""Sanity baselines for a task set: single agent with (a) its own shard only,
(b) the union of all shards. (a) must be ~chance, (b) ~ceiling.
Works with any backend (hf_blackbox / whitebox / api)."""
import argparse, glob, json, os, sys, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from listen.backends import build_backend
from listen.tasks.schema import Task
from listen.engine.protocol import normalise_answer

ap = argparse.ArgumentParser()
ap.add_argument("--tasks-dir", required=True)
ap.add_argument("--backend", default="hf_blackbox", choices=["hf_blackbox", "whitebox", "api"])
ap.add_argument("--model", required=True)
ap.add_argument("--dtype", default="bfloat16")
ap.add_argument("--base-url", default=None)
ap.add_argument("--no-thinking", action="store_true")
ap.add_argument("--limit", type=int, default=50)
ap.add_argument("--agents-per-task", type=int, default=2, help="how many agents to test in own-shard mode")
a = ap.parse_args()
cfg = {"type": a.backend, "model": a.model, "dtype": a.dtype, "base_url": a.base_url,
       "chat_template_kwargs": {"enable_thinking": False} if a.no_thinking else None}
be = build_backend(cfg)
tasks = [Task.load(p) for p in sorted(glob.glob(os.path.join(a.tasks_dir, "*.json")))][: a.limit]
def ask(desc, info, options):
    sys_p = f"{desc}\n\nYOUR INFORMATION\n{info}\n\nAnswer with exactly one option letter from: {', '.join(options)}. Reply as <answer>X</answer>."
    out = be.generate(sys_p, [{"role": "user", "content": "Which option is best? Give your answer now."}], max_new_tokens=300).text
    m = re.search(r"<answer>(.*?)</answer>", out, re.S)
    return normalise_answer(m.group(1) if m else out.strip()[-3:], options)
own_hit = own_n = uni_hit = 0
for t in tasks:
    opts = t.possible_answers
    for ag in t.agents[: a.agents_per_task]:
        own_hit += (ask(t.description, ag.shard_text, opts) == t.correct_answer); own_n += 1
    union = "\n".join(f"- {f.text}" for f in t.facts.values()) if t.facts else "\n\n".join(ag.shard_text for ag in t.agents)
    uni_hit += (ask(t.description, union, opts) == t.correct_answer)
print(json.dumps({"n_tasks": len(tasks), "own_shard_accuracy": own_hit / max(1, own_n),
                  "union_accuracy": uni_hit / max(1, len(tasks)), "chance": 1 / len(tasks[0].possible_answers)}, indent=1))
