#!/usr/bin/env python3
"""Run the pre-registered signal tests S1-S9 (protocol.md Section B).

Example (H100):
  python scripts/probe_tests.py --model Qwen/Qwen3-8B --n-worlds 80 --per-world 8 --out results/probe_qwen3_8b
Example (V100, fp16):
  python scripts/probe_tests.py --model Qwen/Qwen3-8B --dtype float16 --n-worlds 40 --out results/probe_v100
Resumable: existing <out>/<test>.npz captures are reused.
"""
import argparse, json, logging, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from listen.backends import build_backend
from listen.probe_tests.gen_items import gen_items, write_items
from listen.probe_tests.run_probe import capture_items, run_tests

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--dtype", default="bfloat16")
ap.add_argument("--out", required=True)
ap.add_argument("--n-worlds", type=int, default=80, help="worlds (task seeds); half selection, half report")
ap.add_argument("--per-world", type=int, default=8)
ap.add_argument("--seed0", type=int, default=5000, help="first world seed (keep disjoint from experiment task seeds)")
ap.add_argument("--duplication", type=int, default=2)
ap.add_argument("--tests", default="S1,S3,S4,S7,S8")
ap.add_argument("--no-thinking", action="store_true", help="Qwen3: pass enable_thinking=false")
ap.add_argument("--components", default="resid,attn,mlp")
ap.add_argument("--poolings", default="mean,last")
ap.add_argument("--skip-s9", action="store_true")
ap.add_argument("--s9-limit", type=int, default=200)
ap.add_argument("--tiny-random", action="store_true", help="test-only: random weights")
a = ap.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
os.makedirs(a.out, exist_ok=True)
items_path = os.path.join(a.out, "items.jsonl")
if os.path.exists(items_path):
    items = [json.loads(l) for l in open(items_path)]
else:
    items = gen_items(list(range(a.seed0, a.seed0 + a.n_worlds)), per_world=a.per_world,
                      duplication=a.duplication, tests=tuple(a.tests.split(",")))
    write_items(items, items_path)
print(f"{len(items)} items across {len({i['world'] for i in items})} worlds")
ctk = {"enable_thinking": False} if a.no_thinking else None
wb = build_backend({"type": "whitebox", "model": a.model, "dtype": a.dtype, "chat_template_kwargs": ctk,
                    "components": a.components.split(","), "poolings": a.poolings.split(","),
                    "tiny_random": a.tiny_random, "device_map": None if a.tiny_random else "auto"})
paths = capture_items(items, wb, a.out, tests=a.tests.split(","))
rep = run_tests(paths, os.path.join(a.out, "report"), backend=None if a.skip_s9 else wb, s9_limit=a.s9_limit)
print(open(os.path.join(a.out, "report", "report.md")).read())
