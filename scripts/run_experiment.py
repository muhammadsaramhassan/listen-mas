#!/usr/bin/env python3
"""Run a MAS experiment from a YAML config. See configs/."""
import argparse, logging, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from listen.experiment import load_config, run_experiment

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--limit", type=int, default=None, help="only first N tasks")
ap.add_argument("--set", nargs="*", default=[], help="override config keys, e.g. run_name=x rounds.max=6 malice.tier=T0")
a = ap.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
cfg = load_config(a.config)
for kv in a.set:
    k, v = kv.split("=", 1)
    d = cfg
    parts = k.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    try:
        import yaml
        v = yaml.safe_load(v)
    except Exception:
        pass
    d[parts[-1]] = v
res = run_experiment(cfg, limit=a.limit)
n = len(res)
for cond in sorted({r["condition"] for r in res}):
    rs = [r for r in res if r["condition"] == cond]
    acc = sum(1 for r in rs if r["honest_majority_correct"]) / len(rs)
    print(f"{cond}: {len(rs)} runs, honest-majority accuracy {acc:.2f}, mean rounds {sum(r['rounds_used'] for r in rs)/len(rs):.1f}")
