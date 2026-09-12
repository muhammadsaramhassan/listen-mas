#!/usr/bin/env python3
"""Generate hidden-profile task sets (train / eval worlds are disjoint by seed)."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from listen.tasks.convert import generate_task_set

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--seed0", type=int, default=0)
ap.add_argument("--n-agents", type=int, default=6)
ap.add_argument("--duplication", type=int, default=2)
ap.add_argument("--shared-fraction", type=float, default=0.4)
ap.add_argument("--individual-misleading", action="store_true")
a = ap.parse_args()
paths = generate_task_set(a.out, a.n, seed0=a.seed0, n_agents=a.n_agents, duplication=a.duplication,
                          shared_fraction=a.shared_fraction, individual_misleading=a.individual_misleading)
print(f"wrote {len(paths)} tasks to {a.out}")
