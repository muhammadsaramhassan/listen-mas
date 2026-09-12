#!/usr/bin/env python3
"""Message ablation over the white-box readers of finished runs."""
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from listen.backends import build_backend
from listen.ablate import ablate_run

ap = argparse.ArgumentParser()
ap.add_argument("--runs-glob", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--dtype", default="bfloat16")
ap.add_argument("--no-thinking", action="store_true")
a = ap.parse_args()
wb = build_backend({"type": "whitebox", "model": a.model, "dtype": a.dtype,
                    "chat_template_kwargs": {"enable_thinking": False} if a.no_thinking else None})
for d in sorted(glob.glob(a.runs_glob)):
    res = ablate_run(d, wb)
    print(d, len(res), "ablations")
