#!/usr/bin/env python3
"""Calibrate per-reader standardisation and CUSUM threshold on honest runs."""
import argparse, glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from listen.orchestrator.core import DimScorer, MahalanobisScorer, ProbeScorer, calibrate

ap = argparse.ArgumentParser()
ap.add_argument("--honest-glob", required=True, help="e.g. runs/main/honest/*")
ap.add_argument("--scorer", choices=["probe", "mahalanobis", "dim"], default="probe")
ap.add_argument("--scorer-path", default=None)
ap.add_argument("--n-rounds", type=int, default=10)
ap.add_argument("--target-fpr", type=float, default=0.05)
ap.add_argument("--quorum", type=int, default=2)
ap.add_argument("--n-min", type=int, default=3)
ap.add_argument("--out", required=True)
a = ap.parse_args()
sc = {"probe": lambda: ProbeScorer(a.scorer_path), "mahalanobis": lambda: MahalanobisScorer(path=a.scorer_path),
      "dim": lambda: DimScorer()}[a.scorer]()
cal = calibrate(sorted(glob.glob(a.honest_glob)), sc, a.n_rounds, a.target_fpr, quorum=a.quorum, n_min=a.n_min)
cal.save(a.out)
print(f"drift={cal.drift:.3f} threshold={cal.threshold:.3f} readers={list(cal.reader_mu)} notes={cal.notes}")
