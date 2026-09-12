"""Model-free orchestrator.

Pipeline per round:
  scores  x[r,s,m]   from a Scorer applied to reader r's capture of sender s's message m
  standardise        per-reader z-score using honest-run calibration (mean/std of x)
  aggregate          a[s,t] = median over readers of (mean over that reader's messages from s in round t)
  accumulate         CUSUM  S[s,t] = max(0, S[s,t-1] + a[s,t] - drift)
  eject              S > threshold  AND  >= quorum distinct readers contributed  AND  >= n_min messages observed

The threshold is calibrated so that P(any honest sender ejected within T rounds) <= target FPR
on honest runs (see `calibrate`).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..capture.store import ActivationStore


# --------------------------------------------------------------------------
# Scorers: capture record -> scalar (higher = more suspicious)
# --------------------------------------------------------------------------
class Scorer:
    name = "scorer"

    def score(self, store: ActivationStore, rec: Dict[str, Any]) -> float:
        raise NotImplementedError


class DimScorer(Scorer):
    """Test scorer: reads one dimension of one layer (matches MockBackend's planted signal)."""
    name = "dim"

    def __init__(self, component="resid", pooling="mean", layer=2, dim=0):
        self.c, self.p, self.l, self.d = component, pooling, layer, dim

    def score(self, store, rec):
        return float(store.feature(rec, self.c, self.p, self.l)[self.d])


class ProbeScorer(Scorer):
    """Linear probe: score = w . standardised(feature) + b. Loaded from an npz
    produced by probe_tests.run_probe (keys: w, b, mu, sd, component, pooling, layer)."""
    name = "probe"

    def __init__(self, path: str):
        z = np.load(path, allow_pickle=True)
        self.w, self.b = z["w"].astype(np.float32), float(z["b"])
        self.mu, self.sd = z["mu"].astype(np.float32), z["sd"].astype(np.float32)
        self.c, self.p, self.l = str(z["component"]), str(z["pooling"]), int(z["layer"])

    def score(self, store, rec):
        x = store.feature(rec, self.c, self.p, self.l)
        return float(((x - self.mu) / (self.sd + 1e-6)) @ self.w + self.b)


class MahalanobisScorer(Scorer):
    """Unsupervised: robust covariance fit on honest-message activations."""
    name = "mahalanobis"

    def __init__(self, component="resid", pooling="mean", layer=16, path: Optional[str] = None):
        self.c, self.p, self.l = component, pooling, layer
        self.mu = None
        self.P = None
        if path:
            z = np.load(path)
            self.mu, self.P = z["mu"], z["P"]
            self.c, self.p, self.l = str(z["component"]), str(z["pooling"]), int(z["layer"])

    def fit(self, X: np.ndarray, robust: bool = True, shrink: float = 0.1):
        if robust and X.shape[0] > 5 * X.shape[1]:
            from sklearn.covariance import MinCovDet
            mcd = MinCovDet(random_state=0).fit(X)
            mu, C = mcd.location_, mcd.covariance_
        else:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf().fit(X)
            mu, C = lw.location_, lw.covariance_
        C = (1 - shrink) * C + shrink * np.eye(C.shape[0]) * np.trace(C) / C.shape[0]
        self.mu, self.P = mu.astype(np.float32), np.linalg.pinv(C).astype(np.float32)
        return self

    def save(self, path: str):
        np.savez(path, mu=self.mu, P=self.P, component=self.c, pooling=self.p, layer=self.l)

    def score(self, store, rec):
        x = store.feature(rec, self.c, self.p, self.l)
        d = x - self.mu
        return float(np.sqrt(max(d @ self.P @ d, 0.0)))


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------
@dataclass
class Calibration:
    reader_mu: Dict[str, float] = field(default_factory=dict)
    reader_sd: Dict[str, float] = field(default_factory=dict)
    threshold: float = float("inf")
    drift: float = 0.0
    target_fpr: float = 0.05
    n_rounds: int = 10
    notes: Dict[str, Any] = field(default_factory=dict)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=1)

    @staticmethod
    def load(path):
        d = json.load(open(path))
        c = Calibration()
        c.__dict__.update(d)
        return c


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------
class Orchestrator:
    def __init__(self, scorer: Scorer, calib: Optional[Calibration] = None, quorum: int = 2,
                 n_min: int = 3, drift: Optional[float] = None, threshold: Optional[float] = None,
                 mode: str = "cusum", eject: bool = True, log_path: Optional[str] = None):
        self.scorer = scorer
        self.calib = calib or Calibration()
        self.quorum = quorum
        self.n_min = n_min
        self.drift = self.calib.drift if drift is None else drift
        self.threshold = self.calib.threshold if threshold is None else threshold
        self.mode = mode
        self.do_eject = eject
        self.S: Dict[str, float] = {}
        self.cum: Dict[str, float] = {}
        self.readers_seen: Dict[str, set] = {}
        self.n_obs: Dict[str, int] = {}
        self.history: List[Dict[str, Any]] = []
        self.ejected: set = set()
        self.log_path = log_path

    def _z(self, reader: str, x: float) -> float:
        mu = self.calib.reader_mu.get(reader, 0.0)
        sd = self.calib.reader_sd.get(reader, 1.0)
        return (x - mu) / (sd + 1e-6)

    def on_round_end(self, ctx) -> List[str]:
        # gather per (sender, reader) standardised scores for this round
        per: Dict[str, Dict[str, List[float]]] = {}
        for rec in ctx.delivered:
            if "capture_path" not in rec or rec["reader"] not in ctx.readers:
                continue
            rec2 = {**rec, "path": os.path.relpath(rec["capture_path"], ctx.store.root)}
            x = self.scorer.score(ctx.store, rec2)
            z = self._z(rec["reader"], x)
            per.setdefault(rec["sender"], {}).setdefault(rec["reader"], []).append(z)
        out = []
        for s in ctx.active:
            if s in self.ejected:
                continue
            rd = per.get(s, {})
            if rd:
                per_reader = [float(np.mean(v)) for v in rd.values()]
                a = float(np.median(per_reader))
                self.readers_seen.setdefault(s, set()).update(rd.keys())
                self.n_obs[s] = self.n_obs.get(s, 0) + sum(len(v) for v in rd.values())
            else:
                a = 0.0
            self.S[s] = max(0.0, self.S.get(s, 0.0) + a - self.drift)
            self.cum[s] = self.cum.get(s, 0.0) + a
            stat = self.S[s] if self.mode == "cusum" else self.cum[s]
            flag = (stat > self.threshold and len(self.readers_seen.get(s, ())) >= self.quorum
                    and self.n_obs.get(s, 0) >= self.n_min)
            self.history.append({"round": ctx.round_idx, "sender": s, "a": a, "S": self.S[s], "cum": self.cum[s],
                                 "n_readers": len(self.readers_seen.get(s, ())), "n_obs": self.n_obs.get(s, 0),
                                 "flag": bool(flag), "role": ctx.roles.get(s)})
            if flag:
                self.ejected.add(s)
                if self.do_eject:
                    out.append(s)
        if self.log_path:
            with open(self.log_path, "w") as f:
                json.dump(self.history, f, indent=1)
        return out


# --------------------------------------------------------------------------
# Offline: build sender statistics from finished runs, calibrate threshold
# --------------------------------------------------------------------------
def score_run(run_dir: str, scorer: Scorer, readers: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Score every stored capture in a run. Returns records with x (raw score)."""
    store = ActivationStore(run_dir)
    out = []
    for rec in store.iter_index():
        if readers and rec["reader"] not in readers:
            continue
        out.append({**rec, "x": scorer.score(store, rec)})
    return out


def sender_trajectories(scored: List[Dict[str, Any]], calib: Calibration, drift: float,
                        n_rounds: int) -> Dict[str, Dict[str, Any]]:
    """Replay the orchestrator aggregation offline for one run -> per-sender CUSUM path."""
    per_round: Dict[Tuple[str, int], Dict[str, List[float]]] = {}
    for r in scored:
        z = (r["x"] - calib.reader_mu.get(r["reader"], 0.0)) / (calib.reader_sd.get(r["reader"], 1.0) + 1e-6)
        per_round.setdefault((r["sender"], r["round"]), {}).setdefault(r["reader"], []).append(z)
    senders = sorted({r["sender"] for r in scored})
    traj = {}
    for s in senders:
        S, path, readers, n = 0.0, [], set(), 0
        for t in range(1, n_rounds + 1):
            rd = per_round.get((s, t), {})
            a = float(np.median([np.mean(v) for v in rd.values()])) if rd else 0.0
            readers.update(rd.keys())
            n += sum(len(v) for v in rd.values())
            S = max(0.0, S + a - drift)
            path.append(S)
        traj[s] = {"path": path, "max": max(path) if path else 0.0, "final": path[-1] if path else 0.0,
                   "n_readers": len(readers), "n_obs": n,
                   "role": next((r["sender_role"] for r in scored if r["sender"] == s), None)}
    return traj


def calibrate(honest_run_dirs: List[str], scorer: Scorer, n_rounds: int, target_fpr: float = 0.05,
              drift_quantile: float = 0.6, quorum: int = 2, n_min: int = 3) -> Calibration:
    """Per-reader standardisation + CUSUM threshold from honest runs.

    threshold = (1 - target_fpr) quantile of the per-sender max CUSUM over honest runs,
    restricted to senders that satisfy quorum/coverage (others could never be ejected anyway).
    """
    scored_all = [score_run(d, scorer) for d in honest_run_dirs]
    flat = [r for s in scored_all for r in s]
    calib = Calibration(target_fpr=target_fpr, n_rounds=n_rounds)
    for reader in sorted({r["reader"] for r in flat}):
        xs = np.array([r["x"] for r in flat if r["reader"] == reader], dtype=np.float64)
        calib.reader_mu[reader] = float(xs.mean())
        calib.reader_sd[reader] = float(xs.std() + 1e-6)
    # drift: quantile of standardised per-round aggregates (keeps honest CUSUM near zero)
    zs = []
    for s in scored_all:
        for r in s:
            zs.append((r["x"] - calib.reader_mu[r["reader"]]) / calib.reader_sd[r["reader"]])
    calib.drift = float(np.quantile(zs, drift_quantile)) if zs else 0.0
    maxes = []
    for s in scored_all:
        traj = sender_trajectories(s, calib, calib.drift, n_rounds)
        for v in traj.values():
            if v["n_readers"] >= quorum and v["n_obs"] >= n_min:
                maxes.append(v["max"])
    calib.threshold = float(np.quantile(maxes, 1 - target_fpr)) if maxes else float("inf")
    calib.notes = {"n_honest_runs": len(honest_run_dirs), "n_scored": len(flat), "n_eligible_senders": len(maxes)}
    return calib


# --------------------------------------------------------------------------
# Baseline: Among-Us-style per-round majority vote
# --------------------------------------------------------------------------
def vote_baseline(scored: List[Dict[str, Any]], calib: Calibration, n_rounds: int, window: int = 3,
                  vote_z: float = 1.0) -> Dict[str, Any]:
    """Each round each reader votes for its most suspicious sender (if z > vote_z);
    eject the first sender with a strict majority of readers' votes within `window` rounds."""
    readers = sorted({r["reader"] for r in scored})
    votes: Dict[int, Dict[str, set]] = {}
    for t in range(1, n_rounds + 1):
        for rd in readers:
            rows = [r for r in scored if r["round"] == t and r["reader"] == rd]
            if not rows:
                continue
            best = max(rows, key=lambda r: r["x"])
            z = (best["x"] - calib.reader_mu.get(rd, 0.0)) / (calib.reader_sd.get(rd, 1.0) + 1e-6)
            if z > vote_z:
                votes.setdefault(t, {}).setdefault(best["sender"], set()).add(rd)
    for t in range(1, n_rounds + 1):
        tally: Dict[str, set] = {}
        for tt in range(max(1, t - window + 1), t + 1):
            for s, rs in votes.get(tt, {}).items():
                tally.setdefault(s, set()).update(rs)
        for s, rs in tally.items():
            if len(rs) > len(readers) / 2:
                return {"ejected": s, "round": t}
    return {"ejected": None, "round": None}
