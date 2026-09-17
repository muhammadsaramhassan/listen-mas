#!/usr/bin/env python3
"""Fit a linear conflict probe from already-captured MAS activations (CPU, no GPU).

Positive label = the ingested message is a ground-truth false claim (from the
run's labels.jsonl); negative = any message from an honest sender. Sweeps layers
by task-grouped CV AUROC on one component/pooling, refits the best on all data,
and writes best_probe.npz with the keys ProbeScorer expects (w, b, mu, sd,
component, pooling, layer).

NOTE: labels here are sender-truth (false-claim vs honest), so this probe can
ride the sender/lexical confound the Sept-12 audit flagged. It is a real
end-to-end detection number, not the confound-controlled S4 result. Use it to
unblock the orchestrator, not as the paper's headline.
"""
import argparse, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from listen.capture.store import ActivationStore
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score


def load_dataset(run_dirs, component, pooling):
    """Return X[n, L, D], y[n], groups[n] (task_id). Loads every layer once."""
    Xs, ys, groups = [], [], []
    for d in run_dirs:
        false_ids = set()
        lp = os.path.join(d, "labels.jsonl")
        if os.path.exists(lp):
            for line in open(lp):
                r = json.loads(line)
                if r.get("false_claim"):
                    false_ids.add(r["msg_id"])
        store = ActivationStore(d)
        for rec in store.iter_index():
            arr = store.load(rec).get(f"{component}_{pooling}")
            if arr is None:
                continue
            Xs.append(np.nan_to_num(arr.astype(np.float32)))          # [L, D]
            ys.append(1 if rec["msg_id"] in false_ids else 0)
            groups.append(rec.get("task_id", d))
    if not Xs:
        sys.exit("No captures found. Check the globs and that activations/ exist.")
    return np.stack(Xs), np.array(ys), np.array(groups)


def cv_auroc(X, y, groups, C):
    n_groups = len(set(groups))
    k = min(5, n_groups)
    if k < 2 or len(set(y)) < 2:
        return float("nan")
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
        if len(set(y[tr])) < 2:
            oof[te] = y[tr].mean()
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        lr = LogisticRegression(C=C, max_iter=2000).fit((X[tr] - mu) / sd, y[tr])
        oof[te] = lr.predict_proba((X[te] - mu) / sd)[:, 1]
    return roc_auc_score(y, oof)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--honest-glob", required=True)
    ap.add_argument("--attacked-glob", required=True)
    ap.add_argument("--component", default="resid")
    ap.add_argument("--pooling", default="mean")
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dirs = sorted(glob.glob(a.honest_glob)) + sorted(glob.glob(a.attacked_glob))
    X, y, groups = load_dataset(dirs, a.component, a.pooling)
    L = X.shape[1]
    print(f"{len(y)} captures  positives={int(y.sum())}  tasks={len(set(groups))}  layers={L}  D={X.shape[2]}")

    scores = [(cv_auroc(X[:, l], y, groups, a.C), l) for l in range(L)]
    for auc, l in scores:
        print(f"  layer {l:2d}  cv_auroc={auc:.3f}")
    best_auc, best_l = max((s for s in scores if not np.isnan(s[0])), default=(float("nan"), 0))
    print(f"BEST layer {best_l}  cv_auroc={best_auc:.3f}")

    Xl = X[:, best_l]
    mu, sd = Xl.mean(0), Xl.std(0) + 1e-6
    lr = LogisticRegression(C=a.C, max_iter=2000).fit((Xl - mu) / sd, y)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(a.out, w=lr.coef_[0].astype(np.float32), b=np.float32(lr.intercept_[0]),
             mu=mu.astype(np.float32), sd=sd.astype(np.float32),
             component=a.component, pooling=a.pooling, layer=best_l)
    print(f"wrote {a.out}  (component={a.component} pooling={a.pooling} layer={best_l})")


def selfcheck():
    rng = np.random.default_rng(0)
    n, L, D = 200, 3, 8
    y = np.r_[np.zeros(n // 2), np.ones(n // 2)].astype(int)
    X = rng.normal(size=(n, L, D)); X[:, 1, 0] += y * 4.0            # signal only at layer 1
    g = np.arange(n) % 10
    aucs = [cv_auroc(X[:, l], y, g, 0.1) for l in range(L)]
    assert aucs[1] > 0.9 and int(np.nanargmax(aucs)) == 1, aucs
    print("selfcheck ok", [round(x, 2) for x in aucs])


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        selfcheck()
    else:
        main()
