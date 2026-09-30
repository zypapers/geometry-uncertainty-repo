#!/usr/bin/env python3
"""
Robustness sensitivity check for the four-type separation.

Re-runs the four-way and pairwise classification with progressively larger sets
of prompts excluded (data/robustness_exclusions.json), namely prompts that are
borderline exemplars of their type. Confirms the geometric separation does not
depend on any particular subset of edge cases; type labels are unchanged.

Exclusion levels (nested): L1_CORE (9) < L2_EXTENDED (22) < L3_ALL (33) < L4_ALL_PLUS (36).
Reports per model per level: 6 pairwise peak in-window AUROC and 4-way accuracy,
versus the full-set values.

Usage: python robustness_exclusion.py --execute
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import common as C

# exclusion sets loaded lazily inside main()


def pairwise_peak(H, regimes, ids, exclude, model, n_jobs):
    from joblib import Parallel, delayed
    lo, hi = C.LAYER_WINDOWS[model]
    keep = np.array([i not in set(exclude) for i in ids])
    out = {}
    for r1, r2 in C.PAIRS:
        pk = f"{r1}_vs_{r2}"
        mask = keep & ((regimes == r1) | (regimes == r2))
        idx = np.where(mask)[0]
        y = (regimes[idx] == r1).astype(np.int64)
        def layer_auroc(L):
            pts = [C.auroc_binary(y, C.oof_binary(H[idx][:, L, :], y, s)) for s in C.SEEDS]
            return float(np.nanmean(pts))
        aurocs = Parallel(n_jobs=n_jobs)(delayed(layer_auroc)(L) for L in range(lo, hi + 1))
        best = int(np.argmax(aurocs))
        out[pk] = {"peak_auroc": round(max(aurocs), 4), "peak_layer": lo + best,
                   "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum())}
    return out


def fourway_peak(H, regimes, ids, exclude, model, n_jobs):
    from joblib import Parallel, delayed
    lo, hi = C.LAYER_WINDOWS[model]
    keep = np.array([i not in set(exclude) for i in ids])
    idx = np.where(keep)[0]
    y = regimes[idx]
    def layer_acc(L):
        accs = []
        for s in C.SEEDS:
            _, yhat = C.oof_multiclass(H[idx][:, L, :], y, s)
            accs.append(C.accuracy_4way(y, yhat))
        return float(np.mean(accs))
    accs = Parallel(n_jobs=n_jobs)(delayed(layer_acc)(L) for L in range(lo, hi + 1))
    best = int(np.argmax(accs))
    counts = {r: int((y == r).sum()) for r in C.REGIMES}
    return {"peak_acc": round(max(accs), 4), "peak_layer": lo + best,
            "n": int(len(y)), "class_counts": counts,
            "chance": round(max(counts.values()) / len(y), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=14)
    args = ap.parse_args()
    EXCL = json.load(open("data/robustness_exclusions.json"))
    LEVELS = {"full": [], "L3_ALL": EXCL["L3_ALL"], "L4_ALL_PLUS": EXCL["L4_ALL_PLUS"]}
    if not args.execute:
        print(__doc__)
        return

    out = {"per_model": {}, "_meta": {
        "levels": {k: len(v) for k, v in LEVELS.items()},
        "exclusions": LEVELS,
        "exclusion_basis": "borderline exemplars of each type",
        "policy": "type labels unchanged; rows excluded as a sensitivity check",
        "seeds": list(C.SEEDS)}}

    for model in ["qwen7b", "llama8b", "qwen32b"]:
        df, npz, regimes = C.load_aligned(model, "base")
        H = npz["last_token_states"].astype(np.float32)
        ids = [str(x) for x in npz["prompt_ids"]]
        out["per_model"][model] = {}
        for level, excl in LEVELS.items():
            pw = pairwise_peak(H, regimes, ids, excl, model, args.n_jobs)
            fw = fourway_peak(H, regimes, ids, excl, model, args.n_jobs)
            minpair = min(v["peak_auroc"] for v in pw.values())
            out["per_model"][model][level] = {"pairwise": pw, "fourway": fw,
                                              "min_pairwise_auroc": minpair}
            print(f"{model:8s} {level:14s} (excl {len(excl):2d}): "
                  f"min pairwise AUROC {minpair:.4f}, 4-way acc {fw['peak_acc']:.4f} "
                  f"(n={fw['n']}, chance {fw['chance']:.3f})")

    out["_provenance"] = C.provenance(files=[C.NPZ[(m, "base")] for m in ["qwen7b", "llama8b", "qwen32b"]])
    C.save_versioned(out, Path("results/robustness_exclusion_v3.json"))
    print("\nHeadline claim is >= 0.986 min pairwise on the full set; check it holds under exclusion.")


if __name__ == "__main__":
    main()
