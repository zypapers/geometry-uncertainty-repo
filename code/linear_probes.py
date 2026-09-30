#!/usr/bin/env python3
"""
hidden-state linear probes.

For each of 29 layers, 6 pairwise regime comparisons, and 2 representations
(last-token primary, mean-pooled secondary), trains a logistic regression
probe with 5-fold cross-validation and 5 fixed seeds. Reports
AUROC mean across seeds and bootstrap 95% CI (1000 resamples).

Fixed hyperparameters: L2 logistic regression (C=1.0, lbfgs, max_iter=2000) on raw fp32
last-token states; 5-fold cross-validation x 5 seeds (1234, 4567, 7890, 2345, 5678);
bootstrap 95% CI (1000 resamples); shuffled-label control seeds 9001-9005.

Modes:
  --dry-run           Print plan, exit. Stdlib only.
  --synthetic-smoke   Random data, run end-to-end, save to results/_smoke/.
                      Requires sklearn.
  --execute           REAL RUN against hidden_states/hidden_states_qwen7b.npz.
                      Requires sklearn.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

# Stdlib + numpy only at import time. sklearn is lazy-imported.


# ---------- fixed constants (must match surface_baselines.py) ----------

SEEDS = (1234, 4567, 7890, 2345, 5678)
N_BOOTSTRAP = 1000
N_CV_FOLDS = 5
REGIMES = ("known", "unknown", "ambiguous", "misleading")
PAIRS = [
    ("known", "unknown"),
    ("known", "ambiguous"),
    ("known", "misleading"),
    ("unknown", "ambiguous"),
    ("unknown", "misleading"),
    ("ambiguous", "misleading"),
]
PRIMARY_REPRESENTATION = "last_token_states"
SECONDARY_REPRESENTATION = "mean_pooled_states"
REPRESENTATIONS = (PRIMARY_REPRESENTATION, SECONDARY_REPRESENTATION)
# Logistic regression hyperparameters (fixed).
# Note: penalty is L2 by default in sklearn; explicit `penalty='l2'` was
# deprecated in sklearn 1.8, so we leave it at default and document the
# intent here rather than passing it explicitly.
LR_C = 1.0
LR_PENALTY_DOC = "l2 (sklearn default)"
LR_SOLVER = "lbfgs"
LR_MAX_ITER = 2000


# ---------- Core evaluation ----------

def pair_mask(regimes, r1, r2):
    return (regimes == r1) | (regimes == r2)


def auroc(y, p):
    from sklearn.metrics import roc_auc_score
    if len(set(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def bootstrap_auroc_ci(y, p, seed, n_boot=N_BOOTSTRAP):
    import numpy as np
    rng = np.random.default_rng(seed * 1_000_003)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        ys = y[idx]
        ps = p[idx]
        if len(set(ys)) < 2:
            continue
        aucs.append(auroc(ys, ps))
    aucs = np.array(aucs, dtype=np.float64)
    if len(aucs) < 2:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def evaluate_probe_one_seed(X_pair, y, seed):
    """5-fold CV LR; returns out-of-fold probs of length n_pair."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    import numpy as np

    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=np.float64)
    for train_idx, test_idx in cv.split(X_pair, y):
        # Default penalty='l2' (don't pass explicitly — deprecated in sklearn 1.8).
        lr = LogisticRegression(
            C=LR_C, solver=LR_SOLVER,
            max_iter=LR_MAX_ITER, random_state=seed,
        ).fit(X_pair[train_idx], y[train_idx])
        oof[test_idx] = lr.predict_proba(X_pair[test_idx])[:, 1]
    return oof


def evaluate_probe_multiseed(X_pair, y):
    """Returns dict with mean point + per-seed + averaged CI."""
    import numpy as np
    points = []
    cis = []
    for seed in SEEDS:
        oof = evaluate_probe_one_seed(X_pair, y, seed)
        points.append(auroc(y, oof))
        cis.append(bootstrap_auroc_ci(y, oof, seed))
    points = np.array(points)
    lo = np.array([c[0] for c in cis])
    hi = np.array([c[1] for c in cis])
    return {
        "auroc_mean_across_seeds": float(np.nanmean(points)),
        "auroc_per_seed": [float(x) for x in points],
        "auroc_ci_low_across_seeds": float(np.nanmean(lo)),
        "auroc_ci_high_across_seeds": float(np.nanmean(hi)),
    }


# ---------- Orchestration ----------

def run_pipeline(npz, regimes, outpath: Path, log_each_layer=True):
    import numpy as np
    n_prompts, num_layers, hidden_dim = npz[PRIMARY_REPRESENTATION].shape
    print(f"n_prompts={n_prompts}  num_layers={num_layers}  hidden_dim={hidden_dim}")
    results = {"per_pair_per_layer": {}, "_meta": {}}

    for r1, r2 in PAIRS:
        pair_key = f"{r1}_vs_{r2}"
        mask = pair_mask(regimes, r1, r2)
        y = (regimes[mask] == r1).astype(np.int64)
        results["per_pair_per_layer"][pair_key] = {}
        for rep in REPRESENTATIONS:
            H_all = npz[rep]  # (n_prompts, num_layers, hidden_dim)
            H_pair = H_all[mask]  # (n_pair, num_layers, hidden_dim)
            results["per_pair_per_layer"][pair_key][rep] = {}
            for L in range(num_layers):
                X_pair = H_pair[:, L, :]  # (n_pair, hidden_dim)
                res = evaluate_probe_multiseed(X_pair, y)
                results["per_pair_per_layer"][pair_key][rep][str(L)] = res
            if log_each_layer:
                aurocs = [
                    results["per_pair_per_layer"][pair_key][rep][str(L)]["auroc_mean_across_seeds"]
                    for L in range(num_layers)
                ]
                peak_L = int(np.argmax(aurocs))
                print(f"  {pair_key:30s} {rep:18s}  peak_AUROC = {aurocs[peak_L]:.4f} @ L{peak_L}")

    results["_meta"] = {
        "seeds": list(SEEDS),
        "n_cv_folds": N_CV_FOLDS,
        "n_bootstrap": N_BOOTSTRAP,
        "n_layers": int(num_layers),
        "hidden_dim": int(hidden_dim),
        "pairs": [f"{a}_vs_{b}" for a, b in PAIRS],
        "primary_representation": PRIMARY_REPRESENTATION,
        "secondary_representation": SECONDARY_REPRESENTATION,
        "lr_config": {
            "C": LR_C, "penalty": LR_PENALTY_DOC, "solver": LR_SOLVER,
            "max_iter": LR_MAX_ITER, "preprocessing": "none (raw fp32 hidden states)",
        },
        "protocol": "fixed hyperparameters; see code/common.py",
    }

    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved: {outpath}")


# ---------- Shuffled-regime control (shuffled-label control) ----------

def run_shuffled_controls(npz, regimes, shuffle_seeds, outpath: Path):
    """Run the probe pipeline with regime labels permuted per shuffle seed.

    Outputs an aggregate JSON nested per shuffle seed:

        {
          "per_shuffle_seed": {"9001": <real-shape-probe-json>, ...},
          "_meta": {"shuffle_seeds": [...], ...}
        }

    Same shuffled labels as the surface-baselines shuffle (provided that
    script uses the same `np.random.default_rng(seed).permutation` on the
    same input regimes array). This matches the shuffled-label control scoring.
    """
    import numpy as np
    aggregate = {"per_shuffle_seed": {}, "_meta": {
        "shuffle_seeds": list(shuffle_seeds),
        "kind": "shuffled_regime_control_linear_probes",
        "shuffled_control": "label permutation with the same pipeline",
        "shuffle_algorithm": "np.random.default_rng(seed).permutation(regimes)",
    }}
    for seed in shuffle_seeds:
        rng = np.random.default_rng(seed)
        shuffled_regimes = rng.permutation(regimes)
        print(f"\n[shuffle seed {seed}] running probes with permuted labels")
        scratch = outpath.parent / f".scratch_seed{seed}_{outpath.name}"
        run_pipeline(npz, shuffled_regimes, scratch, log_each_layer=False)
        aggregate["per_shuffle_seed"][str(seed)] = json.loads(scratch.read_text())
        scratch.unlink()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved shuffled-controls aggregate: {outpath}")


# ---------- Data loading ----------

def load_real_data(npz_path: Path):
    import numpy as np
    npz = np.load(npz_path, allow_pickle=True)
    expected_arrays = {"regimes", "last_token_states", "mean_pooled_states"}
    missing = expected_arrays - set(npz.files)
    if missing:
        raise SystemExit(f"npz missing required arrays: {sorted(missing)}")
    regimes = np.asarray(npz["regimes"]).astype(str)
    return npz, regimes


def generate_synthetic_data(n_per_regime=20, num_layers=5, hidden_dim=16, rng_seed=0):
    """Random hidden states (small dims) for --synthetic-smoke code-path verification.

    Defaults are intentionally small (5 layers, 16-dim) so the smoke completes
    in ~30 sec rather than ~15 min — it verifies the (layer × pair × rep × seed
    × CV × bootstrap) plumbing without simulating real probes. The real run
    uses the full 29 × 3584 from hidden_states/hidden_states_qwen7b.npz.
    """
    import numpy as np
    rng = np.random.default_rng(rng_seed)
    n = n_per_regime * len(REGIMES)
    regimes = np.array(sum([[r] * n_per_regime for r in REGIMES], []))
    npz_like = {
        "regimes": regimes,
        "last_token_states": rng.standard_normal((n, num_layers, hidden_dim)).astype(np.float32),
        "mean_pooled_states": rng.standard_normal((n, num_layers, hidden_dim)).astype(np.float32),
    }
    return npz_like, regimes


# ---------- Utility ----------

def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_id() -> str:
    return "anonymized"


def print_plan():
    print("=" * 72)
    print("hidden-state linear probes — execution plan")
    print("=" * 72)
    print(f"Seeds: {SEEDS}")
    print(f"CV folds: {N_CV_FOLDS}  (StratifiedKFold, shuffle=True, random_state=seed)")
    print(f"Bootstrap resamples per AUROC point: {N_BOOTSTRAP}")
    print(f"LR: C={LR_C}, penalty={LR_PENALTY_DOC}, solver={LR_SOLVER}, max_iter={LR_MAX_ITER}")
    print()
    print(f"Pairs (6):  " + ", ".join(f"{a}↔{b}" for a, b in PAIRS))
    print(f"Reps (2):   primary={PRIMARY_REPRESENTATION}, secondary={SECONDARY_REPRESENTATION}")
    print(f"Layers:     all 29 (L0=embedding output, L1-L28=transformer block outputs)")
    print()
    print("Output: results/probe_auroc_qwen7b.json")
    print(
        "\nThis is the plan. Pass --synthetic-smoke or --execute to actually run.\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print plan and exit. No imports beyond stdlib.",
    )
    parser.add_argument(
        "--synthetic-smoke", action="store_true",
        help="Random hidden states (small hidden_dim), run end-to-end, save to results/_smoke/.",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="REAL RUN against hidden_states/hidden_states_qwen7b.npz.",
    )
    parser.add_argument(
        "--npz",
        default="hidden_states/hidden_states_qwen7b.npz",
        help="extracted hidden-states NPZ.",
    )
    parser.add_argument(
        "--outdir",
        default="results",
        help="Output directory for probe_auroc_qwen7b.json (and _smoke/ for smoke).",
    )
    parser.add_argument(
        "--output-name",
        default="probe_auroc_qwen7b",
        help="Base name (without .json) for the output file. Default "
             "'probe_auroc_qwen7b'. For other models, override "
             "(e.g. --output-name probe_auroc_llama8b).",
    )
    parser.add_argument(
        "--shuffle-seeds", default=None,
        help="Optional comma-separated shuffle seeds "
             "(e.g. '9001,9002,9003,9004,9005'). When provided in --execute "
             "mode, also runs the full pipeline with regime labels permuted "
             "independently per seed and saves an aggregated shuffled-controls "
             "JSON at <outdir>/<output-name>_shuffled.json. Does NOT affect "
             "the main real-label run.",
    )
    args = parser.parse_args()

    modes = sum(int(x) for x in (args.dry_run, args.synthetic_smoke, args.execute))
    if modes == 0:
        print_plan()
        return
    if modes > 1:
        raise SystemExit("Pick at most one of --dry-run / --synthetic-smoke / --execute.")

    if args.dry_run:
        print_plan()
        return

    if args.synthetic_smoke:
        print(">>> SYNTHETIC SMOKE: fake hidden states with small dims (5 layers, 16-dim) for fast code-path verification. No real artifacts touched.")
        npz, regimes = generate_synthetic_data()
        outpath = Path(args.outdir) / "_smoke" / f"{args.output_name}.json"
        run_pipeline(npz, regimes, outpath)
        if args.shuffle_seeds:
            shuffle_seeds = [int(s.strip()) for s in args.shuffle_seeds.split(",")]
            print(f">>> SYNTHETIC SMOKE: also exercising shuffled-control path with seeds {shuffle_seeds}")
            shuffled_outpath = Path(args.outdir) / "_smoke" / f"{args.output_name}_shuffled.json"
            run_shuffled_controls(npz, regimes, shuffle_seeds, shuffled_outpath)
        print("\nSmoke complete. Verify the JSON shape.")
        return

    # --execute: real run.
    print(">>> EXECUTE: real-data run.")
    npz_path = Path(args.npz)
    npz, regimes = load_real_data(npz_path)
    outpath = Path(args.outdir) / f"{args.output_name}.json"
    run_pipeline(npz, regimes, outpath)

    info = {
        "git_commit_at_run": "anonymized",
        "npz_sha256": sha256_of_file(npz_path),
        "script_sha256": sha256_of_file(Path(__file__)),
        "host": "anonymized",
        "python_version": sys.version.split()[0],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    saved = json.loads(outpath.read_text())
    saved["_provenance"] = info
    outpath.write_text(json.dumps(saved, indent=2) + "\n", encoding="utf-8")
    print(f"Provenance appended: git={info['git_commit_at_run']}")

    # Shuffled-label control. Opt-in via --shuffle-seeds.
    if args.shuffle_seeds:
        shuffle_seeds = [int(s.strip()) for s in args.shuffle_seeds.split(",")]
        print(f"\n>>> shuffled-label control: {len(shuffle_seeds)} seeds {shuffle_seeds}")
        shuffled_outpath = Path(args.outdir) / f"{args.output_name}_shuffled.json"
        run_shuffled_controls(npz, regimes, shuffle_seeds, shuffled_outpath)
        shuffled = json.loads(shuffled_outpath.read_text())
        shuffled["_provenance"] = {**info, "shuffle_seeds": shuffle_seeds}
        shuffled_outpath.write_text(json.dumps(shuffled, indent=2) + "\n", encoding="utf-8")
        print(f"Shuffled-control provenance appended: {shuffled_outpath}")


if __name__ == "__main__":
    main()
