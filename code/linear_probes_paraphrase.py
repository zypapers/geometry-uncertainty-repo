#!/usr/bin/env python3
"""Linear probes on the paraphrase extension (group-aware cross-validation).

  - Loads hidden states from hidden_states/hidden_states_paraphrases_<model>.npz
  - Joins `paraphrase_group_id` from data/paraphrases.csv on `id` (matching the npz row order)
  - Uses StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed) with
    groups = paraphrase_group_id and y = (regime == r1) per pair, so every paraphrase
    of a base prompt is held out together
  - The shuffled-label control permutes labels at the GROUP level: permute the
    group -> label map, then propagate each group's permuted label to its rows

Fixed hyperparameters: L2 logistic regression (C=1.0, lbfgs, max_iter=2000) on raw fp32
features; 5-fold group-aware cross-validation x 5 seeds (1234, 4567, 7890, 2345, 5678);
bootstrap 95% CI (1000 resamples); group-level shuffled-label control seeds 9001-9005.

Usage:
  python3 code/linear_probes_paraphrase.py --execute --npz hidden_states/hidden_states_paraphrases_qwen7b.npz --output-name probe_auroc_v3paraphrases_qwen7b
"""


from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = (1234, 4567, 7890, 2345, 5678)
SHUFFLE_SEEDS = (9001, 9002, 9003, 9004, 9005)
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
LR_C = 1.0
LR_SOLVER = "lbfgs"
LR_MAX_ITER = 2000


def pair_mask(regimes, r1, r2):
    return (regimes == r1) | (regimes == r2)


def auroc(y, p):
    from sklearn.metrics import roc_auc_score
    if len(set(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def bootstrap_auroc_ci(y, p, seed, n_boot=N_BOOTSTRAP):
    rng = np.random.default_rng(seed * 1_000_003)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        ys, ps = y[idx], p[idx]
        if len(set(ys)) < 2:
            continue
        aucs.append(auroc(ys, ps))
    aucs = np.array(aucs, dtype=np.float64)
    if len(aucs) < 2:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def evaluate_probe_one_seed(X_pair, y, groups_pair, seed):
    """5-fold StratifiedGroupKFold LR; returns out-of-fold probs."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold

    cv = StratifiedGroupKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=np.float64)
    for train_idx, test_idx in cv.split(X_pair, y, groups=groups_pair):
        lr = LogisticRegression(
            C=LR_C, solver=LR_SOLVER,
            max_iter=LR_MAX_ITER, random_state=seed,
        ).fit(X_pair[train_idx], y[train_idx])
        oof[test_idx] = lr.predict_proba(X_pair[test_idx])[:, 1]
    return oof


def evaluate_probe_multiseed(X_pair, y, groups_pair):
    points, cis = [], []
    for seed in SEEDS:
        oof = evaluate_probe_one_seed(X_pair, y, groups_pair, seed)
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


def run_pipeline(npz, regimes, groups, outpath: Path, log_each_layer=True):
    n_prompts, num_layers, hidden_dim = npz[PRIMARY_REPRESENTATION].shape
    print(f"n_prompts={n_prompts}  num_layers={num_layers}  hidden_dim={hidden_dim}")
    results = {"per_pair_per_layer": {}, "_meta": {}}

    for r1, r2 in PAIRS:
        pair_key = f"{r1}_vs_{r2}"
        mask = pair_mask(regimes, r1, r2)
        y = (regimes[mask] == r1).astype(np.int64)
        g = groups[mask]
        n_groups = len(set(g))
        results["per_pair_per_layer"][pair_key] = {}
        print(f"\n  {pair_key}  n_rows={mask.sum()}  n_groups={n_groups}")
        for rep in REPRESENTATIONS:
            H_all = npz[rep]
            H_pair = H_all[mask]
            results["per_pair_per_layer"][pair_key][rep] = {}
            for L in range(num_layers):
                X_pair = H_pair[:, L, :]
                res = evaluate_probe_multiseed(X_pair, y, g)
                results["per_pair_per_layer"][pair_key][rep][str(L)] = res
            if log_each_layer:
                aurocs = [
                    results["per_pair_per_layer"][pair_key][rep][str(L)]["auroc_mean_across_seeds"]
                    for L in range(num_layers)
                ]
                peak_L = int(np.argmax(aurocs))
                print(f"    {rep:18s}  peak_AUROC = {aurocs[peak_L]:.4f} @ L{peak_L}")

    results["_meta"] = {
        "seeds": list(SEEDS),
        "n_cv_folds": N_CV_FOLDS,
        "n_bootstrap": N_BOOTSTRAP,
        "n_layers": int(num_layers),
        "hidden_dim": int(hidden_dim),
        "pairs": [f"{a}_vs_{b}" for a, b in PAIRS],
        "primary_representation": PRIMARY_REPRESENTATION,
        "secondary_representation": SECONDARY_REPRESENTATION,
        "cv": "StratifiedGroupKFold(groups=paraphrase_group_id, y=regime)",
        "lr_config": {
            "C": LR_C, "penalty": "l2 (sklearn default)", "solver": LR_SOLVER,
            "max_iter": LR_MAX_ITER, "preprocessing": "none (raw fp32 hidden states)",
        },
    }

    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved: {outpath}")


def group_level_shuffle(regimes, groups, seed):
    """Permute labels at the GROUP level .

    Algorithm:
      1. Build group→regime mapping (each group has a single regime since all
         rows in a group share regime; assertion enforces this).
      2. Permute the regime VALUES across groups (keys stay fixed).
      3. Propagate: each row's shuffled regime = permuted_map[row.group].
    """
    unique_groups = pd.Series(groups).unique()
    # Build group → regime
    group_to_regime = {}
    for g in unique_groups:
        labels_in_group = set(regimes[groups == g])
        if len(labels_in_group) != 1:
            raise SystemExit(f"Group {g} has multiple regimes: {labels_in_group}")
        group_to_regime[g] = next(iter(labels_in_group))
    # Permute regime values across groups (preserves per-regime group count)
    regime_values = np.array([group_to_regime[g] for g in unique_groups])
    rng = np.random.default_rng(seed)
    permuted = rng.permutation(regime_values)
    permuted_map = dict(zip(unique_groups, permuted))
    # Propagate back to row level
    shuffled = np.array([permuted_map[g] for g in groups])
    return shuffled


def run_shuffled_controls(npz, regimes, groups, shuffle_seeds, outpath: Path):
    aggregate = {"per_shuffle_seed": {}, "_meta": {
        "shuffle_seeds": list(shuffle_seeds),
        "kind": "group_level_shuffled_regime_control",
        "shuffle_algorithm": "group→regime map → permute values → propagate to rows",
    }}
    for seed in shuffle_seeds:
        print(f"\n[shuffle seed {seed}] group-level permutation")
        shuffled_regimes = group_level_shuffle(regimes, groups, seed)
        scratch = outpath.parent / f".scratch_seed{seed}_{outpath.name}"
        run_pipeline(npz, shuffled_regimes, groups, scratch, log_each_layer=False)
        aggregate["per_shuffle_seed"][str(seed)] = json.loads(scratch.read_text())
        scratch.unlink()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved shuffled-controls aggregate: {outpath}")


def load_data(npz_path: Path, with_paraphrases_csv: Path):
    npz = np.load(npz_path, allow_pickle=True)
    needed = {"prompt_ids", "regimes", "last_token_states", "mean_pooled_states"}
    missing = needed - set(npz.files)
    if missing:
        raise SystemExit(f"npz missing required arrays: {sorted(missing)}")
    prompt_ids = np.asarray(npz["prompt_ids"]).astype(str)
    regimes = np.asarray(npz["regimes"]).astype(str)

    # Join groups by id
    df = pd.read_csv(with_paraphrases_csv)
    id_to_group = dict(zip(df["id"], df["paraphrase_group_id"]))
    missing_ids = [pid for pid in prompt_ids if pid not in id_to_group]
    if missing_ids:
        raise SystemExit(f"prompt_ids not in csv: {missing_ids[:5]}...")
    groups = np.array([id_to_group[pid] for pid in prompt_ids])
    return npz, regimes, groups


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def current_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--npz", required=True, help="Path to hidden_states/hidden_states_paraphrases_<model>.npz")
    parser.add_argument("--with-paraphrases", default="data/paraphrases.csv")
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--output-name", required=True, help="e.g. probe_auroc_v3paraphrases_qwen7b")
    parser.add_argument("--shuffle-seeds", default=",".join(str(s) for s in SHUFFLE_SEEDS),
                        help="Comma-separated. Default = 9001..9005.")
    parser.add_argument("--skip-shuffles", action="store_true",
                        help="If set, skip the group-level shuffled-label control pass.")
    args = parser.parse_args()

    if args.dry_run:
        print("Paraphrase-set probes plan:")
        print(f"  npz:                   {args.npz}")
        print(f"  with_paraphrases_csv:  {args.with_paraphrases}")
        print(f"  LR-CV seeds:           {SEEDS}")
        print(f"  Shuffle seeds:         {args.shuffle_seeds}")
        print(f"  CV:                    StratifiedGroupKFold(n_splits=5, shuffle=True)")
        print(f"  Output:                {args.outdir}/{args.output_name}.json")
        return

    if not args.execute:
        raise SystemExit("Pass --execute (or --dry-run).")

    npz_path = Path(args.npz)
    csv_path = Path(args.with_paraphrases)
    npz, regimes, groups = load_data(npz_path, csv_path)
    print(f"Loaded {npz_path}: n={len(regimes)}, groups={len(set(groups))}")
    print(f"  regime counts: {dict(pd.Series(regimes).value_counts())}")

    outpath = Path(args.outdir) / f"{args.output_name}.json"
    run_pipeline(npz, regimes, groups, outpath)

    info = {
        "git_commit_at_run": current_git_commit(),
        "npz_sha256": sha256_of_file(npz_path),
        "csv_sha256": sha256_of_file(csv_path),
        "script_sha256": sha256_of_file(Path(__file__)),
        "host": socket.gethostname(),
        "python_version": sys.version.split()[0],
    }
    saved = json.loads(outpath.read_text())
    saved["_provenance"] = info
    outpath.write_text(json.dumps(saved, indent=2) + "\n", encoding="utf-8")
    print(f"Provenance appended: git={info['git_commit_at_run']}")

    if not args.skip_shuffles:
        shuffle_seeds = [int(s.strip()) for s in args.shuffle_seeds.split(",") if s.strip()]
        print(f"\n>>> Group-level shuffled-regime control: {len(shuffle_seeds)} seeds {shuffle_seeds}")
        sh_outpath = Path(args.outdir) / f"{args.output_name}_shuffled.json"
        run_shuffled_controls(npz, regimes, groups, shuffle_seeds, sh_outpath)
        shuffled = json.loads(sh_outpath.read_text())
        shuffled["_provenance"] = {**info, "shuffle_seeds": shuffle_seeds}
        sh_outpath.write_text(json.dumps(shuffled, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
