#!/usr/bin/env python3
"""
surface-feature baselines.

Computes 5-fold cross-validated AUROC + bootstrap 95% CI for each of the
six pairwise regime classifications, using the six surface features used in the
paper, evaluated individually and jointly:

The strongest baseline per pair is the bar that hidden-state probes
must clear by ≥5 AUROC points (separation criterion). This script generates
that bar.

Fixed hyperparameters: L2 logistic regression (C=1.0, lbfgs, max_iter=2000);
5-fold cross-validation x 5 seeds (1234, 4567, 7890, 2345, 5678); bootstrap 95% CI
(1000 resamples); shuffled-label control seeds 9001-9005.

Modes:
  --dry-run           Print the plan and exit. No data load, no imports
                      beyond stdlib+argparse. Use when iterating on
                      the script setup without running anything.
  --synthetic-smoke   Generate random features+labels of the expected
                      shapes, run the full pipeline end-to-end, save
                      results to <outdir>/_smoke/. Verifies code paths
                      without touching real data. Requires sklearn.
  --execute           REAL RUN. Loads real labeled CSV + extracted
                      hidden-states NPZ, encodes sentences, fits LRs,
                      saves results/surface_baselines.json. Requires
                      sklearn + sentence_transformers.

Usage examples:
  python3 code/surface_baselines.py --dry-run
  python3 code/surface_baselines.py --synthetic-smoke
  python3 code/surface_baselines.py --execute
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Stdlib + numpy only at import time. sklearn, pandas, sentence-transformers
# are lazy-imported in functions that need them so --dry-run requires nothing.


# ---------- fixed constants ----------

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
QUESTION_FORMS = ("which", "what", "who", "when", "where", "why", "how", "other")
SENTENCE_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
STATIC_FEATURE_NAMES = (
    "word_count",
    "named_entity_count",
    "question_form",
    "prompt_token_count",
    "next_token_entropy",
)
# Features whose value depends on the CV fold (because they require a
# leakage-safe centroid computed from training Known prompts only).
FOLD_DEPENDENT_FEATURE_NAMES = (
    "sentence_embedding_cos_to_known_centroid",
)
FEATURE_NAMES = STATIC_FEATURE_NAMES + FOLD_DEPENDENT_FEATURE_NAMES + ("all_surface",)


# ---------- Feature extraction ----------

def compute_word_count(questions):
    import numpy as np
    return np.array([len(q.split()) for q in questions], dtype=np.float64).reshape(-1, 1)


def compute_named_entity_count(df):
    import numpy as np
    if "named_entity_count_heuristic" not in df.columns:
        raise SystemExit("labeled CSV missing 'named_entity_count_heuristic' column")
    return df["named_entity_count_heuristic"].astype(np.float64).to_numpy().reshape(-1, 1)


def compute_question_form_onehot(df):
    import numpy as np
    if "question_form" not in df.columns:
        raise SystemExit("labeled CSV missing 'question_form' column")
    vals = df["question_form"].astype(str).str.lower().to_numpy()
    out = np.zeros((len(vals), len(QUESTION_FORMS)), dtype=np.float64)
    for i, v in enumerate(vals):
        if v in QUESTION_FORMS:
            out[i, QUESTION_FORMS.index(v)] = 1.0
        else:
            out[i, QUESTION_FORMS.index("other")] = 1.0
    return out


def compute_prompt_token_count(npz):
    import numpy as np
    return npz["prompt_token_counts"].astype(np.float64).reshape(-1, 1)


def compute_next_token_entropy(npz):
    """Shannon entropy of softmax(next_token_logits) in nats."""
    import numpy as np
    logits = npz["next_token_logits"].astype(np.float64)  # promote from fp16
    # softmax, in numerically stable form
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    p = p / p.sum(axis=1, keepdims=True)
    # entropy
    eps = 1e-12
    h = -(p * np.log(p + eps)).sum(axis=1)
    return h.reshape(-1, 1)


def encode_sentences(questions):
    """Lazy-load all-MiniLM-L6-v2 and encode all prompts. Returns (N, 384)."""
    from sentence_transformers import SentenceTransformer  # lazy
    import numpy as np
    model = SentenceTransformer(SENTENCE_EMBED_MODEL)
    embs = model.encode(list(questions), convert_to_numpy=True, normalize_embeddings=False)
    return embs.astype(np.float64)


# ---------- Pairwise CV evaluation ----------

def pair_mask(regimes, r1, r2):
    """Boolean mask of rows belonging to either of the two regimes in the pair."""
    return (regimes == r1) | (regimes == r2)


def evaluate_static_feature(X, y, seed):
    """5-fold CV on a fold-independent feature. Returns out-of-fold prob predictions."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    import numpy as np

    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=np.float64)
    for train_idx, test_idx in cv.split(X, y):
        lr = LogisticRegression(
            random_state=seed, max_iter=2000, solver="lbfgs"
        ).fit(X[train_idx], y[train_idx])
        oof[test_idx] = lr.predict_proba(X[test_idx])[:, 1]
    return oof


def _cosine(a, b):
    """Row-wise cosine of a (N, D) array against a single (D,) vector."""
    import numpy as np
    num = (a * b).sum(axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b)
    return num / (den + 1e-12)


def _fold_sentence_cos_for_pair(
    sentence_embs_all, regimes_all, pair_indices_in_all, test_orig_indices,
):
    """Compute the fold-specific sentence_cos feature for all pair prompts.

    The Known centroid uses all Known prompts NOT in the held-out test set
    (regardless of regime in the pair). For pairs not containing Known, all Known
    prompts contribute because none are in the held-out test set.

    Returns: (n_pair, 1) array of cosine values.
    """
    import numpy as np
    known_mask_all = (regimes_all == "known")
    train_known_mask = known_mask_all.copy()
    train_known_mask[test_orig_indices] = False  # leakage-safe
    train_known_embs = sentence_embs_all[train_known_mask]
    if len(train_known_embs) == 0:
        raise RuntimeError("no training Known prompts available for centroid")
    centroid = train_known_embs.mean(axis=0)
    embs_pair = sentence_embs_all[pair_indices_in_all]
    return _cosine(embs_pair, centroid).reshape(-1, 1)


def evaluate_sentence_cos_feature(
    sentence_embs_all, regimes_all, pair_mask_arr, y, seed,
):
    """Standalone sentence-cosine-to-Known-centroid feature with leakage-safe centroid.

    Centroid recomputed per fold; only the 1-D cosine fed to the LR.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    import numpy as np

    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=seed)
    pair_indices_in_all = np.where(pair_mask_arr)[0]
    oof = np.zeros(len(y), dtype=np.float64)

    for train_idx, test_idx in cv.split(pair_indices_in_all, y):
        test_orig_indices = pair_indices_in_all[test_idx]
        feat = _fold_sentence_cos_for_pair(
            sentence_embs_all, regimes_all, pair_indices_in_all, test_orig_indices,
        )
        lr = LogisticRegression(
            random_state=seed, max_iter=2000, solver="lbfgs"
        ).fit(feat[train_idx], y[train_idx])
        oof[test_idx] = lr.predict_proba(feat[test_idx])[:, 1]
    return oof


def evaluate_all_surface_combined(
    static_features_concat_all, sentence_embs_all, regimes_all, pair_mask_arr, y, seed,
):
    """all_surface = concat(features 1-5) + sentence_cos (per-fold centroid).

    static_features_concat_all: (N_all, d_static) — concatenation of the five
        fold-independent static features for all prompts.
    sentence_embs_all: (N_all, 384) — full sentence embeddings for all prompts.

    Inside each CV fold within the pair:
      1. Compute training-fold-safe Known centroid (excluding test prompts).
      2. Compute sentence_cos for all pair prompts using that centroid.
      3. Concat static features 1-5 (pair-restricted) + sentence_cos → (n_pair, d_static + 1).
      4. Train LR on training fold, predict on test fold.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    import numpy as np

    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=seed)
    pair_indices_in_all = np.where(pair_mask_arr)[0]
    static_pair = static_features_concat_all[pair_mask_arr]  # (n_pair, d_static)
    oof = np.zeros(len(y), dtype=np.float64)

    for train_idx, test_idx in cv.split(static_pair, y):
        test_orig_indices = pair_indices_in_all[test_idx]
        cos_feat = _fold_sentence_cos_for_pair(
            sentence_embs_all, regimes_all, pair_indices_in_all, test_orig_indices,
        )  # (n_pair, 1)
        X_combined = np.concatenate([static_pair, cos_feat], axis=1)
        lr = LogisticRegression(
            random_state=seed, max_iter=2000, solver="lbfgs"
        ).fit(X_combined[train_idx], y[train_idx])
        oof[test_idx] = lr.predict_proba(X_combined[test_idx])[:, 1]
    return oof


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


def evaluate_feature_on_pair(
    feature_name, feature_data, y, regimes_all, pair_mask_arr, seed,
):
    """Dispatch on feature type. Returns (auc_mean_oof, ci_low, ci_high).

    feature_data depends on feature_name:
      static feature: (N_all, d) array — pair-masking happens inside.
      "sentence_embedding_cos_to_known_centroid": (N_all, 384) sentence embeddings.
      "all_surface": dict with keys 'static' (N_all, d_static) and 'sentence_embs' (N_all, 384).
    """
    import numpy as np
    if feature_name == "sentence_embedding_cos_to_known_centroid":
        oof = evaluate_sentence_cos_feature(
            feature_data, regimes_all, pair_mask_arr, y, seed
        )
    elif feature_name == "all_surface":
        oof = evaluate_all_surface_combined(
            feature_data["static"], feature_data["sentence_embs"],
            regimes_all, pair_mask_arr, y, seed,
        )
    else:
        X_pair = feature_data[pair_mask_arr]
        oof = evaluate_static_feature(X_pair, y, seed)
    point = auroc(y, oof)
    lo, hi = bootstrap_auroc_ci(y, oof, seed)
    return point, lo, hi


def evaluate_feature_multiseed(
    feature_name, feature_data, y, regimes_all, pair_mask_arr,
):
    """Run across all fixed seeds; report mean point estimate and CI."""
    import numpy as np
    points = []
    cis = []
    for seed in SEEDS:
        p, lo, hi = evaluate_feature_on_pair(
            feature_name, feature_data, y, regimes_all, pair_mask_arr, seed
        )
        points.append(p)
        cis.append((lo, hi))
    points = np.array(points)
    lo_arr = np.array([c[0] for c in cis])
    hi_arr = np.array([c[1] for c in cis])
    return {
        "auroc_mean_across_seeds": float(np.nanmean(points)),
        "auroc_per_seed": [float(x) for x in points],
        "auroc_ci_low_across_seeds": float(np.nanmean(lo_arr)),
        "auroc_ci_high_across_seeds": float(np.nanmean(hi_arr)),
    }


# ---------- Data loading ----------

def load_real_data(input_csv: Path, npz_path: Path):
    import numpy as np
    import pandas as pd
    df = pd.read_csv(input_csv)
    required = {"id", "regime", "question", "named_entity_count_heuristic", "question_form"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"labeled CSV missing required columns: {sorted(missing)}")
    npz = np.load(npz_path, allow_pickle=True)
    if len(df) != len(npz["prompt_ids"]):
        raise SystemExit(
            f"row count mismatch: csv={len(df)} vs npz={len(npz['prompt_ids'])}"
        )
    npz_ids = list(npz["prompt_ids"])
    csv_ids = list(df["id"])
    if csv_ids != npz_ids:
        raise SystemExit("id ordering differs between csv and npz; refusing to run")
    regimes = df["regime"].astype(str).to_numpy()
    return df, npz, regimes


def generate_synthetic_data(n_per_regime=20, embed_dim=384, vocab=152064, rng_seed=0):
    """Random data of the expected shape. Used by --synthetic-smoke."""
    import numpy as np
    rng = np.random.default_rng(rng_seed)
    n = n_per_regime * len(REGIMES)
    regimes = np.array(sum([[r] * n_per_regime for r in REGIMES], []))
    questions = [f"synthetic question {i} for regime {regimes[i]}" for i in range(n)]
    df_like = {
        "id": np.array([f"synth_{i:03d}" for i in range(n)]),
        "regime": regimes,
        "question": np.array(questions),
        "named_entity_count_heuristic": rng.integers(0, 6, size=n).astype(np.float64),
        "question_form": rng.choice(QUESTION_FORMS, size=n),
    }
    # Stand in for NPZ:
    npz_like = {
        "prompt_ids": df_like["id"],
        "prompt_token_counts": rng.integers(60, 80, size=n).astype(np.int32),
        "next_token_logits": rng.standard_normal((n, vocab)).astype(np.float16),
    }
    import pandas as pd
    return pd.DataFrame(df_like), npz_like, regimes


# ---------- Orchestration ----------

def compute_all_static_features(df, npz):
    """Returns (feats_dict, static_concat_array).

    feats_dict: maps each static feature name to its (N_all, d) array.
    static_concat_array: (N_all, sum-of-d) concat of features 1-5, used by
        the per-fold all_surface evaluator.

    Sentence embeddings are NOT in feats_dict — they are passed alongside,
    because feature 6 (sentence_cos) and the combined all_surface both
    need a fold-specific centroid.
    """
    import numpy as np
    feats = {
        "word_count": compute_word_count(df["question"].astype(str)),
        "named_entity_count": compute_named_entity_count(df),
        "question_form": compute_question_form_onehot(df),
        "prompt_token_count": compute_prompt_token_count(npz),
        "next_token_entropy": compute_next_token_entropy(npz),
    }
    static_concat = np.concatenate(
        [feats[k] for k in STATIC_FEATURE_NAMES], axis=1,
    )
    return feats, static_concat


def run_pipeline(df, npz, sentence_embs, regimes, outpath: Path):
    import numpy as np
    static_feats, static_concat = compute_all_static_features(df, npz)
    all_surface_data = {"static": static_concat, "sentence_embs": sentence_embs}
    results = {"per_pair": {}, "_meta": {}}
    for r1, r2 in PAIRS:
        pair_key = f"{r1}_vs_{r2}"
        mask = pair_mask(regimes, r1, r2)
        n_pair = int(mask.sum())
        y = (regimes[mask] == r1).astype(np.int64)
        pair_block = {"n_examples": n_pair, "features": {}}
        for fname in FEATURE_NAMES:
            if fname in STATIC_FEATURE_NAMES:
                fdata = static_feats[fname]
            elif fname == "sentence_embedding_cos_to_known_centroid":
                fdata = sentence_embs
            elif fname == "all_surface":
                fdata = all_surface_data
            else:
                raise SystemExit(f"unknown feature: {fname}")
            res = evaluate_feature_multiseed(fname, fdata, y, regimes, mask)
            pair_block["features"][fname] = res
        strongest_name = max(
            pair_block["features"].keys(),
            key=lambda k: pair_block["features"][k]["auroc_mean_across_seeds"],
        )
        pair_block["strongest_feature"] = strongest_name
        pair_block["strongest_auroc"] = pair_block["features"][strongest_name]["auroc_mean_across_seeds"]
        results["per_pair"][pair_key] = pair_block
        print(f"  {pair_key}: strongest={strongest_name} auroc={pair_block['strongest_auroc']:.4f}")
    # Meta
    results["_meta"] = {
        "seeds": list(SEEDS),
        "n_cv_folds": N_CV_FOLDS,
        "n_bootstrap": N_BOOTSTRAP,
        "feature_names": list(FEATURE_NAMES),
        "pairs": [f"{a}_vs_{b}" for a, b in PAIRS],
        "protocol": "fixed hyperparameters; see code/common.py",
    }
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved: {outpath}")


# ---------- Shuffled-regime control (shuffled-label control) ----------

def run_shuffled_controls(df, npz, sembs, regimes, shuffle_seeds, outpath: Path):
    """Run the surface-baseline pipeline with regime labels permuted per shuffle seed.

    Outputs an aggregate JSON with the same shape as the real-label
    surface_baselines.json, but nested per shuffle seed:

        {
          "per_shuffle_seed": {"9001": <real-shape-baseline-json>, ...},
          "_meta": {"shuffle_seeds": [...], ...}
        }

    Same shuffled labels here as in the probe-side shuffle (provided the
    probe script uses the same `np.random.default_rng(seed).permutation`
    on the same input regimes array). This matches the shuffled-label control
    scoring rule which requires apples-to-apples shuffled comparison.
    """
    import numpy as np
    aggregate = {"per_shuffle_seed": {}, "_meta": {
        "shuffle_seeds": list(shuffle_seeds),
        "kind": "shuffled_regime_control_surface_baselines",
        "shuffled_control": "label permutation with the same pipeline",
        "shuffle_algorithm": "np.random.default_rng(seed).permutation(regimes)",
    }}
    for seed in shuffle_seeds:
        rng = np.random.default_rng(seed)
        shuffled_regimes = rng.permutation(regimes)
        print(f"\n[shuffle seed {seed}] running surface baselines with permuted labels")
        # Write to a scratch file per seed, then read back the per-seed result.
        # We reuse run_pipeline's output schema; just redirect output.
        scratch = outpath.parent / f".scratch_seed{seed}_{outpath.name}"
        run_pipeline(df, npz, sembs, shuffled_regimes, scratch)
        aggregate["per_shuffle_seed"][str(seed)] = json.loads(scratch.read_text())
        scratch.unlink()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved shuffled-controls aggregate: {outpath}")


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
    print("surface-feature baselines — execution plan")
    print("=" * 72)
    print(f"Seeds: {SEEDS}  (5 seeds)")
    print(f"CV folds: {N_CV_FOLDS}  (StratifiedKFold, shuffle=True, random_state=seed)")
    print(f"Bootstrap resamples per AUROC point: {N_BOOTSTRAP}")
    print()
    print(f"Pairs (6 total):")
    for r1, r2 in PAIRS:
        print(f"  {r1} vs {r2}")
    print()
    print("Seven baseline configurations (six individual features and their combination), each evaluated per pair:")
    for f in FEATURE_NAMES:
        print(f"  {f}")
    print()
    print("Output: results/surface_baselines.json")
    print(
        "\nThis is the plan. Pass --synthetic-smoke or --execute to actually run.\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the execution plan and exit. No imports beyond stdlib.",
    )
    parser.add_argument(
        "--synthetic-smoke", action="store_true",
        help="Generate random data of the expected shape, run end-to-end, save to <outdir>/_smoke/.",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="REAL RUN against the prompt CSV + extracted hidden states.",
    )
    parser.add_argument(
        "--input",
        default="data/prompts.csv",
        help="Prompt CSV (data/prompts.csv).",
    )
    parser.add_argument(
        "--npz",
        default="hidden_states/hidden_states_qwen7b.npz",
        help="extracted hidden-states NPZ.",
    )
    parser.add_argument(
        "--outdir",
        default="results",
        help="Output directory for surface_baselines.json (and _smoke/ for smoke).",
    )
    parser.add_argument(
        "--output-name",
        default="surface_baselines",
        help="Base name (without .json) for the output file. Default "
             "'surface_baselines'. For other models, override "
             "(e.g. --output-name surface_baselines_llama8b).",
    )
    parser.add_argument(
        "--shuffle-seeds", default=None,
        help="Optional comma-separated shuffle seeds "
             "(e.g. '9001,9002,9003,9004,9005'). When provided in --execute "
             "mode, also runs the full pipeline with regime labels permuted "
             "independently per seed and saves an aggregated shuffled-controls "
             "JSON at <outdir>/<output-name>_shuffled.json. Does NOT affect the "
             "main real-label run.",
    )
    args = parser.parse_args()

    # Mutual exclusivity / safety
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
        print(">>> SYNTHETIC SMOKE: using fake data; no real artifacts touched.")
        df, npz, regimes = generate_synthetic_data()
        import numpy as np
        # Random sentence embeddings (384-dim, MiniLM-style) — skip the encoder.
        rng = np.random.default_rng(0)
        sembs = rng.standard_normal((len(df), 384)).astype(np.float64)
        outpath = Path(args.outdir) / "_smoke" / f"{args.output_name}.json"
        run_pipeline(df, npz, sembs, regimes, outpath)
        # If shuffle-seeds is given in smoke mode, exercise that code path too.
        if args.shuffle_seeds:
            shuffle_seeds = [int(s.strip()) for s in args.shuffle_seeds.split(",")]
            print(f">>> SYNTHETIC SMOKE: also exercising shuffled-control path with seeds {shuffle_seeds}")
            shuffled_outpath = Path(args.outdir) / "_smoke" / f"{args.output_name}_shuffled.json"
            run_shuffled_controls(df, npz, sembs, regimes, shuffle_seeds, shuffled_outpath)
        print("\nSmoke complete. Verify the JSON looks sane.")
        return

    # --execute: real run.
    print(">>> EXECUTE: real-data run.")
    in_path = Path(args.input)
    npz_path = Path(args.npz)
    df, npz, regimes = load_real_data(in_path, npz_path)
    print(f"Loaded {len(df)} prompts from {in_path}")
    print(f"Loaded hidden-states sidecar from {npz_path}")
    print(f"Encoding sentences via {SENTENCE_EMBED_MODEL}...")
    sembs = encode_sentences(df["question"].astype(str).to_list())
    print(f"  embeddings shape: {sembs.shape}")
    outpath = Path(args.outdir) / f"{args.output_name}.json"
    run_pipeline(df, npz, sembs, regimes, outpath)

    # Append provenance footer to the JSON.
    info = {
        "git_commit_at_run": "anonymized",
        "input_csv_sha256": sha256_of_file(in_path),
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
        run_shuffled_controls(df, npz, sembs, regimes, shuffle_seeds, shuffled_outpath)
        shuffled = json.loads(shuffled_outpath.read_text())
        shuffled["_provenance"] = {**info, "shuffle_seeds": shuffle_seeds}
        shuffled_outpath.write_text(json.dumps(shuffled, indent=2) + "\n", encoding="utf-8")
        print(f"Shuffled-control provenance appended: {shuffled_outpath}")


if __name__ == "__main__":
    main()
