#!/usr/bin/env python3
"""Six-feature surface baseline on the 558-prompt paraphrase extension (group-aware CV).

  - Reads data/paraphrases.csv (558 rows, paraphrase_group_id present)
  - Uses StratifiedGroupKFold(groups=paraphrase_group_id) for all cross-validation
  - Group-level shuffled control (same permutation scheme as linear_probes_paraphrase.py)
  - Reads next-token entropy and prompt token count for the model from the
    hidden-state NPZ written by extract_hidden_states.py (no extra GPU pass)

Features (6 = "all_surface"): word count, named-entity count, question form
(8-class one-hot), sentence-embedding cosine to the Known centroid (leakage-safe
per fold), prompt token count, next-token entropy.

Output:
  results/surface_baselines_v3paraphrases_<model>.json           (real labels)
  results/surface_baselines_v3paraphrases_<model>_shuffled.json  (group shuffles)

The paraphrase-set surface baseline used in the paper is the maximum over the
three models' runs, and the criterion baseline is max(base_set, paraphrase_set) per pair.

Usage:
  python3 code/surface_baselines_paraphrase.py --execute --npz hidden_states/hidden_states_paraphrases_qwen7b.npz --slug qwen7b --output-name surface_baselines_v3paraphrases_qwen7b
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


QFORMS = ["what", "who", "when", "where", "why", "how", "which", "other"]
REGIMES = ("known", "unknown", "ambiguous", "misleading")
PAIRS = [
    ("known", "unknown"),
    ("known", "ambiguous"),
    ("known", "misleading"),
    ("unknown", "ambiguous"),
    ("unknown", "misleading"),
    ("ambiguous", "misleading"),
]
SEEDS = (1234, 4567, 7890, 2345, 5678)
SHUFFLE_SEEDS = (9001, 9002, 9003, 9004, 9005)
N_CV_FOLDS = 5


def cosine(a, b):
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)


def featurize(df, slug, entropy_arr_id_map, token_count_id_map):
    """Returns X (n × d), feature_names. Columns:
       wc, ne, prompt_token_count, next_token_entropy, qform_one_hot×8.
    """
    feats = []
    for _, r in df.iterrows():
        rid = r["id"]
        wc = float(r["word_count"])
        ne = float(r["named_entity_count_heuristic"])
        qf = [1.0 if r["question_form"] == q else 0.0 for q in QFORMS]
        ptc = float(token_count_id_map.get(rid, np.nan))
        if np.isnan(ptc):
            ptc = 0.0
        ent = float(entropy_arr_id_map.get(rid, np.nan))
        if np.isnan(ent):
            ent = 0.0
        feats.append([wc, ne, ptc, ent] + qf)
    fnames = ["word_count","named_entity_count_heuristic","prompt_token_count","next_token_entropy"] + [f"qform_{q}" for q in QFORMS]
    return np.array(feats, dtype=np.float64), fnames


def group_level_shuffle(regimes, groups, seed):
    unique_groups = pd.Series(groups).unique()
    group_to_regime = {}
    for g in unique_groups:
        labels = set(regimes[groups == g])
        if len(labels) != 1:
            raise SystemExit(f"Group {g} has mixed regimes: {labels}")
        group_to_regime[g] = next(iter(labels))
    regime_values = np.array([group_to_regime[g] for g in unique_groups])
    rng = np.random.default_rng(seed)
    permuted = rng.permutation(regime_values)
    permuted_map = dict(zip(unique_groups, permuted))
    return np.array([permuted_map[g] for g in groups])


def pair_auroc(df, embs_dict, X_base, regimes, groups, r1, r2, seeds=SEEDS):
    """Per-pair AUROC under the passed `regimes` labels.

    For real-label runs, `regimes` = original regime array (matches df["regime"]).
    For shuffled-label runs, `regimes` = group-level-permuted labels (drift from
    df["regime"]). All label-derived quantities (pair mask, y, leakage-safe
    Known centroid) MUST use the passed `regimes`, NOT df["regime"], otherwise
    shuffled controls leak the original labels via the feature pipeline.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler

    mask = (regimes == r1) | (regimes == r2)
    X_pair = X_base[mask]
    g_pair = groups[mask]
    y = (regimes[mask] == r1).astype(int)  # use PASSED labels, not df["regime"]

    all_embs = np.array([embs_dict[i] for i in df["id"]])
    pair_embs = all_embs[mask]
    orig_idx = np.where(mask)[0]

    aurocs = []
    for seed in seeds:
        cv = StratifiedGroupKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=seed)
        oof = np.zeros(len(y), dtype=np.float64)
        for tr, te in cv.split(X_pair, y, groups=g_pair):
            # leakage-safe sentence_cos_to_known_centroid: "Known" centroid
            # uses Known rows under the PASSED labels (real or shuffled),
            # excluding rows in this test fold (per-fold leakage safety).
            test_orig = orig_idx[te]
            kmask = (regimes == "known")  # passed labels (real or shuffled)
            kmask = kmask.copy()
            kmask[test_orig] = False
            centroid = all_embs[kmask].mean(axis=0)
            sc_pair = np.array([cosine(e, centroid) for e in pair_embs])
            Xc = np.column_stack([X_pair, sc_pair.reshape(-1, 1)])

            scaler = StandardScaler().fit(Xc[tr])
            clf = LogisticRegression(random_state=seed, max_iter=2000, solver="lbfgs")
            clf.fit(scaler.transform(Xc[tr]), y[tr])
            oof[te] = clf.predict_proba(scaler.transform(Xc[te]))[:, 1]
        aurocs.append(float(roc_auc_score(y, oof)))
    return float(np.mean(aurocs)), float(np.std(aurocs)), [float(a) for a in aurocs]


def compute_entropy_from_logits(logits):
    """Returns entropy per row (in nats). logits: (n, vocab)."""
    # Compute log-softmax stably
    m = logits.max(axis=1, keepdims=True)
    z = logits - m
    log_sum = np.log(np.exp(z).sum(axis=1, keepdims=True))
    log_probs = z - log_sum
    probs = np.exp(log_probs)
    ent = -(probs * log_probs).sum(axis=1)
    return ent.astype(np.float64)


def run_pipeline(df, embs_dict, X, regimes, groups, label_to_use, outpath: Path, seeds=SEEDS):
    """Compute per-pair AUROC (5 seeds × 5 fold StratifiedGroupKFold)."""
    results = {"per_pair": {}, "_meta": {
        "seeds": list(seeds),
        "n_cv_folds": N_CV_FOLDS,
        "pairs": [f"{a}_vs_{b}" for a, b in PAIRS],
        "feature_set": "all_surface (6-feature: wc, NE, qform-one-hot, ptc, entropy, sent_cos)",
        "cv": "StratifiedGroupKFold(groups=paraphrase_group_id)",
    }}
    for r1, r2 in PAIRS:
        mean, sd, per_seed = pair_auroc(df, embs_dict, X, label_to_use, groups, r1, r2, seeds=seeds)
        results["per_pair"][f"{r1}_vs_{r2}"] = {
            "auroc_mean_across_seeds": mean,
            "auroc_std_across_seeds": sd,
            "auroc_per_seed": per_seed,
            "n_pair": int(((label_to_use == r1) | (label_to_use == r2)).sum()),
        }
        print(f"  {r1}_vs_{r2:30s} AUROC = {mean:.4f} ± {sd:.4f}")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {outpath}")


def run_shuffled_controls(df, embs_dict, X, regimes, groups, shuffle_seeds, outpath: Path):
    aggregate = {"per_shuffle_seed": {}, "_meta": {
        "shuffle_seeds": list(shuffle_seeds),
        "kind": "group_level_shuffled_regime_control_surface",
    }}
    for seed in shuffle_seeds:
        print(f"\n[shuffle seed {seed}] group-level permutation")
        shuffled = group_level_shuffle(regimes, groups, seed)
        scratch = outpath.parent / f".scratch_surface_seed{seed}_{outpath.name}"
        run_pipeline(df, embs_dict, X, regimes, groups, shuffled, scratch)
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
    parser.add_argument("--npz", required=True)
    parser.add_argument("--slug", required=True, help="qwen7b | llama8b | qwen32b — model-specific feature slug")
    parser.add_argument("--with-paraphrases", default="data/paraphrases.csv")
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--skip-shuffles", action="store_true")
    parser.add_argument("--shuffle-seeds", default=",".join(str(s) for s in SHUFFLE_SEEDS))
    args = parser.parse_args()

    if args.dry_run:
        print("Paraphrase-set surface plan:")
        print(f"  npz:                  {args.npz}")
        print(f"  with_paraphrases_csv: {args.with_paraphrases}")
        print(f"  model slug:           {args.slug}")
        print(f"  seeds:                {SEEDS}")
        print(f"  CV:                   StratifiedGroupKFold(n_splits=5)")
        print(f"  output:               {args.outdir}/{args.output_name}.json")
        return
    if not args.execute:
        raise SystemExit("Pass --execute (or --dry-run).")

    csv_path = Path(args.with_paraphrases)
    df = pd.read_csv(csv_path)
    print(f"Loaded {csv_path}: {len(df)} rows")

    npz_path = Path(args.npz)
    npz = np.load(npz_path, allow_pickle=True)
    print(f"Loaded {npz_path}: {npz['last_token_states'].shape}")

    # Build id-keyed lookups for prompt_token_count + entropy from npz
    npz_ids = np.asarray(npz["prompt_ids"]).astype(str)
    ptcs = np.asarray(npz["prompt_token_counts"]).astype(float)
    logits = np.asarray(npz["next_token_logits"]).astype(np.float32)
    entropies = compute_entropy_from_logits(logits)
    print(f"Computed next_token_entropy from logits: {entropies.shape}, mean={entropies.mean():.3f}")

    token_count_map = dict(zip(npz_ids, ptcs))
    entropy_map = dict(zip(npz_ids, entropies))

    # Reorder df to match npz order so X aligns with npz arrays
    df = df.set_index("id").loc[npz_ids].reset_index()
    regimes = df["regime"].to_numpy()
    groups = df["paraphrase_group_id"].to_numpy()
    print(f"  groups={len(set(groups))}, regime counts: {dict(pd.Series(regimes).value_counts())}")

    # Sentence embeddings (CPU)
    print("\nEncoding sentence embeddings...")
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embs = st.encode(df["question"].tolist(), convert_to_numpy=True, show_progress_bar=False)
    embs_dict = dict(zip(df["id"], embs))

    # Featurize (excluding sentence_cos — added per-fold in pair_auroc)
    X, fnames = featurize(df, args.slug, entropy_map, token_count_map)
    print(f"  features: {fnames}")
    print(f"  X shape: {X.shape}")

    # Real-label run
    print(f"\n=== Real labels (per-pair surface AUROC) ===")
    outpath = Path(args.outdir) / f"{args.output_name}.json"
    run_pipeline(df, embs_dict, X, regimes, groups, regimes, outpath)

    info = {
        "git_commit_at_run": current_git_commit(),
        "npz_sha256": sha256_of_file(npz_path),
        "csv_sha256": sha256_of_file(csv_path),
        "script_sha256": sha256_of_file(Path(__file__)),
        "host": socket.gethostname(),
        "python_version": sys.version.split()[0],
        "slug": args.slug,
    }
    saved = json.loads(outpath.read_text())
    saved["_provenance"] = info
    outpath.write_text(json.dumps(saved, indent=2) + "\n", encoding="utf-8")

    if not args.skip_shuffles:
        shuffle_seeds = [int(s.strip()) for s in args.shuffle_seeds.split(",") if s.strip()]
        sh_outpath = Path(args.outdir) / f"{args.output_name}_shuffled.json"
        print(f"\n=== Group-level shuffled surface ({len(shuffle_seeds)} seeds) ===")
        run_shuffled_controls(df, embs_dict, X, regimes, groups, shuffle_seeds, sh_outpath)
        shuffled = json.loads(sh_outpath.read_text())
        shuffled["_provenance"] = {**info, "shuffle_seeds": shuffle_seeds}
        sh_outpath.write_text(json.dumps(shuffled, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
