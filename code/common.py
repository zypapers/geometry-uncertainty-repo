#!/usr/bin/env python3
"""
Shared helpers for the additional analyses (four-way probe, covariate-adjusted
probes, residualization decomposition).

Fixed hyperparameters: L2 logistic regression (C=1.0, lbfgs, max_iter=2000) on raw fp32
last-token states; 5-fold cross-validation x 5 seeds (1234, 4567, 7890, 2345, 5678);
bootstrap 95% CI (1000 resamples); shuffled-label control seeds 9001-9005.

Every constant here mirrors code/linear_probes.py and code/surface_baselines.py;
the released result files were produced with these values.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

# ---------- fixed constants ----------

SEEDS = (1234, 4567, 7890, 2345, 5678)
SHUFFLE_SEEDS = (9001, 9002, 9003, 9004, 9005)
N_BOOTSTRAP = 1000
N_CV_FOLDS = 5
REGIMES = ("known", "unknown", "ambiguous", "misleading")
CLASS_ORDER = ("ambiguous", "known", "misleading", "unknown")  # sklearn sorted order
PAIRS = [
    ("known", "unknown"),
    ("known", "ambiguous"),
    ("known", "misleading"),
    ("unknown", "ambiguous"),
    ("unknown", "misleading"),
    ("ambiguous", "misleading"),
]
LR_C = 1.0
LR_SOLVER = "lbfgs"
LR_MAX_ITER = 2000

QUESTION_FORMS = ("which", "what", "who", "when", "where", "why", "how", "other")
SENTENCE_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# In-window reported layer ranges per model (inclusive), per separation_gate_v3_*.json.
LAYER_WINDOWS = {"qwen7b": (14, 28), "llama8b": (16, 32), "qwen32b": (32, 64)}

# Hidden-only peak in-window layers on Qwen-7B base, per separation_gate_v3_qwen7b.json.
QWEN7B_PEAK_LAYERS = {
    "known_vs_unknown": 15,
    "known_vs_ambiguous": 16,
    "known_vs_misleading": 19,
    "unknown_vs_ambiguous": 17,
    "unknown_vs_misleading": 28,
    "ambiguous_vs_misleading": 18,
}
QWEN7B_SURFACE_BARS = {
    "known_vs_unknown": 0.763,
    "known_vs_ambiguous": 0.833,
    "known_vs_misleading": 0.867,
    "unknown_vs_ambiguous": 0.753,
    "unknown_vs_misleading": 0.785,
    "ambiguous_vs_misleading": 0.718,
}

BASE_CSV = "data/prompts.csv"
PARA_CSV = "data/paraphrases.csv"
NPZ = {
    ("qwen7b", "base"): "hidden_states/hidden_states_qwen7b.npz",
    ("llama8b", "base"): "hidden_states/hidden_states_llama8b.npz",
    ("qwen32b", "base"): "hidden_states/hidden_states_qwen32b.npz",
    ("qwen7b", "paraphrases"): "hidden_states/hidden_states_paraphrases_qwen7b.npz",
    ("llama8b", "paraphrases"): "hidden_states/hidden_states_paraphrases_llama8b.npz",
    # Qwen-32B paraphrase hidden states were re-extracted (the original file was not retained).
    ("qwen32b", "paraphrases"): "hidden_states/hidden_states_paraphrases_qwen32b.npz",
}


# ---------- Loading ----------

def load_aligned(model: str, dataset: str):
    """Load npz + CSV, hard-verify id alignment. Returns (df, npz, regimes)."""
    import pandas as pd
    csv_path = BASE_CSV if dataset == "base" else PARA_CSV
    npz_path = NPZ[(model, dataset)]
    df = pd.read_csv(csv_path)
    npz = np.load(npz_path, allow_pickle=True)
    csv_ids = [str(x) for x in df["id"]]
    npz_ids = [str(x) for x in npz["prompt_ids"]]
    if csv_ids != npz_ids:
        raise SystemExit(f"id ordering differs: {csv_path} vs {npz_path}; refusing to run")
    regimes = df["regime"].astype(str).to_numpy()
    npz_regimes = np.asarray(npz["regimes"]).astype(str)
    if not (regimes == npz_regimes).all():
        raise SystemExit("regime mismatch between csv and npz; refusing to run")
    return df, npz, regimes


# ---------- Metrics ----------

def auroc_binary(y, p):
    from sklearn.metrics import roc_auc_score
    if len(set(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def bootstrap_ci(metric_fn, y, pred, seed, n_boot=N_BOOTSTRAP):
    """Percentile bootstrap CI on pooled OOF (y, pred). rng mirrors  convention."""
    rng = np.random.default_rng(seed * 1_000_003)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        v = metric_fn(y[idx], pred[idx])
        if not np.isnan(v):
            vals.append(v)
    vals = np.array(vals, dtype=np.float64)
    if len(vals) < 2:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def accuracy_4way(y, yhat):
    return float((y == yhat).mean())


def macro_recall_4way(y, yhat):
    from sklearn.metrics import recall_score
    return float(recall_score(y, yhat, average="macro", zero_division=0))


def macro_ovr_auroc(y, proba, classes=CLASS_ORDER):
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y, proba, multi_class="ovr", average="macro",
                                    labels=list(classes)))
    except ValueError:
        return float("nan")


# ---------- Probes ----------

def make_lr(seed):
    from sklearn.linear_model import LogisticRegression
    # Default penalty='l2' (default; not passed explicitly — deprecated in sklearn 1.8).
    return LogisticRegression(C=LR_C, solver=LR_SOLVER, max_iter=LR_MAX_ITER,
                              random_state=seed)


def oof_binary(X, y, seed):
    """5-fold StratifiedKFold binary LR; pooled out-of-fold P(class=1)."""
    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=np.float64)
    for tr, te in cv.split(X, y):
        lr = make_lr(seed).fit(X[tr], y[tr])
        oof[te] = lr.predict_proba(X[te])[:, 1]
    return oof


def oof_multiclass(X, y, seed, groups=None):
    """5-fold (group-aware if groups given) multinomial LR.

    Returns (proba, yhat): pooled OOF class probabilities in CLASS_ORDER
    column order, and argmax predicted labels.
    """
    from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
    if groups is None:
        cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=seed)
        splits = cv.split(X, y)
    else:
        cv = StratifiedGroupKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=seed)
        splits = cv.split(X, y, groups=groups)
    proba = np.zeros((len(y), len(CLASS_ORDER)), dtype=np.float64)
    for tr, te in splits:
        lr = make_lr(seed).fit(X[tr], y[tr])
        p = lr.predict_proba(X[te])
        # Map columns into the fixed CLASS_ORDER (classes_ is sorted, but be explicit).
        col = {c: i for i, c in enumerate(lr.classes_)}
        for j, c in enumerate(CLASS_ORDER):
            if c in col:
                proba[te, j] = p[:, col[c]]
    yhat = np.array([CLASS_ORDER[i] for i in proba.argmax(axis=1)])
    return proba, yhat


# ---------- Group-level label shuffle (group-level permutation) ----------

def group_level_shuffle(regimes, groups, seed):
    """Permute regime values across paraphrase groups, propagate to rows."""
    import pandas as pd
    unique_groups = pd.Series(groups).unique()
    group_to_regime = {}
    for g in unique_groups:
        labels = set(regimes[groups == g])
        if len(labels) != 1:
            raise SystemExit(f"group {g} has multiple regimes: {labels}")
        group_to_regime[g] = next(iter(labels))
    rng = np.random.default_rng(seed)
    vals = np.array([group_to_regime[g] for g in unique_groups])
    permuted = rng.permutation(vals)
    new_map = dict(zip(unique_groups, permuted))
    return np.array([new_map[g] for g in groups])


# ---------- Surface covariates (mirrors surface_baselines.py) ----------

def static_covariates(df, npz):
    """(N, 12) static covariate block: word count, NE count, 8-form onehot,
    token count, next-token entropy. Order fixed."""
    wc = np.array([len(str(q).split()) for q in df["question"]], dtype=np.float64).reshape(-1, 1)
    ne = df["named_entity_count_heuristic"].astype(np.float64).to_numpy().reshape(-1, 1)
    vals = df["question_form"].astype(str).str.lower().to_numpy()
    onehot = np.zeros((len(vals), len(QUESTION_FORMS)), dtype=np.float64)
    for i, v in enumerate(vals):
        onehot[i, QUESTION_FORMS.index(v if v in QUESTION_FORMS else "other")] = 1.0
    tc = npz["prompt_token_counts"].astype(np.float64).reshape(-1, 1)
    logits = npz["next_token_logits"].astype(np.float64)
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    p = p / p.sum(axis=1, keepdims=True)
    ent = (-(p * np.log(p + 1e-12)).sum(axis=1)).reshape(-1, 1)
    return np.concatenate([wc, ne, onehot, tc, ent], axis=1)


def encode_sentences(questions, model_name=SENTENCE_EMBED_MODEL):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model.encode(list(questions), convert_to_numpy=True,
                        normalize_embeddings=False).astype(np.float64)


def fold_sentence_cos(sembs_all, regimes_all, subset_indices, test_orig_indices):
    """Fold-safe cosine-to-Known-centroid for the rows in subset_indices.

    Centroid = mean of Known sentence embeddings NOT in the held-out test set
    (mirrors surface_baselines._fold_sentence_cos_for_pair exactly).
    """
    known_mask = (regimes_all == "known")
    train_known = known_mask.copy()
    train_known[test_orig_indices] = False
    embs = sembs_all[train_known]
    if len(embs) == 0:
        raise RuntimeError("no training Known prompts for centroid")
    centroid = embs.mean(axis=0)
    sub = sembs_all[subset_indices]
    num = (sub * centroid).sum(axis=-1)
    den = np.linalg.norm(sub, axis=-1) * np.linalg.norm(centroid)
    return (num / (den + 1e-12)).reshape(-1, 1)


def zscore_train(train_X, apply_X):
    """Z-score apply_X using train_X statistics (constant-column guard)."""
    mu = train_X.mean(axis=0)
    sd = train_X.std(axis=0) + 1e-9
    return (apply_X - mu) / sd


# ---------- Provenance ----------

def sha256_of_file(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_id() -> str:
    return "anonymized"


def provenance(extra=None, files=()):
    info = {
        "git_commit_at_run": "anonymized",
        "host": "anonymized",
        "python_version": sys.version.split()[0],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "file_sha256": {str(p): sha256_of_file(p) for p in files},
    }
    if extra:
        info.update(extra)
    return info


def save_versioned(obj, outpath: Path):
    """Never overwrite: suffix _r2, _r3, ... if the target exists."""
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    final = outpath
    k = 2
    while final.exists():
        final = outpath.with_name(outpath.stem + f"_r{k}" + outpath.suffix)
        k += 1
    final.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8")
    print(f"Saved: {final}")
    return final
