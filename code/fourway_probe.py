#!/usr/bin/env python3
"""
Four-way multiclass probe .

Fixed hyperparameters: L2 logistic regression (C=1.0, lbfgs, max_iter=2000) on raw fp32
last-token states; 5-fold cross-validation x 5 seeds (1234, 4567, 7890, 2345, 5678);
bootstrap 95% CI (1000 resamples); shuffled-label control seeds 9001-9005.

Per model x dataset: multinomial LR (C=1.0, lbfgs, max_iter=2000, raw fp32
last-token states) per layer; 5-fold CV x 5 seeds (StratifiedKFold on base,
StratifiedGroupKFold on paraphrases); pooled-OOF 4-way accuracy (primary),
macro-OvR AUROC and macro recall (secondary); bootstrap 95% CIs; confusion
matrix at the peak in-window layer; shuffled-label nulls (9001-9005, global
on base, group-level on paraphrases) over in-window layers.

Usage:
  python3 code/fourway_probe.py --model qwen7b --dataset base --execute
  python3 code/fourway_probe.py --model qwen7b --dataset base --baselines
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import common as C


def eval_layer(X, y, groups):
    """All 5 CV seeds on one layer. Returns per-seed metrics + pooled preds."""
    accs, mrecs, aurocs, per_seed_preds = [], [], [], []
    for seed in C.SEEDS:
        proba, yhat = C.oof_multiclass(X, y, seed, groups=groups)
        accs.append(C.accuracy_4way(y, yhat))
        mrecs.append(C.macro_recall_4way(y, yhat))
        aurocs.append(C.macro_ovr_auroc(y, proba))
        per_seed_preds.append(yhat)
    # Bootstrap CI on accuracy, per seed, averaged ( convention).
    los, his = [], []
    for seed, yhat in zip(C.SEEDS, per_seed_preds):
        lo, hi = C.bootstrap_ci(C.accuracy_4way, np.asarray(y), np.asarray(yhat), seed)
        los.append(lo)
        his.append(hi)
    return {
        "accuracy_mean_across_seeds": float(np.mean(accs)),
        "accuracy_per_seed": [float(a) for a in accs],
        "accuracy_ci_low_across_seeds": float(np.nanmean(los)),
        "accuracy_ci_high_across_seeds": float(np.nanmean(his)),
        "macro_recall_mean_across_seeds": float(np.mean(mrecs)),
        "macro_ovr_auroc_mean_across_seeds": float(np.nanmean(aurocs)),
        "macro_ovr_auroc_per_seed": [float(a) for a in aurocs],
    }, per_seed_preds


def confusion_at_layer(y, per_seed_preds):
    """Row-normalized 4x4 confusion pooled across the 5 seeds' OOF predictions."""
    from sklearn.metrics import confusion_matrix
    ys = np.concatenate([y] * len(per_seed_preds))
    yh = np.concatenate(per_seed_preds)
    cm = confusion_matrix(ys, yh, labels=list(C.CLASS_ORDER)).astype(np.float64)
    counts = cm.copy()
    rowsum = cm.sum(axis=1, keepdims=True)
    cm = cm / np.maximum(rowsum, 1)
    return {
        "class_order": list(C.CLASS_ORDER),
        "counts_pooled_over_seeds": counts.astype(int).tolist(),
        "row_normalized": [[round(float(v), 4) for v in row] for row in cm],
    }


def run_probe(model, dataset, n_jobs):
    from joblib import Parallel, delayed
    df, npz, regimes = C.load_aligned(model, dataset)
    groups = df["paraphrase_group_id"].astype(str).to_numpy() if dataset == "paraphrases" else None
    H = npz["last_token_states"].astype(np.float32)
    n, n_layers, d = H.shape
    lo, hi = C.LAYER_WINDOWS[model]
    print(f"{model}/{dataset}: n={n} layers={n_layers} d={d} window=[{lo},{hi}]")

    layer_results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(eval_layer)(H[:, L, :], regimes, groups) for L in range(n_layers)
    )
    per_layer = {str(L): layer_results[L][0] for L in range(n_layers)}

    in_window = [L for L in range(n_layers) if lo <= L <= hi]
    peak_L = max(in_window, key=lambda L: per_layer[str(L)]["accuracy_mean_across_seeds"])
    peak = per_layer[str(peak_L)]
    confusion = confusion_at_layer(regimes, layer_results[peak_L][1])
    print(f"PEAK in-window: L{peak_L} accuracy={peak['accuracy_mean_across_seeds']:.4f} "
          f"macro-OvR-AUROC={peak['macro_ovr_auroc_mean_across_seeds']:.4f}")

    return {
        "per_layer": per_layer,
        "peak_in_window": {"layer": peak_L, **peak, "confusion_matrix": confusion},
        "_meta": {
            "model": model, "dataset": dataset,
            "n_prompts": int(n), "n_layers": int(n_layers), "hidden_dim": int(d),
            "layer_window_inclusive": [lo, hi],
            "class_counts": {r: int((regimes == r).sum()) for r in C.REGIMES},
            "chance_accuracy": round(max((regimes == r).mean() for r in C.REGIMES), 4),
            "cv": ("StratifiedGroupKFold(groups=paraphrase_group_id)" if groups is not None
                   else "StratifiedKFold"),
            "seeds": list(C.SEEDS), "n_cv_folds": C.N_CV_FOLDS, "n_bootstrap": C.N_BOOTSTRAP,
            "bootstrap_rng": "np.random.default_rng(cv_seed * 1_000_003)",
            "lr_config": {"C": C.LR_C, "penalty": "l2 (sklearn default)",
                          "solver": C.LR_SOLVER, "max_iter": C.LR_MAX_ITER,
                          "loss": "multinomial", "preprocessing": "none (raw fp32 last-token states)"},
            "representation": "last_token_states",
            "protocol": "fixed hyperparameters; see code/common.py",
        },
    }


def run_shuffled(model, dataset, n_jobs):
    """Shuffled-label nulls over in-window layers only."""
    from joblib import Parallel, delayed
    df, npz, regimes = C.load_aligned(model, dataset)
    groups = df["paraphrase_group_id"].astype(str).to_numpy() if dataset == "paraphrases" else None
    H = npz["last_token_states"].astype(np.float32)
    lo, hi = C.LAYER_WINDOWS[model]
    in_window = list(range(lo, hi + 1))
    agg = {"per_shuffle_seed": {}, "_meta": {
        "model": model, "dataset": dataset, "shuffle_seeds": list(C.SHUFFLE_SEEDS),
        "layers": in_window,
        "shuffle_algorithm": ("group-level permutation (group-level permutation)" if groups is not None
                              else "np.random.default_rng(seed).permutation(regimes)"),
        "protocol": "fixed hyperparameters; see code/common.py",
    }}
    for s in C.SHUFFLE_SEEDS:
        if groups is not None:
            y_shuf = C.group_level_shuffle(regimes, groups, s)
        else:
            y_shuf = np.random.default_rng(s).permutation(regimes)
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(eval_layer)(H[:, L, :], y_shuf, groups) for L in in_window
        )
        agg["per_shuffle_seed"][str(s)] = {
            str(L): results[i][0] for i, L in enumerate(in_window)
        }
        mx = max(r[0]["accuracy_mean_across_seeds"] for r in results)
        print(f"  shuffle {s}: max in-window accuracy {mx:.4f}")
    return agg


# ---------- 4-way baselines ----------

def summarize_4way_seeds(regimes, per_seed_yhat, accs, aurocs, mrecs):
    """Seed-mean accuracy with -convention bootstrap CIs + macro recall."""
    los, his = [], []
    for seed, yhat in zip(C.SEEDS, per_seed_yhat):
        lo, hi = C.bootstrap_ci(C.accuracy_4way, np.asarray(regimes), np.asarray(yhat), seed)
        los.append(lo)
        his.append(hi)
    return {"accuracy_mean_across_seeds": float(np.mean(accs)),
            "accuracy_per_seed": [float(a) for a in accs],
            "accuracy_ci_low_across_seeds": float(np.nanmean(los)),
            "accuracy_ci_high_across_seeds": float(np.nanmean(his)),
            "macro_recall_mean_across_seeds": float(np.mean(mrecs)),
            "macro_ovr_auroc_mean_across_seeds": float(np.nanmean(aurocs))}


def surface_4way(df, npz, regimes, groups, sembs):
    """all-surface analog: 12 static dims + fold-safe sentence-cos, multinomial."""
    from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
    static = C.static_covariates(df, npz)
    accs, aurocs, mrecs, per_seed_yhat = [], [], [], []
    for seed in C.SEEDS:
        if groups is None:
            cv = StratifiedKFold(n_splits=C.N_CV_FOLDS, shuffle=True, random_state=seed)
            splits = cv.split(static, regimes)
        else:
            cv = StratifiedGroupKFold(n_splits=C.N_CV_FOLDS, shuffle=True, random_state=seed)
            splits = cv.split(static, regimes, groups=groups)
        proba = np.zeros((len(regimes), len(C.CLASS_ORDER)))
        all_idx = np.arange(len(regimes))
        for tr, te in splits:
            cos = C.fold_sentence_cos(sembs, regimes, all_idx, te)
            X = np.concatenate([static, cos], axis=1)
            lr = C.make_lr(seed).fit(X[tr], regimes[tr])
            p = lr.predict_proba(X[te])
            col = {c: i for i, c in enumerate(lr.classes_)}
            for j, c in enumerate(C.CLASS_ORDER):
                if c in col:
                    proba[te, j] = p[:, col[c]]
        yhat = np.array([C.CLASS_ORDER[i] for i in proba.argmax(axis=1)])
        per_seed_yhat.append(yhat)
        accs.append(C.accuracy_4way(regimes, yhat))
        mrecs.append(C.macro_recall_4way(regimes, yhat))
        aurocs.append(C.macro_ovr_auroc(regimes, proba))
    return summarize_4way_seeds(regimes, per_seed_yhat, accs, aurocs, mrecs)


def tfidf_pipeline(variant, seed):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline, FeatureUnion
    if variant == "word":
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True,
                              min_df=1, analyzer="word")
    elif variant == "char":
        vec = TfidfVectorizer(ngram_range=(3, 5), sublinear_tf=True, lowercase=True,
                              min_df=1, analyzer="char_wb")
    else:
        vec = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                     lowercase=True, min_df=1, analyzer="word")),
            ("char", TfidfVectorizer(ngram_range=(3, 5), sublinear_tf=True,
                                     lowercase=True, min_df=1, analyzer="char_wb")),
        ])
    return Pipeline([("vec", vec), ("lr", C.make_lr(seed))])


def text_or_emb_4way(X_or_texts, regimes, groups, seed, pipeline_variant=None):
    """Generic 4-way CV eval for TF-IDF (texts + Pipeline) or embeddings (array)."""
    from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
    n = len(regimes)
    if groups is None:
        cv = StratifiedKFold(n_splits=C.N_CV_FOLDS, shuffle=True, random_state=seed)
        splits = cv.split(np.zeros(n), regimes)
    else:
        cv = StratifiedGroupKFold(n_splits=C.N_CV_FOLDS, shuffle=True, random_state=seed)
        splits = cv.split(np.zeros(n), regimes, groups=groups)
    proba = np.zeros((n, len(C.CLASS_ORDER)))
    for tr, te in splits:
        if pipeline_variant is not None:
            clf = tfidf_pipeline(pipeline_variant, seed)
            clf.fit([X_or_texts[i] for i in tr], regimes[tr])
            p = clf.predict_proba([X_or_texts[i] for i in te])
            classes = clf.named_steps["lr"].classes_
        else:
            clf = C.make_lr(seed).fit(X_or_texts[tr], regimes[tr])
            p = clf.predict_proba(X_or_texts[te])
            classes = clf.classes_
        col = {c: i for i, c in enumerate(classes)}
        for j, c in enumerate(C.CLASS_ORDER):
            if c in col:
                proba[te, j] = p[:, col[c]]
    yhat = np.array([C.CLASS_ORDER[i] for i in proba.argmax(axis=1)])
    return (C.accuracy_4way(regimes, yhat), C.macro_ovr_auroc(regimes, proba),
            C.macro_recall_4way(regimes, yhat), yhat)


def run_baselines(model, dataset):
    df, npz, regimes = C.load_aligned(model, dataset)
    groups = df["paraphrase_group_id"].astype(str).to_numpy() if dataset == "paraphrases" else None
    texts = df["question"].astype(str).tolist()
    out = {"_meta": {"model": model, "dataset": dataset, "seeds": list(C.SEEDS),
                     "protocol": "fixed hyperparameters; see code/common.py"}}

    print("surface (6-feature, fold-safe centroid), 4-way...")
    sembs = C.encode_sentences(texts)
    out["surface_all"] = surface_4way(df, npz, regimes, groups, sembs)
    print(f"  surface 4-way acc {out['surface_all']['accuracy_mean_across_seeds']:.4f}")

    for variant in ("word", "char", "combined"):
        accs, aurocs, mrecs, yhats = [], [], [], []
        for seed in C.SEEDS:
            a, u, m, yh = text_or_emb_4way(texts, regimes, groups, seed, pipeline_variant=variant)
            accs.append(a)
            aurocs.append(u)
            mrecs.append(m)
            yhats.append(yh)
        out[f"tfidf_{variant}"] = summarize_4way_seeds(regimes, yhats, accs, aurocs, mrecs)
        print(f"  tfidf_{variant} 4-way acc {np.mean(accs):.4f}")

    encoders = [("sentence-transformers/all-MiniLM-L6-v2", "minilm"),
                ("sentence-transformers/all-mpnet-base-v2", "mpnet"),
                ("BAAI/bge-large-en-v1.5", "bge_large")]
    for name, short in encoders:
        embs = sembs if short == "minilm" else C.encode_sentences(texts, name)
        accs, aurocs, mrecs, yhats = [], [], [], []
        for seed in C.SEEDS:
            a, u, m, yh = text_or_emb_4way(embs, regimes, groups, seed)
            accs.append(a)
            aurocs.append(u)
            mrecs.append(m)
            yhats.append(yh)
        out[f"encoder_{short}"] = summarize_4way_seeds(regimes, yhats, accs, aurocs, mrecs)
        print(f"  encoder_{short} 4-way acc {np.mean(accs):.4f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["qwen7b", "llama8b", "qwen32b"])
    ap.add_argument("--dataset", required=True, choices=["base", "paraphrases"])
    ap.add_argument("--execute", action="store_true", help="probe + shuffled nulls")
    ap.add_argument("--baselines", action="store_true", help="4-way baselines")
    ap.add_argument("--n-jobs", type=int, default=8)
    args = ap.parse_args()

    if (args.model, args.dataset) not in C.NPZ:
        raise SystemExit(f"npz not available for {args.model}/{args.dataset} "
                         "(qwen32b paraphrase hidden states must be regenerated first)")
    tag = f"v3_{args.model}" if args.dataset == "base" else f"v3paraphrases_{args.model}"
    files = [C.NPZ[(args.model, args.dataset)],
             C.BASE_CSV if args.dataset == "base" else C.PARA_CSV]

    if args.execute:
        res = run_probe(args.model, args.dataset, args.n_jobs)
        res["_provenance"] = C.provenance(files=files)
        C.save_versioned(res, Path("results") / f"probe_4way_{tag}.json")
        shuf = run_shuffled(args.model, args.dataset, args.n_jobs)
        shuf["_provenance"] = C.provenance(files=files)
        C.save_versioned(shuf, Path("results") / f"probe_4way_{tag}_shuffled.json")
    if args.baselines:
        res = run_baselines(args.model, args.dataset)
        res["_provenance"] = C.provenance(files=files)
        C.save_versioned(res, Path("results") / f"baselines_4way_{tag}.json")


if __name__ == "__main__":
    main()
