#!/usr/bin/env python3
"""
Covariate-adjusted probes .

Fixed hyperparameters: L2 logistic regression (C=1.0, lbfgs, max_iter=2000) on raw fp32
last-token states; 5-fold cross-validation x 5 seeds (1234, 4567, 7890, 2345, 5678);
bootstrap 95% CI (1000 resamples); shuffled-label control seeds 9001-9005.

Qwen-7B base set, window L14-L28, 6 pairs, conditions:
  A hidden-only          (consistency check vs probe_auroc_v3_qwen7b.json)
  B concat               [hidden ; z-scored 13-dim covariates]  (covariate inclusion)
  C residualization      per-training-fold OLS covariates -> each hidden dim,
                         subtract fitted values from train AND test, probe residuals
  D surface-only         re-emitted reference (Table 1 bars)

Covariates (exactly the paper's six surface features): word count, NE count,
question-form one-hot (8), prompt token count, next-token entropy, and the
fold-safe sentence-cos-to-Known-centroid (centroid from training-fold Known
prompts only, recomputed inside every fold). Z-scored on training-fold stats.

Shuffled-label control (9001-9005) on condition C at the six peak layers.

Usage:
  python3 code/covariate_probe.py --execute [--model qwen7b] [--n-jobs 8]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import common as C

CONDITIONS = ("hidden_only", "concat", "residualized")


def load_peaks_bars(model):
    """Per-pair IN-WINDOW hidden-only peak layers + surface bars for a model.

    Peaks are computed as the argmax of the published per-layer probe AUROC
    (probe_auroc_v3_{model}.json, last_token_states) restricted to the model's
    evaluation window. NOTE: the separation-criterion 'peak_layer' field is a peak over
    ALL layers and falls out-of-window for some llama8b/qwen32b pairs; the
    in-window peak is the convention used in the paper and is what this analysis keys to.
    Bars come from the separation-criterion JSON. For qwen7b the derived peaks/bars must
    equal the constants hard-coded in common (sanity-asserted).
    """
    lo, hi = C.LAYER_WINDOWS[model]
    pub = json.load(open(f"results/probe_auroc_v3_{model}.json"))
    peaks = {}
    for pair_key, block in pub["per_pair_per_layer"].items():
        curve = block["last_token_states"]
        peaks[pair_key] = max(range(lo, hi + 1),
                              key=lambda L: curve[str(L)]["auroc_mean_across_seeds"])
    kg = json.load(open(f"results/separation_gate_v3_{model}.json"))
    bars = {p["pair"]: float(p["surface_bar"]) for p in kg["per_pair"]}
    if model == "qwen7b":
        assert peaks == C.QWEN7B_PEAK_LAYERS, (peaks, C.QWEN7B_PEAK_LAYERS)
        assert bars == C.QWEN7B_SURFACE_BARS, (bars, C.QWEN7B_SURFACE_BARS)
    return peaks, bars


def oof_condition(H_layer, static12, sembs, regimes_all, pair_indices, y, seed, condition):
    """Pooled OOF P(class=1) for one (pair, layer, seed, condition).

    H_layer:   (N_all, d) hidden states at this layer (all 200 prompts).
    static12:  (N_all, 12) static covariates.
    sembs:     (N_all, 384) MiniLM embeddings (for the fold-safe cos covariate).
    pair_indices: indices (into all-200) of the pair's rows.
    y: binary labels over the pair rows.
    """
    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=C.N_CV_FOLDS, shuffle=True, random_state=seed)
    Xh = H_layer[pair_indices].astype(np.float64)
    oof = np.zeros(len(y), dtype=np.float64)
    for tr, te in cv.split(Xh, y):
        te_orig = pair_indices[te]
        if condition == "hidden_only":
            Xtr, Xte = Xh[tr], Xh[te]
        else:
            cos = C.fold_sentence_cos(sembs, regimes_all, pair_indices, te_orig)
            cov = np.concatenate([static12[pair_indices], cos], axis=1)  # (n_pair, 13)
            cov_z = C.zscore_train(cov[tr], cov)
            if condition == "concat":
                Xtr = np.concatenate([Xh[tr], cov_z[tr]], axis=1)
                Xte = np.concatenate([Xh[te], cov_z[te]], axis=1)
            elif condition == "residualized":
                # OLS (with intercept) from z-scored covariates to each hidden dim,
                # fit on train only; subtract fitted values from train and test.
                A_tr = np.concatenate([cov_z[tr], np.ones((len(tr), 1))], axis=1)
                A_te = np.concatenate([cov_z[te], np.ones((len(te), 1))], axis=1)
                beta, *_ = np.linalg.lstsq(A_tr, Xh[tr], rcond=None)
                Xtr = Xh[tr] - A_tr @ beta
                Xte = Xh[te] - A_te @ beta
            else:
                raise ValueError(condition)
        lr = C.make_lr(seed).fit(Xtr, y[tr])
        oof[te] = lr.predict_proba(Xte)[:, 1]
    return oof


def eval_cell(H_layer, static12, sembs, regimes_all, pair_indices, y, condition):
    """All 5 seeds for one (pair, layer, condition)."""
    points, los, his = [], [], []
    for seed in C.SEEDS:
        oof = oof_condition(H_layer, static12, sembs, regimes_all, pair_indices, y,
                            seed, condition)
        points.append(C.auroc_binary(y, oof))
        lo, hi = C.bootstrap_ci(C.auroc_binary, np.asarray(y), oof, seed)
        los.append(lo)
        his.append(hi)
    return {
        "auroc_mean_across_seeds": float(np.nanmean(points)),
        "auroc_per_seed": [float(p) for p in points],
        "auroc_ci_low_across_seeds": float(np.nanmean(los)),
        "auroc_ci_high_across_seeds": float(np.nanmean(his)),
    }


def run(model, n_jobs):
    from joblib import Parallel, delayed
    peaks, bars = load_peaks_bars(model)
    df, npz, regimes = C.load_aligned(model, "base")
    H = npz["last_token_states"].astype(np.float32)
    static12 = C.static_covariates(df, npz)
    print("Encoding MiniLM sentence embeddings (fold-safe cos covariate)...")
    sembs = C.encode_sentences(df["question"].astype(str).tolist())
    lo, hi = C.LAYER_WINDOWS[model]
    layers = list(range(lo, hi + 1))

    # Published hidden-only per-layer AUROCs, for the condition-A consistency check.
    pub = json.load(open(f"results/probe_auroc_v3_{model}.json"))
    pub_pp = pub["per_pair_per_layer"]

    jobs = []
    for r1, r2 in C.PAIRS:
        pair_key = f"{r1}_vs_{r2}"
        mask = (regimes == r1) | (regimes == r2)
        pair_indices = np.where(mask)[0]
        y = (regimes[mask] == r1).astype(np.int64)
        for L in layers:
            for cond in CONDITIONS:
                jobs.append((pair_key, L, cond, pair_indices, y))

    print(f"{len(jobs)} (pair, layer, condition) cells on {n_jobs} workers...")
    cells = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(eval_cell)(H[:, L, :], static12, sembs, regimes, idx, y, cond)
        for (_, L, cond, idx, y) in jobs
    )

    out = {"per_pair": {}, "_meta": {
        "model": model, "layer_window_inclusive": [lo, hi],
        "conditions": list(CONDITIONS),
        "covariates": ["word_count", "named_entity_count", "question_form_onehot8",
                       "prompt_token_count", "next_token_entropy",
                       "sentence_cos_to_known_centroid_fold_safe"],
        "zscore": "training-fold statistics, +1e-9 constant-column guard",
        "residualizer": "OLS with intercept, fit on training fold only",
        "seeds": list(C.SEEDS), "n_cv_folds": C.N_CV_FOLDS, "n_bootstrap": C.N_BOOTSTRAP,
        "bootstrap_rng": "np.random.default_rng(cv_seed * 1_000_003)",
        "hidden_dtype_note": "hidden states cast to float64 for the LR fit; verified numerically equivalent to the published fp32-input probe values",
        "lr_config": {"C": C.LR_C, "penalty": "l2 (sklearn default)", "solver": C.LR_SOLVER,
                      "max_iter": C.LR_MAX_ITER},
        "representation": "last_token_states",
        "surface_bars": bars,
        "peak_layers_hidden_only": peaks,
        "protocol": "fixed hyperparameters; see code/common.py",
    }}
    for (pair_key, L, cond, _, _), cell in zip(jobs, cells):
        out["per_pair"].setdefault(pair_key, {c: {} for c in CONDITIONS})
        out["per_pair"][pair_key][cond][str(L)] = cell

    # Peak-layer summary (primary metric).
    summary = {}
    for pair_key, peak_L in peaks.items():
        blocks = out["per_pair"][pair_key]
        a = blocks["hidden_only"][str(peak_L)]["auroc_mean_across_seeds"]
        b = blocks["concat"][str(peak_L)]["auroc_mean_across_seeds"]
        c = blocks["residualized"][str(peak_L)]["auroc_mean_across_seeds"]
        pub_a = pub_pp[pair_key]["last_token_states"][str(peak_L)]["auroc_mean_across_seeds"]
        summary[pair_key] = {
            "peak_layer": peak_L,
            "hidden_only": round(a, 4),
            "hidden_only_published": round(pub_a, 4),
            "consistency_delta_vs_published": round(a - pub_a, 5),
            "concat": round(b, 4),
            "concat_delta_vs_hidden": round(b - a, 5),
            "residualized": round(c, 4),
            "residualized_delta_vs_hidden": round(c - a, 5),
            "surface_bar": bars[pair_key],
            "residualized_minus_surface_bar": round(c - bars[pair_key], 4),
        }
        print(f"  {pair_key:26s} L{peak_L:2d}  A={a:.4f} (pub {pub_a:.4f})  "
              f"B={b:.4f} (d {b-a:+.4f})  C={c:.4f} (d {c-a:+.4f})")
    out["peak_layer_summary"] = summary

    # Condition D (surface-only) re-emitted with per-seed values + CIs for
    # self-containedness, copied from the published baseline JSON.
    surf_path = Path(f"results/surface_baselines_v3_{model}.json")
    if surf_path.exists():
        surf = json.load(open(surf_path))
        out["surface_only"] = {
            pk: surf["per_pair"][pk]["features"]["all_surface"]
            for pk in out["per_pair"] if pk in surf.get("per_pair", {})
        }
        out["_meta"]["surface_only_source"] = str(surf_path)
    else:
        out["_meta"]["surface_only_source"] = f"{surf_path} not found; Table-1 scalar bars only"
    return out, (df, npz, regimes, H, static12, sembs)


def run_shuffled_residualized(model, loaded, n_jobs):
    """Shuffled-label null on condition C at the six peak layers."""
    from joblib import Parallel, delayed
    peaks, _ = load_peaks_bars(model)
    df, npz, regimes, H, static12, sembs = loaded
    agg = {"per_shuffle_seed": {}, "_meta": {
        "model": model, "condition": "residualized",
        "peak_layers": peaks, "shuffle_seeds": list(C.SHUFFLE_SEEDS),
        "shuffle_algorithm": "np.random.default_rng(seed).permutation(regimes)",
        "protocol": "fixed hyperparameters; see code/common.py",
    }}
    for s in C.SHUFFLE_SEEDS:
        y_shuf_all = np.random.default_rng(s).permutation(regimes)
        jobs = []
        for r1, r2 in C.PAIRS:
            pair_key = f"{r1}_vs_{r2}"
            mask = (y_shuf_all == r1) | (y_shuf_all == r2)
            idx = np.where(mask)[0]
            y = (y_shuf_all[mask] == r1).astype(np.int64)
            jobs.append((pair_key, peaks[pair_key], idx, y))
        cells = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(eval_cell)(H[:, L, :], static12, sembs, y_shuf_all, idx, y,
                               "residualized")
            for (_, L, idx, y) in jobs
        )
        agg["per_shuffle_seed"][str(s)] = {
            pk: {"peak_layer": L, **cell}
            for (pk, L, _, _), cell in zip(jobs, cells)
        }
        mx = max(c["auroc_mean_across_seeds"] for c in cells)
        print(f"  shuffle {s}: max residualized peak-layer AUROC {mx:.4f}")
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    # llama8b / qwen32b replications
    # ().
    ap.add_argument("--model", default="qwen7b", choices=["qwen7b", "llama8b", "qwen32b"])
    ap.add_argument("--n-jobs", type=int, default=8)
    args = ap.parse_args()
    if not args.execute:
        print(__doc__)
        return
    files = [C.NPZ[(args.model, "base")], C.BASE_CSV, f"results/probe_auroc_v3_{args.model}.json"]
    out, loaded = run(args.model, args.n_jobs)
    out["_provenance"] = C.provenance(files=files)
    C.save_versioned(out, Path("results") / f"probe_covariate_adjusted_v3_{args.model}.json")
    shuf = run_shuffled_residualized(args.model, loaded, args.n_jobs)
    shuf["_provenance"] = C.provenance(files=files)
    C.save_versioned(shuf, Path("results") / f"probe_covariate_adjusted_v3_{args.model}_shuffled.json")


if __name__ == "__main__":
    main()
