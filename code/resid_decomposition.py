#!/usr/bin/env python3
"""
residualization decomposition ladder + matched-subset probing.

Fixed hyperparameters: L2 logistic regression (C=1.0, lbfgs, max_iter=2000) on raw fp32
last-token states; 5-fold cross-validation x 5 seeds (1234, 4567, 7890, 2345, 5678);
bootstrap 95% CI (1000 resamples); shuffled-label control seeds 9001-9005.

Part A: nested residualization (R_text / R_text_cos / R_text_ent / R_full)
        at each pair's in-window peak layer, all 3 models.
Part B: 1:1 greedy NN covariate matching on standardized [wc, ne, token]
        with exact question-form constraint (caliper 1.0), then hidden-only
        probe + TEXT-balance check on the matched subset.

Usage:
  python3 code/resid_decomposition.py --execute [--n-jobs 14]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import common as C
from covariate_probe import load_peaks_bars

MODELS = ("qwen7b", "llama8b", "qwen32b")
# Column layout of the 13-dim covariate matrix [static12 ; cos]:
#   0 wc, 1 ne, 2-9 form onehot, 10 token count, 11 entropy, 12 cos
TEXT_COLS = list(range(0, 11))
ENT_COL = [11]
COS_COL = [12]
LADDER = {
    "R_text": TEXT_COLS,
    "R_text_cos": TEXT_COLS + COS_COL,
    "R_text_ent": TEXT_COLS + ENT_COL,
    "R_full": TEXT_COLS + ENT_COL + COS_COL,
}
FLAGGED = [("llama8b", "known_vs_misleading"), ("qwen32b", "known_vs_unknown")]


def oof_resid_subset(H_layer, static12, sembs, regimes_all, pair_indices, y, seed, cols):
    """Pooled OOF P(class=1): probe on hidden states residualized (train-fold
    OLS) against the covariate columns in `cols`. Mirrors R2 condition C."""
    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=C.N_CV_FOLDS, shuffle=True, random_state=seed)
    Xh = H_layer[pair_indices].astype(np.float64)
    oof = np.zeros(len(y), dtype=np.float64)
    for tr, te in cv.split(Xh, y):
        te_orig = pair_indices[te]
        cos = C.fold_sentence_cos(sembs, regimes_all, pair_indices, te_orig)
        cov_full = np.concatenate([static12[pair_indices], cos], axis=1)[:, cols]
        cov_z = C.zscore_train(cov_full[tr], cov_full)
        A_tr = np.concatenate([cov_z[tr], np.ones((len(tr), 1))], axis=1)
        A_te = np.concatenate([cov_z[te], np.ones((len(te), 1))], axis=1)
        beta, *_ = np.linalg.lstsq(A_tr, Xh[tr], rcond=None)
        lr = C.make_lr(seed).fit(Xh[tr] - A_tr @ beta, y[tr])
        oof[te] = lr.predict_proba(Xh[te] - A_te @ beta)[:, 1]
    return oof


def eval_ladder_cell(H_layer, static12, sembs, regimes, pair_indices, y, cols):
    points, los, his = [], [], []
    for seed in C.SEEDS:
        oof = oof_resid_subset(H_layer, static12, sembs, regimes, pair_indices, y, seed, cols)
        points.append(C.auroc_binary(y, oof))
        lo, hi = C.bootstrap_ci(C.auroc_binary, np.asarray(y), oof, seed)
        los.append(lo)
        his.append(hi)
    return {"auroc_mean_across_seeds": float(np.nanmean(points)),
            "auroc_per_seed": [float(p) for p in points],
            "auroc_ci_low_across_seeds": float(np.nanmean(los)),
            "auroc_ci_high_across_seeds": float(np.nanmean(his))}


def part_a(n_jobs):
    from joblib import Parallel, delayed
    out = {}
    for model in MODELS:
        peaks, bars = load_peaks_bars(model)
        df, npz, regimes = C.load_aligned(model, "base")
        H = npz["last_token_states"].astype(np.float32)
        static12 = C.static_covariates(df, npz)
        sembs = C.encode_sentences(df["question"].astype(str).tolist())
        r2 = json.load(open(f"results/probe_covariate_adjusted_v3_{model}.json"))

        jobs = []
        for r1, r2p in C.PAIRS:
            pk = f"{r1}_vs_{r2p}"
            mask = (regimes == r1) | (regimes == r2p)
            idx = np.where(mask)[0]
            y = (regimes[mask] == r1).astype(np.int64)
            for cond, cols in LADDER.items():
                jobs.append((pk, cond, cols, idx, y))
        L_of = {pk: peaks[pk] for pk in set(j[0] for j in jobs)}
        cells = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(eval_ladder_cell)(H[:, L_of[pk], :], static12, sembs, regimes, idx, y, cols)
            for (pk, cond, cols, idx, y) in jobs
        )
        m_out = {}
        for (pk, cond, _, _, _), cell in zip(jobs, cells):
            m_out.setdefault(pk, {"peak_layer": peaks[pk], "surface_bar": bars[pk]})
            m_out[pk][cond] = round(cell["auroc_mean_across_seeds"], 4)
            m_out[pk][cond + "_detail"] = cell
        # consistency vs R2 condition C + attribution deltas
        for pk, blk in m_out.items():
            r2c = r2["peak_layer_summary"][pk]["residualized"]
            blk["r2_condition_c"] = r2c
            blk["r_full_minus_r2"] = round(blk["R_full"] - r2c, 5)
            blk["drop_text"] = round(blk["R_text"] - blk["R_full"], 4)  # >0: text-only keeps more
            blk["attrib_cos"] = round(blk["R_text"] - blk["R_text_cos"], 4)
            blk["attrib_ent"] = round(blk["R_text"] - blk["R_text_ent"], 4)
            blk["r_text_minus_bar"] = round(blk["R_text"] - blk["surface_bar"], 4)
            print(f"{model} {pk:26s} L{blk['peak_layer']:2d} bar={blk['surface_bar']:.3f} | "
                  f"R_text={blk['R_text']:.3f} (vs bar {blk['r_text_minus_bar']:+.3f}) "
                  f"R_t+cos={blk['R_text_cos']:.3f} R_t+ent={blk['R_text_ent']:.3f} "
                  f"R_full={blk['R_full']:.3f} (R2 {r2c:.3f})")
        out[model] = m_out
    return out


def match_pairs(feats, forms, y):
    """1:1 greedy NN matching without replacement; exact form; caliper 1.0.

    feats: (n, 3) standardized [wc, ne, token]; y binary; returns matched row indices.
    Deterministic: class-1 rows in dataset order, nearest class-0 candidate,
    ties by dataset order.
    """
    ones = [i for i in range(len(y)) if y[i] == 1]
    zeros = [i for i in range(len(y)) if y[i] == 0]
    used = set()
    matched = []
    for i in ones:
        best, best_d = None, None
        for j in zeros:
            if j in used or forms[i] != forms[j]:
                continue
            d = float(np.linalg.norm(feats[i] - feats[j]))
            if best is None or d < best_d - 1e-12:
                best, best_d = j, d
        if best is not None and best_d <= 1.0:
            used.add(best)
            matched.extend([i, best])
    return sorted(matched)


def eval_matched_cell(model, pk, H, static12, regimes, df, peak_L):
    r1, r2p = pk.split("_vs_")
    mask = (regimes == r1) | (regimes == r2p)
    idx = np.where(mask)[0]
    y = (regimes[mask] == r1).astype(np.int64)
    numeric = static12[idx][:, [0, 1, 10]]  # wc, ne, token count
    z = (numeric - numeric.mean(axis=0)) / (numeric.std(axis=0) + 1e-9)
    forms = df["question_form"].astype(str).str.lower().to_numpy()[idx]

    res = {"pair": pk, "peak_layer": int(peak_L)}
    sel = match_pairs(z, forms, y)
    used_fallback = False
    if len(sel) < 30:  # <15 per side
        used_fallback = True
        sel = match_pairs(z, np.zeros(len(y)), y)  # drop form constraint
    res["exact_form_constraint"] = not used_fallback
    res["n_matched_rows"] = len(sel)
    if len(sel) < 30:
        res["verdict"] = "underpowered (<15 per side even after fallback); not reported"
        return res

    sub_idx = idx[sel]
    y_sub = y[sel]
    text_sub = static12[sub_idx][:, TEXT_COLS[:11]]
    Xh_sub = H[sub_idx][:, peak_L, :].astype(np.float64)

    def cv_auroc(X):
        pts = []
        for seed in C.SEEDS:
            oof = C.oof_binary(X, y_sub, seed)
            pts.append(C.auroc_binary(y_sub, oof))
        return float(np.nanmean(pts)), [float(p) for p in pts]

    bal_mean, bal_seeds = cv_auroc(text_sub)
    probe_mean, probe_seeds = cv_auroc(Xh_sub)
    lo, hi = C.bootstrap_ci(C.auroc_binary, y_sub,
                            C.oof_binary(Xh_sub, y_sub, C.SEEDS[0]), C.SEEDS[0])
    res.update({"text_balance_auroc": round(bal_mean, 4),
                "balance_ok_le_0.60": bool(bal_mean <= 0.60),
                "probe_auroc_matched": round(probe_mean, 4),
                "probe_auroc_per_seed": probe_seeds,
                "probe_ci_seed1234": [round(lo, 4), round(hi, 4)]})
    print(f"{model} {pk:26s} matched n={len(sel)} (form={'exact' if not used_fallback else 'relaxed'}) "
          f"balance={bal_mean:.3f} probe={probe_mean:.3f}")
    return res


def part_b():
    out = {}
    for model in MODELS:
        peaks, _ = load_peaks_bars(model)
        df, npz, regimes = C.load_aligned(model, "base")
        H = npz["last_token_states"].astype(np.float32)
        static12 = C.static_covariates(df, npz)
        out[model] = {pk: eval_matched_cell(model, pk, H, static12, regimes, df, peaks[pk])
                      for pk in [f"{a}_vs_{b}" for a, b in C.PAIRS]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=14)
    args = ap.parse_args()
    if not args.execute:
        print(__doc__)
        return
    files = [C.NPZ[(m, "base")] for m in MODELS] + [C.BASE_CSV]
    meta = {"ladder": {k: v for k, v in LADDER.items()},
            "covariate_layout": "0 wc, 1 ne, 2-9 form, 10 token, 11 entropy, 12 cos (fold-safe)",
            "flagged_cells": FLAGGED,
            "seeds": list(C.SEEDS), "n_bootstrap": C.N_BOOTSTRAP,
            "bootstrap_rng": "np.random.default_rng(cv_seed * 1_000_003),  convention",
            "protocol": "fixed hyperparameters; see code/common.py"}

    print("== Part A: decomposition ladder ==")
    a = part_a(args.n_jobs)
    a["_meta"] = meta
    a["_provenance"] = C.provenance(files=files)
    C.save_versioned(a, Path("results/probe_resid_decomposition_v3.json"))

    print("== Part B: matched subsets ==")
    b = part_b()
    b["_meta"] = {**meta, "matching": "1:1 greedy NN, standardized [wc,ne,token], exact form, "
                                      "caliper 1.0, deterministic; fallback drops form if n<30 rows"}
    b["_provenance"] = C.provenance(files=files)
    C.save_versioned(b, Path("results/probe_matched_subset_v3.json"))


if __name__ == "__main__":
    main()
