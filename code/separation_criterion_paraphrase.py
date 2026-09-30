#!/usr/bin/env python3
"""Paraphrase-set separation criterion (pairwise, group-aware inputs).

A layer qualifies for a pair when all three conditions hold at that layer:
  1. real probe AUROC >= required threshold, where the threshold is
     max(base_set, paraphrase_set) surface baseline + 0.05 and the paraphrase-set
     baseline is the maximum over the three models' surface-baseline runs
  2. real probe AUROC - mean shuffled-label AUROC >= 0.10
  3. every individual shuffled-label seed stays below the required threshold
A pair passes with >= 3 consecutive or >= 4 non-consecutive qualifying layers in the
model-specific window (Qwen2.5-7B L14-L28, Llama-3.1-8B L16-L32, Qwen2.5-32B L32-L64).

Reads:  results/probe_auroc_v3paraphrases_<model>.json, its _shuffled companion, and the
        three results/surface_baselines_v3paraphrases_<model>.json files
Writes: results/separation_gate_v3paraphrases_<model>.json

Usage:
  python3 code/separation_criterion_paraphrase.py --probes results/probe_auroc_v3paraphrases_qwen7b.json --shuffled results/probe_auroc_v3paraphrases_qwen7b_shuffled.json --surface results/surface_baselines_v3paraphrases_qwen7b.json results/surface_baselines_v3paraphrases_llama8b.json results/surface_baselines_v3paraphrases_qwen32b.json --layer-range 14 28 --output results/separation_gate_v3paraphrases_qwen7b.json
"""


from __future__ import annotations

import argparse
import json
from pathlib import Path


PRIMARY_REPRESENTATION = "last_token_states"
MARGIN = 0.05
SHUFFLED_GAP_MIN = 0.10
N_PAIRS_REQUIRED = 2

# Fixed base-set surface baselines (maximum over the three models' surface classifiers).
V3_LOCK_SURFACE_BARS = {
    "known_vs_unknown":         0.763,
    "known_vs_ambiguous":       0.833,
    "known_vs_misleading":      0.867,
    "unknown_vs_ambiguous":     0.753,
    "unknown_vs_misleading":    0.785,
    "ambiguous_vs_misleading":  0.718,
}

PAIRS = [
    ("known", "unknown"),
    ("known", "ambiguous"),
    ("known", "misleading"),
    ("unknown", "ambiguous"),
    ("unknown", "misleading"),
    ("ambiguous", "misleading"),
]


def longest_consecutive_run(sorted_layers):
    if not sorted_layers:
        return 0
    best = current = 1
    for i in range(1, len(sorted_layers)):
        if sorted_layers[i] == sorted_layers[i-1] + 1:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def compute_paraphrase_surface_bar(surface_jsons):
    """Compute paraphrase-set surface bar = max across 3 models per pair."""
    per_pair = {}
    for pk in [f"{a}_vs_{b}" for a, b in PAIRS]:
        vals = []
        for sj in surface_jsons:
            v = sj["per_pair"][pk]["auroc_mean_across_seeds"]
            vals.append(v)
        per_pair[pk] = max(vals)
    return per_pair


def evaluate(probe_path: Path, shuffled_path: Path,
             surface_paths: list[Path], layer_range: tuple[int, int],
             out_path: Path):
    probes = json.loads(probe_path.read_text())
    shuffled = json.loads(shuffled_path.read_text())
    surface_jsons = [json.loads(p.read_text()) for p in surface_paths]

    paraphrase_bars = compute_paraphrase_surface_bar(surface_jsons)
    effective_bars = {
        pk: max(V3_LOCK_SURFACE_BARS[pk], paraphrase_bars[pk])
        for pk in V3_LOCK_SURFACE_BARS
    }

    layer_lo, layer_hi = layer_range
    layers = list(range(layer_lo, layer_hi + 1))

    per_pair = []
    passing = []
    for r1, r2 in PAIRS:
        pk = f"{r1}_vs_{r2}"
        surface_bar = effective_bars[pk]
        required = surface_bar + MARGIN

        # Real probe AUROC per layer
        real_layers = probes["per_pair_per_layer"][pk][PRIMARY_REPRESENTATION]
        real_aurocs = {int(L): real_layers[L]["auroc_mean_across_seeds"] for L in real_layers}

        # Shuffled probe AUROC per layer per seed
        shuf_per_seed = shuffled["per_shuffle_seed"]
        seed_ids = sorted(shuf_per_seed.keys())
        first_seed = seed_ids[0]
        layer_keys = list(shuf_per_seed[first_seed]["per_pair_per_layer"][pk][PRIMARY_REPRESENTATION].keys())

        shuf_per_layer_seed = {}  # L → list of per-seed AUROCs
        for L_str in layer_keys:
            L = int(L_str)
            shuf_per_layer_seed[L] = []
            for sid in seed_ids:
                shuf_per_layer_seed[L].append(
                    shuf_per_seed[sid]["per_pair_per_layer"][pk][PRIMARY_REPRESENTATION][L_str]["auroc_mean_across_seeds"]
                )
        shuf_mean = {L: sum(vs)/len(vs) for L, vs in shuf_per_layer_seed.items()}

        # Per-layer evaluation
        layer_results = []
        qualifying_layers = []
        for L in layers:
            real = real_aurocs.get(L)
            shuf_seeds = shuf_per_layer_seed.get(L)
            shuf_m = shuf_mean.get(L)
            if real is None or shuf_seeds is None:
                continue
            cond_margin = real >= required
            real_vs_shuf = real - shuf_m
            cond_gap = real_vs_shuf >= SHUFFLED_GAP_MIN
            # Empirical-null: every individual shuffle seed must be below required
            cond_empirical_null = all(s < required for s in shuf_seeds)
            spurious_seeds = [seed_ids[i] for i, s in enumerate(shuf_seeds) if s >= required]
            qualifies = cond_margin and cond_gap and cond_empirical_null
            layer_results.append({
                "layer": L,
                "real": real,
                "shuffled_mean": shuf_m,
                "shuffled_per_seed": dict(zip(seed_ids, shuf_seeds)),
                "real_minus_shuffled": real_vs_shuf,
                "margin_condition_pass": cond_margin,
                "gap_condition_pass": cond_gap,
                "empirical_null_pass": cond_empirical_null,
                "spurious_seeds_at_layer": spurious_seeds,
                "qualifies": qualifies,
            })
            if qualifies:
                qualifying_layers.append(L)

        longest = longest_consecutive_run(qualifying_layers)
        n_qual = len(qualifying_layers)
        consec_pass = longest >= 3
        nonconsec_pass = n_qual >= 4
        pair_passes = consec_pass or nonconsec_pass

        peak_real_layer = max(real_aurocs, key=real_aurocs.get) if real_aurocs else None
        peak_real = real_aurocs[peak_real_layer] if peak_real_layer is not None else None

        per_pair.append({
            "pair": pk,
            "v3_lock_surface_bar": V3_LOCK_SURFACE_BARS[pk],
            "paraphrase_surface_bar": paraphrase_bars[pk],
            "effective_surface_bar": surface_bar,
            "required_probe_auroc": required,
            "peak_probe_auroc": peak_real,
            "peak_layer": peak_real_layer,
            "qualifying_layers": qualifying_layers,
            "n_qualifying_layers": n_qual,
            "longest_consecutive_run": longest,
            "consecutive_rule_pass": consec_pass,
            "nonconsecutive_rule_pass": nonconsec_pass,
            "pair_passes": pair_passes,
            "per_layer": layer_results,
        })
        if pair_passes:
            passing.append(pk)

    gate_clears = len(passing) >= N_PAIRS_REQUIRED

    result = {
        "primary_representation": PRIMARY_REPRESENTATION,
        "layer_range_inclusive": list(layer_range),
        "margin": MARGIN,
        "shuffled_gap_min": SHUFFLED_GAP_MIN,
        "n_pairs_required": N_PAIRS_REQUIRED,
        "surface_bars": {
            "v3_lock": V3_LOCK_SURFACE_BARS,
            "paraphrase_set": paraphrase_bars,
            "effective": effective_bars,
            "rule": "max(base_set, paraphrase_set) per pair",
        },
        "n_pairs_pass": len(passing),
        "pairs_pass": passing,
        "gate_clears": gate_clears,
        "per_pair": per_pair,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str))

    print(f"\n=== paraphrase-set separation criterion ===")
    print(f"Layer window: L{layer_range[0]}-L{layer_range[1]}")
    print(f"Per-layer qualifying = (real ≥ surf+{MARGIN}) AND (real - mean_shuf ≥ {SHUFFLED_GAP_MIN}) AND (every shuf seed < surf+{MARGIN})\n")
    print(f"{'pair':30s} {'v3_bar':>8s} {'para':>8s} {'eff':>7s} {'req':>7s} {'peak_real':>10s} {'n_qual':>7s} {'longest':>9s} {'pass':>6s}")
    for p in per_pair:
        flag = "✓" if p["pair_passes"] else "✗"
        peak = f"{p['peak_probe_auroc']:.4f}" if p["peak_probe_auroc"] is not None else "—"
        print(f"  {p['pair']:28s} {p['v3_lock_surface_bar']:>8.3f} {p['paraphrase_surface_bar']:>8.3f} {p['effective_surface_bar']:>7.3f} {p['required_probe_auroc']:>7.3f} {peak:>10s} {p['n_qualifying_layers']:>7d} {p['longest_consecutive_run']:>9d} {flag:>6s}")
    print(f"\nPairs passing: {len(passing)} / 6 — {passing}")
    print(f"CRITERION MET (>= 2 of 6 pairs): {gate_clears}")
    print(f"\nWrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", required=True)
    ap.add_argument("--shuffled", required=True)
    ap.add_argument("--surface", nargs=3, required=True,
                    help="3 surface_baselines JSONs (qwen7b llama8b qwen32b) for max-across-3 paraphrase bar")
    ap.add_argument("--layer-range", type=int, nargs=2, metavar=("LOW","HIGH"), required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    evaluate(
        Path(args.probes), Path(args.shuffled),
        [Path(p) for p in args.surface],
        tuple(args.layer_range), Path(args.output),
    )


if __name__ == "__main__":
    main()
