#!/usr/bin/env python3
"""Base-set separation criterion (pairwise).

A layer qualifies for a pair when both conditions hold at that layer:
  1. real probe AUROC >= surface baseline + 0.05
  2. real probe AUROC - mean shuffled-label AUROC (5 seeds, 9001-9005) >= 0.10
A pair passes with >= 3 consecutive or >= 4 non-consecutive qualifying layers in the
model-specific window (Qwen2.5-7B L14-L28, Llama-3.1-8B L16-L32, Qwen2.5-32B L32-L64).
The surface baseline per pair is fixed: the maximum over the three models' surface
classifiers on the base set (SURFACE_BASELINES below).

Reads:  results/probe_auroc_v3_<model>.json and results/probe_auroc_v3_<model>_shuffled.json
Writes: results/separation_gate_v3_<model>.json

Usage:
  python3 code/separation_criterion.py --probes results/probe_auroc_v3_qwen7b.json --shuffled results/probe_auroc_v3_qwen7b_shuffled.json --layer-range 14 28 --output results/separation_gate_v3_qwen7b.json
"""


from __future__ import annotations

import argparse
import json
from pathlib import Path

PRIMARY_REPRESENTATION = "last_token_states"
MARGIN = 0.05
SHUFFLED_GAP_MIN = 0.10
N_PAIRS_REQUIRED = 2

# Fixed base-set surface baselines: maximum over the three models' surface classifiers
# (max-across-3-models, full 6-feature all_surface).
V3_SURFACE_BARS = {
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


def evaluate(probe_path: Path, shuffled_path: Path, layer_range: tuple[int, int], out_path: Path):
    probes = json.loads(probe_path.read_text())
    shuffled = json.loads(shuffled_path.read_text())
    layer_lo, layer_hi = layer_range
    layers = list(range(layer_lo, layer_hi + 1))

    per_pair = []
    passing = []
    for r1, r2 in PAIRS:
        pk = f"{r1}_vs_{r2}"
        surface_bar = V3_SURFACE_BARS[pk]
        required = surface_bar + MARGIN

        # Real-label probe AUROCs per layer
        real_layers = probes["per_pair_per_layer"][pk][PRIMARY_REPRESENTATION]
        real_aurocs = {int(L): real_layers[L]["auroc_mean_across_seeds"] for L in real_layers}

        # Shuffled probe AUROCs per layer — mean across the 5 shuffle seeds.
        # Format: shuffled["per_shuffle_seed"][seed]["per_pair_per_layer"][pk][rep][L]["auroc_mean_across_seeds"]
        # auroc_mean_across_seeds here is the mean across the LR-CV seeds at that
        # shuffle seed; we further average across the 5 shuffle seeds.
        shuf_per_seed = shuffled["per_shuffle_seed"]
        shuf_aurocs = {}
        # Get layer keys from the first shuffle seed
        first_seed = next(iter(shuf_per_seed))
        layer_keys = list(shuf_per_seed[first_seed]["per_pair_per_layer"][pk][PRIMARY_REPRESENTATION].keys())
        for L_str in layer_keys:
            seeds_aurocs = []
            for seed in shuf_per_seed:
                seeds_aurocs.append(
                    shuf_per_seed[seed]["per_pair_per_layer"][pk][PRIMARY_REPRESENTATION][L_str]["auroc_mean_across_seeds"]
                )
            shuf_aurocs[int(L_str)] = sum(seeds_aurocs) / len(seeds_aurocs)

        # Per-layer evaluation
        layer_results = []
        qualifying_layers = []
        for L in layers:
            real = real_aurocs.get(L)
            shuf = shuf_aurocs.get(L)
            if real is None or shuf is None:
                continue
            cond_margin = real >= required
            real_vs_shuf = real - shuf
            cond_gap = real_vs_shuf >= SHUFFLED_GAP_MIN
            qualifies = cond_margin and cond_gap
            layer_results.append({
                "layer": L,
                "real": real,
                "shuffled_mean": shuf,
                "real_minus_shuffled": real_vs_shuf,
                "margin_condition_pass": cond_margin,
                "gap_condition_pass": cond_gap,
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
            "surface_bar": surface_bar,
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
        "surface_bars_source": "fixed per pair: maximum over the three models' surface-feature classifiers on the base set",
        "n_pairs_pass": len(passing),
        "pairs_pass": passing,
        "gate_clears": gate_clears,
        "per_pair": per_pair,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str))

    # Print summary
    print(f"\n=== base-set separation criterion ===")
    print(f"Layer window: L{layer_range[0]}-L{layer_range[1]}")
    print(f"Pair-local rule: ≥3 consecutive OR ≥4 non-consecutive qualifying layers")
    print(f"Per-layer qualifying = (real ≥ surface+{MARGIN}) AND (real - mean_shuffled ≥ {SHUFFLED_GAP_MIN})\n")
    print(f"{'pair':30s} {'surf_bar':>9s} {'req':>7s} {'peak_real':>10s} {'n_qual':>7s} {'longest':>9s} {'pass':>6s}")
    for p in per_pair:
        flag = "✓" if p["pair_passes"] else "✗"
        peak = f"{p['peak_probe_auroc']:.4f}" if p["peak_probe_auroc"] is not None else "—"
        print(f"  {p['pair']:28s} {p['surface_bar']:>9.3f} {p['required_probe_auroc']:>7.3f} {peak:>10s} {p['n_qualifying_layers']:>7d} {p['longest_consecutive_run']:>9d} {flag:>6s}")
    print(f"\nPairs passing: {len(passing)} / 6 — {passing}")
    print(f"CRITERION MET (>= 2 of 6 pairs): {gate_clears}")
    print(f"\nWrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", required=True, help="Real-label probe AUROC JSON")
    ap.add_argument("--shuffled", required=True, help="Shuffled-label probe AUROC JSON (5 seeds)")
    ap.add_argument("--layer-range", type=int, nargs=2, metavar=("LOW", "HIGH"), required=True,
                    help="Inclusive layer window for this model (e.g. 14 28 for Qwen2.5-7B)")
    ap.add_argument("--output", required=True, help="Output JSON path")
    args = ap.parse_args()
    evaluate(Path(args.probes), Path(args.shuffled), tuple(args.layer_range), Path(args.output))


if __name__ == "__main__":
    main()
