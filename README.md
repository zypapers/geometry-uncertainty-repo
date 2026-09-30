# The Geometry of Uncertainty in Language Model Representations — code and data

Code, data, and result files accompanying the AACL-IJCNLP 2026 paper
*The Geometry of Uncertainty in Language Model Representations* (Zy Li, Seattle
University). This repository lets a reader regenerate the pre-generation hidden
states from the prompts and public model checkpoints, re-run the probing analyses
on them, inspect every reported number, and audit the dataset.

## What is here

```
code/        extract_hidden_states.py   regenerate hidden states from the prompts + public checkpoints
             analysis scripts: linear probes and surface baselines (base set), their
             group-aware paraphrase-set versions (*_paraphrase.py), four-way probe,
             covariate-adjusted probes, residualization decomposition, robustness, and
             the two separation-criterion aggregators
data/        prompts.csv                the 200-prompt diagnostic set (id, type, question,
                                        gold answer, sub-type, per-model screening records)
             paraphrases.csv            the 558-prompt paraphrase extension (186 groups: each base
                                        prompt plus two paraphrases; is_base / paraphrase_group_id columns)
             unknown_paraphrase_exclusions.csv  the Unknown paraphrases that failed the screening filter,
                                        with per-model correct counts (the 14 excluded groups)
             surface_balance_summary.json  per-pair surface AUROC and per-type surface-feature
                                        spreads of the final base set
             robustness_exclusions.json  prompt-id sets for the robustness sensitivity check
             annotation_audit/          the two blind annotators' sheets, the item-id map, and the
                                        verbatim instructions given to the annotators (paper Appendix H)
results/     the JSON output files behind every reported number: pairwise probes and
             shuffled-label nulls, surface / TF-IDF / sentence-encoder baselines, four-way
             probe and baselines, covariate-adjusted probes and residualization
             decomposition, label perturbation, in-sample demonstration, sub-type
             analyses, middle-third robustness, exclusion robustness, annotation-audit
             statistics, and (results/behavior_labels/) the per-prompt behavior labels
             and type x behavior crosstabs (paper Table 4)
requirements.txt
```

## Dataset

`data/prompts.csv` contains 200 hand-authored prompts, 50 per type (Known,
Unknown, Ambiguous, Misleading), each with its gold answer, sub-type, and the
per-model screening records (`model_pretest_*`) that operationalize the
Known/Unknown labels. Known items are Wikipedia-verifiable; Unknown items are
defined by screening failure across the three target models; Ambiguous items
carry their enumerated readings; Misleading items pair the false premise with
its correction. Sourcing, screening, and sub-type definitions are described in
the paper (Section 3 and Appendices A–B).

## Hidden states

The probes run on pre-generation last-token hidden states. These are not bundled
(they are large and hardware-specific); regenerate them from the prompts and the
public model checkpoints:

```
CUDA_VISIBLE_DEVICES=0 python code/extract_hidden_states.py \
    --input data/prompts.csv --model Qwen/Qwen2.5-7B-Instruct \
    --output-name hidden_states_qwen7b
```

This writes `hidden_states/hidden_states_qwen7b.npz` (BF16, sdpa attention, chat
template with `add_generation_prompt=True`), ~2 GPU-hours per model. Note that
BF16 forward passes are hardware- and library-version-sensitive, so regenerated
states are not bit-identical across setups; the reported AUROCs are stable to
that variation (they reproduce to within ~0.002).

## Reproducing the results

1. `pip install -r requirements.txt`
2. Regenerate the hidden states (see "Hidden states" above). Each run writes
   `hidden_states/hidden_states_<model>.npz`:
   ```
   CUDA_VISIBLE_DEVICES=0 python code/extract_hidden_states.py \
       --input data/prompts.csv --model Qwen/Qwen2.5-7B-Instruct \
       --output-name hidden_states_qwen7b
   ```
   For the paraphrase extension, extract from `data/paraphrases.csv` with
   `--output-name hidden_states_paraphrases_qwen7b`. The analysis scripts expect
   `hidden_states/hidden_states_<model>.npz` (base) and
   `hidden_states/hidden_states_paraphrases_<model>.npz` (paraphrases), with
   `<model>` one of `qwen7b`, `llama8b`, `qwen32b`.
3. Run the analyses on the regenerated states, e.g.:
   - Pairwise type separation (base): `python code/linear_probes.py --execute --npz hidden_states/hidden_states_qwen7b.npz --output-name probe_auroc_v3_qwen7b --shuffle-seeds 9001,9002,9003,9004,9005`
   - Surface baselines (base): `python code/surface_baselines.py --execute --npz hidden_states/hidden_states_qwen7b.npz --output-name surface_baselines_v3_qwen7b --shuffle-seeds 9001,9002,9003,9004,9005`
   - Separation criterion (base and paraphrase): `python code/separation_criterion.py --help` and `python code/separation_criterion_paraphrase.py --help` show the exact invocations; they read the probe and baseline JSONs above and write `results/separation_gate_*.json`
   - Paraphrase extension, group-aware: `python code/linear_probes_paraphrase.py --execute --npz hidden_states/hidden_states_paraphrases_qwen7b.npz --output-name probe_auroc_v3paraphrases_qwen7b` and `python code/surface_baselines_paraphrase.py --execute --npz hidden_states/hidden_states_paraphrases_qwen7b.npz --slug qwen7b --output-name surface_baselines_v3paraphrases_qwen7b`
   - Four-way classification: `python code/fourway_probe.py --model qwen7b --dataset base --execute`
   - Covariate-adjusted probes: `python code/covariate_probe.py --execute`
   - Robustness sensitivity check: `python code/robustness_exclusion.py --execute`

Every script writes a JSON matching the corresponding file in `results/`. The
probe is deliberately the simplest variant (L2 logistic regression, C=1.0, no
preprocessing, fixed seeds); cross-validation is 5-fold with 5 seeds and
bootstrap 95% confidence intervals.

## Robustness sensitivity check

`robustness_exclusion.py` re-runs the four-way and pairwise classification with
progressively larger sets of prompts excluded (`data/robustness_exclusions.json`),
namely prompts that are borderline exemplars of their type. The separation is
stable: minimum pairwise AUROC 0.990 / 0.984 / 0.993 across the three models with
all 36 borderline prompts excluded, versus 0.987 / 0.981 / 0.994 on the full set
(`results/robustness_exclusion_v3_r2.json`; in-window peaks).

## Notes

- Hidden states are stored in fp32; no prompt text is included in the archives.
- Result JSONs have been scrubbed of machine-specific provenance (hostnames, paths).
- License: dataset and code are released under a permissive license (see LICENSE).
- The `_meta` fields of the result files record the configuration each analysis was run with.

## Citation

```
@inproceedings{li2026geometry,
  title     = {The Geometry of Uncertainty in Language Model Representations},
  author    = {Li, Zy},
  booktitle = {Proceedings of the 5th Asia-Pacific Chapter of the Association for
               Computational Linguistics and the 15th International Joint Conference
               on Natural Language Processing (AACL-IJCNLP 2026)},
  year      = {2026}
}
```
