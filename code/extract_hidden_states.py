#!/usr/bin/env python3
"""
Hidden-state extraction for the four-type diagnostic set.

Reads the prompt CSV (data/prompts.csv; a behavior_label column is optional),
formats each prompt with the same chat template used for behavior labeling
(SHORT_SYSTEM + SHORT_USER_TEMPLATE, add_generation_prompt=True), runs a
single forward pass per prompt with output_hidden_states=True, and saves
prompt-only hidden states at the position where generation would begin.

Outputs:
  hidden_states/hidden_states_qwen7b.npz             — arrays for downstream geometry
  hidden_states/hidden_states_qwen7b_manifest.json  — provenance sidecar

Default usage (one-shot, 200 prompts on Qwen2.5-7B-Instruct):
  CUDA_VISIBLE_DEVICES=0 python3 extract_hidden_states.py

Defaults match the behavior-labeling generation setup, so the hidden
states correspond to "Qwen's prompt state at the moment a short answer
would begin". Do not change the chat template, dtype, or attn impl mid-run.

IMPORTANT: SHORT_SYSTEM and SHORT_USER_TEMPLATE below must match the
behavior-labeling setup verbatim; the released hidden states were extracted
with exactly this template.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x


# Chat template. See module docstring.
SHORT_SYSTEM = (
    "Answer the user's question as briefly as possible. "
    "If the question is ambiguous, say what needs clarification. "
    "If the question contains a false premise, correct it briefly. "
    "If you do not know, say you do not know."
)

SHORT_USER_TEMPLATE = "Question: {question}\nAnswer:"


def build_short_messages(question: str) -> List[Dict[str, str]]:
    """the labeling step message construction (no premise_check style)."""
    return [
        {"role": "system", "content": SHORT_SYSTEM},
        {"role": "user", "content": SHORT_USER_TEMPLATE.format(question=question)},
    ]


def load_model(model_id: str, dtype: str, attn_implementation: str):
    """Mirror the labeling step load_model: pin to GPU 0, sdpa default, BF16 default."""
    torch_dtype = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map={"": 0},
        attn_implementation=attn_implementation or "sdpa",
    )
    model.eval()
    return model, tokenizer


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_id() -> str:
    return "anonymized"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/prompts.csv",
        help="Prompt CSV (data/prompts.csv). Requires id, regime, question columns; "
             "behavior-label columns are optional.",
    )
    parser.add_argument(
        "--outdir",
        default="hidden_states",
        help="Directory for hidden_states_qwen7b.npz and the manifest sidecar.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model id.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model dtype. BF16 is the design choice used in the paper; do not quantize.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        help="sdpa (default) or flash_attention_2.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Optional: extract only the first N prompts (smoke test).",
    )
    parser.add_argument(
        "--output-name",
        default="hidden_states_qwen7b",
        help="Output base name (without extension).",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    npz_path = outdir / f"{args.output_name}.npz"
    manifest_path = outdir / f"{args.output_name}_manifest.json"

    # Read input.
    df = pd.read_csv(in_path)
    required_cols = {"id", "regime", "question"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(
            f"Input {in_path} is missing required columns: {sorted(missing)}."
        )

    # Behavior-label columns are optional. If absent, fill with empty strings;
    # they are used only for a downstream join, never for the type labels.
    optional_label_cols = ("behavior_label", "auto_behavior_label", "final_label_source")
    for c in optional_label_cols:
        if c not in df.columns:
            df[c] = ""
    has_labels = all(
        (df[c].astype(str) != "").any() for c in optional_label_cols
    )

    if args.limit is not None:
        df = df.head(args.limit).copy()

    n_prompts = len(df)
    print(f"Loading model: {args.model}  (dtype={args.dtype}, attn={args.attn_implementation})")
    model, tokenizer = load_model(args.model, args.dtype, args.attn_implementation)
    device = next(model.parameters()).device
    print(f"Model on device: {device}")

    # Inspect once to know layer count / hidden_dim / vocab_size.
    num_layers = model.config.num_hidden_layers + 1  # +1 for embedding output
    hidden_dim = model.config.hidden_size
    vocab_size = model.config.vocab_size
    print(f"num_layers={num_layers}  hidden_dim={hidden_dim}  vocab_size={vocab_size}")

    # Pre-allocate outputs.
    last_token_states = np.zeros((n_prompts, num_layers, hidden_dim), dtype=np.float32)
    mean_pooled_states = np.zeros((n_prompts, num_layers, hidden_dim), dtype=np.float32)
    next_token_logits = np.zeros((n_prompts, vocab_size), dtype=np.float16)  # logits as fp16 to halve storage
    prompt_token_counts = np.zeros((n_prompts,), dtype=np.int32)
    attention_mask_sums = np.zeros((n_prompts,), dtype=np.int32)

    prompt_ids = df["id"].astype(str).to_numpy()
    regimes = df["regime"].astype(str).to_numpy()
    behavior_labels = df["behavior_label"].astype(str).to_numpy()
    auto_behavior_labels = df["auto_behavior_label"].astype(str).to_numpy()
    final_label_sources = df["final_label_source"].astype(str).to_numpy()

    nan_count = 0

    for i, row in enumerate(tqdm(df.itertuples(index=False), total=n_prompts, desc="extracting")):
        messages = build_short_messages(row.question)
        # tokenize=False then re-tokenize with return_tensors so we have inputs and attention_mask
        # that match the labeling step generate_one path. Single-prompt batch — no padding required.
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer([text], return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        with torch.inference_mode():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        # out.hidden_states: tuple of (num_layers,) tensors, each shape (1, seq_len, hidden_dim).
        # Stack along a new layer dim: (1, num_layers, seq_len, hidden_dim).
        # Cast to fp32 BEFORE pooling so accumulation stays in fp32.
        hs = torch.stack(out.hidden_states, dim=1).float()  # (1, L, T, H)

        # attention_mask shape (1, seq_len). For mean pooling, broadcast across layer + hidden dims.
        mask = attention_mask.unsqueeze(1).unsqueeze(-1).float()  # (1, 1, T, 1)
        mask_sum = mask.sum(dim=2).clamp_min(1.0)  # (1, 1, 1)

        # Last non-pad token index.
        last_idx = (attention_mask.sum(dim=1) - 1).item()  # scalar

        last_token = hs[:, :, last_idx, :]  # (1, L, H)
        mean_pooled = (hs * mask).sum(dim=2) / mask_sum  # (1, L, H)

        # next-token logits: logits at last non-pad position.
        logits = out.logits[:, last_idx, :].float()  # (1, V)

        # Move to CPU and store.
        lt_np = last_token.detach().cpu().numpy()[0]
        mp_np = mean_pooled.detach().cpu().numpy()[0]
        lg_np = logits.detach().cpu().to(torch.float16).numpy()[0]

        last_token_states[i] = lt_np
        mean_pooled_states[i] = mp_np
        next_token_logits[i] = lg_np
        prompt_token_counts[i] = int(input_ids.shape[1])
        attention_mask_sums[i] = int(attention_mask.sum().item())

        # NaN check per prompt; cheap.
        if np.isnan(lt_np).any() or np.isnan(mp_np).any():
            nan_count += 1
            print(f"  WARN: NaN in hidden states for prompt {row.id}")

    # Save .npz.
    np.savez_compressed(
        npz_path,
        prompt_ids=prompt_ids,
        regimes=regimes,
        behavior_labels=behavior_labels,
        auto_behavior_labels=auto_behavior_labels,
        final_label_sources=final_label_sources,
        last_token_states=last_token_states,
        mean_pooled_states=mean_pooled_states,
        next_token_logits=next_token_logits,
        prompt_token_counts=prompt_token_counts,
        attention_mask_sums=attention_mask_sums,
    )

    # Sanity prints.
    print()
    print("=== Sanity checks ===")
    print(f"n_prompts = {n_prompts}")
    print(f"n_layers = {num_layers}")
    print(f"hidden_dim = {hidden_dim}")
    print(f"last_token_states.shape   = {last_token_states.shape}  dtype={last_token_states.dtype}")
    print(f"mean_pooled_states.shape  = {mean_pooled_states.shape}  dtype={mean_pooled_states.dtype}")
    print(f"next_token_logits.shape   = {next_token_logits.shape}  dtype={next_token_logits.dtype}")
    print(f"prompt_token_counts: min={prompt_token_counts.min()}  max={prompt_token_counts.max()}  median={int(np.median(prompt_token_counts))}")
    print(f"attention_mask_sums == prompt_token_counts: {bool((attention_mask_sums == prompt_token_counts).all())}")
    print(f"NaN count across hidden arrays: {nan_count}")
    print(f"npz path: {npz_path}  size={npz_path.stat().st_size} bytes")

    # Write provenance manifest sidecar.
    npz_sha = sha256_of_file(npz_path)
    input_sha = sha256_of_file(in_path)
    script_sha = sha256_of_file(Path(__file__))

    manifest = {
        "extractor_script": "extract_hidden_states.py",
        "extractor_script_sha256": script_sha,
        "run_id": "anonymized",
        "model": {
            "id": args.model,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
            "device": str(device),
            "num_layers_including_embedding": num_layers,
            "hidden_dim": hidden_dim,
            "vocab_size": vocab_size,
        },
        "input": {
            "path": str(in_path),
            "sha256": input_sha,
            "n_rows_used": n_prompts,
            "behavior_label_column": "behavior_label" if has_labels else "(not present)",
            "behavior_label_columns_present": has_labels,
        },
        "extraction_template": "short_temp0",
        "chat_template_marker": {
            "system_prompt_first_chars": SHORT_SYSTEM[:40],
            "user_template": SHORT_USER_TEMPLATE,
            "add_generation_prompt": True,
        },
        "output": {
            "path": str(npz_path),
            "sha256": npz_sha,
            "size_bytes": npz_path.stat().st_size,
            "arrays": {
                "prompt_ids":             {"shape": list(prompt_ids.shape),             "dtype": str(prompt_ids.dtype)},
                "regimes":                {"shape": list(regimes.shape),                "dtype": str(regimes.dtype)},
                "behavior_labels":        {"shape": list(behavior_labels.shape),        "dtype": str(behavior_labels.dtype)},
                "auto_behavior_labels":   {"shape": list(auto_behavior_labels.shape),   "dtype": str(auto_behavior_labels.dtype)},
                "final_label_sources":    {"shape": list(final_label_sources.shape),    "dtype": str(final_label_sources.dtype)},
                "last_token_states":      {"shape": list(last_token_states.shape),      "dtype": str(last_token_states.dtype)},
                "mean_pooled_states":     {"shape": list(mean_pooled_states.shape),     "dtype": str(mean_pooled_states.dtype)},
                "next_token_logits":      {"shape": list(next_token_logits.shape),      "dtype": str(next_token_logits.dtype)},
                "prompt_token_counts":    {"shape": list(prompt_token_counts.shape),    "dtype": str(prompt_token_counts.dtype)},
                "attention_mask_sums":    {"shape": list(attention_mask_sums.shape),    "dtype": str(attention_mask_sums.dtype)},
            },
        },
        "sanity": {
            "nan_count": int(nan_count),
            "prompt_token_counts_min": int(prompt_token_counts.min()),
            "prompt_token_counts_max": int(prompt_token_counts.max()),
            "prompt_token_counts_median": int(np.median(prompt_token_counts)),
            "attention_mask_sums_equal_token_counts": bool((attention_mask_sums == prompt_token_counts).all()),
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "transformers_version": __import__("transformers").__version__,
            "host": "anonymized",
            "platform": platform.platform(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest path: {manifest_path}")


if __name__ == "__main__":
    main()
