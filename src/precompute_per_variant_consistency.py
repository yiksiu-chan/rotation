"""
Compute a per-variant 10-pass consistency score and hallucination_label for
every row of the unfiltered TriviaQA T/F dataset.

For each system-prompt variant V in
``datasets/system_prompts_all/hallucination_prompts_all_variants.json`` and
each row R:
    1. Render the same prompt the cache stage would render (via
       ``src.prompt.chat_template.apply_chat_template`` with
       ``condition_type='experiment'``, ``system_prompt_id=V``).
    2. Sample ``--n_passes`` continuations (default 10) with the configured
       temperature.
    3. Score each continuation correct/incorrect using
       ``src.behavior.scoring.get_control_score`` (matches the convention used
       elsewhere for hallucination's control-side correctness).
    4. ``consistency[V, R] = mean(correct)``.
    5. Apply the dual threshold:
            consistency >= threshold_control     -> hallucination_label = 0
            consistency <= threshold_experiment  -> hallucination_label = 1
            otherwise                            -> hallucination_label = -1 (ambiguous)

Output: one JSON with per-row arrays of length n_variants.

This script requires a GPU (loads model weights).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.behavior.scoring import get_control_score  # noqa: E402
from src.config.loader import load_yaml_config  # noqa: E402
from src.model.inference import load_model_and_tokenizer  # noqa: E402
from src.prompt.chat_template import (  # noqa: E402
    apply_chat_template,
    load_experiment_system_prompts,
)
from src.task_corpus.records import StatementRecord  # noqa: E402
from src.task_corpus.triviaqa_loader import is_wikipedia_page_name, parse_claim  # noqa: E402


def _build_records(rows: list) -> List[StatementRecord]:
    """Wrap unfiltered TF rows into StatementRecord objects, dropping
    Wikipedia-suspect pairs (matches the loader used by the cache stage)."""
    drop_pairs = set()
    for row in rows:
        _, ans = parse_claim(row["claim"])
        if is_wikipedia_page_name(ans):
            drop_pairs.add(int(row["pair_id"]))

    records: List[StatementRecord] = []
    for row in rows:
        if int(row["pair_id"]) in drop_pairs:
            continue
        records.append(
            StatementRecord(
                index=int(row["pair_id"]),
                claim_text=row["claim"],
                label=int(row["label"]),
                pair_id=str(row["pair_id"]),
                meta={"_raw_row_idx": rows.index(row)},  # not used downstream
            )
        )
    return records


@torch.no_grad()
def _sample_one_pass(
    model,
    tokenizer,
    full_prompts: List[str],
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    desc: str,
) -> List[str]:
    """Generate one stochastic continuation per prompt. Returns continuations
    only (the prompt tokens are stripped)."""
    out: List[str] = []
    device = model.device
    for i in tqdm(range(0, len(full_prompts), batch_size), desc=desc, leave=False):
        batch = full_prompts[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=False)
        enc = {k: v.to(device) for k, v in enc.items()}
        prompt_len = enc["input_ids"].shape[1]
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
        cont = gen[:, prompt_len:]
        out.extend(tokenizer.batch_decode(cont, skip_special_tokens=True))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_override",
        type=str,
        default="config/hallucination.yaml",
        help="Override config (layered on top of config/default.yaml).",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="datasets/triviaqa_tf/triviaqa_tf_dataset_meta-llama_Llama-3.1-8B-Instruct.json",
    )
    parser.add_argument("--model_alias", type=str, default="Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help=(
            "Override cfg.model_path (e.g. a new model not named in the config). "
            "model_alias is set separately and controls the output filename."
        ),
    )
    parser.add_argument(
        "--variants",
        type=str,
        default=None,
        help=(
            "Comma-separated 0-indexed list of variants to run (e.g. '0,1'). "
            "Default: all variants in the bank."
        ),
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Limit number of rows for smoke testing. Default: all.",
    )
    parser.add_argument("--n_passes", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--threshold_control", type=float, default=0.8)
    parser.add_argument("--threshold_experiment", type=float, default=0.3)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output JSON path. Default: "
            "datasets/triviaqa_hallucination/triviaqa_hallucination_per_variant_<model_alias>.json"
        ),
    )
    args = parser.parse_args()

    # --- Resolve paths and load config ---
    in_path = Path(args.input)
    if not in_path.is_absolute():
        in_path = (PROJECT_ROOT / in_path).resolve()
    if not in_path.is_file():
        raise FileNotFoundError(f"Input not found: {in_path}")

    if args.output is None:
        out_path = PROJECT_ROOT / "datasets" / "triviaqa_hallucination" / (
            f"triviaqa_hallucination_per_variant_{args.model_alias}.json"
        )
    else:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = (PROJECT_ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml_config(
        PROJECT_ROOT / "config" / "default.yaml",
        override_path=PROJECT_ROOT / args.config_override,
    )
    if args.model_path:
        cfg.model_path = args.model_path  # model_alias is a derived property

    # --- Load rows + records ---
    with open(in_path, encoding="utf-8") as f:
        raw_rows = json.load(f)
    if not isinstance(raw_rows, list):
        raise TypeError(f"Expected list at top of {in_path}.")
    if args.max_rows is not None:
        raw_rows = raw_rows[: args.max_rows]

    records = _build_records(raw_rows)
    print(f"Built {len(records)} records (from {len(raw_rows)} raw rows)")

    # --- Resolve variant selection ---
    variants_all = load_experiment_system_prompts(cfg)
    if args.variants is None:
        variant_ids = list(range(len(variants_all)))
    else:
        variant_ids = sorted({int(s) for s in args.variants.split(",") if s.strip()})
        for v in variant_ids:
            if v < 0 or v >= len(variants_all):
                raise ValueError(
                    f"Variant {v} out of range (bank has {len(variants_all)} entries)."
                )
    print(f"Running {len(variant_ids)} variants: {variant_ids}")

    # --- Load model ---
    print(f"Loading model '{cfg.model_path}' …")
    model, tokenizer = load_model_and_tokenizer(cfg.model_path, dtype="bfloat16")

    # --- Per-row, per-variant consistency ---
    # Storage: dict variant_id -> list[int correct count per row] (length = len(records))
    correct_counts = {v: [0] * len(records) for v in variant_ids}

    t0 = time.time()
    for v in variant_ids:
        full_prompts = apply_chat_template(
            cfg, records, tokenizer,
            condition_type="experiment", system_prompt_id=v,
        )

        for pass_idx in range(args.n_passes):
            completions = _sample_one_pass(
                model,
                tokenizer,
                full_prompts,
                batch_size=args.batch_size,
                max_new_tokens=cfg.max_new_tokens,
                temperature=args.temperature,
                desc=f"variant {v} pass {pass_idx+1}/{args.n_passes}",
            )
            for i, comp in enumerate(completions):
                correct_counts[v][i] += get_control_score(comp, records[i].label, "hallucination")
        done_v = (variant_ids.index(v) + 1)
        elapsed = time.time() - t0
        print(
            f"variant {v} done. cumulative {done_v}/{len(variant_ids)} variants  "
            f"({elapsed/60:.1f} min elapsed)"
        )

    # --- Build per-row output payload ---
    t_ctrl, t_exp = float(args.threshold_control), float(args.threshold_experiment)
    if t_exp >= t_ctrl:
        raise ValueError("threshold_experiment must be < threshold_control")

    out_rows = []
    n_label0_per_v = {v: 0 for v in variant_ids}
    n_label1_per_v = {v: 0 for v in variant_ids}
    n_drop_per_v = {v: 0 for v in variant_ids}

    for i, r in enumerate(records):
        consistencies = {}
        labels = {}
        for v in variant_ids:
            c = correct_counts[v][i] / args.n_passes
            consistencies[v] = c
            if c >= t_ctrl:
                lbl = 0
                n_label0_per_v[v] += 1
            elif c <= t_exp:
                lbl = 1
                n_label1_per_v[v] += 1
            else:
                lbl = -1
                n_drop_per_v[v] += 1
            labels[v] = lbl
        out_rows.append(
            {
                "pair_id": r.pair_id,
                "claim": r.claim_text,
                "label": r.label,
                "consistency_per_variant": [consistencies[v] for v in variant_ids],
                "hallucination_label_per_variant": [labels[v] for v in variant_ids],
            }
        )

    payload = {
        "source_file": str(in_path.relative_to(PROJECT_ROOT))
        if in_path.is_relative_to(PROJECT_ROOT)
        else str(in_path),
        "model_alias": args.model_alias,
        "n_passes": args.n_passes,
        "temperature": args.temperature,
        "threshold_control": t_ctrl,
        "threshold_experiment": t_exp,
        "variant_ids": variant_ids,
        "variant_prompts": [variants_all[v] for v in variant_ids],
        "n_rows": len(out_rows),
        "per_variant_counts": {
            str(v): {
                "n_control": n_label0_per_v[v],
                "n_experiment": n_label1_per_v[v],
                "n_dropped": n_drop_per_v[v],
            }
            for v in variant_ids
        },
        "rows": out_rows,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_path}")
    for v in variant_ids:
        c = n_label0_per_v[v]
        e = n_label1_per_v[v]
        d = n_drop_per_v[v]
        print(f"  variant {v:2d}: control={c}  experiment={e}  dropped={d}")


if __name__ == "__main__":
    main()
