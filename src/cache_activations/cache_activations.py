"""
Cache stage: GT-paired activations across token positions and conditions.

For each ``(system_prompt_id, seed)``:

1. Load two GT-split pools of statements successful (followed the instruction)
   in both conditions:
   - Read ``S_success_idx_both_conditions_GT_0`` and
   ``S_success_idx_both_conditions_GT_1`` from the behavior stage's per-prompt
   ``behavior_results_experiment.json`` and use them as independent sampling
   pools.

2. Sample from each GT pool separately, controlled by ``cfg.cache_sample_method``:
   - ``"balance_set"`` (default) — ``n_sample_cache`` GT_0 + ``n_sample_cache`` GT_1.
     Achieved per-class count = ``min(n_sample_cache, |pool_GT_0|, |pool_GT_1|)``.
   - ``"maximum"`` — up to ``n_sample_cache`` from each GT class independently;
     classes are sampled separately so the two halves may have different counts.

3. For each condition (``control``, ``experiment``):
   - render prompts via :func:`src.prompt.chat_template.apply_chat_template_split`
     (which returns ``(core_prompt, answer_prefix)`` per row so the cache hook
     can locate the "prompt" position at the last token *before* answer_prefix),
   - generate with hooks that cache hidden states across
     ``n_pre + 1 + n_post + n_gen`` token positions
     (:func:`src.cache_activations.hooks.generate_with_multi_pos_cache`),
   - score ``output_label``, 1 means aligned with the instruction
     - For control condition, 1 means model's output matches the GT label
     - For experiment condition, 1 means model's output is the opposite of the GT

4. **Experiment-fail subset.** Read GT-split pools
   ``S_only_fail_in_experiment_GT_0`` and ``S_only_fail_in_experiment_GT_1``
   from the experiment behavior JSON (statements that succeeded in control
   and failed ``output_label == 0`` in this variant's experiment behavior).
   Sample from each GT pool separately with the same
   ``cache_sample_method`` / ``n_sample_cache`` budget, run the experiment
   condition only, and persist under the ``_fail`` filename suffix
   (``completions_experiment_<method>_fail.json`` and
   ``activations_experiment_<method>_GT_<label>_fail.pkl``). Skipped silently
   when both pools are empty for a given ``(SysPrompt, seed)``.

5. Persist completions and per-GT activation pickles under
   ``{output_root}/{model_alias}/{task}_{dataset}/SysPrompt{id:03d}/{seed}_pre{n}_gen{m}/``

Activation tensors shape: (k_per_class, n_positions, n_layers, d_model) per GT split.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import re
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

from src.behavior.paths import behavior_run_dir, control_behavior_path, experiment_behavior_path
from src.behavior.scoring import score_control_alignment, score_experiment_alignment
from src.cache_activations.paths import (
    activations_path,
    cache_seed_dir,
    completions_path,
)
from src.config.cli import add_common_args, load_config_from_cli
from src.config.run_record import dump_run_record
from src.config.schema import ExperimentConfig
from src.paths import PROJECT_ROOT
from src.prompt.chat_template import apply_chat_template_split, load_experiment_system_prompts
from src.task_corpus.dispatch import load_statements_for_config as _load_statements
from src.task_corpus.hallucination_loader import load_hallucination_records
from src.task_corpus.records import StatementRecord


CACHE_SAMPLE_METHODS = ("balance_set", "maximum")

# Matches Llama-3 / Qwen / etc. chat-template special tokens like
# ``<|begin_of_text|>``, ``<|start_header_id|>``, ``<|eot_id|>``. Used only
# when serializing ``full_prompt_example`` for traceability — generation paths
# keep the original tokenized prompt. 
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>]+\|>")


# --------------------------------------------------------------------------- #
# Statement and behavior I/O                                                  #
# --------------------------------------------------------------------------- #


def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON atomically so a crashed run never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def _atomic_write_pickle(path: Path, obj) -> None:
    """Pickle to a tmp file then atomically rename into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Sampling                                                                    #
# --------------------------------------------------------------------------- #


def _sample_indices(
    pool0: Sequence[int],
    pool1: Sequence[int],
    n_sample: int,
    seed: int,
    method: str,
) -> Tuple[List[int], List[int]]:
    """
    Draw GT_0 / GT_1 indices from two pre-split pools according to ``method``.

    Sentinel: ``n_sample == -1`` disables sampling entirely and returns the
    full pools (GT_0 then GT_1) regardless of ``method``; no per-class
    equalization is applied.

    Args:
        pool0: Global statement indices with GT label 0 to sample from.
        pool1: Global statement indices with GT label 1 to sample from.
        n_sample: Per-class cap (``cfg.n_sample_cache``), or ``-1`` to use
            all rows of the pools.
        seed: RNG seed; same ``seed`` always produces the same subset for a
            fixed pair of pools.
        method: ``"balance_set"`` (equal per class, capped to the smaller pool)
            or ``"maximum"`` (each class capped independently at ``n_sample``).

    Returns:
        ``(indices, gt_labels)`` of equal length. For ``balance_set`` both halves
        are the same size ``k = min(n_sample, |pool0|, |pool1|)``.
        For ``maximum`` the two halves may differ in size.
    """
    if n_sample == -1:
        take0 = sorted(pool0)
        take1 = sorted(pool1)
        return take0 + take1, [0] * len(take0) + [1] * len(take1)
    rng = random.Random(seed)
    if method == "balance_set":
        k = min(n_sample, len(pool0), len(pool1))
        if k == 0:
            return [], []
        take0 = sorted(rng.sample(list(pool0), k))
        take1 = sorted(rng.sample(list(pool1), k))
        return take0 + take1, [0] * k + [1] * k
    if method == "maximum":
        k0 = min(n_sample, len(pool0))
        k1 = min(n_sample, len(pool1))
        if k0 == 0 and k1 == 0:
            return [], []
        take0 = sorted(rng.sample(list(pool0), k0)) if k0 else []
        take1 = sorted(rng.sample(list(pool1), k1)) if k1 else []
        return take0 + take1, [0] * k0 + [1] * k1
    raise ValueError(
        f"Unsupported cache_sample_method={method!r}; "
        f"expected one of {CACHE_SAMPLE_METHODS}."
    )


# --------------------------------------------------------------------------- #
# Per-condition cache + persist                                               #
# --------------------------------------------------------------------------- #


def _write_completions(
    cfg: ExperimentConfig,
    system_prompt_id: int,
    seed: int,
    condition: str,
    system_prompt_str: str,
    statement_indices: Sequence[int],
    statements_balanced: Sequence[StatementRecord],
    gt_labels: Sequence[int],
    responses: Sequence[str],
    prompts: Sequence[str],
    sampled_size_GT_0: int,
    sampled_size_GT_1: int,
    fail: bool = False,
) -> None:
    """
    Build and atomically write
    ``completions_{condition}_{method}[_fail].json`` for one cached run.

    When ``fail`` is true the file is written under the ``_fail`` suffix and
    the payload is annotated with ``"is_fail_subset": true`` so consumers can
    distinguish the experiment-fail subset from the standard pool.

    For tasks whose experiment-condition system prompt varies per row
    (currently ``lying_situation_induced``), the top-level ``system_prompt``
    field is set to the sentinel ``"<per-row; see outputs>"`` and each row in
    ``outputs`` carries its own ``system_prompt`` key.
    """
    per_row_sys_prompt = (
        cfg.task == "lying_situation_induced" and condition == "experiment"
    )
    rows = []
    for local_i, (stmt_idx, label, resp) in enumerate(
        zip(statement_indices, gt_labels, responses)
    ):
        if condition == "control":
            output_label = score_control_alignment(resp, label, cfg)
        else:
            output_label = score_experiment_alignment(resp, label, cfg)
        row = {
            "statement_index": int(stmt_idx),
            "statement": statements_balanced[local_i].claim_text,
            "GT_label": int(label),
            "model_output": resp,
            "output_label": int(output_label),
        }
        if per_row_sys_prompt:
            row["system_prompt"] = statements_balanced[local_i].meta["system_prompt"]
        rows.append(row)
    top_level_sys_prompt = (
        "<per-row; see outputs>" if per_row_sys_prompt else system_prompt_str
    )
    payload = {
        "system_prompt": top_level_sys_prompt,
        "condition": condition,
        "system_prompt_id": int(system_prompt_id),
        "seed": int(seed),
        "n_pre_prompt_tokens": int(cfg.n_pre_prompt_tokens),
        "n_generated_tokens": int(cfg.n_generated_tokens),
        "cache_sample_method": cfg.cache_sample_method,
        "sampled_size_GT_0": int(sampled_size_GT_0),
        "sampled_size_GT_1": int(sampled_size_GT_1),
        "is_fail_subset": bool(fail),
        # One fully-rendered prompt example for traceability
        # (chat template + system + user + answer_prefix). Special tokens
        # (``<|...|>``) are stripped to keep the example human-readable.
        "full_prompt_example": (
            _SPECIAL_TOKEN_RE.sub("", prompts[0]) if prompts else ""
        ),
        "outputs": rows,
    }
    _atomic_write_json(
        completions_path(cfg, system_prompt_id, seed, condition, fail=fail), payload
    )


def _persist_activations_split_by_gt(
    cfg: ExperimentConfig,
    system_prompt_id: int,
    seed: int,
    condition: str,
    cache_full,  # torch.Tensor (n_layers, k_total, n_positions, d_model)
    gt_labels: Sequence[int],
    fail: bool = False,
) -> None:
    """
    Permute to ``(k_total, n_positions, n_layers, d_model)``, split by GT, and
    pickle each half. Asserts the spec-required shape before persisting.
    The ``fail`` flag routes the output through the ``_fail`` filename suffix.
    """
    cache_perm = cache_full.permute(1, 2, 0, 3).contiguous()  # (k_total, n_pos, n_layers, d)
    n_layers = cache_full.shape[0]
    d_model = cache_full.shape[3]
    n_positions = cfg.n_pre_prompt_tokens + 1 + cfg.n_generated_tokens

    for gt in (0, 1):
        sel = [i for i, lab in enumerate(gt_labels) if lab == gt]
        if not sel:
            continue
        chunk = cache_perm[sel].cpu()
        expected = (len(sel), n_positions, n_layers, d_model)
        assert tuple(chunk.shape) == expected, (
            f"activation tensor shape {tuple(chunk.shape)} != expected {expected} "
            f"for condition={condition!r}, GT={gt}, fail={fail}"
        )
        _atomic_write_pickle(
            activations_path(cfg, system_prompt_id, seed, condition, gt, fail=fail),
            chunk,
        )


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #


def run_cache_activations_for_config(cfg: ExperimentConfig) -> None:
    """
    Run the cache stage end-to-end against one config.

    Iterates over every ``system_prompt_id`` for which the behavior stage
    produced ``behavior_results_experiment.json``, then over every seed in
    ``cfg.seed_list_cache``. Skips (and logs) any combination whose sampled
    pool ends up empty or whose behavior file is missing.

    Sampling is governed by ``cfg.cache_sample_method`` and draws from the
    GT-split pools ``S_success_idx_both_conditions_GT_0`` and
    ``S_success_idx_both_conditions_GT_1`` independently:
        - ``"balance_set"`` — ``n_sample_cache`` GT_0 + ``n_sample_cache`` GT_1,
          both halves capped to ``min(n_sample_cache, |pool_GT_c|)``.
        - ``"maximum"`` — up to ``n_sample_cache`` per GT class, each class
          capped independently (per-class counts may differ).

    For each condition (``control`` + ``experiment``) the stage runs greedy
    generation with multi-position forward hooks, writes
    ``completions_{condition}_{method}.json`` and per-GT
    ``activations_{condition}_{method}_GT_{label}.pkl``.

    Args:
        cfg: Experiment configuration (artifact roots, sampling sizes, model).

    Raises:
        FileNotFoundError: If the behavior stage hasn't been run for this config.
        ValueError: If ``seed_list_cache`` is empty, ``dataset`` is unknown, or
            ``cache_sample_method`` is not one of :data:`CACHE_SAMPLE_METHODS`.
    """
    if cfg.cache_sample_method not in CACHE_SAMPLE_METHODS:
        raise ValueError(
            f"Unsupported cfg.cache_sample_method={cfg.cache_sample_method!r}; "
            f"expected one of {CACHE_SAMPLE_METHODS}."
        )

    # Lazy GPU imports: importing this module shouldn't require torch/transformers.
    import torch

    from src.cache_activations.hooks import generate_with_multi_pos_cache
    from src.model.inference import load_model_and_tokenizer

    seeds = list(cfg.seed_list_cache or [])
    if not seeds:
        raise ValueError("seed_list_cache is empty; nothing to cache.")

    # For hallucination the run-level control behavior file does not exist
    # (control is per-variant), and ``success_ctrl_set`` is only used to
    # intersect the fail pool — which is empty by design for hallucination.
    # Skip both for hallucination; do them for every other task.
    if cfg.task == "hallucination":
        success_ctrl_set: set[int] = set()
        statements: List[StatementRecord] = []  # populated per-variant inside the loop
    else:
        statements = _load_statements(cfg)
        if not statements:
            raise RuntimeError("No statements loaded; check dataset paths / HF access.")
        if not control_behavior_path(cfg).exists():
            raise FileNotFoundError(
                f"Missing {control_behavior_path(cfg)}. Run the behavior stage first."
            )
        # Control successes are shared across all variants — read once and union
        # the two GT-split lists. Used to intersect the fail pool defensively in
        # case the behavior and cache stages were run against different control
        # results.
        ctrl_data = _read_json(control_behavior_path(cfg))
        success_ctrl_set = {
            int(i) for i in ctrl_data.get("S_success_idx_control_GT_0", [])
        } | {
            int(i) for i in ctrl_data.get("S_success_idx_control_GT_1", [])
        }

    variants = load_experiment_system_prompts(cfg)

    print(f"Loading model {cfg.model_path!r} for caching …")
    model, tokenizer = load_model_and_tokenizer(cfg.model_path, dtype=cfg.model_dtype)

    n_pre = cfg.n_pre_prompt_tokens
    n_gen = cfg.n_generated_tokens

    # ``balanced_set_size_behavior == -1`` opts out of behavior filtering
    # entirely: the cache stage's pool becomes every statement (regardless of
    # success in either condition) and the experiment-fail subset is skipped
    # because "fail" is only defined relative to behavior outcomes.
    #
    # ``cfg.task == "hallucination"`` similarly bypasses the behavior-derived
    # pool: the control/experiment split is encoded per-row in
    # ``meta["hallucination_label"]`` rather than by behavior outcome, so the
    # "succeeded in BOTH conditions" intersection is empty by construction
    # (control scoring and experiment scoring are inverses of each other).
    no_behavior_filter = (
        cfg.balanced_set_size_behavior == -1 or cfg.task == "hallucination"
    )
    # Full per-GT pools (used for non-hallucination tasks; for hallucination
    # these are re-derived per-variant inside the loop because the record set
    # itself changes with the V-th hallucination_label projection).
    if cfg.task == "hallucination":
        full_pool0: List[int] = []
        full_pool1: List[int] = []
    else:
        full_pool0 = [i for i, s in enumerate(statements) if int(s.label) == 0]
        full_pool1 = [i for i, s in enumerate(statements) if int(s.label) == 1]

    for system_prompt_id in cfg.variant_indices(len(variants)):
        # For hallucination: reload the corpus with this variant's per-row
        # hallucination_label projected from hallucination_label_per_variant[V].
        # The record list (and thus its indexing) differs per variant because
        # rows with label_v == -1 are dropped.
        if cfg.task == "hallucination":
            statements = load_hallucination_records(
                cfg.project_root,
                cfg.model_alias,
                cfg.triviaqa_hallucination_dataset_glob,
                variant_id=system_prompt_id,
            )
            full_pool0 = [i for i, s in enumerate(statements) if int(s.label) == 0]
            full_pool1 = [i for i, s in enumerate(statements) if int(s.label) == 1]

        exp_path = experiment_behavior_path(cfg, system_prompt_id)
        if not exp_path.exists():
            print(f"  skip SysPrompt{system_prompt_id:03d}: {exp_path} missing")
            continue
        exp_data = _read_json(exp_path)

        pool0 = [int(i) for i in exp_data.get("S_success_idx_both_conditions_GT_0", [])]
        pool1 = [int(i) for i in exp_data.get("S_success_idx_both_conditions_GT_1", [])]

        # When the behavior-derived intersection is unusable (either explicitly
        # disabled via ``balanced_set_size_behavior == -1`` or empty by design
        # for hallucination), fall back to the full per-GT pools so the
        # condition-specific filtering downstream still has rows to draw from.
        if no_behavior_filter:
            pool0 = list(full_pool0)
            pool1 = list(full_pool1)

        fail_pool0 = sorted(
            int(i)
            for i in exp_data.get("S_only_fail_in_experiment_GT_0", [])
            if int(i) in success_ctrl_set
        )
        fail_pool1 = sorted(
            int(i)
            for i in exp_data.get("S_only_fail_in_experiment_GT_1", [])
            if int(i) in success_ctrl_set
        )

        for seed in seeds:
            indices, gt_labels = _sample_indices(
                pool0,
                pool1,
                cfg.n_sample_cache,
                seed,
                cfg.cache_sample_method,
            )
            k0 = sum(1 for lab in gt_labels if lab == 0)
            k1 = sum(1 for lab in gt_labels if lab == 1)
            if k0 + k1 == 0:
                print(
                    f"  skip SysPrompt{system_prompt_id:03d} seed={seed}: "
                    f"empty sampled pool (gt0={len(pool0)}, gt1={len(pool1)})"
                )
                continue
            # n_sample_cache=-1 explicitly asks for "all rows in pool", so
            # the under-cap warnings below don't apply.
            if cfg.n_sample_cache != -1:
                if cfg.cache_sample_method == "balance_set" and k0 < cfg.n_sample_cache:
                    print(
                        f"  WARNING SysPrompt{system_prompt_id:03d} seed={seed}: "
                        f"k={k0}/class < requested {cfg.n_sample_cache}"
                    )
                elif cfg.cache_sample_method == "maximum" and (
                    k0 < cfg.n_sample_cache or k1 < cfg.n_sample_cache
                ):
                    print(
                        f"  WARNING SysPrompt{system_prompt_id:03d} seed={seed}: "
                        f"sampled (k0={k0}, k1={k1}) below requested {cfg.n_sample_cache}/class"
                    )

            statements_balanced = [statements[i] for i in indices]

            seed_dir = cache_seed_dir(cfg, system_prompt_id, seed)
            (seed_dir / "completions").mkdir(parents=True, exist_ok=True)
            (seed_dir / "activations").mkdir(parents=True, exist_ok=True)

            for condition in ("control", "experiment"):
                # For hallucination, control vs experiment is a per-row
                # property (``meta["hallucination_label"]``), not a per-prompt
                # one. Resample inside the loop from a pool filtered to the
                # rows whose hallucination_label matches the current condition.
                # For other tasks, ``cond_*`` aliases the shared per-seed
                # sample so row alignment between control/experiment caches
                # is preserved.
                if cfg.task == "hallucination":
                    wanted_label = 0 if condition == "control" else 1
                    cond_pool0 = [
                        i for i in pool0
                        if int(statements[i].meta.get("hallucination_label", -1)) == wanted_label
                    ]
                    cond_pool1 = [
                        i for i in pool1
                        if int(statements[i].meta.get("hallucination_label", -1)) == wanted_label
                    ]
                    cond_indices, cond_gt_labels = _sample_indices(
                        cond_pool0,
                        cond_pool1,
                        cfg.n_sample_cache,
                        seed,
                        cfg.cache_sample_method,
                    )
                    cond_k0 = sum(1 for lab in cond_gt_labels if lab == 0)
                    cond_k1 = sum(1 for lab in cond_gt_labels if lab == 1)
                    cond_statements_balanced = [statements[i] for i in cond_indices]
                    if cond_k0 + cond_k1 == 0:
                        print(
                            f"  skip SysPrompt{system_prompt_id:03d} seed={seed} "
                            f"condition={condition}: empty hallucination-filtered pool "
                            f"(gt0={len(cond_pool0)}, gt1={len(cond_pool1)})"
                        )
                        continue
                else:
                    cond_indices = indices
                    cond_gt_labels = gt_labels
                    cond_k0 = k0
                    cond_k1 = k1
                    cond_statements_balanced = statements_balanced

                prompt_pairs = apply_chat_template_split(
                    cfg,
                    cond_statements_balanced,
                    tokenizer,
                    condition_type=condition,
                    system_prompt_id=system_prompt_id,
                )
                full_prompts = [core + ap for core, ap in prompt_pairs]
                responses, cache_full = generate_with_multi_pos_cache(
                    model,
                    tokenizer,
                    prompt_pairs,
                    n_pre=n_pre,
                    n_gen=n_gen,
                    batch_size=cfg.batch_size_cache,
                    max_new_tokens=cfg.max_new_tokens,
                    desc=f"v{system_prompt_id:03d}_seed{seed}_{condition}",
                )

                if condition == "experiment" or cfg.task == "hallucination":
                    # Hallucination uses variant V's prompt for BOTH conditions.
                    system_prompt_str = variants[system_prompt_id]
                else:
                    system_prompt_str = cfg.control_system_prompt_map.get(cfg.task, "")

                _write_completions(
                    cfg,
                    system_prompt_id,
                    seed,
                    condition,
                    system_prompt_str,
                    cond_indices,
                    cond_statements_balanced,
                    cond_gt_labels,
                    responses,
                    prompts=full_prompts,
                    sampled_size_GT_0=cond_k0,
                    sampled_size_GT_1=cond_k1,
                )
                _persist_activations_split_by_gt(
                    cfg,
                    system_prompt_id,
                    seed,
                    condition,
                    cache_full,
                    cond_gt_labels,
                )

                del cache_full
                torch.cuda.empty_cache()

            # ---- Experiment-fail subset ----
            # Cache the experiment condition for statements that succeeded in
            # control but failed in experiment (output_label == 0). Same
            # cache_sample_method / n_sample_cache budget; written under the
            # ``_fail`` suffix. Skipped silently when the fail pool is empty.
            fail_indices, fail_gt_labels = _sample_indices(
                fail_pool0,
                fail_pool1,
                cfg.n_sample_cache,
                seed,
                cfg.cache_sample_method,
            )
            kf0 = sum(1 for lab in fail_gt_labels if lab == 0)
            kf1 = sum(1 for lab in fail_gt_labels if lab == 1)
            if kf0 + kf1 == 0:
                print(
                    f"  note SysPrompt{system_prompt_id:03d} seed={seed}: "
                    f"S_only_fail_in_experiment_GT_{{0,1}} both empty — skipping _fail cache"
                )
            else:
                statements_fail = [statements[i] for i in fail_indices]
                prompt_pairs_fail = apply_chat_template_split(
                    cfg,
                    statements_fail,
                    tokenizer,
                    condition_type="experiment",
                    system_prompt_id=system_prompt_id,
                )
                full_prompts_fail = [core + ap for core, ap in prompt_pairs_fail]
                responses_fail, cache_fail = generate_with_multi_pos_cache(
                    model,
                    tokenizer,
                    prompt_pairs_fail,
                    n_pre=n_pre,
                    n_gen=n_gen,
                    batch_size=cfg.batch_size_cache,
                    max_new_tokens=cfg.max_new_tokens,
                    desc=f"v{system_prompt_id:03d}_seed{seed}_experiment_fail",
                )
                _write_completions(
                    cfg,
                    system_prompt_id,
                    seed,
                    "experiment",
                    variants[system_prompt_id],
                    fail_indices,
                    statements_fail,
                    fail_gt_labels,
                    responses_fail,
                    prompts=full_prompts_fail,
                    sampled_size_GT_0=kf0,
                    sampled_size_GT_1=kf1,
                    fail=True,
                )
                _persist_activations_split_by_gt(
                    cfg,
                    system_prompt_id,
                    seed,
                    "experiment",
                    cache_fail,
                    fail_gt_labels,
                    fail=True,
                )
                del cache_fail
                torch.cuda.empty_cache()

            print(
                f"  done SysPrompt{system_prompt_id:03d} seed={seed} "
                f"(method={cfg.cache_sample_method}, k0={k0}, k1={k1}, "
                f"fail_k0={kf0}, fail_k1={kf1})"
            )


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """CLI for the cache stage."""
    p = argparse.ArgumentParser(
        description="Cache activations for contrastive analysis."
    )
    add_common_args(p)
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    """Entry point: load config and run :func:`run_cache_activations_for_config`."""
    args = parse_args(argv)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    cfg = load_config_from_cli(args)
    dump_run_record(
        cfg,
        out_dir=behavior_run_dir(cfg),
        argv=sys.argv,
        stage="cache",
    )
    run_cache_activations_for_config(cfg)


if __name__ == "__main__":
    main()
