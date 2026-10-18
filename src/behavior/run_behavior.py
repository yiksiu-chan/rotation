"""
Behavior stage: measure control vs experiment alignment and persist index sets.

Writes JSON summaries suitable for the caching stage (successful indices only).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set

from src.behavior.paths import behavior_run_dir, control_behavior_path, experiment_behavior_path
from src.behavior.scoring import score_control_alignment, score_experiment_alignment
from src.config.cli import add_common_args, load_config_from_cli
from src.config.run_record import dump_run_record
from src.task_corpus.dispatch import load_statements_for_config as _load_statements
from src.task_corpus.hallucination_loader import load_hallucination_records
from src.task_corpus.records import StatementRecord
from src.task_corpus.triviaqa_loader import balanced_subsample_indices
from src.paths import PROJECT_ROOT
from src.config.schema import ExperimentConfig
from src.prompt.chat_template import apply_chat_template, load_experiment_system_prompts


def _dump_completions_jsonl(
    out_path: Path,
    responses: List[str],
    labels: List[int],
    indices: List[int],
    scores: List[int],
    correct_answers: List[str] | None = None,
) -> None:
    """Persist raw model completions next to a behavior-results JSON so they can
    be eyeballed after the run. One JSON object per line. When provided,
    ``correct_answers`` is written into a ``correct_answer`` field per row."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for i, (resp, lab, glob_i, s) in enumerate(zip(responses, labels, indices, scores)):
            row = {
                "global_index": int(glob_i),
                "label": int(lab),
                "score": int(s),
                "response": resp,
            }
            if correct_answers is not None:
                row["correct_answer"] = correct_answers[i]
            f.write(json.dumps(row) + "\n")


def _correct_answers_for_condition(
    records: List[StatementRecord], cfg: ExperimentConfig, condition: str
) -> List[str] | None:
    """Per-task expected answer string for each record. Returns None for tasks
    where no canonical answer-token mapping is defined yet."""
    if cfg.task == "spatial_easy":
        key = "control_answer" if condition == "control" else "experiment_answer"
        return [str(r.meta[key]) for r in records]
    return None


def _label_counts(gt_labels: List[int], output_labels: List[int]) -> Dict[str, int]:
    """Aggregate GT/output label counts persisted in behavior result JSONs."""
    pairs = list(zip(gt_labels, output_labels))
    return {
        "n_evaluated_GT_0": sum(1 for g in gt_labels if g == 0),
        "n_evaluated_GT_1": sum(1 for g in gt_labels if g == 1),
        "n_evaluated_output_0": sum(1 for o in output_labels if o == 0),
        "n_evaluated_output_1": sum(1 for o in output_labels if o == 1),
        "n_GT_0_output_0": sum(1 for g, o in pairs if g == 0 and o == 0),
        "n_GT_0_output_1": sum(1 for g, o in pairs if g == 0 and o == 1),
        "n_GT_1_output_0": sum(1 for g, o in pairs if g == 1 and o == 0),
        "n_GT_1_output_1": sum(1 for g, o in pairs if g == 1 and o == 1),
    }


def _select_balanced_statement_indices(
    records: List[StatementRecord], cfg: ExperimentConfig
) -> List[int]:
    """Pick ``balanced_set_size_behavior`` GT_0 + same GT_1 indices from ``records``."""
    labels = [r.label for r in records]
    return balanced_subsample_indices(
        labels, cfg.balanced_set_size_behavior, cfg.behavior_seed
    )


def _select_balanced_indices_for_condition(
    records: List[StatementRecord], cfg: ExperimentConfig, condition: str
) -> List[int]:
    """
    Return balanced GT_0/GT_1 indices into ``records`` for the given condition.

    For most tasks the same balanced subset is used in both conditions, so this
    returns the same list regardless of ``condition``.

    For ``cfg.task == "hallucination"`` the per-row ``meta["hallucination_label"]``
    encodes the condition axis (0=control, 1=experiment), so the per-condition
    balanced subsets are disjoint sub-pools of ``records``. The returned indices
    are still positions in the unified ``records`` list, preserving cross-stage
    index consistency.
    """
    if cfg.task != "hallucination":
        return _select_balanced_statement_indices(records, cfg)

    wanted = 0 if condition == "control" else 1
    cond_local_idx = [
        i for i, r in enumerate(records)
        if int(r.meta.get("hallucination_label", -1)) == wanted
    ]
    cond_labels = [records[i].label for i in cond_local_idx]
    chosen_local = balanced_subsample_indices(
        cond_labels, cfg.balanced_set_size_behavior, cfg.behavior_seed
    )
    return sorted(cond_local_idx[i] for i in chosen_local)


def _run_hallucination_per_variant(cfg, model, tokenizer, base_dir, variants) -> None:
    """
    Behavior pass for the hallucination task.

    Both control and experiment run inside the variant loop because each
    variant uses (a) its own system prompt for both conditions and (b) its own
    per-row ``hallucination_label`` for partitioning rows. Each variant's
    records are loaded by calling :func:`load_hallucination_records` with
    ``variant_id=V`` so the V-th label is projected into
    ``meta["hallucination_label"]``.

    Writes per-variant ``SysPrompt{NNN}/behavior_results_control.json`` and
    ``SysPrompt{NNN}/behavior_results_experiment.json``.
    """
    from src.model.inference import generate_completions

    for system_prompt_id in cfg.variant_indices(len(variants)):
        statements_v = load_hallucination_records(
            cfg.project_root,
            cfg.model_alias,
            cfg.triviaqa_hallucination_dataset_glob,
            variant_id=system_prompt_id,
        )
        if not statements_v:
            print(f"  skip SysPrompt{system_prompt_id:03d}: no records after label projection")
            continue

        selected_ctrl = _select_balanced_indices_for_condition(statements_v, cfg, "control")
        selected_exp = _select_balanced_indices_for_condition(statements_v, cfg, "experiment")
        subset_ctrl = [statements_v[i] for i in selected_ctrl]
        subset_exp = [statements_v[i] for i in selected_exp]
        labels_ctrl = [r.label for r in subset_ctrl]
        labels_exp = [r.label for r in subset_exp]

        sub_dir = base_dir / f"SysPrompt{system_prompt_id:03d}"
        sub_dir.mkdir(parents=True, exist_ok=True)

        # Control under variant V's system prompt
        prompts_ctrl = apply_chat_template(
            cfg, subset_ctrl, tokenizer,
            condition_type="control", system_prompt_id=system_prompt_id,
        )
        responses_ctrl = generate_completions(
            model,
            tokenizer,
            prompts_ctrl,
            batch_size=cfg.batch_size_generation,
            max_new_tokens=cfg.max_new_tokens,
            desc=f"control_{system_prompt_id:03d}",
        )
        success_ctrl_GT_0: List[int] = []
        success_ctrl_GT_1: List[int] = []
        output_labels_ctrl: List[int] = []
        for resp, lab, glob_i in zip(responses_ctrl, labels_ctrl, selected_ctrl):
            out_lab = score_control_alignment(resp, lab, cfg)
            output_labels_ctrl.append(int(out_lab))
            if out_lab == 1:
                (success_ctrl_GT_0 if lab == 0 else success_ctrl_GT_1).append(glob_i)
        accuracy_control = (len(success_ctrl_GT_0) + len(success_ctrl_GT_1)) / max(len(selected_ctrl), 1)
        control_payload = {
            "system_prompt_id": system_prompt_id,
            "Accuracy_control": accuracy_control,
            "n_evaluated": len(selected_ctrl),
            **_label_counts([int(l) for l in labels_ctrl], output_labels_ctrl),
            "S_success_idx_control_GT_0": success_ctrl_GT_0,
            "S_success_idx_control_GT_1": success_ctrl_GT_1,
        }
        cp = control_behavior_path(cfg, system_prompt_id=system_prompt_id)
        cp.parent.mkdir(parents=True, exist_ok=True)
        with open(cp, "w", encoding="utf-8") as f:
            json.dump(control_payload, f, indent=2)
        print(f"Wrote {cp}")

        # Experiment under the same variant V's system prompt
        prompts_exp = apply_chat_template(
            cfg, subset_exp, tokenizer,
            condition_type="experiment", system_prompt_id=system_prompt_id,
        )
        responses_exp = generate_completions(
            model,
            tokenizer,
            prompts_exp,
            batch_size=cfg.batch_size_generation,
            max_new_tokens=cfg.max_new_tokens,
            desc=f"experiment_{system_prompt_id:03d}",
        )
        success_exp_GT_0: List[int] = []
        success_exp_GT_1: List[int] = []
        output_labels_exp: List[int] = []
        for resp, lab, glob_i in zip(responses_exp, labels_exp, selected_exp):
            out_lab = score_experiment_alignment(resp, lab, cfg)
            output_labels_exp.append(int(out_lab))
            if out_lab == 1:
                (success_exp_GT_0 if lab == 0 else success_exp_GT_1).append(glob_i)
        accuracy_experiment = (len(success_exp_GT_0) + len(success_exp_GT_1)) / max(len(selected_exp), 1)
        # For hallucination the control/experiment row pools are disjoint
        # (different hallucination_label), so the "succeeded in both" and
        # "only-fail-in-experiment" sets are empty by construction. Persist
        # them as empty lists for schema parity with other tasks.
        experiment_payload = {
            "system_prompt_id": system_prompt_id,
            "Accuracy_control": accuracy_control,
            "Accuracy_experiment": accuracy_experiment,
            "n_evaluated": len(selected_exp),
            **_label_counts([int(l) for l in labels_exp], output_labels_exp),
            "S_success_idx_both_conditions_GT_0": [],
            "S_success_idx_both_conditions_GT_1": [],
            "S_only_fail_in_experiment_GT_0": [],
            "S_only_fail_in_experiment_GT_1": [],
        }
        ep = experiment_behavior_path(cfg, system_prompt_id)
        with open(ep, "w", encoding="utf-8") as f:
            json.dump(experiment_payload, f, indent=2)
        print(f"Wrote {ep}")
        _dump_completions_jsonl(
            ep.parent / "completions_experiment.jsonl",
            responses_exp, [int(l) for l in labels_exp], selected_exp, output_labels_exp,
            correct_answers=_correct_answers_for_condition(subset_exp, cfg, "experiment"),
        )


def run_behavior_for_config(cfg: ExperimentConfig) -> None:
    """
    Execute the full behavior pass:
        control generations, then per-variant experiment runs.

    For the hallucination task, both conditions run **inside** the variant
    loop (see :func:`_run_hallucination_per_variant`) because each variant
    has its own row partition and its own system prompt.

    Args:
        cfg: Experiment configuration (paths, tasks, sample sizes).

    Outputs:
        Writes ``behavior_results_control.json`` and per-``SysPrompt*`` experiment JSON files.
    """

    from src.model.inference import generate_completions, load_model_and_tokenizer

    base_dir = behavior_run_dir(cfg)
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {cfg.model_path!r} …")
    model, tokenizer = load_model_and_tokenizer(cfg.model_path, dtype=cfg.model_dtype)

    variants = load_experiment_system_prompts(cfg)

    if cfg.task == "hallucination":
        _run_hallucination_per_variant(cfg, model, tokenizer, base_dir, variants)
        return

    statements_all = _load_statements(cfg)
    if not statements_all:
        raise RuntimeError("No statements loaded; check dataset paths / HF access.")

    # Class-balanced behavior subset: balanced_set_size_behavior per GT class.
    selected_ctrl = _select_balanced_indices_for_condition(statements_all, cfg, "control")
    selected_exp = _select_balanced_indices_for_condition(statements_all, cfg, "experiment")
    subset_ctrl = [statements_all[i] for i in selected_ctrl]
    subset_exp = [statements_all[i] for i in selected_exp]

    labels_ctrl = [r.label for r in subset_ctrl]
    labels_exp = [r.label for r in subset_exp]

    # --- Control condition ---
    prompts_ctrl = apply_chat_template(cfg, subset_ctrl, tokenizer, condition_type="control")
    responses_ctrl = generate_completions(
        model,
        tokenizer,
        prompts_ctrl,
        batch_size=cfg.batch_size_generation,
        max_new_tokens=cfg.max_new_tokens,
        desc="control",
    )
    success_ctrl_GT_0: List[int] = []
    success_ctrl_GT_1: List[int] = []
    output_labels_ctrl: List[int] = []
    for resp, lab, glob_i in zip(responses_ctrl, labels_ctrl, selected_ctrl):
        out_lab = score_control_alignment(resp, lab, cfg)
        output_labels_ctrl.append(int(out_lab))
        if out_lab == 1:
            if lab == 0:
                success_ctrl_GT_0.append(glob_i)
            else:
                success_ctrl_GT_1.append(glob_i)

    n_success_ctrl = len(success_ctrl_GT_0) + len(success_ctrl_GT_1)
    accuracy_control = n_success_ctrl / max(len(selected_ctrl), 1)
    control_payload = {
        "Accuracy_control": accuracy_control,
        "n_evaluated": len(selected_ctrl),
        **_label_counts([int(l) for l in labels_ctrl], output_labels_ctrl),
        "S_success_idx_control_GT_0": success_ctrl_GT_0,
        "S_success_idx_control_GT_1": success_ctrl_GT_1,
    }
    cp = control_behavior_path(cfg)
    cp.parent.mkdir(parents=True, exist_ok=True)
    with open(cp, "w", encoding="utf-8") as f:
        json.dump(control_payload, f, indent=2)
    print(f"Wrote {cp}")
    _dump_completions_jsonl(
        cp.parent / "completions_control.jsonl",
        responses_ctrl, [int(l) for l in labels_ctrl], selected_ctrl, output_labels_ctrl,
        correct_answers=_correct_answers_for_condition(subset_ctrl, cfg, "control"),
    )

    success_ctrl_set: Set[int] = set(success_ctrl_GT_0) | set(success_ctrl_GT_1)

    # --- Experiment per system prompt (range-bounded by cfg.start/end_variant) ---
    for system_prompt_id in cfg.variant_indices(len(variants)):
        sub_dir = base_dir / f"SysPrompt{system_prompt_id:03d}"
        sub_dir.mkdir(parents=True, exist_ok=True)

        prompts_exp = apply_chat_template(
            cfg, subset_exp, tokenizer,
            condition_type="experiment", system_prompt_id=system_prompt_id,
        )
        responses_exp = generate_completions(
            model,
            tokenizer,
            prompts_exp,
            batch_size=cfg.batch_size_generation,
            max_new_tokens=cfg.max_new_tokens,
            desc=f"experiment_{system_prompt_id:03d}",
        )

        success_exp_GT_0: List[int] = []
        success_exp_GT_1: List[int] = []
        output_labels_exp: List[int] = []
        only_fail_GT_0: List[int] = []
        only_fail_GT_1: List[int] = []
        for resp, lab, glob_i in zip(responses_exp, labels_exp, selected_exp):
            out_lab = score_experiment_alignment(resp, lab, cfg)
            output_labels_exp.append(int(out_lab))
            if out_lab == 1:
                if lab == 0:
                    success_exp_GT_0.append(glob_i)
                else:
                    success_exp_GT_1.append(glob_i)
            elif glob_i in success_ctrl_set:
                if lab == 0:
                    only_fail_GT_0.append(glob_i)
                else:
                    only_fail_GT_1.append(glob_i)

        both_GT_0 = sorted(set(success_ctrl_GT_0) & set(success_exp_GT_0))
        both_GT_1 = sorted(set(success_ctrl_GT_1) & set(success_exp_GT_1))
        only_fail_GT_0.sort()
        only_fail_GT_1.sort()

        n_success_exp = len(success_exp_GT_0) + len(success_exp_GT_1)
        accuracy_experiment = n_success_exp / max(len(selected_exp), 1)

        experiment_payload = {
            "system_prompt_id": system_prompt_id,
            "Accuracy_control": accuracy_control,
            "Accuracy_experiment": accuracy_experiment,
            "n_evaluated": len(selected_exp),
            **_label_counts([int(l) for l in labels_exp], output_labels_exp),
            "S_success_idx_both_conditions_GT_0": both_GT_0,
            "S_success_idx_both_conditions_GT_1": both_GT_1,
            "S_only_fail_in_experiment_GT_0": only_fail_GT_0,
            "S_only_fail_in_experiment_GT_1": only_fail_GT_1,
        }
        ep = experiment_behavior_path(cfg, system_prompt_id)
        with open(ep, "w", encoding="utf-8") as f:
            json.dump(experiment_payload, f, indent=2)
        print(f"Wrote {ep}")
        _dump_completions_jsonl(
            ep.parent / "completions_experiment.jsonl",
            responses_exp, [int(l) for l in labels_exp], selected_exp, output_labels_exp,
            correct_answers=_correct_answers_for_condition(subset_exp, cfg, "experiment"),
        )


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """CLI for the behavior stage."""
    p = argparse.ArgumentParser(description="Run behavior evaluation (control + experiment prompts).")
    add_common_args(p)
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    """Entry point: load config and run :func:`run_behavior_for_config`."""
    args = parse_args(argv)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    cfg = load_config_from_cli(args)
    dump_run_record(
        cfg,
        out_dir=behavior_run_dir(cfg),
        argv=sys.argv,
        stage="behavior",
    )
    run_behavior_for_config(cfg)


if __name__ == "__main__":
    main()
