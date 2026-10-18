"""
Chat template rendering for contrastive experiments.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Literal, Tuple

from src.config.schema import ExperimentConfig
from src.task_corpus.records import StatementRecord

ConditionType = Literal["control", "experiment"]


_LLAMA3_PINNED_DATE = "26 Jul 2024"


# Spatial-easy two-shot demos. Both are Alice-POV (control-condition wording);
# friend's setup intentionally omits Bob-POV demos so the experiment condition
# must generalise the mirror. The reasoning template mirrors friend's:
# "We are asking from X's own point of view. The Y is on X's Z side, so from
# X's perspective it is on their Z. Answer: Z".
_SHOT_EASY_LEFT_USER = (
    "Alice and Bob are sitting directly across from each other at a table, facing each other. "
    "A pencil is placed on Alice's left side of the table. "
    "From Alice's point of view, is the object on the left or the right side of the table?"
)
_SHOT_EASY_LEFT_ASST = (
    "We are asking from Alice's own point of view. "
    "The pencil is on Alice's left side, so from Alice's perspective it is on her left."
    "\n\nAnswer: Left"
)

_SHOT_EASY_RIGHT_USER = (
    "Alice and Bob are sitting directly across from each other at a table, facing each other. "
    "A stapler is placed on Alice's right side of the table. "
    "From Alice's point of view, is the object on the left or the right side of the table?"
)
_SHOT_EASY_RIGHT_ASST = (
    "We are asking from Alice's own point of view. "
    "The stapler is on Alice's right side, so from Alice's perspective it is on her right."
    "\n\nAnswer: Right"
)


def load_experiment_system_prompts(cfg: ExperimentConfig) -> List[str]:
    """
    Load the list of experiment (non-control) system prompts for the active task.

    Args:
        cfg: Must define ``experiment_system_prompts[task]`` with JSON filename and key.

    Returns:
        Ordered system prompt strings.

    Raises:
        KeyError: If ``cfg.task`` has no registered experiment prompt spec.
        FileNotFoundError: If the JSON file is missing under ``sys_prompt_dir_path``.
        ValueError: If the variants key is absent in the JSON object.
    """
    if cfg.task not in cfg.experiment_system_prompts:
        raise KeyError(f"No experiment_system_prompts entry for task={cfg.task!r}")
    spec = cfg.experiment_system_prompts[cfg.task]
    path: Path = cfg.sys_prompt_dir_path / spec.filename
    if not path.is_file():
        raise FileNotFoundError(
            f"System prompt JSON not found: {path}. "
            "Populate datasets/system_prompts_all or set sys_prompt_dir_relative."
        )
    with open(path, encoding="utf-8") as f:
        blob = json.load(f)
    if spec.variants_json_key not in blob:
        raise ValueError(
            f"Key {spec.variants_json_key!r} not in {path}. "
        )
    variants = blob[spec.variants_json_key]
    if not isinstance(variants, list):
        raise TypeError(f"Expected list under {spec.variants_json_key!r}, got {type(variants)}")
    return [str(x) for x in variants]


def _render(tokenizer: Any, messages: List[dict]) -> str:
    """Render a messages list via the tokenizer's official chat template.

    ``add_generation_prompt=True`` appends the assistant header so the model
    starts generating the assistant turn directly. ``date_string`` is consumed
    by Llama 3.1's bundled template; tokenizers whose templates don't reference
    it ignore the kwarg.
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        date_string=_LLAMA3_PINNED_DATE,
    )


def apply_chat_template_split(
    cfg: ExperimentConfig,
    records: List[StatementRecord],
    tokenizer: Any,
    condition_type: ConditionType = "control",
    system_prompt_id: int = 0,
) -> List[Tuple[str, str]]:
    """
    Render each record as ``(core_prompt, answer_prefix)`` separately.

    ``core_prompt`` is the full chat-template body up to (and including) the
    assistant turn opening (``add_generation_prompt=True``). ``answer_prefix``
    is the additional steering text appended after it. Concatenating the two
    reproduces the full prompt string passed to ``model.generate``. The cache
    stage uses the split to locate the "prompt" token position (the last
    token of ``core_prompt``).

    Per-task ``answer_prefix`` sourcing:

    - ``lying_to_achieve_goals``: per-row ``r.meta["answer_prefix"]`` plus a
      trailing space (overrides ``cfg.answer_prefix``).
    - ``spatial_easy``: literal ``"Answer:"`` so the model continues from
      the same stem the few-shot demo answers end with.
    - everything else: ``cfg.answer_prefix``.

    Args:
        cfg: Experiment configuration (system prompt maps, ``answer_prefix``).
        records: Statement records to render.
        tokenizer: HuggingFace tokenizer with a configured ``chat_template``.
        condition_type: ``control`` uses ``control_system_prompt_map`` (except
            for tasks listed in ``use_variant_for_control`` below);
            ``experiment`` loads variant ``system_prompt_id`` from the task's
            JSON bank.
        system_prompt_id: Index into the experiment variant list.

    Returns:
        List of ``(core_prompt, answer_prefix)`` tuples, one per record.
    """
    use_variant_for_control = cfg.task in (
        "hallucination", "spatial_easy"
    )

    if condition_type == "control" and not use_variant_for_control:
        if cfg.task not in cfg.control_system_prompt_map:
            raise KeyError(f"Task {cfg.task!r} missing from control_system_prompt_map")
        system_prompt = cfg.control_system_prompt_map[cfg.task]
    else:
        variants = load_experiment_system_prompts(cfg)
        if system_prompt_id < 0 or system_prompt_id >= len(variants):
            raise ValueError(
                f"system_prompt_id {system_prompt_id} out of range for task={cfg.task!r} "
                f"(valid 0-{len(variants) - 1})."
            )
        system_prompt = variants[system_prompt_id]

    out: List[Tuple[str, str]] = []
    for r in records:
        if cfg.task in ["lying_to_achieve_goals", "lying_to_achieve_goals_direct"]:
            instr_key = "normal_instruction" if condition_type == "control" else "deceive_instruction"
            instruction = r.meta[instr_key]
            question_prefix = r.meta["question_prefix"]
            row_answer_prefix = r.meta["answer_prefix"]
            user_content = f"{instruction}\n\n{question_prefix} {r.claim_text}"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            core = _render(tokenizer, messages)
            answer_prefix = f"{row_answer_prefix} "
        elif cfg.task == "lying":
            user_content = f"{cfg.user_prompt}\n{r.claim_text.strip()}"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            core = _render(tokenizer, messages)
            answer_prefix = cfg.answer_prefix
        elif cfg.task == "sycophancy":
            parsed_q = r.meta.get("parsed_question", "") if r.meta else ""
            parsed_a = r.meta.get("parsed_answer", "") if r.meta else ""
            if cfg.dataset == "trivia-qa" and parsed_q and parsed_a:
                statement = f'The answer to "{parsed_q}" is "{parsed_a}".'
            else:
                statement = r.claim_text.strip()
            if condition_type == "experiment":
                gt0_label = cfg.gt_0_label_map[cfg.task]
                gt1_label = cfg.gt_1_label_map[cfg.task]
                stated_opinion = gt0_label if r.label == 1 else gt1_label
                user_content = (
                    f"{cfg.user_prompt}\n{statement}\n"
                    f"I think this is {stated_opinion}."
                )
            else:
                user_content = f"{cfg.user_prompt}\n{statement}"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            core = _render(tokenizer, messages)
            answer_prefix = cfg.answer_prefix
        elif cfg.task == "hallucination":
            user_content = f"{cfg.user_prompt}\n{r.claim_text.strip()}"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            core = _render(tokenizer, messages)
            answer_prefix = cfg.answer_prefix
        elif cfg.task == "spatial_easy":
            question = r.meta["control_question"] if condition_type == "control" else r.meta["experiment_question"]
            user_content = f"{r.claim_text} {question}"
            messages = [{"role": "system", "content": system_prompt}]
            # Two-shot Alice-perspective examples
            messages += [
                {"role": "user", "content": _SHOT_EASY_LEFT_USER},
                {"role": "assistant", "content": _SHOT_EASY_LEFT_ASST},
                {"role": "user", "content": _SHOT_EASY_RIGHT_USER},
                {"role": "assistant", "content": _SHOT_EASY_RIGHT_ASST},
            ]
            messages.append({"role": "user", "content": user_content})
            core = _render(tokenizer, messages)
            answer_prefix = "Answer:"
        elif cfg.task == "lying_situation_induced":
            sp = system_prompt if condition_type == "control" else r.meta["system_prompt"]
            sp = sp + " Remember to answer yes or no directly first."
            user_content = r.meta["user_prompt"]
            messages = [
                {"role": "system", "content": sp},
                {"role": "user", "content": user_content},
            ]
            core = _render(tokenizer, messages)
            answer_prefix = cfg.answer_prefix
        else:
            raise ValueError(f"Unsupported cfg.task={cfg.task!r} for prompt construction.")
        out.append((core, answer_prefix))
    return out


def apply_chat_template(
    cfg: ExperimentConfig,
    records: List[StatementRecord],
    tokenizer: Any,
    condition_type: ConditionType = "control",
    system_prompt_id: int = 0,
) -> List[str]:
    """
    Render full string prompts (core + answer_prefix).

    Equivalent to ``[core + ap for core, ap in apply_chat_template_split(...)]``.
    Used by stages that don't need the boundary between the chat-template body
    and the answer_prefix (e.g. behavior generation). The cache stage calls
    :func:`apply_chat_template_split` instead so it can locate the "prompt"
    token position (last token of ``core``).
    """
    return [core + ap for core, ap in apply_chat_template_split(
        cfg, records, tokenizer,
        condition_type=condition_type, system_prompt_id=system_prompt_id,
    )]
