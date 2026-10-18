"""
Real-world lying dataset loader (ecologically valid scenarios).

Each row is a role-play scenario with a paired ``normal_instruction`` (honest)
and ``deceive_instruction`` (lying). Unlike azaria/triviaqa, the control vs
experiment distinction lives **per-row** (in the instruction text and the
assistant ``answer_prefix``), not in a shared system prompt. The chat template
reads these from :attr:`StatementRecord.meta` to assemble each prompt.

Label convention (matches ``gt_*_label_map`` for ``lying_to_achieve_goals``):
    - ``honest_answer == "Yes"`` -> label = 1 (GT_1)
    - ``honest_answer == "No"``  -> label = 0 (GT_0)
Rows whose ``honest_answer`` is not exactly ``"Yes"`` or ``"No"`` are dropped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

from src.task_corpus.records import StatementRecord


_ALLOWED_HONEST_ANSWERS = {"Yes", "No"}


def load_real_world_lying_records(json_path: Path) -> List[StatementRecord]:
    """
    Read the real-world lying JSON file and yield normalized records.

    Args:
        json_path: Absolute path to the dataset file (e.g.
            ``datasets/real_world_lying/real_world_dataset_strong_gpt4o_deepseek.json``).

    Returns:
        Records whose ``meta`` dict carries the per-row strings needed by
        :func:`src.prompt.chat_template.apply_chat_template`:
        ``normal_instruction``, ``deceive_instruction``, ``question_prefix``,
        ``answer_prefix``, ``honest_answer``, ``deceive_answer``.

        ``claim_text`` holds the bare ``question`` string. The record's
        position in the returned list is the global index used downstream.

    Raises:
        FileNotFoundError: If ``json_path`` does not exist.
        ValueError: If the file is not a JSON list of row objects.
    """
    if not json_path.is_file():
        raise FileNotFoundError(f"Real-world lying dataset not found: {json_path}")

    with open(json_path, encoding="utf-8") as f:
        rows: Any = json.load(f)

    if not isinstance(rows, list):
        raise ValueError(
            f"Expected a JSON list at {json_path}, got {type(rows).__name__}."
        )

    out: List[StatementRecord] = []
    for row in rows:
        honest_answer = str(row.get("honest_answer", "")).strip()
        if honest_answer not in _ALLOWED_HONEST_ANSWERS:
            continue
        label = 1 if honest_answer == "Yes" else 0
        meta = {
            "normal_instruction": str(row["normal_instruction"]),
            "deceive_instruction": str(row["deceive_instruction"]),
            "question_prefix": str(row["question_prefix"]),
            "answer_prefix": str(row["answer_prefix"]),
            "honest_answer": honest_answer,
            "deceive_answer": str(row.get("deceive_answer", "")),
        }
        out.append(
            StatementRecord(
                index=len(out),
                claim_text=str(row["question"]),
                label=label,
                pair_id=None,
                meta=meta,
            )
        )
    return out
