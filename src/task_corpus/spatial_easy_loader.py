"""
Loader for the spatial-reasoning (easy) task.

Each row places two named people (``person_a`` and ``person_b``) directly
across from each other at a table, facing each other. An object is placed
on ``person_a``'s left or right side. Both POVs are queried with the same
question wording, only the subject differs:

- Control (person_a's POV):  "From {person_a}'s point of view, is the {object} on the left or the right side of the table?"
- Experiment (person_b's POV): "From {person_b}'s point of view, is the {object} on the left or the right side of the table?"

Because A and B sit directly opposite, every row is "always-flip" by
construction (A's left = B's right).

The chat template renders this task with no demos and no CoT trigger; see the
``cfg.task == "spatial_easy"`` branch in :mod:`src.prompt.chat_template`.

``label`` is the **control** answer (0 = Left, 1 = Right). The dataset is
balanced 50/50 across the two labels, so ``meta["experiment_answer"]`` is
always the opposite of ``label``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from src.task_corpus.records import StatementRecord


def load_spatial_easy_records(project_root: Path, dataset_path_relative: str) -> List[StatementRecord]:
    """
    Load the spatial-easy dataset and return one :class:`StatementRecord` per row.

    Args:
        project_root: Repository root used to resolve a relative dataset path.
        dataset_path_relative: Path under ``project_root`` (or absolute).

    Returns:
        List of records with ``claim_text`` = full scene, ``label`` = control
        answer (0=Left, 1=Right), and ``meta`` carrying the placement metadata
        plus ``control_question`` / ``experiment_question`` / ``experiment_answer``.
    """
    path = Path(dataset_path_relative)
    if not path.is_absolute():
        path = (project_root / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"spatial_easy dataset not found at {path}.")
    with open(path, encoding="utf-8") as f:
        blob = json.load(f)
    if not isinstance(blob, dict) or "rows" not in blob:
        raise ValueError(f"Expected top-level object with 'rows' key in {path}, got {type(blob)}")

    out: List[StatementRecord] = []
    for row in blob["rows"]:
        out.append(
            StatementRecord(
                index=int(row["index"]),
                claim_text=row["scene"],
                label=int(row["label"]),
                pair_id=None,
                meta={
                    "control_question": row["control_question"],
                    "experiment_question": row["experiment_question"],
                    "control_answer": row["control_answer"],
                    "experiment_answer": row["experiment_answer"],
                    "person_a": row["person_a"],
                    "person_b": row["person_b"],
                    "object": row["object"],
                    "placement": row["placement"],
                    "control_scene": row["scene"],
                },
            )
        )
    n_left = sum(1 for r in out if r.label == 0)
    n_right = sum(1 for r in out if r.label == 1)
    print(f"Loaded {len(out)} spatial_easy records (Left={n_left}, Right={n_right})")
    return out
