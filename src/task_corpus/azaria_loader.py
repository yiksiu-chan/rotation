"""
Azaria–Mitchell style claims via Hugging Face (aligned with ``task_and_dataset_lying``).

Uses ``load_dataset("jojoyang/azaria-mitchell_filtered_good")["combined"]``.
"""

from __future__ import annotations

from typing import Any, List

from src.task_corpus.records import StatementRecord


def load_azaria_records(hf_id: str, hf_split: str) -> List[StatementRecord]:
    """
    Load every row of the configured HF dataset as :class:`StatementRecord`.

    Args:
        hf_id: Hub dataset id (e.g. ``jojoyang/azaria-mitchell_filtered_good``).
        hf_split: Config/split name (e.g. ``combined``).

    Returns:
        Records with ``claim_text`` from the ``claim`` column and ``label`` int 0/1.
        The position of each record in the returned list is the global index used
        downstream.
    """
    # Local import so environments without Hugging Face stacks can still import TriviaQA-only code.
    from datasets import load_dataset

    dataset_all = load_dataset(hf_id)[hf_split]

    out: List[StatementRecord] = []
    for i, row in enumerate(dataset_all):
        claim = str(row["claim"])
        label = int(row["label"])
        meta: dict[str, Any] = {k: row[k] for k in row.keys()}
        out.append(
            StatementRecord(
                index=i,
                claim_text=claim,
                label=label,
                pair_id=None,
                meta=meta,
            )
        )
    return out
