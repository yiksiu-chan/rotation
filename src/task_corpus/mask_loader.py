"""
cais/MASK loader (used by the ``lying_situation_induced`` task).

The MASK benchmark ships per-row ``system_prompt`` and ``user_prompt``
fields and a multiple-choice ground truth (``formatted_ground_truth``)
where ``"A"`` corresponds to "yes" and ``"B"`` to "no". Unlike the
azaria/triviaqa loaders, the per-row ``system_prompt`` IS the
experiment-condition system instruction, so it is preserved on
``StatementRecord.meta`` and read back by
:func:`src.prompt.chat_template.apply_chat_template`.

Label convention (matches ``gt_*_label_map`` for ``lying_situation_induced``):
    - ``formatted_ground_truth == "A"`` -> "yes" -> label = 1 (GT_1)
    - ``formatted_ground_truth == "B"`` -> "no"  -> label = 0 (GT_0)
Rows whose ``formatted_ground_truth`` is anything else are dropped.
"""

from __future__ import annotations

from typing import Any, List

from src.task_corpus.records import StatementRecord


_GT_TO_LABEL = {"A": 1, "B": 0}


def load_mask_records(hf_id: str, hf_config: str, hf_split: str) -> List[StatementRecord]:
    """
    Load every row of the configured MASK config as :class:`StatementRecord`.

    Args:
        hf_id: HF Hub dataset id (e.g. ``cais/MASK``).
        hf_config: Subset / config name (e.g. ``known_facts``).
        hf_split: Split name (e.g. ``test``).

    Returns:
        Records with ``claim_text`` set to the per-row ``user_prompt`` (treated
        as the "statement" downstream — the field the cache stage persists in
        ``outputs[*].statement``). The per-row ``system_prompt`` and
        ``user_prompt`` are also retained on ``meta`` so prompt construction
        can read them back. Rows with ``formatted_ground_truth`` outside
        ``{"A", "B"}`` are dropped.
    """
    # Local import so environments without HF stacks can import this module.
    from datasets import load_dataset

    dataset_all = load_dataset(hf_id, hf_config, split=hf_split)

    out: List[StatementRecord] = []
    for row in dataset_all:
        fgt = str(row.get("formatted_ground_truth", "")).strip()
        if fgt not in _GT_TO_LABEL:
            continue
        label = _GT_TO_LABEL[fgt]
        user_prompt = str(row["user_prompt"])
        system_prompt = str(row["system_prompt"])
        meta: dict[str, Any] = {k: row[k] for k in row.keys()}
        meta["system_prompt"] = system_prompt
        meta["user_prompt"] = user_prompt
        meta["formatted_ground_truth"] = fgt
        out.append(
            StatementRecord(
                index=len(out),
                claim_text=user_prompt,
                label=label,
                pair_id=None,
                meta=meta,
            )
        )
    return out
