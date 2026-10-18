"""
Load and filter TriviaQA-derived true/false JSON (same spirit as ``run_triviaqa_strong_syco_v2``).

This module reimplements the loading logic in fresh code; it does not import the script.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, List, Sequence

from src.task_corpus.records import StatementRecord


def parse_claim(claim: str) -> tuple[str, str]:
    """
    Split a composite claim into answer string and question text when present.

    Args:
        claim: Raw ``claim`` field from the TriviaQA T/F JSON object.

    Returns:
        ``(answer_str, question)``; if the delimiter is absent, returns
        ``(claim.strip(), "")``.
    """
    if " is the answer to the question:" in claim:
        ans, q = claim.split(" is the answer to the question:", 1)
        return ans.strip(), q.strip().rstrip("?")
    return claim.strip(), ""


def is_wikipedia_page_name(answer_string: str) -> bool:
    """
    Return True if ``answer_string`` looks like a Wikipedia title or list page.

    Heuristic aligned with ``run_triviaqa_strong_syco_v2.is_wikipedia_page_name``.
    """
    if "(disambiguation)" in answer_string:
        return True
    if answer_string.startswith(
        ("Etymology of", "List of", "History of", "Outline of", "Index of", "Glossary of")
    ):
        return True
    if re.search(
        r"\((film|book|album|novel|song|TV series|video game|game|band|"
        r"character|series|magazine|company|musician|singer|writer|"
        r"actor|director|name|surname)\)",
        answer_string,
        re.IGNORECASE,
    ):
        return True
    if len(answer_string) > 80:
        return True
    return False


def _resolve_dataset_path(project_root: Path, model_alias: str, glob_pattern: str) -> Path:
    """Resolve the configured glob; raise if the file is missing."""
    rel = glob_pattern.format(model_alias=model_alias)
    path = project_root / rel
    if not path.exists():
        raise FileNotFoundError(
            f"No TriviaQA JSON found for model_alias={model_alias!r} at {path}."
        )
    return path


def load_triviaqa_filtered_records(
    project_root: Path,
    model_alias: str,
    triviaqa_dataset_glob: str,
) -> List[StatementRecord]:
    """
    Load JSON rows, drop suspect Wikipedia-style pairs, attach parsed_answer/question.

    Args:
        project_root: Repository root (contains ``datasets/triviaqa_tf`` by default).
        model_alias: Used to locate specific json files.
        triviaqa_dataset_glob: Relative glob pattern with ``{model_alias}`` placeholder.

    Returns:
        List of :class:`StatementRecord` with ``claim_text`` equal to the original ``claim``.
    """
    path = _resolve_dataset_path(project_root, model_alias, triviaqa_dataset_glob)
    with open(path, encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    suspect_pairs: set[Any] = set()
    for row in data:
        claim = row.get("claim", "")
        ans, _ = parse_claim(claim)
        if is_wikipedia_page_name(ans):
            suspect_pairs.add(row["pair_id"])

    filtered = [row for row in data if row["pair_id"] not in suspect_pairs]
    out: List[StatementRecord] = []
    for i, row in enumerate(filtered):
        claim = row.get("claim", "")
        ans, q = parse_claim(claim)
        label = int(row.get("label", 0))
        out.append(
            StatementRecord(
                index=i,
                claim_text=claim,
                label=label,
                pair_id=str(row.get("pair_id", "")),
                meta={"parsed_answer": ans, "parsed_question": q, **row},
            )
        )
    print(f"Loaded {len(out)} records from the dataset")
    return out


def balanced_subsample_indices(
    labels: Sequence[int], k_per_class: int, seed: int
) -> List[int]:
    """
    Return a class-balanced sample of indices into ``labels``.

    Picks up to ``k_per_class`` indices with label ``0`` and ``k_per_class`` with
    label ``1`` (without replacement). Returns the union as a sorted list. The
    achieved per-class count may be smaller when a class has fewer than
    ``k_per_class`` rows.

    Sentinel: ``k_per_class == -1`` disables sampling entirely and returns
    every index (both classes), sorted. The caller gets the full corpus
    without any per-class equalization.

    Args:
        labels: ``labels[i]`` = GT label of statement ``i``.
        k_per_class: Desired count per GT class, or ``-1`` to use all rows.
        seed: RNG seed for reproducibility.

    Returns:
        Sorted list of selected indices into the full statement list.
    """
    import random

    if k_per_class == -1:
        return list(range(len(labels)))

    pool0 = [i for i, lab in enumerate(labels) if lab == 0]
    pool1 = [i for i, lab in enumerate(labels) if lab == 1]
    rng = random.Random(seed)
    k0 = min(k_per_class, len(pool0))
    k1 = min(k_per_class, len(pool1))
    take0 = rng.sample(pool0, k0)
    take1 = rng.sample(pool1, k1)
    return sorted(take0 + take1)
