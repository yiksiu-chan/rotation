"""
Load the per-model hallucination artifact produced by
``src/precompute_per_variant_consistency.py`` (per-variant labels). Also
reads an older single-label format (one ``hallucination_label`` per row,
no per-variant list) if present on disk, for backward compatibility with
artifacts generated before per-variant labels existed; nothing in this
repo generates that format anymore.

Each row is a TriviaQA T/F statement with a per-row, per-variant
``hallucination_label``:

    0 -> non-hallucination (control bucket; model answers correctly >= threshold_control)
    1 -> hallucination     (experiment bucket; model wrong >= 1-threshold_experiment)
    -1 -> ambiguous middle band (dropped under the dual-threshold policy)

The GT axis (True/False of the underlying claim) is preserved as
``StatementRecord.label`` so downstream balanced sampling and GT-split plots
keep their normal semantics. The condition axis is encoded per-row in
``meta["hallucination_label"]`` and is consumed by the behavior and cache
stages to partition rows between control and experiment forward passes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.task_corpus.records import StatementRecord
from src.task_corpus.triviaqa_loader import (
    is_wikipedia_page_name,
    parse_claim,
)


def _resolve_dataset_path(project_root: Path, model_alias: str, glob_pattern: str) -> Path:
    rel = glob_pattern.format(model_alias=model_alias)
    path = project_root / rel
    if not path.exists():
        raise FileNotFoundError(
            f"No hallucination artifact found for model_alias={model_alias!r} at {path}. "
            f"Run src/precompute_per_variant_consistency.py to generate it."
        )
    return path


def load_hallucination_records(
    project_root: Path,
    model_alias: str,
    hallucination_dataset_glob: str,
    variant_id: Optional[int] = None,
) -> List[StatementRecord]:
    """
    Load the baked artifact and return one :class:`StatementRecord` per row.

    Drops the same Wikipedia-style suspect pairs as
    :func:`load_triviaqa_filtered_records` so cross-task comparisons over
    TriviaQA stay row-aligned.

    Two artifact formats are supported:

    1. **Legacy single-label** (rows carry scalar ``hallucination_label``):
       returns every non-Wikipedia row; ``variant_id`` is ignored. Rows with
       ``hallucination_label == -1`` (ambiguous, dual-threshold mode) are
       dropped silently.

    2. **Per-variant** (rows carry ``hallucination_label_per_variant`` lists):
       ``variant_id`` is required. Returns rows where the V-th label is 0 or
       1; rows with ``hallucination_label_per_variant[V] == -1`` are dropped.
       The projected scalar is stored as ``meta["hallucination_label"]`` so
       downstream code is unchanged.

    Args:
        project_root: Repository root.
        model_alias: Used to locate the per-model file.
        hallucination_dataset_glob: Relative glob with ``{model_alias}`` placeholder.
        variant_id: Required for per-variant artifacts; ignored for legacy.

    Returns:
        :class:`StatementRecord` list. ``label`` is the original True/False GT;
        ``meta["hallucination_label"]`` (0/1) drives the condition partition.
    """
    path = _resolve_dataset_path(project_root, model_alias, hallucination_dataset_glob)
    with open(path, encoding="utf-8") as f:
        blob = json.load(f)
    if not isinstance(blob, dict) or "rows" not in blob:
        raise ValueError(
            f"Expected top-level object with 'rows' key in {path}, got {type(blob)}"
        )
    rows: List[Dict[str, Any]] = blob["rows"]

    is_per_variant = bool(rows) and "hallucination_label_per_variant" in rows[0]
    if is_per_variant:
        if variant_id is None:
            raise ValueError(
                f"{path.name} is a per-variant artifact but variant_id was not provided. "
                f"Pass variant_id explicitly when calling load_hallucination_records."
            )
        variant_ids = blob.get("variant_ids", list(range(len(rows[0]["hallucination_label_per_variant"]))))
        if variant_id not in variant_ids:
            raise ValueError(
                f"variant_id={variant_id} not found in artifact's variant_ids={variant_ids}."
            )
        v_index = variant_ids.index(variant_id)
    else:
        v_index = None

    suspect_pairs: set[Any] = set()
    for row in rows:
        ans, _ = parse_claim(row.get("claim", ""))
        if is_wikipedia_page_name(ans):
            suspect_pairs.add(row.get("pair_id"))

    filtered = [r for r in rows if r.get("pair_id") not in suspect_pairs]
    out: List[StatementRecord] = []
    n_dropped_ambiguous = 0
    for i, row in enumerate(filtered):
        if is_per_variant:
            labels = row.get("hallucination_label_per_variant")
            consistencies = row.get("consistency_per_variant", [])
            if labels is None or v_index >= len(labels):
                raise KeyError(
                    f"Row {i} missing valid hallucination_label_per_variant entry for "
                    f"variant_id={variant_id}; regenerate the artifact."
                )
            label_v = int(labels[v_index])
            consistency_v = float(consistencies[v_index]) if v_index < len(consistencies) else 0.0
        else:
            if "hallucination_label" not in row:
                raise KeyError(
                    f"Row {i} is missing 'hallucination_label'. This looks like an older, "
                    f"single-label artifact; regenerate it with "
                    f"src/precompute_per_variant_consistency.py instead."
                )
            label_v = int(row["hallucination_label"])
            consistency_v = float(row.get("consistency", 0.0))

        if label_v == -1:
            n_dropped_ambiguous += 1
            continue

        claim = row.get("claim", "")
        ans, q = parse_claim(claim)
        out.append(
            StatementRecord(
                index=i,
                claim_text=claim,
                label=int(row.get("label", 0)),
                pair_id=str(row.get("pair_id", "")),
                meta={
                    "parsed_answer": ans,
                    "parsed_question": q,
                    "hallucination_label": label_v,
                    "consistency": consistency_v,
                    **{k: v for k, v in row.items() if k not in (
                        "hallucination_label",
                        "hallucination_label_per_variant",
                        "consistency",
                        "consistency_per_variant",
                    )},
                },
            )
        )
    suffix = f" (variant={variant_id})" if is_per_variant else ""
    print(
        f"Loaded {len(out)} hallucination records{suffix} "
        f"(control={sum(1 for r in out if r.meta['hallucination_label'] == 0)}, "
        f"experiment={sum(1 for r in out if r.meta['hallucination_label'] == 1)}, "
        f"dropped_ambiguous={n_dropped_ambiguous})"
    )
    return out
