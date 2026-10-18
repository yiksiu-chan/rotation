"""Single dataset-dispatch entry point shared by every stage/script that
needs the full statement corpus for the configured task.

Previously hand-duplicated (identically) in ``src.behavior.run_behavior`` and
``src.cache_activations.cache_activations``, and reached into from
``src.activation_patching`` via ``cache_activations``'s private
``_load_statements`` -- all four now import this instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from src.config.schema import ExperimentConfig
from src.task_corpus.azaria_loader import load_azaria_records
from src.task_corpus.hallucination_loader import load_hallucination_records
from src.task_corpus.mask_loader import load_mask_records
from src.task_corpus.real_world_lying_loader import load_real_world_lying_records
from src.task_corpus.records import StatementRecord
from src.task_corpus.spatial_easy_loader import load_spatial_easy_records
from src.task_corpus.triviaqa_loader import load_triviaqa_filtered_records


def load_statements_for_config(cfg: ExperimentConfig) -> List[StatementRecord]:
    """
    Load normalized statements for ``cfg.dataset``.

    Returns the full filtered corpus; balanced sampling happens downstream so
    global indices remain stable across the behavior, cache, and
    activation-patching stages.

    Args:
        cfg: Must specify a ``dataset`` recognized below.

    Returns:
        List of :class:`StatementRecord` instances.

    Raises:
        ValueError: If ``dataset`` is unknown.
    """
    if cfg.dataset == "trivia-qa":
        return load_triviaqa_filtered_records(
            cfg.project_root,
            cfg.model_alias,
            cfg.triviaqa_dataset_glob,
        )
    if cfg.dataset == "triviaqa-hallucination":
        return load_hallucination_records(
            cfg.project_root,
            cfg.model_alias,
            cfg.triviaqa_hallucination_dataset_glob,
        )
    if cfg.dataset == "azaria-mitchell":
        return load_azaria_records(cfg.azaria_hf_dataset, cfg.azaria_hf_split)
    if cfg.dataset in cfg.real_world_lying_dataset_paths:
        path = Path(cfg.real_world_lying_dataset_paths[cfg.dataset])
        if not path.is_absolute():
            path = (cfg.project_root / path).resolve()
        return load_real_world_lying_records(path)
    if cfg.dataset in cfg.mask_dataset_configs:
        return load_mask_records(cfg.mask_hf_dataset, cfg.dataset, cfg.mask_hf_split)
    if cfg.dataset == "spatial-easy":
        return load_spatial_easy_records(cfg.project_root, cfg.spatial_easy_dataset_path)
    raise ValueError(f"Unsupported dataset: {cfg.dataset!r}")
