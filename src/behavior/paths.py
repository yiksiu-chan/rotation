"""Filesystem locations for behavior-stage artifacts."""

from __future__ import annotations

from pathlib import Path

from src.config.schema import ExperimentConfig


def behavior_run_dir(cfg: ExperimentConfig) -> Path:
    """
    Return ``{output_root}/{model_alias}/{task}_{dataset}``.

    Args:
        cfg: Experiment configuration.

    Returns:
        Base directory for control JSON and per-system-prompt subfolders.
    """
    return cfg.output_root / cfg.model_alias / cfg.task_dataset_slug()


def control_behavior_path(cfg: ExperimentConfig, system_prompt_id: int | None = None) -> Path:
    """
    Path to ``behavior_results_control.json``.

    For most tasks, control is variant-invariant and lives at the run-level
    directory. For hallucination, control varies per variant (each variant
    uses its own system prompt and its own row partition), so the file lives
    under ``SysPrompt{NNN}/`` alongside the experiment file.

    Args:
        cfg: Experiment configuration.
        system_prompt_id: If given, return the per-variant path under
            ``SysPrompt{id:03d}/``. If None, return the run-level path.
    """
    base = behavior_run_dir(cfg)
    if system_prompt_id is not None:
        base = base / f"SysPrompt{system_prompt_id:03d}"
    return base / "behavior_results_control.json"


def experiment_behavior_path(cfg: ExperimentConfig, system_prompt_id: int) -> Path:
    """Path to ``behavior_results_experiment.json`` for one system prompt index."""
    sub = behavior_run_dir(cfg) / f"SysPrompt{system_prompt_id:03d}"
    return sub / "behavior_results_experiment.json"
