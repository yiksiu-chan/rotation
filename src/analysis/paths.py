"""Filesystem locations for analysis-stage artifacts."""

from __future__ import annotations

from pathlib import Path

from src.cache_activations.paths import cache_seed_dir
from src.config.schema import ExperimentConfig


def stage_stats_dir(cfg: ExperimentConfig, system_prompt_id: int, seed: int) -> Path:
    """``…/SysPrompt{id:03d}/{seed}_pre{n}_gen{m}/stage_stats``."""
    return cache_seed_dir(cfg, system_prompt_id, seed) / "stage_stats"


def rotation_stats_path(cfg: ExperimentConfig, system_prompt_id: int, seed: int) -> Path:
    """Path to ``stage_stats/rotation_stats.json``."""
    return stage_stats_dir(cfg, system_prompt_id, seed) / "rotation_stats.json"


def cos_sim_heatmap_basepath(
    cfg: ExperimentConfig, system_prompt_id: int, seed: int
) -> Path:
    """
    Path **without extension** for the cross-position cosine heatmap.

    Both ``.pdf`` and ``.svg`` are written under this base by the plotting
    helper.
    """
    return stage_stats_dir(cfg, system_prompt_id, seed) / "cos_sim_heatmaps_across_tokens"


def class_separation_stats_path(
    cfg: ExperimentConfig, system_prompt_id: int, seed: int
) -> Path:
    """
    Path to ``stage_stats/class_separation_stats_nPC{n}.json``.

    The ``n_PC`` suffix mirrors the matching plot file
    (:func:`classifier_acc_curves_basepath`) so reruns with different
    ``cfg.n_PC_classifier`` don't overwrite each other's stats.
    """
    return (
        stage_stats_dir(cfg, system_prompt_id, seed)
        / f"class_separation_stats_nPC{cfg.n_PC_classifier}.json"
    )


def classifier_acc_curves_basepath(
    cfg: ExperimentConfig, system_prompt_id: int, seed: int
) -> Path:
    """
    Path **without extension** for the per-seed classifier-accuracy curve
    plot (``stage_stats/classifier_acc_curves_nPC{n}``).

    One file per entry in ``cfg.class_separation_plot_formats`` is written.
    """
    return (
        stage_stats_dir(cfg, system_prompt_id, seed)
        / f"classifier_acc_curves_nPC{cfg.n_PC_classifier}"
    )


def class_separation_per_class_stats_path(
    cfg: ExperimentConfig, system_prompt_id: int, seed: int
) -> Path:
    """
    Path to ``stage_stats/class_separation_per_class_stats_nPC{n}.json``.

    Parallel to :func:`class_separation_stats_path` but stores per-class
    CV recall curves (``cv_recall_per_class_across_layers``: a dict from
    class name to length-``n_layers`` list).
    """
    return (
        stage_stats_dir(cfg, system_prompt_id, seed)
        / f"class_separation_per_class_stats_nPC{cfg.n_PC_classifier}.json"
    )


def classifier_acc_per_class_curves_basepath(
    cfg: ExperimentConfig, system_prompt_id: int, seed: int
) -> Path:
    """
    Path **without extension** for the per-seed 4-line per-class
    classifier-accuracy curve plot
    (``stage_stats/classifier_acc_per_class_curves_nPC{n}``).
    """
    return (
        stage_stats_dir(cfg, system_prompt_id, seed)
        / f"classifier_acc_per_class_curves_nPC{cfg.n_PC_classifier}"
    )


def class_separation_binary_stats_path(
    cfg: ExperimentConfig, system_prompt_id: int, seed: int
) -> Path:
    """
    Path to ``stage_stats/class_separation_binary_stats_nPC{n}.json``.

    Stores per-condition binary classifier accuracy curves (one classifier
    per condition: control GT=0 vs control GT=1, and experiment GT=0 vs
    experiment GT=1). Schema:
    ``cv_accuracy_per_condition_across_layers``: dict from condition name
    to length-``n_layers`` list.
    """
    return (
        stage_stats_dir(cfg, system_prompt_id, seed)
        / f"class_separation_binary_stats_nPC{cfg.n_PC_classifier}.json"
    )


def classifier_acc_binary_curves_basepath(
    cfg: ExperimentConfig, system_prompt_id: int, seed: int
) -> Path:
    """
    Path **without extension** for the per-seed 2-line per-condition
    binary classifier-accuracy curve plot
    (``stage_stats/classifier_acc_binary_curves_nPC{n}``).
    """
    return (
        stage_stats_dir(cfg, system_prompt_id, seed)
        / f"classifier_acc_binary_curves_nPC{cfg.n_PC_classifier}"
    )


def pca_plots_dir(cfg: ExperimentConfig, system_prompt_id: int, seed: int) -> Path:
    """
    ``…/SysPrompt{id:03d}/{seed}_pre{n}_gen{m}/PCA_plots``.

    When ``cfg.pca_plots_output_root`` is set, mirrors the same
    ``{model_alias}/{task}_{dataset}/SysPrompt{NNN}/{seed}_pre{n}_gen{m}/PCA_plots``
    layout under that root instead — used to redirect the PCA grid output when
    the cached-pickle dir is read-only.
    """
    override = getattr(cfg, "pca_plots_output_root", None)
    if override:
        seed_subdir = f"{seed}_pre{cfg.n_pre_prompt_tokens}_gen{cfg.n_generated_tokens}"
        return (
            Path(override)
            / cfg.model_alias
            / cfg.task_dataset_slug()
            / f"SysPrompt{system_prompt_id:03d}"
            / seed_subdir
            / "PCA_plots"
        )
    return cache_seed_dir(cfg, system_prompt_id, seed) / "PCA_plots"


def pca_plot_basepath(
    cfg: ExperimentConfig,
    system_prompt_id: int,
    seed: int,
    position_label: str,
) -> Path:
    """
    Path **without extension** for one position's PCA scatter grid.

    Slashes / spaces in ``position_label`` are replaced so it is filesystem-safe
    (e.g. ``"prompt-1"`` stays ``"prompt-1"``, but odd labels are sanitized).
    """
    safe = position_label.replace("/", "_").replace(" ", "_")
    return pca_plots_dir(cfg, system_prompt_id, seed) / f"pca_{safe}"


# Manifold scatter output paths live on the run_microscopic CLI side now —
# see :mod:`src.run_microscopic` for the ``output_dir/manifolds/...`` layout.
