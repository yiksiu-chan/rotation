"""Filesystem locations for paper-plot figures (under ``cfg.plot_root``)."""

from __future__ import annotations

from pathlib import Path

from src.config.schema import ExperimentConfig


def model_plot_root(cfg: ExperimentConfig) -> Path:
    """Return ``{plot_root}/{model_alias}``."""
    return cfg.plot_root / cfg.model_alias


def cos_sim_heatmap_basepath(cfg: ExperimentConfig, seed: int) -> Path:
    """``{plot_root}/{model_alias}/cos_sim_heatmaps/cos_heatmap_{task}_{dataset}_seed{n}`` (no extension)."""
    sub = model_plot_root(cfg) / "cos_sim_heatmaps"
    return sub / f"cos_heatmap_{cfg.task}_{cfg.dataset}_seed{seed}"


def best_token_position_basepath(cfg: ExperimentConfig, seed: int) -> Path:
    """``…/best_token_position/best_token_{task}_{dataset}_seed{n}`` (no extension)."""
    sub = model_plot_root(cfg) / "best_token_position"
    return sub / f"best_token_{cfg.task}_{cfg.dataset}_seed{seed}"


def rotation_distribution_basepath(cfg: ExperimentConfig, seed: int) -> Path:
    """``…/rotation_distribution/rotation_dist_{task}_{dataset}_seed{n}`` (no extension)."""
    sub = model_plot_root(cfg) / "rotation_distribution"
    return sub / f"rotation_dist_{cfg.task}_{cfg.dataset}_seed{seed}"


def cos_sim_curves_basepath(cfg: ExperimentConfig, seed: int) -> Path:
    """``…/cos_sim_curves/cos_sim_curves_{task}_{dataset}_seed{n}`` (no extension)."""
    sub = model_plot_root(cfg) / "cos_sim_curves"
    return sub / f"cos_sim_curves_{cfg.task}_{cfg.dataset}_seed{seed}"


def class_separation_basepath(cfg: ExperimentConfig, seed: int) -> Path:
    """
    ``…/class_separation/classifier_acc_curves_nPC{n}_{task}_{dataset}_seed{n}``
    (no extension).

    The ``n_PC`` suffix mirrors the per-seed file produced by the analysis
    stage so the aggregated mean-band plot and its inputs stay associated
    when ``cfg.n_PC_classifier`` is varied across reruns.
    """
    sub = model_plot_root(cfg) / "class_separation"
    return (
        sub
        / f"classifier_acc_curves_nPC{cfg.n_PC_classifier}_"
        f"{cfg.task}_{cfg.dataset}_seed{seed}"
    )


def class_separation_per_class_basepath(cfg: ExperimentConfig, seed: int) -> Path:
    """
    ``…/class_separation/classifier_acc_per_class_curves_nPC{n}_{task}_{dataset}_seed{n}``
    (no extension).

    Parallel to :func:`class_separation_basepath`; lives in the same
    subdirectory so all classifier-accuracy figures for a given seed land
    next to each other.
    """
    sub = model_plot_root(cfg) / "class_separation"
    return (
        sub
        / f"classifier_acc_per_class_curves_nPC{cfg.n_PC_classifier}_"
        f"{cfg.task}_{cfg.dataset}_seed{seed}"
    )


def class_separation_binary_basepath(cfg: ExperimentConfig, seed: int) -> Path:
    """
    ``…/class_separation/classifier_acc_binary_curves_nPC{n}_{task}_{dataset}_seed{n}``
    (no extension).

    Per-condition (control vs experiment) binary classifier figure — one
    binary classifier per condition; chance = 0.5.
    """
    sub = model_plot_root(cfg) / "class_separation"
    return (
        sub
        / f"classifier_acc_binary_curves_nPC{cfg.n_PC_classifier}_"
        f"{cfg.task}_{cfg.dataset}_seed{seed}"
    )


def behavior_basepath(cfg: ExperimentConfig) -> Path:
    """``…/behavior/behavior_BarBoxplot_{task}_{dataset}`` (no extension; not seed-keyed)."""
    sub = model_plot_root(cfg) / "behavior"
    return sub / f"behavior_BarBoxplot_{cfg.task}_{cfg.dataset}"


def no_rotation_analysis_basepath(cfg: ExperimentConfig, seed: int) -> Path:
    """``…/no_rotation_analysis/scatter_acc_vs_r_{task}_{dataset}_seed{n}`` (no extension)."""
    sub = model_plot_root(cfg) / "no_rotation_analysis"
    return sub / f"scatter_acc_vs_r_{cfg.task}_{cfg.dataset}_seed{seed}"


def cross_dataset_rotation_scatter_basepath(
    cfg: ExperimentConfig, seed: int, other_dataset: str
) -> Path:
    """``…/no_rotation_analysis/scatter_r_{task}_{dataset}_vs_{other_dataset}_seed{n}`` (no extension)."""
    sub = model_plot_root(cfg) / "no_rotation_analysis"
    return sub / f"scatter_r_{cfg.task}_{cfg.dataset}_vs_{other_dataset}_seed{seed}"
