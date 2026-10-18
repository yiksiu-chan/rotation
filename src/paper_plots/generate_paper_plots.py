"""
Paper-plot stage: cross-system-prompt aggregation figures (Plotly).

Per seed in ``cfg.seed_list_cache``:

- ``cos_sim_heatmaps/cos_heatmap_{task}_{dataset}_seed{n}.{svg,png}`` —
  cosine heatmap (rows = system prompts at their own best position under
  ``cfg.rotation_metric``, ordered by descending rotation; columns = layers
  + a trailing ``r1, r2, …`` block of metric values). Heatmaps are written
  as SVG + high-resolution PNG.
- ``best_token_position/best_token_{task}_{dataset}_seed{n}.{pdf,svg}`` —
  bar chart over canonical token positions, height = variant count.
- ``rotation_distribution/rotation_dist_{task}_{dataset}_seed{n}.{pdf,svg}``
  — box+points distribution of ``rotation_r`` (under ``cfg.rotation_metric``,
  at each variant's best position) across system prompts.
- ``cos_sim_curves/cos_sim_curves_{task}_{dataset}_seed{n}.{pdf,svg}`` —
  mean ± variance-band cosine-similarity curve across system prompts (each
  variant's ``cos_sim_across_layers`` taken at its own best position under
  ``cfg.rotation_metric``). Band kind is configured by
  ``cfg.cos_sim_curves_band_kind`` (``std`` | ``sem`` | ``ci95``).

Across seeds (single, non-seed-keyed file):

- ``behavior/behavior_BarBoxplot_{task}_{dataset}.{pdf,svg}`` — bar at the
  control accuracy and a box of experiment accuracies across variants.
- ``summary_{task}_{dataset}.json`` (also printed to stdout) — the plain-
  language answer for this run: control accuracy, mean experiment flip rate,
  and mean rotation (under ``cfg.rotation_metric``) per seed and overall. Read
  this first before digging into any other artifact below.

All artifacts land under ``{plot_root}/{model_alias}/``. Variants whose
upstream JSON files are missing are silently skipped, so partial sweeps still
produce best-effort figures.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from src.cache_activations.hooks import position_labels
from src.config.cli import add_common_args, load_config_from_cli
from src.config.run_record import dump_run_record
from src.config.schema import ExperimentConfig
from src.paper_plots.aggregate import (
    BehaviorAggregate,
    ClassSeparationAggregate,
    ClassSeparationBinaryAggregate,
    ClassSeparationPerClassAggregate,
    SeedAggregate,
    load_behavior_aggregate,
    load_class_separation_aggregate_for_seed,
    load_class_separation_binary_aggregate_for_seed,
    load_class_separation_per_class_aggregate_for_seed,
    load_rotation_aggregate_for_seed,
)
from src.paper_plots.paths import (
    behavior_basepath,
    best_token_position_basepath,
    class_separation_basepath,
    class_separation_binary_basepath,
    class_separation_per_class_basepath,
    cos_sim_curves_basepath,
    cos_sim_heatmap_basepath,
    cross_dataset_rotation_scatter_basepath,
    model_plot_root,
    no_rotation_analysis_basepath,
    rotation_distribution_basepath,
)
from src.paths import PROJECT_ROOT
from src.prompt.chat_template import load_experiment_system_prompts


# Canonical short keys used by ``cfg.paper_plots_to_generate`` /
# ``cfg.paper_plot_formats`` to gate each figure and pick its output formats.
# Keep these in sync with ``ExperimentConfig.paper_plots_to_generate``'s
# default factory in :mod:`src.config.schema`.
_FIG_COS_SIM_HEATMAP = "cos_sim_heatmap"
_FIG_BEST_TOKEN_POSITION = "best_token_position"
_FIG_ROTATION_DISTRIBUTION = "rotation_distribution"
_FIG_ACCURACY_VS_ROTATION = "accuracy_vs_rotation_scatter"
_FIG_CROSS_DATASET = "cross_dataset_scatter"
_FIG_BEHAVIOR_BAR_BOX = "behavior_bar_box"
_FIG_CLASS_SEPARATION = "class_separation"
_FIG_CLASS_SEPARATION_PER_CLASS = "class_separation_per_class"
_FIG_CLASS_SEPARATION_BINARY = "class_separation_binary"
_FIG_COS_SIM_CURVES = "cos_sim_curves"


def _is_enabled(cfg: ExperimentConfig, key: str) -> bool:
    """Look up ``cfg.paper_plots_to_generate[key]``, defaulting to True."""
    toggles = getattr(cfg, "paper_plots_to_generate", None) or {}
    return bool(toggles.get(key, True))


def _formats_for(cfg: ExperimentConfig, key: str) -> Optional[List[str]]:
    """
    Look up ``cfg.paper_plot_formats[key]``; ``None`` means "use the figure
    function's default". Empty lists are treated the same as ``None`` so a
    user can drop a key without disabling the figure.
    """
    fmt_map = getattr(cfg, "paper_plot_formats", None) or {}
    fmts = fmt_map.get(key)
    if not fmts:
        return None
    return list(fmts)


# --------------------------------------------------------------------------- #
# Per-seed dispatch                                                           #
# --------------------------------------------------------------------------- #


def _figure_title_seed(cfg: ExperimentConfig, seed: int, suffix: str) -> str:
    return f"{cfg.model_alias} | {cfg.task}/{cfg.dataset} | seed={seed} — {suffix}"


def _plot_one_seed(
    cfg: ExperimentConfig,
    seed_data: SeedAggregate,
    class_sep_data: ClassSeparationAggregate,
    class_sep_per_class_data: ClassSeparationPerClassAggregate,
    class_sep_binary_data: ClassSeparationBinaryAggregate,
    accuracy_by_variant_id: Dict[int, float],
) -> int:
    """
    Emit the per-seed figures. Returns the count of figures written.

    The accuracy-vs-rotation scatter is skipped (without failing the seed)
    when no variant has both a rotation summary and a behavior experiment
    JSON; the other three figures always run on a non-empty seed. The
    class-separation mean-band figure is skipped when ``class_sep_data`` is
    empty (no ``class_separation_stats_nPC{n}.json`` found across variants)
    independently of ``seed_data``.
    """
    if (
        seed_data.is_empty
        and class_sep_data.is_empty
        and class_sep_per_class_data.is_empty
        and class_sep_binary_data.is_empty
    ):
        return 0

    # Lazy plot import: keeps `import src.paper_plots` cheap.
    from src.paper_plots.figures import (
        plot_accuracy_vs_rotation_scatter,
        plot_best_token_distribution,
        plot_class_separation_binary_curves_mean_band,
        plot_class_separation_curves_mean_band,
        plot_class_separation_per_class_curves_mean_band,
        plot_cos_sim_curves_across_prompts,
        plot_cos_sim_heatmap_across_prompts,
        plot_rotation_distribution,
    )

    pos_labels = position_labels(cfg.n_pre_prompt_tokens, cfg.n_generated_tokens)
    if not seed_data.is_empty:
        seed = seed_data.seed
    elif not class_sep_data.is_empty:
        seed = class_sep_data.seed
    elif not class_sep_per_class_data.is_empty:
        seed = class_sep_per_class_data.seed
    else:
        seed = class_sep_binary_data.seed
    written = 0
    if not seed_data.is_empty:
        if _is_enabled(cfg, _FIG_COS_SIM_HEATMAP):
            plot_cos_sim_heatmap_across_prompts(
                seed_data,
                basepath=cos_sim_heatmap_basepath(cfg, seed),
                title=_figure_title_seed(cfg, seed, "cosine sim across system prompts (best token)"),
                metric_display_mode=cfg.heatmap_r_metric_display,
                label_display=cfg.heatmap_label_display,
                formats=_formats_for(cfg, _FIG_COS_SIM_HEATMAP),
            )
            written += 1
        if _is_enabled(cfg, _FIG_BEST_TOKEN_POSITION):
            plot_best_token_distribution(
                seed_data,
                canonical_position_labels=pos_labels,
                basepath=best_token_position_basepath(cfg, seed),
                title=_figure_title_seed(cfg, seed, "best token position distribution"),
                formats=_formats_for(cfg, _FIG_BEST_TOKEN_POSITION),
            )
            written += 1
        if _is_enabled(cfg, _FIG_ROTATION_DISTRIBUTION):
            plot_rotation_distribution(
                seed_data,
                basepath=rotation_distribution_basepath(cfg, seed),
                title=_figure_title_seed(cfg, seed, "rotation_r distribution"),
                formats=_formats_for(cfg, _FIG_ROTATION_DISTRIBUTION),
            )
            written += 1
        if _is_enabled(cfg, _FIG_COS_SIM_CURVES):
            plot_cos_sim_curves_across_prompts(
                seed_data,
                basepath=cos_sim_curves_basepath(cfg, seed),
                title=_figure_title_seed(
                    cfg,
                    seed,
                    f"cosine sim across system prompts "
                    f"(mean ± {cfg.cos_sim_curves_band_kind}, best token)",
                ),
                band_kind=cfg.cos_sim_curves_band_kind,
                formats=_formats_for(cfg, _FIG_COS_SIM_CURVES),
                line_color=cfg.cos_sim_curves_line_color,
                line_width=cfg.cos_sim_curves_line_width,
                band_color=cfg.cos_sim_curves_band_color,
                band_alpha=cfg.cos_sim_curves_band_alpha,
                y_range=tuple(cfg.cos_sim_curves_y_range),
            )
            written += 1

        if _is_enabled(cfg, _FIG_ACCURACY_VS_ROTATION):
            has_overlap = any(
                v.system_prompt_id in accuracy_by_variant_id for v in seed_data.variants
            )
            if has_overlap:
                plot_accuracy_vs_rotation_scatter(
                    seed_data,
                    accuracy_by_variant_id=accuracy_by_variant_id,
                    basepath=no_rotation_analysis_basepath(cfg, seed),
                    title=_figure_title_seed(
                        cfg, seed, "experiment accuracy vs rotation_r (per system prompt)"
                    ),
                    x_axis_label=f"{cfg.task}_instruction_following_rate",
                    formats=_formats_for(cfg, _FIG_ACCURACY_VS_ROTATION),
                )
                written += 1
            else:
                print(
                    f"  skip accuracy-vs-rotation scatter for seed={seed}: "
                    "no variant has both rotation_stats.json and behavior_results_experiment.json"
                )

    if _is_enabled(cfg, _FIG_CLASS_SEPARATION) and not class_sep_data.is_empty:
        plot_class_separation_curves_mean_band(
            class_sep_data,
            basepath=class_separation_basepath(cfg, seed),
            title=_figure_title_seed(
                cfg,
                seed,
                f"4-class classifier CV accuracy (mean ± std, "
                f"nPC={class_sep_data.n_PC_classifier})",
            ),
            formats=_formats_for(cfg, _FIG_CLASS_SEPARATION)
            or list(cfg.class_separation_plot_formats),
            line_color=cfg.class_separation_line_color,
            line_width=cfg.class_separation_line_width,
            band_color=cfg.class_separation_band_color,
            band_alpha=cfg.class_separation_band_alpha,
            y_range=tuple(cfg.class_separation_y_range),
        )
        written += 1
    if (
        _is_enabled(cfg, _FIG_CLASS_SEPARATION_PER_CLASS)
        and not class_sep_per_class_data.is_empty
    ):
        plot_class_separation_per_class_curves_mean_band(
            class_sep_per_class_data,
            basepath=class_separation_per_class_basepath(cfg, seed),
            title=_figure_title_seed(
                cfg,
                seed,
                f"per-class classifier CV recall (mean ± std, "
                f"nPC={class_sep_per_class_data.n_PC_classifier})",
            ),
            formats=_formats_for(cfg, _FIG_CLASS_SEPARATION_PER_CLASS)
            or list(cfg.class_separation_plot_formats),
            line_width=cfg.class_separation_line_width,
            band_alpha=cfg.class_separation_band_alpha,
            y_range=tuple(cfg.class_separation_y_range),
        )
        written += 1
    if (
        _is_enabled(cfg, _FIG_CLASS_SEPARATION_BINARY)
        and not class_sep_binary_data.is_empty
    ):
        plot_class_separation_binary_curves_mean_band(
            class_sep_binary_data,
            basepath=class_separation_binary_basepath(cfg, seed),
            title=_figure_title_seed(
                cfg,
                seed,
                f"per-condition binary classifier CV accuracy (mean ± std, "
                f"nPC={class_sep_binary_data.n_PC_classifier})",
            ),
            formats=_formats_for(cfg, _FIG_CLASS_SEPARATION_BINARY)
            or list(cfg.class_separation_plot_formats),
            line_width=cfg.class_separation_line_width,
            band_alpha=cfg.class_separation_band_alpha,
            y_range=tuple(cfg.class_separation_y_range),
        )
        written += 1
    return written


def _plot_cross_dataset_scatters_for_seed(
    cfg: ExperimentConfig,
    seed: int,
    seed_data_x: SeedAggregate,
    variants: List[str],
) -> int:
    """
    For each entry in ``cfg.compare_datasets``, emit one rotation_r scatter
    (cfg.dataset on x vs the other dataset on y) for the given seed. Returns
    the number of figures actually written.
    """
    if not cfg.compare_datasets:
        return 0
    if seed_data_x.is_empty:
        return 0
    if not _is_enabled(cfg, _FIG_CROSS_DATASET):
        return 0

    from src.paper_plots.figures import plot_rotation_r_cross_dataset_scatter

    fmts = _formats_for(cfg, _FIG_CROSS_DATASET)
    written = 0
    for other_dataset in cfg.compare_datasets:
        if other_dataset == cfg.dataset:
            continue
        cfg_other = dataclasses.replace(cfg, dataset=other_dataset)
        seed_data_y = load_rotation_aggregate_for_seed(cfg_other, seed, variants)
        if seed_data_y.is_empty:
            print(
                f"  skip cross-dataset scatter ({cfg.dataset} vs {other_dataset}) "
                f"for seed={seed}: no rotation_stats.json found under {other_dataset}"
            )
            continue
        plot_rotation_r_cross_dataset_scatter(
            seed_data_x=seed_data_x,
            seed_data_y=seed_data_y,
            x_dataset_name=cfg.dataset,
            y_dataset_name=other_dataset,
            basepath=cross_dataset_rotation_scatter_basepath(cfg, seed, other_dataset),
            title=_figure_title_seed(
                cfg,
                seed,
                f"rotation_r across datasets: {cfg.dataset} vs {other_dataset}",
            ),
            formats=fmts,
        )
        written += 1
    return written


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #


def generate_paper_plots_for_config(cfg: ExperimentConfig) -> None:
    """
    Build all paper figures for one config.

    Args:
        cfg: Experiment configuration. ``cfg.seed_list_cache`` enumerates the
            per-seed plots; the behavior figure is single (no seed in name).

    Raises:
        ValueError: If ``seed_list_cache`` is empty.
    """
    seeds = list(cfg.seed_list_cache or [])
    if not seeds:
        raise ValueError("seed_list_cache is empty; nothing to plot.")

    variants = load_experiment_system_prompts(cfg)
    n_variants = len(variants)

    # Behavior is seed-independent; load once and reuse for both the per-seed
    # accuracy-vs-rotation scatter and the across-seed bar/box plot below.
    behavior = load_behavior_aggregate(cfg, n_variants)
    accuracy_by_variant_id: Dict[int, float] = dict(
        zip(behavior.variant_ids_with_experiment, behavior.accuracy_experiments)
    )

    n_per_seed = 0
    n_seeds_with_data = 0
    mean_rotation_r_by_seed: Dict[int, float] = {}
    for seed in seeds:
        seed_data = load_rotation_aggregate_for_seed(cfg, seed, variants)
        class_sep_data = load_class_separation_aggregate_for_seed(
            cfg, seed, variants
        )
        class_sep_per_class_data = (
            load_class_separation_per_class_aggregate_for_seed(
                cfg, seed, variants
            )
        )
        class_sep_binary_data = (
            load_class_separation_binary_aggregate_for_seed(
                cfg, seed, variants
            )
        )
        if (
            seed_data.is_empty
            and class_sep_data.is_empty
            and class_sep_per_class_data.is_empty
            and class_sep_binary_data.is_empty
        ):
            print(
                f"  skip seed={seed}: no rotation_stats.json or "
                "class_separation_stats_*.json found across variants"
            )
            continue
        written = _plot_one_seed(
            cfg,
            seed_data,
            class_sep_data,
            class_sep_per_class_data,
            class_sep_binary_data,
            accuracy_by_variant_id,
        )
        written += _plot_cross_dataset_scatters_for_seed(
            cfg, seed, seed_data, variants
        )
        n_per_seed += written
        n_seeds_with_data += 1
        if not seed_data.is_empty:
            mean_rotation_r_by_seed[seed] = sum(
                v.rotation_r for v in seed_data.variants
            ) / len(seed_data.variants)
        print(
            f"  seed={seed}: wrote {written} figures "
            f"(rotation variants: {len(seed_data.variants)}/{n_variants}; "
            f"class-separation variants: {len(class_sep_data.variants)}/{n_variants}; "
            f"per-class variants: {len(class_sep_per_class_data.variants)}/{n_variants}; "
            f"binary variants: {len(class_sep_binary_data.variants)}/{n_variants})"
        )

    # Behavior bar/box figure (no seed dimension).
    if not _is_enabled(cfg, _FIG_BEHAVIOR_BAR_BOX):
        print("  skip behavior plot: disabled via paper_plots_to_generate")
    elif not behavior.has_data:
        print("  skip behavior plot: no behavior_results_*.json found")
    else:
        from src.paper_plots.figures import plot_behavior_bar_box

        plot_behavior_bar_box(
            behavior,
            basepath=behavior_basepath(cfg),
            title=(
                f"{cfg.model_alias} | {cfg.task}/{cfg.dataset} — "
                "behavior accuracy (control bar / experiment box across variants)"
            ),
            formats=_formats_for(cfg, _FIG_BEHAVIOR_BAR_BOX),
        )
        print(
            f"  behavior plot written ("
            f"control={'set' if behavior.accuracy_control is not None else 'missing'}, "
            f"experiment_n={len(behavior.accuracy_experiments)})"
        )

    print(
        f"Paper plots complete: {n_per_seed} per-seed figures across "
        f"{n_seeds_with_data} seed(s)."
    )

    _write_summary(cfg, behavior, mean_rotation_r_by_seed)


def _write_summary(
    cfg: ExperimentConfig,
    behavior: BehaviorAggregate,
    mean_rotation_r_by_seed: Dict[int, float],
) -> None:
    """
    Print (and persist as ``summary.json``) the plain-language answer this
    task/dataset/model run gives: mean flip rate and mean rotation, so a
    first-time user doesn't have to open nested ``stage_stats/*.json`` files
    to find out "did it work".
    """
    mean_flip_rate = (
        sum(behavior.accuracy_experiments) / len(behavior.accuracy_experiments)
        if behavior.accuracy_experiments
        else None
    )
    overall_mean_rotation_r = (
        sum(mean_rotation_r_by_seed.values()) / len(mean_rotation_r_by_seed)
        if mean_rotation_r_by_seed
        else None
    )
    summary = {
        "model_alias": cfg.model_alias,
        "task": cfg.task,
        "dataset": cfg.dataset,
        "rotation_metric": cfg.rotation_metric,
        "accuracy_control": behavior.accuracy_control,
        "mean_flip_rate_experiment": mean_flip_rate,
        "n_variants_with_behavior": len(behavior.accuracy_experiments),
        "mean_rotation_r_by_seed": mean_rotation_r_by_seed,
        "overall_mean_rotation_r": overall_mean_rotation_r,
    }
    fmt = lambda x: "n/a" if x is None else f"{x:.3f}"
    print(
        f"Summary [{cfg.model_alias} | {cfg.task}/{cfg.dataset}]: "
        f"control accuracy={fmt(behavior.accuracy_control)}, "
        f"mean flip rate (experiment, n={len(behavior.accuracy_experiments)})={fmt(mean_flip_rate)}, "
        f"mean rotation ({cfg.rotation_metric})={fmt(overall_mean_rotation_r)}"
    )
    out_path = model_plot_root(cfg) / f"summary_{cfg.task}_{cfg.dataset}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """CLI for the paper-plot stage."""
    p = argparse.ArgumentParser(description="Generate cross-prompt paper figures.")
    add_common_args(p)
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    """Entry point: load config and run :func:`generate_paper_plots_for_config`."""
    args = parse_args(argv)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    cfg = load_config_from_cli(args)
    dump_run_record(
        cfg,
        out_dir=model_plot_root(cfg),
        argv=sys.argv,
        stage="plots",
    )
    generate_paper_plots_for_config(cfg)


if __name__ == "__main__":
    main()
