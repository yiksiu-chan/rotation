"""
Analysis stage: per-position cosine matrices, PCA scatter grids, and per-metric
rotation scalars.

For every ``(system_prompt_id, seed)`` directory produced by the cache stage:

1. Load the four activation pickles
   (``activations_{control,experiment}_GT_{0,1}.pkl``).
2. Compute the per-``(layer, position)`` cosine matrix between the control and
   experiment contrast directions
   (:func:`src.analysis.metrics.compute_cos_sim_matrix`).
3. Apply **every** registered rotation metric in
   :data:`src.analysis.rotation_plugins.ROTATION_METRIC_REGISTRY` to each token
   position; the metric named by ``cfg.rotation_metric`` drives PCA selection.
4. Persist:

   - ``stage_stats/rotation_stats.json`` — per-position
     ``{rotation_r: {metric_name: float, …}, cos_sim_across_layers}`` plus
     top-level ``best_token_pos_<metric>`` and ``ranked_token_pos_<metric>``
     for each registered metric.
   - ``stage_stats/cos_sim_heatmaps_across_tokens.<fmt>`` — y=positions,
     x=layers heatmap with a trailing block of ``r1, r2, …`` columns whose
     cells are annotated with each metric's value (rounded to one decimal).
     Output formats are controlled by ``cfg.cos_sim_heatmap_formats``
     (default ``["svg", "png"]``).
   - ``PCA_plots/pca_<position>.{pdf,svg}`` — per-layer PCA scatter grid for
     each token position; aesthetics from ``cfg.pca_plot_style``. The PCA
     position selection follows ``ranked_token_pos_<cfg.rotation_metric>``.
     Disabled when ``cfg.save_PCA_plots`` is ``false``.
   Manifold scatter plots (across-layer and set-layer PCA / t-SNE / UMAP)
   are **not** produced by this stage. They live behind
   :mod:`src.run_microscopic`; its ``--manifold_plots`` flag drives that
   output independently.

Skip-with-warning behavior: any ``(system_prompt_id, seed)`` whose four
activation pickles are not all present is silently skipped (printed once), so a
partial cache run can be analyzed incrementally. Set
``cfg.rerun_cache_in_analysis = True`` to invoke the cache stage from the
analysis CLI first (regenerating pickles for the same variant range), then
proceed with the analysis loop.

Variant range: every iteration goes through ``cfg.variant_indices(...)`` so
``cfg.start_variant`` / ``cfg.end_variant`` consistently bound which system
prompts are processed — both for the optional cache rerun and the
plotting / cosine / rotation passes.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.analysis.class_separation import (
    compute_binary_per_condition_cv_accuracy_per_layer,
    compute_cv_accuracy_per_layer,
    compute_per_class_cv_accuracy_per_layer,
)
from src.analysis.metrics import compute_cos_sim_matrix
from src.analysis.paths import (
    class_separation_binary_stats_path,
    class_separation_per_class_stats_path,
    class_separation_stats_path,
    classifier_acc_binary_curves_basepath,
    classifier_acc_curves_basepath,
    classifier_acc_per_class_curves_basepath,
    cos_sim_heatmap_basepath,
    pca_plot_basepath,
    rotation_stats_path,
    stage_stats_dir,
)
from src.analysis.rotation_plugins import (
    ROTATION_METRIC_ORDER,
    ROTATION_METRIC_REGISTRY,
    metric_short_label,
    resolve_displayed_metrics,
    resolve_rotation_metric,
)
from src.behavior.paths import behavior_run_dir
from src.cache_activations.hooks import position_labels
from src.cache_activations.paths import activations_path, completions_path
from src.config.cli import add_common_args, load_config_from_cli
from src.config.run_record import dump_run_record
from src.config.schema import ExperimentConfig
from src.paths import PROJECT_ROOT
from src.prompt.chat_template import load_experiment_system_prompts


# --------------------------------------------------------------------------- #
# I/O helpers                                                                 #
# --------------------------------------------------------------------------- #


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON atomically (matches the cache stage's convention)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def _load_pickle_as_numpy(path: Path) -> np.ndarray:
    """Load a torch tensor pickle and return it as a NumPy float32 array."""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    # The cache stage pickles torch tensors; convert to numpy without keeping
    # torch in this module's hot path (caller already imported torch transitively
    # via the cache stage when producing the file).
    if hasattr(obj, "numpy"):
        return obj.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(obj, dtype=np.float32)


def _load_four_activation_pickles(
    cfg: ExperimentConfig, system_prompt_id: int, seed: int
) -> Optional[Dict[str, np.ndarray]]:
    """
    Return the four activation arrays as a dict, or ``None`` if any are missing.

    Keys: ``"control_GT_0"``, ``"control_GT_1"``, ``"experiment_GT_0"``,
    ``"experiment_GT_1"`` — same order used by the PCA plotting helper.
    """
    out: Dict[str, np.ndarray] = {}
    for cond in ("control", "experiment"):
        for gt in (0, 1):
            p = activations_path(cfg, system_prompt_id, seed, cond, gt)
            if not p.exists():
                return None
            out[f"{cond}_GT_{gt}"] = _load_pickle_as_numpy(p)
    return out


# --------------------------------------------------------------------------- #
# Rotation stats                                                              #
# --------------------------------------------------------------------------- #


def _build_rotation_stats(
    cos_sim: np.ndarray,
    pos_labels: List[str],
) -> dict:
    """
    Reduce the ``(n_layers, n_positions)`` cosine matrix into the JSON shape:

    .. code-block:: python

        {
          "<position_label>": {
            "rotation_r": {
              "late_layer_anti_alignment": float,
              "largest_drop_any_early_to_late": float,
              ...
            },
            "cos_sim_across_layers": [floats, length n_layers]
          },
          ...
          "best_token_pos_<metric>":   "<label>",
          "ranked_token_pos_<metric>": ["<label>", ...],   # descending by score
          ...   # one pair per registered metric
        }
    """
    n_layers, n_positions = cos_sim.shape
    if len(pos_labels) != n_positions:
        raise ValueError(
            f"position_labels length {len(pos_labels)} != cos_sim n_positions {n_positions}"
        )

    out: dict = {}
    for p, label in enumerate(pos_labels):
        layer_curve = [float(x) for x in cos_sim[:, p]]
        per_metric_score: Dict[str, float] = {}
        for metric_name, metric_fn in ROTATION_METRIC_REGISTRY.items():
            per_metric_score[metric_name] = float(
                metric_fn(layer_curve, token_position_index=p, metadata=None)
            )
        out[label] = {
            "rotation_r": per_metric_score,
            "cos_sim_across_layers": layer_curve,
        }

    for metric_name in ROTATION_METRIC_REGISTRY:
        ranked = sorted(
            pos_labels,
            key=lambda lbl, m=metric_name: out[lbl]["rotation_r"][m],
            reverse=True,
        )
        out[f"ranked_token_pos_{metric_name}"] = ranked
        out[f"best_token_pos_{metric_name}"] = ranked[0]
    return out


def _read_sampled_size_per_gt(
    cfg: ExperimentConfig, system_prompt_id: int, seed: int
) -> Optional[tuple[int, int]]:
    """
    Read ``(sampled_size_GT_0, sampled_size_GT_1)`` from one cached completions JSON.

    Tries the control file first, then experiment. Returns ``None`` if neither
    is readable or the fields are missing; the analysis stage uses that to
    leave the count out of plot titles.
    """
    for cond in ("control", "experiment"):
        p = completions_path(cfg, system_prompt_id, seed, cond)
        if not p.exists():
            continue
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        v0 = data.get("sampled_size_GT_0")
        v1 = data.get("sampled_size_GT_1")
        if v0 is not None and v1 is not None:
            return int(v0), int(v1)
    return None


# --------------------------------------------------------------------------- #
# Per-(SysPrompt, seed) work                                                  #
# --------------------------------------------------------------------------- #


def _analyze_one_seed(
    cfg: ExperimentConfig,
    system_prompt_id: int,
    seed: int,
    driving_metric_name: str,
) -> bool:
    """
    Compute and persist all analysis artifacts for one ``(SysPrompt, seed)``.

    PCA grids are written only for the top
    ``cfg.save_top_n_tokens_4_PCAplots`` token positions (ranked by
    ``rotation_r`` of ``driving_metric_name``); set the config field higher to
    emit more.

    Returns:
        True on success; False if pickles were missing (skipped).
    """
    # Lazy plotting imports: avoids needing plotly/sklearn at module-import time.
    from src.analysis.plots import (
        plot_class_separation_binary_curve,
        plot_class_separation_curve,
        plot_class_separation_per_class_curve,
        plot_cos_sim_heatmap,
        plot_pca_grid_for_position,
        resolve_pca_style,
    )

    activations = _load_four_activation_pickles(cfg, system_prompt_id, seed)
    if activations is None:
        return False

    cos_sim = compute_cos_sim_matrix(
        activations["control_GT_0"],
        activations["control_GT_1"],
        activations["experiment_GT_0"],
        activations["experiment_GT_1"],
    )
    pos_labels = position_labels(cfg.n_pre_prompt_tokens, cfg.n_generated_tokens)

    stats = _build_rotation_stats(cos_sim, pos_labels)
    _atomic_write_json(rotation_stats_path(cfg, system_prompt_id, seed), stats)

    # Per-row metric values for the heatmap's r-block. Column order matches
    # `displayed_metrics`, which is gated by cfg.heatmap_r_metric_display
    # ("all" → every registered metric; "chosen_metric" → only `driving_metric_name`).
    displayed_metrics = resolve_displayed_metrics(
        cfg.heatmap_r_metric_display, driving_metric_name
    )
    metric_values_per_row: List[List[float]] = [
        [stats[lbl]["rotation_r"][m] for m in displayed_metrics]
        for lbl in pos_labels
    ]
    metric_short_labels: List[str] = [metric_short_label(m) for m in displayed_metrics]

    stage_stats_dir(cfg, system_prompt_id, seed).mkdir(parents=True, exist_ok=True)
    plot_cos_sim_heatmap(
        cos_sim,
        position_labels=pos_labels,
        basepath=cos_sim_heatmap_basepath(cfg, system_prompt_id, seed),
        title=(
            f"{cfg.task}/{cfg.dataset} | SysPrompt{system_prompt_id:03d} | seed={seed}"
            f" | best[{driving_metric_name}]={stats[f'best_token_pos_{driving_metric_name}']}"
        ),
        metric_values_per_row=metric_values_per_row,
        metric_column_labels=metric_short_labels,
        formats=cfg.cos_sim_heatmap_formats,
    )

    style = resolve_pca_style(getattr(cfg, "pca_plot_style", {}) or {})

    # PCA position selection: an explicit list (config or CLI) wins; otherwise
    # fall back to the top-N positions by the driving metric's rotation_r.
    explicit: List[str] = list(getattr(cfg, "pca_token_positions", []) or [])
    if explicit:
        unknown = [p for p in explicit if p not in pos_labels]
        if unknown:
            print(
                f"  WARNING SysPrompt{system_prompt_id:03d} seed={seed}: "
                f"pca_token_positions has unknown labels (skipped): {unknown}"
            )
        top_positions: List[str] = [p for p in explicit if p in pos_labels]
    else:
        top_n = max(0, int(getattr(cfg, "save_top_n_tokens_4_PCAplots", 1)))
        ranked: List[str] = list(stats[f"ranked_token_pos_{driving_metric_name}"])
        top_positions = ranked[:top_n]

    sampled_k = _read_sampled_size_per_gt(cfg, system_prompt_id, seed)
    if sampled_k is None:
        k_str = "?"
    else:
        k0, k1 = sampled_k
        k_str = f"{k0}" if k0 == k1 else f"GT0={k0}, GT1={k1}"

    # Per-task GT label mapping (e.g. "False" / "True" for lying/sycophancy,
    # "Left" / "Right" for perspective). Falls back to "GT_0" / "GT_1" if the
    # task isn't registered in the config maps.
    gt0_name = cfg.gt_0_label_map.get(cfg.task, "GT_0")
    gt1_name = cfg.gt_1_label_map.get(cfg.task, "GT_1")
    legend_label_map = {
        "control_GT_0":         f"control / {gt0_name}",
        "control_GT_1":         f"control / {gt1_name}",
        "experiment_GT_0":      f"experiment / {gt0_name}",
        "experiment_GT_1":      f"experiment / {gt1_name}",
        "experiment_GT_0_fail": f"experiment-fail / {gt0_name}",
        "experiment_GT_1_fail": f"experiment-fail / {gt1_name}",
    }

    if bool(getattr(cfg, "save_PCA_plots", True)):
        for label in top_positions:
            position_index = pos_labels.index(label)
            title = (
                f"{cfg.task}/{cfg.dataset} | "
                f"SysPrompt{system_prompt_id:03d} | seed={seed} | "
                f"token={label} | "
                f"sampled_size_per_GT_class={k_str} | "
                f"method={cfg.cache_sample_method}"
            )
            plot_pca_grid_for_position(
                activations_by_group=activations,
                position_index=position_index,
                position_label=label,
                basepath=pca_plot_basepath(cfg, system_prompt_id, seed, label),
                style=style,
                title=title,
                legend_label_map=legend_label_map,
                formats=getattr(cfg, "PCA_plot_formats", None),
            )

    # --- 4-class classifier accuracy curve --------------------------------- #
    # Trains a multinomial logistic-regression classifier per layer in a
    # PCA-reduced subspace (n_PC fit per CV fold, no leakage). Position
    # selection: explicit cfg.class_separation_token_position wins; otherwise
    # the variant's best position under the driving rotation metric.
    if bool(getattr(cfg, "save_class_separation_curve", True)):
        explicit_cs = getattr(cfg, "class_separation_token_position", None)
        if explicit_cs and explicit_cs in pos_labels:
            cs_position_label = explicit_cs
        else:
            if explicit_cs:
                print(
                    f"  WARNING SysPrompt{system_prompt_id:03d} seed={seed}: "
                    f"class_separation_token_position={explicit_cs!r} "
                    f"not in pos_labels; falling back to best position "
                    f"under {driving_metric_name}."
                )
            cs_position_label = stats[f"best_token_pos_{driving_metric_name}"]
        cs_position_index = pos_labels.index(cs_position_label)

        try:
            accuracies = compute_cv_accuracy_per_layer(
                activations,
                position_index=cs_position_index,
                n_PC_classifier=int(cfg.n_PC_classifier),
                cv_n_splits=int(cfg.cv_n_splits),
                cv_shuffle=bool(cfg.cv_shuffle),
                cv_random_state=int(cfg.cv_random_state),
                classifier_C=float(cfg.classifier_C),
                classifier_max_iter=int(cfg.classifier_max_iter),
            )
        except ValueError as e:
            print(
                f"  WARNING SysPrompt{system_prompt_id:03d} seed={seed}: "
                f"4-class classifier skipped — {e}"
            )
            accuracies = None
        if accuracies is not None:
            cs_payload = {
                "token_position": cs_position_label,
                "n_PC_classifier": int(cfg.n_PC_classifier),
                "cv_n_splits": int(cfg.cv_n_splits),
                "cv_shuffle": bool(cfg.cv_shuffle),
                "cv_random_state": int(cfg.cv_random_state),
                "classifier_C": float(cfg.classifier_C),
                "classifier_max_iter": int(cfg.classifier_max_iter),
                "cv_accuracy_across_layers": [float(a) for a in accuracies],
            }
            _atomic_write_json(
                class_separation_stats_path(cfg, system_prompt_id, seed), cs_payload
            )
            plot_class_separation_curve(
                accuracies,
                basepath=classifier_acc_curves_basepath(cfg, system_prompt_id, seed),
                title=(
                    f"{cfg.task}/{cfg.dataset} | "
                    f"SysPrompt{system_prompt_id:03d} | seed={seed} | "
                    f"token={cs_position_label} | nPC={cfg.n_PC_classifier} | "
                    f"cv={cfg.cv_n_splits}-fold"
                ),
                formats=cfg.class_separation_plot_formats,
                line_color=cfg.class_separation_line_color,
                line_width=cfg.class_separation_line_width,
                y_range=tuple(cfg.class_separation_y_range),
            )

        # Per-class CV recall — same pipeline / CV splitter, exposed as one
        # curve per (condition, GT) class so we can read off which group the
        # classifier struggles with at any given layer.
        try:
            per_class_acc = compute_per_class_cv_accuracy_per_layer(
                activations,
                position_index=cs_position_index,
                n_PC_classifier=int(cfg.n_PC_classifier),
                cv_n_splits=int(cfg.cv_n_splits),
                cv_shuffle=bool(cfg.cv_shuffle),
                cv_random_state=int(cfg.cv_random_state),
                classifier_C=float(cfg.classifier_C),
                classifier_max_iter=int(cfg.classifier_max_iter),
            )
        except ValueError as e:
            print(
                f"  WARNING SysPrompt{system_prompt_id:03d} seed={seed}: "
                f"per-class classifier skipped — {e}"
            )
            per_class_acc = None
        if per_class_acc is not None:
            cs_per_class_payload = {
                "token_position": cs_position_label,
                "n_PC_classifier": int(cfg.n_PC_classifier),
                "cv_n_splits": int(cfg.cv_n_splits),
                "cv_shuffle": bool(cfg.cv_shuffle),
                "cv_random_state": int(cfg.cv_random_state),
                "classifier_C": float(cfg.classifier_C),
                "classifier_max_iter": int(cfg.classifier_max_iter),
                "cv_recall_per_class_across_layers": {
                    cls_name: [float(a) for a in arr]
                    for cls_name, arr in per_class_acc.items()
                },
            }
            _atomic_write_json(
                class_separation_per_class_stats_path(cfg, system_prompt_id, seed),
                cs_per_class_payload,
            )
            plot_class_separation_per_class_curve(
                per_class_acc,
                basepath=classifier_acc_per_class_curves_basepath(
                    cfg, system_prompt_id, seed
                ),
                title=(
                    f"{cfg.task}/{cfg.dataset} | "
                    f"SysPrompt{system_prompt_id:03d} | seed={seed} | "
                    f"token={cs_position_label} | nPC={cfg.n_PC_classifier} | "
                    f"cv={cfg.cv_n_splits}-fold | per-class recall"
                ),
                formats=cfg.class_separation_plot_formats,
                line_width=cfg.class_separation_line_width,
                y_range=tuple(cfg.class_separation_y_range),
            )

        # Per-condition binary classifiers — one classifier per condition
        # (control GT=0 vs control GT=1, experiment GT=0 vs experiment GT=1).
        # Chance is 0.5 inside each condition.
        try:
            binary_acc = compute_binary_per_condition_cv_accuracy_per_layer(
                activations,
                position_index=cs_position_index,
                n_PC_classifier=int(cfg.n_PC_classifier),
                cv_n_splits=int(cfg.cv_n_splits),
                cv_shuffle=bool(cfg.cv_shuffle),
                cv_random_state=int(cfg.cv_random_state),
                classifier_C=float(cfg.classifier_C),
                classifier_max_iter=int(cfg.classifier_max_iter),
            )
        except ValueError as e:
            print(
                f"  WARNING SysPrompt{system_prompt_id:03d} seed={seed}: "
                f"binary classifier skipped — {e}"
            )
            binary_acc = None
        if binary_acc:
            cs_binary_payload = {
                "token_position": cs_position_label,
                "n_PC_classifier": int(cfg.n_PC_classifier),
                "cv_n_splits": int(cfg.cv_n_splits),
                "cv_shuffle": bool(cfg.cv_shuffle),
                "cv_random_state": int(cfg.cv_random_state),
                "classifier_C": float(cfg.classifier_C),
                "classifier_max_iter": int(cfg.classifier_max_iter),
                "cv_accuracy_per_condition_across_layers": {
                    cond: [float(a) for a in arr]
                    for cond, arr in binary_acc.items()
                },
            }
            _atomic_write_json(
                class_separation_binary_stats_path(cfg, system_prompt_id, seed),
                cs_binary_payload,
            )
            plot_class_separation_binary_curve(
                binary_acc,
                basepath=classifier_acc_binary_curves_basepath(
                    cfg, system_prompt_id, seed
                ),
                title=(
                    f"{cfg.task}/{cfg.dataset} | "
                    f"SysPrompt{system_prompt_id:03d} | seed={seed} | "
                    f"token={cs_position_label} | nPC={cfg.n_PC_classifier} | "
                    f"cv={cfg.cv_n_splits}-fold | per-condition binary"
                ),
                formats=cfg.class_separation_plot_formats,
                line_width=cfg.class_separation_line_width,
                y_range=tuple(cfg.class_separation_y_range),
            )

    return True


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #


def run_analysis_for_config(
    cfg: ExperimentConfig,
    rotation_metric_name: str | None = None,
) -> None:
    """
    Run the analysis stage end-to-end against one config.

    Iterates over every ``system_prompt_id`` in
    ``cfg.variant_indices(len(variants))`` (i.e. honoring
    ``cfg.start_variant`` / ``cfg.end_variant``) and every seed in
    ``cfg.seed_list_cache``. Skips pairs whose activation pickles are not
    all present unless ``cfg.rerun_cache_in_analysis`` is ``True``, in
    which case the cache stage is invoked first to regenerate pickles for
    the same variant range before the analysis loop runs.

    Args:
        cfg: Experiment configuration.
        rotation_metric_name: Optional override for ``cfg.rotation_metric``.
            Must be a key in
            :data:`src.analysis.rotation_plugins.ROTATION_METRIC_REGISTRY`.
            ``None`` → use ``cfg.rotation_metric``.

    Raises:
        ValueError: If ``cfg.seed_list_cache`` is empty or the requested
            metric name is not registered.
    """
    chosen_name, _ = resolve_rotation_metric(
        rotation_metric_name or cfg.rotation_metric
    )
    seeds = list(cfg.seed_list_cache or [])
    if not seeds:
        raise ValueError("seed_list_cache is empty; nothing to analyze.")

    variants = load_experiment_system_prompts(cfg)
    variant_range = cfg.variant_indices(len(variants))
    variant_list = list(variant_range)
    if not variant_list:
        print(
            f"Analysis stage: empty variant range "
            f"(start_variant={cfg.start_variant}, end_variant={cfg.end_variant}, "
            f"variant_subset={cfg.variant_subset}, "
            f"n_variants={len(variants)}); nothing to do."
        )
        return
    print(
        f"Analysis stage: processing system prompts "
        f"[{variant_list[0]}, {variant_list[-1]}] "
        f"({len(variant_list)} variants) × seeds={seeds}"
    )

    # Optional cache rerun: same variant_indices range and config drive the
    # cache stage, so the pickles regenerated below match the analysis loop
    # below 1:1. Lazy import keeps the analysis stage free of torch when this
    # flag is False (the default).
    if bool(getattr(cfg, "rerun_cache_in_analysis", False)):
        from src.cache_activations.cache_activations import (
            run_cache_activations_for_config,
        )
        print("Analysis stage: rerun_cache_in_analysis=True — running cache stage first.")
        run_cache_activations_for_config(cfg)

    n_done = 0
    n_skipped = 0
    for system_prompt_id in variant_list:
        for seed in seeds:
            ok = _analyze_one_seed(cfg, system_prompt_id, seed, chosen_name)
            if ok:
                n_done += 1
                print(f"  analyzed SysPrompt{system_prompt_id:03d} seed={seed}")
            else:
                n_skipped += 1
                print(
                    f"  skip SysPrompt{system_prompt_id:03d} seed={seed}: "
                    "missing activation pickles (run cache stage first)"
                )
    print(f"Analysis stage complete: {n_done} analyzed, {n_skipped} skipped.")


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """CLI for the analysis stage."""
    p = argparse.ArgumentParser(
        description="Run activation analysis (cosine + PCA + rotation metrics)."
    )
    add_common_args(p)
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    """Entry point: load config and run :func:`run_analysis_for_config`."""
    args = parse_args(argv)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    cfg = load_config_from_cli(args)
    dump_run_record(
        cfg,
        out_dir=behavior_run_dir(cfg),
        argv=sys.argv,
        stage="analysis",
    )
    run_analysis_for_config(cfg)


if __name__ == "__main__":
    main()
