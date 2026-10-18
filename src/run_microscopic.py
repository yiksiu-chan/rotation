"""
Microscopic projection + trajectory plots — CLI entry point.

Per-datapoint projection-line plots (existing behavior, now driven by
``--targets``) and 2D layer-trajectory plots (new) are both written for
each ``--targets_data_folder`` against a single
``--task_vector_data_folder``.

The truth direction is built from ``--task_vector_data_folder`` and sliced
at that folder's *top-1* token position (from
``stage_stats/rotation_stats.json``). Activations from each
``--targets_data_folder`` are sliced at that *targets* folder's own
top-N (and/or explicit) token positions — so the truth direction's position
and the activation's position can differ.

Usage
~~~~~

.. code-block:: bash

    python -m src.run_microscopic \\
    --task_vector_data_folder \\
        results/runs/Llama-3.1-8B-Instruct/lying_azaria-mitchell/SysPrompt004/0_pre10_gen5 \\
    --targets_data_folder \\
        results/runs/Llama-3.1-8B-Instruct/lying_azaria-mitchell/SysPrompt004/0_pre10_gen5 \\
        results/runs/Llama-3.1-8B-Instruct/lying_to_achieve_goals_strong_gpt4o_deepseek_filtered/SysPrompt000/0_pre10_gen5 \\
    --token_positions_targets prompt-1 prompt \\
    --best_n_token_position_targets 1 \\
    --targets control_GT_0:0,5 experiment_GT_1:0,2 \\
    --trajectory_residualPC_variants raw unit_x unit_xy \\
    --trajectory_realignment_variants raw_projection cosine_similarity \\
    --trajectory_background none control success_only all \\
    --trajectory_set_layer 8 18 \\
    --trajectory_realignment_footnote \\
    --bg_grey_scale \\
    --trajectory_task_vector_arrow \\
    --gt_style line_style \\
    --output_dir "results/plots/<model_alias>/microscopic" \\
    --formats png \\
    --manifold_PCA_set_layer 8 18 \\
   --token_positions_targets prompt gen+1 gen+2 \\

Both ``--token_positions_targets`` and ``--best_n_token_position_targets``
may be supplied — the union (deduplicated, preserving listed-first /
best-first order) drives the figure rows for that targets folder. When
neither is supplied the script defaults to
``best_n_token_position_targets=1`` so a row always renders.

The task-vector token position defaults to the top-1 from the task-vector
folder's ``ranked_token_pos_<rotation_metric>`` ranking (sentinel value
``["-1"]``). Pass one or more explicit labels via
``--token_positions_task_vector`` (e.g. ``--token_positions_task_vector
prompt-1 prompt gen+2``) to override; one figure set is written per
position. Falls back to ``"prompt-1"`` with a warning if the rotation
stats are missing.

Targets
~~~~~~~

``--targets`` accepts ``group:idx[,idx...]`` tokens. Group names can be
either the user-facing form (``control_GT_0``, ``experiment_GT_1``,
``experiment_GT_0_fail``) or the internal form (``honest_success_GT_0``,
``lie_success_GT_1``, ``lie_fail_GT_0``). The same target list drives
**both** the projection-line plots (per-target single-sample columns) and
the trajectory plots (highlighted layer-trajectories). When the list is
empty, the projection plots fall back to sample 0 of every loaded group
(backward-compatible default) and the trajectory plots are skipped.

Outputs
~~~~~~~

For each targets folder one projection-line figure, zero-or-more trajectory
figures (one per ``(--trajectory_background, --trajectory_set_layer)`` pair),
and zero-or-more manifold figures (across-layer + set-layer subspaces, one
file per ``(method, n_components, group_mode, position[, set_layer])``
combination) are written:

::

    {output_dir}/projection/{task_vector_alias}/{axis_name}/microscopic_tv-{tv_pos}_{targets_alias}_L{lo}-L{hi}.{fmt}
    {output_dir}/trajectory/{task_vector_alias}/trajectory_bg-{bg}_tv-{tv_pos}_setL{L}_targetsL{lo}-L{hi}_{targets_alias}.{fmt}
    {output_dir}/manifolds/{task_vector_alias}/across/{n}D/{method}/manifold_{targets_alias}_pos-{pos}_L{lo}-L{hi}_{fail}.{fmt}
    {output_dir}/manifolds/{task_vector_alias}/set_layer/{n}D/PCA/manifold_{targets_alias}_pos-{pos}_setL{set}_targetsL{tlo}-L{thi}_{fail}.{fmt}

where ``alias`` is ``{task_dataset}_SysPrompt{NNN}_seed{seed}`` derived from
the run-folder path segments (the same names that
:mod:`src.cache_activations.paths` writes), and ``tv_pos`` is the
task-vector token-position label (e.g. ``prompt-1``).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

from src.analysis.microscopic import (
    AXIS_REGISTRY,
    INTERNAL_TO_USER_GROUP,
    compute_truth_directions,
    default_targets_for_groups,
    filter_valid_targets,
    fit_axis_directions,
    load_microscopic_groups,
    parse_target_spec,
    plot_microscopic_grid,
    plot_trajectory_axis_grid,
    plot_trajectory_axis_polar_grid,
    plot_trajectory_grid,
    project_all_groups,
)
from src.cache_activations.hooks import position_labels
from src.config.loader import default_config_path, load_yaml_config
from src.config.schema import ExperimentConfig
from src.paths import PROJECT_ROOT


# --------------------------------------------------------------------------- #
# Path parsing                                                                #
# --------------------------------------------------------------------------- #


_SEED_RE = re.compile(r"^(\d+)_pre(\d+)_gen(\d+)$")
_SYS_PROMPT_RE = re.compile(r"^SysPrompt(\d+)$")


def _parse_run_folder(folder: Path) -> Dict[str, object]:
    """
    Parse a cache-stage run folder into its identifying components.

    Expected layout (matches
    :func:`src.cache_activations.paths.cache_seed_dir`):

    ::

        <output_root>/<model_alias>/<task_dataset>/SysPrompt<NNN>/<seed>_pre<n>_gen<m>

    Returns a dict with ``model_alias``, ``task_dataset``,
    ``sys_prompt_id`` (int), ``sys_prompt_segment`` (str, 3-digit padded),
    ``seed`` (int), ``n_pre`` (int), ``n_gen`` (int), ``seed_segment`` (str)
    and ``alias`` — a single ``{task_dataset}_SysPrompt{NNN}_seed{seed}``
    handle suitable for filenames / output-folder naming.

    Raises:
        ValueError: If the path doesn't match the expected layout. The
            caller is expected to have pointed at the seed-level dir
            (the one containing ``activations/``).
    """
    folder = folder.resolve()
    parts = folder.parts
    if len(parts) < 4:
        raise ValueError(
            f"Run folder {folder} has too few path segments; expected "
            "<...>/<model_alias>/<task_dataset>/SysPrompt<NNN>/<seed>_pre<n>_gen<m>"
        )
    seed_segment = parts[-1]
    sys_prompt_segment = parts[-2]
    task_dataset = parts[-3]
    model_alias = parts[-4]

    seed_match = _SEED_RE.match(seed_segment)
    if not seed_match:
        raise ValueError(
            f"Cannot parse seed segment {seed_segment!r} of {folder}; "
            f"expected '<seed>_pre<n>_gen<m>'."
        )
    sys_match = _SYS_PROMPT_RE.match(sys_prompt_segment)
    if not sys_match:
        raise ValueError(
            f"Cannot parse SysPrompt segment {sys_prompt_segment!r} of {folder}; "
            f"expected 'SysPrompt<NNN>'."
        )

    seed = int(seed_match.group(1))
    n_pre = int(seed_match.group(2))
    n_gen = int(seed_match.group(3))
    sys_prompt_id = int(sys_match.group(1))
    return {
        "model_alias": model_alias,
        "task_dataset": task_dataset,
        "sys_prompt_id": sys_prompt_id,
        "sys_prompt_segment": sys_prompt_segment,
        "seed": seed,
        "n_pre": n_pre,
        "n_gen": n_gen,
        "seed_segment": seed_segment,
        "alias": f"{task_dataset}_{sys_prompt_segment}_seed{seed}",
        "folder": folder,
    }


# --------------------------------------------------------------------------- #
# Token position resolution                                                   #
# --------------------------------------------------------------------------- #


def _load_ranked_token_positions(
    folder: Path,
    rotation_metric: str,
) -> List[str]:
    """
    Read ``ranked_token_pos_<rotation_metric>`` from
    ``folder/stage_stats/rotation_stats.json``.

    Returns the empty list when the file is missing, unreadable, or the
    ranked key is absent — callers are expected to print their own context-
    specific warning.
    """
    rotation_stats_path = folder / "stage_stats" / "rotation_stats.json"
    if not rotation_stats_path.exists():
        return []
    try:
        with open(rotation_stats_path, encoding="utf-8") as f:
            stats = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"  WARN cannot read {rotation_stats_path}: {e}.")
        return []
    return list(stats.get(f"ranked_token_pos_{rotation_metric}", []))


def _resolve_tv_top_position(
    folder: Path,
    pos_labels: Sequence[str],
    rotation_metric: str,
) -> str:
    """
    Resolve the *task-vector* token position: top-1 of the task-vector
    folder's ``ranked_token_pos_<rotation_metric>`` ranking.

    Falls back to ``"prompt-1"`` (or the first available label) with a
    warning when the ranking is missing or its top entry isn't in the run's
    label list.
    """
    ranked = _load_ranked_token_positions(folder, rotation_metric)
    if not ranked:
        fallback = "prompt-1" if "prompt-1" in pos_labels else pos_labels[0]
        print(
            f"  WARN no ranked_token_pos_{rotation_metric} under "
            f"{folder} — falling back to task-vector position {fallback!r}."
        )
        return fallback
    top = ranked[0]
    if top not in pos_labels:
        fallback = "prompt-1" if "prompt-1" in pos_labels else pos_labels[0]
        print(
            f"  WARN top-ranked task-vector label {top!r} not in available "
            f"labels {list(pos_labels)} — falling back to {fallback!r}."
        )
        return fallback
    return top


def _resolve_token_positions(
    folder: Path,
    pos_labels: Sequence[str],
    explicit: Sequence[str],
    best_n: int,
    rotation_metric: str,
) -> List[str]:
    """
    Build the ordered list of *target* token-position labels for one
    targets folder.

    Combines ``explicit`` (CLI ``--token_positions_targets``) and the top
    ``best_n`` entries of the ``ranked_token_pos_<rotation_metric>`` array
    from the *targets* folder's ``stage_stats/rotation_stats.json``.
    Explicit labels come first, then the ranked top-N — duplicates removed
    while preserving order.

    Falls back to ``["prompt-1"]`` when both sources are empty (so a row
    always renders).
    """
    seen: List[str] = []
    for lbl in explicit:
        if lbl in seen:
            continue
        if lbl not in pos_labels:
            print(
                f"  WARN token position {lbl!r} not in available labels "
                f"{list(pos_labels)} — skipping."
            )
            continue
        seen.append(lbl)

    if best_n > 0:
        rotation_stats_path = folder / "stage_stats" / "rotation_stats.json"
        if rotation_stats_path.exists():
            ranked = _load_ranked_token_positions(folder, rotation_metric)
            if not ranked:
                print(
                    f"  WARN ranked_token_pos_{rotation_metric} missing or "
                    f"empty in {rotation_stats_path} — "
                    f"--best_n_token_position_targets will be ignored."
                )
            for lbl in ranked[: int(best_n)]:
                if lbl in seen:
                    continue
                if lbl not in pos_labels:
                    print(
                        f"  WARN ranked label {lbl!r} not in available "
                        f"labels — skipping."
                    )
                    continue
                seen.append(lbl)
        else:
            print(
                f"  WARN no rotation_stats.json under {folder} — "
                f"--best_n_token_position_targets={best_n} will be "
                f"ignored. Run the analysis stage first or pass "
                f"--token_positions_targets explicitly."
            )

    if not seen:
        fallback = "prompt-1" if "prompt-1" in pos_labels else pos_labels[0]
        print(
            f"  no token positions resolved; falling back to {fallback!r}."
        )
        seen = [fallback]
    return seen


# --------------------------------------------------------------------------- #
# Rotation cos_sim sibling aggregation                                        #
# --------------------------------------------------------------------------- #


def _discover_sibling_sysprompt_rotation_stats(
    target_folder: Path,
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Discover sibling SysPrompt*/<seed_segment>/stage_stats/rotation_stats.json
    folders next to ``target_folder`` and load each rotation_stats dict.

    Target folder layout (from
    :func:`src.cache_activations.paths.cache_seed_dir`):

    ::

        <root>/<model>/<task>/SysPrompt<NNN>/<seed_segment>/

    Returns an ordered list of ``(sys_prompt_segment, rotation_stats_dict)``
    sorted by ``sys_prompt_segment``. Missing JSONs or unreadable files are
    skipped with a warning. Includes the target folder's own SysPrompt when
    present.
    """
    target_folder = target_folder.resolve()
    parts = target_folder.parts
    if len(parts) < 4:
        print(
            f"  rotation cos_sim: cannot resolve siblings of {target_folder} "
            f"(unexpected path layout); skipping aggregate."
        )
        return []
    seed_segment = parts[-1]
    task_parent = Path(*parts[:-2])  # strip SysPrompt###/<seed_segment>

    out: List[Tuple[str, Dict[str, Any]]] = []
    for sys_dir in sorted(task_parent.glob("SysPrompt*")):
        if not sys_dir.is_dir():
            continue
        rotation_path = sys_dir / seed_segment / "stage_stats" / "rotation_stats.json"
        if not rotation_path.exists():
            print(
                f"  rotation cos_sim: WARN no rotation_stats.json under "
                f"{sys_dir.name}/{seed_segment} — dropping from aggregate."
            )
            continue
        try:
            with open(rotation_path, encoding="utf-8") as f:
                stats = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(
                f"  rotation cos_sim: WARN cannot read {rotation_path}: {e} — "
                f"dropping from aggregate."
            )
            continue
        out.append((sys_dir.name, stats))
    return out


def _build_rotation_cos_sim_data(
    target_folder: Path,
    positions: Sequence[str],
) -> Dict[str, Dict[str, Optional[np.ndarray]]]:
    """
    Build the ``rotation_cos_sim_data`` payload for
    :func:`plot_microscopic_grid`.

    For each position in ``positions``:

    - ``"individual"``: ``cos_sim_across_layers`` pulled from
      ``target_folder``'s own ``rotation_stats.json`` (None when missing).
    - ``"aggregate"``: ``(n_sysprompts, n_layers)`` stack across sibling
      SysPrompt folders that share ``target_folder``'s seed_segment (None
      when nothing usable was found).

    Sibling curves with mismatched lengths are dropped with a warning.
    """
    siblings = _discover_sibling_sysprompt_rotation_stats(target_folder)
    target_stats_path = target_folder / "stage_stats" / "rotation_stats.json"
    target_stats: Dict[str, Any] = {}
    if target_stats_path.exists():
        try:
            with open(target_stats_path, encoding="utf-8") as f:
                target_stats = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(
                f"  rotation cos_sim: WARN cannot read {target_stats_path}: "
                f"{e} — individual curve will be None."
            )
    else:
        print(
            f"  rotation cos_sim: WARN no rotation_stats.json under "
            f"{target_folder} — individual curve will be None."
        )

    out: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
    for pos_label in positions:
        individual: Optional[np.ndarray] = None
        pos_block = target_stats.get(pos_label)
        if isinstance(pos_block, dict):
            curve = pos_block.get("cos_sim_across_layers")
            if curve is not None:
                individual = np.asarray(curve, dtype=np.float64)

        agg_rows: List[np.ndarray] = []
        ref_len: Optional[int] = (
            individual.shape[0] if individual is not None else None
        )
        for sys_name, stats in siblings:
            block = stats.get(pos_label)
            if not isinstance(block, dict):
                print(
                    f"  rotation cos_sim: WARN position {pos_label!r} missing "
                    f"under {sys_name} — dropping from aggregate."
                )
                continue
            curve = block.get("cos_sim_across_layers")
            if curve is None:
                print(
                    f"  rotation cos_sim: WARN cos_sim_across_layers missing "
                    f"for {sys_name}/{pos_label} — dropping from aggregate."
                )
                continue
            arr = np.asarray(curve, dtype=np.float64)
            if ref_len is None:
                ref_len = int(arr.shape[0])
            if arr.shape[0] != ref_len:
                print(
                    f"  rotation cos_sim: WARN {sys_name}/{pos_label} length "
                    f"{arr.shape[0]} != {ref_len} — dropping from aggregate."
                )
                continue
            agg_rows.append(arr)

        aggregate: Optional[np.ndarray] = (
            np.stack(agg_rows, axis=0) if agg_rows else None
        )
        out[pos_label] = {"individual": individual, "aggregate": aggregate}
    return out


# --------------------------------------------------------------------------- #
# Per-folder run                                                              #
# --------------------------------------------------------------------------- #


def _resolve_gt_label_names(cfg: Optional[ExperimentConfig], task_dataset: str) -> Tuple[str, str]:
    """
    Best-effort GT label names (e.g. "False" / "True" for lying tasks).

    With a config: use ``cfg.gt_0_label_map`` / ``cfg.gt_1_label_map`` keyed
    by the longest task prefix that matches the run folder's
    ``task_dataset`` name (the folder concatenates ``cfg.task`` and
    ``cfg.dataset``, so an exact key lookup won't work for cross-config
    plots).

    Without a config: fall back to ``"GT_0"`` / ``"GT_1"`` so the legend
    still renders cleanly.
    """
    if cfg is None:
        return "GT_0", "GT_1"
    gt0 = cfg.gt_0_label_map.get(cfg.task) or "GT_0"
    gt1 = cfg.gt_1_label_map.get(cfg.task) or "GT_1"
    if cfg.task and task_dataset.startswith(cfg.task):
        return gt0, gt1
    # Cross-task projection — try every registered key and pick the longest
    # prefix match so e.g. ``lying_to_achieve_goals_*`` runs still grab the
    # lying-task labels.
    candidates = sorted(cfg.gt_0_label_map.keys(), key=len, reverse=True)
    for key in candidates:
        if task_dataset.startswith(key):
            return cfg.gt_0_label_map[key], cfg.gt_1_label_map.get(key, "GT_1")
    return "GT_0", "GT_1"


def _manifold_basepath_across(
    *,
    output_dir: Path,
    task_vector_alias: str,
    target_alias: str,
    n_components: int,
    method: str,
    position_label: str,
    layer_lo: int,
    layer_hi: int,
    with_fail: bool,
    gt_filename_suffix: str = "",
) -> Path:
    """``{output_dir}/manifolds/{tv_alias}/across/{n}D/{method}/manifold_{targets_alias}_pos-{pos}_L{lo}-L{hi}_{fail}{gt}``."""
    safe = position_label.replace("/", "_").replace(" ", "_")
    fail_suffix = "with_fail" if with_fail else "without_fail"
    return (
        output_dir
        / "manifolds"
        / task_vector_alias
        / "across"
        / f"{int(n_components)}D"
        / method
        / (
            f"manifold_{target_alias}_pos-{safe}"
            f"_L{int(layer_lo)}-L{int(layer_hi)}_{fail_suffix}{gt_filename_suffix}"
        )
    )


def _manifold_basepath_set_layer(
    *,
    output_dir: Path,
    task_vector_alias: str,
    target_alias: str,
    n_components: int,
    position_label: str,
    set_layer_idx: int,
    target_layer_lo: int,
    target_layer_hi: int,
    with_fail: bool,
    gt_filename_suffix: str = "",
) -> Path:
    """``{output_dir}/manifolds/{tv_alias}/set_layer/{n}D/PCA/manifold_{targets_alias}_pos-{pos}_setL{set}_targetsL{tlo}-L{thi}_{fail}{gt}``."""
    safe = position_label.replace("/", "_").replace(" ", "_")
    fail_suffix = "with_fail" if with_fail else "without_fail"
    return (
        output_dir
        / "manifolds"
        / task_vector_alias
        / "set_layer"
        / f"{int(n_components)}D"
        / "PCA"
        / (
            f"manifold_{target_alias}_pos-{safe}"
            f"_setL{int(set_layer_idx)}"
            f"_targetsL{int(target_layer_lo)}-L{int(target_layer_hi)}"
            f"_{fail_suffix}{gt_filename_suffix}"
        )
    )


def _trajectory_axis_basepath(
    *,
    output_dir: Path,
    task_vector_alias: str,
    axis_names: Sequence[str],
    target_alias: str,
    tv_position_label: str,
    target_position_label: str,
    layer_lo: int,
    layer_hi: int,
    gt_filename_suffix: str = "",
) -> Path:
    """
    ``{output_dir}/trajectory_axis/{tv_alias}/axes_<a>__<b>[__<c>]/
        trajectory_axis_tv-{tv_pos}_pos-{target_pos}_L{lo}-L{hi}_{target_alias}{gt}``.
    """
    safe_tv = tv_position_label.replace("/", "_").replace(" ", "_")
    safe_target = target_position_label.replace("/", "_").replace(" ", "_")
    slug = "axes_" + "__".join(axis_names)
    return (
        output_dir
        / "trajectory_axis"
        / task_vector_alias
        / slug
        / (
            f"trajectory_axis_tv-{safe_tv}_pos-{safe_target}"
            f"_L{int(layer_lo)}-L{int(layer_hi)}_{target_alias}{gt_filename_suffix}"
        )
    )


def _trajectory_axis_polar_basepath(
    *,
    output_dir: Path,
    task_vector_alias: str,
    axis_names: Sequence[str],
    target_alias: str,
    tv_position_label: str,
    set_layer: int,
    background: str,
    layer_lo: int,
    layer_hi: int,
    gt_filename_suffix: str = "",
) -> Path:
    """
    ``{output_dir}/trajectory_axis_polar/{tv_alias}/axes_<a>__<b>[__<c>]/
        trajectory_axis_polar_bg-{bg}_tv-{tv_pos}_setL{set_L}_targetsL{lo}-L{hi}_{target_alias}{gt}``.
    """
    safe_tv = tv_position_label.replace("/", "_").replace(" ", "_")
    slug = "axes_" + "__".join(axis_names)
    return (
        output_dir
        / "trajectory_axis_polar"
        / task_vector_alias
        / slug
        / (
            f"trajectory_axis_polar_bg-{background}_tv-{safe_tv}"
            f"_setL{int(set_layer)}"
            f"_targetsL{int(layer_lo)}-L{int(layer_hi)}_{target_alias}{gt_filename_suffix}"
        )
    )


def _parse_trajectory_axis_axis_sets(
    raw_entries: Sequence[str],
) -> List[List[str]]:
    """
    Parse ``--trajectory_axis_axes`` entries into a list of axis-name lists.

    Each entry is a comma-joined axis name list (1–3 names). Whitespace around
    names is stripped. Empty entries are silently skipped. Each entry yields
    one axis-set / one figure.

    Raises:
        ValueError: When any entry is empty after splitting, holds >3 axes,
            or references a name not in :data:`AXIS_REGISTRY`.
    """
    out: List[List[str]] = []
    for entry in raw_entries:
        names = [n.strip() for n in str(entry).split(",") if n.strip()]
        if not names:
            continue
        if not (1 <= len(names) <= 3):
            raise ValueError(
                f"--trajectory_axis_axes entry {entry!r} must list 1–3 axis "
                f"names; got {len(names)}: {names}"
            )
        unknown = [n for n in names if n not in AXIS_REGISTRY]
        if unknown:
            raise ValueError(
                f"--trajectory_axis_axes entry {entry!r} has unknown axis "
                f"name(s) {unknown}; valid: {sorted(AXIS_REGISTRY)}"
            )
        out.append(names)
    return out


def _optional_int(raw: Any) -> Optional[int]:
    """YAML → ``Optional[int]``. Treats ``None`` / ``""`` / ``"null"`` as
    "no override" (returns ``None``); anything else is coerced via
    ``int(...)`` and a ``ValueError`` propagates as a config error."""
    if raw is None or raw == "" or raw == "null":
        return None
    return int(raw)


def _resolve_optional_float(cli_value: Optional[float], yaml_raw: Any) -> Optional[float]:
    """CLI value (if not ``None``) wins, otherwise fall back to the YAML
    raw value with the same null sentinels accepted by :func:`_optional_int`.
    ``None`` propagates through to "no override" downstream."""
    if cli_value is not None:
        return float(cli_value)
    if yaml_raw is None or yaml_raw == "" or yaml_raw == "null":
        return None
    return float(yaml_raw)


def _parse_axis_ranges(
    raw: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, Tuple[float, float]]]:
    """
    Validate ``trajectory_axis.axis_ranges`` from yaml into
    ``{axis_name: (min, max)}``. ``None`` / empty dict returns ``None``.

    Each value must be a 2-element sequence of distinct floats; axis names
    must appear in :data:`AXIS_REGISTRY`. ``null``-valued entries are
    skipped (treat them as 'no override for this axis'). Passing the
    range in reversed order (``[MAX, MIN]``) flips that axis's direction
    in the plot — both matplotlib (``set_xlim``/``set_ylim``/``set_zlim``)
    and plotly (axis ``range``) invert when given a descending pair.
    """
    if not raw:
        return None
    out: Dict[str, Tuple[float, float]] = {}
    for name, rng in raw.items():
        if rng is None:
            continue
        if name not in AXIS_REGISTRY:
            raise ValueError(
                f"trajectory_axis.axis_ranges has unknown axis name "
                f"{name!r}; valid: {sorted(AXIS_REGISTRY)}"
            )
        try:
            lo, hi = float(rng[0]), float(rng[1])
        except (TypeError, ValueError, IndexError) as e:
            raise ValueError(
                f"trajectory_axis.axis_ranges[{name!r}] must be "
                f"[min, max]; got {rng!r}"
            ) from e
        if lo == hi:
            raise ValueError(
                f"trajectory_axis.axis_ranges[{name!r}] requires distinct "
                f"endpoints; got [{lo}, {hi}]"
            )
        out[name] = (lo, hi)
    return out or None


_VALID_TARGET_GT: Tuple[str, ...] = ("GT_0", "GT_1")
_VALID_TARGET_GROUP: Tuple[str, ...] = ("control", "experiment", "experiment_fail")


def _build_targets_specs_from_triple(
    gt_list: Sequence[str],
    group_list: Sequence[str],
    idx_list: Sequence[int],
) -> List[str]:
    """Build the cross product ``GT × group × idx`` as
    ``parse_target_spec``-compatible ``"<group>_<GT>[_fail]:<idx>"`` tokens.

    The ``experiment_fail`` group expands to ``experiment_<GT>_fail`` to
    match the user-facing key names in
    :data:`src.analysis.microscopic.USER_TO_INTERNAL_GROUP`.
    """
    unknown_gt = [g for g in gt_list if g not in _VALID_TARGET_GT]
    if unknown_gt:
        raise ValueError(
            f"targets.GT has unknown entries {unknown_gt}; "
            f"valid: {list(_VALID_TARGET_GT)}"
        )
    unknown_group = [g for g in group_list if g not in _VALID_TARGET_GROUP]
    if unknown_group:
        raise ValueError(
            f"targets.group has unknown entries {unknown_group}; "
            f"valid: {list(_VALID_TARGET_GROUP)}"
        )
    out: List[str] = []
    for group in group_list:
        for gt in gt_list:
            key = f"experiment_{gt}_fail" if group == "experiment_fail" else f"{group}_{gt}"
            for idx in idx_list:
                out.append(f"{key}:{int(idx)}")
    return out


def _gt_filename_suffix(gt_list: Sequence[str]) -> str:
    """``[GT_0]`` → ``"_GT0"``; ``[GT_1]`` → ``"_GT1"``; both → ``"_GT0-1"``."""
    nums = sorted({int(str(g).split("_")[-1]) for g in gt_list})
    if not nums:
        return ""
    return "_GT" + "-".join(str(n) for n in nums)


def _to_user_facing_groups(
    groups: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """Translate ``honest_success_GT_0`` → ``control_GT_0`` etc."""
    return {INTERNAL_TO_USER_GROUP.get(k, k): v for k, v in groups.items()}


def _to_user_facing_targets(
    targets: Sequence[Tuple[str, int]]
) -> List[Tuple[str, int]]:
    """Translate target group keys from internal to user-facing form."""
    return [(INTERNAL_TO_USER_GROUP.get(g, g), idx) for g, idx in targets]


def _run_manifold_plots_for_targets_folder(
    *,
    activations_user: Dict[str, np.ndarray],
    targets_user: Sequence[Tuple[str, int]],
    pos_labels: Sequence[str],
    chosen_positions: Sequence[str],
    output_dir: Path,
    task_vector_alias: str,
    target_alias: str,
    target_meta: Dict[str, object],
    legend_label_map: Dict[str, str],
    style: Dict[str, object],
    manifold_subspaces: Sequence[str],
    manifold_methods: Sequence[str],
    manifold_groups: Sequence[str],
    manifold_n_components: Sequence[int],
    n_sample_manifold: int,
    manifold_across_start_layer: int,
    manifold_across_end_layer: int,
    manifold_PCA_set_layers: Sequence[int],
    manifold_set_start_layer: int,
    manifold_set_end_layer: int,
    gt_style: str,
    formats: Sequence[str],
    bg_grey_scale: bool = False,
    gt_filename_suffix: str = "",
) -> None:
    """
    Render across-layer and set-layer manifold plots for one targets folder.

    Activations are already in user-facing-key form (``control_GT_0`` etc.) so
    they line up with ``style`` and the plot helpers. Per ``manifold_groups``
    we either include or drop the ``experiment_GT_*_fail`` keys from the
    scatter; ``targets`` always resolves against the full dict.

    The ``--targets`` list (in user-facing form) drives the highlighted
    layer-traversal lines / markers in both subspaces; ``gt_style`` controls
    their dash style (and, in ``"marker_shape"`` mode, the per-target marker
    symbol in the set-layer subspace).
    """
    from src.analysis.plots import (
        plot_manifold_for_position,
        plot_manifold_set_layer_for_position,
    )

    # ---- subsample & build with/without-fail variants ---------------------
    n_sub = int(n_sample_manifold or 0)

    def _apply_n_sub(d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {g: a[:n_sub] for g, a in d.items()} if n_sub > 0 else d

    activations_with_fail = _apply_n_sub(dict(activations_user))
    activations_without_fail = _apply_n_sub(
        {k: v for k, v in activations_user.items() if not k.endswith("_fail")}
    )

    any_acts = next(iter(activations_with_fail.values()))
    n_layers_full = any_acts.shape[2]

    def _resolve_range(start_v: int, end_v: int) -> Tuple[int, int]:
        lo = 0 if int(start_v) == -1 else int(start_v)
        hi = n_layers_full - 1 if int(end_v) == -1 else int(end_v)
        return lo, hi

    across_lo, across_hi = _resolve_range(
        manifold_across_start_layer, manifold_across_end_layer
    )
    target_lo, target_hi = _resolve_range(
        manifold_set_start_layer, manifold_set_end_layer
    )

    set_layers = [int(L) for L in manifold_PCA_set_layers or ()]
    task_dataset = str(target_meta.get("task_dataset", "?"))
    fmts_call = tuple(formats)

    for position_label in chosen_positions:
        if position_label not in pos_labels:
            print(
                f"  manifold WARN position {position_label!r} not in run's "
                f"labels — skipping."
            )
            continue
        position_index = pos_labels.index(position_label)

        for group_mode in manifold_groups:
            with_fail = group_mode == "with_experiment_fail"
            manifold_acts = (
                activations_with_fail if with_fail else activations_without_fail
            )
            fail_label = "with-fail" if with_fail else "without-fail"

            # ---- across-layer subspace ------------------------------------
            if "across_layer" in manifold_subspaces and manifold_methods:
                for method in manifold_methods:
                    for n_components in manifold_n_components:
                        basepath = _manifold_basepath_across(
                            output_dir=output_dir,
                            task_vector_alias=task_vector_alias,
                            target_alias=target_alias,
                            n_components=int(n_components),
                            method=method,
                            position_label=position_label,
                            layer_lo=across_lo,
                            layer_hi=across_hi,
                            with_fail=with_fail,
                            gt_filename_suffix=gt_filename_suffix,
                        )
                        title = (
                            f"{method} {n_components}D across-layer | "
                            f"task={task_dataset} | targets={target_alias} | "
                            f"pos={position_label} | "
                            f"layers=L{across_lo}-L{across_hi} | groups={fail_label}"
                        )
                        plot_manifold_for_position(
                            activations_by_group=manifold_acts,
                            position_index=position_index,
                            position_label=position_label,
                            method=method,
                            n_components=int(n_components),
                            basepath=basepath,
                            style=style,
                            title=title,
                            legend_label_map=legend_label_map,
                            start_layer=across_lo,
                            end_layer=across_hi,
                            formats=fmts_call,
                            targets=targets_user,
                            gt_style=gt_style,
                            bg_grey_scale=bg_grey_scale,
                        )
                        print(
                            f"  wrote {basepath}.{{{', '.join(fmts_call)}}} "
                            f"(across {method} {n_components}D; pos={position_label}; "
                            f"{fail_label})"
                        )

            # ---- set-layer subspace (PCA only) ----------------------------
            if (
                "set_layer" in manifold_subspaces
                and "PCA" in manifold_methods
                and set_layers
            ):
                # Set-layer always passes the **full** (with-fail) dict so
                # targets can resolve any group; manifold_group filters the
                # background scatter via background_groups.
                set_layer_acts = activations_with_fail
                bg_groups = (
                    list(activations_with_fail.keys())
                    if with_fail
                    else list(activations_without_fail.keys())
                )
                for set_layer_idx in set_layers:
                    if not (0 <= set_layer_idx < n_layers_full):
                        print(
                            f"  set-layer manifold skipped: set_layer="
                            f"{set_layer_idx} out of range "
                            f"[0, {n_layers_full - 1}]"
                        )
                        continue
                    for n_components in manifold_n_components:
                        basepath = _manifold_basepath_set_layer(
                            output_dir=output_dir,
                            task_vector_alias=task_vector_alias,
                            target_alias=target_alias,
                            n_components=int(n_components),
                            position_label=position_label,
                            set_layer_idx=set_layer_idx,
                            target_layer_lo=target_lo,
                            target_layer_hi=target_hi,
                            with_fail=with_fail,
                            gt_filename_suffix=gt_filename_suffix,
                        )
                        title = (
                            f"PCA {n_components}D set-layer L{set_layer_idx} | "
                            f"task={task_dataset} | targets={target_alias} | "
                            f"pos={position_label} | "
                            f"targets=L{target_lo}-L{target_hi} | "
                            f"background={fail_label}"
                        )
                        plot_manifold_set_layer_for_position(
                            activations_by_group=set_layer_acts,
                            position_index=position_index,
                            position_label=position_label,
                            set_layer_idx=set_layer_idx,
                            target_layer_lo=target_lo,
                            target_layer_hi=target_hi,
                            n_components=int(n_components),
                            basepath=basepath,
                            style=style,
                            title=title,
                            legend_label_map=legend_label_map,
                            background_groups=bg_groups,
                            formats=fmts_call,
                            targets=targets_user,
                            gt_style=gt_style,
                            bg_grey_scale=bg_grey_scale,
                        )
                        print(
                            f"  wrote {basepath}.{{{', '.join(fmts_call)}}} "
                            f"(set-layer PCA {n_components}D; pos={position_label}; "
                            f"setL{set_layer_idx}; {fail_label})"
                        )


def _load_target_folder_for_subplot(
    folder: Path,
    *,
    truth_dirs: np.ndarray,
    expected_n_pre: int,
    expected_n_gen: int,
    role: str,
) -> Optional[Tuple[Path, Dict[str, object], Dict[str, np.ndarray]]]:
    """
    Load a secondary targets folder used by the trajectory_axis
    right-column override (``targets_data_folder_mean`` /
    ``targets_data_folder_individual``).

    Validates the same preconditions ``_run_one_targets_folder`` enforces
    on its primary folder: folder layout parses, pre/gen layout matches
    the task-vector folder, ``activations/`` subdir exists, at least one
    activation pickle loaded, and every group's
    ``(n_pos, n_layers, d_model)`` matches the truth-direction tensor.

    Returns ``(resolved_folder, meta, groups)`` on success, ``None`` on
    any precondition failure (a context-tagged warning is printed).
    """
    resolved = folder.resolve()
    try:
        meta = _parse_run_folder(resolved)
    except ValueError as e:
        print(f"  skip trajectory_axis {role} folder {resolved}: {e}")
        return None
    if int(meta["n_pre"]) != int(expected_n_pre) or int(meta["n_gen"]) != int(
        expected_n_gen
    ):
        print(
            f"  skip trajectory_axis {role} folder {resolved}: "
            f"pre/gen layout {meta['n_pre']}_pre_gen{meta['n_gen']} "
            f"!= task_vector's {expected_n_pre}_pre_gen{expected_n_gen}."
        )
        return None
    activations_dir = resolved / "activations"
    if not activations_dir.is_dir():
        print(
            f"  skip trajectory_axis {role} folder {resolved}: "
            f"no 'activations' subdir."
        )
        return None
    groups = load_microscopic_groups(activations_dir)
    if not groups:
        print(
            f"  skip trajectory_axis {role} folder {resolved}: "
            f"no recognizable activation pickles."
        )
        return None
    n_pos_truth, n_layers_truth, d_model_truth = truth_dirs.shape
    for g, arr in groups.items():
        _, n_pos_g, n_layers_g, d_model_g = arr.shape
        if (n_pos_g, n_layers_g, d_model_g) != (
            n_pos_truth, n_layers_truth, d_model_truth
        ):
            raise ValueError(
                f"Shape mismatch between truth directions and "
                f"{resolved}/{g} (loaded as trajectory_axis {role}): "
                f"truth=({n_pos_truth}, {n_layers_truth}, {d_model_truth}); "
                f"group={(n_pos_g, n_layers_g, d_model_g)}"
            )
    return resolved, meta, groups


def _run_one_targets_folder(
    *,
    target_folder: Path,
    truth_dirs: np.ndarray,
    pos_labels: Sequence[str],
    tv_position_label: str,
    tv_position_index: int,
    mean_data_folder: Optional[Path] = None,
    indiv_data_folder: Optional[Path] = None,
    token_positions_targets: Sequence[str],
    best_n_token_position_targets: int,
    rotation_metric: str,
    targets_user_specs: Sequence[str],
    output_dir: Path,
    task_vector_alias: str,
    target_meta: Dict[str, object],
    title_prefix: str,
    band_kind: str,
    gt_style: str,
    formats: Sequence[str],
    cfg: Optional[ExperimentConfig],
    trajectory_residualPC_variants: Sequence[str],
    trajectory_realignment_variants: Sequence[str],
    trajectory_realignment_footnote: bool,
    trajectory_backgrounds: Sequence[str],
    trajectory_set_layers: Sequence[int],
    trajectory_task_vector_arrow: bool,
    trajectory_start_layer: int,
    trajectory_end_layer: int,
    trajectory_xlim: Optional[Tuple[float, float]],
    trajectory_ylim: Optional[Tuple[float, float]],
    trajectory_axis_polar_rlim: Optional[float],
    bg_grey_scale: bool,
    projection_start_layer: int,
    projection_end_layer: int,
    plot_kinds: Sequence[str],
    manifold_subspaces: Sequence[str],
    manifold_methods: Sequence[str],
    manifold_groups: Sequence[str],
    manifold_n_components: Sequence[int],
    n_sample_manifold: int,
    manifold_across_start_layer: int,
    manifold_across_end_layer: int,
    manifold_PCA_set_layers: Sequence[int],
    manifold_set_start_layer: int,
    manifold_set_end_layer: int,
    include_rotation_cos_sim_curves: bool,
    rotation_cos_sim_line_color: str,
    rotation_cos_sim_band_color: str,
    axis_dirs: Optional[Dict[str, np.ndarray]] = None,
    trajectory_axis_axis_sets: Sequence[Sequence[str]] = (),
    trajectory_axis_start_layer: int = -1,
    trajectory_axis_end_layer: int = -1,
    trajectory_axis_start_layer_mean: Optional[int] = None,
    trajectory_axis_end_layer_mean: Optional[int] = None,
    trajectory_axis_start_layer_individual: Optional[int] = None,
    trajectory_axis_end_layer_individual: Optional[int] = None,
    trajectory_axis_max_target_per_class: int = 50,
    trajectory_axis_line_width_individual: float = 0.8,
    trajectory_axis_line_width_mean: float = 2.5,
    trajectory_axis_individual_line_alpha: float = 0.18,
    trajectory_axis_scene_show_walls: bool = True,
    trajectory_axis_scene_wall_color: Optional[str] = None,
    trajectory_axis_marker_layer_gradient: bool = True,
    trajectory_axis_marker_gradient_start: float = 0.5,
    trajectory_axis_marker_size_target: Optional[float] = None,
    trajectory_axis_marker_size_mean: Optional[float] = None,
    trajectory_axis_sample_seed: int = 0,
    trajectory_axis_show_arrows: bool = False,
    trajectory_axis_library: str = "plotly",
    trajectory_axis_axis_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    projection_axis_names: Sequence[str] = ("honest_T-F",),
    gt_filename_suffix: str = "",
) -> None:
    """
    Compute projections + render projection-line and trajectory figures for
    one targets folder. Skips quietly with a printed note when no usable
    groups are present at any chosen target position.

    The truth direction is sliced at ``tv_position_index`` (computed once
    from the task-vector folder); the targets-side positions are
    resolved here from this folder's own rotation stats and
    ``--token_positions_targets``.
    """
    activations_dir = target_folder / "activations"
    if not activations_dir.is_dir():
        print(
            f"  skip {target_folder}: no 'activations' subdir."
        )
        return

    groups = load_microscopic_groups(activations_dir)
    if not groups:
        print(f"  skip {target_folder}: no recognizable activation pickles.")
        return

    expected_keys = {
        "honest_success_GT_0",
        "honest_success_GT_1",
        "lie_success_GT_0",
        "lie_success_GT_1",
        "lie_fail_GT_0",
        "lie_fail_GT_1",
    }
    missing = sorted(expected_keys - groups.keys())
    if missing:
        print(
            f"  {target_folder}: missing groups {missing} — those curves "
            f"will be omitted."
        )

    # Validate group shapes against the truth-direction tensor's
    # (n_pos, n_layers, d_model) so a mismatched run produces a clear error
    # rather than a silently-zero plot.
    n_pos_truth, n_layers_truth, d_model_truth = truth_dirs.shape
    for g, arr in groups.items():
        _, n_pos_g, n_layers_g, d_model_g = arr.shape
        if (n_pos_g, n_layers_g, d_model_g) != (n_pos_truth, n_layers_truth, d_model_truth):
            raise ValueError(
                f"Shape mismatch between truth directions and "
                f"{target_folder}/{g}: truth=({n_pos_truth}, {n_layers_truth}, "
                f"{d_model_truth}); group={(n_pos_g, n_layers_g, d_model_g)}"
            )

    # Resolve targets-side token positions from this folder's own
    # rotation_stats — independent of the task-vector folder.
    chosen_positions = _resolve_token_positions(
        target_folder,
        pos_labels=pos_labels,
        explicit=token_positions_targets,
        best_n=int(best_n_token_position_targets),
        rotation_metric=rotation_metric,
    )

    # Per-axis projection happens lazily inside the projection-plot block;
    # here we only validate which requested target positions actually exist
    # for this run (the trajectory / manifold / axis blocks need the same
    # ``usable_positions`` list).
    usable_positions: List[str] = []
    for pos_label in chosen_positions:
        if pos_label not in pos_labels:
            print(
                f"  WARN position {pos_label!r} not in run's labels — skipping."
            )
            continue
        usable_positions.append(pos_label)

    if not usable_positions:
        print(
            f"  skip {target_folder}: no requested token positions matched "
            f"its run layout."
        )
        return

    target_alias = str(target_meta["alias"])

    gt0_name, gt1_name = _resolve_gt_label_names(
        cfg, str(target_meta["task_dataset"])
    )

    # Resolve the (group, idx) target list. Empty CLI input → backward-
    # compatible default (sample 0 of every loaded group).
    if targets_user_specs:
        try:
            requested_targets = parse_target_spec(targets_user_specs)
        except ValueError as e:
            print(f"  skip {target_folder}: bad --targets spec — {e}")
            return
        targets = filter_valid_targets(requested_targets, groups)
    else:
        targets = default_targets_for_groups(groups, sample_idx=0)

    written_fmts = ", ".join(formats)

    # ---- Projection-line figure (output_dir/projection/...) -----------------

    if "projection" in plot_kinds:
        n_layers_full = int(truth_dirs.shape[1])
        proj_lo = 0 if int(projection_start_layer) == -1 else int(projection_start_layer)
        proj_hi = (
            n_layers_full - 1
            if int(projection_end_layer) == -1
            else int(projection_end_layer)
        )
        rotation_cos_sim_data = (
            _build_rotation_cos_sim_data(target_folder, usable_positions)
            if include_rotation_cos_sim_curves
            else None
        )
        for axis_name in projection_axis_names:
            if axis_dirs is not None and axis_name in axis_dirs:
                axis_direction = axis_dirs[axis_name]
            elif axis_name == "honest_T-F":
                # honest_T-F == truth_dirs (up to normalization, which the
                # projector reapplies); use it as a free fallback when the
                # caller hasn't fit the axis registry.
                axis_direction = truth_dirs
            else:
                print(
                    f"  WARN projection axis {axis_name!r} not available "
                    f"in axis_dirs — skipping."
                )
                continue

            projections_per_position: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
            for pos_label in usable_positions:
                target_position_index = pos_labels.index(pos_label)
                projections_per_position[pos_label] = project_all_groups(
                    groups,
                    axis_direction,
                    tv_position_index=tv_position_index,
                    proj_position_index=target_position_index,
                )

            title_proj = (
                f"{title_prefix} | task_vector={task_vector_alias} "
                f"(tv_pos={tv_position_label}) | axis={axis_name} | "
                f"targets={target_alias}"
            )
            proj_subdir = output_dir / "projection" / task_vector_alias / axis_name
            proj_subdir.mkdir(parents=True, exist_ok=True)
            proj_basepath = (
                proj_subdir
                / f"microscopic_tv-{tv_position_label}_{target_alias}_L{proj_lo}-L{proj_hi}{gt_filename_suffix}"
            )
            plot_microscopic_grid(
                projections_per_position=projections_per_position,
                position_labels_in_order=usable_positions,
                targets=targets,
                basepath=proj_basepath,
                title=title_proj,
                gt_0_name=gt0_name,
                gt_1_name=gt1_name,
                band_kind=band_kind,
                gt_style=gt_style,
                layer_lo=int(projection_start_layer),
                layer_hi=int(projection_end_layer),
                formats=formats,
                rotation_cos_sim_data=rotation_cos_sim_data,
                rotation_cos_sim_line_color=rotation_cos_sim_line_color,
                rotation_cos_sim_band_color=rotation_cos_sim_band_color,
            )
            print(
                f"  wrote {proj_basepath}.{{{written_fmts}}} "
                f"(axis={axis_name}; tv_pos={tv_position_label}; "
                f"target_positions: {usable_positions})"
            )

    # ---- Trajectory figure(s) (output_dir/trajectory/...) -------------------

    if "trajectory" in plot_kinds:
        n_traj_variants = (
            len(trajectory_residualPC_variants) + len(trajectory_realignment_variants)
        )
        set_layers_list = [int(L) for L in trajectory_set_layers or ()]
        if not targets or n_traj_variants == 0 or not set_layers_list:
            if not targets:
                print(
                    f"  trajectory plots: no usable targets for {target_folder} — skipping."
                )
            elif n_traj_variants == 0:
                print(
                    f"  trajectory plots: no variants requested — skipping."
                )
            else:
                print(
                    f"  trajectory plots: trajectory_set_layer is empty — skipping."
                )
        else:
            pos_label_to_index = {lbl: pos_labels.index(lbl) for lbl in usable_positions}
            traj_subdir = output_dir / "trajectory" / task_vector_alias
            traj_subdir.mkdir(parents=True, exist_ok=True)

            from src.analysis.plots import resolve_manifold_style  # lazy
            traj_style = resolve_manifold_style(
                getattr(cfg, "manifold_plot_style", {}) or {}
                if cfg is not None else {}
            )

            n_layers_full = int(truth_dirs.shape[1])
            traj_lo = 0 if int(trajectory_start_layer) == -1 else int(trajectory_start_layer)
            traj_hi = (
                n_layers_full - 1
                if int(trajectory_end_layer) == -1
                else int(trajectory_end_layer)
            )

            for set_L in set_layers_list:
                if not (0 <= set_L < n_layers_full):
                    print(
                        f"  trajectory plot skipped: set_layer={set_L} out of "
                        f"range [0, {n_layers_full - 1}]."
                    )
                    continue
                for bg in trajectory_backgrounds:
                    title_traj = (
                        f"trajectory ({bg} bg, setL{set_L}) | "
                        f"task_vector={task_vector_alias} "
                        f"(tv_pos={tv_position_label}) | targets={target_alias}"
                    )
                    traj_basepath = (
                        traj_subdir
                        / (
                            f"trajectory_bg-{bg}_tv-{tv_position_label}"
                            f"_setL{set_L}_targetsL{traj_lo}-L{traj_hi}"
                            f"_{target_alias}{gt_filename_suffix}"
                        )
                    )
                    plot_trajectory_grid(
                        groups=groups,
                        truth_dirs=truth_dirs,
                        targets=targets,
                        position_labels_in_order=usable_positions,
                        pos_label_to_index=pos_label_to_index,
                        tv_position_index=tv_position_index,
                        basepath=traj_basepath,
                        title=title_traj,
                        residualPC_variants=trajectory_residualPC_variants,
                        realignment_variants=trajectory_realignment_variants,
                        background=bg,
                        show_task_vector_arrow=trajectory_task_vector_arrow,
                        layer_lo=int(trajectory_start_layer),
                        layer_hi=int(trajectory_end_layer),
                        set_layer=set_L,
                        gt_0_name=gt0_name,
                        gt_1_name=gt1_name,
                        manifold_style=traj_style,
                        bg_grey_scale=bg_grey_scale,
                        show_in_plane_footnote=trajectory_realignment_footnote,
                        xlim=trajectory_xlim,
                        ylim=trajectory_ylim,
                        formats=formats,
                    )
                    print(
                        f"  wrote {traj_basepath}.{{{written_fmts}}} "
                        f"(bg={bg}; setL={set_L}; n_targets={len(targets)})"
                    )

    # ---- Polar axis-trajectory figure(s) (output_dir/trajectory_axis_polar/...) ----

    if (
        "trajectory_axis_polar" in plot_kinds
        and trajectory_axis_axis_sets
        and axis_dirs is not None
    ):
        polar_set_layers_list = [int(L) for L in trajectory_set_layers or ()]
        polar_axis_sets = [
            list(s) for s in trajectory_axis_axis_sets if 2 <= len(s) <= 3
        ]
        if not targets or not polar_set_layers_list or not polar_axis_sets:
            if not targets:
                print(
                    f"  trajectory_axis_polar plots: no usable targets for "
                    f"{target_folder} — skipping."
                )
            elif not polar_axis_sets:
                print(
                    f"  trajectory_axis_polar plots: no axis-set entry with "
                    f"2 or 3 axes — skipping."
                )
            else:
                print(
                    f"  trajectory_axis_polar plots: trajectory_set_layer is empty — skipping."
                )
        else:
            pos_label_to_index = {lbl: pos_labels.index(lbl) for lbl in usable_positions}

            from src.analysis.plots import resolve_manifold_style  # lazy
            polar_style = resolve_manifold_style(
                getattr(cfg, "manifold_plot_style", {}) or {}
                if cfg is not None else {}
            )

            n_layers_full = int(truth_dirs.shape[1])
            polar_lo = 0 if int(trajectory_start_layer) == -1 else int(trajectory_start_layer)
            polar_hi = (
                n_layers_full - 1
                if int(trajectory_end_layer) == -1
                else int(trajectory_end_layer)
            )

            for axis_set in polar_axis_sets:
                for set_L in polar_set_layers_list:
                    if not (0 <= set_L < n_layers_full):
                        print(
                            f"  trajectory_axis_polar plot skipped: set_layer={set_L} out of "
                            f"range [0, {n_layers_full - 1}]."
                        )
                        continue
                    for bg in trajectory_backgrounds:
                        title_polar = (
                            f"trajectory axis polar ({bg} bg, setL{set_L}) | "
                            f"axes={list(axis_set)} | "
                            f"task_vector={task_vector_alias} "
                            f"(tv_pos={tv_position_label}) | targets={target_alias}"
                        )
                        polar_basepath = _trajectory_axis_polar_basepath(
                            output_dir=output_dir,
                            task_vector_alias=task_vector_alias,
                            axis_names=axis_set,
                            target_alias=target_alias,
                            tv_position_label=tv_position_label,
                            set_layer=set_L,
                            background=bg,
                            layer_lo=polar_lo,
                            layer_hi=polar_hi,
                            gt_filename_suffix=gt_filename_suffix,
                        )
                        plot_trajectory_axis_polar_grid(
                            groups=groups,
                            axis_dirs=axis_dirs,
                            axis_names=list(axis_set),
                            targets=targets,
                            position_labels_in_order=usable_positions,
                            pos_label_to_index=pos_label_to_index,
                            tv_position_index=tv_position_index,
                            basepath=polar_basepath,
                            title=title_polar,
                            background=bg,
                            layer_lo=int(trajectory_start_layer),
                            layer_hi=int(trajectory_end_layer),
                            set_layer=set_L,
                            gt_0_name=gt0_name,
                            gt_1_name=gt1_name,
                            manifold_style=polar_style,
                            bg_grey_scale=bg_grey_scale,
                            rlim=trajectory_axis_polar_rlim,
                            formats=formats,
                        )
                        n_axes_polar = len(axis_set)
                        suffix = " (+ .html)" if n_axes_polar == 3 else ""
                        print(
                            f"  wrote {polar_basepath}.{{{written_fmts}}}{suffix} "
                            f"(axis-polar {n_axes_polar}D; axes={list(axis_set)}; "
                            f"bg={bg}; setL={set_L}; n_targets={len(targets)})"
                        )

    # ---- Axis-trajectory figures (output_dir/trajectory_axis/...) ----------
    #
    # Optional dual-source mode: ``mean_data_folder`` / ``indiv_data_folder``
    # let the right-column subplot pull the bold per-class mean and the
    # faint per-class individual overlay from different cache-run folders.
    # When either override is set, token positions are resolved per source
    # (or shared, if ``token_positions.targets`` is non-empty explicitly).

    if (
        "trajectory_axis" in plot_kinds
        and trajectory_axis_axis_sets
        and axis_dirs is not None
    ):
        n_pre_local = int(target_meta["n_pre"])
        n_gen_local = int(target_meta["n_gen"])

        # Resolve the mean / individual source folders, defaulting to the
        # current loop's primary ``target_folder`` when not overridden.
        if mean_data_folder is not None and mean_data_folder.resolve() != target_folder:
            mean_loaded = _load_target_folder_for_subplot(
                mean_data_folder,
                truth_dirs=truth_dirs,
                expected_n_pre=n_pre_local,
                expected_n_gen=n_gen_local,
                role="mean",
            )
        else:
            mean_loaded = None
        if indiv_data_folder is not None and indiv_data_folder.resolve() != target_folder:
            if (
                mean_loaded is not None
                and indiv_data_folder.resolve() == mean_loaded[0]
            ):
                indiv_loaded = mean_loaded
            else:
                indiv_loaded = _load_target_folder_for_subplot(
                    indiv_data_folder,
                    truth_dirs=truth_dirs,
                    expected_n_pre=n_pre_local,
                    expected_n_gen=n_gen_local,
                    role="individual",
                )
        else:
            indiv_loaded = None

        if mean_loaded is not None:
            mean_folder_eff, mean_meta_eff, mean_groups_eff = mean_loaded
        else:
            mean_folder_eff = target_folder
            mean_meta_eff = target_meta
            mean_groups_eff = groups
        if indiv_loaded is not None:
            indiv_folder_eff, indiv_meta_eff, indiv_groups_eff = indiv_loaded
        else:
            indiv_folder_eff = target_folder
            indiv_meta_eff = target_meta
            indiv_groups_eff = groups

        dual_mode = (
            mean_folder_eff != target_folder
            or indiv_folder_eff != target_folder
        )

        # Position resolution. In dual_mode, an explicit
        # ``token_positions.targets`` list overrides per-folder auto-
        # resolution and applies to both subplots; otherwise we resolve
        # ``best_n`` positions independently from each folder's own
        # ``stage_stats/rotation_stats.json``.
        if dual_mode:
            explicit_positions = [
                lbl for lbl in token_positions_targets if lbl in pos_labels
            ]
            if explicit_positions:
                mean_positions = list(dict.fromkeys(explicit_positions))
                indiv_positions = list(mean_positions)
            else:
                mean_positions = (
                    _resolve_token_positions(
                        mean_folder_eff,
                        pos_labels=pos_labels,
                        explicit=[],
                        best_n=int(best_n_token_position_targets),
                        rotation_metric=rotation_metric,
                    )
                    if mean_folder_eff != target_folder
                    else list(usable_positions)
                )
                indiv_positions = (
                    _resolve_token_positions(
                        indiv_folder_eff,
                        pos_labels=pos_labels,
                        explicit=[],
                        best_n=int(best_n_token_position_targets),
                        rotation_metric=rotation_metric,
                    )
                    if indiv_folder_eff != target_folder
                    else list(usable_positions)
                )
            mean_positions = [lbl for lbl in mean_positions if lbl in pos_labels]
            indiv_positions = [
                lbl for lbl in indiv_positions if lbl in pos_labels
            ]
            position_pairs = [
                (m, i) for m in mean_positions for i in indiv_positions
            ]
        else:
            position_pairs = [(p, p) for p in usable_positions]

        n_layers_full_axis = int(truth_dirs.shape[1])
        at_lo = 0 if int(trajectory_axis_start_layer) == -1 else int(trajectory_axis_start_layer)
        at_hi = (
            n_layers_full_axis - 1
            if int(trajectory_axis_end_layer) == -1
            else int(trajectory_axis_end_layer)
        )
        mean_alias_eff = str(mean_meta_eff["alias"])
        indiv_alias_eff = str(indiv_meta_eff["alias"])
        # Filename primary alias = mean folder's alias (per user choice).
        # When the individual folder differs, encode both aliases.
        if mean_alias_eff != indiv_alias_eff:
            combined_alias = f"{mean_alias_eff}_indiv-{indiv_alias_eff}"
        else:
            combined_alias = mean_alias_eff

        if not targets:
            print(
                f"  trajectory_axis plots: no usable targets for "
                f"{target_folder} — skipping."
            )
        elif not position_pairs:
            print(
                f"  trajectory_axis plots: no resolved target token "
                f"positions for {target_folder} (mean={mean_alias_eff}, "
                f"indiv={indiv_alias_eff}) — skipping."
            )
        else:
            for axis_set in trajectory_axis_axis_sets:
                for mean_pos_label, indiv_pos_label in position_pairs:
                    mean_pos_index = pos_labels.index(mean_pos_label)
                    indiv_pos_index = pos_labels.index(indiv_pos_label)
                    # Left column slicing uses the primary target_folder
                    # context. Pick a sensible single label/index for the
                    # left column: prefer the mean position (filename keeps
                    # the encoding flexible). When dual_mode is off the two
                    # positions are equal anyway.
                    left_pos_label = mean_pos_label
                    left_pos_index = mean_pos_index
                    if mean_pos_label == indiv_pos_label:
                        pos_label_for_filename = mean_pos_label
                    else:
                        pos_label_for_filename = (
                            f"meanP{mean_pos_label}_indivP{indiv_pos_label}"
                        )
                    basepath = _trajectory_axis_basepath(
                        output_dir=output_dir,
                        task_vector_alias=task_vector_alias,
                        axis_names=axis_set,
                        target_alias=combined_alias,
                        tv_position_label=tv_position_label,
                        target_position_label=pos_label_for_filename,
                        layer_lo=at_lo,
                        layer_hi=at_hi,
                        gt_filename_suffix=gt_filename_suffix,
                    )
                    if dual_mode:
                        title = (
                            f"trajectory axis | task_vector={task_vector_alias} "
                            f"(tv_pos={tv_position_label}) | targets={target_alias} "
                            f"| mean={mean_alias_eff} | indiv={indiv_alias_eff}"
                        )
                    else:
                        title = (
                            f"trajectory axis | task_vector={task_vector_alias} "
                            f"(tv_pos={tv_position_label}) | targets={target_alias}"
                        )
                    plot_trajectory_axis_grid(
                        groups=groups,
                        axis_dirs=axis_dirs,
                        axis_names=list(axis_set),
                        targets=targets,
                        target_position_index=left_pos_index,
                        tv_position_index=tv_position_index,
                        target_position_label=left_pos_label,
                        layer_lo=int(trajectory_axis_start_layer),
                        layer_hi=int(trajectory_axis_end_layer),
                        layer_lo_mean=(
                            int(trajectory_axis_start_layer_mean)
                            if trajectory_axis_start_layer_mean is not None
                            else None
                        ),
                        layer_hi_mean=(
                            int(trajectory_axis_end_layer_mean)
                            if trajectory_axis_end_layer_mean is not None
                            else None
                        ),
                        layer_lo_individual=(
                            int(trajectory_axis_start_layer_individual)
                            if trajectory_axis_start_layer_individual is not None
                            else None
                        ),
                        layer_hi_individual=(
                            int(trajectory_axis_end_layer_individual)
                            if trajectory_axis_end_layer_individual is not None
                            else None
                        ),
                        band_kind=band_kind,
                        gt_0_name=gt0_name,
                        gt_1_name=gt1_name,
                        gt_style=gt_style,
                        max_target_per_class=int(trajectory_axis_max_target_per_class),
                        line_width_individual=float(trajectory_axis_line_width_individual),
                        line_width_mean=float(trajectory_axis_line_width_mean),
                        individual_line_alpha=float(trajectory_axis_individual_line_alpha),
                        scene_show_walls=bool(trajectory_axis_scene_show_walls),
                        scene_wall_color=trajectory_axis_scene_wall_color,
                        marker_layer_gradient=bool(trajectory_axis_marker_layer_gradient),
                        marker_gradient_start=float(trajectory_axis_marker_gradient_start),
                        marker_size_target=trajectory_axis_marker_size_target,
                        marker_size_mean=trajectory_axis_marker_size_mean,
                        sample_seed=int(trajectory_axis_sample_seed),
                        show_arrows=bool(trajectory_axis_show_arrows),
                        library=str(trajectory_axis_library),
                        axis_ranges=trajectory_axis_axis_ranges,
                        groups_mean=mean_groups_eff if dual_mode else None,
                        groups_individual=indiv_groups_eff if dual_mode else None,
                        target_position_index_mean=mean_pos_index if dual_mode else None,
                        target_position_index_individual=indiv_pos_index if dual_mode else None,
                        target_position_label_mean=mean_pos_label if dual_mode else None,
                        target_position_label_individual=indiv_pos_label if dual_mode else None,
                        basepath=basepath,
                        title=title,
                        formats=formats,
                    )
                    n_axes = len(axis_set)
                    suffix = " (+ .html)" if n_axes == 3 else ""
                    if dual_mode:
                        pos_log = (
                            f"mean_pos={mean_pos_label}, indiv_pos={indiv_pos_label}"
                        )
                    else:
                        pos_log = f"pos={mean_pos_label}"
                    print(
                        f"  wrote {basepath}.{{{written_fmts}}}{suffix} "
                        f"(axis-traj {n_axes}D; axes={list(axis_set)}; "
                        f"{pos_log})"
                    )

    # ---- Manifold figures (output_dir/manifolds/...) -----------------------

    if "manifold" not in plot_kinds:
        return

    from src.analysis.plots import resolve_manifold_style  # lazy

    style = resolve_manifold_style(
        getattr(cfg, "manifold_plot_style", {}) or {} if cfg is not None else {}
    )
    # plot_manifold_* expects user-facing group keys (control_GT_0 etc.) to
    # line up with the style dict. Translate at the boundary; targets too.
    activations_user = _to_user_facing_groups(groups)
    targets_user = _to_user_facing_targets(targets)
    manifold_legend_label_map: Dict[str, str] = {
        "control_GT_0":         f"control / {gt0_name}",
        "control_GT_1":         f"control / {gt1_name}",
        "experiment_GT_0":      f"experiment / {gt0_name}",
        "experiment_GT_1":      f"experiment / {gt1_name}",
        "experiment_GT_0_fail": f"experiment-fail / {gt0_name}",
        "experiment_GT_1_fail": f"experiment-fail / {gt1_name}",
    }

    _run_manifold_plots_for_targets_folder(
        activations_user=activations_user,
        targets_user=targets_user,
        pos_labels=pos_labels,
        chosen_positions=usable_positions,
        output_dir=output_dir,
        task_vector_alias=task_vector_alias,
        target_alias=target_alias,
        target_meta=target_meta,
        legend_label_map=manifold_legend_label_map,
        style=style,
        manifold_subspaces=manifold_subspaces,
        manifold_methods=manifold_methods,
        manifold_groups=manifold_groups,
        manifold_n_components=manifold_n_components,
        n_sample_manifold=n_sample_manifold,
        manifold_across_start_layer=manifold_across_start_layer,
        manifold_across_end_layer=manifold_across_end_layer,
        manifold_PCA_set_layers=manifold_PCA_set_layers,
        manifold_set_start_layer=manifold_set_start_layer,
        manifold_set_end_layer=manifold_set_end_layer,
        gt_style=gt_style,
        formats=formats,
        bg_grey_scale=bg_grey_scale,
        gt_filename_suffix=gt_filename_suffix,
    )


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #


def run_microscopic(
    task_vector_data_folder: Path,
    targets_data_folders: Sequence[Path],
    *,
    targets_data_folder_mean: Optional[Path] = None,
    targets_data_folder_individual: Optional[Path] = None,
    token_positions_task_vector: Sequence[str] = ("-1",),
    token_positions_targets: Sequence[str] = (),
    best_n_token_position_targets: int = 1,
    targets: Sequence[str] = (),
    output_dir: Optional[Path] = None,
    config_override: Optional[Path] = None,
    rotation_metric: Optional[str] = None,
    band_kind: str = "std",
    gt_style: str = "line_style",
    formats: Sequence[str] = ("pdf", "svg"),
    trajectory_residualPC_variants: Sequence[str] = ("raw", "unit_x", "unit_xy"),
    trajectory_realignment_variants: Sequence[str] = (
        "raw_projection", "cosine_similarity",
    ),
    trajectory_realignment_footnote: bool = False,
    trajectory_backgrounds: Sequence[str] = ("none",),
    trajectory_set_layers: Sequence[int] = (),
    trajectory_task_vector_arrow: bool = False,
    trajectory_start_layer: int = -1,
    trajectory_end_layer: int = -1,
    trajectory_xlim: Optional[Tuple[float, float]] = None,
    trajectory_ylim: Optional[Tuple[float, float]] = None,
    trajectory_axis_polar_rlim: Optional[float] = None,
    bg_grey_scale: bool = False,
    projection_start_layer: int = -1,
    projection_end_layer: int = -1,
    plot_kinds: Sequence[str] = ("manifold", "trajectory", "projection"),
    manifold_subspaces: Sequence[str] = ("across_layer", "set_layer"),
    manifold_methods: Sequence[str] = ("PCA",),
    manifold_groups: Sequence[str] = ("without_experiment_fail",),
    manifold_n_components: Sequence[int] = (2,),
    n_sample_manifold: int = 0,
    manifold_across_start_layer: int = -1,
    manifold_across_end_layer: int = -1,
    manifold_PCA_set_layers: Sequence[int] = (),
    manifold_set_start_layer: int = -1,
    manifold_set_end_layer: int = -1,
    include_rotation_cos_sim_curves: bool = False,
    rotation_cos_sim_line_color: str = "black",
    rotation_cos_sim_band_color: str = "grey",
    trajectory_axis_axis_sets: Sequence[Sequence[str]] = (),
    trajectory_axis_start_layer: int = -1,
    trajectory_axis_end_layer: int = -1,
    trajectory_axis_start_layer_mean: Optional[int] = None,
    trajectory_axis_end_layer_mean: Optional[int] = None,
    trajectory_axis_start_layer_individual: Optional[int] = None,
    trajectory_axis_end_layer_individual: Optional[int] = None,
    trajectory_axis_max_target_per_class: int = 50,
    trajectory_axis_line_width_individual: float = 0.8,
    trajectory_axis_line_width_mean: float = 2.5,
    trajectory_axis_individual_line_alpha: float = 0.18,
    trajectory_axis_scene_show_walls: bool = True,
    trajectory_axis_scene_wall_color: Optional[str] = None,
    trajectory_axis_marker_layer_gradient: bool = True,
    trajectory_axis_marker_gradient_start: float = 0.5,
    trajectory_axis_marker_size_target: Optional[float] = None,
    trajectory_axis_marker_size_mean: Optional[float] = None,
    trajectory_axis_sample_seed: int = 0,
    trajectory_axis_show_arrows: bool = False,
    trajectory_axis_library: str = "plotly",
    trajectory_axis_axis_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    gt_filename_suffix: str = "",
) -> None:
    """
    End-to-end orchestration. See module docstring for usage.
    """
    task_vector_data_folder = task_vector_data_folder.resolve()
    activations_dir = task_vector_data_folder / "activations"
    if not activations_dir.is_dir():
        raise FileNotFoundError(
            f"task_vector_data_folder has no 'activations' subdir: "
            f"{task_vector_data_folder}"
        )

    tv_meta = _parse_run_folder(task_vector_data_folder)
    n_pre = int(tv_meta["n_pre"])
    n_gen = int(tv_meta["n_gen"])
    pos_labels = position_labels(n_pre, n_gen)

    # Build the truth direction tensor from the task-vector folder's control
    # activations. Both control_GT_{0,1} pickles must exist; that's the only
    # hard precondition for this script.
    tv_groups = load_microscopic_groups(
        activations_dir,
        require_honest=True,
    )
    truth_dirs = compute_truth_directions(
        tv_groups["honest_success_GT_0"],
        tv_groups["honest_success_GT_1"],
    )

    # Determine which DiM axes need fitting. We fit the union of:
    #   1. Axes referenced by ``trajectory_axis_axis_sets`` (used for the
    #      trajectory_axis / trajectory_axis_polar plots).
    #   2. ``projection_axis_names`` — the axes the projection-line plots
    #      project onto. When ``trajectory_axis_axis_sets`` is empty this
    #      falls back to ``["honest_T-F"]``, preserving the historical
    #      single-axis projection behavior.
    # An unused registry entry that needs a group missing from ``tv_groups``
    # won't raise because it isn't fit.
    seen_axes: set = set()
    used_axes_unique: List[str] = []
    for axis_set in trajectory_axis_axis_sets:
        for name in axis_set:
            if name in seen_axes:
                continue
            seen_axes.add(name)
            used_axes_unique.append(name)

    projection_axis_names: List[str] = (
        list(used_axes_unique) if used_axes_unique else ["honest_T-F"]
    )
    for name in projection_axis_names:
        if name in seen_axes:
            continue
        seen_axes.add(name)
        used_axes_unique.append(name)

    axis_dirs: Optional[Dict[str, np.ndarray]] = None
    if used_axes_unique:
        axis_dirs = fit_axis_directions(tv_groups, used_axes_unique)

    # Always load ``config/default.yaml`` so the script picks up the project
    # default ``rotation_metric`` (and GT label maps) without requiring
    # ``--config_override``. ``--config_override`` layers on top of the
    # default. ``--rotation_metric`` (when given) takes precedence over both.
    cfg: ExperimentConfig = load_yaml_config(
        default_config_path(), override_path=config_override
    )
    if rotation_metric is None:
        rotation_metric = getattr(cfg, "rotation_metric", "largest_drop_any_early_to_late")

    # Task-vector positions. Default sentinel "-1" means "use top-1 from the
    # task-vector folder's rotation stats"; otherwise use the literal label.
    # Multiple positions emit one figure set each; duplicates are removed
    # while preserving order.
    tv_positions: List[Tuple[str, int, str]] = []
    seen_tv_labels: set = set()
    for raw_label in (token_positions_task_vector or ["-1"]):
        if raw_label == "-1":
            resolved_label = _resolve_tv_top_position(
                task_vector_data_folder,
                pos_labels=pos_labels,
                rotation_metric=rotation_metric,
            )
            source = "top-1 from rotation_stats"
        else:
            if raw_label not in pos_labels:
                raise ValueError(
                    f"--token_positions_task_vector entry "
                    f"{raw_label!r} not in available labels "
                    f"{list(pos_labels)}"
                )
            resolved_label = raw_label
            source = "explicit override"
        if resolved_label in seen_tv_labels:
            continue
        seen_tv_labels.add(resolved_label)
        tv_positions.append(
            (resolved_label, pos_labels.index(resolved_label), source)
        )

    # Default output_dir: results/plots/<model_alias>/microscopic/.
    # Explicit ``--output_dir`` may include placeholder segments that get
    # substituted from the *task-vector* run's parsed metadata, so a single
    # output-root template can be reused across runs without hard-coding
    # the model name. Supported placeholders:
    #
    #   <model_alias>        e.g. 'Llama-3.1-8B-Instruct'
    #   <task_dataset>       e.g. 'lying_azaria-mitchell'
    #   <sys_prompt_segment> e.g. 'SysPrompt048'
    #   <seed>               e.g. '0'
    #   <task_vector_alias>  '<task_dataset>_<sys_prompt_segment>_seed<seed>'
    if output_dir is None or str(output_dir) in ("", "."):
        output_dir = (
            PROJECT_ROOT / "results" / "plots"
            / str(tv_meta["model_alias"]) / "microscopic"
        )
    else:
        placeholders = {
            "<model_alias>":        str(tv_meta["model_alias"]),
            "<task_dataset>":       str(tv_meta["task_dataset"]),
            "<sys_prompt_segment>": str(tv_meta["sys_prompt_segment"]),
            "<seed>":               str(tv_meta["seed"]),
            "<task_vector_alias>":  str(tv_meta["alias"]),
        }
        out_str = str(output_dir)
        for ph, val in placeholders.items():
            out_str = out_str.replace(ph, val)
        output_dir = Path(out_str)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tv_pos_summary = ", ".join(
        f"{label} ({src})" for label, _idx, src in tv_positions
    )
    print(
        f"Microscopic plots:\n"
        f"  task_vector_data_folder            = {task_vector_data_folder}\n"
        f"  targets_data_folders               = {[str(p) for p in targets_data_folders]}\n"
        f"  targets_data_folder_mean           = {targets_data_folder_mean if targets_data_folder_mean is not None else '<inherit per-loop>'}\n"
        f"  targets_data_folder_individual     = {targets_data_folder_individual if targets_data_folder_individual is not None else '<inherit per-loop>'}\n"
        f"  rotation_metric                    = {rotation_metric}\n"
        f"  task_vector token positions        = {tv_pos_summary}\n"
        f"  token_positions_targets            = {list(token_positions_targets)}\n"
        f"  best_n_token_position_targets      = {best_n_token_position_targets}\n"
        f"  targets                            = {list(targets) or '<auto: sample 0 per group>'}\n"
        f"  trajectory residualPC variants     = {list(trajectory_residualPC_variants)}\n"
        f"  trajectory realignment variants    = {list(trajectory_realignment_variants)}\n"
        f"  trajectory realignment footnote    = {trajectory_realignment_footnote}\n"
        f"  trajectory backgrounds             = {list(trajectory_backgrounds)}\n"
        f"  trajectory set_layer indices       = {list(trajectory_set_layers)}\n"
        f"  trajectory task_vector arrow       = {trajectory_task_vector_arrow}\n"
        f"  trajectory layer range             = [{trajectory_start_layer}, {trajectory_end_layer}]\n"
        f"  trajectory xlim                    = {list(trajectory_xlim) if trajectory_xlim is not None else '<auto>'}\n"
        f"  trajectory ylim                    = {list(trajectory_ylim) if trajectory_ylim is not None else '<auto>'}\n"
        f"  trajectory_axis_polar rlim         = {trajectory_axis_polar_rlim if trajectory_axis_polar_rlim is not None else '<auto>'}\n"
        f"  bg_grey_scale                      = {bg_grey_scale}\n"
        f"  projection layer range             = [{projection_start_layer}, {projection_end_layer}]\n"
        f"  projection axis names              = {list(projection_axis_names)}\n"
        f"  gt_style                           = {gt_style}\n"
        f"  plot_kinds                         = {list(plot_kinds)}\n"
        f"  manifold subspaces                 = {list(manifold_subspaces)}\n"
        f"  manifold methods                   = {list(manifold_methods)}\n"
        f"  manifold groups                    = {list(manifold_groups)}\n"
        f"  manifold n_components              = {list(manifold_n_components)}\n"
        f"  n_sample_manifold                  = {n_sample_manifold}\n"
        f"  manifold across-layer range        = [{manifold_across_start_layer}, {manifold_across_end_layer}]\n"
        f"  manifold PCA set_layer indices     = {list(manifold_PCA_set_layers)}\n"
        f"  manifold set-layer target range    = [{manifold_set_start_layer}, {manifold_set_end_layer}]\n"
        f"  trajectory_axis axis sets          = {[list(s) for s in trajectory_axis_axis_sets]}\n"
        f"  trajectory_axis layer range        = [{trajectory_axis_start_layer}, {trajectory_axis_end_layer}]\n"
        f"  trajectory_axis layer range (mean) = "
        f"[{trajectory_axis_start_layer_mean if trajectory_axis_start_layer_mean is not None else '<default>'}, "
        f"{trajectory_axis_end_layer_mean if trajectory_axis_end_layer_mean is not None else '<default>'}]\n"
        f"  trajectory_axis layer range (indiv)= "
        f"[{trajectory_axis_start_layer_individual if trajectory_axis_start_layer_individual is not None else '<default>'}, "
        f"{trajectory_axis_end_layer_individual if trajectory_axis_end_layer_individual is not None else '<default>'}]\n"
        f"  trajectory_axis axis_ranges        = {dict(trajectory_axis_axis_ranges) if trajectory_axis_axis_ranges else '<auto>'}\n"
        f"  output_dir                         = {output_dir}\n"
        f"  task_vector alias                  = {tv_meta['alias']}"
    )

    title_prefix = "microscopic / proj onto truth dir"
    for target_path in targets_data_folders:
        target_path = target_path.resolve()
        try:
            target_meta = _parse_run_folder(target_path)
        except ValueError as e:
            print(f"  skip {target_path}: {e}")
            continue

        # Cross-run sanity check: pickle layouts must agree on n_pre / n_gen
        # so the position labels line up.
        if target_meta["n_pre"] != n_pre or target_meta["n_gen"] != n_gen:
            print(
                f"  skip {target_path}: pre/gen layout {target_meta['n_pre']}_pre"
                f"_gen{target_meta['n_gen']} != task_vector's {n_pre}_pre_gen{n_gen}."
            )
            continue

        for tv_label, tv_index, _tv_source in tv_positions:
            _run_one_targets_folder(
                target_folder=target_path,
                truth_dirs=truth_dirs,
                pos_labels=pos_labels,
                tv_position_label=tv_label,
                tv_position_index=tv_index,
                mean_data_folder=targets_data_folder_mean,
                indiv_data_folder=targets_data_folder_individual,
                token_positions_targets=token_positions_targets,
                best_n_token_position_targets=best_n_token_position_targets,
                rotation_metric=rotation_metric,
                targets_user_specs=targets,
                output_dir=output_dir,
                task_vector_alias=str(tv_meta["alias"]),
                target_meta=target_meta,
                title_prefix=title_prefix,
                band_kind=band_kind,
                gt_style=gt_style,
                formats=tuple(formats),
                cfg=cfg,
                trajectory_residualPC_variants=trajectory_residualPC_variants,
                trajectory_realignment_variants=trajectory_realignment_variants,
                trajectory_realignment_footnote=trajectory_realignment_footnote,
                trajectory_backgrounds=trajectory_backgrounds,
                trajectory_set_layers=trajectory_set_layers,
                trajectory_task_vector_arrow=trajectory_task_vector_arrow,
                trajectory_start_layer=trajectory_start_layer,
                trajectory_end_layer=trajectory_end_layer,
                trajectory_xlim=trajectory_xlim,
                trajectory_ylim=trajectory_ylim,
                trajectory_axis_polar_rlim=trajectory_axis_polar_rlim,
                bg_grey_scale=bg_grey_scale,
                projection_start_layer=projection_start_layer,
                projection_end_layer=projection_end_layer,
                plot_kinds=plot_kinds,
                manifold_subspaces=manifold_subspaces,
                manifold_methods=manifold_methods,
                manifold_groups=manifold_groups,
                manifold_n_components=manifold_n_components,
                n_sample_manifold=n_sample_manifold,
                manifold_across_start_layer=manifold_across_start_layer,
                manifold_across_end_layer=manifold_across_end_layer,
                manifold_PCA_set_layers=manifold_PCA_set_layers,
                manifold_set_start_layer=manifold_set_start_layer,
                manifold_set_end_layer=manifold_set_end_layer,
                include_rotation_cos_sim_curves=include_rotation_cos_sim_curves,
                rotation_cos_sim_line_color=rotation_cos_sim_line_color,
                rotation_cos_sim_band_color=rotation_cos_sim_band_color,
                axis_dirs=axis_dirs,
                trajectory_axis_axis_sets=trajectory_axis_axis_sets,
                trajectory_axis_start_layer=trajectory_axis_start_layer,
                trajectory_axis_end_layer=trajectory_axis_end_layer,
                trajectory_axis_start_layer_mean=trajectory_axis_start_layer_mean,
                trajectory_axis_end_layer_mean=trajectory_axis_end_layer_mean,
                trajectory_axis_start_layer_individual=trajectory_axis_start_layer_individual,
                trajectory_axis_end_layer_individual=trajectory_axis_end_layer_individual,
                trajectory_axis_max_target_per_class=trajectory_axis_max_target_per_class,
                trajectory_axis_line_width_individual=trajectory_axis_line_width_individual,
                trajectory_axis_line_width_mean=trajectory_axis_line_width_mean,
                trajectory_axis_individual_line_alpha=trajectory_axis_individual_line_alpha,
                trajectory_axis_scene_show_walls=trajectory_axis_scene_show_walls,
                trajectory_axis_scene_wall_color=trajectory_axis_scene_wall_color,
                trajectory_axis_marker_layer_gradient=trajectory_axis_marker_layer_gradient,
                trajectory_axis_marker_gradient_start=trajectory_axis_marker_gradient_start,
                trajectory_axis_marker_size_target=trajectory_axis_marker_size_target,
                trajectory_axis_marker_size_mean=trajectory_axis_marker_size_mean,
                trajectory_axis_sample_seed=trajectory_axis_sample_seed,
                trajectory_axis_show_arrows=trajectory_axis_show_arrows,
                trajectory_axis_library=trajectory_axis_library,
                trajectory_axis_axis_ranges=trajectory_axis_axis_ranges,
                projection_axis_names=projection_axis_names,
                gt_filename_suffix=gt_filename_suffix,
            )


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def _load_microscopic_config(path: Path) -> Dict[str, Any]:
    """Load the microscopic YAML config (no merging — self-contained)."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"Top-level YAML in {path} must be a mapping; got "
            f"{type(raw).__name__}."
        )
    return raw


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """CLI for the microscopic projection + trajectory plots.

    All knobs live in ``config/microscopic.yaml``; the CLI exposes only a
    small override allow-list. Pass ``--config <path>`` to point at a
    different YAML; the listed flags override the matching keys.
    """
    p = argparse.ArgumentParser(
        description=(
            "Plot per-datapoint projections onto the per-layer truth direction "
            "and 2D layer trajectories. All parameters live in the YAML "
            "config; CLI flags below override the matching keys."
        )
    )
    p.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "microscopic.yaml",
        help=(
            "Path to the microscopic YAML config. Default: "
            "config/microscopic.yaml."
        ),
    )
    # Override allow-list — high-frequency knobs that are convenient to
    # change without editing the YAML. Each one overrides the matching
    # top-level key in the config when supplied.
    p.add_argument(
        "--task_vector_data_folder",
        type=Path,
        default=None,
        help="Override io.task_vector_data_folder.",
    )
    p.add_argument(
        "--targets_data_folder",
        type=Path,
        nargs="+",
        default=None,
        help="Override io.targets_data_folder.",
    )
    p.add_argument(
        "--targets_data_folder_mean",
        type=Path,
        default=None,
        help="Override io.targets_data_folder_mean (single path).",
    )
    p.add_argument(
        "--targets_data_folder_individual",
        type=Path,
        default=None,
        help="Override io.targets_data_folder_individual (single path).",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Override io.output_dir.",
    )
    p.add_argument(
        "--plot_kinds",
        type=str,
        nargs="+",
        default=None,
        choices=(
            "manifold", "trajectory", "trajectory_axis_polar", "projection",
            "trajectory_axis",
        ),
        help="Override plot_kinds.",
    )
    p.add_argument(
        "--formats",
        type=str,
        nargs="+",
        default=None,
        help="Override io.formats.",
    )
    p.add_argument(
        "--trajectory_axis_start_layer",
        type=int,
        default=-1,
        help="Override trajectory_axis.start_layer. -1 = use YAML value.",
    )
    p.add_argument(
        "--trajectory_axis_end_layer",
        type=int,
        default=-1,
        help="Override trajectory_axis.end_layer. -1 = use YAML value.",
    )
    p.add_argument(
        "--trajectory_axis_marker_size_target",
        type=float,
        default=None,
        help=(
            "Override trajectory_axis.marker_size_target (left subplot, "
            "per-target sample). In Plotly pixel units / matplotlib points. "
            "Omitted = use YAML value (or per-dim defaults if YAML is null)."
        ),
    )
    p.add_argument(
        "--trajectory_axis_marker_size_mean",
        type=float,
        default=None,
        help=(
            "Override trajectory_axis.marker_size_mean (right subplot, "
            "per-class mean). In Plotly pixel units / matplotlib points. "
            "Omitted = use YAML value (or per-dim defaults if YAML is null)."
        ),
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point.

    Loads ``config/microscopic.yaml`` (or ``--config <path>``), applies the
    CLI override allow-list on top, and calls :func:`run_microscopic`.
    """
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    args = parse_args(argv)

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (PROJECT_ROOT / cfg_path).resolve()
    cfg = _load_microscopic_config(cfg_path)

    io_cfg: Mapping[str, Any] = cfg.get("io") or {}
    tp_cfg: Mapping[str, Any] = cfg.get("token_positions") or {}
    proj_cfg: Mapping[str, Any] = cfg.get("projection") or {}
    traj_cfg: Mapping[str, Any] = cfg.get("trajectory") or {}
    taxis_cfg: Mapping[str, Any] = cfg.get("trajectory_axis") or {}
    tpolar_cfg: Mapping[str, Any] = cfg.get("trajectory_axis_polar") or {}
    man_cfg: Mapping[str, Any] = cfg.get("manifold") or {}

    # --- I/O with CLI overrides ---
    task_vector_data_folder = (
        args.task_vector_data_folder
        if args.task_vector_data_folder is not None
        else io_cfg.get("task_vector_data_folder")
    )
    if task_vector_data_folder is None:
        raise ValueError(
            "task_vector_data_folder is required: set io.task_vector_data_folder "
            "in the YAML or pass --task_vector_data_folder."
        )
    task_vector_data_folder = Path(task_vector_data_folder)

    targets_data_folders_raw = (
        args.targets_data_folder
        if args.targets_data_folder is not None
        else io_cfg.get("targets_data_folder") or []
    )
    if not targets_data_folders_raw:
        raise ValueError(
            "At least one targets data folder is required: set "
            "io.targets_data_folder in the YAML or pass --targets_data_folder."
        )
    targets_data_folders = [Path(p) for p in targets_data_folders_raw]

    # Optional per-subplot overrides for the trajectory_axis right column.
    # Each takes a SINGLE path; null/empty = fall back to the current loop
    # entry from ``targets_data_folder``.
    def _opt_single_path(raw: Any) -> Optional[Path]:
        if raw in (None, "", "null"):
            return None
        if isinstance(raw, (list, tuple)):
            if not raw:
                return None
            if len(raw) > 1:
                raise ValueError(
                    f"io.targets_data_folder_mean / "
                    f"io.targets_data_folder_individual must be a single "
                    f"path, not a list ({raw!r})."
                )
            raw = raw[0]
        return Path(raw)

    targets_data_folder_mean = _opt_single_path(
        args.targets_data_folder_mean
        if args.targets_data_folder_mean is not None
        else io_cfg.get("targets_data_folder_mean")
    )
    targets_data_folder_individual = _opt_single_path(
        args.targets_data_folder_individual
        if args.targets_data_folder_individual is not None
        else io_cfg.get("targets_data_folder_individual")
    )

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else Path(io_cfg.get("output_dir") or "results/plots/<model_alias>/microscopic")
    )

    formats = (
        args.formats
        if args.formats is not None
        else io_cfg.get("formats") or ["png"]
    )

    plot_kinds = (
        args.plot_kinds
        if args.plot_kinds is not None
        else cfg.get("plot_kinds") or ["manifold", "trajectory", "projection"]
    )

    # --- trajectory_axis axis sets ---
    trajectory_axis_axis_sets = _parse_trajectory_axis_axis_sets(
        taxis_cfg.get("axes") or []
    )

    # --- targets: cross product of (GT × group × idx) ---
    targets_cfg: Mapping[str, Any] = cfg.get("targets") or {}
    target_gt_list = list(targets_cfg.get("GT") or ["GT_0", "GT_1"])
    target_group_list = list(
        targets_cfg.get("group") or ["control", "experiment", "experiment_fail"]
    )
    target_idx_list = [int(i) for i in (targets_cfg.get("idx") or [0])]
    target_specs = _build_targets_specs_from_triple(
        target_gt_list, target_group_list, target_idx_list
    )
    gt_filename_suffix = _gt_filename_suffix(target_gt_list)

    xlim_raw = traj_cfg.get("xlim")
    ylim_raw = traj_cfg.get("ylim")
    rlim_raw = tpolar_cfg.get("rlim")

    run_microscopic(
        task_vector_data_folder=task_vector_data_folder,
        targets_data_folders=targets_data_folders,
        targets_data_folder_mean=targets_data_folder_mean,
        targets_data_folder_individual=targets_data_folder_individual,
        token_positions_task_vector=tp_cfg.get("task_vector") or ["-1"],
        token_positions_targets=tp_cfg.get("targets") or [],
        best_n_token_position_targets=int(tp_cfg.get("best_n_targets", 1)),
        targets=target_specs,
        output_dir=output_dir,
        config_override=(
            Path(cfg["config_override"])
            if cfg.get("config_override")
            else None
        ),
        rotation_metric=cfg.get("rotation_metric"),
        band_kind=cfg.get("band_kind", "std"),
        gt_style=cfg.get("gt_style", "line_style"),
        formats=formats,
        trajectory_residualPC_variants=traj_cfg.get(
            "residualPC_variants"
        ) or ["raw", "unit_x", "unit_xy"],
        trajectory_realignment_variants=traj_cfg.get(
            "realignment_variants"
        ) or ["raw_projection", "cosine_similarity"],
        trajectory_realignment_footnote=bool(
            traj_cfg.get("realignment_footnote", False)
        ),
        trajectory_backgrounds=traj_cfg.get("background") or ["success_only", "all"],
        trajectory_set_layers=traj_cfg.get("set_layer") or [],
        trajectory_task_vector_arrow=bool(traj_cfg.get("task_vector_arrow", False)),
        trajectory_start_layer=int(traj_cfg.get("start_layer", -1)),
        trajectory_end_layer=int(traj_cfg.get("end_layer", -1)),
        trajectory_xlim=tuple(xlim_raw) if xlim_raw is not None else None,
        trajectory_ylim=tuple(ylim_raw) if ylim_raw is not None else None,
        trajectory_axis_polar_rlim=float(rlim_raw) if rlim_raw is not None else None,
        bg_grey_scale=bool(traj_cfg.get("bg_grey_scale", False)),
        projection_start_layer=int(proj_cfg.get("start_layer", -1)),
        projection_end_layer=int(proj_cfg.get("end_layer", -1)),
        plot_kinds=plot_kinds,
        manifold_subspaces=man_cfg.get("subspace") or ["set_layer"],
        manifold_methods=man_cfg.get("method") or ["PCA"],
        manifold_groups=man_cfg.get("group") or ["without_experiment_fail"],
        manifold_n_components=man_cfg.get("n_components") or [2, 3],
        n_sample_manifold=int(man_cfg.get("n_sample", 0)),
        manifold_across_start_layer=int(man_cfg.get("across_start_layer", -1)),
        manifold_across_end_layer=int(man_cfg.get("across_end_layer", -1)),
        manifold_PCA_set_layers=man_cfg.get("PCA_set_layer") or [],
        manifold_set_start_layer=int(man_cfg.get("set_start_layer", -1)),
        manifold_set_end_layer=int(man_cfg.get("set_end_layer", -1)),
        include_rotation_cos_sim_curves=bool(
            proj_cfg.get("include_rotation_cos_sim_curves", False)
        ),
        rotation_cos_sim_line_color=proj_cfg.get("rotation_cos_sim_line_color", "black"),
        rotation_cos_sim_band_color=proj_cfg.get("rotation_cos_sim_band_color", "grey"),
        trajectory_axis_axis_sets=trajectory_axis_axis_sets,
        trajectory_axis_start_layer=(
            int(args.trajectory_axis_start_layer)
            if int(args.trajectory_axis_start_layer) != -1
            else int(taxis_cfg.get("start_layer", -1))
        ),
        trajectory_axis_end_layer=(
            int(args.trajectory_axis_end_layer)
            if int(args.trajectory_axis_end_layer) != -1
            else int(taxis_cfg.get("end_layer", -1))
        ),
        trajectory_axis_start_layer_mean=_optional_int(
            taxis_cfg.get("start_layer_mean")
        ),
        trajectory_axis_end_layer_mean=_optional_int(
            taxis_cfg.get("end_layer_mean")
        ),
        trajectory_axis_start_layer_individual=_optional_int(
            taxis_cfg.get("start_layer_individual")
        ),
        trajectory_axis_end_layer_individual=_optional_int(
            taxis_cfg.get("end_layer_individual")
        ),
        trajectory_axis_max_target_per_class=int(
            taxis_cfg.get("max_target_per_class", 50)
        ),
        trajectory_axis_line_width_individual=float(
            taxis_cfg.get("line_width_individual", 0.8)
        ),
        trajectory_axis_line_width_mean=float(
            taxis_cfg.get("line_width_mean", 2.5)
        ),
        trajectory_axis_individual_line_alpha=float(
            taxis_cfg.get("individual_line_alpha", 0.18)
        ),
        trajectory_axis_scene_show_walls=bool(
            taxis_cfg.get("scene_show_walls", True)
        ),
        trajectory_axis_scene_wall_color=(
            None
            if taxis_cfg.get("scene_wall_color") in (None, "", "null")
            else str(taxis_cfg.get("scene_wall_color"))
        ),
        trajectory_axis_marker_layer_gradient=bool(
            taxis_cfg.get("marker_layer_gradient", True)
        ),
        trajectory_axis_marker_gradient_start=float(
            taxis_cfg.get("marker_gradient_start", 0.5)
        ),
        trajectory_axis_marker_size_target=_resolve_optional_float(
            args.trajectory_axis_marker_size_target,
            taxis_cfg.get("marker_size_target"),
        ),
        trajectory_axis_marker_size_mean=_resolve_optional_float(
            args.trajectory_axis_marker_size_mean,
            taxis_cfg.get("marker_size_mean"),
        ),
        trajectory_axis_sample_seed=int(taxis_cfg.get("sample_seed", 0)),
        trajectory_axis_show_arrows=bool(taxis_cfg.get("show_arrows", False)),
        trajectory_axis_library=str(taxis_cfg.get("library", "plotly")),
        trajectory_axis_axis_ranges=_parse_axis_ranges(taxis_cfg.get("axis_ranges")),
        gt_filename_suffix=gt_filename_suffix,
    )


if __name__ == "__main__":
    main()
