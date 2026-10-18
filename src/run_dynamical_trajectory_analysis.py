"""
Train and evaluate kinematic-descriptor predictors of the microscopic class
(honest_success / lie_success / lie_fail) from cached layer trajectories.

Run with the project's standard layered-YAML pattern:

.. code-block:: bash

    python -m src.run_dynamical_trajectory_analysis \\
        --config config/dynamical_trajectory.yaml \\
        --data_folder results/runs/<model>/<task_dataset>/SysPrompt000/0_pre10_gen10

The script consumes :mod:`config/dynamical_trajectory.yaml` directly (no
merge into :class:`ExperimentConfig`). It is self-contained: it reads the
six microscopic-class activation pickles from one cache-stage seed folder,
computes the configured descriptors, then for every entry in
``class_subsets`` (e.g. ``{all: [honest_success, lie_success, lie_fail],
lie_only: [lie_success, lie_fail]}``) trains a multinomial logistic
regression on the subset's rows for every individual descriptor and every
combined predictor. Supervised direction-fit descriptors keep using the
full population to fit their DiM vectors — only the predictor's training
rows are restricted.

Outputs (one folder per class subset, joined by ``+``):

::

    <results_dir>/<run_alias>/<position>/<classes-joined-by-+>/metrics.json
    <results_dir>/<run_alias>/<position>/<classes-joined-by-+>/pipeline.pkl
    <plots_dir>/<run_alias>/<position>/<classes-joined-by-+>/
        mean_descriptor_curves.png
        pca_<predictor>_<N>D.png
        coef_heatmap_<predictor>.png
        cv_accuracy.png
        confusion_matrices.png
    <results_dir>/<run_alias>/summary.json

``<position>`` is ``<token_position>`` when the full layer trajectory is used
(``data.layer_start`` and ``data.layer_end`` both ``-1``), and
``<token_position>_L<lo>-L<hi>`` when a layer range is set, where ``<lo>`` /
``<hi>`` are the *resolved* absolute layer indices (a ``-1`` end resolves to
the last layer). The range is configurable via ``data.layer_start`` /
``data.layer_end`` in the YAML or the ``--layer_start`` / ``--layer_end`` CLI
flags.

When ``test.data_folders`` is non-empty (or ``--data_folder_test`` is
passed), the saved ``pipeline.pkl`` is applied to each test cache folder
and one subfolder per test is written under:

::

    <results_dir>/<run_alias>/<position>/<subset>/test/<test_alias>/<test_token_position>/metrics.json
    <plots_dir>/<run_alias>/<position>/<subset>/test/<test_alias>/<test_token_position>/
        descriptor_curves.png   (if test.plots.descriptor_curves)
        pca_<predictor>_<N>D.png (if test.plots.pca)
        confusion_matrices.png  (if test.plots.confusion_matrix)
        accuracy.png            (if test.plots.accuracy)

CLI flags layer over the YAML; the YAML stays the canonical source.
``--class_subsets <name>...`` picks a subset of the YAML-defined subsets
to run. ``--data_folder_test <path>...`` overrides ``test.data_folders``.
``--layer_start`` / ``--layer_end`` override ``data.layer_start`` /
``data.layer_end`` (the analyzed layer range).
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

from src.analysis.dynamical_trajectory import (
    CLASS_NAMES,
    CVResult,
    ClassTrajectories,
    DESCRIPTOR_REGISTRY,
    SUPERVISED_DESCRIPTOR_REGISTRY,
    FeatureMap,
    PerPredictorPipeline,
    PredictorMatrix,
    TrainedPipelineBundle,
    all_known_descriptors,
    build_predictor_matrix,
    compute_dynamical_direction_cosines,
    compute_feature_map,
    evaluate_pipeline_on_test,
    load_class_trajectories,
    plot_confusion_matrices,
    plot_coefficient_heatmap,
    plot_cv_accuracy,
    plot_dynamical_direction_cosines,
    plot_mean_descriptor_curves,
    plot_pca_scatter,
    plot_test_accuracy,
    plot_test_accuracy_by_mode,
    plot_test_confusion_matrices,
    restrict_class_trajectories,
    run_cv,
    save_pipeline_bundle,
    _resolve_top_token_position,
)
from src.paths import PROJECT_ROOT


_SEED_RE = re.compile(r"^(\d+)_pre(\d+)_gen(\d+)$")
_SYS_PROMPT_RE = re.compile(r"^SysPrompt(\d+)$")


# --------------------------------------------------------------------------- #
# Config loading                                                              #
# --------------------------------------------------------------------------- #


def _load_config(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Top-level YAML in {path} must be a mapping; got {type(raw).__name__}.")
    return raw


def _resolve_path(p: str) -> Path:
    """Resolve relative paths against ``PROJECT_ROOT``."""
    path = Path(p).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _run_alias(data_folder: Path) -> str:
    """
    Derive a filesystem-safe handle from the cache folder layout
    ``<root>/<model>/<task_dataset>/SysPrompt<NNN>/<seed>_pre<N>_gen<M>``.
    Returns ``"<model>__<task_dataset>__SysPrompt<NNN>__seed<seed>"``; falls
    back to the trailing two segments joined with double-underscores when
    the layout doesn't match.
    """
    parts = data_folder.resolve().parts
    if len(parts) >= 4:
        model = parts[-4]
        task_dataset = parts[-3]
        sys_prompt = parts[-2]
        seed_seg = parts[-1]
        seed_match = _SEED_RE.match(seed_seg)
        sys_match = _SYS_PROMPT_RE.match(sys_prompt)
        if seed_match and sys_match:
            return f"{model}__{task_dataset}__SysPrompt{sys_match.group(1)}__seed{seed_match.group(1)}"
    return "__".join(parts[-2:])


# --------------------------------------------------------------------------- #
# Per-position run                                                            #
# --------------------------------------------------------------------------- #


def _descriptor_specs(individual_cfg: Mapping[str, Any]) -> List[Tuple[str, str, Mapping[str, Any]]]:
    """
    Translate the YAML ``descriptors.individual`` block into the tuple list
    consumed by :func:`compute_feature_map`. Each spec resolves the function
    name (or defaults to the key) and pulls optional ``kwargs``. The
    function must be registered in either
    :data:`DESCRIPTOR_REGISTRY` (unsupervised) or
    :data:`SUPERVISED_DESCRIPTOR_REGISTRY` (DiM projection).
    """
    specs: List[Tuple[str, str, Mapping[str, Any]]] = []
    for spec_name, body in individual_cfg.items():
        body = body or {}
        fn_name = str(body.get("function", spec_name))
        kwargs = body.get("kwargs") or {}
        if fn_name not in DESCRIPTOR_REGISTRY and fn_name not in SUPERVISED_DESCRIPTOR_REGISTRY:
            raise KeyError(
                f"Descriptor function {fn_name!r} (for spec {spec_name!r}) is not "
                f"a known descriptor. Known: {all_known_descriptors()}"
            )
        specs.append((spec_name, fn_name, kwargs))
    return specs


def _predictor_specs(individual_cfg: Mapping[str, Any],
                     combined_cfg: Mapping[str, Any]) -> List[Tuple[str, List[str]]]:
    """
    Ordered list of ``(predictor_name, [descriptor_names])`` covering every
    individual descriptor (1-element subset) followed by every combined
    predictor (multi-element subset).
    """
    out: List[Tuple[str, List[str]]] = []
    for name in individual_cfg.keys():
        out.append((str(name), [str(name)]))
    for combined_name, body in (combined_cfg or {}).items():
        body = body or {}
        descriptors = list(body.get("descriptors") or [])
        if not descriptors:
            raise ValueError(
                f"Combined predictor {combined_name!r} has no descriptors listed."
            )
        for d in descriptors:
            if d not in individual_cfg:
                raise KeyError(
                    f"Combined predictor {combined_name!r} references "
                    f"descriptor {d!r} which is not in descriptors.individual."
                )
        out.append((str(combined_name), [str(d) for d in descriptors]))
    return out


def _subset_folder_name(subset_classes: Sequence[str]) -> str:
    """Folder slug for a class subset, e.g. ``honest_success+lie_success``."""
    return "+".join(subset_classes)


def _layer_folder_suffix(
    layer_start: int, layer_end: int, layer_indices: np.ndarray
) -> str:
    """Folder suffix encoding the layer slice.

    Both bounds at ``-1`` (the default "all layers" intent) -> ``""`` so the
    folder is just the token position. Otherwise -> ``"_L{lo}-L{hi}"`` using
    the *resolved* absolute layer indices (so a ``-1`` end resolves to the
    last layer's index rather than appearing literally in the path).
    """
    if layer_start < 0 and layer_end < 0:
        return ""
    lo = int(layer_indices[0])
    hi = int(layer_indices[-1])
    return f"_L{lo}-L{hi}"


def _test_plot_toggles(cfg: Mapping[str, Any]) -> Dict[str, bool]:
    """
    Read ``test.plots`` from the YAML config and return a dict with explicit
    booleans for every supported toggle. Keys: ``descriptor_curves``,
    ``pca``, ``confusion_matrix``, ``accuracy``, ``direction_cosines``.
    Unset entries default to ``True`` when the test block exists at all,
    so users only opt out.
    """
    test_cfg = cfg.get("test", {}) or {}
    raw = test_cfg.get("plots", {}) or {}
    keys = (
        "descriptor_curves", "pca", "confusion_matrix", "accuracy",
        "direction_cosines",
    )
    return {k: bool(raw.get(k, True)) for k in keys}


def _parse_test_modes(
    cfg: Mapping[str, Any],
    key: str,
    *,
    valid: Sequence[str] = ("frozen", "refit_on_test"),
    default: Sequence[str] = ("frozen", "refit_on_test"),
) -> List[str]:
    """Parse ``test.<key>`` as a deduped list of mode strings."""
    test_cfg = cfg.get("test", {}) or {}
    raw = test_cfg.get(key, list(default))
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        raise ValueError(
            f"test.{key} must be a list of mode strings; got {type(raw).__name__}."
        )
    out: List[str] = []
    for m in raw:
        m = str(m)
        if m not in valid:
            raise ValueError(
                f"test.{key}: {m!r} not in valid set {list(valid)}."
            )
        if m not in out:
            out.append(m)
    if not out:
        raise ValueError(f"test.{key} produced empty mode list.")
    return out


def _mode_combos_dyn(
    bundle: TrainedPipelineBundle,
    std_modes: Sequence[str],
    dir_modes: Sequence[str],
) -> List[Tuple[str, str]]:
    """
    Expand ``std_modes × dir_modes``. When the bundle has no supervised
    directions, ``direction_mode`` is collapsed to ``"frozen"`` (no-op).
    """
    has_supervised = bool(bundle.supervised_directions)
    combos: List[Tuple[str, str]] = []
    if has_supervised:
        for s in std_modes:
            for d in dir_modes:
                combos.append((s, d))
    else:
        for s in std_modes:
            combo = (s, "frozen")
            if combo not in combos:
                combos.append(combo)
    return combos


def _mode_slug_dyn(std_mode: str, dir_mode: str, *, has_supervised: bool) -> str:
    std_short = "frozen" if std_mode == "frozen" else "refit"
    dir_short = "frozen" if dir_mode == "frozen" else "refit"
    if has_supervised:
        return f"std-{std_short}__dir-{dir_short}"
    return f"std-{std_short}"


def _build_pipeline_bundle(
    *,
    cv_results: Mapping[str, "CVResult"],
    feat: "FeatureMap",
    descriptor_specs: Sequence[Tuple[str, str, Mapping[str, Any]]],
    individual_descriptor_names: Sequence[str],
    predictors: Sequence[Tuple[str, List[str]]],
    training_token_position: str,
    training_data_folder: Path,
    smoothing_enabled: bool,
    smoothing_sigma: float,
) -> TrainedPipelineBundle:
    """
    Pack the per-predictor full-data refits + their column metadata into a
    single :class:`TrainedPipelineBundle` ready to pickle.

    Only the descriptors that any predictor actually consumes are kept in
    the bundle's ``descriptor_specs`` / ``supervised_directions`` /
    ``valid_mask`` — test-time evaluation should not have to compute
    descriptors that the saved predictors never reference.
    """
    used_descriptors: List[str] = []
    seen: set = set()
    for _, subset in predictors:
        for d in subset:
            if d not in seen:
                seen.add(d)
                used_descriptors.append(d)
    spec_lookup = {name: (fn, kwargs) for name, fn, kwargs in descriptor_specs}
    pruned_specs: List[Tuple[str, str, Mapping[str, Any]]] = []
    for d in used_descriptors:
        if d not in spec_lookup:
            raise KeyError(
                f"_build_pipeline_bundle: predictor referenced descriptor "
                f"{d!r} that is not in descriptor_specs."
            )
        fn, kwargs = spec_lookup[d]
        pruned_specs.append((d, fn, dict(kwargs)))

    supervised_directions = {
        name: np.asarray(arr, dtype=np.float32)
        for name, arr in feat.supervised_directions.items()
        if name in used_descriptors
    }
    valid_mask = {
        name: np.asarray(feat.valid_mask[name], dtype=bool)
        for name in used_descriptors
        if name in feat.valid_mask
    }

    per_predictor: Dict[str, PerPredictorPipeline] = {}
    for predictor_name, descriptor_subset in predictors:
        cv = cv_results[predictor_name]
        per_predictor[predictor_name] = PerPredictorPipeline(
            predictor_name=predictor_name,
            descriptor_subset=list(descriptor_subset),
            scaler=cv.full_scaler,
            model=cv.full_model,
            column_names=list(cv.column_names),
            column_descriptors=list(cv.column_descriptors),
            column_layers=list(cv.column_layers),
            class_names=list(cv.class_names),
        )

    return TrainedPipelineBundle(
        descriptor_specs=pruned_specs,
        training_token_position=training_token_position,
        training_layer_indices=np.asarray(feat.layer_indices, dtype=np.int64).copy(),
        training_data_folder=str(training_data_folder),
        supervised_directions=supervised_directions,
        valid_mask=valid_mask,
        smoothing_enabled=bool(smoothing_enabled),
        smoothing_sigma=float(smoothing_sigma),
        predictors=per_predictor,
    )


def _resolve_test_folders(
    cfg: Mapping[str, Any],
    cli_override: Optional[Sequence[Path]],
) -> List[Path]:
    """
    Resolve ``test.data_folders`` from the YAML, overridden by
    ``--data_folder_test`` when provided. Returns the resolved absolute
    paths (empty list = no test evaluation requested).
    """
    if cli_override is not None:
        raw_list: Sequence[Any] = list(cli_override)
    else:
        test_cfg = cfg.get("test", {}) or {}
        raw_list = list(test_cfg.get("data_folders", []) or [])
    resolved: List[Path] = []
    for entry in raw_list:
        p = _resolve_path(str(entry))
        if not (p / "activations").exists():
            raise FileNotFoundError(
                f"test data folder {p} does not contain an 'activations/' subdir."
            )
        resolved.append(p)
    return resolved


def _run_tests(
    *,
    bundle: TrainedPipelineBundle,
    test_folders: Sequence[Path],
    cfg: Mapping[str, Any],
    available_classes: Sequence[str],
    results_dir: Path,
    plots_dir: Path,
    formats: Sequence[str],
    class_colors: Mapping[str, str],
    pca_dims: Sequence[int],
    mean_curve_band: str,
    plot_toggles: Mapping[str, bool],
    data_seed: int,
) -> Dict[str, Any]:
    """
    For each entry in ``test_folders``: resolve the test token position
    (per ``test.token_position_policy``), load test trajectories restricted
    to the training subset, evaluate every predictor in ``bundle`` for
    every requested ``(standardization_mode, direction_mode)`` combination,
    and emit the toggled plots into a per-test subfolder.

    Each mode combination writes to its own subfolder (e.g.
    ``std-frozen__dir-frozen/``). Specs without supervised descriptors
    collapse the direction-mode axis (``direction_mode=refit_on_test``
    would be a no-op).

    Returns a metrics dict keyed by test alias.
    """
    if not test_folders:
        return {}
    data_cfg = cfg.get("data", {}) or {}
    test_cfg = cfg.get("test", {}) or {}
    rotation_metric = str(
        test_cfg.get("rotation_metric")
        or data_cfg.get("rotation_metric", "largest_drop_any_early_to_late")
    )
    token_position_policy = str(test_cfg.get("token_position_policy", "test_top1"))
    explicit_pos = test_cfg.get("token_position_explicit")
    std_modes = _parse_test_modes(cfg, "standardization_modes")
    dir_modes = _parse_test_modes(cfg, "direction_modes")
    report_direction_cosine = bool(test_cfg.get("report_direction_cosine", True))

    has_supervised = bool(bundle.supervised_directions)
    combos = _mode_combos_dyn(bundle, std_modes, dir_modes)

    out: Dict[str, Any] = {}
    for test_folder in test_folders:
        test_alias = _run_alias(test_folder)
        if token_position_policy == "match_train":
            test_top: Optional[str] = bundle.training_token_position
        elif token_position_policy == "explicit":
            if not explicit_pos:
                raise ValueError(
                    "test.token_position_policy='explicit' requires "
                    "test.token_position_explicit to be set."
                )
            test_top = str(explicit_pos)
        else:
            test_top = _resolve_top_token_position(test_folder, rotation_metric)
            if test_top is None:
                raise FileNotFoundError(
                    f"Cannot resolve top-1 token position for test folder "
                    f"{test_folder} (looked for "
                    f"ranked_token_pos_{rotation_metric} in "
                    f"{test_folder}/stage_stats/rotation_stats.json)."
                )
            if test_top != bundle.training_token_position:
                print(
                    f"  [test {test_alias!r}] WARNING: test top-1 token "
                    f"position={test_top!r} differs from training "
                    f"{bundle.training_token_position!r}. This is a transfer "
                    "experiment, not a held-out test in the usual sense."
                )

        sub_results = results_dir / "test" / test_alias / test_top
        sub_plots = plots_dir / "test" / test_alias / test_top
        sub_results.mkdir(parents=True, exist_ok=True)
        sub_plots.mkdir(parents=True, exist_ok=True)
        print(f"\n  [test {test_alias!r}] folder={test_folder} "
              f"top_pos={test_top!r} -> {sub_plots}")

        rng = np.random.default_rng(int(data_cfg.get("data_seed", data_seed)))
        full_test = load_class_trajectories(
            test_folder,
            token_position=test_top,
            layer_start=int(data_cfg.get("layer_start", -1)),
            layer_end=int(data_cfg.get("layer_end", -1)),
            n_samples_per_class=int(data_cfg.get("n_samples_per_class", -1)),
            rng=rng,
        )
        # Restrict to the classes the model was trained on. Predictors all
        # share the same training class set in this runner, so any
        # predictor's class_names works.
        train_classes = next(iter(bundle.predictors.values())).class_names
        available_test = [c for c in train_classes if c in full_test.trajectories]
        if not available_test:
            raise ValueError(
                f"Test folder {test_folder} has no rows from the training "
                f"class set {train_classes} (available: "
                f"{list(full_test.trajectories)})."
            )
        if available_test != list(train_classes):
            dropped = [c for c in train_classes if c not in available_test]
            print(f"    note: test missing class(es) {dropped}; "
                  "scoring on intersection only.")
        if list(full_test.trajectories.keys()) == available_test:
            test_cls_traj = full_test
        else:
            test_cls_traj, _ = restrict_class_trajectories(full_test, available_test)

        results_per_mode: Dict[str, Dict[str, Any]] = {}
        accuracy_table: Dict[str, Dict[str, float]] = {}
        for std_mode, dir_mode in combos:
            slug = _mode_slug_dyn(std_mode, dir_mode, has_supervised=has_supervised)
            test_results, test_feat = evaluate_pipeline_on_test(
                bundle, test_cls_traj,
                standardization_mode=std_mode,
                direction_mode=dir_mode,
                direction_fit_data=full_test,
            )
            results_per_mode[slug] = test_results
            mode_results_dir = sub_results / slug
            mode_plots_dir = sub_plots / slug
            mode_results_dir.mkdir(parents=True, exist_ok=True)
            mode_plots_dir.mkdir(parents=True, exist_ok=True)

            wrote: List[Path] = []
            if plot_toggles.get("descriptor_curves", True):
                wrote.extend(plot_mean_descriptor_curves(
                    test_feat,
                    basepath=mode_plots_dir / "descriptor_curves",
                    band=mean_curve_band,
                    class_colors=class_colors,
                    formats=formats,
                    class_names=available_test,
                ))
            if plot_toggles.get("pca", True):
                for predictor_name, tr in test_results.items():
                    wrote.extend(plot_pca_scatter(
                        tr.X_test_scaled, tr.y_true,
                        title=f"{predictor_name} (test, {slug})",
                        basepath=mode_plots_dir / f"pca_{predictor_name}",
                        dims=list(pca_dims),
                        class_colors=class_colors,
                        scaler=None,
                        formats=formats,
                        class_names=available_test,
                    ))
            if plot_toggles.get("confusion_matrix", True):
                wrote.extend(plot_test_confusion_matrices(
                    test_results,
                    basepath=mode_plots_dir / "confusion_matrices",
                    formats=formats,
                ))
            if plot_toggles.get("accuracy", True):
                wrote.extend(plot_test_accuracy(
                    test_results,
                    basepath=mode_plots_dir / "accuracy",
                    formats=formats,
                ))
            for p in wrote:
                print(f"    wrote {p}")

            mode_metrics = {
                "standardization_mode": std_mode,
                "direction_mode": dir_mode,
                "test_folder": str(test_folder),
                "test_alias": test_alias,
                "test_token_position": test_top,
                "training_token_position": bundle.training_token_position,
                "rotation_metric": rotation_metric,
                "classes_evaluated": available_test,
                "predictors": {
                    name: {
                        "test_accuracy": tr.test_accuracy,
                        "confusion_matrix": tr.confusion_matrix.tolist(),
                        "confusion_matrix_classes": tr.class_names,
                        "classes_present": tr.classes_present,
                        "n_per_class": tr.n_per_class,
                    }
                    for name, tr in test_results.items()
                },
            }
            mode_metrics_path = mode_results_dir / "metrics.json"
            mode_metrics_path.write_text(json.dumps(mode_metrics, indent=2))
            print(f"    [{slug}] wrote {mode_metrics_path}")
            accuracy_table[slug] = {
                name: float(tr.test_accuracy)
                for name, tr in test_results.items()
            }

        # Cross-mode comparison plot (one figure per test cache).
        if plot_toggles.get("accuracy", True) and len(combos) > 1:
            wrote_cmp = plot_test_accuracy_by_mode(
                results_per_mode,
                basepath=sub_plots / "accuracy_by_mode",
                formats=formats,
            )
            for p in wrote_cmp:
                print(f"    wrote {p}")

        # Direction cosine diagnostic (free; supervised-spec only).
        cosines_summary: Optional[Dict[str, List[float]]] = None
        if report_direction_cosine and has_supervised:
            cosines = compute_dynamical_direction_cosines(bundle, full_test)
            if cosines:
                cosines_summary = {n: a.tolist() for n, a in cosines.items()}
                cos_path = sub_results / "direction_cosines.json"
                cos_path.write_text(json.dumps({
                    "layer_indices": [int(l) for l in test_cls_traj.layer_indices],
                    "cosines": cosines_summary,
                }, indent=2))
                print(f"    wrote {cos_path}")
                if plot_toggles.get("direction_cosines", True):
                    cos_plot = plot_dynamical_direction_cosines(
                        cosines,
                        [int(l) for l in test_cls_traj.layer_indices],
                        basepath=sub_plots / "direction_cosines",
                        formats=formats,
                    )
                    for p in cos_plot:
                        print(f"    wrote {p}")

        out[test_alias] = {
            "plots_dir": str(sub_plots),
            "test_token_position": test_top,
            "modes": list(accuracy_table.keys()),
            "accuracy_by_mode": accuracy_table,
            "direction_cosines": cosines_summary,
        }
    return out


def _resolve_class_subsets(
    cfg: Mapping[str, Any],
    cli_override: Optional[Sequence[str]],
) -> List[Tuple[str, List[str]]]:
    """
    Read ``class_subsets`` from the YAML (top-level mapping of
    ``subset_name -> [class_names]``) and intersect with ``cli_override``
    (a list of subset *names*) when provided. Returns the ordered list of
    ``(subset_name, [class_names])`` to run.

    When the YAML block is empty / missing and no CLI override is given,
    defaults to a single subset called ``all`` covering every entry in
    :data:`CLASS_NAMES`. The output is always non-empty.
    """
    raw = cfg.get("class_subsets") or {}
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"class_subsets must be a mapping (subset_name -> [class_names]); "
            f"got {type(raw).__name__}."
        )

    subsets: List[Tuple[str, List[str]]] = []
    for name, body in raw.items():
        classes = list(body) if body is not None else []
        unknown = [c for c in classes if c not in CLASS_NAMES]
        if unknown:
            raise ValueError(
                f"class_subsets.{name}: unknown class(es) {unknown}; "
                f"valid classes are {list(CLASS_NAMES)}."
            )
        if not classes:
            raise ValueError(f"class_subsets.{name}: at least one class required.")
        # Preserve CLASS_NAMES ordering so folder names are deterministic
        # regardless of how the user listed them in YAML.
        ordered = [c for c in CLASS_NAMES if c in classes]
        subsets.append((str(name), ordered))

    if cli_override:
        wanted = set(cli_override)
        names = {n for n, _ in subsets}
        missing = wanted - names
        if missing:
            raise ValueError(
                f"--class_subsets references unknown subset name(s) {sorted(missing)}; "
                f"YAML defines {sorted(names)}."
            )
        subsets = [(n, c) for n, c in subsets if n in wanted]
        if not subsets:
            raise ValueError("--class_subsets selection produced an empty list.")

    if not subsets:
        subsets = [("all", list(CLASS_NAMES))]
    return subsets


def _run_for_subset(
    *,
    cfg: Mapping[str, Any],
    full_cls_traj: ClassTrajectories,
    subset_classes: Sequence[str],
    subset_name: str,
    token_position: str,
    data_folder: Path,
    test_folders: Sequence[Path],
    results_dir: Path,
    plots_dir: Path,
) -> Dict[str, Any]:
    """
    Run the descriptor / CV pipeline on one (position, class-subset) pair.

    ``full_cls_traj`` is the all-classes load used as the supervised
    direction-fitting pool (so e.g. ``proj_2_lie-honest_direction`` keeps
    seeing ``honest_success`` even when it isn't in the prediction subset).
    The predictor's X / y come only from rows in ``subset_classes``.
    """
    plot_cfg = cfg.get("plots", {}) or {}
    descriptors_cfg = cfg.get("descriptors", {}) or {}
    individual_cfg = descriptors_cfg.get("individual", {}) or {}
    combined_cfg = cfg.get("combined_predictors", {}) or {}
    preprocessing_cfg = cfg.get("preprocessing", {}) or {}
    smoothing_cfg = preprocessing_cfg.get("smoothing", {}) or {}
    standardization_cfg = preprocessing_cfg.get("standardization", {}) or {}
    predictor_cfg = cfg.get("predictor", {}) or {}
    cv_cfg = cfg.get("cv", {}) or {}
    io_cfg = cfg.get("io", {}) or {}
    formats = list(io_cfg.get("plot_formats", ["png"]))
    class_colors = plot_cfg.get("class_colors", {}) or {}

    print(f"\n  [subset {subset_name!r}] classes={list(subset_classes)}")

    available = [c for c in subset_classes if c in full_cls_traj.trajectories]
    if not available:
        raise ValueError(
            f"Subset {subset_name!r} requests {list(subset_classes)} but none "
            f"of those classes were loaded from the cache (available: "
            f"{list(full_cls_traj.trajectories)})."
        )
    if available != list(subset_classes):
        dropped = [c for c in subset_classes if c not in available]
        print(f"    note: dropping {dropped} (not present in cache).")

    if list(available) == list(full_cls_traj.trajectories):
        # Subset covers everything that was loaded — short-circuit the
        # restriction so we don't pay for a redundant copy.
        cls_traj = full_cls_traj
        subset_to_full_idx = None
        sup_fit_data = None
    else:
        cls_traj, subset_to_full_idx = restrict_class_trajectories(
            full_cls_traj, available
        )
        sup_fit_data = full_cls_traj

    print(f"    rows: {{ {', '.join(f'{c}: {a.shape[0]}' for c, a in cls_traj.trajectories.items())} }}, "
          f"total={cls_traj.feature_data.shape[0]}, "
          f"fit_pool={'full' if sup_fit_data is full_cls_traj else 'subset'}")

    specs = _descriptor_specs(individual_cfg)
    feat = compute_feature_map(
        cls_traj,
        descriptor_specs=specs,
        smoothing_enabled=bool(smoothing_cfg.get("enabled", False)),
        smoothing_sigma=float(smoothing_cfg.get("sigma", 1.0)),
        supervised_fit_data=sup_fit_data,
    )

    predictors = _predictor_specs(individual_cfg, combined_cfg)

    plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: mean descriptor curves (only the predicted classes).
    written = plot_mean_descriptor_curves(
        feat,
        basepath=plots_dir / "mean_descriptor_curves",
        band=str(plot_cfg.get("mean_curve_band", "ci95")),
        class_colors=class_colors,
        formats=formats,
        class_names=available,
    )
    for p in written:
        print(f"    wrote {p}")

    cv_results: Dict[str, CVResult] = {}
    for predictor_name, descriptor_subset in predictors:
        # ``pm_viz`` uses the in-sample supervised fit (from feat); it's only
        # for visualizing the predictor in PCA space. The CV accuracy below
        # rebuilds the predictor matrix per fold with per-fold DiM refits.
        pm_viz = build_predictor_matrix(feat, descriptor_subset)
        dropped = sum(
            int(np.sum(~feat.valid_mask.get(d, np.ones(len(cls_traj.layer_indices), dtype=bool))))
            for d in descriptor_subset
        )
        note = f" (dropped {dropped} invalid layer-columns)" if dropped else ""
        has_supervised = any(d in SUPERVISED_DESCRIPTOR_REGISTRY for d in descriptor_subset)
        sup_note = "  [per-fold DiM refit]" if has_supervised else ""
        print(f"    predictor={predictor_name!r}  X.shape={pm_viz.X.shape}  "
              f"descriptors={descriptor_subset}{note}{sup_note}")
        cv = run_cv(
            cls_traj=cls_traj,
            feat=feat,
            descriptor_subset=descriptor_subset,
            predictor_cfg=predictor_cfg,
            cv_cfg=cv_cfg,
            scaler_cfg=standardization_cfg,
            smoothing_enabled=bool(smoothing_cfg.get("enabled", False)),
            smoothing_sigma=float(smoothing_cfg.get("sigma", 1.0)),
            supervised_fit_data=sup_fit_data,
            subset_to_full_idx=subset_to_full_idx,
        )
        cv_results[predictor_name] = cv

        pca_paths = plot_pca_scatter(
            pm_viz.X, cls_traj.labels,
            title=f"{predictor_name}",
            basepath=plots_dir / f"pca_{predictor_name}",
            dims=list(plot_cfg.get("pca_dims", [2])),
            class_colors=class_colors,
            scaler=cv.full_scaler,
            formats=formats,
            class_names=available,
        )
        for p in pca_paths:
            print(f"    wrote {p}")

        heat_paths = plot_coefficient_heatmap(
            cv,
            descriptor_subset=descriptor_subset,
            layer_indices=cls_traj.layer_indices,
            title=f"Coefficients — {predictor_name}",
            basepath=plots_dir / f"coef_heatmap_{predictor_name}",
            cmap=str(plot_cfg.get("coef_cmap", "RdBu_r")),
            formats=formats,
        )
        for p in heat_paths:
            print(f"    wrote {p}")

    acc_paths = plot_cv_accuracy(
        cv_results,
        basepath=plots_dir / "cv_accuracy",
        formats=formats,
        n_classes=len(available),
    )
    for p in acc_paths:
        print(f"    wrote {p}")

    cm_paths = plot_confusion_matrices(
        cv_results,
        basepath=plots_dir / "confusion_matrices",
        formats=formats,
    )
    for p in cm_paths:
        print(f"    wrote {p}")

    metrics: Dict[str, Any] = {
        "token_position": token_position,
        "subset_name": subset_name,
        "classes": list(available),
        "layer_indices": cls_traj.layer_indices.tolist(),
        "n_per_class": feat.n_per_class,
        "predictors": {},
    }
    for predictor_name, cv in cv_results.items():
        metrics["predictors"][predictor_name] = {
            "fold_accuracy": cv.fold_accuracy.tolist(),
            "mean_accuracy": float(cv.fold_accuracy.mean()),
            "std_accuracy": float(cv.fold_accuracy.std()),
            "confusion_matrix": cv.confusion_matrix.tolist(),
            "confusion_matrix_classes": cv.class_names,
            "classes_present": cv.classes_present,
            "n_features": int(cv.coef_full.shape[1]),
        }
    metrics_path = results_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"    wrote {metrics_path}")

    # ---- Save the full-data refit pipeline + run held-out tests ---- #
    bundle = _build_pipeline_bundle(
        cv_results=cv_results,
        feat=feat,
        descriptor_specs=specs,
        individual_descriptor_names=list(individual_cfg.keys()),
        predictors=predictors,
        training_token_position=token_position,
        training_data_folder=data_folder,
        smoothing_enabled=bool(smoothing_cfg.get("enabled", False)),
        smoothing_sigma=float(smoothing_cfg.get("sigma", 1.0)),
    )
    pipeline_path = save_pipeline_bundle(bundle, results_dir / "pipeline.pkl")
    print(f"    wrote {pipeline_path}")
    metrics["pipeline_path"] = str(pipeline_path)

    test_plot_toggles = _test_plot_toggles(cfg)
    test_metrics = _run_tests(
        bundle=bundle,
        test_folders=test_folders,
        cfg=cfg,
        available_classes=available,
        results_dir=results_dir,
        plots_dir=plots_dir,
        formats=formats,
        class_colors=class_colors,
        pca_dims=list(plot_cfg.get("pca_dims", [2])),
        mean_curve_band=str(plot_cfg.get("mean_curve_band", "ci95")),
        plot_toggles=test_plot_toggles,
        data_seed=int((cfg.get("data") or {}).get("data_seed", 0)),
    )
    if test_metrics:
        metrics["tests"] = test_metrics
        metrics_path.write_text(json.dumps(metrics, indent=2))

    return metrics


def run_for_position(
    *,
    cfg: Mapping[str, Any],
    data_folder: Path,
    token_position: str,
    class_subsets: Sequence[Tuple[str, List[str]]],
    test_folders: Sequence[Path],
    results_dir: Path,
    plots_dir: Path,
) -> Dict[str, Any]:
    """
    Load trajectories once for ``token_position``, then run the descriptor /
    CV pipeline for every ``(subset_name, [class_names])`` in
    ``class_subsets``. Each subset writes into its own ``<class_join>/``
    subfolder under ``results_dir`` / ``plots_dir``.

    When the configured layer range is not the full trajectory (i.e. either
    ``data.layer_start`` or ``data.layer_end`` is set), the leaf
    ``<token_position>`` folder is suffixed with the resolved range
    (``<token_position>_L<lo>-L<hi>``); the suffix propagates to every subset
    and test subfolder. Returns ``{"position_folder", "by_subset"}`` where
    ``position_folder`` is the (possibly suffixed) leaf folder name.
    """
    print(f"\n[dyn-traj] token_position={token_position!r}")
    data_cfg = cfg.get("data", {}) or {}

    layer_start = int(data_cfg.get("layer_start", -1))
    layer_end = int(data_cfg.get("layer_end", -1))
    rng = np.random.default_rng(int(data_cfg.get("data_seed", 0)))
    full_cls_traj = load_class_trajectories(
        data_folder,
        token_position=token_position,
        layer_start=layer_start,
        layer_end=layer_end,
        n_samples_per_class=int(data_cfg.get("n_samples_per_class", -1)),
        rng=rng,
    )

    # Encode the resolved layer slice into the leaf folder name and push it
    # onto the dirs handed down to subsets / test eval.
    suffix = _layer_folder_suffix(layer_start, layer_end, full_cls_traj.layer_indices)
    if suffix:
        results_dir = results_dir.parent / (results_dir.name + suffix)
        plots_dir = plots_dir.parent / (plots_dir.name + suffix)
    position_folder = results_dir.name

    print(f"  classes loaded: {list(full_cls_traj.trajectories)} "
          f"(n_per_class={{ {', '.join(f'{c}: {a.shape[0]}' for c, a in full_cls_traj.trajectories.items())} }}, "
          f"layers={len(full_cls_traj.layer_indices)}"
          f"{f' [L{int(full_cls_traj.layer_indices[0])}-L{int(full_cls_traj.layer_indices[-1])}]' if suffix else ''})")

    by_subset: Dict[str, Any] = {}
    for subset_name, subset_classes in class_subsets:
        subset_slug = _subset_folder_name(
            [c for c in subset_classes if c in full_cls_traj.trajectories]
        ) or _subset_folder_name(subset_classes)
        sub_results = results_dir / subset_slug
        sub_plots = plots_dir / subset_slug
        metrics = _run_for_subset(
            cfg=cfg,
            full_cls_traj=full_cls_traj,
            subset_classes=subset_classes,
            subset_name=subset_name,
            token_position=token_position,
            data_folder=data_folder,
            test_folders=test_folders,
            results_dir=sub_results,
            plots_dir=sub_plots,
        )
        by_subset[subset_name] = {
            "folder": subset_slug,
            "classes": metrics["classes"],
            "metrics_path": str(sub_results / "metrics.json"),
            "plots_dir": str(sub_plots),
            "mean_accuracy_by_predictor": {
                name: m["mean_accuracy"]
                for name, m in metrics["predictors"].items()
            },
        }
    return {"position_folder": position_folder, "by_subset": by_subset}


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "config" / "dynamical_trajectory.yaml",
        help="Path to the dynamical-trajectory YAML config.",
    )
    parser.add_argument(
        "--data_folder", type=Path, default=None,
        help="Cache seed folder (contains activations/). Overrides "
             "data.data_folder in the YAML.",
    )
    parser.add_argument(
        "--token_positions", type=str, nargs="*", default=None,
        help="Token-position labels to analyze (one run per label). "
             "Overrides data.token_positions in the YAML. Empty / unset = use "
             "top-1 from rotation_stats.json under the data folder.",
    )
    parser.add_argument(
        "--layer_start", type=int, default=None,
        help="Inclusive lower layer bound (-1 = first layer). Overrides "
             "data.layer_start in the YAML.",
    )
    parser.add_argument(
        "--layer_end", type=int, default=None,
        help="Inclusive upper layer bound (-1 = last layer). Overrides "
             "data.layer_end in the YAML.",
    )
    parser.add_argument(
        "--class_subsets", type=str, nargs="*", default=None,
        help="Names of class_subsets entries from the YAML to run. Empty = "
             "run every subset defined in class_subsets (or the implicit "
             "'all' subset when the YAML defines none).",
    )
    parser.add_argument(
        "--data_folder_test", type=Path, nargs="*", default=None,
        help="Cache seed folder(s) to evaluate the trained pipeline on. "
             "Each folder becomes a subfolder under "
             "<plots_dir>/<run_alias>/<train_token_position>/test/. "
             "Overrides test.data_folders in the YAML.",
    )
    args = parser.parse_args(argv)

    cfg = _load_config(args.config)
    data_cfg = cfg.setdefault("data", {})
    io_cfg = cfg.setdefault("io", {})

    if args.data_folder is not None:
        data_cfg["data_folder"] = str(args.data_folder)
    if args.token_positions is not None:
        data_cfg["token_positions"] = list(args.token_positions)
    if args.layer_start is not None:
        data_cfg["layer_start"] = int(args.layer_start)
    if args.layer_end is not None:
        data_cfg["layer_end"] = int(args.layer_end)
    class_subsets = _resolve_class_subsets(cfg, args.class_subsets)
    test_folders = _resolve_test_folders(cfg, args.data_folder_test)

    data_folder_raw = data_cfg.get("data_folder")
    if not data_folder_raw:
        parser.error(
            "data.data_folder is required (set in YAML or via --data_folder)."
        )
    data_folder = _resolve_path(data_folder_raw)
    if not (data_folder / "activations").exists():
        parser.error(
            f"--data_folder {data_folder} does not contain an 'activations/' subdir."
        )

    requested_positions = list(data_cfg.get("token_positions") or [])
    if not requested_positions:
        top = _resolve_top_token_position(
            data_folder, str(data_cfg.get("rotation_metric", "largest_drop_any_early_to_late"))
        )
        if top is None:
            parser.error(
                f"Cannot resolve top-1 token position from "
                f"{data_folder}/stage_stats/rotation_stats.json. Pass "
                "--token_positions or set data.token_positions in the YAML."
            )
        requested_positions = [top]
    print(f"[dyn-traj] data_folder={data_folder}")
    print(f"[dyn-traj] token_positions={requested_positions}")
    _ls, _le = int(data_cfg.get("layer_start", -1)), int(data_cfg.get("layer_end", -1))
    print(f"[dyn-traj] layer_range=[{_ls}, {_le}]"
          f"{' (all layers)' if _ls < 0 and _le < 0 else ''}")
    print(f"[dyn-traj] class_subsets="
          f"{[(n, c) for n, c in class_subsets]}")

    results_root = _resolve_path(io_cfg.get("results_dir", "results/dynamical_trajectory"))
    plots_root = _resolve_path(io_cfg.get("plots_dir", "results/plots/dynamical_trajectory"))
    alias = _run_alias(data_folder)

    if test_folders:
        print(f"[dyn-traj] test data_folders={[str(p) for p in test_folders]}")
    summary: Dict[str, Any] = {
        "data_folder": str(data_folder),
        "class_subsets": {n: c for n, c in class_subsets},
        "test_data_folders": [str(p) for p in test_folders],
        "results_by_position": {},
    }
    for position in requested_positions:
        results_dir = results_root / alias / position
        plots_dir = plots_root / alias / position
        res = run_for_position(
            cfg=cfg,
            data_folder=data_folder,
            token_position=position,
            class_subsets=class_subsets,
            test_folders=test_folders,
            results_dir=results_dir,
            plots_dir=plots_dir,
        )
        summary["results_by_position"][res["position_folder"]] = {
            "by_subset": res["by_subset"]
        }

    summary_path = results_root / alias / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[dyn-traj] summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.exit(main())
