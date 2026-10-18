"""
Train and evaluate static (single-layer) detectors of the microscopic
class (``honest_success`` / ``lie_success`` / ``lie_fail``) from cached
layer activations.

For every entry under ``feature_specs``, the script sweeps every layer in
the configured range, building a feature matrix per layer (e.g. three
projection scalars for ``proj_3way``, or the full ``d_model``-dim
activation for ``raw_activation``), running stratified K-fold logistic
regression with per-fold refit of any supervised DiM directions, and
emitting:

::

    <results_dir>/<run_alias>/<token_position>/<classes-joined-by-+>/<spec>/metrics.json
    <plots_dir>/<run_alias>/<token_position>/<classes-joined-by-+>/<spec>_cv_accuracy.png
    <plots_dir>/<run_alias>/<token_position>/<classes-joined-by-+>/<spec>_confusion_matrix_Lbest.png
    <plots_dir>/<run_alias>/<token_position>/<classes-joined-by-+>/<spec>_pca_Lbest_{2,3}D.png  (projection specs only)

Plus a per-subset roll-up at
``<results_dir>/<run_alias>/<token_position>/<classes-joined-by-+>/metrics.json``
and a top-level ``<results_dir>/<run_alias>/summary.json``.

Run with the project's standard layered-YAML pattern:

.. code-block:: bash

    python -m src.run_static_detector \\
        --config config/static_detector.yaml \\
        --data_folder results/runs/<model>/<task_dataset>/SysPrompt000/0_pre10_gen10
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import yaml

from src.analysis.dynamical_trajectory import (
    CLASS_NAMES,
    ClassTrajectories,
    _resolve_top_token_position,
    load_class_trajectories,
    restrict_class_trajectories,
)
from src.analysis.static_detector import (
    FeatureSpec,
    LayerSweepResult,
    StaticCVResult,
    TestSweepResult,
    TrainedStaticPipelineBundle,
    build_static_pipeline_bundle,
    compute_static_direction_cosines,
    evaluate_static_pipeline_on_test,
    parse_feature_specs,
    plot_confusion_matrix,
    plot_cv_accuracy_vs_layer,
    plot_direction_cosines_by_layer,
    plot_pca_at_layer,
    plot_test_accuracy_by_layer,
    plot_test_confusion_matrix_at_layer,
    run_layer_sweep,
    save_static_pipeline_bundle,
    _mode_slug,
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
        raise ValueError(
            f"Top-level YAML in {path} must be a mapping; got {type(raw).__name__}."
        )
    return raw


def _resolve_path(p: str) -> Path:
    """Resolve relative paths against ``PROJECT_ROOT``."""
    path = Path(p).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _run_alias(data_folder: Path) -> str:
    """Same handle derivation used by the dynamical runner."""
    parts = data_folder.resolve().parts
    if len(parts) >= 4:
        model = parts[-4]
        task_dataset = parts[-3]
        sys_prompt = parts[-2]
        seed_seg = parts[-1]
        seed_match = _SEED_RE.match(seed_seg)
        sys_match = _SYS_PROMPT_RE.match(sys_prompt)
        if seed_match and sys_match:
            return (
                f"{model}__{task_dataset}__SysPrompt{sys_match.group(1)}"
                f"__seed{seed_match.group(1)}"
            )
    return "__".join(parts[-2:])


def _subset_folder_name(subset_classes: Sequence[str]) -> str:
    """Folder slug for a class subset, e.g. ``honest_success+lie_success``."""
    return "+".join(subset_classes)


def _resolve_class_subsets(
    cfg: Mapping[str, Any],
    cli_override: Optional[Sequence[str]],
) -> List[Tuple[str, List[str]]]:
    """
    Read ``class_subsets`` from the YAML and intersect with ``cli_override``.

    Same semantics as ``run_dynamical_trajectory_analysis``: each entry is
    a mapping ``subset_name -> [class_names]``. CLI override is by *name*,
    not by class list. Defaults to a single ``all`` subset covering every
    entry in :data:`CLASS_NAMES` when both YAML and CLI are empty.
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


def _filter_feature_specs(
    specs: Sequence[FeatureSpec],
    cli_override: Optional[Sequence[str]],
) -> List[FeatureSpec]:
    """Restrict ``specs`` to the names listed in ``cli_override`` (if any)."""
    if not cli_override:
        return list(specs)
    wanted = set(cli_override)
    known = {s.name for s in specs}
    missing = wanted - known
    if missing:
        raise ValueError(
            f"--feature_specs references unknown spec(s) {sorted(missing)}; "
            f"YAML defines {sorted(known)}."
        )
    out = [s for s in specs if s.name in wanted]
    if not out:
        raise ValueError("--feature_specs selection produced an empty list.")
    return out


def _resolve_test_folders(
    cfg: Mapping[str, Any],
    cli_override: Optional[Sequence[Path]],
    *,
    require_activations: bool = True,
) -> List[Path]:
    """
    Resolve ``test.data_folder_test`` from the YAML, overridden by
    ``--data_folder_test`` when provided. Returns the resolved absolute
    paths (empty list = no test evaluation requested). When
    ``require_activations`` is False, skip the cache existence check so
    replot mode works even after activations have been pruned.
    """
    if cli_override is not None:
        raw_list: Sequence[Any] = list(cli_override)
    else:
        test_cfg = cfg.get("test", {}) or {}
        raw_list = list(test_cfg.get("data_folder_test", []) or [])
    resolved: List[Path] = []
    for entry in raw_list:
        p = _resolve_path(str(entry))
        if require_activations and not (p / "activations").exists():
            raise FileNotFoundError(
                f"test data folder {p} does not contain an 'activations/' subdir."
            )
        resolved.append(p)
    return resolved


def _parse_modes(
    cfg: Mapping[str, Any],
    key: str,
    *,
    valid: Sequence[str] = ("frozen", "refit_on_test"),
    default: Sequence[str] = ("frozen", "refit_on_test"),
) -> List[str]:
    """Parse ``test.<key>`` as a list of mode strings, with validation."""
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


def _mode_combos(
    spec_type: str,
    std_modes: Sequence[str],
    dir_modes: Sequence[str],
) -> List[Tuple[str, str]]:
    """
    Expand ``std_modes × dir_modes``. For specs without supervised
    directions (e.g. ``raw_activation``), direction_mode is collapsed to
    ``"frozen"`` because refitting nothing is a no-op.
    """
    combos: List[Tuple[str, str]] = []
    if spec_type == "projection":
        for s in std_modes:
            for d in dir_modes:
                combos.append((s, d))
    else:
        for s in std_modes:
            combo = (s, "frozen")
            if combo not in combos:
                combos.append(combo)
    return combos


# --------------------------------------------------------------------------- #
# Per-subset run                                                              #
# --------------------------------------------------------------------------- #


def _spec_metrics(sweep: LayerSweepResult) -> Dict[str, Any]:
    """Serialize a layer sweep to JSON-friendly metrics."""
    layers = [int(l) for l in sweep.layer_indices]
    per_layer: Dict[str, Any] = {}
    for l in layers:
        cv = sweep.per_layer[l]
        per_layer[str(l)] = {
            "fold_accuracy": cv.fold_accuracy.tolist(),
            "mean_accuracy": float(cv.fold_accuracy.mean()),
            "std_accuracy": float(cv.fold_accuracy.std()),
            "confusion_matrix": cv.confusion_matrix.tolist(),
            "confusion_matrix_classes": cv.class_names,
            "classes_present": cv.classes_present,
            "n_features": int(cv.X_full_scaled.shape[1]),
            "column_names": cv.column_names,
        }
    return {
        "spec_name": sweep.spec_name,
        "spec_type": sweep.spec_type,
        "layer_indices": layers,
        "mean_accuracy_by_layer": sweep.mean_acc_by_layer.tolist(),
        "std_accuracy_by_layer": sweep.std_acc_by_layer.tolist(),
        "best_layer": int(sweep.best_layer_abs),
        "best_mean_accuracy": float(sweep.mean_acc_by_layer.max()),
        "classes_evaluated": sweep.class_names,
        "per_layer": per_layer,
    }


def _test_plot_toggles(cfg: Mapping[str, Any]) -> Dict[str, bool]:
    """
    Read ``test.plots`` and return explicit booleans for each supported
    toggle. Unset = ``True`` so users only opt out.
    """
    test_cfg = cfg.get("test", {}) or {}
    raw = test_cfg.get("plots", {}) or {}
    keys = ("accuracy_by_layer", "confusion_matrix", "direction_cosines")
    return {k: bool(raw.get(k, True)) for k in keys}


def _run_tests_for_spec(
    *,
    spec: FeatureSpec,
    bundle: TrainedStaticPipelineBundle,
    train_sweep: LayerSweepResult,
    test_folders: Sequence[Path],
    cfg: Mapping[str, Any],
    results_dir: Path,
    plots_dir: Path,
    formats: Sequence[str],
    std_modes: Sequence[str],
    dir_modes: Sequence[str],
    report_direction_cosine: bool,
    plot_toggles: Mapping[str, bool],
) -> Dict[str, Any]:
    """
    For each test folder, evaluate the bundle for every requested
    ``(std_mode, dir_mode)`` combination plus (optionally) compute
    train-vs-test direction cosines. Returns a metrics dict keyed by
    test alias.
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
    combos = _mode_combos(spec.type, std_modes, dir_modes)

    out: Dict[str, Any] = {}
    for test_folder in test_folders:
        test_alias = _run_alias(test_folder)
        if token_position_policy == "match_train":
            test_pos: Optional[str] = bundle.training_token_position
        elif token_position_policy == "explicit":
            if not explicit_pos:
                raise ValueError(
                    "test.token_position_policy='explicit' requires "
                    "test.token_position_explicit to be set."
                )
            test_pos = str(explicit_pos)
        else:
            test_pos = _resolve_top_token_position(test_folder, rotation_metric)
            if test_pos is None:
                raise FileNotFoundError(
                    f"Cannot resolve top-1 token position for test folder "
                    f"{test_folder} (looked for "
                    f"ranked_token_pos_{rotation_metric} in "
                    f"{test_folder}/stage_stats/rotation_stats.json)."
                )
            if test_pos != bundle.training_token_position:
                print(
                    f"      [test {test_alias!r}] WARNING: test top-1 token "
                    f"position={test_pos!r} differs from training "
                    f"{bundle.training_token_position!r}. This is a transfer "
                    "experiment, not a held-out test."
                )

        sub_results = results_dir / "test" / test_alias / test_pos
        sub_plots = plots_dir / "test" / test_alias / test_pos
        sub_results.mkdir(parents=True, exist_ok=True)
        sub_plots.mkdir(parents=True, exist_ok=True)
        print(f"      [test {test_alias!r}] folder={test_folder} "
              f"top_pos={test_pos!r} -> {sub_plots}")

        rng = np.random.default_rng(int(data_cfg.get("data_seed", 0)))
        full_test = load_class_trajectories(
            test_folder,
            token_position=test_pos,
            layer_start=int(data_cfg.get("layer_start", -1)),
            layer_end=int(data_cfg.get("layer_end", -1)),
            n_samples_per_class=int(data_cfg.get("n_samples_per_class", -1)),
            rng=rng,
        )
        available_test = [c for c in bundle.class_names if c in full_test.trajectories]
        if not available_test:
            raise ValueError(
                f"Test folder {test_folder} has no rows from training class "
                f"set {bundle.class_names} (available: "
                f"{list(full_test.trajectories)})."
            )
        if available_test != list(bundle.class_names):
            dropped = [c for c in bundle.class_names if c not in available_test]
            print(f"        note: test missing class(es) {dropped}; "
                  "scoring on intersection only.")
        if list(full_test.trajectories.keys()) == available_test:
            test_cls_traj = full_test
        else:
            test_cls_traj, _ = restrict_class_trajectories(full_test, available_test)

        test_sweeps_for_plot: List[TestSweepResult] = []
        combo_metrics: Dict[str, Any] = {}
        for std_mode, dir_mode in combos:
            slug = _mode_slug(std_mode, dir_mode, spec_type=spec.type)
            sweep_test = evaluate_static_pipeline_on_test(
                bundle, test_cls_traj,
                standardization_mode=std_mode,
                direction_mode=dir_mode,
                direction_fit_data=full_test,
            )
            test_sweeps_for_plot.append(sweep_test)
            mode_dir = sub_results / slug
            mode_dir.mkdir(parents=True, exist_ok=True)
            best_train_layer = train_sweep.best_layer_abs
            best_test_layer = sweep_test.best_layer_abs
            per_layer_dump = {
                str(int(l)): {
                    "test_accuracy": float(sweep_test.per_layer[int(l)].test_accuracy),
                    "confusion_matrix": sweep_test.per_layer[int(l)].confusion_matrix.tolist(),
                    "confusion_matrix_classes": sweep_test.per_layer[int(l)].class_names,
                    "classes_present": sweep_test.per_layer[int(l)].classes_present,
                    "n_per_class": sweep_test.per_layer[int(l)].n_per_class,
                }
                for l in sweep_test.layer_indices
            }
            metrics = {
                "spec_name": spec.name,
                "spec_type": spec.type,
                "standardization_mode": std_mode,
                "direction_mode": dir_mode,
                "test_folder": str(test_folder),
                "test_alias": test_alias,
                "test_token_position": test_pos,
                "training_token_position": bundle.training_token_position,
                "rotation_metric": rotation_metric,
                "classes_evaluated": available_test,
                "layer_indices": [int(l) for l in sweep_test.layer_indices],
                "accuracy_by_layer": sweep_test.accuracy_by_layer.tolist(),
                "best_train_layer": int(best_train_layer),
                "best_train_layer_test_accuracy": float(
                    sweep_test.per_layer[int(best_train_layer)].test_accuracy
                ),
                "best_test_layer": int(best_test_layer),
                "best_test_layer_accuracy": float(
                    sweep_test.per_layer[int(best_test_layer)].test_accuracy
                ),
                "per_layer": per_layer_dump,
            }
            metrics_path = mode_dir / "metrics.json"
            metrics_path.write_text(json.dumps(metrics, indent=2))
            print(
                f"        [{slug}] test_acc@Ltrain={best_train_layer}: "
                f"{metrics['best_train_layer_test_accuracy']:.3f} | "
                f"test_acc@Ltest={best_test_layer}: "
                f"{metrics['best_test_layer_accuracy']:.3f}"
            )
            combo_metrics[slug] = {
                "metrics_path": str(metrics_path),
                "best_train_layer_test_accuracy": metrics["best_train_layer_test_accuracy"],
                "best_test_layer": metrics["best_test_layer"],
                "best_test_layer_accuracy": metrics["best_test_layer_accuracy"],
            }

            if plot_toggles.get("confusion_matrix", True):
                cm_path = plot_test_confusion_matrix_at_layer(
                    sweep_test.per_layer[int(best_train_layer)],
                    basepath=sub_plots / f"confusion_matrix_{slug}_Ltrainbest",
                    title=(
                        f"{spec.name} [{slug}] — L{best_train_layer} "
                        f"(train-best)  acc="
                        f"{metrics['best_train_layer_test_accuracy']:.3f}"
                    ),
                    formats=formats,
                )
                for p in cm_path:
                    print(f"        wrote {p}")

        if plot_toggles.get("accuracy_by_layer", True):
            acc_path = plot_test_accuracy_by_layer(
                train_sweep, test_sweeps_for_plot,
                basepath=sub_plots / "accuracy_by_layer",
                formats=formats,
            )
            for p in acc_path:
                print(f"        wrote {p}")

        cosines_summary: Optional[Dict[str, List[float]]] = None
        if report_direction_cosine and spec.type == "projection":
            cosines = compute_static_direction_cosines(bundle, full_test)
            if cosines:
                cosines_summary = {
                    name: arr.tolist() for name, arr in cosines.items()
                }
                cos_path = sub_results / "direction_cosines.json"
                cos_path.write_text(json.dumps({
                    "layer_indices": [int(l) for l in sweep_test.layer_indices],
                    "cosines": cosines_summary,
                }, indent=2))
                print(f"        wrote {cos_path}")
                if plot_toggles.get("direction_cosines", True):
                    cos_plot = plot_direction_cosines_by_layer(
                        cosines,
                        [int(l) for l in sweep_test.layer_indices],
                        basepath=sub_plots / "direction_cosines",
                        title=(
                            f"{spec.name} — direction cosine similarity "
                            f"(train vs test-refit)"
                        ),
                        formats=formats,
                    )
                    for p in cos_plot:
                        print(f"        wrote {p}")

        out[test_alias] = {
            "test_folder": str(test_folder),
            "test_token_position": test_pos,
            "classes_evaluated": available_test,
            "combos": combo_metrics,
            "direction_cosines": cosines_summary,
            "plots_dir": str(sub_plots),
        }
    return out


def _run_for_subset(
    *,
    cfg: Mapping[str, Any],
    full_cls_traj: ClassTrajectories,
    subset_classes: Sequence[str],
    subset_name: str,
    token_position: str,
    feature_specs: Sequence[FeatureSpec],
    test_folders: Sequence[Path],
    data_folder: Path,
    results_dir: Path,
    plots_dir: Path,
) -> Dict[str, Any]:
    """Run every feature_spec for one (token_position, class_subset)."""
    plot_cfg = cfg.get("plots", {}) or {}
    preprocessing_cfg = cfg.get("preprocessing", {}) or {}
    standardization_cfg = preprocessing_cfg.get("standardization", {}) or {}
    predictor_cfg = cfg.get("predictor", {}) or {}
    cv_cfg = cfg.get("cv", {}) or {}
    io_cfg = cfg.get("io", {}) or {}
    test_cfg = cfg.get("test", {}) or {}
    formats = list(io_cfg.get("plot_formats", ["png"]))
    class_colors = plot_cfg.get("class_colors", {}) or {}
    pca_dims = list(plot_cfg.get("pca_dims", [2, 3]))
    std_modes = _parse_modes(cfg, "standardization_modes")
    dir_modes = _parse_modes(cfg, "direction_modes")
    report_direction_cosine = bool(test_cfg.get("report_direction_cosine", True))
    test_plot_toggles = _test_plot_toggles(cfg)

    print(f"\n  [subset {subset_name!r}] classes={list(subset_classes)}")

    available = [c for c in subset_classes if c in full_cls_traj.trajectories]
    if not available:
        raise ValueError(
            f"Subset {subset_name!r} requests {list(subset_classes)} but none of "
            f"those classes are loaded (available: {list(full_cls_traj.trajectories)})."
        )
    if available != list(subset_classes):
        dropped = [c for c in subset_classes if c not in available]
        print(f"    note: dropping {dropped} (not present in cache).")

    if list(available) == list(full_cls_traj.trajectories):
        cls_traj = full_cls_traj
        subset_to_full_idx = None
        sup_fit_data: Optional[ClassTrajectories] = None
    else:
        cls_traj, subset_to_full_idx = restrict_class_trajectories(
            full_cls_traj, available
        )
        sup_fit_data = full_cls_traj

    print(
        f"    rows: {{ {', '.join(f'{c}: {a.shape[0]}' for c, a in cls_traj.trajectories.items())} }}, "
        f"total={cls_traj.feature_data.shape[0]}, "
        f"fit_pool={'full' if sup_fit_data is full_cls_traj else 'subset'}"
    )

    plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    by_spec: Dict[str, Any] = {}
    for spec in feature_specs:
        n_required = len(spec.kwargs.get("directions") or []) if spec.type == "projection" else None
        print(
            f"    spec={spec.name!r} type={spec.type}"
            + (f" directions={spec.kwargs.get('directions')}" if n_required else "")
        )
        sweep = run_layer_sweep(
            cls_traj=cls_traj,
            spec=spec,
            predictor_cfg=predictor_cfg,
            cv_cfg=cv_cfg,
            scaler_cfg=standardization_cfg,
            supervised_fit_data=sup_fit_data,
            subset_to_full_idx=subset_to_full_idx,
        )
        best_abs = sweep.best_layer_abs
        best_cv: StaticCVResult = sweep.per_layer[best_abs]
        best_mean = float(sweep.mean_acc_by_layer.max())
        print(
            f"      best layer L{best_abs}: mean_acc={best_mean:.3f}"
            f"  (n_features={best_cv.X_full_scaled.shape[1]})"
        )

        # Metrics
        spec_dir = results_dir / spec.name
        spec_dir.mkdir(parents=True, exist_ok=True)
        metrics = _spec_metrics(sweep)
        metrics_path = spec_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2))
        print(f"      wrote {metrics_path}")

        # Plots — all share the subset's plot dir, prefixed by spec.name.
        wrote: List[Path] = []
        wrote.extend(plot_cv_accuracy_vs_layer(
            sweep,
            basepath=plots_dir / f"{spec.name}_cv_accuracy",
            formats=formats,
        ))
        wrote.extend(plot_confusion_matrix(
            best_cv,
            basepath=plots_dir / f"{spec.name}_confusion_matrix_Lbest",
            title=(
                f"{spec.name} — confusion matrix at L{best_abs} "
                f"(CV-summed, row-norm)"
            ),
            formats=formats,
        ))
        if spec.type == "projection":
            wrote.extend(plot_pca_at_layer(
                best_cv,
                sweep.labels,
                title=f"{spec.name} (L{best_abs})",
                basepath=plots_dir / f"{spec.name}_pca_Lbest",
                dims=pca_dims,
                class_colors=class_colors,
                formats=formats,
            ))
        for p in wrote:
            print(f"      wrote {p}")

        # Save the trained pipeline bundle for this spec.
        bundle = build_static_pipeline_bundle(
            sweep,
            spec=spec,
            training_token_position=token_position,
            training_data_folder=data_folder,
            standardization_enabled=bool(standardization_cfg.get("enabled", True)),
            standardization_name=str(standardization_cfg.get("scaler", "standard")),
        )
        bundle_path = save_static_pipeline_bundle(bundle, spec_dir / "pipeline.pkl")
        print(f"      wrote {bundle_path}")

        # Held-out test evaluation (no-op when test_folders is empty).
        test_summary = _run_tests_for_spec(
            spec=spec,
            bundle=bundle,
            train_sweep=sweep,
            test_folders=test_folders,
            cfg=cfg,
            results_dir=spec_dir,
            plots_dir=plots_dir / spec.name,
            formats=formats,
            std_modes=std_modes,
            dir_modes=dir_modes,
            report_direction_cosine=report_direction_cosine,
            plot_toggles=test_plot_toggles,
        )

        by_spec[spec.name] = {
            "spec_type": spec.type,
            "metrics_path": str(metrics_path),
            "pipeline_path": str(bundle_path),
            "best_layer": int(best_abs),
            "best_mean_accuracy": best_mean,
            "n_features_best": int(best_cv.X_full_scaled.shape[1]),
            "tests": test_summary,
        }

    subset_metrics = {
        "token_position": token_position,
        "subset_name": subset_name,
        "classes": list(available),
        "layer_indices": [int(l) for l in cls_traj.layer_indices],
        "specs": by_spec,
    }
    subset_metrics_path = results_dir / "metrics.json"
    subset_metrics_path.write_text(json.dumps(subset_metrics, indent=2))
    print(f"    wrote {subset_metrics_path}")
    return subset_metrics


def run_for_position(
    *,
    cfg: Mapping[str, Any],
    data_folder: Path,
    token_position: str,
    class_subsets: Sequence[Tuple[str, List[str]]],
    feature_specs: Sequence[FeatureSpec],
    test_folders: Sequence[Path],
    results_dir: Path,
    plots_dir: Path,
) -> Dict[str, Any]:
    """Load one token position, then loop over class subsets."""
    print(f"\n[static-detector] token_position={token_position!r}")
    data_cfg = cfg.get("data", {}) or {}

    rng = np.random.default_rng(int(data_cfg.get("data_seed", 0)))
    full_cls_traj = load_class_trajectories(
        data_folder,
        token_position=token_position,
        layer_start=int(data_cfg.get("layer_start", -1)),
        layer_end=int(data_cfg.get("layer_end", -1)),
        n_samples_per_class=int(data_cfg.get("n_samples_per_class", -1)),
        rng=rng,
    )
    print(
        f"  classes loaded: {list(full_cls_traj.trajectories)} "
        f"(n_per_class={{ "
        f"{', '.join(f'{c}: {a.shape[0]}' for c, a in full_cls_traj.trajectories.items())} }}, "
        f"layers={len(full_cls_traj.layer_indices)})"
    )

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
            feature_specs=feature_specs,
            test_folders=test_folders,
            data_folder=data_folder,
            results_dir=sub_results,
            plots_dir=sub_plots,
        )
        by_subset[subset_name] = {
            "folder": subset_slug,
            "classes": metrics["classes"],
            "metrics_path": str(sub_results / "metrics.json"),
            "plots_dir": str(sub_plots),
            "specs": {
                name: {
                    "best_layer": body["best_layer"],
                    "best_mean_accuracy": body["best_mean_accuracy"],
                }
                for name, body in metrics["specs"].items()
            },
        }
    return by_subset


# --------------------------------------------------------------------------- #
# Replot from cached metrics                                                  #
# --------------------------------------------------------------------------- #


def _resolve_test_token_position_for_replot(
    test_folder: Path,
    cfg: Mapping[str, Any],
    train_token_position: str,
) -> str:
    """Mirror ``_run_tests_for_spec``'s token-position policy, without a bundle."""
    data_cfg = cfg.get("data", {}) or {}
    test_cfg = cfg.get("test", {}) or {}
    policy = str(test_cfg.get("token_position_policy", "test_top1"))
    if policy == "match_train":
        return train_token_position
    if policy == "explicit":
        explicit_pos = test_cfg.get("token_position_explicit")
        if not explicit_pos:
            raise ValueError(
                "test.token_position_policy='explicit' requires "
                "test.token_position_explicit to be set."
            )
        return str(explicit_pos)
    rotation_metric = str(
        test_cfg.get("rotation_metric")
        or data_cfg.get("rotation_metric", "largest_drop_any_early_to_late")
    )
    pos = _resolve_top_token_position(test_folder, rotation_metric)
    if pos is None:
        raise FileNotFoundError(
            f"Cannot resolve top-1 token position for test folder {test_folder} "
            f"(looked for ranked_token_pos_{rotation_metric} in "
            f"{test_folder}/stage_stats/rotation_stats.json)."
        )
    return pos


def _train_sweep_stub_from_json(metrics: Mapping[str, Any]) -> SimpleNamespace:
    """Minimal duck-typed stand-in for ``LayerSweepResult`` used by replot."""
    return SimpleNamespace(
        spec_name=metrics["spec_name"],
        spec_type=metrics["spec_type"],
        layer_indices=np.asarray(metrics["layer_indices"], dtype=np.int64),
        mean_acc_by_layer=np.asarray(
            metrics["mean_accuracy_by_layer"], dtype=np.float64
        ),
        std_acc_by_layer=np.asarray(
            metrics["std_accuracy_by_layer"], dtype=np.float64
        ),
        class_names=list(metrics.get("classes_evaluated") or CLASS_NAMES),
    )


def _test_sweep_stub_from_json(metrics: Mapping[str, Any]) -> SimpleNamespace:
    """Minimal duck-typed stand-in for ``TestSweepResult`` used by replot."""
    return SimpleNamespace(
        spec_name=metrics["spec_name"],
        spec_type=metrics["spec_type"],
        standardization_mode=metrics["standardization_mode"],
        direction_mode=metrics["direction_mode"],
        layer_indices=np.asarray(metrics["layer_indices"], dtype=np.int64),
        accuracy_by_layer=np.asarray(metrics["accuracy_by_layer"], dtype=np.float64),
        class_names=list(metrics.get("classes_evaluated") or CLASS_NAMES),
    )


def _test_layer_result_stub_from_json(
    per_layer_info: Mapping[str, Any], layer_idx: int
) -> SimpleNamespace:
    """Minimal duck-typed stand-in for ``TestLayerResult`` used by replot."""
    return SimpleNamespace(
        layer_idx=int(layer_idx),
        test_accuracy=float(per_layer_info["test_accuracy"]),
        confusion_matrix=np.asarray(
            per_layer_info["confusion_matrix"], dtype=np.float64
        ),
        class_names=list(per_layer_info.get("confusion_matrix_classes") or []),
        classes_present=list(per_layer_info.get("classes_present") or []),
        n_per_class=dict(per_layer_info.get("n_per_class") or {}),
    )


def _replot_tests_for_spec(
    *,
    spec_name: str,
    spec_type: str,
    train_sweep_stub: SimpleNamespace,
    train_token_position: str,
    test_folders: Sequence[Path],
    cfg: Mapping[str, Any],
    results_dir: Path,
    plots_dir: Path,
    formats: Sequence[str],
    std_modes: Sequence[str],
    dir_modes: Sequence[str],
    combo_filter: Optional[Set[str]],
    report_direction_cosine: bool,
    plot_toggles: Mapping[str, bool],
) -> None:
    """Regenerate every test plot for one spec from cached JSON."""
    if not test_folders:
        return
    combos = _mode_combos(spec_type, std_modes, dir_modes)
    for test_folder in test_folders:
        test_alias = _run_alias(test_folder)
        test_pos = _resolve_test_token_position_for_replot(
            test_folder, cfg, train_token_position
        )
        if test_pos != train_token_position:
            print(
                f"      [test {test_alias!r}] note: test top-1 token "
                f"position={test_pos!r} differs from training "
                f"{train_token_position!r}."
            )
        sub_results = results_dir / "test" / test_alias / test_pos
        sub_plots = plots_dir / "test" / test_alias / test_pos
        if not sub_results.exists():
            print(f"      [test {test_alias!r}] missing {sub_results}; skipping")
            continue
        sub_plots.mkdir(parents=True, exist_ok=True)
        print(
            f"      [test {test_alias!r}] folder={test_folder} "
            f"top_pos={test_pos!r} -> {sub_plots}"
        )

        loaded_combos: List[Tuple[str, SimpleNamespace, Dict[str, Any]]] = []
        for std_mode, dir_mode in combos:
            slug = _mode_slug(std_mode, dir_mode, spec_type=spec_type)
            metrics_path = sub_results / slug / "metrics.json"
            if not metrics_path.exists():
                print(f"        [{slug}] missing {metrics_path}; skipping combo")
                continue
            metrics = json.loads(metrics_path.read_text())
            sweep_stub = _test_sweep_stub_from_json(metrics)
            loaded_combos.append((slug, sweep_stub, metrics))

            if plot_toggles.get("confusion_matrix", True):
                best_train_layer = int(metrics["best_train_layer"])
                per_layer_dump = metrics.get("per_layer") or {}
                key = str(best_train_layer)
                if key not in per_layer_dump:
                    print(
                        f"        [{slug}] per_layer[{key}] missing in "
                        f"{metrics_path}; skipping CM"
                    )
                else:
                    tlr_stub = _test_layer_result_stub_from_json(
                        per_layer_dump[key], best_train_layer
                    )
                    cm_path = plot_test_confusion_matrix_at_layer(
                        tlr_stub,
                        basepath=sub_plots / f"confusion_matrix_{slug}_Ltrainbest",
                        title=(
                            f"{spec_name} [{slug}] — L{best_train_layer} "
                            f"(train-best)  acc="
                            f"{metrics['best_train_layer_test_accuracy']:.3f}"
                        ),
                        formats=formats,
                    )
                    for p in cm_path:
                        print(f"        wrote {p}")

        if plot_toggles.get("accuracy_by_layer", True) and loaded_combos:
            if combo_filter is None:
                selected = [s for _, s, _ in loaded_combos]
                selected_slugs = [slug for slug, _, _ in loaded_combos]
            else:
                selected = [
                    s for slug, s, _ in loaded_combos if slug in combo_filter
                ]
                selected_slugs = [
                    slug for slug, _, _ in loaded_combos if slug in combo_filter
                ]
                if not selected:
                    available = [slug for slug, _, _ in loaded_combos]
                    print(
                        f"        accuracy_by_layer: --replot_combos "
                        f"{sorted(combo_filter)} matched none of {available}; "
                        "skipping"
                    )
            if selected:
                if combo_filter is None:
                    base_name = "accuracy_by_layer"
                else:
                    base_name = "accuracy_by_layer__" + "+".join(selected_slugs)
                print(f"        {base_name}: lines={selected_slugs}")
                acc_path = plot_test_accuracy_by_layer(
                    train_sweep_stub, selected,
                    basepath=sub_plots / base_name,
                    formats=formats,
                )
                for p in acc_path:
                    print(f"        wrote {p}")

        if (
            report_direction_cosine
            and spec_type == "projection"
            and plot_toggles.get("direction_cosines", True)
        ):
            cos_path = sub_results / "direction_cosines.json"
            if cos_path.exists():
                cos_data = json.loads(cos_path.read_text())
                cosines = {
                    name: np.asarray(arr, dtype=np.float64)
                    for name, arr in (cos_data.get("cosines") or {}).items()
                }
                if cosines:
                    cos_plot = plot_direction_cosines_by_layer(
                        cosines,
                        [int(l) for l in cos_data.get("layer_indices", [])],
                        basepath=sub_plots / "direction_cosines",
                        title=(
                            f"{spec_name} — direction cosine similarity "
                            f"(train vs test-refit)"
                        ),
                        formats=formats,
                    )
                    for p in cos_plot:
                        print(f"        wrote {p}")


def _replot_for_subset(
    *,
    cfg: Mapping[str, Any],
    subset_classes: Sequence[str],
    subset_name: str,
    token_position: str,
    feature_specs: Sequence[FeatureSpec],
    test_folders: Sequence[Path],
    results_dir: Path,
    plots_dir: Path,
    combo_filter: Optional[Set[str]],
) -> None:
    """Replot all test plots for one (token_position, class_subset) folder."""
    io_cfg = cfg.get("io", {}) or {}
    test_cfg = cfg.get("test", {}) or {}
    formats = list(io_cfg.get("plot_formats", ["png"]))
    std_modes = _parse_modes(cfg, "standardization_modes")
    dir_modes = _parse_modes(cfg, "direction_modes")
    report_direction_cosine = bool(test_cfg.get("report_direction_cosine", True))
    plot_toggles = _test_plot_toggles(cfg)

    print(f"\n  [subset {subset_name!r}] classes={list(subset_classes)}  (replot)")

    for spec in feature_specs:
        spec_dir = results_dir / spec.name
        train_metrics_path = spec_dir / "metrics.json"
        if not train_metrics_path.exists():
            print(
                f"    spec={spec.name!r}: missing {train_metrics_path}; skipping spec"
            )
            continue
        train_metrics = json.loads(train_metrics_path.read_text())
        train_sweep_stub = _train_sweep_stub_from_json(train_metrics)
        print(
            f"    spec={spec.name!r} type={spec.type}  "
            f"(train metrics: {train_metrics_path})"
        )
        _replot_tests_for_spec(
            spec_name=spec.name,
            spec_type=spec.type,
            train_sweep_stub=train_sweep_stub,
            train_token_position=token_position,
            test_folders=test_folders,
            cfg=cfg,
            results_dir=spec_dir,
            plots_dir=plots_dir / spec.name,
            formats=formats,
            std_modes=std_modes,
            dir_modes=dir_modes,
            combo_filter=combo_filter,
            report_direction_cosine=report_direction_cosine,
            plot_toggles=plot_toggles,
        )


def _resolve_replot_subset_dir(
    results_dir: Path,
    subset_classes: Sequence[str],
) -> Optional[Path]:
    """
    Find the on-disk folder corresponding to ``subset_classes`` under
    ``results_dir``. Tries the canonical ``a+b+c`` slug first, then falls
    back to any subfolder whose classes are a subset (e.g. when training
    dropped a missing class). Returns ``None`` if no match found.
    """
    canonical = results_dir / _subset_folder_name(subset_classes)
    if (canonical / "metrics.json").exists():
        return canonical
    if not results_dir.exists():
        return None
    wanted = set(subset_classes)
    candidates = [
        cand for cand in results_dir.iterdir()
        if cand.is_dir()
        and set(cand.name.split("+")) <= wanted
        and (cand / "metrics.json").exists()
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: -len(p.name.split("+")))
    return candidates[0]


def replot_for_position(
    *,
    cfg: Mapping[str, Any],
    token_position: str,
    class_subsets: Sequence[Tuple[str, List[str]]],
    feature_specs: Sequence[FeatureSpec],
    test_folders: Sequence[Path],
    results_dir: Path,
    plots_dir: Path,
    combo_filter: Optional[Set[str]],
) -> None:
    """Replot every (subset × spec × test) combination for one token position."""
    print(f"\n[static-detector replot] token_position={token_position!r}")
    if not results_dir.exists():
        print(f"  missing results dir {results_dir}; nothing to replot")
        return
    for subset_name, subset_classes in class_subsets:
        sub_results = _resolve_replot_subset_dir(results_dir, subset_classes)
        if sub_results is None:
            print(
                f"  [subset {subset_name!r}] no matching results folder under "
                f"{results_dir} (looked for "
                f"{_subset_folder_name(subset_classes)}); skipping"
            )
            continue
        sub_plots = plots_dir / sub_results.name
        _replot_for_subset(
            cfg=cfg,
            subset_classes=subset_classes,
            subset_name=subset_name,
            token_position=token_position,
            feature_specs=feature_specs,
            test_folders=test_folders,
            results_dir=sub_results,
            plots_dir=sub_plots,
            combo_filter=combo_filter,
        )


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "config" / "static_detector.yaml",
        help="Path to the static-detector YAML config.",
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
        "--class_subsets", type=str, nargs="*", default=None,
        help="Names of class_subsets entries from the YAML to run. Empty = "
             "run every subset defined in class_subsets (or the implicit "
             "'all' subset when the YAML defines none).",
    )
    parser.add_argument(
        "--feature_specs", type=str, nargs="*", default=None,
        help="Names of feature_specs entries from the YAML to run. Empty = "
             "run every feature_spec.",
    )
    parser.add_argument(
        "--data_folder_test", type=Path, nargs="*", default=None,
        help="Cache seed folder(s) to evaluate the trained pipeline on. "
             "Each folder becomes a subfolder under "
             "<plots_dir>/<alias>/<train_token>/<subset>/<spec>/test/. "
             "Overrides test.data_folder_test in the YAML.",
    )
    parser.add_argument(
        "--replot_only", action="store_true",
        help="Skip training/evaluation; regenerate test plots "
             "(accuracy_by_layer, confusion_matrix_*_Ltrainbest, "
             "direction_cosines) from cached metrics.json under results_dir. "
             "Overwrites the existing PNG/PDF files in place.",
    )
    parser.add_argument(
        "--replot_combos", type=str, nargs="*", default=None,
        help="(replot only) Restrict the accuracy_by_layer plot to these "
             "(std_mode, dir_mode) combo slugs. When set, the figure is "
             "written to accuracy_by_layer__<slug1>+<slug2>+...{png,pdf} "
             "instead of overwriting the original accuracy_by_layer.{png,pdf}. "
             "Confusion-matrix and direction-cosine plots are unaffected. "
             "Examples: 'std-frozen__dir-frozen', 'std-refit__dir-refit' "
             "(projection specs); 'std-frozen', 'std-refit' (non-projection "
             "specs).",
    )
    args = parser.parse_args(argv)

    if args.replot_combos is not None and not args.replot_only:
        parser.error("--replot_combos requires --replot_only.")
    combo_filter: Optional[Set[str]] = (
        set(args.replot_combos) if args.replot_combos else None
    )

    cfg = _load_config(args.config)
    data_cfg = cfg.setdefault("data", {})
    io_cfg = cfg.setdefault("io", {})

    if args.data_folder is not None:
        data_cfg["data_folder"] = str(args.data_folder)
    if args.token_positions is not None:
        data_cfg["token_positions"] = list(args.token_positions)
    class_subsets = _resolve_class_subsets(cfg, args.class_subsets)
    feature_specs = _filter_feature_specs(
        parse_feature_specs(cfg.get("feature_specs") or {}),
        args.feature_specs,
    )
    test_folders = _resolve_test_folders(
        cfg, args.data_folder_test, require_activations=not args.replot_only
    )

    data_folder_raw = data_cfg.get("data_folder")
    if not data_folder_raw:
        parser.error(
            "data.data_folder is required (set in YAML or via --data_folder)."
        )
    data_folder = _resolve_path(data_folder_raw)
    if not args.replot_only and not (data_folder / "activations").exists():
        parser.error(
            f"--data_folder {data_folder} does not contain an 'activations/' subdir."
        )

    requested_positions = list(data_cfg.get("token_positions") or [])
    if not requested_positions:
        top = _resolve_top_token_position(
            data_folder,
            str(data_cfg.get("rotation_metric", "largest_drop_any_early_to_late")),
        )
        if top is None:
            parser.error(
                f"Cannot resolve top-1 token position from "
                f"{data_folder}/stage_stats/rotation_stats.json. Pass "
                "--token_positions or set data.token_positions in the YAML."
            )
        requested_positions = [top]

    print(f"[static-detector] data_folder={data_folder}")
    print(f"[static-detector] token_positions={requested_positions}")
    print(
        f"[static-detector] class_subsets={[(n, c) for n, c in class_subsets]}"
    )
    print(
        f"[static-detector] feature_specs="
        f"{[(s.name, s.type) for s in feature_specs]}"
    )
    if test_folders:
        print(f"[static-detector] test data_folders={[str(p) for p in test_folders]}")

    results_root = _resolve_path(io_cfg.get("results_dir", "results/static_detector"))
    plots_root = _resolve_path(
        io_cfg.get("plots_dir", "results/plots/static_detector")
    )
    alias = _run_alias(data_folder)

    if args.replot_only:
        if combo_filter is not None:
            print(f"[static-detector replot] combo_filter={sorted(combo_filter)}")
        if not test_folders:
            print(
                "[static-detector replot] no test data_folders configured; "
                "nothing to replot (test plots only)."
            )
            return 0
        for position in requested_positions:
            results_dir = results_root / alias / position
            plots_dir = plots_root / alias / position
            replot_for_position(
                cfg=cfg,
                token_position=position,
                class_subsets=class_subsets,
                feature_specs=feature_specs,
                test_folders=test_folders,
                results_dir=results_dir,
                plots_dir=plots_dir,
                combo_filter=combo_filter,
            )
        print("\n[static-detector replot] done.")
        return 0

    summary: Dict[str, Any] = {
        "data_folder": str(data_folder),
        "class_subsets": {n: c for n, c in class_subsets},
        "feature_specs": [
            {"name": s.name, "type": s.type, "kwargs": dict(s.kwargs)}
            for s in feature_specs
        ],
        "test_data_folders": [str(p) for p in test_folders],
        "results_by_position": {},
    }
    for position in requested_positions:
        results_dir = results_root / alias / position
        plots_dir = plots_root / alias / position
        by_subset = run_for_position(
            cfg=cfg,
            data_folder=data_folder,
            token_position=position,
            class_subsets=class_subsets,
            feature_specs=feature_specs,
            test_folders=test_folders,
            results_dir=results_dir,
            plots_dir=plots_dir,
        )
        summary["results_by_position"][position] = {"by_subset": by_subset}

    summary_path = results_root / alias / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[static-detector] summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.exit(main())
