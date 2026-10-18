"""
Static (single-layer) detector of the microscopic class
(``honest_success`` / ``lie_success`` / ``lie_fail``).

The dynamical pipeline in :mod:`src.analysis.dynamical_trajectory` flattens
descriptors across every layer into one feature vector. This module instead
sweeps each layer in isolation: for every layer ``L`` and every named
``feature_spec`` we build a feature matrix ``(n_samples, n_features_L)``
and fit a multinomial logistic regression with the same CV protocol used
by the dynamical pipeline (per-fold scaler + LR; per-fold refit of any
supervised DiM directions). Comparing layers gives a CV-accuracy-vs-layer
curve per spec.

Feature specs are plug-in builders. Two are registered out of the box:

- ``projection``: requires ``directions: [name, ...]`` where each name keys
  :data:`SUPERVISED_DESCRIPTOR_REGISTRY`. At layer ``L`` we take the per-
  layer DiM direction at index ``L`` and project the layer-``L`` activation
  onto it, producing ``len(directions)`` scalar features per sample.
- ``raw_activation``: features are the layer-``L`` activation vector itself,
  i.e. ``cls_traj.feature_data[:, L, :]`` (``d_model`` features).

Register additional builders by adding entries to
:data:`FEATURE_BUILDER_REGISTRY` — each callable takes a
``ClassTrajectories`` slab, a layer position, the spec's kwargs, and a
mapping of pre-fit supervised directions (``{name: (L, D) array}``), and
returns ``(X, column_names)`` where ``X.shape == (n_samples, n_features)``.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt

from src.analysis.dynamical_trajectory import (
    CLASS_NAMES,
    ClassTrajectories,
    SUPERVISED_DESCRIPTOR_REGISTRY,
    _make_cv_splitter,
    _make_scaler,
    _save_figure,
    is_supervised_descriptor,
    make_logistic_regression,
    plot_pca_scatter,
)


# --------------------------------------------------------------------------- #
# Types                                                                       #
# --------------------------------------------------------------------------- #


FeatureBuilder = Callable[
    [ClassTrajectories, int, Mapping[str, Any], Mapping[str, np.ndarray]],
    Tuple[np.ndarray, List[str]],
]


@dataclass
class FeatureSpec:
    """One static feature_spec entry from the YAML."""

    name: str
    type: str
    kwargs: Mapping[str, Any]


@dataclass
class StaticCVResult:
    """CV outcome for one feature_spec at one absolute layer."""

    layer_idx: int
    fold_accuracy: np.ndarray
    confusion_matrix: np.ndarray
    classes_present: List[str]
    class_names: List[str]
    column_names: List[str]
    full_model: LogisticRegression
    full_scaler: Optional[Any]
    X_full_scaled: np.ndarray


@dataclass
class LayerSweepResult:
    """All-layer sweep for one feature_spec."""

    spec_name: str
    spec_type: str
    layer_indices: np.ndarray
    per_layer: Dict[int, StaticCVResult]
    class_names: List[str]
    classes_present: List[str]
    labels: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    # ``{direction_name: (n_layers, d_model)}`` DiM directions fit on the
    # full supervised pool (after CV). Empty for non-projection specs.
    # Kept alongside the per-layer pipelines so the full bundle can be
    # serialized and applied to test data without recomputing the fit.
    supervised_directions: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def mean_acc_by_layer(self) -> np.ndarray:
        return np.asarray(
            [self.per_layer[int(l)].fold_accuracy.mean() for l in self.layer_indices],
            dtype=np.float64,
        )

    @property
    def std_acc_by_layer(self) -> np.ndarray:
        return np.asarray(
            [self.per_layer[int(l)].fold_accuracy.std() for l in self.layer_indices],
            dtype=np.float64,
        )

    @property
    def best_layer_abs(self) -> int:
        means = self.mean_acc_by_layer
        return int(self.layer_indices[int(np.argmax(means))])


# --------------------------------------------------------------------------- #
# Feature builders                                                            #
# --------------------------------------------------------------------------- #


def build_projection_features(
    cls_traj: ClassTrajectories,
    layer_pos: int,
    spec_kwargs: Mapping[str, Any],
    fit_directions: Mapping[str, np.ndarray],
) -> Tuple[np.ndarray, List[str]]:
    """
    Project the layer-``layer_pos`` activations onto every named direction.

    ``spec_kwargs['directions']`` must be a non-empty list of names that key
    :data:`SUPERVISED_DESCRIPTOR_REGISTRY`. ``fit_directions[name]`` is the
    ``(n_layers, d_model)`` DiM matrix fit elsewhere (per-fold inside CV,
    full-pool for the visualization refit); this builder just slices row
    ``layer_pos`` and dots it against the activations at that layer.
    """
    directions = list(spec_kwargs.get("directions") or [])
    if not directions:
        raise ValueError(
            "projection feature_spec requires non-empty 'directions' kwarg."
        )
    acts = cls_traj.feature_data[:, layer_pos, :]  # (n, D)
    cols: List[np.ndarray] = []
    names: List[str] = []
    for d in directions:
        if d not in fit_directions:
            raise KeyError(
                f"projection: direction {d!r} not in pre-fit directions "
                f"({list(fit_directions)})."
            )
        D_l = np.asarray(fit_directions[d][layer_pos], dtype=np.float32)
        if D_l.shape != (acts.shape[1],):
            raise ValueError(
                f"projection: direction {d!r} layer-{layer_pos} shape {D_l.shape} "
                f"does not match d_model={acts.shape[1]}."
            )
        proj = (acts.astype(np.float32, copy=False) @ D_l).astype(np.float32, copy=False)
        cols.append(proj[:, None])
        names.append(d)
    X = np.concatenate(cols, axis=1).astype(np.float32, copy=False)
    return X, names


def build_raw_activation_features(
    cls_traj: ClassTrajectories,
    layer_pos: int,
    spec_kwargs: Mapping[str, Any],
    fit_directions: Mapping[str, np.ndarray],  # noqa: ARG001 — unused
) -> Tuple[np.ndarray, List[str]]:
    """Features = the layer-``layer_pos`` activation vector (``d_model`` features)."""
    arr = cls_traj.feature_data[:, layer_pos, :].astype(np.float32, copy=False)
    names = [f"act_dim_{i}" for i in range(arr.shape[1])]
    return arr, names


FEATURE_BUILDER_REGISTRY: Dict[str, FeatureBuilder] = {
    "projection": build_projection_features,
    "raw_activation": build_raw_activation_features,
}


def feature_spec_required_directions(spec: FeatureSpec) -> List[str]:
    """Names this spec needs pre-fit before its builder can run."""
    if spec.type == "projection":
        return list(spec.kwargs.get("directions") or [])
    return []


def parse_feature_specs(raw: Mapping[str, Any]) -> List[FeatureSpec]:
    """Translate the YAML ``feature_specs`` mapping into typed specs."""
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"feature_specs must be a mapping (name -> {{type, ...}}); "
            f"got {type(raw).__name__}."
        )
    out: List[FeatureSpec] = []
    for name, body in raw.items():
        body = body or {}
        if "type" not in body:
            raise KeyError(f"feature_specs.{name}: missing required 'type' field.")
        spec_type = str(body["type"])
        if spec_type not in FEATURE_BUILDER_REGISTRY:
            raise KeyError(
                f"feature_specs.{name}: unknown type {spec_type!r}; "
                f"known: {sorted(FEATURE_BUILDER_REGISTRY)}."
            )
        kwargs = {k: v for k, v in body.items() if k != "type"}
        # Validate directions reference the supervised registry up front.
        if spec_type == "projection":
            for d in (kwargs.get("directions") or []):
                if not is_supervised_descriptor(str(d)):
                    raise KeyError(
                        f"feature_specs.{name}: direction {d!r} is not in "
                        f"SUPERVISED_DESCRIPTOR_REGISTRY."
                    )
        out.append(FeatureSpec(name=str(name), type=spec_type, kwargs=kwargs))
    if not out:
        raise ValueError("feature_specs is empty — at least one spec required.")
    return out


# --------------------------------------------------------------------------- #
# Layer sweep CV                                                              #
# --------------------------------------------------------------------------- #


def _fit_directions(
    names: Sequence[str],
    fit_traj: np.ndarray,
    fit_class: np.ndarray,
    fit_gt: np.ndarray,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for d in names:
        if d not in SUPERVISED_DESCRIPTOR_REGISTRY:
            raise KeyError(
                f"_fit_directions: {d!r} is not in SUPERVISED_DESCRIPTOR_REGISTRY."
            )
        out[d] = SUPERVISED_DESCRIPTOR_REGISTRY[d].fit(fit_traj, fit_class, fit_gt)
    return out


def run_layer_sweep(
    cls_traj: ClassTrajectories,
    spec: FeatureSpec,
    *,
    predictor_cfg: Mapping[str, Any],
    cv_cfg: Mapping[str, Any],
    scaler_cfg: Mapping[str, Any],
    supervised_fit_data: Optional[ClassTrajectories] = None,
    subset_to_full_idx: Optional[np.ndarray] = None,
) -> LayerSweepResult:
    """
    Sweep every layer in ``cls_traj.layer_indices`` for one feature_spec.

    For each fold, supervised DiM directions are refit on the training rows
    (mirroring :func:`src.analysis.dynamical_trajectory.run_cv`). When
    ``supervised_fit_data`` and ``subset_to_full_idx`` are provided,
    direction fitting uses the full slab minus the rows that correspond to
    the held-out subset test fold, so out-of-subset classes (e.g.
    ``honest_success`` in a ``lie_only`` run) still contribute to the DiM.

    Returns a :class:`LayerSweepResult` whose ``per_layer`` dict maps each
    absolute layer index to a :class:`StaticCVResult`.
    """
    if spec.type not in FEATURE_BUILDER_REGISTRY:
        raise KeyError(
            f"unknown feature_spec type {spec.type!r}; "
            f"known: {sorted(FEATURE_BUILDER_REGISTRY)}."
        )
    builder = FEATURE_BUILDER_REGISTRY[spec.type]
    required_dirs = feature_spec_required_directions(spec)

    layer_indices = cls_traj.layer_indices
    n_layers = len(layer_indices)
    y = cls_traj.labels
    n_samples = cls_traj.feature_data.shape[0]

    classes_present = sorted(int(c) for c in np.unique(y))
    classes_present_names = [CLASS_NAMES[i] for i in classes_present]
    subset_label_indices = sorted(
        CLASS_NAMES.index(c) for c in cls_traj.trajectories.keys()
    )
    class_names = [CLASS_NAMES[i] for i in subset_label_indices]

    use_external_fit_pool = (
        required_dirs
        and supervised_fit_data is not None
        and subset_to_full_idx is not None
    )
    if use_external_fit_pool:
        full_n = supervised_fit_data.feature_data.shape[0]

    splitter = _make_cv_splitter(cv_cfg)
    splitter_input = np.zeros((n_samples, 1), dtype=np.float32)
    fold_iter = list(
        splitter.split(splitter_input, y)
        if hasattr(splitter, "split") else splitter.split(splitter_input)
    )
    n_splits = len(fold_iter)
    accs = np.zeros((n_splits, n_layers), dtype=np.float64)
    cm_per_layer = np.zeros(
        (n_layers, len(class_names), len(class_names)), dtype=np.int64
    )

    scaler_enabled = bool(scaler_cfg.get("enabled", True))
    scaler_name = str(scaler_cfg.get("scaler", "standard"))

    for fold_idx, (train_idx, test_idx) in enumerate(fold_iter):
        if required_dirs:
            if use_external_fit_pool:
                full_test_idx = subset_to_full_idx[test_idx]
                fit_pool_mask = np.ones(full_n, dtype=bool)
                fit_pool_mask[full_test_idx] = False
                fit_traj = supervised_fit_data.feature_data[fit_pool_mask]
                fit_class = supervised_fit_data.labels[fit_pool_mask]
                fit_gt = supervised_fit_data.gt_labels[fit_pool_mask]
            else:
                fit_traj = cls_traj.feature_data[train_idx]
                fit_class = cls_traj.labels[train_idx]
                fit_gt = cls_traj.gt_labels[train_idx]
            fit_dirs = _fit_directions(required_dirs, fit_traj, fit_class, fit_gt)
        else:
            fit_dirs = {}

        for layer_pos in range(n_layers):
            X_layer, _ = builder(cls_traj, layer_pos, spec.kwargs, fit_dirs)
            scaler = _make_scaler(scaler_name) if scaler_enabled else None
            if scaler is not None:
                X_tr = scaler.fit_transform(X_layer[train_idx])
                X_te = scaler.transform(X_layer[test_idx])
            else:
                X_tr = X_layer[train_idx]
                X_te = X_layer[test_idx]
            model = make_logistic_regression(predictor_cfg)
            model.fit(X_tr, y[train_idx])
            y_pred = model.predict(X_te)
            accs[fold_idx, layer_pos] = float(accuracy_score(y[test_idx], y_pred))
            cm_per_layer[layer_pos] += confusion_matrix(
                y[test_idx], y_pred, labels=subset_label_indices
            ).astype(np.int64, copy=False)

    # Full-data refit per layer (in-sample direction fit on the full
    # supervised pool). Used for the per-layer PCA visualization and the
    # confusion matrix at the best layer is taken from the CV-summed cm.
    if required_dirs:
        if supervised_fit_data is not None:
            fit_pool = supervised_fit_data
        else:
            fit_pool = cls_traj
        fit_dirs_full = _fit_directions(
            required_dirs,
            fit_pool.feature_data,
            fit_pool.labels,
            fit_pool.gt_labels,
        )
    else:
        fit_dirs_full = {}

    per_layer: Dict[int, StaticCVResult] = {}
    for layer_pos in range(n_layers):
        abs_layer = int(layer_indices[layer_pos])
        X_layer, col_names = builder(cls_traj, layer_pos, spec.kwargs, fit_dirs_full)
        scaler_full = _make_scaler(scaler_name) if scaler_enabled else None
        X_full = scaler_full.fit_transform(X_layer) if scaler_full is not None else X_layer
        model_full = make_logistic_regression(predictor_cfg)
        model_full.fit(X_full, y)
        per_layer[abs_layer] = StaticCVResult(
            layer_idx=abs_layer,
            fold_accuracy=accs[:, layer_pos].copy(),
            confusion_matrix=cm_per_layer[layer_pos].copy(),
            classes_present=classes_present_names,
            class_names=class_names,
            column_names=col_names,
            full_model=model_full,
            full_scaler=scaler_full,
            X_full_scaled=np.asarray(X_full, dtype=np.float32),
        )

    return LayerSweepResult(
        spec_name=spec.name,
        spec_type=spec.type,
        layer_indices=layer_indices.copy(),
        per_layer=per_layer,
        class_names=class_names,
        classes_present=classes_present_names,
        labels=y.copy(),
        supervised_directions={
            name: np.asarray(arr, dtype=np.float32)
            for name, arr in fit_dirs_full.items()
        },
    )


# --------------------------------------------------------------------------- #
# Plotting                                                                    #
# --------------------------------------------------------------------------- #


def plot_cv_accuracy_vs_layer(
    sweep: LayerSweepResult,
    *,
    basepath: Path,
    formats: Sequence[str] = ("png",),
) -> List[Path]:
    """Mean ± std CV accuracy across layers for one feature_spec."""
    layers = np.asarray(sweep.layer_indices)
    means = sweep.mean_acc_by_layer
    stds = sweep.std_acc_by_layer
    n_classes = len(sweep.class_names) or len(CLASS_NAMES)
    chance = 1.0 / max(1, n_classes)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(layers, means, color="#4C72B0", marker="o", linewidth=1.5,
            label=sweep.spec_name)
    ax.fill_between(layers, means - stds, means + stds, color="#4C72B0", alpha=0.25)
    ax.axhline(chance, color="grey", linestyle="--", linewidth=1.0,
               label=f"chance ({chance:.2f})")
    best_idx = int(np.argmax(means))
    best_layer = int(layers[best_idx])
    ax.axvline(best_layer, color="red", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.scatter(
        [best_layer], [means[best_idx]],
        color="red", s=40, zorder=5,
        label=f"best L{best_layer}: {means[best_idx]:.3f}",
    )
    ax.set_xlabel("layer")
    ax.set_ylabel("CV accuracy (mean ± std)")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.suptitle(
        f"CV accuracy by layer — {sweep.spec_name} ({sweep.spec_type})",
        fontsize=11,
    )
    fig.tight_layout()
    return _save_figure(fig, basepath, formats)


def plot_confusion_matrix(
    cv: StaticCVResult,
    *,
    basepath: Path,
    title: Optional[str] = None,
    formats: Sequence[str] = ("png",),
    normalize: bool = True,
) -> List[Path]:
    """Single-panel CV-summed confusion matrix (row-normalized by default)."""
    cm = cv.confusion_matrix.astype(np.float64)
    tick_labels = list(cv.class_names) if cv.class_names else list(CLASS_NAMES)
    k = len(tick_labels)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_show = cm / row_sums
        vmax = 1.0
    else:
        cm_show = cm
        vmax = cm.max() if cm.size else 1.0

    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    im = ax.imshow(cm_show, cmap="Blues", vmin=0.0, vmax=vmax)
    ax.set_xticks(np.arange(k))
    ax.set_yticks(np.arange(k))
    ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(tick_labels, fontsize=8)
    for i in range(k):
        for j in range(k):
            text = f"{cm_show[i, j]:.2f}" if normalize else f"{int(cm_show[i, j])}"
            ax.text(
                j, i, text,
                ha="center", va="center",
                color="white" if cm_show[i, j] > vmax * 0.6 else "black",
                fontsize=8,
            )
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        title or f"Confusion matrix — L{cv.layer_idx} (CV-summed, row-norm)",
        fontsize=10,
    )
    fig.tight_layout()
    return _save_figure(fig, basepath, formats)


def plot_pca_at_layer(
    cv: StaticCVResult,
    labels: np.ndarray,
    *,
    title: str,
    basepath: Path,
    dims: Sequence[int] = (2, 3),
    class_colors: Optional[Mapping[str, str]] = None,
    formats: Sequence[str] = ("png",),
) -> List[Path]:
    """PCA scatter of the full-data scaled features at one layer."""
    return plot_pca_scatter(
        cv.X_full_scaled, labels,
        title=title,
        basepath=basepath,
        dims=dims,
        class_colors=class_colors,
        scaler=None,  # X_full_scaled is already scaled.
        formats=formats,
        class_names=cv.class_names,
    )


# --------------------------------------------------------------------------- #
# Trained-pipeline serialization + held-out test evaluation                   #
# --------------------------------------------------------------------------- #


@dataclass
class TrainedStaticPipelineBundle:
    """
    Per-spec, per-layer trained pipeline ready to apply to test data.

    Attributes:
        spec_name / spec_type / spec_kwargs: Identify how to rebuild test
            features (which builder, which directions / kwargs to pass).
        training_token_position / training_layer_indices /
        training_data_folder: Training-time provenance. Layer indices pin
            the layer count so a mismatched test cache (e.g. different
            model) raises rather than silently mis-applying.
        class_names: Classes the predictors were trained to discriminate
            (in :data:`CLASS_NAMES` order). Test rows are restricted to
            these before scoring.
        scalers: ``{abs_layer: scaler}`` — sklearn scaler fit on the full
            training matrix for that layer (or ``None`` if standardization
            was disabled).
        models: ``{abs_layer: LogisticRegression}`` — full-data refit per
            layer.
        supervised_directions: ``{direction_name: (n_layers, d_model)}``
            DiM matrices fit on the full training supervised pool. Empty
            for non-projection specs. Used by ``direction_mode='frozen'``
            test evaluation; ignored when ``direction_mode='refit_on_test'``.
        standardization_enabled / standardization_name: Reproduce the same
            scaler choice when refitting on test under
            ``standardization_mode='refit_on_test'``.
    """

    spec_name: str
    spec_type: str
    spec_kwargs: Dict[str, Any]
    training_token_position: str
    training_layer_indices: np.ndarray
    training_data_folder: str
    class_names: List[str]
    scalers: Dict[int, Optional[Any]]
    models: Dict[int, LogisticRegression]
    supervised_directions: Dict[str, np.ndarray]
    standardization_enabled: bool
    standardization_name: str


def save_static_pipeline_bundle(
    bundle: TrainedStaticPipelineBundle, path: Path
) -> Path:
    """Pickle ``bundle`` to ``path`` (creating parents). Returns ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_static_pipeline_bundle(path: Path) -> TrainedStaticPipelineBundle:
    """Inverse of :func:`save_static_pipeline_bundle`."""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, TrainedStaticPipelineBundle):
        raise TypeError(
            f"{path} did not deserialize to TrainedStaticPipelineBundle "
            f"(got {type(obj).__name__})."
        )
    return obj


def build_static_pipeline_bundle(
    sweep: LayerSweepResult,
    *,
    spec: FeatureSpec,
    training_token_position: str,
    training_data_folder: Path,
    standardization_enabled: bool,
    standardization_name: str,
) -> TrainedStaticPipelineBundle:
    """Pack a :class:`LayerSweepResult` into a serializable bundle."""
    scalers: Dict[int, Optional[Any]] = {}
    models: Dict[int, LogisticRegression] = {}
    for abs_layer, cv in sweep.per_layer.items():
        scalers[int(abs_layer)] = cv.full_scaler
        models[int(abs_layer)] = cv.full_model
    return TrainedStaticPipelineBundle(
        spec_name=spec.name,
        spec_type=spec.type,
        spec_kwargs=dict(spec.kwargs),
        training_token_position=training_token_position,
        training_layer_indices=np.asarray(sweep.layer_indices, dtype=np.int64).copy(),
        training_data_folder=str(training_data_folder),
        class_names=list(sweep.class_names),
        scalers=scalers,
        models=models,
        supervised_directions={
            name: np.asarray(arr, dtype=np.float32)
            for name, arr in sweep.supervised_directions.items()
        },
        standardization_enabled=bool(standardization_enabled),
        standardization_name=str(standardization_name),
    )


@dataclass
class TestLayerResult:
    """Outcome of applying a static pipeline at one layer to a test slab."""

    layer_idx: int
    test_accuracy: float
    confusion_matrix: np.ndarray
    class_names: List[str]
    classes_present: List[str]
    n_per_class: Dict[str, int]
    y_true: np.ndarray
    y_pred: np.ndarray
    X_test_scaled: np.ndarray
    column_names: List[str]


@dataclass
class TestSweepResult:
    """
    All-layer test evaluation for one ``(standardization_mode, direction_mode)``
    combination on one test cache.
    """

    spec_name: str
    spec_type: str
    standardization_mode: str
    direction_mode: str
    layer_indices: np.ndarray
    per_layer: Dict[int, TestLayerResult]
    class_names: List[str]
    classes_present: List[str]

    @property
    def accuracy_by_layer(self) -> np.ndarray:
        return np.asarray(
            [self.per_layer[int(l)].test_accuracy for l in self.layer_indices],
            dtype=np.float64,
        )

    @property
    def best_layer_abs(self) -> int:
        return int(self.layer_indices[int(np.argmax(self.accuracy_by_layer))])


def _validate_test_compat(
    bundle: TrainedStaticPipelineBundle, test_cls_traj: ClassTrajectories
) -> None:
    """Layer count + d_model checks so a stale bundle errors cleanly."""
    n_test_layers = test_cls_traj.layer_indices.size
    n_train_layers = bundle.training_layer_indices.size
    if n_test_layers != n_train_layers:
        raise ValueError(
            f"Test data has n_layers={n_test_layers} but bundle was trained on "
            f"n_layers={n_train_layers}. Cannot apply pipeline."
        )
    test_d_model = test_cls_traj.feature_data.shape[-1]
    for name, direction in bundle.supervised_directions.items():
        if direction.shape[0] != n_test_layers:
            raise ValueError(
                f"Direction {name!r} has n_layers={direction.shape[0]} but "
                f"test has {n_test_layers}."
            )
        if direction.shape[1] != test_d_model:
            raise ValueError(
                f"Direction {name!r} has d_model={direction.shape[1]} but "
                f"test has d_model={test_d_model}."
            )


def _refit_test_directions(
    bundle: TrainedStaticPipelineBundle,
    direction_fit_data: ClassTrajectories,
) -> Dict[str, np.ndarray]:
    """Refit every direction the bundle uses on ``direction_fit_data``."""
    out: Dict[str, np.ndarray] = {}
    for name in bundle.supervised_directions.keys():
        if name not in SUPERVISED_DESCRIPTOR_REGISTRY:
            raise KeyError(
                f"Direction {name!r} from bundle is not in "
                f"SUPERVISED_DESCRIPTOR_REGISTRY."
            )
        out[name] = SUPERVISED_DESCRIPTOR_REGISTRY[name].fit(
            direction_fit_data.feature_data,
            direction_fit_data.labels,
            direction_fit_data.gt_labels,
        )
    return out


def evaluate_static_pipeline_on_test(
    bundle: TrainedStaticPipelineBundle,
    test_cls_traj: ClassTrajectories,
    *,
    standardization_mode: str = "frozen",
    direction_mode: str = "frozen",
    direction_fit_data: Optional[ClassTrajectories] = None,
) -> TestSweepResult:
    """
    Apply ``bundle`` to ``test_cls_traj``, one prediction per layer.

    ``standardization_mode``:
        - ``"frozen"``: use the saved scaler in ``bundle.scalers[layer]``.
        - ``"refit_on_test"``: fit a fresh scaler (same class as training)
          on the test features at that layer, then apply it. The LR
          coefficients themselves are not retrained.

    ``direction_mode`` (projection specs only):
        - ``"frozen"``: project test activations onto
          ``bundle.supervised_directions``.
        - ``"refit_on_test"``: refit every direction on
          ``direction_fit_data`` (defaults to ``test_cls_traj``) and
          project test activations onto those new directions. Ignored for
          specs with no supervised directions (no-op).

    ``test_cls_traj`` is the set of rows to evaluate on — usually
    restricted to ``bundle.class_names``. ``direction_fit_data`` is the
    pool used to *fit* refit-mode directions; pass the full 3-class test
    slab to keep direction quality high when ``test_cls_traj`` is a
    subset, otherwise pass ``None`` and the subset itself is used.
    """
    if standardization_mode not in {"frozen", "refit_on_test"}:
        raise ValueError(
            f"standardization_mode={standardization_mode!r}; "
            "expected 'frozen' or 'refit_on_test'."
        )
    if direction_mode not in {"frozen", "refit_on_test"}:
        raise ValueError(
            f"direction_mode={direction_mode!r}; "
            "expected 'frozen' or 'refit_on_test'."
        )
    _validate_test_compat(bundle, test_cls_traj)

    builder = FEATURE_BUILDER_REGISTRY[bundle.spec_type]
    # Decide which direction set to project onto. Frozen + no-supervised =
    # empty dict; raw_activation builders ignore it regardless.
    if direction_mode == "refit_on_test" and bundle.supervised_directions:
        fit_pool = direction_fit_data if direction_fit_data is not None else test_cls_traj
        fit_dirs = _refit_test_directions(bundle, fit_pool)
    else:
        fit_dirs = dict(bundle.supervised_directions)

    train_label_indices = [CLASS_NAMES.index(c) for c in bundle.class_names]
    y_full = test_cls_traj.labels
    scored_mask = np.isin(y_full, train_label_indices)
    if not np.any(scored_mask):
        raise ValueError(
            f"No test rows match the training class set {bundle.class_names} "
            f"(test labels present: {sorted(set(y_full.tolist()))})."
        )

    per_layer: Dict[int, TestLayerResult] = {}
    for layer_pos, abs_layer in enumerate(test_cls_traj.layer_indices.tolist()):
        abs_layer = int(abs_layer)
        if abs_layer not in bundle.models:
            raise KeyError(
                f"Bundle has no model for absolute layer {abs_layer}; "
                f"known layers {sorted(bundle.models)}."
            )
        X_layer, col_names = builder(
            test_cls_traj, layer_pos, bundle.spec_kwargs, fit_dirs
        )
        if standardization_mode == "frozen":
            scaler = bundle.scalers.get(abs_layer)
        else:
            scaler = (
                _make_scaler(bundle.standardization_name)
                if bundle.standardization_enabled else None
            )
            if scaler is not None:
                scaler = scaler.fit(X_layer)
        X_scaled = scaler.transform(X_layer) if scaler is not None else X_layer

        model = bundle.models[abs_layer]
        if X_scaled.shape[1] != model.n_features_in_:
            raise ValueError(
                f"Layer {abs_layer}: test feature width {X_scaled.shape[1]} "
                f"does not match saved model n_features_in_="
                f"{model.n_features_in_}."
            )
        y_pred_full = model.predict(X_scaled)

        y_true_scored = y_full[scored_mask]
        y_pred_scored = y_pred_full[scored_mask]
        X_scored = X_scaled[scored_mask]
        acc = float(accuracy_score(y_true_scored, y_pred_scored))
        cm = confusion_matrix(
            y_true_scored, y_pred_scored, labels=train_label_indices
        ).astype(np.int64, copy=False)
        present = [
            c for c in bundle.class_names
            if int(np.sum(y_true_scored == CLASS_NAMES.index(c))) > 0
        ]
        n_per_class = {
            c: int(np.sum(y_true_scored == CLASS_NAMES.index(c)))
            for c in bundle.class_names
        }
        per_layer[abs_layer] = TestLayerResult(
            layer_idx=abs_layer,
            test_accuracy=acc,
            confusion_matrix=cm,
            class_names=list(bundle.class_names),
            classes_present=present,
            n_per_class=n_per_class,
            y_true=y_true_scored.astype(np.int64, copy=False),
            y_pred=y_pred_scored.astype(np.int64, copy=False),
            X_test_scaled=np.asarray(X_scored, dtype=np.float32),
            column_names=list(col_names),
        )

    classes_present_total = sorted(
        {c for cv in per_layer.values() for c in cv.classes_present},
        key=lambda c: CLASS_NAMES.index(c) if c in CLASS_NAMES else 999,
    )
    return TestSweepResult(
        spec_name=bundle.spec_name,
        spec_type=bundle.spec_type,
        standardization_mode=standardization_mode,
        direction_mode=direction_mode,
        layer_indices=np.asarray(test_cls_traj.layer_indices, dtype=np.int64).copy(),
        per_layer=per_layer,
        class_names=list(bundle.class_names),
        classes_present=classes_present_total,
    )


def compute_static_direction_cosines(
    bundle: TrainedStaticPipelineBundle,
    direction_fit_data: ClassTrajectories,
) -> Dict[str, np.ndarray]:
    """
    Per-direction, per-layer cosine similarity between the bundle's
    training-fit direction and a refit-on-``direction_fit_data`` direction.

    Returns ``{name: (n_layers,) float64}``. Empty dict for non-projection
    specs. Layers where either direction is degenerate (zero norm) get
    NaN so the plot shows a gap rather than a spurious zero.
    """
    if bundle.spec_type != "projection" or not bundle.supervised_directions:
        return {}
    _validate_test_compat(bundle, direction_fit_data)
    refit = _refit_test_directions(bundle, direction_fit_data)
    out: Dict[str, np.ndarray] = {}
    for name, D_train in bundle.supervised_directions.items():
        D_test = refit[name]
        num = np.sum(D_train * D_test, axis=-1)
        norms = (
            np.linalg.norm(D_train, axis=-1)
            * np.linalg.norm(D_test, axis=-1)
        )
        cos = np.full(num.shape, np.nan, dtype=np.float64)
        good = norms > 1e-8
        cos[good] = (num[good] / norms[good]).astype(np.float64)
        out[name] = cos
    return out


# --------------------------------------------------------------------------- #
# Test-time plotting                                                          #
# --------------------------------------------------------------------------- #


def _mode_label(std_mode: str, dir_mode: str, *, spec_type: str) -> str:
    """Compact legend label for a (std_mode, dir_mode) combination."""
    if spec_type == "projection":
        return f"std={std_mode} | dir={dir_mode}"
    return f"std={std_mode}"


def _mode_slug(std_mode: str, dir_mode: str, *, spec_type: str) -> str:
    """Filename-safe slug for one combination."""
    std_short = "frozen" if std_mode == "frozen" else "refit"
    dir_short = "frozen" if dir_mode == "frozen" else "refit"
    if spec_type == "projection":
        return f"std-{std_short}__dir-{dir_short}"
    return f"std-{std_short}"


def plot_test_accuracy_by_layer(
    train_sweep: LayerSweepResult,
    test_sweeps: Sequence[TestSweepResult],
    *,
    basepath: Path,
    formats: Sequence[str] = ("png",),
) -> List[Path]:
    """
    Overlay the training CV mean ± std curve with one accuracy curve per
    test ``(std_mode, dir_mode)`` combination. One figure per call (call
    once per spec).
    """
    layers_train = np.asarray(train_sweep.layer_indices)
    train_mean = train_sweep.mean_acc_by_layer
    train_std = train_sweep.std_acc_by_layer
    n_classes = len(train_sweep.class_names) or len(CLASS_NAMES)
    chance = 1.0 / max(1, n_classes)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(layers_train, train_mean, color="#4C72B0", linewidth=1.8,
            marker="o", markersize=4, label="train CV (mean ± std)")
    ax.fill_between(
        layers_train, train_mean - train_std, train_mean + train_std,
        color="#4C72B0", alpha=0.18,
    )

    palette = ["#DD8452", "#55A868", "#C44E52", "#8172B2"]
    for i, sweep in enumerate(test_sweeps):
        color = palette[i % len(palette)]
        accs = sweep.accuracy_by_layer
        layers_test = np.asarray(sweep.layer_indices)
        ax.plot(
            layers_test, accs, color=color, linewidth=1.4,
            marker="s", markersize=4,
            label=f"test [{_mode_label(sweep.standardization_mode, sweep.direction_mode, spec_type=sweep.spec_type)}]",
        )
    ax.axhline(
        chance, color="grey", linestyle="--", linewidth=1.0,
        label=f"chance ({chance:.2f})",
    )
    ax.set_xlabel("layer")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.suptitle(
        f"Train CV vs held-out test — {train_sweep.spec_name}",
        fontsize=11,
    )
    fig.tight_layout()
    return _save_figure(fig, basepath, formats)


def plot_test_confusion_matrix_at_layer(
    test_result: TestLayerResult,
    *,
    basepath: Path,
    title: Optional[str] = None,
    formats: Sequence[str] = ("png",),
    normalize: bool = True,
) -> List[Path]:
    """Single-panel test confusion matrix at one layer (row-normalized by default)."""
    cm = test_result.confusion_matrix.astype(np.float64)
    tick_labels = list(test_result.class_names) if test_result.class_names else list(CLASS_NAMES)
    k = len(tick_labels)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_show = cm / row_sums
        vmax = 1.0
    else:
        cm_show = cm
        vmax = cm.max() if cm.size else 1.0

    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    im = ax.imshow(cm_show, cmap="Blues", vmin=0.0, vmax=vmax)
    ax.set_xticks(np.arange(k))
    ax.set_yticks(np.arange(k))
    ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(tick_labels, fontsize=8)
    for i in range(k):
        for j in range(k):
            text = f"{cm_show[i, j]:.2f}" if normalize else f"{int(cm_show[i, j])}"
            ax.text(
                j, i, text,
                ha="center", va="center",
                color="white" if cm_show[i, j] > vmax * 0.6 else "black",
                fontsize=8,
            )
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        title or
        f"Test CM — L{test_result.layer_idx}  (acc={test_result.test_accuracy:.3f})",
        fontsize=10,
    )
    fig.tight_layout()
    return _save_figure(fig, basepath, formats)


def plot_direction_cosines_by_layer(
    cosines: Mapping[str, np.ndarray],
    layer_indices: Sequence[int],
    *,
    basepath: Path,
    title: str = "Direction cosine similarity (train vs test-refit)",
    formats: Sequence[str] = ("png",),
) -> List[Path]:
    """One line per direction across layers; ``1.0`` = identical axis."""
    if not cosines:
        return []
    layers = np.asarray(list(layer_indices), dtype=np.int64)
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
    for i, (name, arr) in enumerate(cosines.items()):
        ax.plot(
            layers, np.asarray(arr, dtype=np.float64),
            marker="o", markersize=4, linewidth=1.3,
            color=palette[i % len(palette)],
            label=name,
        )
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.8)
    ax.axhline(0.0, color="grey", linestyle=":", linewidth=0.8)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("layer")
    ax.set_ylabel("cosine similarity")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return _save_figure(fig, basepath, formats)
