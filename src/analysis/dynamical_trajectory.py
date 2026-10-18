"""
Dynamical-trajectory analysis: predict the microscopic class
(``honest_success`` / ``lie_success`` / ``lie_fail``) of each cached sample
from kinematic descriptors of its per-layer hidden-state trajectory.

The activations cached by :mod:`src.cache_activations` have shape
``(k, n_pos, n_layers, d_model)``. After slicing one token position, each
sample becomes a curve through ``R^{d_model}`` indexed by layer index ``l``.
The descriptors registered in this module turn that curve into a per-layer
scalar (e.g. velocity ``v_l``, curvature ``kappa_l``); concatenating ``K``
descriptors gives a feature vector of length ``L*K``.

There are two descriptor flavors:

- **Unsupervised** (:data:`DESCRIPTOR_REGISTRY`):
  ``Callable[[ndarray], ndarray]`` — pure functions of the trajectory.
  Examples: ``velocity``, ``arc_length_cum``, ``angle_curvature_cum``.

- **Supervised** (:data:`SUPERVISED_DESCRIPTOR_REGISTRY`): pair of ``fit``
  (consumes per-sample class + GT labels to compute a per-layer
  difference-in-means direction) and ``transform`` (projects the
  trajectory onto that direction). Examples:
  ``proj_2_lie-honest_direction``, ``proj_2_lie_success-fail_direction``,
  ``proj_2_truthful_direction_cum``.
  These are refit per training fold inside :func:`run_cv` so the
  test-fold rows are projected onto an axis that never saw them.

Pipeline (one position per call):

1. :func:`load_class_trajectories` — read pickles from one cache folder,
   collapse GT_0+GT_1 into the three target classes (tracking per-sample
   GT separately for supervised descriptors), slice to the chosen token
   position and layer range.
2. :func:`compute_feature_map` — apply every descriptor (in-sample fit
   for supervised), optionally smooth along the layer axis (1D Gaussian),
   and return the feature matrix ``(n_samples, n_layers, n_descriptors)``
   plus the per-(layer, descriptor) raw curves (kept for the mean-curve
   plot).
3. :func:`build_predictor_matrix` — flatten one subset of descriptors
   into ``X = (n_samples, n_layers*n_subset)`` with column names
   ``<desc>_L<layer>``. Accepts ``column_overrides`` so :func:`run_cv`
   can inject per-fold supervised refits.
4. :func:`run_cv` — KFold or ShuffleSplit. On each fold: refit supervised
   DiMs on the training split, build the fold predictor matrix, fit a
   fresh scaler + logistic regression. Returns per-fold accuracy, the
   summed confusion matrix, and a full-data refit (used for the
   coefficient heatmap and the PCA scatter; the latter inherits the
   in-sample supervised fit, which is acceptable for visualization).
5. ``plot_*`` helpers — five matplotlib figures matching the script's
   visualization spec.

Add new unsupervised descriptors by registering a ``f(traj) -> (n, L)``
function in :data:`DESCRIPTOR_REGISTRY`. Add new supervised descriptors
by adding a :class:`SupervisedDescriptorSpec` (``fit`` + ``transform``)
to :data:`SUPERVISED_DESCRIPTOR_REGISTRY`. The config picks them up by
name.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import RobustScaler, StandardScaler

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers projection='3d')


# --------------------------------------------------------------------------- #
# Class definitions                                                           #
# --------------------------------------------------------------------------- #


# Order matters: it drives the row/column ordering in confusion matrices and
# the class indexing inside the logistic regression's ``classes_`` array.
CLASS_NAMES: Tuple[str, ...] = ("honest_success", "lie_success", "lie_fail")


# Each entry maps a class to the ``(condition, fail)`` pair used to locate the
# pickles written by the cache stage. GT_0 + GT_1 are pooled.
_CLASS_SOURCES: Dict[str, Tuple[str, bool]] = {
    "honest_success": ("control", False),
    "lie_success":    ("experiment", False),
    "lie_fail":       ("experiment", True),
}


# --------------------------------------------------------------------------- #
# Descriptor registry                                                         #
# --------------------------------------------------------------------------- #


DescriptorFn = Callable[..., np.ndarray]

# Numerical floor used in denominators of cosine / kinematic curvature so a
# degenerate (zero-norm) displacement maps to zero instead of NaN.
_DESCRIPTOR_EPS: float = 1e-8


def _place_with_edges(values: np.ndarray, target_len: int, valid_start: int) -> np.ndarray:
    """
    Pad ``values`` (per-sample row of length ``n``, defined at layer indices
    ``[valid_start, valid_start + n - 1]``) up to length ``target_len`` by
    repeating the nearest valid value at each missing index. Both ends are
    handled independently so a descriptor that's only defined for
    ``l in [2, L-2]`` (jerk) gets two left-pad copies and one right-pad
    copy.

    When ``values`` is entirely empty (sub-3-layer slabs), returns zeros so
    the rest of the pipeline keeps the expected ``(n_samples, n_layers)``
    contract.
    """
    n_samples, n = values.shape[0], values.shape[1]
    if n == 0:
        return np.zeros((n_samples, target_len), dtype=values.dtype)
    parts: List[np.ndarray] = []
    left_pad = max(0, int(valid_start))
    if left_pad:
        parts.append(np.repeat(values[:, :1], left_pad, axis=1))
    parts.append(values)
    right_pad = target_len - left_pad - n
    if right_pad > 0:
        parts.append(np.repeat(values[:, -1:], right_pad, axis=1))
    out = np.concatenate(parts, axis=1)
    # Final guard: clamp to target length if rounding produced a longer array
    # (only possible when callers pass inconsistent inputs).
    return out[:, :target_len].astype(np.float32, copy=False)


def _check_traj(traj: np.ndarray, name: str) -> None:
    if traj.ndim != 3:
        raise ValueError(
            f"{name} expects (n_samples, n_layers, d_model); got {traj.shape}"
        )


def velocity(traj: np.ndarray) -> np.ndarray:
    """
    Scalar velocity ``v_l := ||Δh_l||_2`` with ``Δh_l = h_{l+1} - h_l``.

    ``v_l`` is well-defined for ``l in [0, L-2]``; the last position is
    padded by repeating ``v_{L-2}`` so the output length equals
    ``n_layers``.

    Args:
        traj: ``(n_samples, n_layers, d_model)``.

    Returns:
        ``(n_samples, n_layers)``.
    """
    _check_traj(traj, "velocity")
    L = traj.shape[1]
    if L < 2:
        return np.zeros((traj.shape[0], L), dtype=np.float32)
    speed = np.linalg.norm(np.diff(traj, axis=1), axis=-1)  # (n, L-1) for l in [0, L-2]
    return _place_with_edges(speed, L, valid_start=0)


def acceleration(traj: np.ndarray) -> np.ndarray:
    """
    Scalar acceleration ``a_l := v_l - v_{l-1}`` (gradient of scalar
    velocity).

    Defined for ``l in [1, L-2]``; ``l = 0`` and ``l = L-1`` are padded by
    repeating the nearest valid value.
    """
    _check_traj(traj, "acceleration")
    L = traj.shape[1]
    if L < 3:
        return np.zeros((traj.shape[0], L), dtype=np.float32)
    v_raw = np.linalg.norm(np.diff(traj, axis=1), axis=-1)  # (n, L-1)
    a = np.diff(v_raw, axis=1)  # (n, L-2) for l in [1, L-2]
    return _place_with_edges(a, L, valid_start=1)


def jerk(traj: np.ndarray) -> np.ndarray:
    """
    Scalar jerk ``j_l := a_l - a_{l-1}`` (gradient of scalar acceleration).

    Defined for ``l in [2, L-2]``; the boundary positions are padded by
    repeating the nearest valid value.
    """
    _check_traj(traj, "jerk")
    L = traj.shape[1]
    if L < 4:
        return np.zeros((traj.shape[0], L), dtype=np.float32)
    v_raw = np.linalg.norm(np.diff(traj, axis=1), axis=-1)  # (n, L-1)
    a_raw = np.diff(v_raw, axis=1)                          # (n, L-2)
    j = np.diff(a_raw, axis=1)                              # (n, L-3) for l in [2, L-2]
    return _place_with_edges(j, L, valid_start=2)


def directional_curvature(traj: np.ndarray) -> np.ndarray:
    """
    Directional curvature ``κ_l := <Δh_l, Δh_{l-1}> / (||Δh_l|| ||Δh_{l-1}||)``
    — cosine similarity between consecutive displacements. Defined for
    ``l in [1, L-2]``; boundary positions padded by repeating the nearest
    valid value.

    Note: this returns a cosine in ``[-1, 1]``. As written, higher values
    mean displacements are *aligned* (small directional change). If you
    want "amount of directional change" use ``1 - κ_l`` downstream.
    """
    _check_traj(traj, "directional_curvature")
    L = traj.shape[1]
    if L < 3:
        return np.zeros((traj.shape[0], L), dtype=np.float32)
    delta = np.diff(traj, axis=1)  # (n, L-1, D) for l in [0, L-2]
    delta_l = delta[:, 1:, :]      # (n, L-2, D) for l in [1, L-2]
    delta_lm1 = delta[:, :-1, :]   # (n, L-2, D) for l in [1, L-2]
    dot = (delta_l * delta_lm1).sum(axis=-1)                  # (n, L-2)
    norms = np.linalg.norm(delta, axis=-1)                     # (n, L-1)
    denom = norms[:, 1:] * norms[:, :-1]                       # (n, L-2)
    safe = np.where(denom < _DESCRIPTOR_EPS, 1.0, denom)
    cos = dot / safe
    cos = np.where(denom < _DESCRIPTOR_EPS, 0.0, cos)
    return _place_with_edges(cos, L, valid_start=1)


def kinematic_curvature(traj: np.ndarray) -> np.ndarray:
    """
    Kinematic curvature ``κ^kin_l := ||a_l|| / ||v_l||^2``, where
    ``v_l = Δh_l = h_{l+1} - h_l`` is the *vector* velocity and
    ``a_l = h_{l+1} - 2 h_l + h_{l-1}`` is the *vector* acceleration.

    Distinct from the scalar :func:`acceleration` descriptor (which is the
    gradient of *scalar* speed). Defined for ``l in [1, L-2]``; boundary
    positions padded by repeating the nearest valid value.
    """
    _check_traj(traj, "kinematic_curvature")
    L = traj.shape[1]
    if L < 3:
        return np.zeros((traj.shape[0], L), dtype=np.float32)
    delta = np.diff(traj, axis=1)               # (n, L-1, D)  v_l for l in [0, L-2]
    a_vec = np.diff(delta, axis=1)              # (n, L-2, D)  a_l for l in [1, L-2]
    a_norm = np.linalg.norm(a_vec, axis=-1)     # (n, L-2)
    v_norm = np.linalg.norm(delta[:, 1:, :], axis=-1)  # (n, L-2)  ||v_l|| for l in [1, L-2]
    denom = v_norm ** 2
    safe = np.where(denom < _DESCRIPTOR_EPS, 1.0, denom)
    kappa = a_norm / safe
    kappa = np.where(denom < _DESCRIPTOR_EPS, 0.0, kappa)
    return _place_with_edges(kappa, L, valid_start=1)


def vector_angular_curvature(traj: np.ndarray) -> np.ndarray:
    """
    Vector angular curvature ``c_L := arccos(<v_{L+1}, v_L> / (||v_{L+1}|| ||v_L||))``
    where ``v_L = x_L - x_{L-1}`` is the residual-stream update added by
    layer ``L``. Output is in **degrees**, range ``[0, 180]``.

    Layout (length matches ``n_layers``):

    - Position ``0``: ``NaN`` — this descriptor needs ``v_L`` and
      ``v_{L+1}``, so the very first layer has no defined value. The
      analysis pipeline drops this column from the predictor matrix and
      renders the corresponding heatmap cell blank.
    - Positions ``1 .. L-2``: arccos value (``L-2`` total).
    - Position ``L-1``: padded by repeating the last valid value so the
      descriptor exposes ``L-1`` non-NaN values overall (matches the PDF
      note "for a model with n layers, this metric only has n-1 values
      per example").
    """
    _check_traj(traj, "vector_angular_curvature")
    n_samples, L = traj.shape[0], traj.shape[1]
    out = np.full((n_samples, L), np.nan, dtype=np.float32)
    if L < 3:
        return out
    delta = np.diff(traj, axis=1)              # (n, L-1, D); delta[l] = x_{l+1} - x_l
    # User's v_L = delta[L-1], v_{L+1} = delta[L]; both required so L in [1, L-2].
    v_Lm1 = delta[:, :-1, :]                   # (n, L-2, D)
    v_L = delta[:, 1:, :]                      # (n, L-2, D)
    dot = (v_Lm1 * v_L).sum(axis=-1)           # (n, L-2)
    n_a = np.linalg.norm(v_Lm1, axis=-1)
    n_b = np.linalg.norm(v_L, axis=-1)
    denom = n_a * n_b
    safe = np.where(denom < _DESCRIPTOR_EPS, 1.0, denom)
    cos_theta = np.clip(dot / safe, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_theta)).astype(np.float32, copy=False)
    angle_deg = np.where(denom < _DESCRIPTOR_EPS, np.nan, angle_deg)
    out[:, 1:L - 1] = angle_deg                # positions [1, L-2]
    out[:, L - 1] = angle_deg[:, -1]           # right-pad last valid
    return out


def _per_layer_angular_curvature(traj: np.ndarray) -> np.ndarray:
    """
    Internal helper: per-layer angular curvature (degrees) with degenerate
    segments mapped to ``0`` rather than ``NaN``. Output shape
    ``(n_samples, L-2)`` covering layers ``[1, L-2]``. Used by the cumulative
    and sum reductions, where ``NaN`` would poison the aggregate.
    """
    delta = np.diff(traj, axis=1)
    v_Lm1 = delta[:, :-1, :]
    v_L = delta[:, 1:, :]
    dot = (v_Lm1 * v_L).sum(axis=-1)
    n_a = np.linalg.norm(v_Lm1, axis=-1)
    n_b = np.linalg.norm(v_L, axis=-1)
    denom = n_a * n_b
    safe = np.where(denom < _DESCRIPTOR_EPS, 1.0, denom)
    cos_theta = np.clip(dot / safe, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_theta)).astype(np.float32, copy=False)
    return np.where(denom < _DESCRIPTOR_EPS, 0.0, angle_deg)


def angle_curvature_cum(traj: np.ndarray) -> np.ndarray:
    """
    Per-layer cumulative vector angular curvature
    ``S_l := Σ_{i=1}^{l} c_i``, where ``c_i`` is the
    :func:`vector_angular_curvature` value at layer ``i`` (degrees,
    ``[0, 180]``).

    Layout (length matches ``n_layers``), all positions are finite:

    - ``S_0 = 0`` (no angle change accumulated yet).
    - ``S_l`` for ``l in [1, L-2]``: running sum of the ``L-2``
      well-defined ``c_i`` values.
    - ``S_{L-1}``: same as ``S_{L-2}`` (the global cumulative angle —
      there is no ``c_{L-1}`` to add).

    Degenerate segments (zero-norm displacements) contribute ``0`` rather
    than ``NaN`` so the running sum stays defined at every layer.
    """
    _check_traj(traj, "angle_curvature_cum")
    n_samples, L = traj.shape[0], traj.shape[1]
    out = np.zeros((n_samples, L), dtype=np.float32)
    if L < 3:
        return out
    angle_deg = _per_layer_angular_curvature(traj)
    cum = np.cumsum(angle_deg, axis=1)  # (n, L-2) -> S_1..S_{L-2}
    out[:, 1:L - 1] = cum
    out[:, L - 1] = cum[:, -1]
    return out


def angular_curvature_sum(traj: np.ndarray) -> np.ndarray:
    """
    Global angular curvature sum ``S := Σ_{i=1}^{L-2} c_i``, where ``c_i``
    is :func:`vector_angular_curvature` (degrees, ``[0, 180]``).

    The PDF defines this as a single scalar per trajectory; to fit the
    ``(n_samples, n_layers)`` contract this descriptor broadcasts that
    scalar across every layer column (every layer-tagged feature is
    therefore identical). Equivalent to the final entry of
    :func:`angle_curvature_cum` repeated across all layers. Use
    :func:`angle_curvature_cum` instead when you want a per-layer signal.
    """
    _check_traj(traj, "angular_curvature_sum")
    n_samples, L = traj.shape[0], traj.shape[1]
    if L < 3:
        return np.zeros((n_samples, L), dtype=np.float32)
    angle_deg = _per_layer_angular_curvature(traj)
    total = angle_deg.sum(axis=1, keepdims=True)  # (n, 1)
    return np.broadcast_to(total, (n_samples, L)).astype(np.float32, copy=True)


def arc_length(traj: np.ndarray) -> np.ndarray:
    """
    Global arc length ``S := Σ_{l=0}^{L-2} ||h_{l+1} - h_l||_2``.

    The PDF defines arc length as a single scalar per trajectory. To fit
    the ``(n_samples, n_layers)`` contract this descriptor *broadcasts*
    that scalar across every layer column — every layer-tagged feature is
    therefore identical. Use :func:`arc_length_cum` instead when you want
    a per-layer signal (its final entry equals the global ``S``).
    """
    _check_traj(traj, "arc_length")
    L = traj.shape[1]
    if L < 2:
        return np.zeros((traj.shape[0], L), dtype=np.float32)
    seg = np.linalg.norm(np.diff(traj, axis=1), axis=-1)  # (n, L-1)
    total = seg.sum(axis=1, keepdims=True)                # (n, 1)
    return np.broadcast_to(total, (total.shape[0], L)).astype(np.float32, copy=True)


def arc_length_cum(traj: np.ndarray) -> np.ndarray:
    """
    Cumulative arc length ``S_l := Σ_{i=0}^{l-1} ||h_{i+1} - h_i||_2`` with
    ``S_0 = 0``. ``S_{L-1}`` equals the global arc length :func:`arc_length`
    returns. Per-layer, monotonically non-decreasing.
    """
    _check_traj(traj, "arc_length_cum")
    L = traj.shape[1]
    if L < 2:
        return np.zeros((traj.shape[0], L), dtype=np.float32)
    seg = np.linalg.norm(np.diff(traj, axis=1), axis=-1)  # (n, L-1)
    cum = np.cumsum(seg, axis=1)                          # (n, L-1) -> S_1..S_{L-1}
    zero = np.zeros((traj.shape[0], 1), dtype=cum.dtype)
    return np.concatenate([zero, cum], axis=1).astype(np.float32, copy=False)


# Registry of named unsupervised descriptors. Each entry's signature is
# ``fn(traj: (n, L, D), **kwargs) -> (n, L)`` and depends only on the
# trajectory (no class/GT labels needed). Definitions follow the PDF:
#   1. velocity              v_l = ||Δh_l||
#   2. acceleration          a_l = v_l - v_{l-1}        (scalar)
#   3. jerk                  j_l = a_l - a_{l-1}        (scalar)
#   4. directional_curvature κ_l = cos(Δh_l, Δh_{l-1})
#   5. kinematic_curvature   κ^kin_l = ||a_l|| / ||v_l||^2  (vector a, v)
#   6. vector_angular_curvature  c_l = arccos(cos)  (degrees)
#   7. angle_curvature_cum   S_l = Σ_{i<=l} c_i   (per-layer)
#   8. angular_curvature_sum S = Σ_l c_l          (global, broadcast)
#   9. arc_length            S    (global, broadcast across layers)
#  10. arc_length_cum        S_l = Σ_{i<l} ||Δh_i||      (per-layer)
# Supervised descriptors (projection onto DiM vectors) live in
# :data:`SUPERVISED_DESCRIPTOR_REGISTRY` below.
DESCRIPTOR_REGISTRY: Dict[str, DescriptorFn] = {
    "velocity":                   velocity,
    "acceleration":               acceleration,
    "jerk":                       jerk,
    "directional_curvature":      directional_curvature,
    "kinematic_curvature":        kinematic_curvature,
    "vector_angular_curvature":   vector_angular_curvature,
    "angle_curvature_cum":        angle_curvature_cum,
    "angular_curvature_sum":      angular_curvature_sum,
    "arc_length":                 arc_length,
    "arc_length_cum":             arc_length_cum,
}


# --------------------------------------------------------------------------- #
# Supervised descriptor registry                                              #
# --------------------------------------------------------------------------- #
#
# These descriptors need per-class label information (``class_labels`` and/or
# ``gt_labels``) to fit a per-layer direction vector from training data, then
# project arbitrary trajectories onto it. The fit/transform split is what lets
# :func:`run_cv` refit the direction on the training fold each iteration so
# the CV accuracy is not leaked-through.
#
# Direction-fit functions return ``direction: (L, D)`` unit-normalized per
# layer. Transform functions take ``(traj, direction) -> (n, L)``.

# Class indices into :data:`CLASS_NAMES`.
_HONEST_SUCCESS = 0
_LIE_SUCCESS = 1
_LIE_FAIL = 2


def _fit_unit_dim(
    traj: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """
    Per-layer unit-normalized Difference-in-Means
    ``D_l = unit(mean(traj[mask_a, l]) - mean(traj[mask_b, l]))``.

    Args:
        traj: ``(n_samples, n_layers, d_model)`` — typically the *training*
            slab, since the direction is what gets refit per fold.
        mask_a, mask_b: Boolean masks selecting the two groups.
        name: Used for error messages.

    Returns:
        ``(n_layers, d_model)`` unit vector per layer. Layers where one
        side is empty raise; layers where the difference is degenerate
        (zero-norm) get a zero vector — projections onto them will be 0.
    """
    if not np.any(mask_a) or not np.any(mask_b):
        raise ValueError(
            f"{name}: at least one of the DiM groups is empty (|a|={int(mask_a.sum())}, "
            f"|b|={int(mask_b.sum())}). Need samples in both groups to fit."
        )
    mean_a = traj[mask_a].mean(axis=0)   # (L, D)
    mean_b = traj[mask_b].mean(axis=0)   # (L, D)
    diff = (mean_a - mean_b).astype(np.float32, copy=False)
    norms = np.linalg.norm(diff, axis=-1, keepdims=True)
    safe = np.where(norms < _DESCRIPTOR_EPS, 1.0, norms)
    unit = diff / safe
    unit = np.where(norms < _DESCRIPTOR_EPS, 0.0, unit)
    return unit.astype(np.float32, copy=False)


def fit_lie_honest_dim(
    traj: np.ndarray,
    class_labels: np.ndarray,
    gt_labels: np.ndarray,  # noqa: ARG001 — unused; kept for uniform interface
) -> np.ndarray:
    """Per-layer DiM = unit(mean(lie_success) - mean(honest_success))."""
    return _fit_unit_dim(
        traj,
        mask_a=(class_labels == _LIE_SUCCESS),
        mask_b=(class_labels == _HONEST_SUCCESS),
        name="lie-honest direction",
    )


def fit_lie_success_fail_dim(
    traj: np.ndarray,
    class_labels: np.ndarray,
    gt_labels: np.ndarray,  # noqa: ARG001 — unused; kept for uniform interface
) -> np.ndarray:
    """Per-layer DiM = unit(mean(lie_success) - mean(lie_fail))."""
    return _fit_unit_dim(
        traj,
        mask_a=(class_labels == _LIE_SUCCESS),
        mask_b=(class_labels == _LIE_FAIL),
        name="lie_success-fail direction",
    )


def fit_truthful_dim(
    traj: np.ndarray,
    class_labels: np.ndarray,
    gt_labels: np.ndarray,
) -> np.ndarray:
    """
    Per-layer DiM = unit(mean(honest_success, GT_1) - mean(honest_success, GT_0)).

    The PDF specifies the truthful axis as ``honest-GT_1 - honest-GT_0``;
    we read "honest" as ``honest_success`` (the same class used as the
    honest side of the lie-honest axis), which keeps the two directions
    fit on the same population of samples.
    """
    honest = (class_labels == _HONEST_SUCCESS)
    return _fit_unit_dim(
        traj,
        mask_a=honest & (gt_labels == 1),
        mask_b=honest & (gt_labels == 0),
        name="truthful direction",
    )


def project_activations(traj: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """``out[n, l] = <h_l, D_l>`` — per-layer projection of the raw activations."""
    if traj.shape[1:] != direction.shape:
        raise ValueError(
            f"project_activations: traj layers/dim {traj.shape[1:]} "
            f"does not match direction shape {direction.shape}."
        )
    return np.einsum("nld,ld->nl", traj, direction).astype(np.float32, copy=False)


def project_updates(traj: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """
    ``out[n, l] = <Δh_l, D_l>`` for ``l in [0, L-2]`` with ``Δh_l = h_{l+1} - h_l``.

    ``D_l`` is the direction at the *starting* layer of the update; this
    matches the indexing convention used by :func:`project_cum` so the
    cumulative variant is exactly the running sum of this descriptor.
    Position ``L-1`` is padded by repeating the last valid value.
    """
    if traj.shape[1:] != direction.shape:
        raise ValueError(
            f"project_updates: traj layers/dim {traj.shape[1:]} "
            f"does not match direction shape {direction.shape}."
        )
    n, L, _ = traj.shape
    if L < 2:
        return np.zeros((n, L), dtype=np.float32)
    delta = np.diff(traj, axis=1)               # (n, L-1, D)
    proj = np.einsum("nld,ld->nl", delta, direction[: L - 1])  # (n, L-1)
    out = np.empty((n, L), dtype=np.float32)
    out[:, : L - 1] = proj.astype(np.float32, copy=False)
    out[:, L - 1] = proj[:, -1]                  # pad
    return out


def project_cum(traj: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """
    Cumulative update projection
    ``Path_l = Σ_{i=0}^{l-1} <Δh_i, D_i>`` with ``Path_0 = 0``. Matches the
    PDF formula ``Path_ℓ = Σ_{i=1}^{ℓ} (Δh_i · D)`` under a 0-indexed
    convention; the final entry ``Path_{L-1}`` is the global path length
    along the direction axis.
    """
    if traj.shape[1:] != direction.shape:
        raise ValueError(
            f"project_cum: traj layers/dim {traj.shape[1:]} "
            f"does not match direction shape {direction.shape}."
        )
    n, L, _ = traj.shape
    if L < 2:
        return np.zeros((n, L), dtype=np.float32)
    delta = np.diff(traj, axis=1)
    proj = np.einsum("nld,ld->nl", delta, direction[: L - 1])  # (n, L-1)
    cum = np.cumsum(proj, axis=1)
    out = np.zeros((n, L), dtype=np.float32)
    out[:, 1:L] = cum.astype(np.float32, copy=False)
    return out


def project_sum(traj: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """
    Sum of per-layer activation projections ``S = Σ_l <h_l, D_l>``,
    broadcast across every layer column (mirrors the
    :func:`arc_length` convention). Distinct from :func:`project_cum`,
    which sums *update* projections; this one sums *activation*
    projections.
    """
    if traj.shape[1:] != direction.shape:
        raise ValueError(
            f"project_sum: traj layers/dim {traj.shape[1:]} "
            f"does not match direction shape {direction.shape}."
        )
    n, L, _ = traj.shape
    proj = np.einsum("nld,ld->nl", traj, direction)  # (n, L)
    total = proj.sum(axis=1, keepdims=True)
    return np.broadcast_to(total, (n, L)).astype(np.float32, copy=True)


# Each entry pairs a direction-fit function with a transform function.
# fit:       (traj_train, class_labels_train, gt_labels_train) -> direction (L, D)
# transform: (traj, direction)                                  -> (n, L)
@dataclass
class SupervisedDescriptorSpec:
    fit: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    transform: Callable[[np.ndarray, np.ndarray], np.ndarray]


SUPERVISED_DESCRIPTOR_REGISTRY: Dict[str, SupervisedDescriptorSpec] = {
    # Lie-honest direction (uses behavior labels: lie_success vs honest_success).
    # Inside :func:`run_cv` this is refit per training fold; the in-sample fit
    # produced by :func:`compute_feature_map` is *only* used for visualization
    # plots (mean curves / PCA) and is not used by the CV accuracy numbers.
    "proj_2_lie-honest_direction":         SupervisedDescriptorSpec(fit_lie_honest_dim, project_activations),
    "proj_2_lie-honest_direction_updates": SupervisedDescriptorSpec(fit_lie_honest_dim, project_updates),
    "proj_2_lie-honest_direction_cum":     SupervisedDescriptorSpec(fit_lie_honest_dim, project_cum),
    "proj_2_lie-honest_direction_sum":     SupervisedDescriptorSpec(fit_lie_honest_dim, project_sum),
    # Lie-success-vs-fail direction (uses behavior labels: lie_success vs lie_fail).
    # Fit pool follows the same convention as the lie-honest axis.
    "proj_2_lie_success-fail_direction":         SupervisedDescriptorSpec(fit_lie_success_fail_dim, project_activations),
    "proj_2_lie_success-fail_direction_updates": SupervisedDescriptorSpec(fit_lie_success_fail_dim, project_updates),
    "proj_2_lie_success-fail_direction_cum":     SupervisedDescriptorSpec(fit_lie_success_fail_dim, project_cum),
    "proj_2_lie_success-fail_direction_sum":     SupervisedDescriptorSpec(fit_lie_success_fail_dim, project_sum),
    # Truthful direction (uses GT labels within honest_success; orthogonal to
    # the behavior-class prediction target, so in-sample fit has no leakage —
    # but we refit per-fold anyway for symmetry with the lie-honest axis).
    "proj_2_truthful_direction":           SupervisedDescriptorSpec(fit_truthful_dim, project_activations),
    "proj_2_truthful_direction_updates":   SupervisedDescriptorSpec(fit_truthful_dim, project_updates),
    "proj_2_truthful_direction_cum":       SupervisedDescriptorSpec(fit_truthful_dim, project_cum),
    "proj_2_truthful_direction_sum":       SupervisedDescriptorSpec(fit_truthful_dim, project_sum),
}


def is_supervised_descriptor(name: str) -> bool:
    """True iff ``name`` is in :data:`SUPERVISED_DESCRIPTOR_REGISTRY`."""
    return name in SUPERVISED_DESCRIPTOR_REGISTRY


def all_known_descriptors() -> List[str]:
    """Union of unsupervised and supervised registry keys (sorted)."""
    return sorted(set(DESCRIPTOR_REGISTRY) | set(SUPERVISED_DESCRIPTOR_REGISTRY))


# --------------------------------------------------------------------------- #
# Data loading                                                                #
# --------------------------------------------------------------------------- #


def _discover_activation_pickle(
    activations_dir: Path,
    condition: str,
    gt_label: int,
    fail: bool,
) -> Optional[Path]:
    """
    Locate ``activations_{condition}[_<method>]_GT_{gt}[_fail].pkl`` inside
    ``activations_dir``. Mirrors the loader in :mod:`src.analysis.microscopic`
    so the two analyses agree on which pickles belong to which class.
    """
    suffix = "_fail" if fail else ""
    patterns = [
        f"activations_{condition}_GT_{gt_label}{suffix}.pkl",
        f"activations_{condition}_*_GT_{gt_label}{suffix}.pkl",
    ]
    seen: List[Path] = []
    for pat in patterns:
        for p in activations_dir.glob(pat):
            if not fail and p.name.endswith("_fail.pkl"):
                continue
            if p not in seen:
                seen.append(p)
    if not seen:
        return None
    seen.sort(key=lambda p: (-len(p.name), p.name))
    return seen[0]


def _load_pickle_as_numpy(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if hasattr(obj, "numpy"):
        return obj.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(obj, dtype=np.float32)


def _resolve_top_token_position(
    data_folder: Path,
    rotation_metric: str,
) -> Optional[str]:
    """
    Read ``ranked_token_pos_<rotation_metric>`` from
    ``data_folder/stage_stats/rotation_stats.json`` and return its top entry.
    Returns ``None`` if the file or the ranking is missing.
    """
    stats_path = data_folder / "stage_stats" / "rotation_stats.json"
    if not stats_path.exists():
        return None
    try:
        with open(stats_path, encoding="utf-8") as f:
            stats = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    ranked = stats.get(f"ranked_token_pos_{rotation_metric}")
    if not ranked:
        return None
    return str(ranked[0])


def _position_labels(n_pre: int, n_post: int) -> List[str]:
    """
    Reproduce the position label scheme from
    :func:`src.cache_activations.hooks.position_labels` without importing
    that module (this analysis runs from the cached pickles only).

    The cache layout is ``[prompt-N, ..., prompt-1, prompt, gen+1, ..., gen+M]``
    with ``N == n_pre_prompt_tokens`` and ``M == n_generated_tokens``.
    """
    pre = [f"prompt-{i}" for i in range(n_pre, 0, -1)]
    post = [f"gen+{i}" for i in range(1, n_post + 1)]
    return pre + ["prompt"] + post


def _infer_position_labels(activations_dir: Path) -> List[str]:
    """
    Derive position labels by looking at the seed-folder name
    ``<seed>_pre<N>_gen<M>``. Falls back to numeric labels if the parent
    directory name does not match the expected pattern.
    """
    seed_segment = activations_dir.parent.name
    parts = seed_segment.split("_")
    n_pre = n_post = None
    for token in parts:
        if token.startswith("pre"):
            try:
                n_pre = int(token[3:])
            except ValueError:
                pass
        elif token.startswith("gen"):
            try:
                n_post = int(token[3:])
            except ValueError:
                pass
    if n_pre is None or n_post is None:
        # Probe an activation pickle for the n_pos dimension and label
        # positions numerically — keeps the analysis runnable on folders
        # whose name has drifted.
        for cond, gt, fail in (
            ("control", 0, False),
            ("control", 1, False),
            ("experiment", 0, False),
            ("experiment", 1, False),
        ):
            p = _discover_activation_pickle(activations_dir, cond, gt, fail)
            if p is not None:
                n_pos = _load_pickle_as_numpy(p).shape[1]
                return [f"pos_{i}" for i in range(n_pos)]
        raise FileNotFoundError(
            f"No activation pickles under {activations_dir} to infer position layout."
        )
    return _position_labels(n_pre, n_post)


@dataclass
class ClassTrajectories:
    """
    Per-class trajectories sliced to one token position and one layer range.

    Attributes:
        trajectories: ``{class_name: (n_samples, n_layers, d_model)}``.
        labels: 1D integer array indexing :data:`CLASS_NAMES`. Concatenation
            order matches :data:`CLASS_NAMES`.
        gt_labels: 1D integer array (values in ``{0, 1}``) of per-sample
            ground-truth labels, aligned with ``labels`` / ``feature_data``.
            Used by supervised descriptors that fit on the truthful axis
            (``honest_GT_1 - honest_GT_0``).
        feature_data: ``(n_total_samples, n_layers, d_model)`` stack of the
            trajectories in label order. Equivalent to
            ``np.concatenate([trajectories[c] for c in CLASS_NAMES])`` but
            kept here so callers don't recompute it.
        token_position: Token-position label used for slicing.
        layer_indices: Absolute layer indices used (after the layer-range
            slice). Length matches the second dim of ``feature_data``.
    """

    trajectories: Dict[str, np.ndarray]
    labels: np.ndarray
    gt_labels: np.ndarray
    feature_data: np.ndarray
    token_position: str
    layer_indices: np.ndarray


def load_class_trajectories(
    data_folder: Path,
    token_position: str,
    *,
    layer_start: int = -1,
    layer_end: int = -1,
    n_samples_per_class: int = -1,
    rng: Optional[np.random.Generator] = None,
) -> ClassTrajectories:
    """
    Load every cached pickle in ``data_folder/activations/``, pool GT_0 +
    GT_1 within each of the three classes, and slice to ``token_position``
    and ``[layer_start, layer_end]``.

    Classes whose pickles are entirely missing are silently dropped (e.g.
    ``lie_fail`` is only present when the cache stage saved failures).

    Args:
        data_folder: Cache seed folder (the parent of ``activations/``).
        token_position: Label such as ``"prompt-1"`` or ``"prompt"``.
        layer_start: Inclusive lower bound; ``-1`` = 0.
        layer_end: Inclusive upper bound; ``-1`` = last layer.
        n_samples_per_class: Cap per class after pooling GT_0+GT_1
            (``-1`` keeps all).
        rng: Optional NumPy generator for the per-class subsample.

    Returns:
        :class:`ClassTrajectories` with the slice resolved.
    """
    activations_dir = data_folder / "activations"
    if not activations_dir.exists():
        raise FileNotFoundError(f"{activations_dir} does not exist.")
    pos_labels = _infer_position_labels(activations_dir)
    if token_position not in pos_labels:
        raise ValueError(
            f"token_position {token_position!r} not in available labels {pos_labels}."
        )
    pos_idx = pos_labels.index(token_position)

    rng = rng if rng is not None else np.random.default_rng(0)

    trajectories: Dict[str, np.ndarray] = {}
    gts_per_class: Dict[str, np.ndarray] = {}
    for class_name in CLASS_NAMES:
        condition, fail = _CLASS_SOURCES[class_name]
        chunks: List[np.ndarray] = []
        gt_chunks: List[np.ndarray] = []
        for gt in (0, 1):
            path = _discover_activation_pickle(activations_dir, condition, gt, fail)
            if path is None:
                continue
            arr = _load_pickle_as_numpy(path)
            if arr.shape[0] == 0:
                continue
            chunks.append(arr[:, pos_idx, :, :])
            gt_chunks.append(np.full(arr.shape[0], gt, dtype=np.int64))
        if not chunks:
            continue
        pooled = np.concatenate(chunks, axis=0)
        pooled_gt = np.concatenate(gt_chunks, axis=0)
        if n_samples_per_class > 0 and pooled.shape[0] > n_samples_per_class:
            idx = np.sort(rng.choice(pooled.shape[0], size=n_samples_per_class, replace=False))
            pooled = pooled[idx]
            pooled_gt = pooled_gt[idx]
        trajectories[class_name] = pooled
        gts_per_class[class_name] = pooled_gt

    if not trajectories:
        raise FileNotFoundError(
            f"No usable activation pickles found under {activations_dir}."
        )

    sample = next(iter(trajectories.values()))
    n_layers_full = sample.shape[1]
    lo = 0 if layer_start < 0 else int(layer_start)
    hi = n_layers_full - 1 if layer_end < 0 else int(layer_end)
    if not (0 <= lo <= hi < n_layers_full):
        raise ValueError(
            f"Layer range [{layer_start}, {layer_end}] resolved to [{lo}, {hi}] "
            f"out of bounds for n_layers={n_layers_full}."
        )
    layer_indices = np.arange(lo, hi + 1, dtype=np.int64)

    sliced: Dict[str, np.ndarray] = {}
    feature_parts: List[np.ndarray] = []
    label_parts: List[np.ndarray] = []
    gt_parts: List[np.ndarray] = []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        if class_name not in trajectories:
            continue
        arr = trajectories[class_name][:, layer_indices, :]
        sliced[class_name] = arr
        feature_parts.append(arr)
        label_parts.append(np.full(arr.shape[0], class_idx, dtype=np.int64))
        gt_parts.append(gts_per_class[class_name])

    feature_data = np.concatenate(feature_parts, axis=0)
    labels = np.concatenate(label_parts, axis=0)
    gt_labels = np.concatenate(gt_parts, axis=0)

    return ClassTrajectories(
        trajectories=sliced,
        labels=labels,
        gt_labels=gt_labels,
        feature_data=feature_data,
        token_position=token_position,
        layer_indices=layer_indices,
    )


def restrict_class_trajectories(
    full: ClassTrajectories,
    subset_class_names: Sequence[str],
) -> Tuple[ClassTrajectories, np.ndarray]:
    """
    Project ``full`` onto the rows whose class is in ``subset_class_names``.

    Used by the runner when the user wants to predict only a subset of the
    microscopic classes (e.g. ``{lie_success, lie_fail}``). Labels in the
    returned :class:`ClassTrajectories` keep their original
    :data:`CLASS_NAMES` indices (so a "lie_success" row still has
    ``label == 1``) — this preserves the meaning of every plot and the
    semantics of the supervised-descriptor fit functions, which key off
    those constants.

    Returns:
        Tuple ``(subset, subset_to_full_idx)``:

        - ``subset``: A new :class:`ClassTrajectories` whose
          ``trajectories``, ``feature_data``, ``labels`` and ``gt_labels``
          contain only the rows belonging to ``subset_class_names``. Order
          follows :data:`CLASS_NAMES` (so concatenation stays consistent
          with the full slab's ordering).
        - ``subset_to_full_idx``: Array of length ``n_subset`` mapping each
          subset row back to its index in ``full.feature_data``. Required
          by :func:`run_cv` to identify which rows are held out of the
          per-fold supervised-direction-fitting pool.
    """
    requested = list(subset_class_names)
    for c in requested:
        if c not in CLASS_NAMES:
            raise ValueError(
                f"Unknown class {c!r}; expected one of {list(CLASS_NAMES)}."
            )
    missing = [c for c in requested if c not in full.trajectories]
    if missing:
        raise ValueError(
            f"Requested class(es) {missing} are not present in the loaded "
            f"data (available: {list(full.trajectories)})."
        )

    subset_to_full_parts: List[np.ndarray] = []
    sliced: Dict[str, np.ndarray] = {}
    label_parts: List[np.ndarray] = []
    gt_parts: List[np.ndarray] = []
    feature_parts: List[np.ndarray] = []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        if class_name not in requested or class_name not in full.trajectories:
            continue
        full_idx = np.flatnonzero(full.labels == class_idx)
        if full_idx.size == 0:
            continue
        subset_to_full_parts.append(full_idx)
        arr = full.feature_data[full_idx]
        sliced[class_name] = arr
        feature_parts.append(arr)
        label_parts.append(full.labels[full_idx])
        gt_parts.append(full.gt_labels[full_idx])

    if not sliced:
        raise ValueError(
            f"Subset {requested!r} produced zero rows from the loaded data."
        )

    subset_to_full_idx = np.concatenate(subset_to_full_parts).astype(np.int64, copy=False)
    subset = ClassTrajectories(
        trajectories=sliced,
        labels=np.concatenate(label_parts),
        gt_labels=np.concatenate(gt_parts),
        feature_data=np.concatenate(feature_parts, axis=0),
        token_position=full.token_position,
        layer_indices=full.layer_indices.copy(),
    )
    return subset, subset_to_full_idx


# --------------------------------------------------------------------------- #
# Descriptor computation                                                      #
# --------------------------------------------------------------------------- #


@dataclass
class FeatureMap:
    """
    Computed descriptor map for one set of trajectories.

    Attributes:
        descriptor_names: Ordered descriptor names (matches the 3rd axis of
            ``per_sample``).
        per_sample: ``(n_samples, n_layers, n_descriptors)`` — used by
            :func:`build_predictor_matrix` to assemble per-predictor X.
            Layers where the descriptor is undefined are filled with NaN.
            For supervised descriptors, this is computed using the
            **in-sample** DiM fit and is therefore intended for
            visualization (mean curves / PCA) and the full-data refit of
            the coefficient heatmap. Per-fold CV recomputes these columns
            using a training-fold DiM inside :func:`run_cv`.
        layer_indices: Absolute layer indices (length = ``n_layers``).
        mean_per_class: ``{class_name: (n_descriptors, n_layers)}`` —
            class-conditional mean of each descriptor across layers
            (``np.nanmean`` so undefined layers stay NaN and propagate to
            a gap in the line plot).
        std_per_class: Same shape as ``mean_per_class`` — class-conditional
            stddev (``np.nanstd``).
        n_per_class: ``{class_name: int}`` sample count per class.
        valid_mask: ``{descriptor_name: bool array of length n_layers}``
            indicating which layer positions are valid (non-NaN) for each
            descriptor. Currently only :func:`vector_angular_curvature`
            sets the layer-0 entry to ``False``.
        supervised_directions: ``{descriptor_name: (n_layers, d_model)}``
            in-sample DiM directions for every supervised descriptor in
            ``descriptor_names``. Empty when no supervised descriptors
            were requested. Reused by the full-data refit in
            :func:`run_cv`.
    """

    descriptor_names: List[str]
    per_sample: np.ndarray
    layer_indices: np.ndarray
    mean_per_class: Dict[str, np.ndarray]
    std_per_class: Dict[str, np.ndarray]
    n_per_class: Dict[str, int]
    valid_mask: Dict[str, np.ndarray]
    supervised_directions: Dict[str, np.ndarray] = field(default_factory=dict)


def _maybe_smooth(out: np.ndarray, smoothing_enabled: bool, smoothing_sigma: float) -> np.ndarray:
    """Optional 1D Gaussian smoothing along the layer axis."""
    if smoothing_enabled and float(smoothing_sigma) > 0.0 and out.shape[1] > 1:
        out = gaussian_filter1d(out, sigma=float(smoothing_sigma), axis=1, mode="nearest")
    return out.astype(np.float32, copy=False)


def _resolve_descriptor_fn(fn_name: str):
    """Look up ``fn_name`` in either descriptor registry; raise if missing."""
    if fn_name in DESCRIPTOR_REGISTRY:
        return ("unsupervised", DESCRIPTOR_REGISTRY[fn_name])
    if fn_name in SUPERVISED_DESCRIPTOR_REGISTRY:
        return ("supervised", SUPERVISED_DESCRIPTOR_REGISTRY[fn_name])
    raise KeyError(
        f"Descriptor {fn_name!r} not in DESCRIPTOR_REGISTRY or "
        f"SUPERVISED_DESCRIPTOR_REGISTRY. Known: {all_known_descriptors()}"
    )


def compute_feature_map(
    cls_traj: ClassTrajectories,
    *,
    descriptor_specs: Sequence[Tuple[str, str, Mapping[str, Any]]],
    smoothing_enabled: bool = False,
    smoothing_sigma: float = 1.0,
    supervised_fit_data: Optional[ClassTrajectories] = None,
    supervised_directions_override: Optional[Mapping[str, np.ndarray]] = None,
) -> FeatureMap:
    """
    Apply every descriptor in ``descriptor_specs`` to ``cls_traj.feature_data``
    and to each class slab. Optionally smooth along the layer axis with a 1D
    Gaussian (zero-pad mode keeps the boundary tractable).

    Supervised descriptors (entries in
    :data:`SUPERVISED_DESCRIPTOR_REGISTRY`) are fit **once** on the full
    ``cls_traj.feature_data`` slab — their per-sample features are in-sample
    and useful for visualization (mean-descriptor curves, PCA scatter) but
    are not used by per-fold CV accuracy (which refits the direction inside
    :func:`run_cv`).

    Args:
        cls_traj: Output of :func:`load_class_trajectories`, or the
            subset view from :func:`restrict_class_trajectories`. This is
            the data that gets *projected* (for supervised descriptors) or
            *processed* (for unsupervised) and ends up in ``per_sample``.
        descriptor_specs: Sequence of ``(spec_name, function_name, kwargs)``
            tuples. ``function_name`` keys either descriptor registry;
            ``spec_name`` is the display name the output uses (typically
            equal to ``function_name``).
        smoothing_enabled: Apply Gaussian-1D smoothing across the layer axis
            of each descriptor's output (independent per sample).
        smoothing_sigma: Standard deviation (in layer units) for the
            Gaussian smoother.
        supervised_fit_data: When fitting supervised-descriptor directions
            (DiM vectors), use the trajectories / labels from this object
            instead of ``cls_traj``. This is how the class-subset runner
            keeps the lie-honest direction fit on the full population
            (which always includes ``honest_success``) even when the
            prediction subset excludes one of those classes. ``None`` =
            fit and transform on the same slab.
        supervised_directions_override: When provided, skip the DiM fit
            and use these pre-computed directions for the listed
            supervised descriptors. Used at test time so the test rows are
            projected onto the *training-time* DiM (not a refit on test
            labels). Every supervised entry in ``descriptor_specs`` must
            appear as a key in this mapping.

    Returns:
        :class:`FeatureMap`.
    """
    if not descriptor_specs:
        raise ValueError("descriptor_specs is empty — nothing to compute.")

    fit_data = supervised_fit_data if supervised_fit_data is not None else cls_traj

    # In-sample fit for each supervised descriptor (used for plotting + the
    # full-data refit in :func:`run_cv`). ``kwargs`` is currently unused by
    # supervised descriptors but accepted for forward compatibility.
    supervised_directions: Dict[str, np.ndarray] = {}
    for spec_name, fn_name, _ in descriptor_specs:
        kind, ref = _resolve_descriptor_fn(fn_name)
        if kind != "supervised":
            continue
        if supervised_directions_override is not None:
            if spec_name not in supervised_directions_override:
                raise KeyError(
                    f"supervised_directions_override is missing entry for "
                    f"supervised descriptor {spec_name!r}."
                )
            supervised_directions[spec_name] = np.asarray(
                supervised_directions_override[spec_name], dtype=np.float32
            )
            continue
        supervised_directions[spec_name] = ref.fit(
            fit_data.feature_data,
            fit_data.labels,
            fit_data.gt_labels,
        )

    def _apply(traj: np.ndarray) -> np.ndarray:
        per_desc: List[np.ndarray] = []
        for spec_name, fn_name, kwargs in descriptor_specs:
            kind, ref = _resolve_descriptor_fn(fn_name)
            if kind == "unsupervised":
                out = ref(traj, **dict(kwargs))
            else:
                out = ref.transform(traj, supervised_directions[spec_name])
            if out.ndim != 2 or out.shape[:2] != (traj.shape[0], traj.shape[1]):
                raise ValueError(
                    f"Descriptor {fn_name!r} returned shape {out.shape}; "
                    f"expected ({traj.shape[0]}, {traj.shape[1]})."
                )
            per_desc.append(_maybe_smooth(out, smoothing_enabled, smoothing_sigma))
        return np.stack(per_desc, axis=-1)  # (n_samples, n_layers, n_descriptors)

    per_sample = _apply(cls_traj.feature_data)

    # Per-descriptor validity mask: a layer is valid iff at least one sample
    # produced a non-NaN value there (so a single NaN-padded layer of a
    # well-defined descriptor is still treated as invalid). Validity is
    # sample-independent for the descriptors in :data:`DESCRIPTOR_REGISTRY`,
    # so checking ``any`` over samples is enough.
    descriptor_names = [name for name, _, _ in descriptor_specs]
    valid_mask: Dict[str, np.ndarray] = {}
    for d_idx, name in enumerate(descriptor_names):
        col_any_finite = np.any(~np.isnan(per_sample[:, :, d_idx]), axis=0)
        valid_mask[name] = col_any_finite

    mean_per_class: Dict[str, np.ndarray] = {}
    std_per_class: Dict[str, np.ndarray] = {}
    n_per_class: Dict[str, int] = {}
    for class_name, arr in cls_traj.trajectories.items():
        desc_arr = _apply(arr)  # (n_class, n_layers, n_desc)
        # nanmean / nanstd so invalid layers stay NaN and propagate to a
        # visible gap in the line plot, while valid layers still get the
        # right per-class statistic. RuntimeWarning for all-NaN slices is
        # benign here (it just means the layer is invalid for that desc).
        with np.errstate(all="ignore"):
            mean_per_class[class_name] = np.nanmean(desc_arr, axis=0).T
            std_per_class[class_name] = np.nanstd(desc_arr, axis=0).T
        n_per_class[class_name] = int(arr.shape[0])

    return FeatureMap(
        descriptor_names=descriptor_names,
        per_sample=per_sample,
        layer_indices=cls_traj.layer_indices.copy(),
        mean_per_class=mean_per_class,
        std_per_class=std_per_class,
        n_per_class=n_per_class,
        valid_mask=valid_mask,
        supervised_directions=supervised_directions,
    )


@dataclass
class PredictorMatrix:
    """
    Output of :func:`build_predictor_matrix` — per-predictor design matrix
    plus column-level metadata so the coefficient heatmap can scatter
    weights back onto the full ``(descriptor, layer)`` grid even when some
    columns were dropped because the descriptor is undefined at that
    layer.

    Attributes:
        X: ``(n_samples, n_features)`` feature matrix.
        column_names: ``"<desc>_L<layer>"`` for each column.
        column_descriptors: Descriptor name per column.
        column_layers: Absolute layer index per column.
    """

    X: np.ndarray
    column_names: List[str]
    column_descriptors: List[str]
    column_layers: List[int]


def build_predictor_matrix(
    feat: FeatureMap,
    descriptor_subset: Sequence[str],
    *,
    column_overrides: Optional[Mapping[str, np.ndarray]] = None,
) -> PredictorMatrix:
    """
    Flatten the chosen descriptors into a design matrix. Layer columns
    where the descriptor is undefined (per :attr:`FeatureMap.valid_mask`)
    are dropped, so ``n_features <= n_layers * len(descriptor_subset)``.

    Column order is descriptor-major (every valid layer of descriptor 0
    first, then descriptor 1, …); layer order within each descriptor
    follows :attr:`FeatureMap.layer_indices`. Column names use the
    ``"<desc>_L<layer>"`` format.

    Args:
        feat: precomputed feature map (in-sample fit for supervised
            descriptors).
        descriptor_subset: which descriptors to include.
        column_overrides: optional ``{descriptor_name: (n_samples, n_layers)}``
            overrides. When given for descriptor ``d``, the override array
            replaces ``feat.per_sample[:, :, idx_of(d)]`` before slicing on
            ``feat.valid_mask[d]``. Used by :func:`run_cv` to inject
            per-fold supervised-descriptor recomputations while leaving
            the cached unsupervised columns alone.
    """
    overrides = dict(column_overrides or {})
    name_to_idx = {name: i for i, name in enumerate(feat.descriptor_names)}
    cols: List[np.ndarray] = []
    col_names: List[str] = []
    col_descriptors: List[str] = []
    col_layers: List[int] = []
    for desc in descriptor_subset:
        if desc not in name_to_idx:
            raise KeyError(
                f"Descriptor {desc!r} not in computed feature map "
                f"(have {feat.descriptor_names})."
            )
        mask = feat.valid_mask.get(desc)
        if mask is None:
            mask = np.ones(feat.per_sample.shape[1], dtype=bool)
        keep = np.asarray(mask, dtype=bool)
        if not np.any(keep):
            raise ValueError(
                f"Descriptor {desc!r} has no valid layers in this feature map."
            )
        if desc in overrides:
            full = overrides[desc]
            if full.shape != feat.per_sample.shape[:2]:
                raise ValueError(
                    f"column_overrides[{desc!r}] shape {full.shape} does not match "
                    f"feature map per-sample shape {feat.per_sample.shape[:2]}."
                )
            sub = full[:, keep]
        else:
            idx = name_to_idx[desc]
            sub = feat.per_sample[:, keep, idx]
        cols.append(sub)
        kept_layers = feat.layer_indices[keep]
        for l in kept_layers:
            col_names.append(f"{desc}_L{int(l)}")
            col_descriptors.append(desc)
            col_layers.append(int(l))
    X = np.concatenate(cols, axis=1).astype(np.float32, copy=False)
    return PredictorMatrix(
        X=X,
        column_names=col_names,
        column_descriptors=col_descriptors,
        column_layers=col_layers,
    )


def _compute_supervised_columns(
    fit_traj: np.ndarray,
    fit_class_labels: np.ndarray,
    fit_gt_labels: np.ndarray,
    target_traj: np.ndarray,
    descriptor_names: Sequence[str],
    *,
    smoothing_enabled: bool = False,
    smoothing_sigma: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Refit each supervised descriptor's DiM on the ``fit_*`` arrays, then
    project ``target_traj`` onto that direction.

    The fit pool is decoupled from the projection target so the per-fold
    refit in :func:`run_cv` can pull direction-fitting samples from the
    full data (subset-train ∪ all out-of-subset rows that the descriptor
    needs) while still projecting only the subset rows that the predictor
    will consume.

    Returns a mapping ``descriptor_name -> (n_target_samples, n_layers)``
    suitable for :func:`build_predictor_matrix`'s ``column_overrides``.
    """
    out: Dict[str, np.ndarray] = {}
    for name in descriptor_names:
        if name not in SUPERVISED_DESCRIPTOR_REGISTRY:
            raise KeyError(
                f"Descriptor {name!r} is not a supervised descriptor "
                f"(known supervised: {sorted(SUPERVISED_DESCRIPTOR_REGISTRY)})."
            )
        spec = SUPERVISED_DESCRIPTOR_REGISTRY[name]
        direction = spec.fit(fit_traj, fit_class_labels, fit_gt_labels)
        arr = spec.transform(target_traj, direction)
        out[name] = _maybe_smooth(arr, smoothing_enabled, smoothing_sigma)
    return out


# --------------------------------------------------------------------------- #
# Predictor                                                                   #
# --------------------------------------------------------------------------- #


def _make_scaler(name: str):
    n = (name or "standard").lower()
    if n == "standard":
        return StandardScaler()
    if n == "robust":
        return RobustScaler()
    if n == "none":
        return None
    raise ValueError(f"Unknown scaler {name!r}. Choose from 'standard' | 'robust' | 'none'.")


def _resolve_solver(penalty: str, solver: Optional[str]) -> str:
    if solver:
        return solver
    p = penalty.lower()
    if p in ("l1", "elasticnet"):
        return "saga"
    if p in ("l2", "none"):
        return "lbfgs"
    raise ValueError(f"Unknown penalty {penalty!r}.")


def make_logistic_regression(predictor_cfg: Mapping[str, Any]) -> LogisticRegression:
    """
    Construct a :class:`LogisticRegression` from the config block. Handles
    the solver auto-pick and the ``l1_ratio`` requirement for elasticnet.
    Multinomial behavior is implicit in sklearn 1.6+ (the ``multi_class``
    kwarg is deprecated and removed in 1.7).
    """
    penalty = str(predictor_cfg.get("penalty", "l2")).lower()
    solver = _resolve_solver(penalty, predictor_cfg.get("solver"))
    kwargs: Dict[str, Any] = dict(
        penalty=None if penalty == "none" else penalty,
        C=float(predictor_cfg.get("C", 1.0)),
        solver=solver,
        max_iter=int(predictor_cfg.get("max_iter", 5000)),
        random_state=int(predictor_cfg.get("random_state", 0)),
    )
    cw = predictor_cfg.get("class_weight")
    if cw is not None:
        kwargs["class_weight"] = cw
    if penalty == "elasticnet":
        kwargs["l1_ratio"] = float(predictor_cfg.get("l1_ratio", 0.5))
    return LogisticRegression(**kwargs)


# --------------------------------------------------------------------------- #
# Cross-validation                                                            #
# --------------------------------------------------------------------------- #


def _make_cv_splitter(cv_cfg: Mapping[str, Any]):
    method = str(cv_cfg.get("method", "kfold")).lower()
    n_splits = int(cv_cfg.get("n_splits", 5))
    shuffle = bool(cv_cfg.get("shuffle", True))
    stratified = bool(cv_cfg.get("stratified", True))
    random_state = int(cv_cfg.get("random_state", 0))
    if method == "kfold":
        if stratified:
            return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state if shuffle else None)
        return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state if shuffle else None)
    if method == "shuffle_split":
        test_size = float(cv_cfg.get("test_size", 0.2))
        if stratified:
            return StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
        return ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    raise ValueError(f"Unknown cv.method {method!r}.")


@dataclass
class CVResult:
    """
    Outcome of :func:`run_cv` for one predictor.

    Attributes:
        fold_accuracy: ``(n_splits,)`` test accuracy per fold.
        confusion_matrix: ``(n_classes, n_classes)`` summed across folds. Rows
            are true labels, columns predicted, indexed by :data:`CLASS_NAMES`.
        classes_present: List of class names actually represented in the
            data (drops classes whose pickles were absent).
        coef_full: ``(n_classes_present, n_features)`` coefficients from the
            full-data refit. Used by :func:`plot_coefficient_heatmap`.
        intercept_full: ``(n_classes_present,)``.
        full_model: The refit estimator (with its preceding scaler applied
            externally — held here for reuse by the PCA plot).
        full_scaler: The scaler fit on the full data (or ``None``).
        column_names: ``<desc>_L<layer>`` column labels, in coefficient order.
        column_descriptors: Descriptor name per column.
        column_layers: Absolute layer index per column.
    """

    fold_accuracy: np.ndarray
    confusion_matrix: np.ndarray
    classes_present: List[str]
    coef_full: np.ndarray
    intercept_full: np.ndarray
    full_model: LogisticRegression
    full_scaler: Optional[Any]
    column_names: List[str]
    column_descriptors: List[str] = field(default_factory=list)
    column_layers: List[int] = field(default_factory=list)
    # Class names that the classifier was trained to discriminate. Equal to
    # :data:`CLASS_NAMES` for the all-three-class default; restricted to the
    # subset (still in :data:`CLASS_NAMES` order) when the runner predicts
    # only a subset of the microscopic classes. Indexes
    # ``confusion_matrix``'s rows / columns.
    class_names: List[str] = field(default_factory=lambda: list(CLASS_NAMES))


def run_cv(
    cls_traj: ClassTrajectories,
    feat: FeatureMap,
    descriptor_subset: Sequence[str],
    *,
    predictor_cfg: Mapping[str, Any],
    cv_cfg: Mapping[str, Any],
    scaler_cfg: Mapping[str, Any],
    smoothing_enabled: bool = False,
    smoothing_sigma: float = 1.0,
    supervised_fit_data: Optional[ClassTrajectories] = None,
    subset_to_full_idx: Optional[np.ndarray] = None,
) -> CVResult:
    """
    Per-fold scaler + logistic regression + accuracy + summed confusion
    matrix, plus a single full-data refit reused by the coefficient heatmap
    and the PCA visualization.

    Supervised descriptors (entries in :data:`SUPERVISED_DESCRIPTOR_REGISTRY`)
    have their direction vectors **refit on each fold's training split**, so
    the held-out test rows are projected onto a DiM that never saw them.
    Unsupervised columns are taken from ``feat.per_sample`` and never depend
    on the split.

    When ``supervised_fit_data`` and ``subset_to_full_idx`` are provided,
    ``cls_traj`` is treated as a class-subset view of the larger
    ``supervised_fit_data`` slab. The classifier is fit / evaluated only on
    subset rows, but each per-fold supervised-direction refit uses
    ``supervised_fit_data ∖ (subset rows in the held-out test fold)`` —
    so out-of-subset rows (e.g. ``honest_success`` when predicting only
    ``{lie_success, lie_fail}``) participate in direction fitting but
    never appear as training rows in the predictor.

    The full-data refit uses the in-sample directions cached in
    ``feat.per_sample`` (fit on ``supervised_fit_data`` when provided);
    downstream visualizations (PCA scatter, coefficient heatmap) inherit
    those features. CV accuracy numbers come from the per-fold path only.
    """
    y = cls_traj.labels
    splitter = _make_cv_splitter(cv_cfg)
    classes_present_idx = sorted(int(c) for c in np.unique(y))
    classes_present = [CLASS_NAMES[i] for i in classes_present_idx]
    # Class labels the classifier is allowed to discriminate among (in
    # CLASS_NAMES order). For the default all-three-class run this is
    # CLASS_NAMES; for a subset view it's only the rows actually loaded.
    subset_label_indices = sorted(
        CLASS_NAMES.index(c) for c in cls_traj.trajectories.keys()
    )
    class_names = [CLASS_NAMES[i] for i in subset_label_indices]

    supervised_in_subset = [d for d in descriptor_subset if is_supervised_descriptor(d)]
    use_external_fit_pool = (
        supervised_fit_data is not None and subset_to_full_idx is not None
    )
    if use_external_fit_pool:
        full_n = supervised_fit_data.feature_data.shape[0]

    accs: List[float] = []
    cm_total = np.zeros((len(class_names), len(class_names)), dtype=np.int64)

    # Any (n_samples, …) array works as splitter input — use feat.per_sample
    # so the splitter sees exactly the same n_samples as everything downstream.
    splitter_input = feat.per_sample
    fold_iter = (
        splitter.split(splitter_input, y)
        if hasattr(splitter, "split") else splitter.split(splitter_input)
    )
    for fold_idx, (train_idx, test_idx) in enumerate(fold_iter):
        if supervised_in_subset:
            if use_external_fit_pool:
                # Direction-fit pool: everything in the full slab except the
                # full-slab rows that correspond to the held-out subset test
                # fold. Out-of-subset rows are always included (they're never
                # test data here).
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
            sup_overrides = _compute_supervised_columns(
                fit_traj,
                fit_class,
                fit_gt,
                cls_traj.feature_data,
                supervised_in_subset,
                smoothing_enabled=smoothing_enabled,
                smoothing_sigma=smoothing_sigma,
            )
        else:
            sup_overrides = {}
        pm_fold = build_predictor_matrix(
            feat, descriptor_subset, column_overrides=sup_overrides
        )
        X = pm_fold.X

        scaler = _make_scaler(str(scaler_cfg.get("scaler", "standard"))) if scaler_cfg.get("enabled", True) else None
        if scaler is not None:
            X_tr = scaler.fit_transform(X[train_idx])
            X_te = scaler.transform(X[test_idx])
        else:
            X_tr = X[train_idx]
            X_te = X[test_idx]
        model = make_logistic_regression(predictor_cfg)
        model.fit(X_tr, y[train_idx])
        y_pred = model.predict(X_te)
        accs.append(float(accuracy_score(y[test_idx], y_pred)))
        cm_fold = confusion_matrix(y[test_idx], y_pred, labels=subset_label_indices)
        cm_total += cm_fold.astype(np.int64, copy=False)

    # Full-data refit — supervised columns use the in-sample directions
    # already baked into ``feat.per_sample``.
    pm_full = build_predictor_matrix(feat, descriptor_subset)
    scaler_full = _make_scaler(str(scaler_cfg.get("scaler", "standard"))) if scaler_cfg.get("enabled", True) else None
    X_full = scaler_full.fit_transform(pm_full.X) if scaler_full is not None else pm_full.X
    model_full = make_logistic_regression(predictor_cfg)
    model_full.fit(X_full, y)

    return CVResult(
        fold_accuracy=np.asarray(accs, dtype=np.float64),
        confusion_matrix=cm_total,
        classes_present=classes_present,
        coef_full=np.asarray(model_full.coef_, dtype=np.float64),
        intercept_full=np.asarray(model_full.intercept_, dtype=np.float64),
        full_model=model_full,
        full_scaler=scaler_full,
        column_names=list(pm_full.column_names),
        column_descriptors=list(pm_full.column_descriptors),
        column_layers=list(pm_full.column_layers),
        class_names=class_names,
    )


# --------------------------------------------------------------------------- #
# Plotting helpers                                                            #
# --------------------------------------------------------------------------- #


def _save_figure(fig: plt.Figure, basepath: Path, formats: Sequence[str]) -> List[Path]:
    """Save ``fig`` under ``basepath.<fmt>`` for each ``fmt``; return the paths."""
    basepath.parent.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for fmt in formats:
        out_path = basepath.with_suffix(f".{fmt.lstrip('.')}")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        written.append(out_path)
    plt.close(fig)
    return written


def plot_mean_descriptor_curves(
    feat: FeatureMap,
    basepath: Path,
    *,
    band: str = "ci95",
    class_colors: Optional[Mapping[str, str]] = None,
    formats: Sequence[str] = ("png",),
    class_names: Optional[Sequence[str]] = None,
) -> List[Path]:
    """
    One row per descriptor; X = layer index, Y = descriptor value; mean line
    + shaded band (``band='stddev'`` or ``'ci95'``) per class.

    ``class_names`` (defaulting to :data:`CLASS_NAMES`) controls which
    classes to render and in what order — pass the predictor's subset
    when the run only discriminates among a subset of the microscopic
    classes.
    """
    class_colors = dict(class_colors or {})
    iter_classes = list(class_names) if class_names is not None else list(CLASS_NAMES)
    descriptors = feat.descriptor_names
    fig, axes = plt.subplots(
        len(descriptors), 1,
        figsize=(7.5, 2.6 * len(descriptors) + 0.5),
        sharex=True,
        squeeze=False,
    )
    layers = feat.layer_indices
    for d_idx, desc_name in enumerate(descriptors):
        ax = axes[d_idx, 0]
        for class_name in iter_classes:
            if class_name not in feat.mean_per_class:
                continue
            mean = feat.mean_per_class[class_name][d_idx]
            std = feat.std_per_class[class_name][d_idx]
            n = max(1, feat.n_per_class.get(class_name, 1))
            if band == "ci95":
                half = 1.96 * std / np.sqrt(n)
            else:
                half = std
            color = class_colors.get(class_name, None)
            ax.plot(layers, mean, label=class_name, color=color, linewidth=1.6)
            ax.fill_between(layers, mean - half, mean + half, color=color, alpha=0.18, linewidth=0)
        ax.set_ylabel(desc_name)
        ax.grid(True, alpha=0.3)
        if d_idx == 0:
            ax.legend(loc="best", fontsize=8, frameon=False)
    axes[-1, 0].set_xlabel("layer")
    fig.suptitle("Mean descriptor curves per class", fontsize=11)
    fig.tight_layout()
    return _save_figure(fig, basepath, formats)


def plot_pca_scatter(
    X: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    basepath: Path,
    dims: Sequence[int] = (2,),
    class_colors: Optional[Mapping[str, str]] = None,
    scaler: Optional[Any] = None,
    formats: Sequence[str] = ("png",),
    class_names: Optional[Sequence[str]] = None,
) -> List[Path]:
    """
    PCA-project the flattened predictor matrix into each ``n_components`` in
    ``dims``. One figure file is written per dimension.

    If ``scaler`` is provided it is applied first (matches the predictor's
    training-time scaling, so the PCA matches the space the classifier sees).

    ``class_names`` selects which classes to overlay (defaults to all of
    :data:`CLASS_NAMES`); ``y`` is interpreted in :data:`CLASS_NAMES`
    indexing regardless of subset.
    """
    class_colors = dict(class_colors or {})
    iter_classes = list(class_names) if class_names is not None else list(CLASS_NAMES)
    Xs = scaler.transform(X) if scaler is not None else X
    written: List[Path] = []
    for n_components in dims:
        n_components = int(n_components)
        if n_components < 1:
            continue
        n_components = min(n_components, Xs.shape[0], Xs.shape[1])
        pca = PCA(n_components=n_components)
        Z = pca.fit_transform(Xs)
        evr = pca.explained_variance_ratio_

        fig = plt.figure(figsize=(6.0, 5.0))
        if n_components >= 3:
            ax = fig.add_subplot(111, projection="3d")
            for class_name in iter_classes:
                c_idx = CLASS_NAMES.index(class_name)
                mask = (y == c_idx)
                if not np.any(mask):
                    continue
                color = class_colors.get(class_name, None)
                ax.scatter(Z[mask, 0], Z[mask, 1], Z[mask, 2],
                           label=class_name, color=color, s=12, alpha=0.7)
            ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
            ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
            ax.set_zlabel(f"PC3 ({evr[2]*100:.1f}%)")
        else:
            ax = fig.add_subplot(111)
            for class_name in iter_classes:
                c_idx = CLASS_NAMES.index(class_name)
                mask = (y == c_idx)
                if not np.any(mask):
                    continue
                color = class_colors.get(class_name, None)
                if n_components == 1:
                    ax.scatter(Z[mask, 0], np.zeros(int(mask.sum())), label=class_name,
                               color=color, s=12, alpha=0.7)
                else:
                    ax.scatter(Z[mask, 0], Z[mask, 1], label=class_name,
                               color=color, s=12, alpha=0.7)
            ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
            if n_components >= 2:
                ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
            ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8, frameon=False)
        fig.suptitle(f"{title}  ({n_components}D PCA)", fontsize=11)
        fig.tight_layout()
        written.extend(_save_figure(fig, basepath.with_name(f"{basepath.name}_{n_components}D"), formats))
    return written


def plot_coefficient_heatmap(
    cv: CVResult,
    descriptor_subset: Sequence[str],
    layer_indices: np.ndarray,
    *,
    title: str,
    basepath: Path,
    cmap: str = "RdBu_r",
    formats: Sequence[str] = ("png",),
) -> List[Path]:
    """
    For each class present, plot a 2D heatmap of logistic-regression
    coefficients with descriptors on rows and layers on columns. The
    colormap is centered at zero so red = positive correlation with the
    class, blue = negative.

    Columns the predictor matrix dropped (because the descriptor is
    undefined at that layer — currently only
    :func:`vector_angular_curvature` at layer 0) are rendered as light-grey
    "blank" cells using the colormap's bad-value color.
    """
    n_desc = len(descriptor_subset)
    n_layers = len(layer_indices)
    desc_to_row = {d: i for i, d in enumerate(descriptor_subset)}
    layer_to_col = {int(l): i for i, l in enumerate(layer_indices)}

    if cv.coef_full.shape[1] != len(cv.column_descriptors):
        raise ValueError(
            f"Coefficient column count {cv.coef_full.shape[1]} doesn't match "
            f"column metadata length {len(cv.column_descriptors)}."
        )

    n_rows = cv.coef_full.shape[0]
    # sklearn's binary LogisticRegression returns a single coefficient row
    # (log-odds of classes_present[1]); the multinomial case has one row per
    # class.
    if n_rows == 1 and len(cv.classes_present) == 2:
        row_labels = [f"{cv.classes_present[1]} vs {cv.classes_present[0]}"]
    else:
        row_labels = list(cv.classes_present)
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(max(4.5, n_layers * 0.25 + 2.5), 1.4 * n_desc + 0.7 + (n_rows * 0.6)),
        squeeze=False,
    )
    vmax = float(np.max(np.abs(cv.coef_full))) or 1e-8
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="lightgrey")
    for c_idx, class_name in enumerate(row_labels):
        ax = axes[c_idx, 0]
        # Scatter the (variable-length) coefficient vector back onto a
        # full (n_desc, n_layers) grid using the per-column metadata.
        # Untouched cells stay NaN and render via cmap.set_bad.
        grid = np.full((n_desc, n_layers), np.nan, dtype=np.float64)
        for col_idx, (desc, layer) in enumerate(
            zip(cv.column_descriptors, cv.column_layers)
        ):
            if desc not in desc_to_row or int(layer) not in layer_to_col:
                continue
            grid[desc_to_row[desc], layer_to_col[int(layer)]] = cv.coef_full[c_idx, col_idx]
        masked = np.ma.masked_invalid(grid)
        im = ax.imshow(masked, aspect="auto", cmap=cmap_obj, norm=norm)
        ax.set_yticks(np.arange(n_desc))
        ax.set_yticklabels(list(descriptor_subset))
        ax.set_xticks(np.arange(n_layers))
        ax.set_xticklabels([str(int(l)) for l in layer_indices], rotation=0, fontsize=8)
        ax.set_title(f"class={class_name}", fontsize=9)
        if c_idx == n_rows - 1:
            ax.set_xlabel("layer")
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save_figure(fig, basepath, formats)


def plot_cv_accuracy(
    cv_results: Mapping[str, CVResult],
    *,
    basepath: Path,
    formats: Sequence[str] = ("png",),
    n_classes: Optional[int] = None,
) -> List[Path]:
    """
    One bar per predictor (individual + combined), height = mean fold
    accuracy, error bar = stddev across folds.

    ``n_classes`` controls the chance reference line. Defaults to the
    discriminated-class count taken from the first :class:`CVResult`
    (falling back to ``len(CLASS_NAMES)``) so subset runs show the right
    chance baseline.
    """
    names = list(cv_results.keys())
    means = [float(cv_results[n].fold_accuracy.mean()) for n in names]
    stds = [float(cv_results[n].fold_accuracy.std()) for n in names]
    fig, ax = plt.subplots(figsize=(max(5.5, 0.6 * len(names) + 2.0), 4.0))
    xs = np.arange(len(names))
    ax.bar(xs, means, yerr=stds, capsize=4, color="#4C72B0")
    if n_classes is None:
        if names:
            n_classes = len(cv_results[names[0]].class_names) or len(CLASS_NAMES)
        else:
            n_classes = len(CLASS_NAMES)
    chance = 1.0 / max(1, int(n_classes))
    ax.axhline(chance, color="grey", linestyle="--", linewidth=1.0, label=f"chance ({chance:.2f})")
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel("CV accuracy (mean ± std)")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.suptitle("Cross-validation accuracy", fontsize=11)
    fig.tight_layout()
    return _save_figure(fig, basepath, formats)


def plot_confusion_matrices(
    cv_results: Mapping[str, CVResult],
    *,
    basepath: Path,
    formats: Sequence[str] = ("png",),
    normalize: bool = True,
) -> List[Path]:
    """
    One subplot per predictor showing the summed-across-folds confusion
    matrix. Rows = true, columns = predicted, in each predictor's
    ``class_names`` order (defaults to :data:`CLASS_NAMES`; restricted to
    the subset on subset runs). Cells are normalized per row when
    ``normalize=True``.
    """
    names = list(cv_results.keys())
    n = len(names)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.4 * rows), squeeze=False)
    for idx, predictor_name in enumerate(names):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        cv = cv_results[predictor_name]
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
        im = ax.imshow(cm_show, cmap="Blues", vmin=0.0, vmax=vmax)
        ax.set_xticks(np.arange(k))
        ax.set_yticks(np.arange(k))
        ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(tick_labels, fontsize=8)
        for i in range(k):
            for j in range(k):
                text = f"{cm_show[i, j]:.2f}" if normalize else f"{int(cm_show[i, j])}"
                ax.text(j, i, text, ha="center", va="center",
                        color="white" if cm_show[i, j] > vmax * 0.6 else "black",
                        fontsize=8)
        ax.set_title(predictor_name, fontsize=9)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # Hide unused axes.
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")
    fig.suptitle("Confusion matrices (CV-summed, row-normalized)" if normalize
                 else "Confusion matrices (CV-summed)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save_figure(fig, basepath, formats)


# --------------------------------------------------------------------------- #
# Trained-pipeline serialization + held-out test evaluation                   #
# --------------------------------------------------------------------------- #


@dataclass
class PerPredictorPipeline:
    """
    The preprocessing scaler + logistic regression for one predictor, plus
    the column metadata required to rebuild the feature matrix at test time.

    Attributes:
        predictor_name: Display name (matches the key under
            ``descriptors.individual`` / ``combined_predictors``).
        descriptor_subset: Descriptor names used by this predictor (in
            column order).
        scaler: The fit preprocessing scaler (``StandardScaler`` /
            ``RobustScaler`` / ``None``). Reused at test time as
            ``scaler.transform(X_test)``.
        model: The fit :class:`LogisticRegression`.
        column_names: ``"<desc>_L<layer>"`` per feature column.
        column_descriptors: Descriptor name per feature column.
        column_layers: Absolute layer index per feature column.
        class_names: Classes the model was trained to discriminate (in
            :data:`CLASS_NAMES` order).
    """

    predictor_name: str
    descriptor_subset: List[str]
    scaler: Optional[Any]
    model: LogisticRegression
    column_names: List[str]
    column_descriptors: List[str]
    column_layers: List[int]
    class_names: List[str]


@dataclass
class TrainedPipelineBundle:
    """
    All artifacts needed to re-run inference on a fresh cache folder
    *without* recomputing the training-time supervised DiM fits.

    Saved per (training run alias, training token position). One bundle
    holds every predictor (individual + combined) trained at that position
    so all downstream test-time work shares the same pre-processing
    decisions.

    Attributes:
        descriptor_specs: ``[(spec_name, fn_name, kwargs), ...]`` covering
            every descriptor any predictor needs.
        training_token_position: Token-position label used at training.
        training_layer_indices: Absolute layer indices used at training.
        training_data_folder: Source cache folder (for provenance).
        supervised_directions: ``{descriptor_name: (n_layers, d_model)}``
            in-sample DiM directions, fit on the full training slab. Test
            data is projected onto these.
        valid_mask: ``{descriptor_name: bool array of length n_layers}``
            from training. Reused at test time so the column count is
            identical and the saved scaler / model apply unchanged.
        smoothing_enabled / smoothing_sigma: 1D-Gaussian smoothing config
            used at training.
        predictors: ``{predictor_name: PerPredictorPipeline}``.
    """

    descriptor_specs: List[Tuple[str, str, Mapping[str, Any]]]
    training_token_position: str
    training_layer_indices: np.ndarray
    training_data_folder: str
    supervised_directions: Dict[str, np.ndarray]
    valid_mask: Dict[str, np.ndarray]
    smoothing_enabled: bool
    smoothing_sigma: float
    predictors: Dict[str, PerPredictorPipeline]


def save_pipeline_bundle(bundle: TrainedPipelineBundle, path: Path) -> Path:
    """Pickle ``bundle`` to ``path`` (creating parents). Returns ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_pipeline_bundle(path: Path) -> TrainedPipelineBundle:
    """Inverse of :func:`save_pipeline_bundle`."""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, TrainedPipelineBundle):
        raise TypeError(
            f"{path} did not deserialize to TrainedPipelineBundle (got "
            f"{type(obj).__name__})."
        )
    return obj


@dataclass
class TestResult:
    """
    Outcome of applying one predictor's :class:`PerPredictorPipeline` to a
    held-out test slab.

    Attributes:
        predictor_name: Name of the predictor evaluated.
        test_accuracy: Accuracy on the test rows, in ``[0, 1]``.
        confusion_matrix: ``(n_classes, n_classes)`` int matrix indexed by
            ``class_names`` (rows = true, cols = predicted).
        class_names: Same as ``PerPredictorPipeline.class_names``.
        classes_present: Class names actually represented in this test
            slab (may be a subset).
        n_per_class: ``{class_name: int}`` test sample counts.
        y_true / y_pred: Per-sample labels (in :data:`CLASS_NAMES` indexing).
        X_test_scaled: Test feature matrix after the saved scaler. Used by
            the PCA plot.
    """

    predictor_name: str
    test_accuracy: float
    confusion_matrix: np.ndarray
    class_names: List[str]
    classes_present: List[str]
    n_per_class: Dict[str, int]
    y_true: np.ndarray
    y_pred: np.ndarray
    X_test_scaled: np.ndarray


def evaluate_pipeline_on_test(
    bundle: TrainedPipelineBundle,
    test_cls_traj: ClassTrajectories,
    *,
    standardization_mode: str = "frozen",
    direction_mode: str = "frozen",
    direction_fit_data: Optional[ClassTrajectories] = None,
) -> Tuple[Dict[str, TestResult], FeatureMap]:
    """
    Apply every predictor in ``bundle`` to ``test_cls_traj``.

    ``standardization_mode``:
        - ``"frozen"`` (default): use the saved per-predictor scaler.
        - ``"refit_on_test"``: refit a fresh scaler (same class as the
          one bundled with the predictor) on the test predictor matrix,
          then apply it. The LR coefficients themselves stay frozen.

    ``direction_mode``:
        - ``"frozen"`` (default): supervised descriptors project test
          rows onto the bundle's training-time DiMs.
        - ``"refit_on_test"``: refit each supervised DiM on
          ``direction_fit_data`` (defaults to ``test_cls_traj``) and
          project test rows onto those new directions. The training-fit
          DiMs in ``bundle.supervised_directions`` are unused. Specs with
          no supervised descriptors treat this as a no-op.

    The training-time ``valid_mask`` is forced onto the test feature map so
    the predictor matrix shapes line up with the saved scaler and
    coefficients even when (e.g.) a degenerate test column would have
    been masked differently if refit independently.

    Returns:
        ``(per_predictor_results, test_feat)``.
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
    if test_cls_traj.layer_indices.shape != bundle.training_layer_indices.shape:
        raise ValueError(
            f"Test data has n_layers={test_cls_traj.layer_indices.size} but "
            f"training used n_layers={bundle.training_layer_indices.size}. "
            "The saved pipeline cannot be applied to a different layer count."
        )
    # DiM shape compatibility: every supervised DiM is (n_layers, d_model).
    sample_dmodel = test_cls_traj.feature_data.shape[-1]
    for name, direction in bundle.supervised_directions.items():
        if direction.shape[0] != test_cls_traj.layer_indices.size:
            raise ValueError(
                f"Supervised DiM {name!r} has n_layers={direction.shape[0]} "
                f"but test layer count is {test_cls_traj.layer_indices.size}."
            )
        if direction.shape[1] != sample_dmodel:
            raise ValueError(
                f"Supervised DiM {name!r} has d_model={direction.shape[1]} "
                f"but test d_model is {sample_dmodel}. The pipeline cannot "
                "be applied across different model widths."
            )

    if direction_mode == "refit_on_test":
        fit_pool = direction_fit_data if direction_fit_data is not None else test_cls_traj
        test_feat = compute_feature_map(
            test_cls_traj,
            descriptor_specs=bundle.descriptor_specs,
            smoothing_enabled=bundle.smoothing_enabled,
            smoothing_sigma=bundle.smoothing_sigma,
            supervised_fit_data=fit_pool,
        )
    else:
        test_feat = compute_feature_map(
            test_cls_traj,
            descriptor_specs=bundle.descriptor_specs,
            smoothing_enabled=bundle.smoothing_enabled,
            smoothing_sigma=bundle.smoothing_sigma,
            supervised_directions_override=bundle.supervised_directions,
        )
    # Force training-time validity onto the test feature map so column
    # counts match the saved scaler / coefficients exactly.
    test_feat.valid_mask = {k: np.asarray(v, dtype=bool) for k, v in bundle.valid_mask.items()}

    results: Dict[str, TestResult] = {}
    y_full = test_cls_traj.labels
    for predictor_name, pipe in bundle.predictors.items():
        pm = build_predictor_matrix(test_feat, pipe.descriptor_subset)
        if pm.X.shape[1] != len(pipe.column_names):
            raise ValueError(
                f"Predictor {predictor_name!r}: test feature matrix has "
                f"{pm.X.shape[1]} columns but the saved pipeline expects "
                f"{len(pipe.column_names)}. This usually means the test "
                "data triggered a different valid_mask than training."
            )
        if standardization_mode == "frozen":
            X_test = pipe.scaler.transform(pm.X) if pipe.scaler is not None else pm.X
        else:
            # Refit a fresh scaler of the same class on test, ignore the
            # training-fit one. ``None`` => predictor was trained without
            # scaling; nothing to do.
            if pipe.scaler is None:
                X_test = pm.X
            else:
                refit_scaler = type(pipe.scaler)()
                X_test = refit_scaler.fit_transform(pm.X)
        y_pred = pipe.model.predict(X_test)
        # Restrict the scored rows to classes the model was trained on.
        train_label_indices = [CLASS_NAMES.index(c) for c in pipe.class_names]
        scored_mask = np.isin(y_full, train_label_indices)
        if not np.any(scored_mask):
            raise ValueError(
                f"Predictor {predictor_name!r}: no test rows match the "
                f"training class set {pipe.class_names}."
            )
        y_true_scored = y_full[scored_mask]
        y_pred_scored = y_pred[scored_mask]
        acc = float(accuracy_score(y_true_scored, y_pred_scored))
        cm = confusion_matrix(
            y_true_scored, y_pred_scored, labels=train_label_indices
        ).astype(np.int64, copy=False)
        present = [
            c for c in pipe.class_names
            if int(np.sum(y_true_scored == CLASS_NAMES.index(c))) > 0
        ]
        n_per_class = {
            c: int(np.sum(y_true_scored == CLASS_NAMES.index(c)))
            for c in pipe.class_names
        }
        results[predictor_name] = TestResult(
            predictor_name=predictor_name,
            test_accuracy=acc,
            confusion_matrix=cm,
            class_names=list(pipe.class_names),
            classes_present=present,
            n_per_class=n_per_class,
            y_true=y_true_scored.astype(np.int64, copy=False),
            y_pred=y_pred_scored.astype(np.int64, copy=False),
            X_test_scaled=X_test.astype(np.float32, copy=False),
        )
    return results, test_feat


def plot_test_accuracy(
    test_results: Mapping[str, TestResult],
    *,
    basepath: Path,
    formats: Sequence[str] = ("png",),
) -> List[Path]:
    """
    One bar per predictor showing held-out test accuracy. Dashed line at
    1/n_classes_trained marks chance (assumed uniform priors).
    """
    names = list(test_results.keys())
    if not names:
        raise ValueError("plot_test_accuracy: no test results to plot.")
    accs = [float(test_results[n].test_accuracy) for n in names]
    n_classes = max(len(test_results[n].class_names) for n in names)
    fig, ax = plt.subplots(figsize=(max(5.5, 0.6 * len(names) + 2.0), 4.0))
    xs = np.arange(len(names))
    ax.bar(xs, accs, color="#55A868")
    chance = 1.0 / max(1, n_classes)
    ax.axhline(chance, color="grey", linestyle="--", linewidth=1.0,
               label=f"chance ({chance:.2f})")
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.suptitle("Held-out test accuracy", fontsize=11)
    fig.tight_layout()
    return _save_figure(fig, basepath, formats)


def plot_test_confusion_matrices(
    test_results: Mapping[str, TestResult],
    *,
    basepath: Path,
    formats: Sequence[str] = ("png",),
    normalize: bool = True,
) -> List[Path]:
    """
    One subplot per predictor. Rows = true, columns = predicted, in the
    order of each predictor's ``class_names``. Cells row-normalized when
    ``normalize=True``.
    """
    names = list(test_results.keys())
    if not names:
        raise ValueError("plot_test_confusion_matrices: no test results.")
    n = len(names)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.4 * rows), squeeze=False)
    for idx, predictor_name in enumerate(names):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        tr = test_results[predictor_name]
        cm = tr.confusion_matrix.astype(np.float64)
        tick_labels = list(tr.class_names)
        k = len(tick_labels)
        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            cm_show = cm / row_sums
            vmax = 1.0
        else:
            cm_show = cm
            vmax = cm.max() if cm.size else 1.0
        im = ax.imshow(cm_show, cmap="Blues", vmin=0.0, vmax=vmax)
        ax.set_xticks(np.arange(k))
        ax.set_yticks(np.arange(k))
        ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(tick_labels, fontsize=8)
        for i in range(k):
            for j in range(k):
                text = f"{cm_show[i, j]:.2f}" if normalize else f"{int(cm_show[i, j])}"
                ax.text(j, i, text, ha="center", va="center",
                        color="white" if cm_show[i, j] > vmax * 0.6 else "black",
                        fontsize=8)
        ax.set_title(f"{predictor_name}\nacc={tr.test_accuracy:.3f}", fontsize=9)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")
    fig.suptitle("Test confusion matrices (row-normalized)" if normalize
                 else "Test confusion matrices", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save_figure(fig, basepath, formats)


def compute_dynamical_direction_cosines(
    bundle: TrainedPipelineBundle,
    direction_fit_data: ClassTrajectories,
) -> Dict[str, np.ndarray]:
    """
    Per-direction, per-layer cosine similarity between each training-fit
    DiM in ``bundle.supervised_directions`` and a refit-on-test DiM.

    Returns ``{descriptor_name: (n_layers,) float64}``. Empty dict when
    the bundle has no supervised directions. Layers where either side is
    degenerate (zero norm) come back as NaN so plots show gaps rather
    than spurious zeros.
    """
    if not bundle.supervised_directions:
        return {}
    refit: Dict[str, np.ndarray] = {}
    for name in bundle.supervised_directions:
        if name not in SUPERVISED_DESCRIPTOR_REGISTRY:
            raise KeyError(
                f"Direction {name!r} from bundle is not in "
                f"SUPERVISED_DESCRIPTOR_REGISTRY."
            )
        refit[name] = SUPERVISED_DESCRIPTOR_REGISTRY[name].fit(
            direction_fit_data.feature_data,
            direction_fit_data.labels,
            direction_fit_data.gt_labels,
        )
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


def plot_dynamical_direction_cosines(
    cosines: Mapping[str, np.ndarray],
    layer_indices: Sequence[int],
    *,
    basepath: Path,
    title: str = "Direction cosine similarity (train vs test-refit)",
    formats: Sequence[str] = ("png",),
) -> List[Path]:
    """One line per supervised descriptor across layers; 1.0 = identical axis."""
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


def plot_test_accuracy_by_mode(
    test_results_per_mode: Mapping[str, Mapping[str, TestResult]],
    *,
    basepath: Path,
    formats: Sequence[str] = ("png",),
) -> List[Path]:
    """
    Grouped bar chart: x-axis is predictor name; one bar per
    ``(std_mode, dir_mode)`` combination. Same shape regardless of how
    many modes were requested (1 = one bar each, 4 = four bars each).

    ``test_results_per_mode`` maps a mode slug (e.g.
    ``"std-frozen__dir-frozen"``) to its ``{predictor_name: TestResult}``
    mapping.
    """
    if not test_results_per_mode:
        return []
    mode_slugs = list(test_results_per_mode.keys())
    sample_mode = test_results_per_mode[mode_slugs[0]]
    predictor_names = list(sample_mode.keys())
    if not predictor_names:
        return []
    n_classes = max(len(sample_mode[p].class_names) for p in predictor_names)
    chance = 1.0 / max(1, int(n_classes))

    fig, ax = plt.subplots(
        figsize=(max(6.5, 0.7 * len(predictor_names) * len(mode_slugs) + 2.5), 4.2)
    )
    x = np.arange(len(predictor_names))
    width = 0.8 / max(1, len(mode_slugs))
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    for i, slug in enumerate(mode_slugs):
        accs = [
            float(test_results_per_mode[slug][p].test_accuracy)
            for p in predictor_names
        ]
        offset = (i - (len(mode_slugs) - 1) / 2.0) * width
        ax.bar(
            x + offset, accs, width=width,
            label=slug, color=palette[i % len(palette)],
        )
    ax.axhline(chance, color="grey", linestyle="--", linewidth=1.0,
               label=f"chance ({chance:.2f})")
    ax.set_xticks(x)
    ax.set_xticklabels(predictor_names, rotation=35, ha="right")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.suptitle("Held-out test accuracy by mode", fontsize=11)
    fig.tight_layout()
    return _save_figure(fig, basepath, formats)
