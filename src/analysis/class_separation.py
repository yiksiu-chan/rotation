"""
Per-layer 4-class classifier accuracy curves.

For one ``(SysPrompt, seed)`` pair, the four cached activation pickles
(``control_GT_0`` / ``control_GT_1`` / ``experiment_GT_0`` /
``experiment_GT_1``) are treated as four classes of a balanced classification
problem. At one chosen token position, for each layer, we fit a multinomial
logistic-regression classifier on the per-sample activations inside a
stratified k-fold cross-validation loop and record the mean test-fold
accuracy. Chance is ``1 / n_classes = 0.25`` for the standard four-group
setup.

Pipeline (per training fold, applied in this order):

1. :class:`~sklearn.preprocessing.StandardScaler` — z-score the
   ``d_model``-dimensional features so PCA isn't dominated by high-variance
   coordinates.
2. :class:`~sklearn.decomposition.PCA` (``n_components=n_PC_classifier``,
   ``whiten=True``) — compress to a low-dimensional subspace fit on training
   samples only (no leakage into the test fold).
3. :class:`~sklearn.linear_model.LogisticRegression`
   (``solver="lbfgs"``, ``C=cfg.classifier_C``) — L2-regularized softmax over
   the classes (sklearn's lbfgs solver defaults to multinomial loss on
   1.5+).

Refitting PCA per fold rather than once on the full dataset keeps the
reported CV accuracy as an unbiased estimate of generalization — the
held-out fold never contributed to picking the principal components.

The minimum train-fold size under ``StratifiedKFold(n_splits=k)`` is
``(k - 1) * min(class_sizes)``. PCA needs at least ``n_components + 1``
samples to fit; we precompute this bound and raise a :class:`ValueError`
*before* running CV if it would be violated, so partially-cached sweeps
fail loudly rather than silently dropping to a lower-dim subspace.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# Class order fed to the classifier — encoded as integer labels 0..3 in this
# exact sequence. Keep stable: any downstream code that reads
# ``class_separation_stats_*.json`` and wants to introspect the label map
# should look here.
CLASS_GROUPS: Tuple[str, ...] = (
    "control_GT_0",
    "control_GT_1",
    "experiment_GT_0",
    "experiment_GT_1",
)


def _build_xy_per_layer(
    activations_by_group: Dict[str, np.ndarray],
    position_index: int,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Stack the four ``(condition, GT)`` activation pickles into one
    ``(N, n_layers, d_model)`` feature tensor + ``(N,)`` integer label vector
    sliced at ``position_index``.

    Returns:
        ``(X_all, y_all, class_sizes)`` where ``X_all[i, L, :]`` is the
        layer-``L`` activation of sample ``i`` at the chosen position,
        ``y_all[i]`` ∈ ``{0, 1, 2, 3}`` indexes :data:`CLASS_GROUPS`, and
        ``class_sizes[c]`` is the count of samples for class ``c``.

    Raises:
        ValueError: If fewer than two non-empty groups are present (a
        classifier needs at least two classes).
    """
    Xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    class_sizes: List[int] = []
    for cls_id, group in enumerate(CLASS_GROUPS):
        arr = activations_by_group.get(group)
        if arr is None or arr.shape[0] == 0:
            class_sizes.append(0)
            continue
        # arr: (k, n_pos, n_layers, d_model) -> slice the chosen position
        # to (k, n_layers, d_model).
        Xs.append(arr[:, position_index, :, :])
        ys.append(np.full(arr.shape[0], cls_id, dtype=np.int64))
        class_sizes.append(int(arr.shape[0]))

    if sum(1 for n in class_sizes if n > 0) < 2:
        raise ValueError(
            "Need at least two non-empty groups to fit a classifier; got "
            f"class sizes {dict(zip(CLASS_GROUPS, class_sizes))}."
        )

    X_all = np.concatenate(Xs, axis=0).astype(np.float32, copy=False)
    y_all = np.concatenate(ys, axis=0)
    return X_all, y_all, class_sizes


def _validate_pca_fit_feasible(
    n_PC_classifier: int,
    cv_n_splits: int,
    class_sizes: List[int],
) -> None:
    """
    Raise :class:`ValueError` if ``n_PC_classifier + 1`` exceeds the smallest
    training-fold size achievable under ``StratifiedKFold(n_splits=cv_n_splits)``.

    Stratified k-fold guarantees each fold's training portion contains at
    least ``(k - 1) * min(class_sizes_nonzero)`` samples — that's the bound
    we check.
    """
    non_empty = [n for n in class_sizes if n > 0]
    if not non_empty:
        raise ValueError("All four activation groups are empty.")
    min_class = min(non_empty)
    if min_class < cv_n_splits:
        raise ValueError(
            f"StratifiedKFold(n_splits={cv_n_splits}) requires every class to "
            f"have at least {cv_n_splits} members, but the smallest class has "
            f"only {min_class}. Class sizes = {class_sizes}. Reduce "
            f"cv_n_splits or increase cache n_sample_cache."
        )
    min_train_size = (cv_n_splits - 1) * min_class
    if min_train_size < n_PC_classifier + 1:
        raise ValueError(
            f"PCA n_components={n_PC_classifier} requires at least "
            f"{n_PC_classifier + 1} samples per training fold, but the "
            f"minimum training fold size under StratifiedKFold("
            f"n_splits={cv_n_splits}) is {min_train_size} "
            f"= (k - 1) * min(class_size) = ({cv_n_splits} - 1) * {min_class}. "
            "Reduce n_PC_classifier, reduce cv_n_splits, or increase "
            "cache n_sample_cache."
        )


def compute_cv_accuracy_per_layer(
    activations_by_group: Dict[str, np.ndarray],
    position_index: int,
    *,
    n_PC_classifier: int,
    cv_n_splits: int,
    cv_shuffle: bool,
    cv_random_state: int,
    classifier_C: float,
    classifier_max_iter: int,
) -> np.ndarray:
    """
    For each layer, train a 4-class classifier on the per-sample activations
    at ``position_index`` and return the mean stratified-k-fold CV accuracy.

    Args:
        activations_by_group: Mapping from group key (one of
            :data:`CLASS_GROUPS`) to ``(k, n_pos, n_layers, d_model)`` arrays
            as loaded from the cache stage's pickles.
        position_index: Column index of the token position to slice features
            at (must agree with ``cfg.n_pre_prompt_tokens`` /
            ``cfg.n_generated_tokens``).
        n_PC_classifier: Number of PCA components kept before logistic
            regression. PCA is fit per training fold (leakage-free).
        cv_n_splits: ``StratifiedKFold(n_splits=...)``.
        cv_shuffle: Whether to shuffle samples before splitting into folds.
        cv_random_state: Seed for the shuffle. Ignored when ``cv_shuffle`` is
            ``False``.
        classifier_C: Inverse L2 regularization strength.
        classifier_max_iter: Max optimizer iterations.

    Returns:
        Length-``n_layers`` ``np.ndarray[float64]`` of mean CV accuracies in
        ``[0, 1]``. Chance = ``0.25`` for the standard four-group setup.

    Raises:
        ValueError: When fewer than two non-empty groups are present, or
        when ``n_PC_classifier + 1`` exceeds the minimum train-fold size
        achievable under :class:`StratifiedKFold`.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X_all, y_all, class_sizes = _build_xy_per_layer(
        activations_by_group, position_index
    )
    _validate_pca_fit_feasible(n_PC_classifier, cv_n_splits, class_sizes)

    n_layers = X_all.shape[1]
    cv = StratifiedKFold(
        n_splits=cv_n_splits,
        shuffle=cv_shuffle,
        random_state=cv_random_state if cv_shuffle else None,
    )

    accuracies = np.zeros(n_layers, dtype=np.float64)
    for L in range(n_layers):
        X_L = X_all[:, L, :]  # (N, d_model)
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_PC_classifier, whiten=True)),
                (
                    "logreg",
                    LogisticRegression(
                        C=classifier_C,
                        max_iter=classifier_max_iter,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        scores = cross_val_score(pipe, X_L, y_all, cv=cv, scoring="accuracy")
        accuracies[L] = float(scores.mean())
    return accuracies


def compute_per_class_cv_accuracy_per_layer(
    activations_by_group: Dict[str, np.ndarray],
    position_index: int,
    *,
    n_PC_classifier: int,
    cv_n_splits: int,
    cv_shuffle: bool,
    cv_random_state: int,
    classifier_C: float,
    classifier_max_iter: int,
) -> Dict[str, np.ndarray]:
    """
    Per-class CV recall (= per-class accuracy) per layer.

    Mirrors :func:`compute_cv_accuracy_per_layer` (same per-fold
    ``StandardScaler → PCA(n_components=n_PC_classifier, whiten=True) →
    LogisticRegression`` pipeline, same CV splitter) but uses
    ``cross_val_predict`` so we get one prediction per sample across the
    held-out folds. From that we compute recall per class — i.e.
    ``P(pred = c | true = c)`` — which is the natural per-class analogue of
    the pooled accuracy.

    Returns:
        Mapping from class name (entry of :data:`CLASS_GROUPS`) to a
        length-``n_layers`` ``np.ndarray[float64]`` of recalls in ``[0, 1]``.
        Classes absent from ``activations_by_group`` (or with empty arrays)
        are omitted from the returned dict — downstream code should
        introspect the keys rather than assume all four are present.

    Raises:
        ValueError: Same conditions as :func:`compute_cv_accuracy_per_layer`
        — fewer than two non-empty groups, or PCA fit infeasible under the
        chosen ``cv_n_splits``.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X_all, y_all, class_sizes = _build_xy_per_layer(
        activations_by_group, position_index
    )
    _validate_pca_fit_feasible(n_PC_classifier, cv_n_splits, class_sizes)

    n_layers = X_all.shape[1]
    cv = StratifiedKFold(
        n_splits=cv_n_splits,
        shuffle=cv_shuffle,
        random_state=cv_random_state if cv_shuffle else None,
    )

    present_classes = [
        cls_id for cls_id, n in enumerate(class_sizes) if n > 0
    ]
    per_class = {
        CLASS_GROUPS[cls_id]: np.zeros(n_layers, dtype=np.float64)
        for cls_id in present_classes
    }

    for L in range(n_layers):
        X_L = X_all[:, L, :]
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_PC_classifier, whiten=True)),
                (
                    "logreg",
                    LogisticRegression(
                        C=classifier_C,
                        max_iter=classifier_max_iter,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        y_pred = cross_val_predict(pipe, X_L, y_all, cv=cv)
        for cls_id in present_classes:
            mask = y_all == cls_id
            per_class[CLASS_GROUPS[cls_id]][L] = float(
                (y_pred[mask] == cls_id).mean()
            )
    return per_class


# Conditions handled by the per-condition binary classifier (one classifier
# per entry — GT 0 vs GT 1 inside that condition).
BINARY_CONDITIONS: Tuple[str, ...] = ("control", "experiment")


def compute_binary_per_condition_cv_accuracy_per_layer(
    activations_by_group: Dict[str, np.ndarray],
    position_index: int,
    *,
    n_PC_classifier: int,
    cv_n_splits: int,
    cv_shuffle: bool,
    cv_random_state: int,
    classifier_C: float,
    classifier_max_iter: int,
) -> Dict[str, np.ndarray]:
    """
    Two-classifier variant: one binary classifier per condition.

    For each condition in :data:`BINARY_CONDITIONS` (``control`` and
    ``experiment``), we stack the two ``GT=0`` / ``GT=1`` activation arrays
    inside that condition into a balanced two-class problem, then fit the
    same per-fold pipeline as :func:`compute_cv_accuracy_per_layer`
    (``StandardScaler → PCA(n_components=n_PC_classifier, whiten=True) →
    LogisticRegression``) inside a stratified k-fold CV. Chance is ``0.5``
    for the standard balanced binary setup.

    Returns:
        Mapping ``"control" / "experiment" -> length-n_layers
        np.ndarray[float64]`` of mean CV accuracies in ``[0, 1]``. A
        condition is silently skipped (omitted from the returned dict) when
        either of its GT groups is empty, when its smaller GT group has
        fewer than ``cv_n_splits`` samples (StratifiedKFold would leave
        one fold single-class and LogisticRegression would return NaN),
        or when PCA fit is infeasible under the chosen ``cv_n_splits``.

    Raises:
        ValueError: When ``activations_by_group`` is missing *both*
        conditions, i.e. neither condition has both GT halves present
        (otherwise there's nothing to train on).
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    n_layers: Optional[int] = None
    out: Dict[str, np.ndarray] = {}
    for cond in BINARY_CONDITIONS:
        a0 = activations_by_group.get(f"{cond}_GT_0")
        a1 = activations_by_group.get(f"{cond}_GT_1")
        if a0 is None or a1 is None or a0.shape[0] == 0 or a1.shape[0] == 0:
            continue
        # Same StratifiedKFold feasibility guard as the 4-class path, plus
        # an extra binary-only check: each class needs at least ``cv_n_splits``
        # samples or one fold's train portion ends up single-class and
        # LogisticRegression returns NaN for that fold (poisoning the mean).
        # The 4-class path tolerates a 1-sample minority because the other
        # classes keep the train fold multi-class; binary has nowhere to hide.
        sizes = [int(a0.shape[0]), int(a1.shape[0])]
        if min(sizes) < cv_n_splits:
            continue
        try:
            _validate_pca_fit_feasible(n_PC_classifier, cv_n_splits, sizes)
        except ValueError:
            # Skip this condition silently — caller can detect via the
            # returned dict's keys. We don't want one degenerate condition
            # to kill the figure for the other.
            continue
        # Stack: GT=0 first (label 0), then GT=1 (label 1).
        X_a0 = a0[:, position_index, :, :].astype(np.float32, copy=False)
        X_a1 = a1[:, position_index, :, :].astype(np.float32, copy=False)
        X = np.concatenate([X_a0, X_a1], axis=0)  # (N, n_layers, d_model)
        y = np.concatenate(
            [np.zeros(X_a0.shape[0], dtype=np.int64),
             np.ones(X_a1.shape[0], dtype=np.int64)]
        )

        if n_layers is None:
            n_layers = X.shape[1]
        elif X.shape[1] != n_layers:
            raise ValueError(
                f"Layer count mismatch between conditions: "
                f"previous n_layers={n_layers}, "
                f"{cond} sees {X.shape[1]}."
            )

        cv = StratifiedKFold(
            n_splits=cv_n_splits,
            shuffle=cv_shuffle,
            random_state=cv_random_state if cv_shuffle else None,
        )
        accs = np.zeros(n_layers, dtype=np.float64)
        for L in range(n_layers):
            X_L = X[:, L, :]
            pipe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=n_PC_classifier, whiten=True)),
                    (
                        "logreg",
                        LogisticRegression(
                            C=classifier_C,
                            max_iter=classifier_max_iter,
                            solver="lbfgs",
                        ),
                    ),
                ]
            )
            scores = cross_val_score(pipe, X_L, y, cv=cv, scoring="accuracy")
            accs[L] = float(scores.mean())
        out[cond] = accs

    if not out:
        group_sizes = {
            g: (int(arr.shape[0]) if arr is not None else 0)
            for g, arr in activations_by_group.items()
        }
        raise ValueError(
            "No condition has both GT halves present with enough samples "
            f"(cv_n_splits={cv_n_splits}) to train a binary classifier. "
            f"Group sizes: {group_sizes}"
        )
    return out
