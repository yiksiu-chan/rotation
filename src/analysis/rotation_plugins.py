"""
Pluggable rotation metrics for comparing contrast directions across layers.

Every per-``(layer, position)`` cosine curve is reduced to one or more scalar
"rotation" scores; the analysis stage stores values for **every** registered
metric (so downstream consumers can pick a different one without re-running),
while a single metric — ``cfg.rotation_metric`` — drives:

- the "best token position" used to choose PCA grids in the analysis stage, and
- the "best token position" / row ordering of the cross-prompt heatmap in the
  paper-plot stage.

Each metric also has a stable short index (``r1``, ``r2``, …) used as the column
header in the cosine-heatmap "value" panel; see :data:`ROTATION_METRIC_ORDER`.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Protocol, Sequence, Tuple

import numpy as np

RotationMetricCallable = Callable[..., float]


class RotationMetricFn(Protocol):
    """
    Callable protocol for reducing a per-layer cosine curve to one scalar.

    Larger output should mean **more** rotation, so that the "best token
    position" selection (``argmax``) is a maximum-rotation pick.
    """

    def __call__(
        self,
        cos_sim_across_layers: Sequence[float],
        *,
        token_position_index: int,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        """Return a scalar rotation score (larger = stronger rotation)."""
        ...


def late_layer_anti_alignment(
    cos_sim_across_layers: Sequence[float],
    *,
    token_position_index: int = 0,
    metadata: dict[str, Any] | None = None,
) -> float:
    """
    ``1 - mean(cos_sim over the late half of layers)``.

    Late-half: trailing ``ceil(n_layers / 2)`` layers. Override the half via
    ``metadata={"late_fraction": <0..1>}``. Score is in ``[0, 2]`` (modulo
    numerical noise); larger = more anti-aligned in late layers.

    Note that this metric does **not** require a sign flip from early to late
    layers — a position that is anti-aligned from layer 0 onward will score
    near 2 even though it never rotated.
    """
    del token_position_index
    n = len(cos_sim_across_layers)
    if n == 0:
        return 0.0
    late_fraction = 0.5
    if metadata and "late_fraction" in metadata:
        late_fraction = float(metadata["late_fraction"])
        if not 0.0 < late_fraction <= 1.0:
            raise ValueError(f"late_fraction must be in (0, 1]; got {late_fraction!r}")

    cutoff = max(1, int(round(n * (1.0 - late_fraction))))
    late = list(cos_sim_across_layers)[cutoff:]
    if not late:
        late = list(cos_sim_across_layers)[-1:]
    return 1.0 - float(sum(late) / len(late))


def largest_drop_any_early_to_late(
    cos_sim_across_layers: Sequence[float],
    *,
    token_position_index: int = 0,
    metadata: dict[str, Any] | None = None,
) -> float:
    """
    ``r = max_{i<j} (s_i - s_j)`` — largest descent across layers.

    Measures rotation as *change*: a position that is anti-aligned throughout
    scores 0 (no descent), while a position that goes ``+1 → -1`` scores 2.
    Returns 0 for ``n < 2``. Result can be negative for a strictly ascending
    curve (the max of all-negative differences is the least-negative one).
    """
    del token_position_index, metadata
    s = np.asarray(list(cos_sim_across_layers), dtype=np.float64)
    n = s.size
    if n < 2:
        return 0.0
    diffs = s[:, None] - s[None, :]
    iu = np.triu_indices(n, k=1)
    return float(diffs[iu].max())


def largest_drop_any_early_to_late_signed(
    cos_sim_across_layers: Sequence[float],
    *,
    token_position_index: int = 0,
    metadata: dict[str, Any] | None = None,
) -> float:
    """
    ``r = max_{i<j, s_i>0, s_j<0} (s_i - s_j)`` — largest *sign-flipping* drop.

    Stricter variant of :func:`largest_drop_any_early_to_late`: only counts
    pairs where the early layer is positive and the later layer is negative,
    so a curve that descends but never crosses zero scores 0. Returns 0 if no
    positive→negative pair exists.
    """
    del token_position_index, metadata
    s = np.asarray(list(cos_sim_across_layers), dtype=np.float64)
    n = s.size
    if n < 2:
        return 0.0
    pos_idx = np.where(s > 0.0)[0]
    neg_idx = np.where(s < 0.0)[0]
    if pos_idx.size == 0 or neg_idx.size == 0:
        return 0.0
    best = 0.0
    for i in pos_idx:
        for j in neg_idx:
            if j <= i:
                continue
            d = float(s[i] - s[j])
            if d > best:
                best = d
    return best


def first_to_final_cosine_drop(
    cos_sim_across_layers: Sequence[float],
    *,
    token_position_index: int = 0,
    metadata: dict[str, Any] | None = None,
) -> float:
    """
    ``s_0 - s_final`` — drop from the first to the final layer.

    This is the reviewer's "final-layer cosine drop" metric in its most direct
    form. Positive values mean the two condition directions are less aligned at
    the final layer than at the first layer. It can be negative when alignment
    increases across layers.
    """
    del token_position_index, metadata
    s = list(cos_sim_across_layers)
    if len(s) < 2:
        return 0.0
    return float(s[0] - s[-1])


def minimum_cosine_rotation(
    cos_sim_across_layers: Sequence[float],
    *,
    token_position_index: int = 0,
    metadata: dict[str, Any] | None = None,
) -> float:
    """
    ``1 - min_l s_l`` — strongest anti-alignment reached at any layer.

    A value of 0 means the curve never drops below perfect alignment; a value
    near 2 means at least one layer reaches near-complete sign reversal.
    """
    del token_position_index, metadata
    s = list(cos_sim_across_layers)
    if not s:
        return 0.0
    return 1.0 - float(min(s))


def late_layer_rotation_auc(
    cos_sim_across_layers: Sequence[float],
    *,
    token_position_index: int = 0,
    metadata: dict[str, Any] | None = None,
) -> float:
    """
    Mean ``1 - cosine`` over the late half of layers.

    This operationalizes an area-under-rotation-curve score after the early
    class-organization phase: high values require sustained late-layer
    misalignment rather than a single-layer dip. The late-half cutoff matches
    :func:`late_layer_anti_alignment`; this separate name makes the reviewer-
    suggested AUC formulation explicit in outputs.
    """
    return late_layer_anti_alignment(
        cos_sim_across_layers,
        token_position_index=token_position_index,
        metadata=metadata,
    )


def early_to_late_minimum_cosine_drop(
    cos_sim_across_layers: Sequence[float],
    *,
    token_position_index: int = 0,
    metadata: dict[str, Any] | None = None,
) -> float:
    """
    ``max(early-half cosine) - min(late-half cosine)``.

    This is a drop-based version of the reviewer's "minimum cosine" idea. It
    rewards curves that start with an aligned reference direction and later
    reach a low cosine, while not rewarding curves that are already low or
    anti-aligned from the beginning.
    """
    del token_position_index, metadata
    s = np.asarray(list(cos_sim_across_layers), dtype=np.float64)
    if s.size < 2:
        return 0.0
    cutoff = max(1, int(round(s.size * 0.5)))
    early = s[:cutoff]
    late = s[cutoff:]
    if late.size == 0:
        late = s[-1:]
    return float(np.max(early) - np.min(late))


def post_separation_drop_auc(
    cos_sim_across_layers: Sequence[float],
    *,
    token_position_index: int = 0,
    metadata: dict[str, Any] | None = None,
) -> float:
    """
    Mean late-half drop from the best early-half alignment.

    Formally, ``mean_late(max(early cosine) - cosine_l)``. This implements a
    sustained post-separation rotation score: a brief dip is insufficient, and
    curves that start low receive little credit compared with curves that move
    from high early alignment to low late alignment.
    """
    del token_position_index, metadata
    s = np.asarray(list(cos_sim_across_layers), dtype=np.float64)
    if s.size < 2:
        return 0.0
    cutoff = max(1, int(round(s.size * 0.5)))
    early = s[:cutoff]
    late = s[cutoff:]
    if late.size == 0:
        late = s[-1:]
    return float(np.mean(np.max(early) - late))


# --------------------------------------------------------------------------- #
# Registry                                                                    #
# --------------------------------------------------------------------------- #


# Stable order: index → registry name. Dictates the ``r{i}`` short labels
# shown in cosine-similarity heatmaps; appending new metrics is fine,
# reordering breaks the index ↔ name mapping.
ROTATION_METRIC_ORDER: Tuple[str, ...] = (
    "late_layer_anti_alignment",
    "largest_drop_any_early_to_late",
    "largest_drop_any_early_to_late_signed",
    "first_to_final_cosine_drop",
    "minimum_cosine_rotation",
    "late_layer_rotation_auc",
    "early_to_late_minimum_cosine_drop",
    "post_separation_drop_auc",
)


ROTATION_METRIC_REGISTRY: Dict[str, RotationMetricCallable] = {
    "late_layer_anti_alignment": late_layer_anti_alignment,
    "largest_drop_any_early_to_late": largest_drop_any_early_to_late,
    "largest_drop_any_early_to_late_signed": largest_drop_any_early_to_late_signed,
    "first_to_final_cosine_drop": first_to_final_cosine_drop,
    "minimum_cosine_rotation": minimum_cosine_rotation,
    "late_layer_rotation_auc": late_layer_rotation_auc,
    "early_to_late_minimum_cosine_drop": early_to_late_minimum_cosine_drop,
    "post_separation_drop_auc": post_separation_drop_auc,
}


# Default if cfg.rotation_metric is missing.
DEFAULT_ROTATION_METRIC_NAME: str = "late_layer_anti_alignment"


def resolve_rotation_metric(name: str | None) -> Tuple[str, RotationMetricCallable]:
    """
    Look up ``name`` in :data:`ROTATION_METRIC_REGISTRY`.

    Returns:
        ``(resolved_name, callable)``. ``None`` falls back to
        :data:`DEFAULT_ROTATION_METRIC_NAME`.

    Raises:
        ValueError: If ``name`` is not registered.
    """
    resolved = name or DEFAULT_ROTATION_METRIC_NAME
    if resolved not in ROTATION_METRIC_REGISTRY:
        raise ValueError(
            f"Unknown rotation_metric {resolved!r}. "
            f"Registered: {sorted(ROTATION_METRIC_REGISTRY)}."
        )
    return resolved, ROTATION_METRIC_REGISTRY[resolved]


def metric_short_label(name: str) -> str:
    """
    Return the ``r{i}`` short label for ``name`` (1-based index in
    :data:`ROTATION_METRIC_ORDER`).
    """
    if name not in ROTATION_METRIC_ORDER:
        raise ValueError(f"Metric {name!r} not in ROTATION_METRIC_ORDER.")
    return f"r{ROTATION_METRIC_ORDER.index(name) + 1}"


def resolve_displayed_metrics(
    display_mode: str,
    chosen_metric_name: str,
) -> Tuple[str, ...]:
    """
    Pick the metric set shown in cosine-heatmap ``r``-blocks, based on
    ``cfg.heatmap_r_metric_display``.

    Args:
        display_mode: ``"all"`` → every metric in
            :data:`ROTATION_METRIC_ORDER`; ``"chosen_metric"`` → only
            ``chosen_metric_name``.
        chosen_metric_name: The driving metric (``cfg.rotation_metric``);
            already validated by :func:`resolve_rotation_metric`.

    Returns:
        Tuple of metric names in the order they should appear as columns.

    Raises:
        ValueError: If ``display_mode`` is not one of the recognized modes.
    """
    if display_mode == "all":
        return ROTATION_METRIC_ORDER
    if display_mode == "chosen_metric":
        return (chosen_metric_name,)
    raise ValueError(
        f"Unknown heatmap_r_metric_display {display_mode!r}; "
        "expected 'all' or 'chosen_metric'."
    )
