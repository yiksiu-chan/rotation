"""
Plotly helpers for the analysis stage.

Three plot kinds:

- :func:`plot_cos_sim_heatmap` — single heatmap of cosine similarities with
  token positions on the y-axis and layers on the x-axis. A trailing block of
  ``r1, r2, …`` columns shows each registered rotation metric's value per
  row, rounded to one decimal. Output formats are controlled by the caller
  (typically ``cfg.cos_sim_heatmap_formats``); the project default is
  ``["svg", "png"]`` (vector + high-DPI raster).
- :func:`plot_pca_grid_for_position` — grid of per-layer PCA scatter plots for
  one token position, with the four ``(condition, GT)`` groups distinguished by
  marker color and symbol. Exported as ``.pdf`` + ``.svg``.
- :func:`plot_manifold_for_position` — single 2D or 3D manifold scatter
  (PCA / t-SNE / UMAP) at one token position. Activations are concatenated
  across samples × layers × ``(condition, GT)`` groups before fitting the
  reducer once; per-trace coloring encodes ``(condition, GT)`` via the base
  color and **layer index** via a light → dark shade gradient. 2D writes
  ``.pdf + .svg``; 3D additionally writes interactive ``.html``. Optional
  layer-traversal lines connect the markers of caller-supplied
  ``(group, sample_idx)`` rows across layers; line dash style encodes GT via
  the caller's ``gt_style`` mode (``"line_style"``: solid / dash;
  ``"marker_shape"``: solid, with the marker symbol carrying the GT
  distinction in the set-layer subspace).

Aesthetics (colors, symbols, alpha, n_components) for the PCA plots are
sourced from ``cfg.pca_plot_style``; manifold-plot aesthetics (per-group
color/symbol, layer-shade range) live in ``cfg.manifold_plot_style``. Both
are merged onto module-level defaults so fields can be overridden piecemeal.
The list of highlighted target traversals is **not** part of the style dict —
it is passed in explicitly as ``targets`` by the caller (e.g.
:mod:`src.run_microscopic`'s ``--targets`` flag).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# PCA aesthetics resolution                                                   #
# --------------------------------------------------------------------------- #


_DEFAULT_PCA_STYLE: Dict[str, Any] = {
    "n_components": 2,
    "marker_alpha": 0.65,
    "marker_size": 5,
    "control_GT_0":    {"color": "goldenrod",  "symbol": "circle"},
    "control_GT_1":    {"color": "goldenrod",  "symbol": "star"},
    "experiment_GT_0": {"color": "royalblue", "symbol": "circle"},
    "experiment_GT_1": {"color": "royalblue", "symbol": "star"},
}


def resolve_pca_style(style_overrides: Dict[str, Any] | None) -> Dict[str, Any]:
    """
    Merge user overrides on top of :data:`_DEFAULT_PCA_STYLE`.

    Per-group dicts (``control_GT_0`` etc.) are merged shallowly so a user can
    override only ``color`` and inherit ``symbol``.
    """
    merged: Dict[str, Any] = dict(_DEFAULT_PCA_STYLE)
    overrides = style_overrides or {}
    for key, val in overrides.items():
        if (
            isinstance(val, dict)
            and isinstance(merged.get(key), dict)
        ):
            merged[key] = {**merged[key], **val}
        else:
            merged[key] = val
    return merged


# --------------------------------------------------------------------------- #
# Plot writers                                                                #
# --------------------------------------------------------------------------- #


_VECTOR_FORMATS: frozenset = frozenset({"pdf", "svg", "eps"})
_RASTER_FORMATS: frozenset = frozenset({"png", "jpg", "jpeg", "webp"})
_KALEIDO_FORMATS: frozenset = _VECTOR_FORMATS | _RASTER_FORMATS
_INTERACTIVE_FORMATS: frozenset = frozenset({"html"})
_SUPPORTED_FORMATS: frozenset = _KALEIDO_FORMATS | _INTERACTIVE_FORMATS


def _normalize_format(fmt: str) -> str:
    """Strip leading dot, lowercase, and validate against the supported set."""
    f = str(fmt).lower().lstrip(".")
    if f == "jpg":
        f = "jpeg"
    if f not in _SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported plot format {fmt!r}; expected one of "
            f"{sorted(_SUPPORTED_FORMATS)}."
        )
    return f


def _write_figure(
    fig,
    basepath: Path,
    formats: Sequence[str],
    scale: float = 2.0,
    png_scale: float = 4.0,
) -> None:
    """
    Export ``fig`` to ``{basepath}.<fmt>`` for each ``fmt`` in ``formats``.

    Vector formats (``pdf`` / ``svg`` / ``eps``) and raster formats (``png`` /
    ``jpg`` / ``webp``) go through Plotly's kaleido-backed
    :func:`plotly.io.write_image`; raster formats use ``png_scale`` for
    higher DPI. ``html`` writes an interactive standalone file via
    :meth:`plotly.graph_objects.Figure.write_html`. Unknown formats raise
    :class:`ValueError`. Repeats and aliases (``"jpg"`` / ``"jpeg"``) are
    silently de-duplicated.
    """
    import plotly.io as pio  # lazy: heavy import, only needed at plot time

    basepath.parent.mkdir(parents=True, exist_ok=True)
    seen: set = set()
    for fmt in formats:
        fmt_norm = _normalize_format(fmt)
        if fmt_norm in seen:
            continue
        seen.add(fmt_norm)
        out = f"{basepath}.{fmt_norm}"
        if fmt_norm == "html":
            fig.write_html(out, include_plotlyjs="cdn", full_html=True)
        elif fmt_norm in _RASTER_FORMATS:
            pio.write_image(fig, out, scale=png_scale)
        else:  # vector
            pio.write_image(fig, out, scale=scale)


def plot_cos_sim_heatmap(
    cos_sim: np.ndarray,
    position_labels: Sequence[str],
    basepath: Path,
    title: str,
    metric_values_per_row: Optional[Sequence[Sequence[float]]] = None,
    metric_column_labels: Optional[Sequence[str]] = None,
    formats: Sequence[str] = ("svg", "png"),
) -> None:
    """
    Cross-position cosine heatmap (y-axis = token positions, x-axis = layers).

    When ``metric_values_per_row`` and ``metric_column_labels`` are provided, a
    trailing block of columns is appended at the right with each registered
    rotation metric's value per row, rounded to one decimal. Those cells are
    annotation-only (no fill); they never share the cosine ``[-1, 1]``
    colorscale. With a single column (``len(metric_column_labels) == 1``)
    the column header is rendered as a plain ``r`` floated above the column
    instead of an ``rN`` tick at the bottom.

    Args:
        cos_sim: ``(n_layers, n_positions)`` cosine matrix as produced by
            :func:`src.analysis.metrics.compute_cos_sim_matrix`.
        position_labels: Length-``n_positions`` human-readable row names
            (y-axis tick labels).
        basepath: Output path **without extension**; one file per entry in
            ``formats`` is written under it.
        title: Plot title.
        metric_values_per_row: Optional ``(n_positions, n_metrics)`` matrix of
            scalar metric values; row ``i`` corresponds to ``position_labels[i]``,
            column ``j`` to ``metric_column_labels[j]``.
        metric_column_labels: Optional length-``n_metrics`` column headers
            (e.g. ``["r1", "r2", "r3"]``). Required when
            ``metric_values_per_row`` is provided.
        formats: Output file formats (any combo of ``pdf`` / ``svg`` /
            ``eps`` / ``png`` / ``jpg`` / ``webp`` / ``html``). Defaults to
            ``("svg", "png")`` — the project's prior on-disk convention.
    """
    import plotly.graph_objects as go  # lazy

    z = cos_sim.T  # (n_positions, n_layers) — y=positions, x=layers
    n_positions, n_layers = z.shape
    y_labels = list(position_labels)

    has_metric_block = (
        metric_values_per_row is not None
        and metric_column_labels is not None
        and len(metric_column_labels) > 0
    )

    n_metrics = len(metric_column_labels) if has_metric_block else 0
    single_metric = has_metric_block and n_metrics == 1
    gap = 1 if has_metric_block else 0  # blank visual gap between the two blocks
    n_cols_total = n_layers + gap + n_metrics

    # Tick labels: layer numbers at every ~n_layers/8 step; r-cols labelled
    # by metric_column_labels (suppressed in single-metric mode — replaced by
    # a floating "r" header above the column); gap untouched.
    layer_tick_step = max(1, n_layers // 8)
    layer_tickvals = list(range(0, n_layers, layer_tick_step))
    layer_ticktext = [str(v) for v in layer_tickvals]
    metric_tickvals: List[int] = []
    if has_metric_block:
        metric_tickvals = [n_layers + gap + j for j in range(n_metrics)]
        if not single_metric:
            layer_tickvals = layer_tickvals + metric_tickvals
            layer_ticktext = layer_ticktext + list(metric_column_labels)

    # Cosine block: real values on x=0..n_layers-1 (RdBu, [-1, 1]).
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=list(range(n_layers)),
            y=y_labels,
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Cos sim"),
        )
    )

    if has_metric_block:
        metric_arr = np.asarray(metric_values_per_row, dtype=np.float64)
        if metric_arr.shape != (n_positions, n_metrics):
            raise ValueError(
                f"metric_values_per_row shape {metric_arr.shape} "
                f"!= ({n_positions}, {n_metrics})"
            )
        # Per-cell value annotations (1 decimal). The r-block is annotation-only
        # — no background fill — so it inherits the white plot background.
        for i, lbl in enumerate(y_labels):
            for j, col_x in enumerate(metric_tickvals):
                fig.add_annotation(
                    x=col_x,
                    y=lbl,
                    xref="x",
                    yref="y",
                    text=f"{metric_arr[i, j]:.1f}",
                    showarrow=False,
                    font=dict(size=11, color="black"),
                )
        if single_metric:
            fig.add_annotation(
                x=metric_tickvals[0],
                y=1.0,
                xref="x",
                yref="paper",
                yanchor="bottom",
                text="r",
                showarrow=False,
                font=dict(size=12, color="black"),
            )

    fig.update_layout(
        title=title,
        xaxis_title="Layer",
        yaxis_title="Token position",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            tickmode="array",
            tickvals=layer_tickvals,
            ticktext=layer_ticktext,
            range=[-0.5, n_cols_total - 0.5],
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
        ),
        height=max(360, 40 + 24 * n_positions),
        width=max(420, 80 + 22 * n_layers) + (28 * n_metrics + 30 if has_metric_block else 0),
        font=dict(size=12),
    )
    _write_figure(fig, basepath, formats)


def _group_traces_for_pca(
    proj: np.ndarray,
    group_sizes: Dict[str, int],
    style: Dict[str, Any],
    show_legend_for: Sequence[str] = (),
    legend_label_map: Dict[str, str] | None = None,
) -> List[Any]:
    """
    Build one Plotly scatter trace per ``(condition, GT)`` group from a single
    layer's PCA projection.

    Args:
        proj: ``(N, n_components)`` projection of the stacked activations.
        group_sizes: Ordered ``{group_name: count}``; rows of ``proj`` are
            sliced in this order. Group names must match keys in ``style``
            (e.g. ``"control_GT_0"``).
        style: Output of :func:`resolve_pca_style`.
        show_legend_for: Group names whose trace should advertise itself in
            the figure legend. Pass an empty tuple to suppress legend entries
            for this layer (used for all but the first layer to avoid
            duplicate entries).
        legend_label_map: Optional ``{group_name: display_label}`` mapping
            applied to the legend / hover labels (e.g.
            ``{"control_GT_0": "control / False"}``). Group keys absent from
            the map fall through to the raw group name.

    Returns:
        List of ``go.Scatter`` traces.
    """
    import plotly.graph_objects as go

    traces: List[Any] = []
    cursor = 0
    show_legend_for_set = set(show_legend_for)
    for group_name, n in group_sizes.items():
        if n == 0:
            continue
        slc = proj[cursor : cursor + n]
        cursor += n
        spec = style.get(group_name, {})
        display_name = (
            legend_label_map.get(group_name, group_name)
            if legend_label_map
            else group_name
        )
        traces.append(
            go.Scatter(
                x=slc[:, 0],
                y=slc[:, 1] if proj.shape[1] >= 2 else np.zeros(n),
                mode="markers",
                marker=dict(
                    color=spec.get("color", "gray"),
                    symbol=spec.get("symbol", "circle"),
                    size=style.get("marker_size", 5),
                    opacity=style.get("marker_alpha", 0.65),
                    line=dict(width=0.3, color="darkslategrey"),
                ),
                name=display_name,
                legendgroup=display_name,
                showlegend=group_name in show_legend_for_set,
            )
        )
    return traces


def plot_pca_grid_for_position(
    activations_by_group: Dict[str, np.ndarray],
    position_index: int,
    position_label: str,
    basepath: Path,
    style: Dict[str, Any],
    title: str | None = None,
    legend_label_map: Dict[str, str] | None = None,
    formats: Optional[Sequence[str]] = None,
) -> None:
    """
    PCA scatter grid (one subplot per layer) for a single token position.

    Args:
        activations_by_group: Ordered mapping
            ``{"control_GT_0": ndarray(k, n_pos, n_layers, d), ...}`` for the
            four ``(condition, GT)`` splits. Group order in the returned plot
            follows iteration order of this dict.
        position_index: Column index ``p`` of the position to project on.
        position_label: Human-readable position name (e.g. ``"prompt-1"``);
            used only as the suptitle prefix when no ``title`` is given.
        basepath: Output path without extension.
        style: Output of :func:`resolve_pca_style`.
        title: Optional explicit figure title. Falls back to a position-only
            title when ``None``.
        legend_label_map: Optional ``{group_name: display_label}`` mapping
            for legend entries (e.g. relabel ``"control_GT_0"`` →
            ``"control / False"`` based on ``cfg.gt_0_label_map``).
        formats: Output file formats (e.g. ``("pdf", "svg", "png", "html")``).
            Defaults to ``("pdf", "svg")`` when ``None``. Honors any
            combination of vector / raster / interactive formats supported by
            :func:`_write_figure`.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from sklearn.decomposition import PCA

    fmts = tuple(formats) if formats else ("pdf", "svg")

    n_components = int(style.get("n_components", 2))
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1; got {n_components}")

    # Stack groups along axis 0 → (sum_k, n_pos, n_layers, d)
    group_names: List[str] = list(activations_by_group.keys())
    group_sizes = {g: activations_by_group[g].shape[0] for g in group_names}
    stacked = np.concatenate(
        [activations_by_group[g] for g in group_names], axis=0
    )  # (sum_k, n_pos, n_layers, d)
    n_layers = stacked.shape[2]
    d_model = stacked.shape[3]

    cols = 8
    rows = math.ceil(n_layers / cols)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"L{n}" for n in range(n_layers)],
        horizontal_spacing=0.035,
        vertical_spacing=0.08,
    )

    # Show one legend entry per group on the first layer's traces, suppress for the rest.
    first_layer_groups = [g for g in group_names if group_sizes.get(g, 0) > 0]
    for L in range(n_layers):
        slab = stacked[:, position_index, L, :].astype(np.float32)  # (sum_k, d)
        if slab.shape[0] < n_components:
            # Not enough rows for PCA at this layer; pad projection with zeros.
            proj = np.zeros((slab.shape[0], n_components), dtype=np.float32)
        else:
            proj = PCA(n_components=n_components).fit_transform(slab)
        traces = _group_traces_for_pca(
            proj,
            group_sizes,
            style,
            show_legend_for=first_layer_groups if L == 0 else (),
            legend_label_map=legend_label_map,
        )
        r, c = L // cols + 1, L % cols + 1
        for tr in traces:
            fig.add_trace(tr, row=r, col=c)

    fig.update_layout(
        height=140 * rows,
        width=160 * cols,
        font=dict(size=10),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
        margin=dict(t=40, b=20, l=20, r=20),
    )
    _write_figure(fig, basepath, fmts, png_scale=3.0)


# --------------------------------------------------------------------------- #
# Manifold (PCA / t-SNE / UMAP) aesthetics + writer                           #
# --------------------------------------------------------------------------- #


_DEFAULT_MANIFOLD_STYLE: Dict[str, Any] = {
    "marker_alpha": 0.7,
    "marker_size": 4,
    # Width of the dark outline drawn around every marker (Plotly's
    # ``marker.line.width``). Increase for stronger separation between
    # overlapping points; ``0`` = no outline. Applied to both the across-layer
    # and set-layer plot helpers.
    "marker_border_width": 0.3,
    # Outline width for the **target** markers in the set-layer plot. Targets
    # are visually heavier than the background scatter, so the default is a
    # touch bolder. Per-entry ``marker_border_width`` on a target overrides
    # this global default.
    "target_marker_border_width": 0.6,
    # Lightest shade (early layers) sits at this fraction along the per-group
    # colorscale; darkest shade (final layer) sits at 1.0. The default
    # colorscales are Plotly's perceptually-uniform single-hue scales so the
    # light → dark gradient stays distinguishable across all layers.
    "layer_shade_min": 0.25,
    # Per-(condition, GT) styling. ``color`` is the group's representative
    # hue (used for the legend / line fallback); ``colorscale`` is a Plotly
    # continuous colorscale name (e.g. ``"Blues"``, ``"Oranges"``, ``"Greens"``,
    # ``"Purples"``) used as the layer gradient. When ``colorscale`` is absent
    # the marker scale falls back to ``[[0, "white"], [1, color]]`` (lower
    # contrast — provided only for back-compat with custom groups).
    # ``symbol_3d`` overrides ``symbol`` when ``n_components == 3`` (Plotly's
    # 3D scatter does not support every 2D symbol — notably ``"star"``).
    "control_GT_0":         {"color": "goldenrod", "colorscale": "Oranges", "symbol": "circle"},
    "control_GT_1":         {"color": "goldenrod", "colorscale": "Oranges", "symbol": "star",   "symbol_3d": "diamond"},
    "experiment_GT_0":      {"color": "royalblue", "colorscale": "Blues",   "symbol": "circle"},
    "experiment_GT_1":      {"color": "royalblue", "colorscale": "Blues",   "symbol": "star",   "symbol_3d": "diamond"},
    # Experiment-fail subset (statements that succeeded in control but failed
    # the instruction in experiment). Drawn only when the matching pickles
    # exist; the cache stage emits them under the ``_fail`` filename suffix.
    "experiment_GT_0_fail": {"color": "purple",    "colorscale": "Purples", "symbol": "circle"},
    "experiment_GT_1_fail": {"color": "purple",    "colorscale": "Purples", "symbol": "star",   "symbol_3d": "diamond"},
}


# Default line width for the per-target traversal polyline in both manifold
# subspaces. Kept as a module constant rather than a style key so callers
# never have to look two places to disable a feature.
_MANIFOLD_TARGET_LINE_WIDTH: float = 1.0

# Per-GT dash style applied to the per-target traversal line when
# ``gt_style == "line_style"``. Mirrors :data:`src.analysis.microscopic._GT_DASH`.
_GT_DASH_MANIFOLD: Dict[int, str] = {1: "solid", 0: "dash"}

# Per-GT marker symbol override applied to the per-target traversal markers
# when ``gt_style == "marker_shape"``. 2D uses Plotly's ``"x"``; 3D falls
# back to ``"cross"`` (the closest valid 3D symbol).
_GT_MARKER_MANIFOLD_2D: Dict[int, str] = {1: "circle", 0: "x"}
_GT_MARKER_MANIFOLD_3D: Dict[int, str] = {1: "circle", 0: "cross"}


_VALID_3D_SYMBOLS: frozenset = frozenset(
    {
        "circle", "circle-open", "cross", "diamond", "diamond-open",
        "square", "square-open", "x",
    }
)


def resolve_manifold_style(style_overrides: Dict[str, Any] | None) -> Dict[str, Any]:
    """
    Merge user overrides on top of :data:`_DEFAULT_MANIFOLD_STYLE`.

    Per-group dicts (``control_GT_0`` …) are merged shallowly so a user can
    override e.g. only ``color`` and inherit ``symbol`` / ``symbol_3d``.
    """
    merged: Dict[str, Any] = {
        k: (dict(v) if isinstance(v, dict) else v)
        for k, v in _DEFAULT_MANIFOLD_STYLE.items()
    }
    overrides = style_overrides or {}
    for key, val in overrides.items():
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **val}
        else:
            merged[key] = val
    return merged


def _layer_color_fraction(layer_idx: int, n_layers: int, light_min: float) -> float:
    """Fraction along ``[white, base_color]`` for ``layer_idx`` (early → light)."""
    if n_layers <= 1:
        return 1.0
    return light_min + (1.0 - light_min) * layer_idx / (n_layers - 1)


def _resolve_manifold_symbol(spec: Dict[str, Any], n_components: int) -> str:
    """Pick the per-group marker symbol for a 2D vs 3D manifold trace."""
    if n_components == 3:
        sym = spec.get("symbol_3d", spec.get("symbol", "circle"))
        if sym not in _VALID_3D_SYMBOLS:
            sym = "diamond"
        return sym
    return spec.get("symbol", "circle")


def _gt_from_group(group_key: str) -> int:
    """``control_GT_0`` / ``experiment_GT_1_fail`` → ``0`` / ``1``."""
    if "_GT_0" in group_key:
        return 0
    if "_GT_1" in group_key:
        return 1
    raise ValueError(f"Cannot infer GT from group key {group_key!r}")


def _gt_style_target_aesthetics(
    group_key: str,
    gt_style: str,
    group_symbol: str,
    n_components: int,
) -> Tuple[str, str]:
    """
    Resolve ``(line_dash, marker_symbol)`` for a per-target manifold trace
    under the requested ``gt_style``.

    - ``"line_style"``: dash encodes GT (solid GT_1 / dash GT_0); marker
      keeps the per-group symbol.
    - ``"marker_shape"``: line is solid; marker is overridden by GT
      (circle GT_1 / x [2D] or cross [3D] GT_0).
    """
    if gt_style not in ("line_style", "marker_shape"):
        raise ValueError(
            f"gt_style must be 'line_style' | 'marker_shape'; got {gt_style!r}"
        )
    gt = _gt_from_group(group_key)
    if gt_style == "line_style":
        return _GT_DASH_MANIFOLD[gt], group_symbol
    table = _GT_MARKER_MANIFOLD_3D if n_components == 3 else _GT_MARKER_MANIFOLD_2D
    return "solid", table[gt]


def _reduce_to_manifold(
    method: str, n_components: int, X: np.ndarray
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Project ``X`` (``(n_points, d_model)``) into ``n_components`` with the
    requested manifold method (``"PCA"``, ``"t-SNE"``, or ``"UMAP"``).

    Returns ``(proj, variance_ratios)``. ``variance_ratios`` is the
    length-``n_components`` ``explained_variance_ratio_`` for PCA and
    ``None`` for t-SNE / UMAP (which have no closed-form variance attribution).
    Falls back to a zero ``proj`` and ``None`` ratios when the input is too
    small for the reducer to fit (mirrors the PCA grid's
    ``slab.shape[0] < n_components`` fallback).
    """
    if X.shape[0] < n_components + 1:
        return np.zeros((X.shape[0], n_components), dtype=np.float32), None

    if method == "PCA":
        from sklearn.decomposition import PCA  # lazy
        pca = PCA(n_components=n_components)
        proj = pca.fit_transform(X).astype(np.float32)
        return proj, np.asarray(pca.explained_variance_ratio_, dtype=np.float64)

    if method == "t-SNE":
        from sklearn.manifold import TSNE  # lazy
        # perplexity must be < n_samples; clamp into a sane range.
        perplexity = float(min(30, max(5, X.shape[0] // 4)))
        if X.shape[0] <= perplexity:
            perplexity = max(2.0, float(X.shape[0] - 1))
        proj = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            init="pca",
            random_state=0,
        ).fit_transform(X).astype(np.float32)
        return proj, None

    if method == "UMAP":
        try:
            import umap  # type: ignore  # lazy; provided by ``umap-learn``
        except ImportError as e:
            raise ImportError(
                "UMAP manifold plots require the 'umap-learn' package. "
                "Install with `pip install umap-learn`."
            ) from e
        n_neighbors = int(min(15, max(2, X.shape[0] - 1)))
        proj = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            random_state=0,
        ).fit_transform(X).astype(np.float32)
        return proj, None

    raise ValueError(f"Unknown manifold method: {method!r}")


def _axis_titles(method: str, n_components: int, variance_ratios) -> List[str]:
    """
    Build per-axis titles for a manifold scatter.

    PCA → ``f"PC{n}_{var*100:.1f}%variance"`` (per the project convention);
    t-SNE / UMAP → simple numbered axes (no variance attribution).
    """
    if variance_ratios is not None and method == "PCA":
        return [
            f"PC{i + 1}_{float(variance_ratios[i]) * 100:.1f}%variance"
            for i in range(n_components)
        ]
    return [f"{method} {i + 1}" for i in range(n_components)]


def plot_manifold_for_position(
    activations_by_group: Dict[str, np.ndarray],
    position_index: int,
    position_label: str,
    method: str,
    n_components: int,
    basepath: Path,
    style: Dict[str, Any],
    title: str | None = None,
    legend_label_map: Dict[str, str] | None = None,
    start_layer: int = -1,
    end_layer: int = -1,
    formats: Optional[Sequence[str]] = None,
    targets: Sequence[Tuple[str, int]] = (),
    gt_style: str = "line_style",
    bg_grey_scale: bool = False,
) -> None:
    """
    Single 2D or 3D manifold scatter for one token position.

    Activations from all configured ``(condition, GT)`` groups (the four base
    groups plus the optional ``experiment_GT_*_fail`` groups when their
    ``_fail`` pickles are present) are concatenated along ``axis=0``, sliced
    at ``position_index``, then flattened across ``samples × layers`` before
    fitting the reducer once on the resulting ``(sum_k * n_layers, d_model)``
    matrix. Each ``(group, layer)`` is then drawn with the group's per-trace
    Plotly colorscale shaded by ``layer_idx`` (light = early, dark = late)
    and the group's marker symbol (or ``symbol_3d`` in 3D).

    Legend handling: the figure carries one stub trace per non-empty group
    (drawn with the group's solid base color + symbol) so the legend always
    has a clean per-group entry — even when the layer-shaded marker arrays
    fail to produce a usable legend swatch on their own. Up to three
    layer-shade colorbars (``control``, ``experiment``, ``experiment-fail``)
    are added on the right margin so the layer index along each colorscale
    is readable directly from the figure.

    For each ``(group_key, sample_idx)`` in ``targets`` a polyline is drawn
    connecting the markers of that row across the (possibly sliced) layer
    range. The polyline uses the group's base color and a fixed width of
    :data:`_MANIFOLD_TARGET_LINE_WIDTH`; its dash style is derived from
    ``gt_style`` + the group's GT (see :func:`_gt_style_target_aesthetics`).
    Skipped targets emit a one-line diagnostic to stdout so missing ``_fail``
    pickles or out-of-range ``sample_idx`` values are easy to spot.

    ``start_layer`` / ``end_layer`` (both inclusive, both default ``-1``)
    narrow the layer range fed into the reducer. ``-1`` on either bound
    means "no bound" (``-1, -1`` = all layers, ``-1, k`` = ``[0, k]``,
    ``j, -1`` = ``[j, n_layers - 1]``). Hover labels and the right-margin
    colorbars tick at **absolute** layer indices, so narrowing the range
    never confuses the layer numbering.

    Args:
        activations_by_group: Ordered mapping
            ``{"control_GT_0": ndarray(k, n_pos, n_layers, d), ...}``.
        position_index: Column index of the token position to project.
        position_label: Human-readable position name (used in titles / files).
        method: One of ``"PCA"``, ``"t-SNE"``, ``"UMAP"``.
        n_components: ``2`` or ``3``.
        basepath: Output path **without extension**; ``.pdf`` + ``.svg`` are
            always written, ``.html`` is additionally written for 3D.
        style: Output of :func:`resolve_manifold_style`.
        title: Optional explicit figure title.
        legend_label_map: Optional ``{group_name: display_label}`` mapping
            for legend / hover labels.
        targets: ``(group_key, sample_idx)`` pairs to draw as highlighted
            layer-traversal lines. Group keys must be in user-facing form
            (e.g. ``"control_GT_0"``, ``"experiment_GT_1_fail"``) and match a
            key in ``activations_by_group``; out-of-range / unknown entries
            are skipped with a printed note.
        gt_style: ``"line_style"`` (default) encodes GT via line dash
            (solid GT_1 / dash GT_0); ``"marker_shape"`` keeps the line solid
            (markers carry GT in the set-layer subspace — the across-layer
            subspace draws lines only, so ``marker_shape`` is equivalent to
            "all solid" here).
    """
    import plotly.graph_objects as go  # lazy

    if n_components not in (2, 3):
        raise ValueError(f"n_components must be 2 or 3; got {n_components}")
    if method not in ("PCA", "t-SNE", "UMAP"):
        raise ValueError(f"method must be PCA / t-SNE / UMAP; got {method!r}")

    fmts = tuple(formats) if formats else ("pdf", "svg")

    group_names: List[str] = list(activations_by_group.keys())
    group_sizes = {g: activations_by_group[g].shape[0] for g in group_names}
    if sum(group_sizes.values()) == 0:
        raise ValueError("activations_by_group is empty across all groups.")

    stacked = np.concatenate(
        [activations_by_group[g] for g in group_names], axis=0
    )  # (sum_k, n_pos, n_layers_full, d_model)
    sum_k, _, n_layers_full, d_model = stacked.shape

    # Resolve ``start_layer`` / ``end_layer`` against the actual ``n_layers``
    # of this run. Either bound at ``-1`` means "no bound".
    layer_lo = 0 if int(start_layer) == -1 else int(start_layer)
    layer_hi = n_layers_full - 1 if int(end_layer) == -1 else int(end_layer)
    if not (0 <= layer_lo <= layer_hi <= n_layers_full - 1):
        raise ValueError(
            f"Invalid layer range start_layer={start_layer}, end_layer={end_layer} "
            f"resolved to [{layer_lo}, {layer_hi}] for n_layers={n_layers_full}"
        )
    n_layers = layer_hi - layer_lo + 1

    # Slice the chosen position and the requested layer range, then flatten
    # samples × layers so the reducer sees one combined point cloud and the
    # per-layer projections share an axis frame across groups.
    slab = stacked[:, position_index, layer_lo : layer_hi + 1, :].astype(np.float32)
    flat = slab.reshape(sum_k * n_layers, d_model)
    proj_flat, variance_ratios = _reduce_to_manifold(method, n_components, flat)
    proj = proj_flat.reshape(sum_k, n_layers, n_components)
    axis_titles = _axis_titles(method, n_components, variance_ratios)
    marker_border_width = float(style.get("marker_border_width", 0.3))

    fig = go.Figure()
    light_min = float(style.get("layer_shade_min", 0.25))
    marker_size = style.get("marker_size", 4)
    marker_alpha = style.get("marker_alpha", 0.7)

    groups_with_data: List[str] = [
        g for g in group_names if group_sizes.get(g, 0) > 0
    ]

    # Legend stubs: one invisible point per group, drawn in the group's solid
    # ``base_color`` + symbol. The per-(group, layer) marker traces below all
    # set ``showlegend=False`` and share ``legendgroup`` with the stub, so the
    # legend stays clean (one entry per group) and clicks still toggle the
    # whole group's points. This is more reliable than relying on the legend
    # swatch of a marker.colorscale-driven trace, which can render incorrectly
    # for some scale + array combinations (and was the cause of missing
    # ``experiment-fail / *`` legend rows).
    for g in groups_with_data:
        spec = style.get(g, {})
        base_color = spec.get("color", "gray")
        symbol = _resolve_manifold_symbol(spec, n_components)
        display = (legend_label_map or {}).get(g, g)
        stub_kwargs: Dict[str, Any] = dict(
            mode="markers",
            marker=dict(
                color=base_color,
                symbol=symbol,
                size=marker_size + 1,
                opacity=1.0,
                line=dict(
                    width=max(0.4, marker_border_width + 0.2),
                    color="darkslategrey",
                ),
            ),
            name=display,
            legendgroup=display,
            showlegend=True,
            hoverinfo="skip",
            legend="legend",
        )
        if n_components == 2:
            fig.add_trace(go.Scatter(x=[None], y=[None], **stub_kwargs))
        else:
            fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], **stub_kwargs))

    cursor = 0
    for g in group_names:
        n_g = group_sizes[g]
        if n_g == 0:
            continue
        spec = style.get(g, {})
        base_color = spec.get("color", "gray")
        # Prefer a named Plotly colorscale (e.g. "Blues") for layer shading —
        # those are perceptually uniform and stay distinguishable through the
        # whole light → dark range. Fall back to a plain white → base_color
        # blend only when no colorscale is configured for this group.
        layer_colorscale = (
            spec.get("colorscale")
            or [[0.0, "white"], [1.0, base_color]]
        )
        symbol = _resolve_manifold_symbol(spec, n_components)
        display = (legend_label_map or {}).get(g, g)
        for L in range(n_layers):
            frac = _layer_color_fraction(L, n_layers, light_min)
            slc = proj[cursor : cursor + n_g, L, :]
            if bg_grey_scale:
                marker_kwargs = dict(
                    color="lightgray",
                    symbol=symbol,
                    size=marker_size,
                    opacity=marker_alpha,
                    line=dict(width=marker_border_width, color="darkslategrey"),
                    showscale=False,
                )
            else:
                marker_kwargs = dict(
                    color=[frac] * slc.shape[0],
                    colorscale=layer_colorscale,
                    cmin=0.0,
                    cmax=1.0,
                    symbol=symbol,
                    size=marker_size,
                    opacity=marker_alpha,
                    line=dict(width=marker_border_width, color="darkslategrey"),
                    showscale=False,
                )
            hover = f"{display} | L{layer_lo + L}"
            # Legend handled by the per-group stub above; per-(group, layer)
            # traces ride the same legendgroup so toggling stays whole-group.
            # ``legend="legend"`` keeps these in the *first* legend row
            # (group markers); target / line traces below use ``legend2``.
            if n_components == 2:
                fig.add_trace(
                    go.Scatter(
                        x=slc[:, 0],
                        y=slc[:, 1] if slc.shape[1] >= 2 else np.zeros(slc.shape[0]),
                        mode="markers",
                        marker=marker_kwargs,
                        name=display,
                        legendgroup=display,
                        showlegend=False,
                        hovertemplate=hover + "<extra></extra>",
                        legend="legend",
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter3d(
                        x=slc[:, 0],
                        y=slc[:, 1],
                        z=slc[:, 2],
                        mode="markers",
                        marker=marker_kwargs,
                        name=display,
                        legendgroup=display,
                        showlegend=False,
                        hovertemplate=hover + "<extra></extra>",
                        legend="legend",
                    )
                )
        cursor += n_g

    # Per-condition colorbars: render up to three layer-shade colorbars on the
    # right side of the figure (control / experiment / experiment-fail) so the
    # reader can read off the layer index from each subset's colorscale. Only
    # families with at least one non-empty group contribute a bar. Suppressed
    # entirely when ``bg_grey_scale`` is on (no per-group layer gradient to
    # decode).
    colorbar_specs: List[tuple] = []
    colorbar_iter: Tuple[Tuple[str, str], ...] = () if bg_grey_scale else (
        ("control",         "control"),
        ("experiment",      "experiment"),
        ("experiment_fail", "experiment-fail"),
    )
    for cond_key, label in colorbar_iter:
        if cond_key == "experiment_fail":
            candidates = ["experiment_GT_0_fail", "experiment_GT_1_fail"]
        else:
            candidates = [f"{cond_key}_GT_0", f"{cond_key}_GT_1"]
        rep = next((c for c in candidates if c in groups_with_data), None)
        if rep is None:
            continue
        spec = style.get(rep, {})
        scale = spec.get("colorscale")
        if scale is None:
            scale = [[0.0, "white"], [1.0, spec.get("color", "gray")]]
        colorbar_specs.append((label, scale))

    n_bars = len(colorbar_specs)
    bar_x_start = 1.04
    bar_step = 0.13
    layer_mid_abs = (layer_lo + layer_hi) // 2
    tickvals = [light_min, (light_min + 1.0) / 2.0, 1.0]
    ticktext = [f"L{layer_lo}", f"L{layer_mid_abs}", f"L{layer_hi}"]
    for i, (label, scale) in enumerate(colorbar_specs):
        bar_x = bar_x_start + i * bar_step
        cb_kwargs: Dict[str, Any] = dict(
            mode="markers",
            marker=dict(
                color=[light_min, 1.0],
                colorscale=scale,
                cmin=light_min,
                cmax=1.0,
                size=0.001,
                showscale=True,
                colorbar=dict(
                    title=dict(text=f"layer<br>({label})", side="top", font=dict(size=10)),
                    x=bar_x,
                    y=0.5,
                    len=0.7,
                    thickness=12,
                    tickmode="array",
                    tickvals=tickvals,
                    ticktext=ticktext,
                    tickfont=dict(size=9),
                ),
            ),
            showlegend=False,
            hoverinfo="skip",
        )
        if n_components == 2:
            fig.add_trace(go.Scatter(x=[None, None], y=[None, None], **cb_kwargs))
        else:
            fig.add_trace(
                go.Scatter3d(x=[None, None], y=[None, None], z=[None, None], **cb_kwargs)
            )

    # Per-target traversal lines. Each ``(group_key, sample_idx)`` pair in
    # ``targets`` produces a polyline through ``proj[row, :, :]`` in the
    # group's base color; dash style encodes GT via ``gt_style``. Unknown
    # groups / out-of-range indices are reported once and skipped — typically
    # the ``_fail`` pickle is missing or that GT class had no failures.
    for target_group, sample_idx in targets:
        if target_group not in group_names:
            print(
                f"  manifold line skipped: group {target_group!r} not in "
                f"activations (have: {sorted(group_names)})"
            )
            continue
        local_n = group_sizes[target_group]
        if not 0 <= int(sample_idx) < local_n:
            print(
                f"  manifold line skipped: sample_idx={sample_idx} out of "
                f"range for group {target_group!r} (size={local_n})"
            )
            continue
        offset = 0
        for g in group_names:
            if g == target_group:
                break
            offset += group_sizes[g]
        row = offset + int(sample_idx)
        path = proj[row]  # (n_layers, n_components)
        spec = style.get(target_group, {})
        line_color = spec.get("color", "black")
        group_symbol = _resolve_manifold_symbol(spec, n_components)
        dash, _marker_sym = _gt_style_target_aesthetics(
            target_group, gt_style, group_symbol, n_components
        )
        display = (legend_label_map or {}).get(target_group, target_group)
        line_kwargs = dict(
            color=line_color, width=_MANIFOLD_TARGET_LINE_WIDTH, dash=dash
        )
        line_name = f"line: {display}[{sample_idx}] ({dash})"
        if n_components == 2:
            fig.add_trace(
                go.Scatter(
                    x=path[:, 0],
                    y=path[:, 1],
                    mode="lines",
                    line=line_kwargs,
                    name=line_name,
                    legendgroup=line_name,
                    showlegend=True,
                    legend="legend2",
                )
            )
        else:
            fig.add_trace(
                go.Scatter3d(
                    x=path[:, 0],
                    y=path[:, 1],
                    z=path[:, 2],
                    mode="lines",
                    line=line_kwargs,
                    name=line_name,
                    legendgroup=line_name,
                    showlegend=True,
                    legend="legend2",
                )
            )

    figure_title = (
        title if title is not None
        else f"{method} {n_components}D manifold @ {position_label}"
    )
    # Title at top, two legend rows below the plot (groups on row 1, target
    # lines on row 2 — set via per-trace ``legend="legend"`` /
    # ``legend="legend2"``); shade-gradient annotation sits below both.
    # Colorbars (when present) sit on the right margin — widen the figure
    # and push the right margin to make room.
    right_margin = 60 + 60 * n_bars if n_bars else 40
    layout_kwargs: Dict[str, Any] = dict(
        title=dict(
            text=figure_title,
            x=0.5,
            xanchor="center",
            y=0.97,
            yanchor="top",
        ),
        font=dict(size=11),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.10,
            xanchor="center",
            x=0.5,
            title=dict(text="groups:", font=dict(size=10)),
        ),
        legend2=dict(
            orientation="h",
            yanchor="top",
            y=-0.20,
            xanchor="center",
            x=0.5,
            title=dict(text="targets:", font=dict(size=10)),
        ),
        margin=dict(t=70, b=150, l=60, r=right_margin),
        annotations=[
            dict(
                text="Lighter shade = earlier layer · darker shade = later layer",
                xref="paper", yref="paper",
                x=0.5, xanchor="center",
                y=-0.30, yanchor="top",
                showarrow=False,
                font=dict(size=10, color="dimgray"),
            )
        ],
    )
    # PC axis labels with variance for PCA; method-specific labels otherwise.
    if n_components == 2:
        layout_kwargs.update(
            xaxis_title=axis_titles[0],
            yaxis_title=axis_titles[1],
        )
    else:
        layout_kwargs["scene"] = dict(
            xaxis=dict(title=axis_titles[0]),
            yaxis=dict(title=axis_titles[1]),
            zaxis=dict(title=axis_titles[2]),
        )
    base_w_2d, base_w_3d = 760, 860
    extra_w = 90 * n_bars if n_bars else 0
    if n_components == 2:
        layout_kwargs.update(
            width=base_w_2d + extra_w, height=760, plot_bgcolor="white"
        )
    else:
        layout_kwargs.update(width=base_w_3d + extra_w, height=840)
    fig.update_layout(**layout_kwargs)

    _write_figure(fig, basepath, fmts)


# --------------------------------------------------------------------------- #
# Set-layer PCA subspace                                                      #
# --------------------------------------------------------------------------- #


def plot_manifold_set_layer_for_position(
    activations_by_group: Dict[str, np.ndarray],
    position_index: int,
    position_label: str,
    set_layer_idx: int,
    target_layer_lo: int,
    target_layer_hi: int,
    n_components: int,
    basepath: Path,
    style: Dict[str, Any],
    title: str | None = None,
    legend_label_map: Dict[str, str] | None = None,
    background_groups: Optional[Sequence[str]] = None,
    formats: Optional[Sequence[str]] = None,
    targets: Sequence[Tuple[str, int]] = (),
    gt_style: str = "line_style",
    bg_grey_scale: bool = False,
) -> None:
    """
    Set-layer PCA manifold plot for one token position.

    Fits one PCA on the activations of **all** groups at the fixed
    ``set_layer_idx`` (one row per sample, no layer concatenation), then:

    1. Draws every group's set-layer activations as solid markers in the
       PC plane (one trace per group, ``marker_alpha`` / ``marker_size`` from
       :data:`_DEFAULT_MANIFOLD_STYLE`). No layer gradient is needed because
       the scatter is sliced at exactly one layer.
    2. For each ``(group_key, sample_idx)`` in ``targets`` whose group is
       present, projects that sample's activations across
       ``[target_layer_lo, target_layer_hi]`` (inclusive, absolute indices)
       through the **same** PCA — so the reader can see where each target's
       layer trajectory lands in the set-layer subspace. The target traces
       draw layer-shaded markers (the group's colorscale, light =
       ``target_layer_lo``, dark = ``target_layer_hi``) plus a connecting
       polyline whose dash style encodes GT via ``gt_style``.

    All layer references in titles, hover labels, and the right-margin
    colorbars are **absolute** layer indices, so set-layer plots never
    confuse the layer numbering relative to the cached pickles.

    Args:
        activations_by_group: Same shape contract as
            :func:`plot_manifold_for_position`. Always supply the **full**
            dict (all available groups including ``experiment_GT_*_fail``)
            so target traversals can resolve any group; use
            ``background_groups`` to filter what is rendered as the base
            scatter (the PCA fit and the layer-shaded scatter follow
            ``background_groups``, but targets resolve against the full dict).
        position_index: Token-position column to slice.
        position_label: Human-readable position name.
        set_layer_idx: Absolute layer index used to fit the PCA + draw the
            base scatter.
        target_layer_lo, target_layer_hi: Inclusive absolute layer range
            projected through the set-layer PCA for each target traversal.
        n_components: ``2`` or ``3``.
        basepath: Output path **without extension**; ``.pdf`` + ``.svg`` are
            always written, ``.html`` is additionally written for 3D.
        style: Output of :func:`resolve_manifold_style`.
        title: Optional explicit figure title.
        legend_label_map: Optional ``{group: display}`` mapping.
        background_groups: Optional subset of ``activations_by_group`` keys
            controlling which groups appear as the fixed-layer scatter (and
            which the PCA is fit on). ``None`` (default) = use every
            non-empty group. ``targets`` are **always** drawn when their
            group exists in ``activations_by_group``, regardless of this
            filter — so callers can pass the four-group "without_fail"
            background while still rendering experiment-fail target
            traversals projected through the four-group PCA.
        targets: ``(group_key, sample_idx)`` pairs to draw as highlighted
            set-layer traversals. Same convention as
            :func:`plot_manifold_for_position`.
        gt_style: ``"line_style"`` (default) encodes GT via line dash;
            ``"marker_shape"`` keeps the line solid and overrides the
            per-target marker symbol with circle (GT_1) / x (GT_0).
    """
    import plotly.graph_objects as go  # lazy
    from sklearn.decomposition import PCA  # lazy

    if n_components not in (2, 3):
        raise ValueError(f"n_components must be 2 or 3; got {n_components}")

    fmts = tuple(formats) if formats else ("pdf", "svg")

    group_names: List[str] = list(activations_by_group.keys())
    group_sizes = {g: activations_by_group[g].shape[0] for g in group_names}
    if sum(group_sizes.values()) == 0:
        raise ValueError("activations_by_group is empty across all groups.")

    # Resolve which groups are drawn as the background scatter (and used to
    # fit the PCA). Default = every non-empty group present. ``manifold_group``
    # in the caller uses this to drop the ``experiment_GT_*_fail`` groups
    # from the scatter while still keeping them resolvable for ``targets``
    # traversals (passed through against the full activations dict).
    if background_groups is None:
        bg_groups: List[str] = [
            g for g in group_names if group_sizes.get(g, 0) > 0
        ]
    else:
        bg_groups = [
            g for g in background_groups
            if g in activations_by_group and group_sizes.get(g, 0) > 0
        ]
    if not bg_groups:
        raise ValueError(
            "No background groups available for the set-layer scatter."
        )

    bg_stacked = np.concatenate(
        [activations_by_group[g] for g in bg_groups], axis=0
    )  # (sum_bg, n_pos, n_layers_full, d_model)
    n_layers_full = bg_stacked.shape[2]

    if not (0 <= int(set_layer_idx) < n_layers_full):
        raise ValueError(
            f"set_layer_idx={set_layer_idx} out of range [0, {n_layers_full - 1}]"
        )
    tlo = 0 if int(target_layer_lo) == -1 else int(target_layer_lo)
    thi = n_layers_full - 1 if int(target_layer_hi) == -1 else int(target_layer_hi)
    if not (0 <= tlo <= thi <= n_layers_full - 1):
        raise ValueError(
            f"Invalid target layer range [{tlo}, {thi}] for n_layers={n_layers_full}"
        )

    # Fit PCA on the **background** groups only (one row per sample).
    # ``manifold_group`` thus controls the geometric subspace the targets
    # are projected into, while leaving target lookups fully decoupled
    # from the filter.
    slab_set = bg_stacked[:, position_index, int(set_layer_idx), :].astype(np.float32)
    variance_ratios: Optional[np.ndarray] = None
    if slab_set.shape[0] < n_components + 1:
        proj_set = np.zeros((slab_set.shape[0], n_components), dtype=np.float32)
        pca: Optional[PCA] = None
    else:
        pca = PCA(n_components=n_components)
        proj_set = pca.fit_transform(slab_set).astype(np.float32)
        variance_ratios = np.asarray(
            pca.explained_variance_ratio_, dtype=np.float64
        )
    axis_titles = _axis_titles("PCA", n_components, variance_ratios)

    fig = go.Figure()
    marker_size = style.get("marker_size", 4)
    marker_alpha = style.get("marker_alpha", 0.7)
    light_min = float(style.get("layer_shade_min", 0.25))
    marker_border_width = float(style.get("marker_border_width", 0.3))
    target_border_default = float(
        style.get("target_marker_border_width", marker_border_width + 0.3)
    )

    # Per-group set-layer scatter: one trace per **background** group, solid
    # base color. ``legend="legend"`` keeps these in the *first* legend row
    # (groups); target traces below ride on ``legend2``.
    cursor = 0
    for g in bg_groups:
        n_g = group_sizes[g]
        if n_g == 0:
            continue
        spec = style.get(g, {})
        base_color = spec.get("color", "gray")
        symbol = _resolve_manifold_symbol(spec, n_components)
        display = (legend_label_map or {}).get(g, g)
        slc = proj_set[cursor : cursor + n_g]
        cursor += n_g
        bg_color = "lightgray" if bg_grey_scale else base_color
        marker_kwargs = dict(
            color=bg_color,
            symbol=symbol,
            size=marker_size,
            opacity=marker_alpha,
            line=dict(width=marker_border_width, color="darkslategrey"),
        )
        hover = f"{display} | L{int(set_layer_idx)}"
        if n_components == 2:
            fig.add_trace(
                go.Scatter(
                    x=slc[:, 0],
                    y=slc[:, 1] if slc.shape[1] >= 2 else np.zeros(n_g),
                    mode="markers",
                    marker=marker_kwargs,
                    name=display,
                    legendgroup=display,
                    showlegend=True,
                    hovertemplate=hover + "<extra></extra>",
                    legend="legend",
                )
            )
        else:
            fig.add_trace(
                go.Scatter3d(
                    x=slc[:, 0], y=slc[:, 1], z=slc[:, 2],
                    mode="markers",
                    marker=marker_kwargs,
                    name=display,
                    legendgroup=display,
                    showlegend=True,
                    hovertemplate=hover + "<extra></extra>",
                    legend="legend",
                )
            )

    # Per-target traversal: project the chosen sample's activations across
    # [tlo, thi] through the set-layer PCA and draw layer-shaded markers
    # plus a connecting line. Marker aesthetics: the group's existing
    # colorscale, an emphasized size / alpha, and (when ``gt_style`` is
    # ``marker_shape``) a GT-driven symbol override. The line dash encodes
    # GT via ``gt_style``. Targets resolve against the **full** activations
    # dict, not the ``bg_groups`` filter — so ``without_experiment_fail``
    # runs still render an experiment-fail target as long as its pickle
    # exists. Track unique colorscales so we can render one colorbar per
    # scale on the right margin (de-duplicated when targets reuse the same
    # colorscale).
    target_colorbar_specs: List[tuple] = []  # [(label, scale)]
    seen_scale_keys: set = set()

    def _scale_key(scale: Any) -> str:
        return scale if isinstance(scale, str) else repr(scale)

    if pca is not None:
        n_target_layers = thi - tlo + 1
        target_customdata = list(range(tlo, thi + 1))
        for target_group, sample_idx in targets:
            if target_group not in activations_by_group:
                print(
                    f"  set-layer manifold target skipped: group "
                    f"{target_group!r} not in activations "
                    f"(have: {sorted(activations_by_group)})"
                )
                continue
            local_n = group_sizes[target_group]
            sample_idx = int(sample_idx)
            if not 0 <= sample_idx < local_n:
                print(
                    f"  set-layer manifold target skipped: sample_idx="
                    f"{sample_idx} out of range for {target_group!r} (size={local_n})"
                )
                continue
            target_acts = activations_by_group[target_group][
                sample_idx, position_index, tlo : thi + 1, :
            ].astype(np.float32)
            target_proj = pca.transform(target_acts).astype(np.float32)

            spec = style.get(target_group, {})
            target_base = spec.get("color", "black")
            target_scale = (
                spec.get("colorscale")
                or [[0.0, "white"], [1.0, target_base]]
            )
            group_symbol = _resolve_manifold_symbol(spec, n_components)
            dash, target_symbol = _gt_style_target_aesthetics(
                target_group, gt_style, group_symbol, n_components
            )
            t_marker_size = float(marker_size) + 2.0
            t_marker_alpha = 1.0
            t_border_width = target_border_default
            layer_fracs = [
                _layer_color_fraction(L, n_target_layers, light_min)
                for L in range(n_target_layers)
            ]
            display = (legend_label_map or {}).get(target_group, target_group)
            target_name = f"{display} [sample {sample_idx}]"

            # Record this target's colorscale for the right-margin colorbar
            # (deduplicated — reused scales render one bar only).
            sk = _scale_key(target_scale)
            if sk not in seen_scale_keys:
                seen_scale_keys.add(sk)
                cb_label = (
                    str(target_scale)
                    if isinstance(target_scale, str)
                    else "custom"
                )
                target_colorbar_specs.append((cb_label, target_scale))

            marker_kwargs = dict(
                color=layer_fracs,
                colorscale=target_scale,
                cmin=0.0, cmax=1.0,
                symbol=target_symbol,
                size=t_marker_size,
                opacity=t_marker_alpha,
                line=dict(width=t_border_width, color="darkslategrey"),
                showscale=False,
            )
            trace_kwargs: Dict[str, Any] = dict(
                mode="lines+markers",
                marker=marker_kwargs,
                line=dict(
                    color=target_base,
                    width=_MANIFOLD_TARGET_LINE_WIDTH,
                    dash=dash,
                ),
                name=target_name,
                legendgroup=target_name,
                showlegend=True,
                customdata=target_customdata,
                hovertemplate=(
                    f"{display}[{sample_idx}] | L%{{customdata}}<extra></extra>"
                ),
                legend="legend2",
            )

            if n_components == 2:
                fig.add_trace(
                    go.Scatter(
                        x=target_proj[:, 0],
                        y=(
                            target_proj[:, 1]
                            if target_proj.shape[1] >= 2
                            else np.zeros(n_target_layers)
                        ),
                        **trace_kwargs,
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter3d(
                        x=target_proj[:, 0],
                        y=target_proj[:, 1],
                        z=target_proj[:, 2],
                        **trace_kwargs,
                    )
                )

    # Right-margin colorbars — one per unique target colorscale, ticked at
    # the target-traversal layer indices ([tlo, mid, thi]).
    n_target_layers_for_ticks = thi - tlo + 1
    target_layer_mid = (tlo + thi) // 2 if n_target_layers_for_ticks >= 2 else tlo
    bar_tickvals = [light_min, (light_min + 1.0) / 2.0, 1.0]
    bar_ticktext = [f"L{tlo}", f"L{target_layer_mid}", f"L{thi}"]
    n_bars = len(target_colorbar_specs)
    bar_x_start = 1.04
    bar_step = 0.13
    for i, (label, scale) in enumerate(target_colorbar_specs):
        bar_x = bar_x_start + i * bar_step
        cb_kwargs: Dict[str, Any] = dict(
            mode="markers",
            marker=dict(
                color=[light_min, 1.0],
                colorscale=scale,
                cmin=light_min,
                cmax=1.0,
                size=0.001,
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text=f"target<br>({label})",
                        side="top",
                        font=dict(size=10),
                    ),
                    x=bar_x,
                    y=0.5,
                    len=0.7,
                    thickness=12,
                    tickmode="array",
                    tickvals=bar_tickvals,
                    ticktext=bar_ticktext,
                    tickfont=dict(size=9),
                ),
            ),
            showlegend=False,
            hoverinfo="skip",
        )
        if n_components == 2:
            fig.add_trace(go.Scatter(x=[None, None], y=[None, None], **cb_kwargs))
        else:
            fig.add_trace(
                go.Scatter3d(x=[None, None], y=[None, None], z=[None, None], **cb_kwargs)
            )

    figure_title = (
        title if title is not None
        else f"PCA {n_components}D fit @ L{int(set_layer_idx)} | {position_label}"
    )
    # Subtitle sits **just under the title** (between title and plot area)
    # so it never overlaps the legend / plot. Top margin grows to give the
    # subtitle dedicated whitespace; bottom margin grows to fit two legend
    # rows (groups + targets).
    right_margin = 60 + 60 * n_bars if n_bars else 40
    subtitle_text = (
        f"PCA fitted on layer L{int(set_layer_idx)}; "
        f"target traversals projected from L{tlo} → L{thi}."
    )
    layout_kwargs: Dict[str, Any] = dict(
        title=dict(
            text=figure_title,
            x=0.5, xanchor="center",
            y=0.97, yanchor="top",
        ),
        font=dict(size=11),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.10,
            xanchor="center",
            x=0.5,
            title=dict(text="groups:", font=dict(size=10)),
        ),
        legend2=dict(
            orientation="h",
            yanchor="top",
            y=-0.20,
            xanchor="center",
            x=0.5,
            title=dict(text="targets:", font=dict(size=10)),
        ),
        margin=dict(t=110, b=150, l=60, r=right_margin),
        annotations=[
            dict(
                text=subtitle_text,
                xref="paper", yref="paper",
                x=0.5, xanchor="center",
                y=1.06, yanchor="bottom",
                showarrow=False,
                font=dict(size=10, color="dimgray"),
            ),
        ],
    )
    # PC axis labels with variance.
    if n_components == 2:
        layout_kwargs.update(
            xaxis_title=axis_titles[0],
            yaxis_title=axis_titles[1],
        )
    else:
        layout_kwargs["scene"] = dict(
            xaxis=dict(title=axis_titles[0]),
            yaxis=dict(title=axis_titles[1]),
            zaxis=dict(title=axis_titles[2]),
        )
    base_w_2d, base_w_3d = 760, 860
    extra_w = 90 * n_bars if n_bars else 0
    if n_components == 2:
        layout_kwargs.update(
            width=base_w_2d + extra_w, height=780, plot_bgcolor="white"
        )
    else:
        layout_kwargs.update(width=base_w_3d + extra_w, height=860)
    fig.update_layout(**layout_kwargs)

    _write_figure(fig, basepath, fmts)


# --------------------------------------------------------------------------- #
# Per-layer classifier-accuracy curve (single SysPrompt, single seed)         #
# --------------------------------------------------------------------------- #


def plot_class_separation_curve(
    accuracies: Sequence[float],
    basepath: Path,
    title: str,
    formats: Sequence[str] = ("png", "svg"),
    chance_line: Optional[float] = 0.25,
    line_color: str = "firebrick",
    line_width: float = 2.0,
    y_range: Sequence[float] = (0.0, 1.05),
) -> None:
    """
    Single-curve line plot of per-layer 4-class CV accuracy.

    Args:
        accuracies: Length-``n_layers`` sequence in ``[0, 1]``.
        basepath: Output path without extension.
        title: Figure title.
        formats: Output file formats (any combination supported by
            :func:`_write_figure`). Defaults to ``("png", "svg")``.
        chance_line: Optional reference horizontal line value (e.g. ``0.25``
            for four balanced classes). ``None`` suppresses the line.
        line_color: Plotly-acceptable color for the accuracy curve.
        line_width: Pixel width of the accuracy curve.
    """
    import plotly.graph_objects as go  # lazy

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(accuracies))),
            y=[float(a) for a in accuracies],
            mode="lines+markers",
            line=dict(color=line_color, width=float(line_width)),
            marker=dict(size=4),
            name="CV accuracy",
        )
    )
    if chance_line is not None:
        fig.add_hline(
            y=float(chance_line),
            line_dash="dash",
            line_color="gray",
        )
    fig.update_layout(
        title=title,
        xaxis_title="Layer",
        yaxis_title="Cross-validation accuracy",
        yaxis=dict(range=[float(y_range[0]), float(y_range[1])]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=420,
        width=720,
        font=dict(size=12),
        legend=dict(itemwidth=40),
    )
    _write_figure(fig, basepath, formats)


# Per-class color + dash convention. Matches the PCA scatter convention in
# ``cfg.pca_plot_style``: color encodes condition (control=goldenrod,
# experiment=royalblue) and dash encodes GT (dash=0, solid=1).
_PER_CLASS_STYLE: Dict[str, Dict[str, str]] = {
    "control_GT_0":    {"color": "goldenrod", "dash": "dash"},
    "control_GT_1":    {"color": "goldenrod", "dash": "solid"},
    "experiment_GT_0": {"color": "royalblue", "dash": "dash"},
    "experiment_GT_1": {"color": "royalblue", "dash": "solid"},
}


def plot_class_separation_per_class_curve(
    per_class_accuracies: Dict[str, Sequence[float]],
    basepath: Path,
    title: str,
    formats: Sequence[str] = ("png", "svg"),
    chance_line: Optional[float] = 0.25,
    line_width: float = 2.0,
    y_range: Sequence[float] = (0.0, 1.05),
) -> None:
    """
    Four-line per-layer CV recall plot, one line per ``(condition, GT)`` class.

    Args:
        per_class_accuracies: Mapping ``class_name -> length-n_layers list``
            as produced by
            :func:`src.analysis.class_separation.compute_per_class_cv_accuracy_per_layer`.
            Missing classes are simply skipped.
        basepath: Output path **without extension**.
        title: Figure title.
        formats: Output file formats (any combination supported by
            :func:`_write_figure`). Defaults to ``("png", "svg")``.
        chance_line: Optional dashed horizontal reference line (default
            ``0.25`` for four balanced classes). ``None`` suppresses it.
        line_width: Pixel width of each curve.
        y_range: Y-axis range.
    """
    import plotly.graph_objects as go  # lazy

    fig = go.Figure()
    for cls_name, style in _PER_CLASS_STYLE.items():
        accs = per_class_accuracies.get(cls_name)
        if accs is None:
            continue
        fig.add_trace(
            go.Scatter(
                x=list(range(len(accs))),
                y=[float(a) for a in accs],
                mode="lines+markers",
                line=dict(
                    color=style["color"],
                    width=float(line_width),
                    dash=style["dash"],
                ),
                marker=dict(size=4),
                name=cls_name,
            )
        )
    if chance_line is not None:
        fig.add_hline(
            y=float(chance_line),
            line_dash="dash",
            line_color="gray",
        )
    fig.update_layout(
        title=title,
        xaxis_title="Layer",
        yaxis_title="Per-class CV recall",
        yaxis=dict(range=[float(y_range[0]), float(y_range[1])]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=440,
        width=760,
        font=dict(size=12),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, x=0.0, itemwidth=40
        ),
    )
    _write_figure(fig, basepath, formats)


# Per-condition binary classifier style: one classifier per condition
# (control / experiment). Solid lines, same color palette as the per-class
# plot above (goldenrod=control, royalblue=experiment).
_BINARY_CONDITION_STYLE: Dict[str, Dict[str, str]] = {
    "control":    {"color": "goldenrod"},
    "experiment": {"color": "royalblue"},
}


def plot_class_separation_binary_curve(
    per_condition_accuracies: Dict[str, Sequence[float]],
    basepath: Path,
    title: str,
    formats: Sequence[str] = ("png", "svg"),
    chance_line: Optional[float] = 0.5,
    line_width: float = 2.0,
    y_range: Sequence[float] = (0.0, 1.05),
) -> None:
    """
    Two-line per-layer CV accuracy plot, one line per condition (control,
    experiment). Each line is a separately-trained binary classifier
    distinguishing GT=0 from GT=1 *within* that condition.

    Args:
        per_condition_accuracies: Mapping ``"control" / "experiment" ->
            length-n_layers list`` as produced by
            :func:`src.analysis.class_separation.compute_binary_per_condition_cv_accuracy_per_layer`.
            A condition absent from the dict is silently skipped.
        basepath: Output path **without extension**.
        title: Figure title.
        formats: Output file formats (any combination supported by
            :func:`_write_figure`). Defaults to ``("png", "svg")``.
        chance_line: Optional dashed horizontal reference (default ``0.5``
            for a balanced binary task).
        line_width: Pixel width of each curve.
        y_range: Y-axis range.
    """
    import plotly.graph_objects as go  # lazy

    fig = go.Figure()
    for cond, style in _BINARY_CONDITION_STYLE.items():
        accs = per_condition_accuracies.get(cond)
        if accs is None:
            continue
        fig.add_trace(
            go.Scatter(
                x=list(range(len(accs))),
                y=[float(a) for a in accs],
                mode="lines+markers",
                line=dict(color=style["color"], width=float(line_width)),
                marker=dict(size=4),
                name=cond,
            )
        )
    if chance_line is not None:
        fig.add_hline(
            y=float(chance_line),
            line_dash="dash",
            line_color="gray",
        )
    fig.update_layout(
        title=title,
        xaxis_title="Layer",
        yaxis_title="Binary CV accuracy",
        yaxis=dict(range=[float(y_range[0]), float(y_range[1])]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=440,
        width=760,
        font=dict(size=12),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, x=0.0, itemwidth=40
        ),
    )
    _write_figure(fig, basepath, formats)
