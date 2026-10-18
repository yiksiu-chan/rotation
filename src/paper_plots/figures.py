"""
Plotly figure builders for the paper-plot stage.

Each plot accepts a ``formats`` list (``["pdf", "svg", "png", "html", ...]``)
that the orchestrator pulls from ``cfg.paper_plot_formats``. Defaults match
the project's prior convention — the cosine heatmap defaults to
``["svg", "png"]`` (high-DPI raster + vector) and every other figure defaults
to ``["pdf", "svg"]``. Inputs are the small dataclasses defined in
:mod:`src.paper_plots.aggregate`; nothing here touches disk except
:func:`_write_figure`.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from src.analysis.rotation_plugins import (
    metric_short_label,
    resolve_displayed_metrics,
)
from src.paper_plots.aggregate import (
    BehaviorAggregate,
    ClassSeparationAggregate,
    ClassSeparationBinaryAggregate,
    ClassSeparationPerClassAggregate,
    SeedAggregate,
)


# --------------------------------------------------------------------------- #
# Shared helpers                                                              #
# --------------------------------------------------------------------------- #


_RASTER_FORMATS = frozenset({"png", "jpg", "jpeg", "webp"})
_VECTOR_FORMATS = frozenset({"pdf", "svg", "eps"})
_SUPPORTED_FORMATS = _RASTER_FORMATS | _VECTOR_FORMATS | {"html"}


# Common CSS named colors used by the existing palette + a small selection of
# neutral / accent colors users are likely to type. Anything outside this set
# should be passed in as hex (``#rrggbb``), ``rgb(r,g,b)``, or ``rgba(...)``.
_CSS_NAMED_COLOR_RGB: Dict[str, tuple] = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
    "lightgray": (211, 211, 211),
    "lightgrey": (211, 211, 211),
    "darkgray": (169, 169, 169),
    "darkgrey": (169, 169, 169),
    "darkslategrey": (47, 79, 79),
    "darkslategray": (47, 79, 79),
    "red": (255, 0, 0),
    "green": (0, 128, 0),
    "blue": (0, 0, 255),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "firebrick": (178, 34, 34),
    "steelblue": (70, 130, 180),
    "royalblue": (65, 105, 225),
    "goldenrod": (218, 165, 32),
    "teal": (0, 128, 128),
    "navy": (0, 0, 128),
    "olive": (128, 128, 0),
    "maroon": (128, 0, 0),
    "indianred": (205, 92, 92),
    "salmon": (250, 128, 114),
    "tomato": (255, 99, 71),
    "seagreen": (46, 139, 87),
    "forestgreen": (34, 139, 34),
    "mediumseagreen": (60, 179, 113),
    "skyblue": (135, 206, 235),
    "dodgerblue": (30, 144, 255),
    "cornflowerblue": (100, 149, 237),
    "slateblue": (106, 90, 205),
    "mediumpurple": (147, 112, 219),
    "violet": (238, 130, 238),
    "magenta": (255, 0, 255),
    "pink": (255, 192, 203),
    "gold": (255, 215, 0),
    "khaki": (240, 230, 140),
    "tan": (210, 180, 140),
    "chocolate": (210, 105, 30),
    "sienna": (160, 82, 45),
    "brown": (165, 42, 42),
    "cyan": (0, 255, 255),
    "yellow": (255, 255, 0),
}


def _to_rgba(color: str, alpha: float) -> str:
    """
    Convert a Plotly-acceptable color string to ``rgba(r,g,b,alpha)``.

    Accepted forms: hex (``#rgb`` / ``#rrggbb``), ``rgb(r,g,b)``,
    ``rgba(r,g,b,a)`` (existing alpha is replaced), or a common CSS named
    color in :data:`_CSS_NAMED_COLOR_RGB`. Raises :class:`ValueError` on
    unrecognized input so style typos surface loudly rather than silently
    drawing the wrong shade.
    """
    if not isinstance(color, str):
        raise ValueError(f"_to_rgba: color must be a string; got {type(color).__name__}.")
    c = color.strip()
    a = float(alpha)
    if c.startswith("rgba(") and c.endswith(")"):
        parts = [p.strip() for p in c[5:-1].split(",")]
        if len(parts) == 4:
            r, g, b = (int(float(p)) for p in parts[:3])
            return f"rgba({r},{g},{b},{a})"
    if c.startswith("rgb(") and c.endswith(")"):
        parts = [p.strip() for p in c[4:-1].split(",")]
        if len(parts) == 3:
            r, g, b = (int(float(p)) for p in parts)
            return f"rgba({r},{g},{b},{a})"
    if c.startswith("#"):
        h = c.lstrip("#")
        if len(h) == 3:
            h = "".join(ch * 2 for ch in h)
        if len(h) == 6:
            r = int(h[0:2], 16)
            g = int(h[2:4], 16)
            b = int(h[4:6], 16)
            return f"rgba({r},{g},{b},{a})"
    rgb = _CSS_NAMED_COLOR_RGB.get(c.lower())
    if rgb is not None:
        r, g, b = rgb
        return f"rgba({r},{g},{b},{a})"
    raise ValueError(
        f"_to_rgba: cannot interpret color {color!r}. Use hex (#rrggbb), "
        f"rgb()/rgba() string, or one of: {sorted(_CSS_NAMED_COLOR_RGB)}."
    )


def _normalize_format(fmt: str) -> str:
    f = fmt.lower().lstrip(".")
    if f == "jpeg":
        f = "jpg"
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
    higher DPI. ``html`` writes an interactive standalone file. Unknown
    formats raise :class:`ValueError`. Repeats and aliases (``"jpg"`` /
    ``"jpeg"``) are silently de-duplicated.
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


# --------------------------------------------------------------------------- #
# 1. Cross-prompt cosine heatmap (best token per variant)                     #
# --------------------------------------------------------------------------- #


def plot_cos_sim_heatmap_across_prompts(
    seed_data: SeedAggregate,
    basepath: Path,
    title: str,
    metric_display_mode: str = "all",
    label_display: bool = True,
    formats: Optional[Sequence[str]] = None,
    label_format: str = "full",
    show_metric_block: bool = True,
    show_title: bool = True,
    label_px_per_char: float = 6.0,
    label_px_pad: int = 40,
    png_scale: float = 4.0,
) -> None:
    """
    Heatmap with one row per system prompt, columns = layers, color = cosine
    similarity at that variant's **own** best token position (under
    ``cfg.rotation_metric``). Rows are ordered top→bottom from highest to
    lowest ``rotation_r`` of the chosen metric. Each row's y-label is

    .. code-block::

        <system_prompt_idx>  [<best_token_position>]  <system_prompt_text>

    The system-prompt text is rendered on a **single line** (no wrapping); the
    figure widens to fit the longest label.

    A trailing block of ``r1, r2, …`` columns shows each metric's value at
    that variant's displayed position (rounded to one decimal). Which metrics
    appear is controlled by ``metric_display_mode`` — ``"all"`` (every metric
    in :data:`ROTATION_METRIC_ORDER`) or ``"chosen_metric"`` (only the
    seed's driving metric). In ``"chosen_metric"`` mode the column is
    headed by ``r`` placed *above* the column instead of the usual ``rN``
    tick at the bottom.

    Args:
        seed_data: Aggregated rotation summaries for one seed.
        basepath: Output path **without extension**; ``.svg`` and ``.png`` are
            written under it.
        title: Figure title.
        metric_display_mode: One of ``"all"`` / ``"chosen_metric"``.
        label_display: Whether to render the per-row text label
            (``<idx> [<pos>] <prompt>``) on the left of the figure. False
            hides the labels and shrinks the left margin accordingly.

    Raises:
        ValueError: If variants disagree on layer count (shouldn't happen
            within one seed if all came from the same model).
    """
    import plotly.graph_objects as go  # lazy

    if seed_data.is_empty:
        raise ValueError("Cannot plot cos-sim heatmap: no variants with rotation_stats.json.")

    layer_counts = {len(v.cos_sim_across_layers) for v in seed_data.variants}
    if len(layer_counts) != 1:
        raise ValueError(
            f"Variants disagree on n_layers (got {layer_counts}); cannot stack into a heatmap."
        )
    n_layers = next(iter(layer_counts))

    # Sort variants by chosen-metric rotation_r descending — highest at top
    # in the rendered heatmap (we use yaxis autorange='reversed' below since
    # Plotly's default puts y[0] at the bottom).
    variants_sorted = sorted(
        seed_data.variants, key=lambda v: v.rotation_r, reverse=True
    )

    z = np.array([v.cos_sim_across_layers for v in variants_sorted], dtype=np.float64)

    # Y-label format: index, best position, raw system-prompt content. Kept
    # on a single line — the figure widens for long prompts rather than
    # wrapping. Prefixing the variant id keeps labels unique even if two
    # prompts have identical text. ``label_format="prompt_only"`` drops the
    # variant id and ``[best_position]`` prefix, leaving the raw prompt text.
    if label_format == "prompt_only":
        raw_labels = [v.system_prompt_text for v in variants_sorted]
        # Plotly Heatmap requires unique y-tick strings; append a zero-width
        # space per row so identical prompts (rare) don't collapse onto the
        # same row. The padding is invisible in the rendered label.
        y_labels = [
            lbl + ("​" * i) for i, lbl in enumerate(raw_labels)
        ]
    elif label_format == "full":
        y_labels = [
            f"{v.system_prompt_id}  [{v.best_position_label}]  {v.system_prompt_text}"
            for v in variants_sorted
        ]
    else:
        raise ValueError(
            f"label_format must be 'full' or 'prompt_only'; got {label_format!r}"
        )
    n_rows = len(variants_sorted)
    max_label_chars = max((len(lbl) for lbl in y_labels), default=1)

    metric_names = list(
        resolve_displayed_metrics(metric_display_mode, seed_data.driving_metric_name)
    ) if show_metric_block else []
    n_metrics = len(metric_names)
    single_metric = n_metrics == 1 and show_metric_block
    # In "chosen_metric" mode (single column) we drop the rN tick at the
    # bottom and float a plain "r" header above the column instead.
    metric_short_labels = (
        [] if single_metric else [metric_short_label(m) for m in metric_names]
    )
    metric_arr = np.array(
        [
            [v.rotation_r_by_metric.get(m, np.nan) for m in metric_names]
            for v in variants_sorted
        ],
        dtype=np.float64,
    )

    gap = 1 if show_metric_block and n_metrics > 0 else 0
    n_cols_total = n_layers + gap + n_metrics

    layer_tick_step = max(1, n_layers // 8)
    layer_tickvals = list(range(0, n_layers, layer_tick_step))
    layer_ticktext = [str(v) for v in layer_tickvals]
    metric_tickvals = [n_layers + gap + j for j in range(n_metrics)]
    full_tickvals = layer_tickvals + (metric_tickvals if not single_metric else [])
    full_ticktext = layer_ticktext + metric_short_labels

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
    if show_metric_block and n_metrics > 0:
        for i, lbl in enumerate(y_labels):
            for j, col_x in enumerate(metric_tickvals):
                val = metric_arr[i, j]
                text = "n/a" if np.isnan(val) else f"{val:.1f}"
                fig.add_annotation(
                    x=col_x,
                    y=lbl,
                    xref="x",
                    yref="y",
                    text=text,
                    showarrow=False,
                    font=dict(size=11, color="black"),
                )
    # In chosen-metric mode, place a single "r" header above the lone column.
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

    # Per-character reservation for the left margin's label band. Defaults
    # (6.0 + 40 pad) are deliberately generous so the original 'full' labels
    # never clip; callers rendering the minimal `prompt_only` variant pass
    # tighter values (e.g. 5.0 + 10) to remove the unused whitespace at the
    # left of the figure. Cap at 2200 px to avoid runaway widths on
    # pathologically long prompts.
    label_px = (
        min(2200, max(160, int(max_label_chars * label_px_per_char) + label_px_pad))
        if label_display
        else 0
    )
    fig.update_layout(
        title=(title if show_title else None),
        xaxis_title="Layer",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            tickmode="array",
            tickvals=full_tickvals,
            ticktext=full_ticktext,
            range=[-0.5, n_cols_total - 0.5],
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            autorange="reversed",  # highest rotation at top
            tickfont=dict(size=10),
            showticklabels=label_display,
            showgrid=False,
            zeroline=False,
        ),
        height=max(360, 60 + 22 * n_rows),
        width=label_px + max(520, 120 + 22 * n_layers) + 28 * n_metrics + 30,
        font=dict(size=12),
        # Left margin matches the projected label width so single-line prompts
        # render without truncation; Plotly's default would clip them. When
        # labels are hidden we collapse the margin to a small constant.
        margin=dict(
            l=label_px if label_display else 40,
            r=40,
            t=80 if show_title else 20,
            b=60,
        ),
    )
    _write_figure(fig, basepath, formats or ("svg", "png"), png_scale=png_scale)


# --------------------------------------------------------------------------- #
# 2. Distribution of best token positions across variants                     #
# --------------------------------------------------------------------------- #


def plot_best_token_distribution(
    seed_data: SeedAggregate,
    canonical_position_labels: Sequence[str],
    basepath: Path,
    title: str,
    formats: Optional[Sequence[str]] = None,
) -> None:
    """
    Bar chart: x = token positions in canonical order (no skipping), y = count
    of variants whose best token landed on that position (under
    ``cfg.rotation_metric``).

    Args:
        seed_data: Aggregated rotation summaries for one seed.
        canonical_position_labels: Ordered token-position labels
            (``position_labels(n_pre, n_gen)``); positions absent from the
            counts still appear with height 0.
        basepath: Output path **without extension**.
        title: Figure title.
    """
    import plotly.graph_objects as go  # lazy

    counts = Counter(v.best_position_label for v in seed_data.variants)
    y_vals = [counts.get(lbl, 0) for lbl in canonical_position_labels]

    fig = go.Figure(
        data=go.Bar(
            x=list(canonical_position_labels),
            y=y_vals,
            marker=dict(color="steelblue"),
            text=[str(c) if c > 0 else "" for c in y_vals],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Token position (best per variant)",
        yaxis_title="Variant count",
        height=420,
        width=max(520, 80 + 60 * len(canonical_position_labels)),
        font=dict(size=12),
    )
    _write_figure(fig, basepath, formats or ("pdf", "svg"))


# --------------------------------------------------------------------------- #
# 3. Distribution of rotation_r across variants                               #
# --------------------------------------------------------------------------- #


def plot_rotation_distribution(
    seed_data: SeedAggregate,
    basepath: Path,
    title: str,
    formats: Optional[Sequence[str]] = None,
) -> None:
    """
    Box-plus-points distribution of ``rotation_r`` (taken at each variant's
    best token position under ``cfg.rotation_metric``) across all variants
    for one seed.
    """
    import plotly.graph_objects as go  # lazy

    rotation_rs = [v.rotation_r for v in seed_data.variants]
    fig = go.Figure(
        data=go.Box(
            y=rotation_rs,
            name=f"rotation_r [{seed_data.driving_metric_name}]",
            boxpoints="all",
            jitter=0.4,
            pointpos=0,
            marker=dict(color="steelblue", size=5, opacity=0.7),
            line=dict(color="steelblue"),
        )
    )
    fig.update_layout(
        title=title,
        yaxis_title=f"rotation_r [{seed_data.driving_metric_name}] (best per variant)",
        xaxis=dict(showticklabels=False),
        height=480,
        width=420,
        font=dict(size=12),
    )
    _write_figure(fig, basepath, formats or ("pdf", "svg"))


# --------------------------------------------------------------------------- #
# 3a-bis. Per-seed cosine-similarity curve, mean ± band across system prompts #
# --------------------------------------------------------------------------- #


def plot_cos_sim_curves_across_prompts(
    seed_data: SeedAggregate,
    basepath: Path,
    title: str,
    band_kind: str = "std",
    formats: Optional[Sequence[str]] = None,
    line_color: str = "steelblue",
    line_width: float = 2.0,
    band_color: str = "steelblue",
    band_alpha: float = 0.2,
    y_range: Sequence[float] = (-1.05, 1.05),
) -> None:
    """
    Mean ± variance band of the rotation cosine-similarity curve across system
    prompts for one seed. Each variant contributes its
    ``cos_sim_across_layers`` taken at *its own* best position under
    ``cfg.rotation_metric``; curves are stacked and aggregated layer-wise.

    Args:
        seed_data: Per-seed aggregate from
            :func:`src.paper_plots.aggregate.load_rotation_aggregate_for_seed`.
        basepath: Output path **without extension** (use
            :func:`src.paper_plots.paths.cos_sim_curves_basepath`).
        title: Figure title.
        band_kind: Variance metric for the shaded band — ``"std"`` (1-σ),
            ``"sem"``, or ``"ci95"`` (1.96 × SEM).
        formats: Output formats; ``None`` falls back to ``("pdf", "svg")``.
        line_color: Color for the mean curve (hex / rgb()/rgba() string /
            common CSS name; see :func:`_to_rgba`).
        line_width: Pixel width of the mean curve.
        band_color: Color of the variance band; combined with ``band_alpha``
            into an ``rgba(...)`` fill string.
        band_alpha: Opacity of the variance band in ``[0, 1]``.

    Curves of unequal length (which should not happen in a normally
    configured run since ``n_layers`` is a model property) are aligned on
    the leading ``min(len(curve))`` layers before stacking.
    """
    import plotly.graph_objects as go  # lazy

    if band_kind not in ("std", "sem", "ci95"):
        raise ValueError(
            f"band_kind must be 'std' | 'sem' | 'ci95'; got {band_kind!r}"
        )

    curves = [
        np.asarray(v.cos_sim_across_layers, dtype=np.float64)
        for v in seed_data.variants
        if v.cos_sim_across_layers
    ]
    if not curves:
        print(
            f"  skip cos_sim_curves (seed={seed_data.seed}): "
            "no cos_sim_across_layers curves to plot."
        )
        return

    n_layers = min(int(c.shape[0]) for c in curves)
    stacked = np.stack([c[:n_layers] for c in curves], axis=0)
    mean_curve = stacked.mean(axis=0)
    if band_kind == "std":
        spread = stacked.std(axis=0, ddof=0)
    elif band_kind == "sem":
        spread = stacked.std(axis=0, ddof=1) / np.sqrt(max(1, stacked.shape[0]))
    else:  # ci95
        spread = (
            1.96 * stacked.std(axis=0, ddof=1) / np.sqrt(max(1, stacked.shape[0]))
        )
    upper = (mean_curve + spread).tolist()
    lower = (mean_curve - spread).tolist()
    x = list(range(n_layers))

    band_label = f"± {band_kind}" if band_kind != "ci95" else "± 95% CI"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=upper,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=lower,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor=_to_rgba(band_color, band_alpha),
            name=band_label,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=mean_curve.tolist(),
            mode="lines",
            line=dict(color=line_color, width=float(line_width)),
            name=f"mean across {len(curves)} prompts",
            hovertemplate="L%{x} | y=%{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=0.0,
        line=dict(color="lightgray", width=1, dash="dot"),
    )
    fig.update_layout(
        title=title,
        xaxis_title="Layer",
        yaxis_title=(
            f"Cosine sim to truth dir [{seed_data.driving_metric_name}, "
            "best per variant]"
        ),
        yaxis=dict(range=[float(y_range[0]), float(y_range[1])]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=440,
        width=760,
        font=dict(size=12),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, x=0.0, itemwidth=40
        ),
    )
    _write_figure(fig, basepath, formats or ("pdf", "svg"))


# --------------------------------------------------------------------------- #
# 3b. Accuracy (experiment) vs rotation_r scatter, per system prompt          #
# --------------------------------------------------------------------------- #


def plot_accuracy_vs_rotation_scatter(
    seed_data: SeedAggregate,
    accuracy_by_variant_id: Dict[int, float],
    basepath: Path,
    title: str,
    x_axis_label: str,
    formats: Optional[Sequence[str]] = None,
) -> None:
    """
    Scatter of experiment-condition instruction-following accuracy (x) against
    ``rotation_r`` (y) at each variant's best token position under the seed's
    driving metric. One marker per system prompt that has both a rotation
    summary and a behavior_results_experiment.json.
    """
    import plotly.graph_objects as go  # lazy

    xs: list[float] = []
    ys: list[float] = []
    ids: list[int] = []
    for v in seed_data.variants:
        acc = accuracy_by_variant_id.get(v.system_prompt_id)
        if acc is None:
            continue
        xs.append(float(acc))
        ys.append(float(v.rotation_r))
        ids.append(v.system_prompt_id)

    if not xs:
        raise ValueError(
            "Cannot plot accuracy-vs-rotation scatter: no variants have both "
            "rotation_stats.json and behavior_results_experiment.json."
        )

    fig = go.Figure(
        data=go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(color="steelblue", size=8, opacity=0.75),
            text=[f"SysPrompt {i}" for i in ids],
            hovertemplate="%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=x_axis_label,
        yaxis_title=f"rotation metric r [{seed_data.driving_metric_name}]",
        xaxis=dict(range=[0.0, 1.05]),
        height=480,
        width=560,
        font=dict(size=12),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    _write_figure(fig, basepath, formats or ("pdf", "svg"))


# --------------------------------------------------------------------------- #
# 3c. rotation_r scatter across two datasets, per system prompt               #
# --------------------------------------------------------------------------- #


def plot_rotation_r_cross_dataset_scatter(
    seed_data_x: SeedAggregate,
    seed_data_y: SeedAggregate,
    x_dataset_name: str,
    y_dataset_name: str,
    basepath: Path,
    title: str,
    formats: Optional[Sequence[str]] = None,
) -> None:
    """
    Scatter of ``rotation_r`` for one task across two datasets, paired by
    ``system_prompt_id``. Each axis takes the variant's own best-position
    rotation_r in that dataset (under the seed's driving rotation metric).
    """
    import plotly.graph_objects as go  # lazy

    by_id_x = {v.system_prompt_id: float(v.rotation_r) for v in seed_data_x.variants}
    by_id_y = {v.system_prompt_id: float(v.rotation_r) for v in seed_data_y.variants}
    common_ids = sorted(set(by_id_x) & set(by_id_y))
    if not common_ids:
        raise ValueError(
            "Cannot plot cross-dataset rotation_r scatter: no system prompt has "
            f"rotation_stats.json under both datasets ({x_dataset_name!r} vs "
            f"{y_dataset_name!r})."
        )

    xs = [by_id_x[i] for i in common_ids]
    ys = [by_id_y[i] for i in common_ids]

    lo = min(min(xs), min(ys))
    hi = max(max(xs), max(ys))
    pad = 0.05 * (hi - lo) if hi > lo else 0.1
    axis_range = [lo - pad, hi + pad]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=axis_range,
            y=axis_range,
            mode="lines",
            line=dict(color="lightgray", dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(color="steelblue", size=8, opacity=0.75),
            text=[f"SysPrompt {i}" for i in common_ids],
            hovertemplate=(
                "%{text}<br>"
                + f"{x_dataset_name}=" + "%{x:.3f}<br>"
                + f"{y_dataset_name}=" + "%{y:.3f}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=x_dataset_name,
        yaxis_title=y_dataset_name,
        # ``matches="x"`` makes the y-axis mirror the x-axis layout (range AND
        # ticks). A plain shared range alone is not enough — without
        # ``matches`` Plotly's auto-tick can pick different step sizes for the
        # two axes when figure width != height, and ``scaleanchor`` distorts
        # the explicit range to keep pixel-per-unit constant. ``matches``
        # avoids both pitfalls.
        xaxis=dict(range=axis_range),
        yaxis=dict(matches="x"),
        height=560,
        width=560,
        font=dict(size=12),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    _write_figure(fig, basepath, formats or ("pdf", "svg"))


# --------------------------------------------------------------------------- #
# 4. Behavior bar (control) + box (experiment)                                #
# --------------------------------------------------------------------------- #


def plot_behavior_bar_box(
    behavior: BehaviorAggregate,
    basepath: Path,
    title: str,
    formats: Optional[Sequence[str]] = None,
) -> None:
    """
    Mixed plot:

    - Bar at ``"control"`` showing ``Accuracy_control``.
    - Box at ``"experiment"`` showing the distribution of
      ``Accuracy_experiment`` across system prompts.

    Args:
        behavior: Aggregated behavior data.
        basepath: Output path **without extension**.
        title: Figure title.
    """
    import plotly.graph_objects as go  # lazy

    if not behavior.has_data:
        raise ValueError("Cannot plot behavior summary: no behavior JSONs found.")

    fig = go.Figure()
    if behavior.accuracy_control is not None:
        fig.add_trace(
            go.Bar(
                x=["control"],
                y=[behavior.accuracy_control],
                name="control accuracy",
                marker=dict(color="goldenrod"),
                text=[f"{behavior.accuracy_control:.2f}"],
                textposition="outside",
                width=[0.55],
            )
        )
    if behavior.accuracy_experiments:
        fig.add_trace(
            go.Box(
                x=["experiment"] * len(behavior.accuracy_experiments),
                y=behavior.accuracy_experiments,
                name="experiment accuracy (across variants)",
                boxpoints="all",
                jitter=0.4,
                pointpos=0,
                marker=dict(color="royalblue", size=5, opacity=0.7),
                line=dict(color="royalblue"),
            )
        )
    fig.update_layout(
        title=title,
        yaxis_title="Accuracy",
        yaxis=dict(range=[0.0, 1.05]),
        height=480,
        width=560,
        font=dict(size=12),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
    )
    _write_figure(fig, basepath, formats or ("pdf", "svg"))


# --------------------------------------------------------------------------- #
# Class separation: mean + std band across system prompts                     #
# --------------------------------------------------------------------------- #


def plot_class_separation_curves_mean_band(
    aggregate: ClassSeparationAggregate,
    basepath: Path,
    title: str,
    formats: Optional[Sequence[str]] = None,
    chance_line: Optional[float] = 0.25,
    line_color: str = "firebrick",
    line_width: float = 2.0,
    band_color: str = "firebrick",
    band_alpha: float = 0.2,
    y_range: Sequence[float] = (0.0, 1.05),
) -> None:
    """
    Mean ± 1 std (across system prompts) per-layer 4-class CV accuracy curve.

    Args:
        aggregate: One seed's :class:`ClassSeparationAggregate` loaded via
            :func:`src.paper_plots.aggregate.load_class_separation_aggregate_for_seed`.
        basepath: Output path **without extension** — typically
            :func:`src.paper_plots.paths.class_separation_basepath`.
        title: Figure title.
        formats: Output formats; ``None`` falls back to ``("png", "svg")``.
        chance_line: Optional dashed horizontal reference line (default ``0.25``
            for four balanced classes). ``None`` suppresses the line.
        line_color: Color for the mean curve (hex / rgb()/rgba() string /
            common CSS name; see :func:`_to_rgba`).
        line_width: Pixel width of the mean curve.
        band_color: Color of the ±1 std band; combined with ``band_alpha``
            into an ``rgba(...)`` fill string.
        band_alpha: Opacity of the band in ``[0, 1]``.

    Notes:
        Variants whose ``cv_accuracy_across_layers`` is empty are dropped.
        Curves of unequal length (which should not happen in a normally
        configured run, since ``n_layers`` is a model property) are aligned
        on the leading ``min(len(curve))`` layers before stacking.
    """
    import plotly.graph_objects as go  # lazy

    fmts = tuple(formats) if formats else ("png", "svg")

    curves: list = [
        np.asarray(v.cv_accuracy_across_layers, dtype=np.float64)
        for v in aggregate.variants
        if v.cv_accuracy_across_layers
    ]
    if not curves:
        print(
            f"  skip class-separation mean-band (seed={aggregate.seed}, "
            f"nPC={aggregate.n_PC_classifier}): no curves to plot."
        )
        return

    n_layers = min(int(c.shape[0]) for c in curves)
    stacked = np.stack([c[:n_layers] for c in curves], axis=0)
    mean_curve = stacked.mean(axis=0)
    std_curve = stacked.std(axis=0)
    upper = (mean_curve + std_curve).tolist()
    lower = (mean_curve - std_curve).tolist()
    x = list(range(n_layers))

    fig = go.Figure()
    # Upper bound (transparent line) — needed as the "previous trace" for the
    # filled lower-bound trace below.
    fig.add_trace(
        go.Scatter(
            x=x,
            y=upper,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    # Lower bound (filled to previous trace) — paints the ±1 std band.
    fig.add_trace(
        go.Scatter(
            x=x,
            y=lower,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor=_to_rgba(band_color, band_alpha),
            name="± 1 std",
            hoverinfo="skip",
        )
    )
    # Mean line on top.
    fig.add_trace(
        go.Scatter(
            x=x,
            y=mean_curve.tolist(),
            mode="lines+markers",
            line=dict(color=line_color, width=float(line_width)),
            marker=dict(size=4),
            name=f"mean across {len(curves)} prompts",
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
        height=440,
        width=760,
        font=dict(size=12),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, x=0.0, itemwidth=40
        ),
    )
    _write_figure(fig, basepath, fmts)


# Color + dash convention for the per-class mean-band figure. Matches the
# analysis-stage per-seed plot in :mod:`src.analysis.plots` so the per-class
# legend reads the same in both places.
_PER_CLASS_BAND_STYLE: Dict[str, Dict[str, str]] = {
    "control_GT_0":    {"color": "goldenrod", "dash": "dash"},
    "control_GT_1":    {"color": "goldenrod", "dash": "solid"},
    "experiment_GT_0": {"color": "royalblue", "dash": "dash"},
    "experiment_GT_1": {"color": "royalblue", "dash": "solid"},
}


def plot_class_separation_per_class_curves_mean_band(
    aggregate: ClassSeparationPerClassAggregate,
    basepath: Path,
    title: str,
    formats: Optional[Sequence[str]] = None,
    chance_line: Optional[float] = 0.25,
    line_width: float = 2.0,
    band_alpha: float = 0.2,
    y_range: Sequence[float] = (0.0, 1.05),
) -> None:
    """
    Mean ± 1 std (across system prompts) per-layer CV recall curve, one line
    per ``(condition, GT)`` class.

    The plot has up to four mean lines (one per class) and four shaded
    bands; per-class colors + dash patterns mirror the per-seed plot from
    :func:`src.analysis.plots.plot_class_separation_per_class_curve`.

    Args:
        aggregate: One seed's
            :class:`ClassSeparationPerClassAggregate` loaded via
            :func:`src.paper_plots.aggregate.load_class_separation_per_class_aggregate_for_seed`.
        basepath: Output path **without extension**.
        title: Figure title.
        formats: Output formats; ``None`` falls back to ``("png", "svg")``.
        chance_line: Optional dashed horizontal reference line (default
            ``0.25`` for four balanced classes).
        line_width: Pixel width of the mean curves.
        band_alpha: Opacity of the std bands.
        y_range: Y-axis range.
    """
    import plotly.graph_objects as go  # lazy

    fmts = tuple(formats) if formats else ("png", "svg")

    if not aggregate.variants:
        print(
            f"  skip per-class class-separation mean-band (seed={aggregate.seed}, "
            f"nPC={aggregate.n_PC_classifier}): no per-class curves to plot."
        )
        return

    fig = go.Figure()
    n_plotted = 0
    n_variants_per_class: Dict[str, int] = {}
    for cls_name, style in _PER_CLASS_BAND_STYLE.items():
        curves = [
            np.asarray(v.cv_recall_per_class_across_layers[cls_name], dtype=np.float64)
            for v in aggregate.variants
            if cls_name in v.cv_recall_per_class_across_layers
            and v.cv_recall_per_class_across_layers[cls_name]
        ]
        if not curves:
            continue
        n_layers = min(int(c.shape[0]) for c in curves)
        stacked = np.stack([c[:n_layers] for c in curves], axis=0)
        mean_curve = stacked.mean(axis=0)
        std_curve = stacked.std(axis=0)
        x = list(range(n_layers))
        # Upper bound (transparent) then lower bound (filled) for the std band.
        fig.add_trace(
            go.Scatter(
                x=x,
                y=(mean_curve + std_curve).tolist(),
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=(mean_curve - std_curve).tolist(),
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=_to_rgba(style["color"], band_alpha),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=mean_curve.tolist(),
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
        n_plotted += 1
        n_variants_per_class[cls_name] = len(curves)

    if n_plotted == 0:
        print(
            f"  skip per-class class-separation mean-band (seed={aggregate.seed}, "
            f"nPC={aggregate.n_PC_classifier}): all variants empty."
        )
        return

    if chance_line is not None:
        fig.add_hline(
            y=float(chance_line),
            line_dash="dash",
            line_color="gray",
        )
    # If all four classes were drawn from the same prompt count, advertise that
    # in the legend title; otherwise just say "± 1 std across prompts".
    counts = set(n_variants_per_class.values())
    if len(counts) == 1:
        legend_title = f"mean ± 1 std across {counts.pop()} prompts"
    else:
        legend_title = "mean ± 1 std across prompts"
    fig.update_layout(
        title=title,
        xaxis_title="Layer",
        yaxis_title="Per-class CV recall",
        yaxis=dict(range=[float(y_range[0]), float(y_range[1])]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=460,
        width=820,
        font=dict(size=12),
        showlegend=True,
        legend=dict(
            title=dict(text=legend_title),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0.0,
            itemwidth=40,
        ),
    )
    _write_figure(fig, basepath, fmts)


# Per-condition binary classifier style for the paper-plot figure — solid
# lines, goldenrod=control / royalblue=experiment, matching the per-seed
# plot in :mod:`src.analysis.plots`.
_BINARY_CONDITION_BAND_STYLE: Dict[str, Dict[str, str]] = {
    "control":    {"color": "goldenrod"},
    "experiment": {"color": "royalblue"},
}


def plot_class_separation_binary_curves_mean_band(
    aggregate: ClassSeparationBinaryAggregate,
    basepath: Path,
    title: str,
    formats: Optional[Sequence[str]] = None,
    chance_line: Optional[float] = 0.5,
    line_width: float = 2.0,
    band_alpha: float = 0.2,
    y_range: Sequence[float] = (0.0, 1.05),
) -> None:
    """
    Mean ± 1 std (across system prompts) per-layer CV accuracy curve, one
    line per condition (control / experiment). Each line is the
    cross-prompt mean of separately-trained binary classifiers
    distinguishing GT=0 vs GT=1 *within* that condition.

    Args:
        aggregate: One seed's
            :class:`ClassSeparationBinaryAggregate` loaded via
            :func:`src.paper_plots.aggregate.load_class_separation_binary_aggregate_for_seed`.
        basepath: Output path **without extension**.
        title: Figure title.
        formats: Output formats; ``None`` falls back to ``("png", "svg")``.
        chance_line: Dashed reference at ``0.5`` (balanced binary). ``None``
            suppresses the line.
        line_width: Pixel width of the mean curves.
        band_alpha: Opacity of the std bands.
        y_range: Y-axis range.
    """
    import plotly.graph_objects as go  # lazy

    fmts = tuple(formats) if formats else ("png", "svg")

    if not aggregate.variants:
        print(
            f"  skip binary class-separation mean-band (seed={aggregate.seed}, "
            f"nPC={aggregate.n_PC_classifier}): no curves to plot."
        )
        return

    fig = go.Figure()
    n_plotted = 0
    n_variants_per_cond: Dict[str, int] = {}
    for cond, style in _BINARY_CONDITION_BAND_STYLE.items():
        # Drop curves that are all-NaN (a single one would poison .mean()).
        # Older runs wrote NaN-filled stats when a condition's minority GT
        # had fewer than cv_n_splits samples — see
        # compute_binary_per_condition_cv_accuracy_per_layer for the
        # forward-fix that now skips writing those entries.
        curves = []
        for v in aggregate.variants:
            arr = v.cv_accuracy_per_condition_across_layers.get(cond)
            if not arr:
                continue
            c = np.asarray(arr, dtype=np.float64)
            if not np.isfinite(c).any():
                continue
            curves.append(c)
        if not curves:
            continue
        n_layers = min(int(c.shape[0]) for c in curves)
        stacked = np.stack([c[:n_layers] for c in curves], axis=0)
        mean_curve = np.nanmean(stacked, axis=0)
        std_curve = np.nanstd(stacked, axis=0)
        x = list(range(n_layers))
        fig.add_trace(
            go.Scatter(
                x=x,
                y=(mean_curve + std_curve).tolist(),
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=(mean_curve - std_curve).tolist(),
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=_to_rgba(style["color"], band_alpha),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=mean_curve.tolist(),
                mode="lines+markers",
                line=dict(color=style["color"], width=float(line_width)),
                marker=dict(size=4),
                name=cond,
            )
        )
        n_plotted += 1
        n_variants_per_cond[cond] = len(curves)

    if n_plotted == 0:
        print(
            f"  skip binary class-separation mean-band (seed={aggregate.seed}, "
            f"nPC={aggregate.n_PC_classifier}): all variants empty."
        )
        return

    if chance_line is not None:
        fig.add_hline(
            y=float(chance_line),
            line_dash="dash",
            line_color="gray",
        )
    counts = set(n_variants_per_cond.values())
    if len(counts) == 1:
        legend_title = f"mean ± 1 std across {counts.pop()} prompts"
    else:
        legend_title = "mean ± 1 std across prompts"
    fig.update_layout(
        title=title,
        xaxis_title="Layer",
        yaxis_title="Binary CV accuracy",
        yaxis=dict(range=[float(y_range[0]), float(y_range[1])]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=460,
        width=820,
        font=dict(size=12),
        showlegend=True,
        legend=dict(
            title=dict(text=legend_title),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0.0,
            itemwidth=40,
        ),
    )
    _write_figure(fig, basepath, fmts)
