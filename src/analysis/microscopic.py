"""
Microscopic projection plots: per-datapoint signed projection (and cosine
similarity) onto the per-layer truth direction, plus 2D trajectory plots
that visualize how each target moves across layers relative to the
truth/task direction.

Pipeline
~~~~~~~~

1. From a ``task_vector_data_folder`` (the ``activations/`` dir of one
   ``(SysPrompt, seed)`` cache run), compute the per-(position, layer) **truth
   direction** from the **control** activations:

   .. code-block:: python

       truth_dir[p, L, :] = mean(act_control_GT_1[:, p, L, :])
                          - mean(act_control_GT_0[:, p, L, :])

   The truth direction is then sliced at a single ``tv_position_index`` (the
   top-1 from that folder's ``rotation_stats.json``).

2. From each ``targets_data_folder`` (possibly the same), load up to six
   ``(category, GT)`` activation groups:

   - ``honest_success_GT_{0,1}`` ← ``activations_control[_<method>]_GT_{0,1}.pkl``
   - ``lie_success_GT_{0,1}``    ← ``activations_experiment[_<method>]_GT_{0,1}.pkl``
   - ``lie_fail_GT_{0,1}``       ← ``activations_experiment[_<method>]_GT_{0,1}_fail.pkl``
     (best-effort — written by the cache stage only when the
     ``S_only_fail_in_experiment`` pool was non-empty)

   The same folder drives both projection-line plots and trajectory plots,
   and the same ``--targets`` (``group:idx[,idx...]`` tokens) selects which
   ``(group, sample_idx)`` pairs are highlighted in each plot.

3. **Projection-line plots** (``plot_microscopic_grid``): for every
   requested *target* token position render a row of four subplots
   (X = layer, Y = scalar value):

   - **Col 1.** Per-target sample (one trace per target), projection.
   - **Col 2.** Mean ± band over all samples in each loaded group, projection.
   - **Col 3.** Per-target sample, cosine similarity.
   - **Col 4.** Mean ± band, cosine similarity.

   All target positions share one figure (rows = target positions, cols = 4).

4. **Trajectory plots** (``plot_trajectory_grid``): for every requested
   target position render a row of 2D scatter cells, one per variant of two
   methodologies:

   - ``task_vector_residualPC``: x = task vector direction, y = first PC of
     the orthogonal residual (deflation).
   - ``task_vector_realignment``: x = full unit task vector, y = the unit
     vector in span(PC1, PC2) orthogonal to the task vector.

   Each cell overlays per-target layer-shaded markers + connecting lines
   across the (possibly sliced) layer range, with optional task-vector
   arrows and a faint background scatter.

The renderer uses Plotly subplots so the export pipeline lines up with the
rest of the analysis stage (``_write_figure`` from
:mod:`src.analysis.plots`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Group naming                                                                #
# --------------------------------------------------------------------------- #


# Microscopic categories, in display order. The first half of the tuple is the
# category key used in plot legends; the second half is the underlying
# ``(condition, fail)`` pair used to find the activation pickles. ``GT`` is
# resolved separately.
_CATEGORY_SOURCES: Tuple[Tuple[str, str, bool], ...] = (
    ("honest_success", "control",    False),
    ("lie_success",    "experiment", False),
    ("lie_fail",       "experiment", True),
)


# Group key = ``"{category}_GT_{gt}"``. Iteration order here drives plot order.
def microscopic_group_keys() -> List[str]:
    """Return the six microscopic group keys in display order."""
    keys: List[str] = []
    for category, _, _ in _CATEGORY_SOURCES:
        for gt in (0, 1):
            keys.append(f"{category}_GT_{gt}")
    return keys


# --------------------------------------------------------------------------- #
# User-facing group names ↔ internal loader keys                              #
# --------------------------------------------------------------------------- #


# CLI / public API uses ``control / experiment / experiment_fail`` (matching
# :data:`src.analysis.plots._DEFAULT_MANIFOLD_STYLE` and the cache stage's
# pickle filename roots), while the on-disk loader returns the
# ``honest_success / lie_success / lie_fail`` keys. The two halves of this
# dict translate at the boundary so the rest of the module never has to mix
# spellings.
USER_TO_INTERNAL_GROUP: Dict[str, str] = {
    "control_GT_0":         "honest_success_GT_0",
    "control_GT_1":         "honest_success_GT_1",
    "experiment_GT_0":      "lie_success_GT_0",
    "experiment_GT_1":      "lie_success_GT_1",
    "experiment_GT_0_fail": "lie_fail_GT_0",
    "experiment_GT_1_fail": "lie_fail_GT_1",
}
INTERNAL_TO_USER_GROUP: Dict[str, str] = {v: k for k, v in USER_TO_INTERNAL_GROUP.items()}


def _split_target_argv(specs: Sequence[str]) -> List[str]:
    """
    Pre-flatten the raw CLI tokens into one canonical
    ``"group:idx[,idx...]"`` string per output element.

    The user-friendly forms accepted as input:
        * Space-separated specs (the "standard" form): each argv element is
          already one spec — pass through.
        * Semicolon-separated specs in a single argv element:
          ``"control_GT_0:0,5; experiment_GT_1:0,2"``.
        * Comma-separated specs in a single argv element:
          ``"control_GT_0:0, experiment_GT_1:0"`` — distinguished from a
          comma-separated *index list* by the fact that the next token
          contains its own ``:``.

    Returns a flat list of canonical specs, ready for the per-spec parser.
    """
    out: List[str] = []
    for raw in specs:
        for chunk in str(raw).split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            # Within a single chunk, walk comma-separated tokens. A token
            # containing ``:`` starts a new spec (the previous tokens were
            # an idx list belonging to the previous spec).
            buffer: List[str] = []
            for tok in chunk.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                if ":" in tok and buffer:
                    out.append(",".join(buffer))
                    buffer = [tok]
                else:
                    buffer.append(tok)
            if buffer:
                out.append(",".join(buffer))
    return out


def parse_target_spec(specs: Sequence[str]) -> List[Tuple[str, int]]:
    """
    Parse ``"group:idx[,idx...]"`` CLI tokens into a flat list of
    ``(internal_group_name, sample_idx)`` pairs.

    Accepts both user-facing names (``control_GT_0`` …) and the equivalent
    internal names (``honest_success_GT_0`` …) — the former are translated
    via :data:`USER_TO_INTERNAL_GROUP`. Iteration order matches the CLI
    argument order; duplicates are preserved (the caller may want to draw
    the same ``(group, idx)`` twice in a multi-cell figure).

    Multiple specs combined into a single argv element with ``;`` or ``,``
    separators are tolerated — see :func:`_split_target_argv` — so the user
    can write any of:

        --targets control_GT_0:0,5 experiment_GT_1:0,2
        --targets "control_GT_0:0,5; experiment_GT_1:0,2"
        --targets "control_GT_0:0,5, experiment_GT_1:0,2"

    Args:
        specs: Sequence of strings of the form ``"group:idx"`` or
            ``"group:idx1,idx2,..."``. Whitespace around the colon and
            commas is stripped.

    Returns:
        ``[(internal_group, sample_idx), ...]``.

    Raises:
        ValueError: When a token doesn't match the expected format, the
            group name isn't recognized, or any index isn't a non-negative
            integer.
    """
    out: List[Tuple[str, int]] = []
    for spec in _split_target_argv(specs):
        if ":" not in spec:
            raise ValueError(
                f"Target spec {spec!r} is missing ':<idx>'; expected "
                f"'group:idx[,idx...]' (e.g. 'control_GT_0:0,5')."
            )
        group_raw, idx_block = spec.split(":", 1)
        group_user = group_raw.strip()
        if group_user in USER_TO_INTERNAL_GROUP:
            internal = USER_TO_INTERNAL_GROUP[group_user]
        elif group_user in INTERNAL_TO_USER_GROUP:
            internal = group_user
        else:
            raise ValueError(
                f"Target spec {spec!r} has unknown group {group_user!r}. "
                f"Valid names: {sorted(USER_TO_INTERNAL_GROUP)} (or their "
                f"internal equivalents)."
            )
        for raw in idx_block.split(","):
            raw = raw.strip()
            if not raw:
                continue
            try:
                idx = int(raw)
            except ValueError as e:
                raise ValueError(
                    f"Target spec {spec!r} has non-integer index "
                    f"{raw!r}."
                ) from e
            if idx < 0:
                raise ValueError(
                    f"Target spec {spec!r} has negative index {idx}."
                )
            out.append((internal, idx))
    return out


def default_targets_for_groups(
    groups: Dict[str, np.ndarray],
    sample_idx: int = 0,
) -> List[Tuple[str, int]]:
    """
    Build the default target list when ``--targets`` is empty: one target
    per available group at ``sample_idx`` (clamped to the group's size).
    Iteration order matches :func:`microscopic_group_keys`.
    """
    out: List[Tuple[str, int]] = []
    for key in microscopic_group_keys():
        arr = groups.get(key)
        if arr is None or arr.shape[0] == 0:
            continue
        out.append((key, min(int(sample_idx), arr.shape[0] - 1)))
    return out


def filter_valid_targets(
    targets: Sequence[Tuple[str, int]],
    groups: Dict[str, np.ndarray],
) -> List[Tuple[str, int]]:
    """
    Drop targets whose group isn't loaded or whose index is out of range,
    printing a one-line note per skipped entry. Returns the kept targets in
    iteration order.
    """
    kept: List[Tuple[str, int]] = []
    for group_key, idx in targets:
        arr = groups.get(group_key)
        if arr is None:
            print(
                f"  target skipped: group "
                f"{INTERNAL_TO_USER_GROUP.get(group_key, group_key)!r} not "
                f"loaded for this folder."
            )
            continue
        if not 0 <= idx < arr.shape[0]:
            print(
                f"  target skipped: sample_idx={idx} out of range for "
                f"{INTERNAL_TO_USER_GROUP.get(group_key, group_key)!r} "
                f"(size={arr.shape[0]})."
            )
            continue
        kept.append((group_key, idx))
    return kept


# --------------------------------------------------------------------------- #
# Pickle discovery                                                            #
# --------------------------------------------------------------------------- #


def _discover_activation_pickle(
    activations_dir: Path,
    condition: str,
    gt_label: int,
    fail: bool,
) -> Optional[Path]:
    """
    Locate an activation pickle by globbing.

    Matches the cache-stage filename convention
    ``activations_{condition}[_<method>]_GT_{label}[_fail].pkl`` — both with
    and without the ``_<method>`` infix that
    :func:`src.cache_activations.paths.activations_path` adds when sampling
    is enabled.

    Returns ``None`` when no candidate is found, the best-fit path when one
    or more candidates exist (longer / method-tagged filename wins so an
    explicit ``_maximum`` file beats a stray ``_fail``-stripped duplicate).
    """
    suffix = "_fail" if fail else ""
    patterns = [
        f"activations_{condition}_GT_{gt_label}{suffix}.pkl",
        f"activations_{condition}_*_GT_{gt_label}{suffix}.pkl",
    ]
    seen: List[Path] = []
    for pat in patterns:
        for p in activations_dir.glob(pat):
            # Skip the ``_fail`` companion when looking for the base file —
            # the bare-condition glob would otherwise include both.
            if not fail and p.name.endswith("_fail.pkl"):
                continue
            if p not in seen:
                seen.append(p)
    if not seen:
        return None
    # Prefer the longest filename so method-tagged variants outrank bare names
    # (deterministic tie-break = lexicographic).
    seen.sort(key=lambda p: (-len(p.name), p.name))
    return seen[0]


def _load_pickle_as_numpy(path: Path) -> np.ndarray:
    """Load a torch tensor pickle as float32 numpy."""
    import pickle  # stdlib; lazy keeps top-level import light

    with open(path, "rb") as f:
        obj = pickle.load(f)
    if hasattr(obj, "numpy"):
        return obj.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(obj, dtype=np.float32)


def load_microscopic_groups(
    activations_dir: Path,
    *,
    require_honest: bool = False,
    require_lie_success: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Load every available ``(category, GT)`` activation pickle from one
    ``activations/`` directory.

    Args:
        activations_dir: Directory containing the cache-stage pickles.
        require_honest: When ``True``, raise ``FileNotFoundError`` if the
            two ``honest_success_GT_*`` (control) pickles are missing.
        require_lie_success: Same, but for the ``lie_success_GT_*``
            (experiment) pickles.

    Returns:
        Mapping ``group_key -> ndarray(k, n_pos, n_layers, d_model)``.
        Missing optional pickles (typically ``lie_fail_*``) are simply
        absent from the dict; callers are expected to skip those groups
        with a printed note.
    """
    out: Dict[str, np.ndarray] = {}
    for category, condition, fail in _CATEGORY_SOURCES:
        for gt in (0, 1):
            key = f"{category}_GT_{gt}"
            path = _discover_activation_pickle(activations_dir, condition, gt, fail)
            if path is None:
                continue
            out[key] = _load_pickle_as_numpy(path)

    missing_honest = [k for k in ("honest_success_GT_0", "honest_success_GT_1") if k not in out]
    missing_lie = [k for k in ("lie_success_GT_0", "lie_success_GT_1") if k not in out]
    if require_honest and missing_honest:
        raise FileNotFoundError(
            f"Missing honest_success activations in {activations_dir}: "
            f"{missing_honest}"
        )
    if require_lie_success and missing_lie:
        raise FileNotFoundError(
            f"Missing lie_success activations in {activations_dir}: "
            f"{missing_lie}"
        )
    return out


# --------------------------------------------------------------------------- #
# Truth direction + projection                                                #
# --------------------------------------------------------------------------- #


def compute_truth_directions(
    act_ctrl_gt0: np.ndarray, act_ctrl_gt1: np.ndarray
) -> np.ndarray:
    """
    Per-(position, layer) truth direction ``mean(GT_1) - mean(GT_0)``.

    Args:
        act_ctrl_gt0: ``(k, n_pos, n_layers, d_model)`` control / GT_0.
        act_ctrl_gt1: ``(k, n_pos, n_layers, d_model)`` control / GT_1.

    Returns:
        ``(n_pos, n_layers, d_model)`` truth-direction tensor. ``k`` may
        differ between the two inputs — both are reduced via mean along
        axis 0 first.
    """
    if act_ctrl_gt0.shape[1:] != act_ctrl_gt1.shape[1:]:
        raise ValueError(
            f"Control activation shapes disagree on (n_pos, n_layers, d_model): "
            f"{act_ctrl_gt0.shape} vs {act_ctrl_gt1.shape}"
        )
    mean_gt1 = act_ctrl_gt1.mean(axis=0)  # (n_pos, n_layers, d_model)
    mean_gt0 = act_ctrl_gt0.mean(axis=0)
    return (mean_gt1 - mean_gt0).astype(np.float32, copy=False)


def project_group_onto_truth(
    activations: np.ndarray,
    truth_dirs: np.ndarray,
    *,
    tv_position_index: int,
    proj_position_index: int,
    eps: float = 1e-8,
) -> Dict[str, np.ndarray]:
    """
    Project every datapoint of one group onto the per-layer unit truth
    direction. The truth direction is sliced at ``tv_position_index`` (from
    the task-vector folder); the activations are sliced at
    ``proj_position_index`` (from the projection folder). The two indices may
    differ.

    Args:
        activations: ``(k, n_pos, n_layers, d_model)`` group activations.
        truth_dirs: ``(n_pos, n_layers, d_model)`` truth directions, as
            returned by :func:`compute_truth_directions`.
        tv_position_index: Position-axis index for ``truth_dirs``.
        proj_position_index: Position-axis index for ``activations``.
        eps: Numerical floor for ``||truth_dir||`` and ``||x||`` — layers /
            samples whose vector is degenerate produce zeros (so the curve
            drops to zero rather than NaN-ing out the whole figure).

    Returns:
        ``{"projection": ndarray(k, n_layers), "cosine": ndarray(k, n_layers)}``

        - ``projection``: signed scalar projection ``dot(x, t/||t||)``.
        - ``cosine``: cosine similarity ``dot(x/||x||, t/||t||)``.
    """
    if truth_dirs.shape[0] != activations.shape[1]:
        raise ValueError(
            f"truth_dirs n_pos {truth_dirs.shape[0]} != "
            f"activations n_pos {activations.shape[1]}"
        )
    if truth_dirs.shape[1] != activations.shape[2]:
        raise ValueError(
            f"truth_dirs n_layers {truth_dirs.shape[1]} != "
            f"activations n_layers {activations.shape[2]}"
        )
    if truth_dirs.shape[2] != activations.shape[3]:
        raise ValueError(
            f"truth_dirs d_model {truth_dirs.shape[2]} != "
            f"activations d_model {activations.shape[3]}"
        )
    if not 0 <= int(tv_position_index) < truth_dirs.shape[0]:
        raise ValueError(
            f"tv_position_index {tv_position_index} out of range "
            f"[0, {truth_dirs.shape[0] - 1}]"
        )
    if not 0 <= int(proj_position_index) < activations.shape[1]:
        raise ValueError(
            f"proj_position_index {proj_position_index} out of range "
            f"[0, {activations.shape[1] - 1}]"
        )

    slab = activations[:, int(proj_position_index), :, :].astype(np.float32, copy=False)
    direction = truth_dirs[int(tv_position_index), :, :].astype(np.float32, copy=False)
    norms_t = np.linalg.norm(direction, axis=-1)  # (n_layers,)
    safe_t = np.where(norms_t < eps, 1.0, norms_t)
    unit_t = direction / safe_t[:, None]
    proj = np.einsum("kld,ld->kl", slab, unit_t)  # (k, n_layers)
    bad_t = norms_t[None, :] < eps
    proj = np.where(bad_t, 0.0, proj)

    norms_x = np.linalg.norm(slab, axis=-1)  # (k, n_layers)
    safe_x = np.where(norms_x < eps, 1.0, norms_x)
    cosine = proj / safe_x
    cosine = np.where(bad_t | (norms_x < eps), 0.0, cosine)

    return {
        "projection": proj.astype(np.float32, copy=False),
        "cosine": cosine.astype(np.float32, copy=False),
    }


def project_all_groups(
    groups: Dict[str, np.ndarray],
    truth_dirs: np.ndarray,
    *,
    tv_position_index: int,
    proj_position_index: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Apply :func:`project_group_onto_truth` to every group in ``groups``.

    Returns ``{group_key: {"projection": (k, n_layers), "cosine": (k, n_layers)}}``.
    Groups absent from ``groups`` (e.g. missing ``_fail`` pickles) are not
    fabricated — the caller iterates over whichever keys came back.
    """
    return {
        g: project_group_onto_truth(
            arr,
            truth_dirs,
            tv_position_index=tv_position_index,
            proj_position_index=proj_position_index,
        )
        for g, arr in groups.items()
    }


# --------------------------------------------------------------------------- #
# Plotting                                                                    #
# --------------------------------------------------------------------------- #


# Color encodes category; GT distinction is encoded either via line dash style
# (``gt_style="line_style"``) or via marker shape (``gt_style="marker_shape"``).
# Colors mirror the manifold-plot palette in src/analysis/plots.py.
_CATEGORY_COLORS: Dict[str, str] = {
    "honest_success": "goldenrod",
    "lie_success":    "royalblue",
    "lie_fail":       "purple",
}

# GT_1 = "True" side, GT_0 = "False" side.
_GT_DASH: Dict[int, str] = {1: "solid", 0: "dash"}
_GT_MARKER: Dict[int, str] = {1: "circle", 0: "x"}


def _build_group_style(gt_style: str) -> Dict[str, Dict[str, str]]:
    """
    Build the per-group ``{color, dash, marker, mode}`` style dict for a given
    ``gt_style`` mode.

    - ``"line_style"``: GT encoded by dash (solid / dash); pure-line traces.
    - ``"marker_shape"``: GT encoded by marker symbol (circle / x); lines all
      solid, ``mode="lines+markers"``.
    """
    if gt_style not in ("line_style", "marker_shape"):
        raise ValueError(
            f"gt_style must be 'line_style' | 'marker_shape'; got {gt_style!r}"
        )
    out: Dict[str, Dict[str, str]] = {}
    for category, color in _CATEGORY_COLORS.items():
        for gt in (0, 1):
            key = f"{category}_GT_{gt}"
            if gt_style == "line_style":
                out[key] = {
                    "color": color,
                    "dash": _GT_DASH[gt],
                    "marker": "circle",
                    "mode": "lines",
                }
            else:
                out[key] = {
                    "color": color,
                    "dash": "solid",
                    "marker": _GT_MARKER[gt],
                    "mode": "lines+markers",
                }
    return out


def _hex_to_rgba(color: str, alpha: float) -> str:
    """
    Build a ``rgba(...)`` Plotly color string from a named CSS color or hex.

    Falls back to passing the color through unchanged when the input is
    already an ``rgb(...)`` / ``rgba(...)`` literal — Plotly accepts that
    directly. Used to render translucent variance bands in the same hue as
    the mean line.
    """
    import matplotlib.colors as mcolors  # lazy

    if color.startswith("rgba"):
        return color
    rgb = mcolors.to_rgb(color)  # (r, g, b) in [0, 1]
    r, g, b = (int(round(c * 255)) for c in rgb)
    return f"rgba({r},{g},{b},{alpha:.3f})"


def _parse_color_to_rgb01(color: str) -> Tuple[float, float, float]:
    """
    Parse a color string into ``(r, g, b)`` floats in ``[0, 1]``.

    Accepts CSS ``rgb(r,g,b)`` / ``rgba(r,g,b,a)`` (channels in 0–255)
    in addition to anything ``matplotlib.colors.to_rgb`` understands
    (named colors, ``#rrggbb``). Plotly emits ``rgb(...)`` strings,
    so callers that round-trip Plotly colors need this wrapper.
    """
    import re
    import matplotlib.colors as mcolors  # lazy

    s = str(color).strip()
    m = re.match(r"\s*rgba?\(([^)]+)\)\s*$", s, flags=re.IGNORECASE)
    if m:
        parts = [p.strip() for p in m.group(1).split(",")]
        r, g, b = (float(parts[0]), float(parts[1]), float(parts[2]))
        return (r / 255.0, g / 255.0, b / 255.0)
    return mcolors.to_rgb(s)


def _darken_color(color: str, fraction: float) -> str:
    """
    Multiply each RGB channel by ``fraction`` to get a darker variant.

    Used to derive a visible gridline / zero-line color from the 3D
    scene wall color so the lines don't disappear against the wall.
    ``fraction=1.0`` returns the original color; ``fraction=0.0``
    returns black.
    """
    f = float(np.clip(fraction, 0.0, 1.0))
    r, g, b = _parse_color_to_rgb01(color)
    return (
        f"rgb({int(round(r*f*255))},"
        f"{int(round(g*f*255))},"
        f"{int(round(b*f*255))})"
    )


def _mix_with_white(color: str, fraction: float) -> str:
    """
    Interpolate ``color`` along the white→color axis.

    - ``fraction=0.0`` returns pure white.
    - ``fraction=1.0`` returns the original color.
    - ``fraction=0.5`` returns the midpoint (a half-tinted version).

    Used to shift the *start* of the marker depth-gradient away from pure
    white so early-layer markers stay visibly the class color.
    """
    import matplotlib.colors as mcolors  # lazy

    f = float(np.clip(fraction, 0.0, 1.0))
    r, g, b = mcolors.to_rgb(color)
    rr = 1.0 - (1.0 - r) * f
    gg = 1.0 - (1.0 - g) * f
    bb = 1.0 - (1.0 - b) * f
    return (
        f"rgb({int(round(rr*255))},"
        f"{int(round(gg*255))},"
        f"{int(round(bb*255))})"
    )


def _legend_label(group_key: str, gt_0_name: str, gt_1_name: str) -> str:
    """Map ``honest_success_GT_0`` → ``"honest_success / False"`` etc."""
    if group_key.endswith("_GT_0"):
        return f"{group_key[: -len('_GT_0')]} / {gt_0_name}"
    if group_key.endswith("_GT_1"):
        return f"{group_key[: -len('_GT_1')]} / {gt_1_name}"
    return group_key


def plot_microscopic_grid(
    projections_per_position: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    position_labels_in_order: Sequence[str],
    targets: Sequence[Tuple[str, int]],
    basepath: Path,
    title: str,
    *,
    gt_0_name: str = "GT_0",
    gt_1_name: str = "GT_1",
    band_kind: str = "std",
    gt_style: str = "line_style",
    layer_lo: int = -1,
    layer_hi: int = -1,
    formats: Optional[Sequence[str]] = None,
    rotation_cos_sim_data: Optional[
        Dict[str, Dict[str, Optional[np.ndarray]]]
    ] = None,
    rotation_cos_sim_line_color: str = "black",
    rotation_cos_sim_band_color: str = "grey",
) -> None:
    """
    Render the four-column microscopic figure for one targets folder.

    Args:
        projections_per_position:
            ``{position_label: {group_key: {"projection": ndarray(k, n_layers),
            "cosine": ndarray(k, n_layers)}}}``. Group keys are a subset of
            :func:`microscopic_group_keys`. Missing groups (typically
            ``lie_fail_*``) are simply skipped.
        position_labels_in_order: Iteration order for the rows
            (top → bottom). Each label is a *target* token position.
        targets: ``(internal_group_name, sample_idx)`` pairs to draw in
            cols 1 and 3 (per-target sample views). One trace per target.
            Each target's group must appear in
            ``projections_per_position[position_label]`` for that row, and
            its index must be in range — out-of-range entries are skipped
            silently per row (the caller is expected to have validated
            with :func:`filter_valid_targets`). The unique *types* (group
            keys) appearing in ``targets`` also drive the mean ± band
            columns (cols 2 & 4) — sample indices are ignored there;
            curves render in canonical
            :func:`microscopic_group_keys` order.
        basepath: Output path **without extension**.
        title: Figure-level title.
        gt_0_name / gt_1_name: Human-readable GT labels for the legend.
        band_kind: Variance metric for cols 2 and 4 — ``"std"`` (1-σ,
            default), ``"sem"`` or ``"ci95"`` (1.96 × SEM).
        gt_style: ``"line_style"`` (default) encodes GT via dash style;
            ``"marker_shape"`` encodes GT via marker symbol with all-solid
            lines.
        layer_lo / layer_hi: Inclusive absolute layer range; ``-1`` on either
            bound means "no bound". Resolved via :func:`_resolve_layer_range`.
            Layers outside the range are dropped from every trace; mean / band
            statistics are computed over all samples but plotted only inside
            the range, so the y-axis auto-fits to the visible layers.
        formats: Output formats forwarded to ``_write_figure``. Defaults to
            ``("pdf", "svg")``.
        rotation_cos_sim_data: When provided, append one extra row per token
            position below the existing projection rows showing the rotation
            cosine-similarity curve (``cos_sim_across_layers`` from
            ``stage_stats/rotation_stats.json``). Outer key = position label;
            inner keys ``"individual"`` (``ndarray(n_layers,)`` or ``None``)
            drives cols 1 & 3, and ``"aggregate"``
            (``ndarray(n_sysprompts, n_layers)`` or ``None``) drives the mean
            ± band in cols 2 & 4. When ``None`` the figure layout is
            unchanged.
        rotation_cos_sim_line_color: Line color for the rotation cos_sim
            curves (individual + mean). Default ``"black"``.
        rotation_cos_sim_band_color: Fill color for the rotation cos_sim
            variance band. Default ``"grey"``.
    """
    import plotly.graph_objects as go  # lazy
    from plotly.subplots import make_subplots

    from src.analysis.plots import _write_figure  # internal helper, OK to reuse

    fmts = tuple(formats) if formats else ("pdf", "svg")
    if band_kind not in ("std", "sem", "ci95"):
        raise ValueError(
            f"band_kind must be 'std' | 'sem' | 'ci95'; got {band_kind!r}"
        )

    pos_labels = list(position_labels_in_order)
    n_rows = len(pos_labels)
    if n_rows == 0:
        raise ValueError("position_labels_in_order is empty")

    style_map = _build_group_style(gt_style)
    targets_list = list(targets)

    # Mean-band columns iterate over the *types* of group keys appearing in
    # ``targets`` (sample indices ignored), preserving the canonical
    # ``microscopic_group_keys()`` order so row layout stays predictable.
    target_group_types: set = {gk for gk, _ in targets_list}

    # Column metadata: (column index, kind, metric).
    # kind ∈ {"sample", "mean_band"}; metric ∈ {"projection", "cosine"}.
    col_specs: List[Tuple[int, str, str]] = [
        (1, "sample",    "projection"),
        (2, "mean_band", "projection"),
        (3, "sample",    "cosine"),
        (4, "mean_band", "cosine"),
    ]
    n_cols = len(col_specs)

    include_rot_rows = rotation_cos_sim_data is not None
    n_rotation_rows = n_rows if include_rot_rows else 0
    n_total_rows = n_rows + n_rotation_rows

    subplot_titles: List[str] = []
    for lbl in pos_labels:
        for _, kind, metric in col_specs:
            metric_label = "proj" if metric == "projection" else "cos"
            if kind == "sample":
                subplot_titles.append(
                    f"per-target | {metric_label} | {lbl}"
                )
            else:
                subplot_titles.append(
                    f"mean ± {band_kind} | {metric_label} | {lbl}"
                )
    if include_rot_rows:
        for lbl in pos_labels:
            for _, kind, _metric in col_specs:
                if kind == "sample":
                    subplot_titles.append(f"rot. cos_sim | pos={lbl}")
                else:
                    subplot_titles.append(
                        f"rot. cos_sim mean ± {band_kind} | pos={lbl}"
                    )

    fig = make_subplots(
        rows=n_total_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.06,
        vertical_spacing=0.10 if n_total_rows > 1 else 0.0,
        shared_xaxes=False,
    )

    legend_seen: set = set()
    group_order = microscopic_group_keys()
    y_axis_labels = {
        "projection": "Projection onto truth dir",
        "cosine": "Cosine sim to truth dir",
    }

    for r, pos_label in enumerate(pos_labels, start=1):
        per_group = projections_per_position.get(pos_label, {})
        if not per_group:
            print(
                f"  microscopic plot: no projections for token position "
                f"{pos_label!r} — row left blank."
            )
            continue
        any_metric_dict = next(iter(per_group.values()))
        any_arr = any_metric_dict["projection"]
        n_layers_full = any_arr.shape[1]
        abs_lo, abs_hi = _resolve_layer_range(layer_lo, layer_hi, n_layers_full)
        layer_slice = slice(abs_lo, abs_hi + 1)
        layers = np.arange(abs_lo, abs_hi + 1, dtype=np.int32)

        # Per-target sample columns (cols 1, 3): one trace per (group, idx).
        for group_key, idx in targets_list:
            metric_dict = per_group.get(group_key)
            if metric_dict is None:
                continue
            proj_arr = metric_dict.get("projection")
            cos_arr = metric_dict.get("cosine")
            if proj_arr is None or proj_arr.size == 0:
                continue
            if not 0 <= idx < proj_arr.shape[0]:
                continue
            style = style_map.get(
                group_key,
                {"color": "gray", "dash": "solid", "marker": "circle", "mode": "lines"},
            )
            color = style["color"]
            dash = style["dash"]
            marker_symbol = style["marker"]
            line_mode = style["mode"]
            base_label = _legend_label(group_key, gt_0_name, gt_1_name)
            target_label = f"{base_label} [sample {idx}]"

            metric_arrays = {"projection": proj_arr, "cosine": cos_arr}
            for col, kind, metric in col_specs:
                if kind != "sample":
                    continue
                arr = metric_arrays.get(metric)
                if arr is None or arr.size == 0:
                    continue
                show_legend = (col == 1) and (target_label not in legend_seen)
                if show_legend:
                    legend_seen.add(target_label)
                fig.add_trace(
                    go.Scatter(
                        x=layers,
                        y=arr[idx, layer_slice],
                        mode=line_mode,
                        line=dict(color=color, dash=dash, width=2),
                        marker=dict(symbol=marker_symbol, size=6, color=color),
                        name=target_label,
                        legendgroup=target_label,
                        showlegend=show_legend,
                        hovertemplate=(
                            f"{target_label} | {metric}"
                            f" | L%{{x}} | y=%{{y:.3f}}<extra></extra>"
                        ),
                    ),
                    row=r,
                    col=col,
                )

        # Mean ± band columns (cols 2, 4): one band per group *type* listed
        # in ``targets`` (sample indices ignored), iterated in canonical
        # ``microscopic_group_keys()`` order.
        for group_key in group_order:
            if group_key not in target_group_types:
                continue
            metric_dict = per_group.get(group_key)
            if metric_dict is None:
                continue
            proj_arr = metric_dict.get("projection")
            cos_arr = metric_dict.get("cosine")
            if proj_arr is None or proj_arr.size == 0:
                continue
            style = style_map.get(
                group_key,
                {"color": "gray", "dash": "solid", "marker": "circle", "mode": "lines"},
            )
            color = style["color"]
            dash = style["dash"]
            marker_symbol = style["marker"]
            line_mode = style["mode"]
            band_label = _legend_label(group_key, gt_0_name, gt_1_name)
            band_color = _hex_to_rgba(color, 0.18)

            metric_arrays = {"projection": proj_arr, "cosine": cos_arr}
            for col, kind, metric in col_specs:
                if kind != "mean_band":
                    continue
                arr = metric_arrays.get(metric)
                if arr is None or arr.size == 0:
                    continue
                arr = arr[:, layer_slice]
                mean = arr.mean(axis=0)
                if band_kind == "std":
                    spread = arr.std(axis=0, ddof=0)
                elif band_kind == "sem":
                    spread = arr.std(axis=0, ddof=1) / np.sqrt(max(1, arr.shape[0]))
                else:  # ci95
                    spread = (
                        1.96 * arr.std(axis=0, ddof=1) / np.sqrt(max(1, arr.shape[0]))
                    )
                upper = mean + spread
                lower = mean - spread
                show_legend = (col == 2) and (band_label not in legend_seen)
                if show_legend:
                    legend_seen.add(band_label)

                fig.add_trace(
                    go.Scatter(
                        x=layers,
                        y=upper,
                        mode="lines",
                        line=dict(color="rgba(0,0,0,0)", width=0),
                        name=band_label,
                        legendgroup=band_label,
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=r,
                    col=col,
                )
                fig.add_trace(
                    go.Scatter(
                        x=layers,
                        y=lower,
                        mode="lines",
                        line=dict(color="rgba(0,0,0,0)", width=0),
                        fill="tonexty",
                        fillcolor=band_color,
                        name=band_label,
                        legendgroup=band_label,
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=r,
                    col=col,
                )
                fig.add_trace(
                    go.Scatter(
                        x=layers,
                        y=mean,
                        mode=line_mode,
                        line=dict(color=color, dash=dash, width=2),
                        marker=dict(symbol=marker_symbol, size=6, color=color),
                        name=band_label,
                        legendgroup=band_label,
                        showlegend=show_legend,
                        hovertemplate=(
                            f"{band_label} | mean | {metric}"
                            f" | L%{{x}} | y=%{{y:.3f}}<extra></extra>"
                        ),
                    ),
                    row=r,
                    col=col,
                )

        for col, _, metric in col_specs:
            fig.add_hline(
                y=0,
                line=dict(color="lightgray", width=1, dash="dot"),
                row=r,
                col=col,
            )
            fig.update_xaxes(title_text="Layer", row=r, col=col)
            fig.update_yaxes(title_text=y_axis_labels[metric], row=r, col=col)

    # ---- rotation cos_sim rows (one per position, below projection rows) ---
    if include_rot_rows:
        rot_individual_legend_done = False
        rot_mean_legend_done = False
        rot_band_label = "rotation cos_sim mean"
        rot_indiv_label = "rotation cos_sim (this SysPrompt)"
        rot_band_fill = _hex_to_rgba(rotation_cos_sim_band_color, 0.35)
        rot_y_title = "Cosine sim across layers"
        for k, pos_label in enumerate(pos_labels):
            r_rot = n_rows + k + 1  # 1-based row index in the doubled grid
            row_data = (rotation_cos_sim_data or {}).get(pos_label) or {}
            individual = row_data.get("individual")
            aggregate = row_data.get("aggregate")

            if individual is not None:
                n_layers_full_rot = int(np.asarray(individual).shape[0])
            elif aggregate is not None and np.asarray(aggregate).size > 0:
                n_layers_full_rot = int(np.asarray(aggregate).shape[1])
            else:
                continue
            abs_lo_rot, abs_hi_rot = _resolve_layer_range(
                layer_lo, layer_hi, n_layers_full_rot
            )
            rot_slice = slice(abs_lo_rot, abs_hi_rot + 1)
            rot_layers = np.arange(abs_lo_rot, abs_hi_rot + 1, dtype=np.int32)

            if individual is not None:
                indiv_arr = np.asarray(individual, dtype=np.float64)[rot_slice]
                for col in (1, 3):
                    show_legend = (col == 1) and not rot_individual_legend_done
                    if show_legend:
                        rot_individual_legend_done = True
                    fig.add_trace(
                        go.Scatter(
                            x=rot_layers,
                            y=indiv_arr,
                            mode="lines",
                            line=dict(color=rotation_cos_sim_line_color, width=2),
                            name=rot_indiv_label,
                            legendgroup="rot_individual",
                            showlegend=show_legend,
                            hovertemplate=(
                                f"{rot_indiv_label} | pos={pos_label}"
                                f" | L%{{x}} | y=%{{y:.3f}}<extra></extra>"
                            ),
                        ),
                        row=r_rot,
                        col=col,
                    )

            if aggregate is not None:
                agg_arr = np.asarray(aggregate, dtype=np.float64)
                if agg_arr.size > 0 and agg_arr.shape[0] >= 1:
                    arr = agg_arr[:, rot_slice]
                    mean = arr.mean(axis=0)
                    if band_kind == "std":
                        spread = arr.std(axis=0, ddof=0)
                    elif band_kind == "sem":
                        spread = arr.std(axis=0, ddof=1) / np.sqrt(max(1, arr.shape[0]))
                    else:  # ci95
                        spread = (
                            1.96 * arr.std(axis=0, ddof=1)
                            / np.sqrt(max(1, arr.shape[0]))
                        )
                    upper = mean + spread
                    lower = mean - spread
                    for col in (2, 4):
                        fig.add_trace(
                            go.Scatter(
                                x=rot_layers,
                                y=upper,
                                mode="lines",
                                line=dict(color="rgba(0,0,0,0)", width=0),
                                name=rot_band_label,
                                legendgroup="rot_mean",
                                showlegend=False,
                                hoverinfo="skip",
                            ),
                            row=r_rot,
                            col=col,
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=rot_layers,
                                y=lower,
                                mode="lines",
                                line=dict(color="rgba(0,0,0,0)", width=0),
                                fill="tonexty",
                                fillcolor=rot_band_fill,
                                name=rot_band_label,
                                legendgroup="rot_mean",
                                showlegend=False,
                                hoverinfo="skip",
                            ),
                            row=r_rot,
                            col=col,
                        )
                        show_legend = (col == 2) and not rot_mean_legend_done
                        if show_legend:
                            rot_mean_legend_done = True
                        fig.add_trace(
                            go.Scatter(
                                x=rot_layers,
                                y=mean,
                                mode="lines",
                                line=dict(color=rotation_cos_sim_line_color, width=2),
                                name=rot_band_label,
                                legendgroup="rot_mean",
                                showlegend=show_legend,
                                hovertemplate=(
                                    f"{rot_band_label} | pos={pos_label}"
                                    f" | L%{{x}} | y=%{{y:.3f}}<extra></extra>"
                                ),
                            ),
                            row=r_rot,
                            col=col,
                        )

            for col in (1, 2, 3, 4):
                fig.add_hline(
                    y=0,
                    line=dict(color="lightgray", width=1, dash="dot"),
                    row=r_rot,
                    col=col,
                )
                fig.update_xaxes(title_text="Layer", row=r_rot, col=col)
                fig.update_yaxes(title_text=rot_y_title, row=r_rot, col=col)

    # Legend sits below the bottom row's x-axis title. Both compete for the
    # same paper-coordinate band, so we (a) reserve enough bottom margin for
    # the title + a multi-line legend and (b) anchor the legend's *top* edge
    # well below y=0 so it never rides up over the title.
    legend_y = -90.0 / (320 * n_total_rows + 100 + 200)  # ~90px below plot area
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        height=320 * n_total_rows + 100 + 100,  # +100 for legend room
        width=1800,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=11),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=legend_y,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(t=80, b=160, l=70, r=40),
    )
    _write_figure(fig, basepath, fmts)


# --------------------------------------------------------------------------- #
# Trajectory plots: 2D per-layer frames                                       #
# --------------------------------------------------------------------------- #


# Two methodologies are rendered side-by-side in the same figure; their
# variant catalogs are kept separate so the CLI can subset each independently.
_RESIDUALPC_VARIANTS: Tuple[str, ...] = ("raw", "unit_x", "unit_xy")
_REALIGNMENT_VARIANTS: Tuple[str, ...] = ("raw_projection", "cosine_similarity")


# PC source for the per-layer frame fitting (residual PCA / realignment PCA).
# Default mirrors the user's earlier methodological choice — successful trials
# from both conditions, excluding ``_fail`` so the PC source matches the
# task-vector cohort.
_PC_SOURCE_GROUPS: Tuple[str, ...] = (
    "honest_success_GT_0",
    "honest_success_GT_1",
    "lie_success_GT_0",
    "lie_success_GT_1",
)


_BACKGROUND_GROUP_KEYS: Dict[str, Tuple[str, ...]] = {
    "none": (),
    "control": ("honest_success_GT_0", "honest_success_GT_1"),
    "success_only": (
        "honest_success_GT_0", "honest_success_GT_1",
        "lie_success_GT_0",    "lie_success_GT_1",
    ),
    "all": (
        "honest_success_GT_0", "honest_success_GT_1",
        "lie_success_GT_0",    "lie_success_GT_1",
        "lie_fail_GT_0",       "lie_fail_GT_1",
    ),
}


def _category_for_group(group_key: str) -> str:
    """``honest_success_GT_0`` → ``honest_success``; ``lie_fail_GT_1`` → ``lie_fail``."""
    return group_key.rsplit("_GT_", 1)[0]


def _stack_groups_at(
    groups: Dict[str, np.ndarray],
    group_keys: Sequence[str],
    position_idx: int,
    layer_idx: int,
) -> np.ndarray:
    """
    Concatenate ``groups[k][:, position_idx, layer_idx, :]`` for each present
    ``k``. Returns ``(0, 0)`` when no listed group is loaded — callers fall
    back to a degenerate frame in that case.
    """
    parts: List[np.ndarray] = []
    for k in group_keys:
        arr = groups.get(k)
        if arr is None or arr.shape[0] == 0:
            continue
        parts.append(arr[:, position_idx, layer_idx, :].astype(np.float32, copy=False))
    if not parts:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(parts, axis=0)


def _residualPC_frame_at_layer(
    pc_source: np.ndarray,
    task_vec: np.ndarray,
    prev_pc1: Optional[np.ndarray] = None,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Build the ``task_vector_residualPC`` frame at one (position, layer).

    Returns ``(unit_task_vec, unit_pc1_residual, task_vec_norm, residual_pc1_std)``.
    PC1 of the residual is sign-flipped so it aligns with ``prev_pc1`` when
    given (``unit_pc1 · prev_pc1 > 0``), giving a continuous y-axis across
    layers when trajectories are overlaid. The residual is built by removing
    each row's component along ``task_vec``, then PCA-fitting a single
    component on the remaining structure.
    """
    norm_t = float(np.linalg.norm(task_vec))
    if norm_t < eps:
        unit_t = np.zeros_like(task_vec, dtype=np.float32)
        return unit_t, np.zeros_like(unit_t), 0.0, 1.0
    unit_t = (task_vec / norm_t).astype(np.float32, copy=False)

    if pc_source.shape[0] < 2:
        unit_pc1 = np.zeros_like(unit_t)
        return unit_t, unit_pc1, norm_t, 1.0

    proj_t = pc_source @ unit_t  # (N,)
    residual = pc_source - np.outer(proj_t, unit_t)  # (N, d)

    from sklearn.decomposition import PCA  # lazy
    try:
        pca = PCA(n_components=1)
        pca.fit(residual)
        unit_pc1 = pca.components_[0].astype(np.float32, copy=False)
        std_pc1 = float(np.sqrt(max(pca.explained_variance_[0], eps)))
    except Exception:  # pragma: no cover — degenerate covariance
        unit_pc1 = np.zeros_like(unit_t)
        std_pc1 = 1.0

    if prev_pc1 is not None and float(unit_pc1 @ prev_pc1) < 0:
        unit_pc1 = -unit_pc1
    return unit_t, unit_pc1, norm_t, std_pc1


def _realignment_frame_at_layer(
    pc_source: np.ndarray,
    task_vec: np.ndarray,
    prev_perp: Optional[np.ndarray] = None,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Build the ``task_vector_realignment`` frame at one (position, layer).

    The x-axis is the **full** unit task vector (it may have an out-of-plane
    component relative to the top-2 PC plane). The y-axis is the unit vector
    in ``span(PC1, PC2)`` that is orthogonal to ``u_t`` — algebraically the
    unit perpendicular within the PC plane to the in-plane projection of
    ``u_t``, which is also orthogonal to ``u_t`` itself because the
    out-of-plane part of ``u_t`` is orthogonal to anything in the PC plane.

    Returns
    ~~~~~~~

    - ``u_t`` — ``(d,)`` unit task vector.
    - ``u_perp`` — ``(d,)`` unit vector in the PC plane, orthogonal to
      ``u_t``. Sign-anchored to ``prev_perp`` when given for cross-layer
      continuity.
    - ``in_plane_fraction`` — ``‖proj_{PC plane}(u_t)‖`` (in [0, 1]). Kept
      as a diagnostic so the optional footnote can flag layers where most
      of the task direction lives outside the PC plane.
    """
    d = int(task_vec.shape[0])
    norm_t = float(np.linalg.norm(task_vec))
    if norm_t < eps:
        return (
            np.zeros(d, dtype=np.float32),
            np.zeros(d, dtype=np.float32),
            0.0,
        )
    u_t = (task_vec / norm_t).astype(np.float32, copy=False)

    if pc_source.shape[0] < 3 or d < 2:
        return u_t, np.zeros(d, dtype=np.float32), 0.0

    from sklearn.decomposition import PCA  # lazy
    pca = PCA(n_components=2)
    pca.fit(pc_source)
    pc_components = pca.components_.astype(np.float32, copy=False)  # (2, d)

    # In-plane coordinates of ``u_t`` w.r.t. (PC1, PC2). Since ``u_t`` is unit
    # the L2 norm of these coefficients equals the in-plane fraction.
    coeffs = pc_components @ u_t  # (2,)
    in_plane_fraction = float(np.linalg.norm(coeffs))
    if in_plane_fraction < eps:
        # Task direction is (effectively) orthogonal to the PC plane. Either
        # PC axis is a valid in-plane direction perpendicular to ``u_t``;
        # pick PC1 deterministically.
        u_perp = pc_components[0].astype(np.float32, copy=False)
    else:
        # PC-plane unit perpendicular to in-plane(u_t): rotate by 90°.
        u_x_pc = coeffs / in_plane_fraction
        u_y_pc = np.array([-u_x_pc[1], u_x_pc[0]], dtype=np.float32)
        u_perp = u_y_pc[0] * pc_components[0] + u_y_pc[1] * pc_components[1]

    if prev_perp is not None and float(u_perp @ prev_perp) < 0:
        u_perp = -u_perp

    return u_t, u_perp.astype(np.float32, copy=False), in_plane_fraction


def _residualPC_project(
    activations: np.ndarray,
    unit_task: np.ndarray,
    unit_pc1: np.ndarray,
    task_vec_norm: float,
    residual_std: float,
    variant: str,
) -> np.ndarray:
    """
    Project ``activations`` ``(M, d)`` into the residualPC frame at one layer.

    Returns ``(M, 2)`` coordinates for the requested variant.
    """
    x_raw = activations @ unit_task  # (M,)
    y_raw = activations @ unit_pc1   # (M,)
    if variant == "raw":
        return np.stack([x_raw, y_raw], axis=1).astype(np.float32, copy=False)
    if variant == "unit_x":
        denom = task_vec_norm if task_vec_norm > 0 else 1.0
        return np.stack([x_raw / denom, y_raw], axis=1).astype(np.float32, copy=False)
    if variant == "unit_xy":
        denom_x = task_vec_norm if task_vec_norm > 0 else 1.0
        denom_y = residual_std if residual_std > 0 else 1.0
        return np.stack(
            [x_raw / denom_x, y_raw / denom_y], axis=1
        ).astype(np.float32, copy=False)
    raise ValueError(f"Unknown task_vector_residualPC variant {variant!r}")


def _realignment_project(
    activations: np.ndarray,
    u_t: np.ndarray,
    u_perp: np.ndarray,
    variant: str,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Project ``activations`` ``(M, d)`` into the realignment frame at one layer.

    Returns ``(M, 2)`` coordinates for the requested variant.

    ``raw_projection`` returns the raw dot products onto ``(u_t, u_perp)``.
    ``cosine_similarity`` additionally divides each row by ``‖a‖`` so the
    coordinates land on (or inside) the unit circle: rows fully aligned with
    ``u_t`` reach ``(1, 0)``, rows along ``u_perp`` reach ``(0, ±1)``.
    """
    x_raw = activations @ u_t        # (M,)
    y_raw = activations @ u_perp     # (M,)
    if variant == "raw_projection":
        return np.stack([x_raw, y_raw], axis=1).astype(np.float32, copy=False)
    if variant == "cosine_similarity":
        norms = np.linalg.norm(activations, axis=1)  # (M,)
        denom = np.where(norms > eps, norms, 1.0)
        return np.stack(
            [x_raw / denom, y_raw / denom], axis=1
        ).astype(np.float32, copy=False)
    raise ValueError(f"Unknown task_vector_realignment variant {variant!r}")


def _resolve_layer_range(
    layer_lo: int, layer_hi: int, n_layers_full: int
) -> Tuple[int, int]:
    """``-1`` on either bound means 'no bound'; mirror plot_manifold_for_position."""
    lo = 0 if int(layer_lo) == -1 else int(layer_lo)
    hi = n_layers_full - 1 if int(layer_hi) == -1 else int(layer_hi)
    if not (0 <= lo <= hi <= n_layers_full - 1):
        raise ValueError(
            f"Invalid layer range layer_lo={layer_lo}, layer_hi={layer_hi} "
            f"resolved to [{lo}, {hi}] for n_layers={n_layers_full}"
        )
    return lo, hi


def _layer_color_fraction(
    layer_idx_local: int, n_layers_local: int, light_min: float
) -> float:
    """Fraction along ``[white, base_color]`` for a layer in ``[0, n_layers_local)``."""
    if n_layers_local <= 1:
        return 1.0
    return light_min + (1.0 - light_min) * layer_idx_local / (n_layers_local - 1)


def _build_per_layer_frames(
    groups: Dict[str, np.ndarray],
    truth_dirs: np.ndarray,
    *,
    tv_position_index: int,
    target_position_index: int,
    layer_lo: int,
    layer_hi: int,
) -> Dict[str, np.ndarray]:
    """
    Precompute per-layer frame data shared across all variants of both
    methodologies, for one (target_position, layer-range) cell-column.

    Returns a dict bundle:
        unit_task_vecs     : (n_layers_local, d) — also used as the realignment
                             x-axis (full unit task direction).
        unit_pc1_residuals : (n_layers_local, d) — sign-anchored
        task_vec_norms     : (n_layers_local,)
        residual_pc1_stds  : (n_layers_local,)
        realign_u_perp     : (n_layers_local, d) — unit vector in span(PC1,PC2)
                             ⊥ to ``u_t``; sign-anchored across layers.
        in_plane_fractions : (n_layers_local,) — diagnostic only (footnote).
    """
    n_layers_local = layer_hi - layer_lo + 1
    d_model = truth_dirs.shape[2]

    unit_task_vecs = np.zeros((n_layers_local, d_model), dtype=np.float32)
    unit_pc1_residuals = np.zeros((n_layers_local, d_model), dtype=np.float32)
    task_vec_norms = np.zeros(n_layers_local, dtype=np.float32)
    residual_pc1_stds = np.ones(n_layers_local, dtype=np.float32)

    realign_u_perp = np.zeros((n_layers_local, d_model), dtype=np.float32)
    in_plane_fractions = np.zeros(n_layers_local, dtype=np.float32)

    prev_pc1: Optional[np.ndarray] = None
    prev_perp: Optional[np.ndarray] = None

    for local_L, abs_L in enumerate(range(layer_lo, layer_hi + 1)):
        task_vec = truth_dirs[tv_position_index, abs_L, :]
        pc_source = _stack_groups_at(
            groups, _PC_SOURCE_GROUPS, target_position_index, abs_L
        )

        unit_t, unit_pc1, norm_t, std_pc1 = _residualPC_frame_at_layer(
            pc_source, task_vec, prev_pc1=prev_pc1
        )
        unit_task_vecs[local_L] = unit_t
        unit_pc1_residuals[local_L] = unit_pc1
        task_vec_norms[local_L] = norm_t
        residual_pc1_stds[local_L] = std_pc1
        prev_pc1 = unit_pc1 if np.linalg.norm(unit_pc1) > 0 else prev_pc1

        _u_t, u_perp, in_plane_frac = _realignment_frame_at_layer(
            pc_source, task_vec, prev_perp=prev_perp
        )
        realign_u_perp[local_L] = u_perp
        in_plane_fractions[local_L] = in_plane_frac
        prev_perp = u_perp if np.linalg.norm(u_perp) > 0 else prev_perp

    return {
        "unit_task_vecs":      unit_task_vecs,
        "unit_pc1_residuals":  unit_pc1_residuals,
        "task_vec_norms":      task_vec_norms,
        "residual_pc1_stds":   residual_pc1_stds,
        "realign_u_perp":      realign_u_perp,
        "in_plane_fractions":  in_plane_fractions,
    }


def _project_target_trajectory(
    target_acts_layers: np.ndarray,  # (n_layers_local, d)
    frames: Dict[str, np.ndarray],
    option: str,
    variant: str,
) -> np.ndarray:
    """
    Project one target's per-layer activations into the cell's 2D frame.

    Returns ``(n_layers_local, 2)``.
    """
    n_layers_local = target_acts_layers.shape[0]
    out = np.zeros((n_layers_local, 2), dtype=np.float32)
    if option == "task_vector_residualPC":
        for L in range(n_layers_local):
            xy = _residualPC_project(
                target_acts_layers[L : L + 1, :],
                frames["unit_task_vecs"][L],
                frames["unit_pc1_residuals"][L],
                float(frames["task_vec_norms"][L]),
                float(frames["residual_pc1_stds"][L]),
                variant,
            )
            out[L] = xy[0]
    elif option == "task_vector_realignment":
        for L in range(n_layers_local):
            xy = _realignment_project(
                target_acts_layers[L : L + 1, :],
                frames["unit_task_vecs"][L],
                frames["realign_u_perp"][L],
                variant,
            )
            out[L] = xy[0]
    else:
        raise ValueError(f"Unknown trajectory option {option!r}")
    return out


def _project_background_per_group(
    groups: Dict[str, np.ndarray],
    bg_keys: Sequence[str],
    target_position_index: int,
    layer_lo: int,
    layer_hi: int,
    frames: Dict[str, np.ndarray],
    option: str,
    variant: str,
    set_layer_abs: Optional[int] = None,
    light_min: float = 0.25,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Project background activations into the cell's 2D frame, keeping per-group
    arrays so the caller can color each group separately.

    Returns ``{group_key: (coords_(M, 2), layer_fracs_(M,))}`` for each
    background group that has at least one row in the (possibly restricted)
    layer range. ``layer_fracs`` index the layer-gradient colorscale; when
    ``set_layer_abs`` is given the fracs collapse to ``1.0`` (single layer).
    """
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    if not bg_keys:
        return out
    if set_layer_abs is not None:
        if not (layer_lo <= int(set_layer_abs) <= layer_hi):
            return out
        layer_iter = [(int(set_layer_abs) - layer_lo, int(set_layer_abs))]
    else:
        layer_iter = list(enumerate(range(layer_lo, layer_hi + 1)))
    n_layers_local = layer_hi - layer_lo + 1

    for key in bg_keys:
        coord_chunks: List[np.ndarray] = []
        frac_chunks: List[np.ndarray] = []
        arr = groups.get(key)
        if arr is None or arr.shape[0] == 0:
            continue
        for local_L, abs_L in layer_iter:
            slab = arr[:, target_position_index, abs_L, :].astype(
                np.float32, copy=False
            )
            if slab.shape[0] == 0:
                continue
            if option == "task_vector_residualPC":
                xy = _residualPC_project(
                    slab,
                    frames["unit_task_vecs"][local_L],
                    frames["unit_pc1_residuals"][local_L],
                    float(frames["task_vec_norms"][local_L]),
                    float(frames["residual_pc1_stds"][local_L]),
                    variant,
                )
            else:
                xy = _realignment_project(
                    slab,
                    frames["unit_task_vecs"][local_L],
                    frames["realign_u_perp"][local_L],
                    variant,
                )
            coord_chunks.append(xy)
            frac_chunks.append(
                np.full(
                    xy.shape[0],
                    _layer_color_fraction(local_L, n_layers_local, light_min),
                    dtype=np.float32,
                )
            )
        if coord_chunks:
            out[key] = (
                np.concatenate(coord_chunks, axis=0),
                np.concatenate(frac_chunks, axis=0),
            )
    return out


def _task_vector_arrow_x(
    frames: Dict[str, np.ndarray],
    option: str,
    variant: str,
    layer_local: int,
) -> float:
    """Return the x-coord of the task-vector tip at this layer + variant."""
    if option == "task_vector_residualPC":
        if variant == "raw":
            return float(frames["task_vec_norms"][layer_local])
        return 1.0  # unit_x and unit_xy both place the task vec at x=1
    # realignment: x = a · u_t. For a = task_vec this is ‖task_vec‖;
    # the cosine_similarity variant divides by ‖a‖ = ‖task_vec‖ → 1.0.
    if variant == "raw_projection":
        return float(frames["task_vec_norms"][layer_local])
    if variant == "cosine_similarity":
        return 1.0
    raise ValueError(f"Unknown trajectory variant {variant!r}")


def plot_trajectory_grid(
    groups: Dict[str, np.ndarray],
    truth_dirs: np.ndarray,
    targets: Sequence[Tuple[str, int]],
    position_labels_in_order: Sequence[str],
    pos_label_to_index: Dict[str, int],
    tv_position_index: int,
    basepath: Path,
    title: str,
    *,
    residualPC_variants: Sequence[str] = _RESIDUALPC_VARIANTS,
    realignment_variants: Sequence[str] = _REALIGNMENT_VARIANTS,
    background: str = "none",
    show_task_vector_arrow: bool = False,
    layer_lo: int = -1,
    layer_hi: int = -1,
    set_layer: Optional[int] = None,
    gt_0_name: str = "GT_0",
    gt_1_name: str = "GT_1",
    layer_shade_min: float = 0.25,
    manifold_style: Optional[Dict[str, object]] = None,
    bg_grey_scale: bool = False,
    show_in_plane_footnote: bool = False,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    formats: Optional[Sequence[str]] = None,
) -> None:
    """
    Render the trajectory figure: rows = target token positions, columns =
    variants of ``task_vector_residualPC`` followed by ``task_vector_realignment``.

    Each cell overlays per-target layer-shaded markers + connecting line
    across ``[layer_lo, layer_hi]`` (sentinel ``-1`` = no bound). Optionally
    overlays a per-group background scatter (``background ∈ {none, control,
    success_only, all}``) and a per-layer task-vector arrow.

    Args:
        groups: Mapping from internal group key (``honest_success_GT_0`` …)
            to ``(k, n_pos, n_layers, d_model)``.
        truth_dirs: ``(n_pos, n_layers, d_model)`` truth directions, sliced
            at ``tv_position_index`` for the per-layer task vector.
        targets: ``(internal_group_name, sample_idx)`` pairs to highlight.
        position_labels_in_order: Iteration order for the rows (top → bottom).
        pos_label_to_index: ``{position_label: position_axis_index}`` for the
            target activations and the truth-direction tensor (they share the
            same position axis).
        tv_position_index: Position-axis index for ``truth_dirs``.
        basepath: Output path **without extension**.
        title: Figure-level title.
        residualPC_variants / realignment_variants: Per-option variant lists.
            Pass an empty sequence to drop an entire option block.
        background: One of :data:`_BACKGROUND_GROUP_KEYS`. Picks the
            background-scatter source groups.
        show_task_vector_arrow: When ``True``, draws a layer-shaded arrow
            from the origin to ``(x_tip, 0)`` at every layer in every cell.
        layer_lo / layer_hi: Inclusive absolute layer range; ``-1`` on
            either bound means 'no bound'.
        set_layer: Absolute layer index. When given, the background scatter
            is restricted to that single layer (per-target trajectory still
            spans ``[layer_lo, layer_hi]``). ``None`` = today's behavior
            (background spans every layer in the range).
        gt_0_name / gt_1_name: Human-readable GT labels for legend rows.
        layer_shade_min: Lightest shade fraction (early layers); darkest = 1.0.
        manifold_style: Output of :func:`src.analysis.plots.resolve_manifold_style`.
            Per-group color / colorscale / marker symbol are read from this
            dict using user-facing keys (``control_GT_0`` …). ``None`` falls
            back to the manifold defaults.
        bg_grey_scale: When ``True``, every background-scatter trace renders
            as a uniform light-grey (no per-group coloring, no layer
            gradient). Per-target / per-layer trajectory markers keep their
            colors.
        show_in_plane_footnote: When ``True``, each realignment cell gets a
            small footnote with the in-plane-fraction range over the layer
            window. Default ``False``.
        xlim / ylim: Optional ``(min, max)`` pairs applied to every cell's
            x- / y-axis. ``None`` keeps Plotly's auto-range. The same range
            is forced on every column even though variants live on different
            scales (e.g. ``raw_projection`` vs ``cosine_similarity``); pick a
            window that suits the variants you're rendering.
        formats: Output formats forwarded to ``_write_figure``. Defaults to
            ``("pdf", "svg")``.
    """
    import plotly.graph_objects as go  # lazy
    from plotly.subplots import make_subplots

    from src.analysis.plots import _write_figure, resolve_manifold_style  # reuse

    fmts = tuple(formats) if formats else ("pdf", "svg")
    if background not in _BACKGROUND_GROUP_KEYS:
        raise ValueError(
            f"background must be one of {sorted(_BACKGROUND_GROUP_KEYS)}; "
            f"got {background!r}"
        )
    style = resolve_manifold_style(dict(manifold_style) if manifold_style else None)

    pos_labels = list(position_labels_in_order)
    n_rows = len(pos_labels)
    if n_rows == 0:
        raise ValueError("position_labels_in_order is empty")

    residualPC_list = list(residualPC_variants)
    realignment_list = list(realignment_variants)
    for v in residualPC_list:
        if v not in _RESIDUALPC_VARIANTS:
            raise ValueError(
                f"Unknown residualPC variant {v!r}; expected one of "
                f"{_RESIDUALPC_VARIANTS}"
            )
    for v in realignment_list:
        if v not in _REALIGNMENT_VARIANTS:
            raise ValueError(
                f"Unknown realignment variant {v!r}; expected one of "
                f"{_REALIGNMENT_VARIANTS}"
            )

    column_specs: List[Tuple[str, str]] = (
        [("task_vector_residualPC", v) for v in residualPC_list]
        + [("task_vector_realignment", v) for v in realignment_list]
    )
    n_cols = len(column_specs)
    if n_cols == 0:
        raise ValueError(
            "No trajectory variants requested — "
            "residualPC_variants and realignment_variants are both empty."
        )

    n_layers_full = truth_dirs.shape[1]
    abs_lo, abs_hi = _resolve_layer_range(layer_lo, layer_hi, n_layers_full)
    n_layers_local = abs_hi - abs_lo + 1

    # Validate target shapes vs truth-direction tensor.
    expected_d = truth_dirs.shape[2]
    for group_key, idx in targets:
        arr = groups.get(group_key)
        if arr is None:
            continue
        if arr.shape[3] != expected_d:
            raise ValueError(
                f"Target group {group_key} d_model={arr.shape[3]} disagrees "
                f"with truth direction d_model={expected_d}"
            )

    # Subplot titles only on the top row; row labels are added later as
    # paper-coordinate annotations on the left margin.
    subplot_titles: List[str] = []
    for r in range(n_rows):
        for c, (option, variant) in enumerate(column_specs):
            if r == 0:
                subplot_titles.append(f"{option}/{variant}")
            else:
                subplot_titles.append("")

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.05,
        vertical_spacing=0.10 if n_rows > 1 else 0.0,
        shared_xaxes=False,
        shared_yaxes=False,
    )

    bg_keys = _BACKGROUND_GROUP_KEYS[background]
    legend_seen: set = set()

    # Decouple the background's frame from the trajectory's frame so the
    # caller can pick a set-layer outside ``[abs_lo, abs_hi]``. Trajectory
    # markers + polylines always live in ``frames`` (built over the
    # trajectory range); the background uses ``bg_frames`` (a separate
    # single-layer build at ``set_layer`` when that layer is outside the
    # trajectory range, otherwise just ``frames`` itself).
    set_layer_abs: Optional[int] = (
        int(set_layer) if set_layer is not None else None
    )
    if set_layer_abs is not None and not (0 <= set_layer_abs < n_layers_full):
        raise ValueError(
            f"set_layer={set_layer_abs} out of range [0, {n_layers_full - 1}]"
        )
    bg_outside_traj = (
        set_layer_abs is not None and not (abs_lo <= set_layer_abs <= abs_hi)
    )

    for r, pos_label in enumerate(pos_labels, start=1):
        target_position_index = pos_label_to_index[pos_label]
        frames = _build_per_layer_frames(
            groups,
            truth_dirs,
            tv_position_index=tv_position_index,
            target_position_index=target_position_index,
            layer_lo=abs_lo,
            layer_hi=abs_hi,
        )
        if bg_outside_traj:
            bg_frames = _build_per_layer_frames(
                groups,
                truth_dirs,
                tv_position_index=tv_position_index,
                target_position_index=target_position_index,
                layer_lo=int(set_layer_abs),
                layer_hi=int(set_layer_abs),
            )
            bg_frame_lo = int(set_layer_abs)
            bg_frame_hi = int(set_layer_abs)
        else:
            bg_frames = frames
            bg_frame_lo = abs_lo
            bg_frame_hi = abs_hi

        # Pre-extract per-target layer-sliced activations once per position.
        per_target_acts: List[Tuple[Tuple[str, int], np.ndarray]] = []
        for group_key, idx in targets:
            arr = groups.get(group_key)
            if arr is None or not 0 <= idx < arr.shape[0]:
                continue
            per_target_acts.append((
                (group_key, idx),
                arr[idx, target_position_index, abs_lo : abs_hi + 1, :].astype(
                    np.float32, copy=False
                ),
            ))

        for c, (option, variant) in enumerate(column_specs, start=1):
            # Background scatter — per-group, optionally restricted to a
            # single set layer. ``bg_grey_scale`` collapses every group to a
            # uniform light-grey; otherwise each group gets its own
            # color/colorscale via ``manifold_style``. Marker size / symbol /
            # opacity / border width all come from ``manifold_style`` so the
            # trajectory background matches the manifold plot conventions.
            if bg_keys:
                bg_per_group = _project_background_per_group(
                    groups, bg_keys,
                    target_position_index=target_position_index,
                    layer_lo=bg_frame_lo, layer_hi=bg_frame_hi,
                    frames=bg_frames, option=option, variant=variant,
                    set_layer_abs=set_layer_abs,
                    light_min=layer_shade_min,
                )
                bg_marker_size = float(style.get("marker_size", 4)) if isinstance(style, dict) else 4
                bg_marker_alpha = float(style.get("marker_alpha", 0.7)) if isinstance(style, dict) else 0.7
                bg_border_width = float(
                    style.get("marker_border_width", 0.3)
                ) if isinstance(style, dict) else 0.3
                for bg_key, (bg_coords, bg_fracs) in bg_per_group.items():
                    user_key = INTERNAL_TO_USER_GROUP.get(bg_key, bg_key)
                    spec = style.get(user_key, {}) if isinstance(style, dict) else {}
                    base_color = spec.get("color", "gray")
                    bg_symbol = spec.get(
                        "symbol", "star" if bg_key.endswith("_GT_1") else "circle"
                    )
                    if bg_grey_scale:
                        marker_kwargs = dict(
                            color="lightgray",
                            symbol=bg_symbol,
                            size=bg_marker_size,
                            opacity=bg_marker_alpha,
                            line=dict(width=bg_border_width, color="darkslategrey"),
                            showscale=False,
                        )
                    else:
                        bg_scale = spec.get(
                            "colorscale", [[0.0, "white"], [1.0, base_color]]
                        )
                        marker_kwargs = dict(
                            color=bg_fracs,
                            colorscale=bg_scale,
                            cmin=0.0, cmax=1.0,
                            symbol=bg_symbol,
                            size=bg_marker_size,
                            opacity=bg_marker_alpha,
                            line=dict(width=bg_border_width, color="darkslategrey"),
                            showscale=False,
                        )
                    fig.add_trace(
                        go.Scatter(
                            x=bg_coords[:, 0],
                            y=bg_coords[:, 1],
                            mode="markers",
                            marker=marker_kwargs,
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=r, col=c,
                    )

            # Per-target trajectories (layer-shaded markers + connecting line).
            for (group_key, idx), target_acts in per_target_acts:
                user_key = INTERNAL_TO_USER_GROUP.get(group_key, group_key)
                spec = style.get(user_key, {}) if isinstance(style, dict) else {}
                base_color = spec.get("color", "gray")
                colorscale = spec.get(
                    "colorscale", [[0.0, "white"], [1.0, base_color]]
                )
                symbol = spec.get(
                    "symbol", "star" if group_key.endswith("_GT_1") else "circle"
                )
                base_label = _legend_label(group_key, gt_0_name, gt_1_name)
                target_label = f"{base_label} [sample {idx}]"
                show_legend = (r, c) == (1, 1) and target_label not in legend_seen
                if show_legend:
                    legend_seen.add(target_label)

                xy = _project_target_trajectory(
                    target_acts, frames, option=option, variant=variant
                )
                layer_fracs = [
                    _layer_color_fraction(L, n_layers_local, layer_shade_min)
                    for L in range(n_layers_local)
                ]
                hover = [
                    f"{target_label} | L{abs_lo + L} | x=%.3f, y=%.3f" % (
                        xy[L, 0], xy[L, 1]
                    )
                    for L in range(n_layers_local)
                ]
                fig.add_trace(
                    go.Scatter(
                        x=xy[:, 0],
                        y=xy[:, 1],
                        mode="lines+markers",
                        line=dict(color=base_color, width=1.2),
                        marker=dict(
                            color=layer_fracs,
                            colorscale=colorscale,
                            cmin=0.0, cmax=1.0,
                            size=7,
                            symbol=symbol,
                            opacity=0.95,
                            line=dict(width=0.5, color="darkslategrey"),
                            showscale=False,
                        ),
                        name=target_label,
                        legendgroup=target_label,
                        showlegend=show_legend,
                        hovertext=hover,
                        hoverinfo="text",
                    ),
                    row=r, col=c,
                )

            # Per-layer task-vector arrows.
            if show_task_vector_arrow:
                for L in range(n_layers_local):
                    x_tip = _task_vector_arrow_x(frames, option, variant, L)
                    if x_tip == 0.0:
                        continue
                    shade = _layer_color_fraction(L, n_layers_local, layer_shade_min)
                    rgba = f"rgba(60,60,60,{0.25 + 0.6 * shade:.3f})"
                    fig.add_trace(
                        go.Scatter(
                            x=[0.0, x_tip],
                            y=[0.0, 0.0],
                            mode="lines+markers",
                            line=dict(color=rgba, width=1.0),
                            marker=dict(
                                color=[rgba, rgba],
                                size=[0, 8],
                                symbol=["circle", "triangle-right"],
                                line=dict(width=0),
                            ),
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=r, col=c,
                    )

            # Origin crosshairs for context.
            fig.add_hline(
                y=0, line=dict(color="lightgray", width=0.6, dash="dot"),
                row=r, col=c,
            )
            fig.add_vline(
                x=0, line=dict(color="lightgray", width=0.6, dash="dot"),
                row=r, col=c,
            )

            # Per-cell footnote: in-plane fraction range (realignment only,
            # opt-in via ``show_in_plane_footnote``).
            if show_in_plane_footnote and option == "task_vector_realignment":
                fracs = frames["in_plane_fractions"]
                cell_idx = (r - 1) * n_cols + c  # 1-based subplot index
                xref = "x" if cell_idx == 1 else f"x{cell_idx}"
                yref = "y" if cell_idx == 1 else f"y{cell_idx}"
                fig.add_annotation(
                    text=(
                        f"in-plane: "
                        f"{fracs.min():.2f}–{fracs.max():.2f}"
                    ),
                    xref=f"{xref} domain",
                    yref=f"{yref} domain",
                    x=0.98, y=0.02,
                    xanchor="right", yanchor="bottom",
                    showarrow=False,
                    font=dict(size=9, color="dimgray"),
                )

            # Axis labels.
            x_title, y_title = _trajectory_axis_titles(option, variant)
            x_kwargs: Dict[str, object] = {"title_text": x_title}
            y_kwargs: Dict[str, object] = {"title_text": y_title, "scaleanchor": None}
            if xlim is not None:
                x_kwargs["range"] = [float(xlim[0]), float(xlim[1])]
            if ylim is not None:
                y_kwargs["range"] = [float(ylim[0]), float(ylim[1])]
            fig.update_xaxes(row=r, col=c, **x_kwargs)
            fig.update_yaxes(row=r, col=c, **y_kwargs)

    # Per-row labels (token positions) on the left margin.
    plot_top = 1.0
    plot_bottom = 0.0
    row_height = (plot_top - plot_bottom) / n_rows
    for r, pos_label in enumerate(pos_labels):
        y_center = plot_top - (r + 0.5) * row_height
        fig.add_annotation(
            text=f"<b>{pos_label}</b>",
            xref="paper", yref="paper",
            x=-0.04, y=y_center,
            xanchor="right", yanchor="middle",
            showarrow=False,
            font=dict(size=12, color="black"),
            textangle=-90,
        )

    legend_y = -90.0 / (360 * n_rows + 100 + 200)
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        height=360 * n_rows + 100 + 100,
        width=max(900, 320 * n_cols + 120),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=11),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=legend_y,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(t=80, b=160, l=110, r=40),
    )
    _write_figure(fig, basepath, fmts)


def _xy_to_polar(xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``(N, 2) → (r, theta_deg)``. ``theta`` is measured CCW from +x."""
    r = np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2).astype(np.float32, copy=False)
    theta = np.degrees(np.arctan2(xy[:, 1], xy[:, 0])).astype(np.float32, copy=False)
    return r, theta


def plot_trajectory_axis_polar_grid(
    *,
    groups: Dict[str, np.ndarray],
    axis_dirs: Dict[str, np.ndarray],
    axis_names: Sequence[str],
    targets: Sequence[Tuple[str, int]],
    position_labels_in_order: Sequence[str],
    pos_label_to_index: Dict[str, int],
    tv_position_index: int,
    basepath: Path,
    title: str,
    background: str = "none",
    layer_lo: int = -1,
    layer_hi: int = -1,
    set_layer: Optional[int] = None,
    gt_0_name: str = "GT_0",
    gt_1_name: str = "GT_1",
    layer_shade_min: float = 0.25,
    manifold_style: Optional[Dict[str, object]] = None,
    bg_grey_scale: bool = False,
    rlim: Optional[float] = None,
    formats: Optional[Sequence[str]] = None,
) -> None:
    """
    Polar / spherical twin of :func:`plot_trajectory_axis_grid`.

    Each row corresponds to a target token position; each column maps the
    per-sample projections onto the chosen DiM axes into polar (2D, two
    axes) or 3D Cartesian (three axes) coordinates:

    - **Col 1: raw_projection** — ``(x, y[, z]) = (a · u_axis_i)`` in raw
      activation units. 2D: mapped to ``(r, θ)`` with ``θ = 0`` along the
      first axis. 3D: plotted directly as ``(x, y, z)`` in a ``scatter3d``
      scene where each Cartesian axis corresponds to one DiM axis.
    - **Col 2: cosine_similarity** — same projections divided by ``‖a‖``.
      2D ``r ∈ [0, 1]`` = fraction of ``‖a‖`` inside the axis plane; 3D
      points live inside the unit sphere.

    ``axis_names`` must list 2 or 3 entries (1D is rendered by
    :func:`plot_trajectory_axis_grid` instead). For 3 axes a standalone
    ``.html`` is always written so the scene can be rotated.

    Args:
        axis_dirs: ``{name: (n_pos, n_layers, d_model)}`` from
            :func:`fit_axis_directions`. Must contain every name in
            ``axis_names``.
        axis_names: Ordered axis names (2 or 3 entries).
        set_layer: Layer at which to project the background-group snapshot
            when it falls outside the trajectory ``[layer_lo, layer_hi]``
            range. Ignored when ``set_layer`` lies inside the range.
        rlim: Optional ``r_max`` for 2D polar cells. ``None`` keeps Plotly's
            auto-range. Ignored in 3D.
    """
    import plotly.graph_objects as go  # lazy
    from plotly.subplots import make_subplots

    from src.analysis.plots import _write_figure, resolve_manifold_style  # reuse

    fmts = tuple(formats) if formats else ("pdf", "svg")
    if background not in _BACKGROUND_GROUP_KEYS:
        raise ValueError(
            f"background must be one of {sorted(_BACKGROUND_GROUP_KEYS)}; "
            f"got {background!r}"
        )
    n_axes = len(axis_names)
    if n_axes not in (2, 3):
        raise ValueError(
            f"axis_names must have 2 or 3 entries (use "
            f"plot_trajectory_axis_grid for 1D); got {n_axes}: "
            f"{list(axis_names)}"
        )
    missing = [n for n in axis_names if n not in axis_dirs]
    if missing:
        raise ValueError(
            f"axis_names {missing} missing from axis_dirs "
            f"(have {sorted(axis_dirs)})"
        )
    is_3d = n_axes == 3
    style = resolve_manifold_style(dict(manifold_style) if manifold_style else None)

    pos_labels = list(position_labels_in_order)
    n_rows = len(pos_labels)
    if n_rows == 0:
        raise ValueError("position_labels_in_order is empty")

    variants: Tuple[str, ...] = ("raw_projection", "cosine_similarity")
    n_cols = len(variants)

    any_arr = next(iter(groups.values()))
    n_layers_full = int(any_arr.shape[2])
    abs_lo, abs_hi = _resolve_layer_range(layer_lo, layer_hi, n_layers_full)
    n_layers_local = abs_hi - abs_lo + 1

    subplot_titles: List[str] = []
    for r in range(n_rows):
        for variant in variants:
            subplot_titles.append(variant if r == 0 else "")

    cell_type = "scene" if is_3d else "polar"
    specs = [[{"type": cell_type} for _ in range(n_cols)] for _ in range(n_rows)]
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=specs,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08 if is_3d else 0.05,
        vertical_spacing=0.10 if n_rows > 1 else 0.0,
    )

    bg_keys = _BACKGROUND_GROUP_KEYS[background]
    legend_seen: set = set()

    set_layer_abs: Optional[int] = (
        int(set_layer) if set_layer is not None else None
    )
    if set_layer_abs is not None and not (0 <= set_layer_abs < n_layers_full):
        raise ValueError(
            f"set_layer={set_layer_abs} out of range [0, {n_layers_full - 1}]"
        )

    def _proj_range(
        arr: np.ndarray, target_pos: int, lo: int, hi: int
    ) -> np.ndarray:
        """Return (k, hi-lo+1, n_axes) axis projections."""
        proj = project_onto_axes(
            arr,
            axis_dirs,
            axis_names,
            tv_position_index=tv_position_index,
            proj_position_index=target_pos,
        )
        return proj[:, lo : hi + 1, :].astype(np.float32, copy=False)

    def _norm_range(
        arr: np.ndarray, target_pos: int, lo: int, hi: int
    ) -> np.ndarray:
        """Return (k, hi-lo+1) per-sample activation norms."""
        slab = arr[:, target_pos, lo : hi + 1, :].astype(np.float32, copy=False)
        return np.linalg.norm(slab, axis=-1)

    bg_marker_size = float(style.get("marker_size", 4)) if isinstance(style, dict) else 4
    bg_marker_alpha = float(style.get("marker_alpha", 0.7)) if isinstance(style, dict) else 0.7
    bg_border_width = float(
        style.get("marker_border_width", 0.3)
    ) if isinstance(style, dict) else 0.3

    for r, pos_label in enumerate(pos_labels, start=1):
        target_position_index = pos_label_to_index[pos_label]

        # Per-target trajectory projections (one sample per (group, idx)).
        per_target: List[Tuple[Tuple[str, int], np.ndarray, np.ndarray]] = []
        for group_key, idx in targets:
            arr = groups.get(group_key)
            if arr is None or not 0 <= idx < arr.shape[0]:
                continue
            proj = _proj_range(arr, target_position_index, abs_lo, abs_hi)
            norm = _norm_range(arr, target_position_index, abs_lo, abs_hi)
            per_target.append(((group_key, idx), proj[idx], norm[idx]))

        # Single-layer background snapshot at set_layer (matches the original
        # polar plot's semantics — `_project_background_per_group` collapsed
        # to one layer when set_layer was provided).
        bg_per_group: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        if set_layer_abs is not None and bg_keys:
            for bg_key in bg_keys:
                arr = groups.get(bg_key)
                if arr is None or arr.shape[0] == 0:
                    continue
                proj = _proj_range(
                    arr, target_position_index,
                    int(set_layer_abs), int(set_layer_abs),
                )[:, 0, :]  # (k, n_axes)
                norm = _norm_range(
                    arr, target_position_index,
                    int(set_layer_abs), int(set_layer_abs),
                )[:, 0]  # (k,)
                bg_per_group[bg_key] = (proj, norm)

        for c, variant in enumerate(variants, start=1):
            # ---- Background traces (single-layer snapshot at set_layer) ----
            for bg_key, (bg_proj, bg_norm) in bg_per_group.items():
                user_key = INTERNAL_TO_USER_GROUP.get(bg_key, bg_key)
                spec = style.get(user_key, {}) if isinstance(style, dict) else {}
                base_color = spec.get("color", "gray")
                bg_symbol_2d = spec.get(
                    "symbol", "star" if bg_key.endswith("_GT_1") else "circle"
                )
                bg_symbol_3d = (
                    "circle" if bg_key.endswith("_GT_1") else "diamond"
                )
                bg_symbol = bg_symbol_3d if is_3d else bg_symbol_2d
                coords = bg_proj
                if variant == "cosine_similarity":
                    coords = coords / np.maximum(bg_norm[:, None], 1e-12)

                if bg_grey_scale:
                    marker_kwargs: Dict[str, object] = dict(
                        color="lightgray",
                        symbol=bg_symbol,
                        size=bg_marker_size,
                        opacity=bg_marker_alpha,
                        showscale=False,
                    )
                else:
                    marker_kwargs = dict(
                        color=base_color,
                        symbol=bg_symbol,
                        size=bg_marker_size,
                        opacity=bg_marker_alpha,
                        showscale=False,
                    )
                if not is_3d:
                    marker_kwargs["line"] = dict(
                        width=bg_border_width, color="darkslategrey"
                    )

                if is_3d:
                    fig.add_trace(
                        go.Scatter3d(
                            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                            mode="markers",
                            marker=marker_kwargs,
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=r, col=c,
                    )
                else:
                    bg_r, bg_theta = _xy_to_polar(coords)
                    fig.add_trace(
                        go.Scatterpolar(
                            r=bg_r,
                            theta=bg_theta,
                            mode="markers",
                            marker=marker_kwargs,
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=r, col=c,
                    )

            # ---- Per-target trajectories ----------------------------------
            for (group_key, idx), proj_one, norm_one in per_target:
                user_key = INTERNAL_TO_USER_GROUP.get(group_key, group_key)
                spec = style.get(user_key, {}) if isinstance(style, dict) else {}
                base_color = spec.get("color", "gray")
                colorscale = spec.get(
                    "colorscale", [[0.0, "white"], [1.0, base_color]]
                )
                symbol = spec.get(
                    "symbol", "star" if group_key.endswith("_GT_1") else "circle"
                )
                marker_symbol_3d = (
                    "circle" if group_key.endswith("_GT_1") else "diamond"
                )
                base_label = _legend_label(group_key, gt_0_name, gt_1_name)
                target_label = f"{base_label} [sample {idx}]"
                show_legend = (r, c) == (1, 1) and target_label not in legend_seen
                if show_legend:
                    legend_seen.add(target_label)

                coords = proj_one
                if variant == "cosine_similarity":
                    coords = coords / np.maximum(norm_one[:, None], 1e-12)

                layer_fracs = [
                    _layer_color_fraction(L, n_layers_local, layer_shade_min)
                    for L in range(n_layers_local)
                ]

                if is_3d:
                    hover = [
                        f"{target_label} | L{abs_lo + L} | "
                        f"({axis_names[0]}={coords[L, 0]:.3f}, "
                        f"{axis_names[1]}={coords[L, 1]:.3f}, "
                        f"{axis_names[2]}={coords[L, 2]:.3f})"
                        for L in range(n_layers_local)
                    ]
                    fig.add_trace(
                        go.Scatter3d(
                            x=coords[:, 0],
                            y=coords[:, 1],
                            z=coords[:, 2],
                            mode="lines+markers",
                            line=dict(color=base_color, width=4),
                            marker=dict(
                                color=layer_fracs,
                                colorscale=colorscale,
                                cmin=0.0, cmax=1.0,
                                size=5,
                                symbol=marker_symbol_3d,
                                opacity=0.95,
                                showscale=False,
                            ),
                            name=target_label,
                            legendgroup=target_label,
                            showlegend=show_legend,
                            hovertext=hover,
                            hoverinfo="text",
                        ),
                        row=r, col=c,
                    )
                else:
                    r_vals, theta_vals = _xy_to_polar(coords)
                    hover = [
                        f"{target_label} | L{abs_lo + L} | "
                        f"r={r_vals[L]:.3f}, θ={theta_vals[L]:.1f}° "
                        f"({axis_names[0]}={coords[L, 0]:.3f}, "
                        f"{axis_names[1]}={coords[L, 1]:.3f})"
                        for L in range(n_layers_local)
                    ]
                    fig.add_trace(
                        go.Scatterpolar(
                            r=r_vals,
                            theta=theta_vals,
                            mode="lines+markers",
                            line=dict(color=base_color, width=1.2),
                            marker=dict(
                                color=layer_fracs,
                                colorscale=colorscale,
                                cmin=0.0, cmax=1.0,
                                size=7,
                                symbol=symbol,
                                opacity=0.95,
                                line=dict(width=0.5, color="darkslategrey"),
                                showscale=False,
                            ),
                            name=target_label,
                            legendgroup=target_label,
                            showlegend=show_legend,
                            hovertext=hover,
                            hoverinfo="text",
                        ),
                        row=r, col=c,
                    )

            # ---- Per-cell layout (radial range / scene titles) -------------
            cell_idx = (r - 1) * n_cols + c  # 1-based subplot index
            if is_3d:
                scene_key = "scene" if cell_idx == 1 else f"scene{cell_idx}"
                if variant == "cosine_similarity":
                    fig.update_layout(
                        **{
                            scene_key: dict(
                                xaxis=dict(title=axis_names[0], range=[-1.0, 1.0]),
                                yaxis=dict(title=axis_names[1], range=[-1.0, 1.0]),
                                zaxis=dict(title=axis_names[2], range=[-1.0, 1.0]),
                            )
                        }
                    )
                else:
                    fig.update_layout(
                        **{
                            scene_key: dict(
                                xaxis=dict(title=axis_names[0]),
                                yaxis=dict(title=axis_names[1]),
                                zaxis=dict(title=axis_names[2]),
                            )
                        }
                    )
            else:
                polar_key = "polar" if cell_idx == 1 else f"polar{cell_idx}"
                radial_kwargs: Dict[str, object] = {
                    "showline": True,
                    "linewidth": 0.6,
                    "gridcolor": "lightgray",
                }
                if variant == "cosine_similarity":
                    radial_kwargs["range"] = [0.0, 1.0]
                elif rlim is not None:
                    radial_kwargs["range"] = [0.0, float(rlim)]
                angular_kwargs: Dict[str, object] = {
                    "direction": "counterclockwise",
                    "rotation": 0,
                    "gridcolor": "lightgray",
                    "tickmode": "array",
                    "tickvals": [0, 45, 90, 135, 180, 225, 270, 315],
                }
                fig.update_layout(
                    **{
                        polar_key: dict(
                            bgcolor="white",
                            radialaxis=radial_kwargs,
                            angularaxis=angular_kwargs,
                        )
                    }
                )

    plot_top = 1.0
    plot_bottom = 0.0
    row_height = (plot_top - plot_bottom) / n_rows
    for r, pos_label in enumerate(pos_labels):
        y_center = plot_top - (r + 0.5) * row_height
        fig.add_annotation(
            text=f"<b>{pos_label}</b>",
            xref="paper", yref="paper",
            x=-0.04, y=y_center,
            xanchor="right", yanchor="middle",
            showarrow=False,
            font=dict(size=12, color="black"),
            textangle=-90,
        )

    row_h = 420 if is_3d else 360
    legend_y = -90.0 / (row_h * n_rows + 100 + 200)
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        height=row_h * n_rows + 100 + 100,
        width=max(900, 360 * n_cols + 120),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=11),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=legend_y,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(t=80, b=160, l=110, r=40),
    )
    out_fmts = list(fmts)
    if is_3d and "html" not in [f.lower() for f in out_fmts]:
        out_fmts.append("html")
    _write_figure(fig, basepath, tuple(out_fmts))


def _trajectory_axis_titles(option: str, variant: str) -> Tuple[str, str]:
    """Per-cell axis-title strings; encode the variant's scaling convention."""
    if option == "task_vector_residualPC":
        if variant == "raw":
            return ("proj on task_vec", "proj on PC1(residual)")
        if variant == "unit_x":
            return ("proj on task_vec / ‖task_vec‖", "proj on PC1(residual)")
        if variant == "unit_xy":
            return (
                "proj on task_vec / ‖task_vec‖",
                "proj on PC1(residual) / std",
            )
    elif option == "task_vector_realignment":
        if variant == "raw_projection":
            return ("proj on task_vec", "proj on PC⊥(task_vec)")
        if variant == "cosine_similarity":
            return ("cos sim w/ task_vec", "cos sim w/ PC⊥(task_vec)")
    return (option, variant)


# --------------------------------------------------------------------------- #
# Axis-trajectory plots: per-DiM-axis projection trajectories                 #
# --------------------------------------------------------------------------- #
#
# These plots project per-sample, per-layer activations onto user-chosen
# named DiM (Difference-in-Means) axes, fit from the *task-vector* folder's
# microscopic groups. Each axis is a per-(position, layer) unit vector
# pointing from one group mean to another (e.g. ``honest_success_GT_1`` →
# ``honest_success_GT_0`` for ``honest_T-F``).
#
# Specifying 1 / 2 / 3 axes selects a 1D / 2D / 3D plot. The figure has two
# columns: left = per-target sample trajectories, right = per-class mean
# trajectories (with a 1-σ / CI95 band in 1D, faint per-sample traces in 2D,
# and mean-only lines in 3D where bands are visually unreadable).


# Each entry: ``(pos_groups, neg_groups)``. DiM direction is
# ``mean(pos_groups) - mean(neg_groups)`` per ``(position, layer)``, then
# unit-normalized along ``d_model``. Multi-group entries pool the named
# groups together before averaging (matches the user-facing description:
# ``honest-lie`` = honest_success (all GT) − lie_success (all GT)).
AXIS_REGISTRY: Dict[str, Tuple[List[str], List[str]]] = {
    "honest_T-F": (
        ["honest_success_GT_1"],
        ["honest_success_GT_0"],
    ),
    "honest-lie": (
        ["honest_success_GT_0", "honest_success_GT_1"],
        ["lie_success_GT_0", "lie_success_GT_1"],
    ),
    "lie_success-fail": (
        ["lie_success_GT_0", "lie_success_GT_1"],
        ["lie_fail_GT_0", "lie_fail_GT_1"],
    ),
}


def fit_axis_directions(
    tv_groups: Dict[str, np.ndarray],
    axis_names: Sequence[str],
    *,
    eps: float = 1e-8,
) -> Dict[str, np.ndarray]:
    """
    Fit per-(position, layer) unit DiM directions for each named axis.

    Args:
        tv_groups: ``{group_key: (k, n_pos, n_layers, d_model)}`` as
            returned by :func:`load_microscopic_groups`. Typically the
            task-vector folder's groups.
        axis_names: Subset of :data:`AXIS_REGISTRY` keys (deduplicated).
        eps: Numerical floor for the difference norm. Layers where the
            two group means coincide return a zero vector — projections
            onto those layers will be ``0``.

    Returns:
        ``{axis_name: (n_pos, n_layers, d_model)}`` unit DiM vectors.

    Raises:
        KeyError: For unregistered axis names.
        ValueError: When any required group from the registry entry is
            missing from ``tv_groups``.
    """
    out: Dict[str, np.ndarray] = {}
    for name in axis_names:
        if name not in AXIS_REGISTRY:
            raise KeyError(
                f"Unknown axis {name!r}; valid names: {sorted(AXIS_REGISTRY)}"
            )
        pos_groups, neg_groups = AXIS_REGISTRY[name]
        missing = [g for g in list(pos_groups) + list(neg_groups) if g not in tv_groups]
        if missing:
            raise ValueError(
                f"Axis {name!r} requires groups {missing} but they are not "
                f"present in the task-vector folder's groups "
                f"(available: {sorted(tv_groups)})."
            )
        pos_stack = np.concatenate([tv_groups[g] for g in pos_groups], axis=0)
        neg_stack = np.concatenate([tv_groups[g] for g in neg_groups], axis=0)
        pos_mean = pos_stack.mean(axis=0)  # (n_pos, n_layers, d_model)
        neg_mean = neg_stack.mean(axis=0)
        diff = (pos_mean - neg_mean).astype(np.float32, copy=False)
        norms = np.linalg.norm(diff, axis=-1, keepdims=True)
        safe = np.where(norms < eps, 1.0, norms)
        unit = diff / safe
        unit = np.where(norms < eps, 0.0, unit)
        out[name] = unit.astype(np.float32, copy=False)
    return out


def project_onto_axes(
    activations: np.ndarray,
    axis_dirs: Dict[str, np.ndarray],
    axis_names: Sequence[str],
    *,
    tv_position_index: int,
    proj_position_index: int,
) -> np.ndarray:
    """
    Project ``activations`` onto each named DiM axis at one
    ``(tv_position_index, proj_position_index)`` frame.

    Args:
        activations: ``(k, n_pos, n_layers, d_model)`` source activations.
        axis_dirs: ``{name: (n_pos, n_layers, d_model)}`` from
            :func:`fit_axis_directions`.
        axis_names: Order of the output axes (must all be in ``axis_dirs``).
        tv_position_index: Position-axis index for the *direction* (task-vector
            frame). The direction tensor is sliced here.
        proj_position_index: Position-axis index for the *activations*. May
            differ from ``tv_position_index`` (cross-position projection).

    Returns:
        ``(k, n_layers, len(axis_names))`` per-sample, per-layer scalar
        projections in ``axis_names`` order.
    """
    slab = activations[:, int(proj_position_index), :, :].astype(
        np.float32, copy=False
    )  # (k, n_layers, d_model)
    cols: List[np.ndarray] = []
    for name in axis_names:
        direction = axis_dirs[name][int(tv_position_index), :, :].astype(
            np.float32, copy=False
        )  # (n_layers, d_model)
        cols.append(np.einsum("kld,ld->kl", slab, direction))
    return np.stack(cols, axis=-1).astype(np.float32, copy=False)


def _axis_set_slug(axis_names: Sequence[str]) -> str:
    """Subfolder name encoding the chosen axis set: ``axes_<a>__<b>[__<c>]``."""
    return "axes_" + "__".join(axis_names)


def _add_segment_arrows_2d(
    fig,
    *,
    xs: np.ndarray,
    ys: np.ndarray,
    color: str,
    subplot_col: int,
) -> None:
    """
    Overlay one arrow-shaft annotation per (layer L → L+1) segment in the
    given 1x2 subplot column. Tail = (xs[i], ys[i]); head = (xs[i+1],
    ys[i+1]); both in data coords by pinning ``axref/ayref`` to the same
    subplot axes as ``xref/yref``.

    Used by :func:`plot_trajectory_axis_grid` to render arrows along the
    bold per-class mean trace in the right column. The arrow shaft lies on
    top of the mean line (intentional — the mean line is the baseline,
    the arrows add direction at each segment endpoint).
    """
    # In ``make_subplots(rows=1, cols=2)``, col=1 → x/y, col=2 → x2/y2.
    axis_suffix = "" if int(subplot_col) == 1 else str(int(subplot_col))
    xref = f"x{axis_suffix}"
    yref = f"y{axis_suffix}"
    n = int(len(xs))
    for i in range(n - 1):
        fig.add_annotation(
            x=float(xs[i + 1]),
            y=float(ys[i + 1]),
            ax=float(xs[i]),
            ay=float(ys[i]),
            xref=xref, yref=yref,
            axref=xref, ayref=yref,
            showarrow=True,
            arrowhead=2,
            arrowsize=1.0,
            arrowwidth=1.5,
            arrowcolor=color,
            standoff=4,
            startstandoff=4,
        )


def _add_segment_arrows_3d(
    fig,
    *,
    pts: np.ndarray,
    color: str,
    legendgroup: str,
    subplot_col: int,
) -> None:
    """
    Overlay a single ``go.Cone`` trace whose glyphs render an arrowhead at
    each (layer L → L+1) segment endpoint in the 3D right-column subplot.
    Each cone is anchored at the segment tip (``anchor="tip"``) with its
    direction vector = (pts[L+1] - pts[L]); the resulting glyph reads as
    an arrowhead pointing at the next layer's data point.

    Plotly has no data-coord arrow tails in 3D scene annotations, so we
    use ``go.Cone`` instead of annotations for the 3D case.

    ``sizeref`` is set to a constant fraction of the median segment length
    so cones are visible without overwhelming short segments.
    """
    import plotly.graph_objects as go  # lazy

    if pts.shape[0] < 2:
        return
    diffs = pts[1:] - pts[:-1]                        # (n-1, 3)
    anchors = pts[1:]                                 # (n-1, 3)
    seg_lens = np.linalg.norm(diffs, axis=1)
    typical = float(np.median(seg_lens)) if seg_lens.size else 0.0
    sizeref = max(typical * 0.35, 1e-3)
    fig.add_trace(
        go.Cone(
            x=anchors[:, 0], y=anchors[:, 1], z=anchors[:, 2],
            u=diffs[:, 0], v=diffs[:, 1], w=diffs[:, 2],
            colorscale=[[0.0, color], [1.0, color]],
            showscale=False,
            sizemode="absolute",
            sizeref=sizeref,
            anchor="tip",
            legendgroup=legendgroup,
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1, col=int(subplot_col),
    )


def plot_trajectory_axis_grid(
    *,
    groups: Dict[str, np.ndarray],
    axis_dirs: Dict[str, np.ndarray],
    axis_names: Sequence[str],
    targets: Sequence[Tuple[str, int]],
    target_position_index: int,
    tv_position_index: int,
    target_position_label: str,
    layer_lo: int = -1,
    layer_hi: int = -1,
    layer_lo_mean: Optional[int] = None,
    layer_hi_mean: Optional[int] = None,
    layer_lo_individual: Optional[int] = None,
    layer_hi_individual: Optional[int] = None,
    band_kind: str = "std",
    gt_0_name: str = "GT_0",
    gt_1_name: str = "GT_1",
    gt_style: str = "line_style",
    layer_shade_min: float = 0.25,
    max_target_per_class: int = 50,
    line_width_individual: float = 0.8,
    line_width_mean: float = 2.5,
    individual_line_alpha: float = 0.18,
    scene_show_walls: bool = True,
    scene_wall_color: Optional[str] = None,
    marker_layer_gradient: bool = True,
    marker_gradient_start: float = 0.5,
    marker_size_target: Optional[float] = None,
    marker_size_mean: Optional[float] = None,
    sample_seed: int = 0,
    show_arrows: bool = False,
    library: str = "plotly",
    axis_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    groups_mean: Optional[Dict[str, np.ndarray]] = None,
    groups_individual: Optional[Dict[str, np.ndarray]] = None,
    target_position_index_mean: Optional[int] = None,
    target_position_index_individual: Optional[int] = None,
    target_position_label_mean: Optional[str] = None,
    target_position_label_individual: Optional[str] = None,
    basepath: Path,
    title: str,
    formats: Sequence[str] = ("png",),
) -> None:
    """
    Render a 2-column axis-trajectory figure for one (target folder, target
    position) cell.

    Left column: per-target sample trajectories (one trace per ``(group,
    sample_idx)`` in ``targets``), colored by category with GT distinction
    via dash / marker (configurable via ``gt_style``) in 1D / 2D; 3D uses
    color + marker symbol because Plotly's ``scatter3d`` ignores ``dash``.

    Right column: per-group mean trajectory overlaid on a seeded subsample
    of individual sample trajectories. In all dimensionalities (1D / 2D /
    3D) up to ``max_target_per_class`` per-sample lines are drawn behind
    the bold mean line. ``band_kind`` is no longer consulted by this plot
    (kept on the signature for backward compatibility).

    Args:
        groups: ``{group_key: (k, n_pos, n_layers, d_model)}`` target
            activations.
        axis_dirs: Output of :func:`fit_axis_directions`. Must contain every
            name in ``axis_names``.
        axis_names: 1, 2, or 3 entries selecting the dimensionality.
        targets: ``(internal_group_name, sample_idx)`` pairs. Their
            deduplicated group set drives the mean+individuals column.
        target_position_index / tv_position_index: As in
            :func:`project_onto_axes`.
        target_position_label: Human-readable label used in the title.
        layer_lo / layer_hi: Inclusive absolute layer range; ``-1`` = no
            bound. Drives the left subplot (per-target sample) and
            provides the default window for the right subplot's mean
            line and faint per-sample overlays when the
            ``layer_*_mean`` / ``layer_*_individual`` overrides below
            are ``None``.
        layer_lo_mean / layer_hi_mean: Optional independent window for
            the right subplot's bold per-class mean trace.
            ``None`` (default) falls back to ``layer_lo`` / ``layer_hi``.
            ``-1`` on either side means "no bound on that side"; the
            usual ``min < max`` rule from :func:`_resolve_layer_range`
            applies once the sentinels are filled in.
        layer_lo_individual / layer_hi_individual: Optional independent
            window for the right subplot's faint per-sample overlay
            traces. Same fallback / sentinel semantics as
            ``layer_*_mean``.
        band_kind: Retained for backward compatibility; unused by this
            plot (individuals overlay replaces the previous band).
        gt_0_name / gt_1_name: Legend label fragments.
        gt_style: ``"line_style"`` (dash for GT in 1D / 2D) or
            ``"marker_shape"`` (always shape).
        layer_shade_min: Lightest layer color fraction; darkest = 1.0.
        max_target_per_class: Cap on per-class individual trajectories
            drawn under the mean. ``-1`` = draw every sample. When the
            class has more than this many samples, a seeded random subset
            is drawn.
        line_width_individual: Line width for the per-sample (faint)
            overlay traces.
        line_width_mean: Line width for the bold per-class mean trace.
        individual_line_alpha: Alpha (``[0, 1]``) blended onto the class
            color for the faint per-sample overlay traces. Lower = more
            washed out. Default ``0.18`` reads as near-white for light
            class colors; bump to ``0.35``–``0.45`` to make individuals
            visibly the class hue in static (SVG/PNG) exports.
        scene_show_walls: 3D scenes only. ``True`` keeps Plotly's default
            cube back-walls (each axis renders an opaque colored panel
            behind the data). ``False`` hides the wall panels (sets
            ``showbackground=False`` on every scene axis) — fixes the
            "white wall" artifact seen in static SVG export where the
            wall panels render as flat opaque white. 1D / 2D ignore.
        scene_wall_color: 3D scenes only; ignored when
            ``scene_show_walls=False``. ``None`` lets Plotly choose
            (the default, light-grey panels that often rasterize as
            near-white in SVG export). Pass an explicit CSS color
            string (e.g. ``"rgb(220,220,220)"``) to force the wall
            panels to a visible color so the static export shows the
            cube rather than blending into a white page.
        marker_layer_gradient: 2D / 3D only. ``True`` (default) shades
            each marker by layer index along a white→class-color (left
            subplot) / ``marker_gradient_start``→class-color (right
            subplot) gradient. ``False`` paints every marker the flat
            class color and ignores ``marker_gradient_start``. 1D has
            no per-marker gradient regardless.
        marker_gradient_start: Starting tint of the per-marker depth
            gradient along the mean trajectory (2D / 3D only, ignored
            when ``marker_layer_gradient=False``). ``0.0`` = pure white
            at the first layer (the legacy look — early markers look
            like white blobs in static SVG). ``1.0`` = uniform class
            color (no gradient — every marker the same). ``0.5``
            (default) = start at a half-tinted class color so the depth
            cue is preserved but the first markers are visibly the
            class hue.
        marker_size_target / marker_size_mean: Optional uniform marker
            sizes for the left (per-target sample) and right (per-class
            mean) subplots respectively. The value is in Plotly pixel
            units (also matplotlib ``plot()``'s ``markersize`` points);
            matplotlib ``scatter()`` (2D/3D markers) uses ``size**2``
            because its ``s=`` kwarg is area in points². ``None``
            (default) keeps the legacy per-dimension defaults
            (left: 1D=6, 2D=7, 3D=4; right: 1D=6, 2D=8, 3D=5).
        sample_seed: RNG seed for the per-class subsample selection.
        show_arrows: Draw arrows pointing from each layer's data point to
            the next along both the left-column per-target sample
            trajectories and the right-column bold per-class mean trace.
            1D / 2D render via Plotly ``layout.annotations`` (real arrow
            shafts + heads in data coords); 3D renders via ``go.Cone``
            glyphs anchored at each segment endpoint with direction along
            the segment (Plotly's scene annotations have no data-coord
            arrow tails in 3D).
        basepath: Output path *without extension*.
        title: Figure-level title.
        library: Backend used to render the figure. ``"plotly"`` (default)
            preserves the legacy interactive figure with HTML export for 3D.
            ``"matplotlib"`` renders the same 1D / 2D / 3D layout via
            matplotlib; ``scene_show_walls`` / ``scene_wall_color`` are
            silently ignored, and ``"html"`` is dropped from ``formats``
            (matplotlib has no interactive HTML export for these figures).
        axis_ranges: Optional ``{axis_name: (min, max)}`` map keyed by the
            DiM axis names appearing in ``axis_names``. Same range applied
            to both left and right subplots. Mapping: 1D → axis_names[0]
            sets the y-axis range (x stays "Layer"); 2D → axis_names[0]
            sets x, axis_names[1] sets y; 3D → x, y, z in order. Axis
            names missing from the dict are left on auto. ``None`` /
            empty dict = every axis on auto.
        groups_mean / groups_individual: Optional separate activation
            dictionaries that drive the right-column subplot's mean
            (``groups_mean``) and faint individual overlay
            (``groups_individual``). ``None`` (default) makes both fall
            back to ``groups`` — preserving legacy behavior where both
            right-column traces come from the same source as the left
            column.
        target_position_index_mean / target_position_index_individual:
            Optional per-source overrides for the token position index
            used when slicing the mean / individual data. ``None``
            (default) falls back to ``target_position_index``.
        target_position_label_mean / target_position_label_individual:
            Human-readable counterparts for the two indices above (used
            only when both differ from ``target_position_label`` to
            extend the figure title with both labels).
        formats: Output formats forwarded to :func:`_write_figure`. ``html``
            is automatically added for 3D plots so the scene can be rotated.
    """
    if library == "matplotlib":
        _plot_trajectory_axis_grid_matplotlib(
            groups=groups,
            axis_dirs=axis_dirs,
            axis_names=axis_names,
            targets=targets,
            target_position_index=target_position_index,
            tv_position_index=tv_position_index,
            target_position_label=target_position_label,
            layer_lo=layer_lo,
            layer_hi=layer_hi,
            layer_lo_mean=layer_lo_mean,
            layer_hi_mean=layer_hi_mean,
            layer_lo_individual=layer_lo_individual,
            layer_hi_individual=layer_hi_individual,
            gt_0_name=gt_0_name,
            gt_1_name=gt_1_name,
            gt_style=gt_style,
            layer_shade_min=layer_shade_min,
            max_target_per_class=max_target_per_class,
            line_width_individual=line_width_individual,
            line_width_mean=line_width_mean,
            individual_line_alpha=individual_line_alpha,
            marker_layer_gradient=marker_layer_gradient,
            marker_gradient_start=marker_gradient_start,
            marker_size_target=marker_size_target,
            marker_size_mean=marker_size_mean,
            sample_seed=sample_seed,
            show_arrows=show_arrows,
            axis_ranges=axis_ranges,
            groups_mean=groups_mean,
            groups_individual=groups_individual,
            target_position_index_mean=target_position_index_mean,
            target_position_index_individual=target_position_index_individual,
            target_position_label_mean=target_position_label_mean,
            target_position_label_individual=target_position_label_individual,
            basepath=basepath,
            title=title,
            formats=formats,
        )
        return
    if library != "plotly":
        raise ValueError(
            f"library must be 'plotly' | 'matplotlib'; got {library!r}"
        )

    import plotly.graph_objects as go  # lazy
    from plotly.subplots import make_subplots

    from src.analysis.plots import _write_figure  # reuse

    n_axes = len(axis_names)
    if n_axes not in (1, 2, 3):
        raise ValueError(
            f"axis_names must have 1, 2, or 3 entries; got {n_axes}: "
            f"{list(axis_names)}"
        )
    if band_kind not in ("std", "sem", "ci95"):
        raise ValueError(
            f"band_kind must be 'std' | 'sem' | 'ci95'; got {band_kind!r}"
        )
    for name in axis_names:
        if name not in axis_dirs:
            raise ValueError(
                f"axis {name!r} missing from axis_dirs (got {sorted(axis_dirs)})"
            )

    # Determine per-source layer ranges — peek at any target group to
    # infer ``n_layers_full``. The left subplot (``groups`` /
    # ``target_position_index``) always uses ``layer_lo`` / ``layer_hi``;
    # the right subplot's mean line and faint individual overlays each
    # have an optional independent window with the same sentinel.
    any_arr = next(iter(groups.values()))
    n_layers_full = any_arr.shape[2]
    abs_lo, abs_hi = _resolve_layer_range(layer_lo, layer_hi, n_layers_full)
    n_layers_local = abs_hi - abs_lo + 1
    layer_indices = np.arange(abs_lo, abs_hi + 1, dtype=np.int32)

    def _resolve_or_default(
        lo_override: Optional[int], hi_override: Optional[int]
    ) -> Tuple[int, int]:
        lo_eff = int(layer_lo) if lo_override is None else int(lo_override)
        hi_eff = int(layer_hi) if hi_override is None else int(hi_override)
        return _resolve_layer_range(lo_eff, hi_eff, n_layers_full)

    abs_lo_mean, abs_hi_mean = _resolve_or_default(layer_lo_mean, layer_hi_mean)
    abs_lo_indiv, abs_hi_indiv = _resolve_or_default(
        layer_lo_individual, layer_hi_individual
    )
    n_layers_local_mean = abs_hi_mean - abs_lo_mean + 1
    n_layers_local_indiv = abs_hi_indiv - abs_lo_indiv + 1
    layer_indices_mean = np.arange(abs_lo_mean, abs_hi_mean + 1, dtype=np.int32)
    layer_indices_indiv = np.arange(abs_lo_indiv, abs_hi_indiv + 1, dtype=np.int32)

    style_map = _build_group_style(gt_style)

    # Right-column source fallbacks: ``groups_mean`` / ``groups_individual``
    # default to ``groups``; the per-source position indices default to
    # ``target_position_index``. With all three defaults active this matches
    # the legacy single-source behavior.
    groups_mean_eff = groups_mean if groups_mean is not None else groups
    groups_indiv_eff = (
        groups_individual if groups_individual is not None else groups
    )
    pos_index_mean = (
        int(target_position_index_mean)
        if target_position_index_mean is not None
        else int(target_position_index)
    )
    pos_index_indiv = (
        int(target_position_index_individual)
        if target_position_index_individual is not None
        else int(target_position_index)
    )

    # Targets drive both columns: left col = each (group, idx); right col =
    # one mean line per *group* in targets (deduplicated, in first-seen
    # order so legends stay deterministic). The right column uses
    # ``groups_mean_eff`` / ``groups_indiv_eff`` for membership.
    seen_groups: set = set()
    target_groups_in_order: List[str] = []
    for group_key, _ in targets:
        if group_key in seen_groups:
            continue
        if (group_key not in groups_mean_eff) and (
            group_key not in groups_indiv_eff
        ):
            continue
        seen_groups.add(group_key)
        target_groups_in_order.append(group_key)

    # ---- Per-sample projections, cached per group and per source ----------
    # axis_names → shape (k, n_layers_window, n_axes) per group. Each
    # source (left / right-mean / right-indiv) gets its own layer window
    # so the three subplots can show different layer ranges.
    def _build_per_group_proj(
        src: Dict[str, np.ndarray],
        pos_index: int,
        keys: set,
        win_lo: int,
        win_hi: int,
    ) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for group_key in keys:
            arr = src.get(group_key)
            if arr is None:
                continue
            proj = project_onto_axes(
                arr,
                axis_dirs,
                axis_names,
                tv_position_index=tv_position_index,
                proj_position_index=pos_index,
            )  # (k, n_layers, n_axes)
            out[group_key] = proj[:, win_lo : win_hi + 1, :]
        return out

    left_keys = {g for g, _ in targets}
    right_keys = set(target_groups_in_order)
    per_group_proj = _build_per_group_proj(
        groups, int(target_position_index), left_keys, abs_lo, abs_hi
    )
    same_mean_source = (
        groups_mean_eff is groups
        and pos_index_mean == int(target_position_index)
        and abs_lo_mean == abs_lo
        and abs_hi_mean == abs_hi
    )
    if same_mean_source:
        per_group_proj_mean = per_group_proj
    else:
        per_group_proj_mean = _build_per_group_proj(
            groups_mean_eff, pos_index_mean, right_keys, abs_lo_mean, abs_hi_mean
        )
    if (
        groups_indiv_eff is groups
        and pos_index_indiv == int(target_position_index)
        and abs_lo_indiv == abs_lo
        and abs_hi_indiv == abs_hi
    ):
        per_group_proj_indiv = per_group_proj
    elif (
        groups_indiv_eff is groups_mean_eff
        and pos_index_indiv == pos_index_mean
        and abs_lo_indiv == abs_lo_mean
        and abs_hi_indiv == abs_hi_mean
    ):
        per_group_proj_indiv = per_group_proj_mean
    else:
        per_group_proj_indiv = _build_per_group_proj(
            groups_indiv_eff,
            pos_index_indiv,
            right_keys,
            abs_lo_indiv,
            abs_hi_indiv,
        )

    # ---- Subplot layout ----------------------------------------------------
    is_3d = (n_axes == 3)
    specs: Optional[List[List[Dict[str, str]]]]
    if is_3d:
        specs = [[{"type": "scene"}, {"type": "scene"}]]
    else:
        specs = None

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("per-target sample", f"per-class mean"),
        horizontal_spacing=0.08,
        specs=specs,
    )

    legend_seen: set = set()

    def _style_for(group_key: str) -> Dict[str, str]:
        return style_map.get(
            group_key,
            {"color": "gray", "dash": "solid", "marker": "circle", "mode": "lines"},
        )

    # Pre-compute layer color fractions per source. Each subplot
    # (left / right-mean / right-indiv) may have a different layer window
    # so the white→class-color shade map must be sized to match.
    def _make_layer_fracs(n: int) -> np.ndarray:
        return np.array(
            [_layer_color_fraction(L, n, layer_shade_min) for L in range(n)],
            dtype=np.float32,
        )

    layer_fracs = _make_layer_fracs(n_layers_local)
    if n_layers_local_mean == n_layers_local:
        layer_fracs_mean = layer_fracs
    else:
        layer_fracs_mean = _make_layer_fracs(n_layers_local_mean)

    # ---- Left column: per-target sample trajectories ----------------------
    for group_key, idx in targets:
        proj = per_group_proj.get(group_key)
        if proj is None or not 0 <= idx < proj.shape[0]:
            continue
        style = _style_for(group_key)
        color = style["color"]
        dash = style["dash"]
        marker_symbol_2d = style["marker"]
        line_mode = style["mode"]
        # In 3D plotly ignores ``dash`` — use marker symbol to distinguish GT.
        marker_symbol_3d = (
            "circle" if group_key.endswith("_GT_1") else "diamond"
        )
        base_label = _legend_label(group_key, gt_0_name, gt_1_name)
        trace_label = f"{base_label} [sample {idx}]"
        show_legend = trace_label not in legend_seen
        if show_legend:
            legend_seen.add(trace_label)

        traj = proj[idx]  # (n_layers_local, n_axes)
        colorscale = [[0.0, "white"], [1.0, color]]

        if n_axes == 1:
            fig.add_trace(
                go.Scatter(
                    x=layer_indices,
                    y=traj[:, 0],
                    mode=line_mode,
                    line=dict(color=color, dash=dash, width=2),
                    marker=dict(
                        symbol=marker_symbol_2d,
                        size=(float(marker_size_target) if marker_size_target is not None else 6),
                        color=color,
                    ),
                    name=trace_label,
                    legendgroup=trace_label,
                    showlegend=show_legend,
                    hovertemplate=(
                        f"{trace_label} | L%{{x}} | {axis_names[0]}=%{{y:.3f}}"
                        "<extra></extra>"
                    ),
                ),
                row=1, col=1,
            )
            if show_arrows and n_layers_local >= 2:
                _add_segment_arrows_2d(
                    fig,
                    xs=layer_indices.astype(np.float64),
                    ys=traj[:, 0].astype(np.float64),
                    color=color,
                    subplot_col=1,
                )
        elif n_axes == 2:
            marker_color_kw = (
                dict(color=layer_fracs, colorscale=colorscale, cmin=0.0, cmax=1.0)
                if marker_layer_gradient
                else dict(color=color)
            )
            fig.add_trace(
                go.Scatter(
                    x=traj[:, 0],
                    y=traj[:, 1],
                    mode="lines+markers",
                    line=dict(color=color, dash=dash, width=1.5),
                    marker=dict(
                        **marker_color_kw,
                        size=(float(marker_size_target) if marker_size_target is not None else 7),
                        symbol=marker_symbol_2d,
                        opacity=0.95,
                        line=dict(width=0.5, color="darkslategrey"),
                        showscale=False,
                    ),
                    name=trace_label,
                    legendgroup=trace_label,
                    showlegend=show_legend,
                ),
                row=1, col=1,
            )
            if show_arrows and n_layers_local >= 2:
                _add_segment_arrows_2d(
                    fig,
                    xs=traj[:, 0].astype(np.float64),
                    ys=traj[:, 1].astype(np.float64),
                    color=color,
                    subplot_col=1,
                )
        else:  # 3D
            marker_color_kw = (
                dict(color=layer_fracs, colorscale=colorscale, cmin=0.0, cmax=1.0)
                if marker_layer_gradient
                else dict(color=color)
            )
            fig.add_trace(
                go.Scatter3d(
                    x=traj[:, 0],
                    y=traj[:, 1],
                    z=traj[:, 2],
                    mode="lines+markers",
                    line=dict(color=color, width=4),
                    marker=dict(
                        **marker_color_kw,
                        size=(float(marker_size_target) if marker_size_target is not None else 4),
                        symbol=marker_symbol_3d,
                        opacity=0.95,
                        showscale=False,
                    ),
                    name=trace_label,
                    legendgroup=trace_label,
                    showlegend=show_legend,
                ),
                row=1, col=1,
            )
            if show_arrows and n_layers_local >= 2:
                _add_segment_arrows_3d(
                    fig,
                    pts=traj.astype(np.float64),
                    color=color,
                    legendgroup=trace_label,
                    subplot_col=1,
                )

    # ---- Right column: per-class individuals overlaid by mean ------------
    rng = np.random.default_rng(int(sample_seed))
    for group_key in target_groups_in_order:
        proj_mean_src = per_group_proj_mean.get(group_key)
        proj_indiv_src = per_group_proj_indiv.get(group_key)
        if (
            proj_mean_src is None
            and proj_indiv_src is None
        ):
            continue
        style = _style_for(group_key)
        color = style["color"]
        dash = style["dash"]
        marker_symbol_2d = style["marker"]
        line_mode = style["mode"]
        marker_symbol_3d = (
            "circle" if group_key.endswith("_GT_1") else "diamond"
        )
        band_label = _legend_label(group_key, gt_0_name, gt_1_name)
        show_legend = band_label not in legend_seen
        if show_legend:
            legend_seen.add(band_label)

        # Mean trace comes from ``groups_mean_eff``; individuals subsample
        # comes from ``groups_indiv_eff``. The two may be the same data
        # (legacy) or different folders (dual-source override).
        if proj_mean_src is not None and proj_mean_src.shape[0] > 0:
            mean = proj_mean_src.mean(axis=0)  # (n_layers_local, n_axes)
        else:
            mean = None

        if proj_indiv_src is not None and proj_indiv_src.shape[0] > 0:
            k_total = proj_indiv_src.shape[0]
            if int(max_target_per_class) < 0 or k_total <= int(max_target_per_class):
                sample_idx = np.arange(k_total, dtype=np.int64)
            else:
                sample_idx = np.sort(
                    rng.choice(
                        k_total, size=int(max_target_per_class), replace=False
                    )
                )
        else:
            sample_idx = np.empty((0,), dtype=np.int64)
        faint_rgba = _hex_to_rgba(color, float(individual_line_alpha))

        if n_axes == 1:
            for k in sample_idx:
                fig.add_trace(
                    go.Scatter(
                        x=layer_indices_indiv, y=proj_indiv_src[k, :, 0],
                        mode="lines",
                        line=dict(color=faint_rgba, width=float(line_width_individual)),
                        legendgroup=band_label,
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=1, col=2,
                )
            if mean is None:
                continue
            fig.add_trace(
                go.Scatter(
                    x=layer_indices_mean, y=mean[:, 0],
                    mode=line_mode,
                    line=dict(color=color, dash=dash, width=float(line_width_mean)),
                    marker=dict(
                        symbol=marker_symbol_2d,
                        size=(float(marker_size_mean) if marker_size_mean is not None else 6),
                        color=color,
                    ),
                    name=band_label, legendgroup=band_label,
                    showlegend=show_legend,
                    hovertemplate=(
                        f"{band_label} | mean | L%{{x}} | "
                        f"{axis_names[0]}=%{{y:.3f}}<extra></extra>"
                    ),
                ),
                row=1, col=2,
            )
            if show_arrows and n_layers_local_mean >= 2:
                _add_segment_arrows_2d(
                    fig,
                    xs=layer_indices_mean.astype(np.float64),
                    ys=mean[:, 0].astype(np.float64),
                    color=color,
                    subplot_col=2,
                )

        elif n_axes == 2:
            for k in sample_idx:
                fig.add_trace(
                    go.Scatter(
                        x=proj_indiv_src[k, :, 0], y=proj_indiv_src[k, :, 1],
                        mode="lines",
                        line=dict(color=faint_rgba, width=float(line_width_individual)),
                        legendgroup=band_label,
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=1, col=2,
                )
            if mean is None:
                continue
            gradient_start = _mix_with_white(color, float(marker_gradient_start))
            colorscale = [[0.0, gradient_start], [1.0, color]]
            marker_color_kw = (
                dict(color=layer_fracs_mean, colorscale=colorscale, cmin=0.0, cmax=1.0)
                if marker_layer_gradient
                else dict(color=color)
            )
            fig.add_trace(
                go.Scatter(
                    x=mean[:, 0], y=mean[:, 1],
                    mode="lines+markers",
                    line=dict(color=color, dash=dash, width=float(line_width_mean)),
                    marker=dict(
                        **marker_color_kw,
                        size=(float(marker_size_mean) if marker_size_mean is not None else 8),
                        symbol=marker_symbol_2d,
                        opacity=1.0,
                        line=dict(width=0.6, color="darkslategrey"),
                        showscale=False,
                    ),
                    name=band_label,
                    legendgroup=band_label,
                    showlegend=show_legend,
                ),
                row=1, col=2,
            )
            if show_arrows and n_layers_local_mean >= 2:
                _add_segment_arrows_2d(
                    fig,
                    xs=mean[:, 0].astype(np.float64),
                    ys=mean[:, 1].astype(np.float64),
                    color=color,
                    subplot_col=2,
                )

        else:  # 3D
            for k in sample_idx:
                fig.add_trace(
                    go.Scatter3d(
                        x=proj_indiv_src[k, :, 0],
                        y=proj_indiv_src[k, :, 1],
                        z=proj_indiv_src[k, :, 2],
                        mode="lines",
                        line=dict(color=faint_rgba, width=float(line_width_individual)),
                        legendgroup=band_label,
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=1, col=2,
                )
            if mean is None:
                continue
            gradient_start = _mix_with_white(color, float(marker_gradient_start))
            colorscale = [[0.0, gradient_start], [1.0, color]]
            marker_color_kw = (
                dict(color=layer_fracs_mean, colorscale=colorscale, cmin=0.0, cmax=1.0)
                if marker_layer_gradient
                else dict(color=color)
            )
            fig.add_trace(
                go.Scatter3d(
                    x=mean[:, 0], y=mean[:, 1], z=mean[:, 2],
                    mode="lines+markers",
                    line=dict(color=color, width=float(line_width_mean)),
                    marker=dict(
                        **marker_color_kw,
                        size=(float(marker_size_mean) if marker_size_mean is not None else 5),
                        symbol=marker_symbol_3d,
                        opacity=1.0,
                        showscale=False,
                    ),
                    name=band_label,
                    legendgroup=band_label,
                    showlegend=show_legend,
                ),
                row=1, col=2,
            )
            if show_arrows and n_layers_local_mean >= 2:
                _add_segment_arrows_3d(
                    fig,
                    pts=mean.astype(np.float64),
                    color=color,
                    legendgroup=band_label,
                    subplot_col=2,
                )

    # ---- Axis titles / layout ---------------------------------------------
    range_map = axis_ranges or {}

    def _range_for(name: str) -> Optional[List[float]]:
        rng = range_map.get(name)
        if rng is None:
            return None
        return [float(rng[0]), float(rng[1])]

    if n_axes == 1:
        y_range = _range_for(axis_names[0])
        for col in (1, 2):
            fig.update_xaxes(title_text="Layer", row=1, col=col)
            y_kwargs: Dict[str, object] = {
                "title_text": f"proj on {axis_names[0]}",
                "row": 1, "col": col,
            }
            if y_range is not None:
                y_kwargs["range"] = y_range
            fig.update_yaxes(**y_kwargs)
            fig.add_hline(
                y=0,
                line=dict(color="lightgray", width=1, dash="dot"),
                row=1, col=col,
            )
    elif n_axes == 2:
        x_range = _range_for(axis_names[0])
        y_range = _range_for(axis_names[1])
        for col in (1, 2):
            x_kwargs: Dict[str, object] = {
                "title_text": f"proj on {axis_names[0]}",
                "row": 1, "col": col,
            }
            if x_range is not None:
                x_kwargs["range"] = x_range
            y_kwargs2: Dict[str, object] = {
                "title_text": f"proj on {axis_names[1]}",
                "row": 1, "col": col,
            }
            if y_range is not None:
                y_kwargs2["range"] = y_range
            fig.update_xaxes(**x_kwargs)
            fig.update_yaxes(**y_kwargs2)
    else:  # 3D — plotly stores per-scene axis titles on layout.scene{n}
        show_bg = bool(scene_show_walls)
        # Per-axis wall config. ``showbackground=False`` hides the cube
        # back-walls entirely; gridlines remain on the floor only by way
        # of the axis lines themselves. When walls are shown and a
        # ``scene_wall_color`` is given, we set every white-able surface
        # on the 3D scene to a contrasting tone (wall background, axis
        # gridlines, zero-lines) so Kaleido's SVG export can't leave any
        # of them as flat white. The grid/zero colors are derived from
        # the wall color (slightly darker) so the gridlines stay visible
        # against the wall.
        axis_wall = dict(showbackground=show_bg)
        scene_bgcolor: Optional[str] = None
        if show_bg and scene_wall_color is not None:
            wall = str(scene_wall_color)
            axis_wall["backgroundcolor"] = wall
            axis_wall["gridcolor"] = _darken_color(wall, 0.85)
            axis_wall["zerolinecolor"] = _darken_color(wall, 0.7)
            scene_bgcolor = wall
        x_range_3d = _range_for(axis_names[0])
        y_range_3d = _range_for(axis_names[1])
        z_range_3d = _range_for(axis_names[2])
        x_axis_3d = dict(title=f"proj on {axis_names[0]}", **axis_wall)
        y_axis_3d = dict(title=f"proj on {axis_names[1]}", **axis_wall)
        z_axis_3d = dict(title=f"proj on {axis_names[2]}", **axis_wall)
        if x_range_3d is not None:
            x_axis_3d["range"] = x_range_3d
        if y_range_3d is not None:
            y_axis_3d["range"] = y_range_3d
        if z_range_3d is not None:
            z_axis_3d["range"] = z_range_3d
        scene_axes = dict(
            xaxis=x_axis_3d,
            yaxis=y_axis_3d,
            zaxis=z_axis_3d,
        )
        if scene_bgcolor is not None:
            scene_kwargs = dict(scene_axes, bgcolor=scene_bgcolor)
        else:
            scene_kwargs = scene_axes
        fig.update_layout(scene=scene_kwargs, scene2=scene_kwargs)

    n_targets = sum(
        1 for g, idx in targets
        if g in groups and 0 <= idx < groups[g].shape[0]
    )
    mean_label_eff = (
        str(target_position_label_mean)
        if target_position_label_mean is not None
        else str(target_position_label)
    )
    indiv_label_eff = (
        str(target_position_label_individual)
        if target_position_label_individual is not None
        else str(target_position_label)
    )
    if mean_label_eff == str(target_position_label) and indiv_label_eff == str(
        target_position_label
    ):
        pos_text = f"pos={target_position_label}"
    else:
        pos_text = (
            f"pos(left)={target_position_label} | "
            f"pos(mean)={mean_label_eff} | pos(indiv)={indiv_label_eff}"
        )
    if (abs_lo_mean, abs_hi_mean) == (abs_lo, abs_hi) and (
        abs_lo_indiv, abs_hi_indiv
    ) == (abs_lo, abs_hi):
        layer_text = f"L{abs_lo}-L{abs_hi}"
    else:
        layer_text = (
            f"L(left)={abs_lo}-{abs_hi} | "
            f"L(mean)={abs_lo_mean}-{abs_hi_mean} | "
            f"L(indiv)={abs_lo_indiv}-{abs_hi_indiv}"
        )
    figure_title = (
        f"{title} | {pos_text} | "
        f"{layer_text} | "
        f"axes={list(axis_names)} | n_targets={n_targets}"
    )
    fig.update_layout(
        title=dict(text=figure_title, x=0.5, xanchor="center"),
        height=520 if not is_3d else 620,
        width=1100,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=11),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.12 if not is_3d else -0.05,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(t=80, b=120, l=80, r=40),
    )

    fmts = list(formats) if formats else ["png"]
    if is_3d and "html" not in [f.lower() for f in fmts]:
        fmts.append("html")
    _write_figure(fig, basepath, tuple(fmts))


# --------------------------------------------------------------------------- #
# Matplotlib backend for plot_trajectory_axis_grid                            #
# --------------------------------------------------------------------------- #

# Plotly → matplotlib translation tables for line dash and marker symbol.
_MPL_DASH: Dict[str, str] = {
    "solid":   "-",
    "dash":    "--",
    "dot":     ":",
    "dashdot": "-.",
}

_MPL_MARKER: Dict[str, str] = {
    "circle":  "o",
    "x":       "x",
    "diamond": "D",
}


def _write_mpl_figure(
    fig,
    basepath: Path,
    formats: Sequence[str],
    dpi: int = 200,
) -> None:
    """
    Save a matplotlib ``Figure`` to ``{basepath}.<fmt>`` for each ``fmt``.

    Drops the ``html`` format silently — matplotlib has no static HTML
    export and the trajectory-axis plot uses a static layout anyway.
    Aliases ``jpg`` → ``jpeg`` and de-duplicates repeats.
    """
    basepath.parent.mkdir(parents=True, exist_ok=True)
    seen: set = set()
    for fmt in formats:
        f = str(fmt).lower().lstrip(".")
        if f == "jpg":
            f = "jpeg"
        if f == "html":
            continue
        if f in seen:
            continue
        seen.add(f)
        out = f"{basepath}.{f}"
        if f in ("png", "jpeg", "webp"):
            fig.savefig(out, dpi=dpi, bbox_inches="tight")
        else:
            fig.savefig(out, bbox_inches="tight")


def _add_segment_arrows_mpl_2d(
    ax,
    xs: np.ndarray,
    ys: np.ndarray,
    color: str,
) -> None:
    """
    Overlay one ``ax.annotate`` arrow per (layer L → L+1) segment on a
    matplotlib 2D Axes. Tail = ``(xs[i], ys[i])``; head = ``(xs[i+1],
    ys[i+1])``. Mirrors the Plotly ``add_annotation`` arrow path used by
    :func:`_add_segment_arrows_2d`.
    """
    n = int(len(xs))
    for i in range(n - 1):
        ax.annotate(
            "",
            xy=(float(xs[i + 1]), float(ys[i + 1])),
            xytext=(float(xs[i]), float(ys[i])),
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                linewidth=1.5,
                shrinkA=4,
                shrinkB=4,
            ),
        )


def _add_segment_arrows_mpl_3d(
    ax,
    pts: np.ndarray,
    color: str,
) -> None:
    """
    Overlay a single ``Axes3D.quiver`` call whose shafts run from each
    layer's data point to the next, rendering an arrowhead at the tip.
    Mirrors the Plotly ``go.Cone`` arrowheads used by
    :func:`_add_segment_arrows_3d` but, because matplotlib's 3D quiver
    accepts data-coord shafts, the tail of each arrow is drawn too.
    """
    if pts.shape[0] < 2:
        return
    diffs = pts[1:] - pts[:-1]
    starts = pts[:-1]
    ax.quiver(
        starts[:, 0], starts[:, 1], starts[:, 2],
        diffs[:, 0], diffs[:, 1], diffs[:, 2],
        color=color,
        arrow_length_ratio=0.2,
        linewidth=1.2,
        length=1.0,
        normalize=False,
    )


def _plot_trajectory_axis_grid_matplotlib(
    *,
    groups: Dict[str, np.ndarray],
    axis_dirs: Dict[str, np.ndarray],
    axis_names: Sequence[str],
    targets: Sequence[Tuple[str, int]],
    target_position_index: int,
    tv_position_index: int,
    target_position_label: str,
    layer_lo: int,
    layer_hi: int,
    layer_lo_mean: Optional[int] = None,
    layer_hi_mean: Optional[int] = None,
    layer_lo_individual: Optional[int] = None,
    layer_hi_individual: Optional[int] = None,
    gt_0_name: str,
    gt_1_name: str,
    gt_style: str,
    layer_shade_min: float,
    max_target_per_class: int,
    line_width_individual: float,
    line_width_mean: float,
    individual_line_alpha: float,
    marker_layer_gradient: bool,
    marker_gradient_start: float,
    marker_size_target: Optional[float],
    marker_size_mean: Optional[float],
    sample_seed: int,
    show_arrows: bool,
    axis_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    groups_mean: Optional[Dict[str, np.ndarray]] = None,
    groups_individual: Optional[Dict[str, np.ndarray]] = None,
    target_position_index_mean: Optional[int] = None,
    target_position_index_individual: Optional[int] = None,
    target_position_label_mean: Optional[str] = None,
    target_position_label_individual: Optional[str] = None,
    basepath: Path,
    title: str,
    formats: Sequence[str],
) -> None:
    """
    Matplotlib backend for :func:`plot_trajectory_axis_grid`.

    Renders the same 2-column figure (left = per-target sample
    trajectories; right = bold per-class mean overlaid on a seeded
    subsample of individual trajectories) for 1D / 2D / 3D axis sets.
    The Plotly-only knobs (``scene_show_walls``, ``scene_wall_color``)
    are intentionally absent — the matplotlib 3D scene uses its default
    panes. The ``html`` output format is dropped at save time.
    """
    import matplotlib
    matplotlib.use("Agg", force=False)  # lazy, headless-safe
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D proj

    n_axes = len(axis_names)
    if n_axes not in (1, 2, 3):
        raise ValueError(
            f"axis_names must have 1, 2, or 3 entries; got {n_axes}: "
            f"{list(axis_names)}"
        )
    for name in axis_names:
        if name not in axis_dirs:
            raise ValueError(
                f"axis {name!r} missing from axis_dirs (got {sorted(axis_dirs)})"
            )

    any_arr = next(iter(groups.values()))
    n_layers_full = any_arr.shape[2]
    abs_lo, abs_hi = _resolve_layer_range(layer_lo, layer_hi, n_layers_full)
    n_layers_local = abs_hi - abs_lo + 1
    layer_indices = np.arange(abs_lo, abs_hi + 1, dtype=np.int32)

    def _resolve_or_default(
        lo_override: Optional[int], hi_override: Optional[int]
    ) -> Tuple[int, int]:
        lo_eff = int(layer_lo) if lo_override is None else int(lo_override)
        hi_eff = int(layer_hi) if hi_override is None else int(hi_override)
        return _resolve_layer_range(lo_eff, hi_eff, n_layers_full)

    abs_lo_mean, abs_hi_mean = _resolve_or_default(layer_lo_mean, layer_hi_mean)
    abs_lo_indiv, abs_hi_indiv = _resolve_or_default(
        layer_lo_individual, layer_hi_individual
    )
    n_layers_local_mean = abs_hi_mean - abs_lo_mean + 1
    n_layers_local_indiv = abs_hi_indiv - abs_lo_indiv + 1
    layer_indices_mean = np.arange(abs_lo_mean, abs_hi_mean + 1, dtype=np.int32)
    layer_indices_indiv = np.arange(abs_lo_indiv, abs_hi_indiv + 1, dtype=np.int32)

    style_map = _build_group_style(gt_style)

    # Right-column source fallbacks (parallel to the plotly path).
    groups_mean_eff = groups_mean if groups_mean is not None else groups
    groups_indiv_eff = (
        groups_individual if groups_individual is not None else groups
    )
    pos_index_mean = (
        int(target_position_index_mean)
        if target_position_index_mean is not None
        else int(target_position_index)
    )
    pos_index_indiv = (
        int(target_position_index_individual)
        if target_position_index_individual is not None
        else int(target_position_index)
    )

    seen_groups: set = set()
    target_groups_in_order: List[str] = []
    for group_key, _ in targets:
        if group_key in seen_groups:
            continue
        if (group_key not in groups_mean_eff) and (
            group_key not in groups_indiv_eff
        ):
            continue
        seen_groups.add(group_key)
        target_groups_in_order.append(group_key)

    def _build_per_group_proj(
        src: Dict[str, np.ndarray],
        pos_index: int,
        keys: set,
        win_lo: int,
        win_hi: int,
    ) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for group_key in keys:
            arr = src.get(group_key)
            if arr is None:
                continue
            proj = project_onto_axes(
                arr,
                axis_dirs,
                axis_names,
                tv_position_index=tv_position_index,
                proj_position_index=pos_index,
            )
            out[group_key] = proj[:, win_lo : win_hi + 1, :]
        return out

    left_keys = {g for g, _ in targets}
    right_keys = set(target_groups_in_order)
    per_group_proj = _build_per_group_proj(
        groups, int(target_position_index), left_keys, abs_lo, abs_hi
    )
    same_mean_source = (
        groups_mean_eff is groups
        and pos_index_mean == int(target_position_index)
        and abs_lo_mean == abs_lo
        and abs_hi_mean == abs_hi
    )
    if same_mean_source:
        per_group_proj_mean = per_group_proj
    else:
        per_group_proj_mean = _build_per_group_proj(
            groups_mean_eff, pos_index_mean, right_keys, abs_lo_mean, abs_hi_mean
        )
    if (
        groups_indiv_eff is groups
        and pos_index_indiv == int(target_position_index)
        and abs_lo_indiv == abs_lo
        and abs_hi_indiv == abs_hi
    ):
        per_group_proj_indiv = per_group_proj
    elif (
        groups_indiv_eff is groups_mean_eff
        and pos_index_indiv == pos_index_mean
        and abs_lo_indiv == abs_lo_mean
        and abs_hi_indiv == abs_hi_mean
    ):
        per_group_proj_indiv = per_group_proj_mean
    else:
        per_group_proj_indiv = _build_per_group_proj(
            groups_indiv_eff,
            pos_index_indiv,
            right_keys,
            abs_lo_indiv,
            abs_hi_indiv,
        )

    is_3d = (n_axes == 3)

    figsize = (11.0, 6.2 if is_3d else 5.2)
    fig = plt.figure(figsize=figsize, facecolor="white")
    if is_3d:
        ax_left = fig.add_subplot(1, 2, 1, projection="3d")
        ax_right = fig.add_subplot(1, 2, 2, projection="3d")
    else:
        ax_left = fig.add_subplot(1, 2, 1)
        ax_right = fig.add_subplot(1, 2, 2)
    # ax_left.set_title("per-target sample", fontsize=11)
    # ax_right.set_title("per-class mean", fontsize=11)

    def _make_layer_fracs(n: int) -> np.ndarray:
        return np.array(
            [_layer_color_fraction(L, n, layer_shade_min) for L in range(n)],
            dtype=np.float32,
        )

    layer_fracs = _make_layer_fracs(n_layers_local)
    if n_layers_local_mean == n_layers_local:
        layer_fracs_mean = layer_fracs
    else:
        layer_fracs_mean = _make_layer_fracs(n_layers_local_mean)

    legend_handles: Dict[str, Any] = {}

    def _style_for(group_key: str) -> Dict[str, str]:
        return style_map.get(
            group_key,
            {"color": "gray", "dash": "solid", "marker": "circle", "mode": "lines"},
        )

    # ---- Left column: per-target sample trajectories ---------------------
    for group_key, idx in targets:
        proj = per_group_proj.get(group_key)
        if proj is None or not 0 <= idx < proj.shape[0]:
            continue
        style = _style_for(group_key)
        color = style["color"]
        linestyle = _MPL_DASH.get(style["dash"], "-")
        marker_2d = _MPL_MARKER.get(style["marker"], "o")
        line_mode = style["mode"]
        marker_3d = "o" if group_key.endswith("_GT_1") else "D"
        base_label = _legend_label(group_key, gt_0_name, gt_1_name)
        trace_label = f"{base_label} [sample {idx}]"
        traj = proj[idx]  # (n_layers_local, n_axes)

        if n_axes == 1:
            include_markers = (line_mode == "lines+markers")
            (h,) = ax_left.plot(
                layer_indices,
                traj[:, 0],
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
                marker=(marker_2d if include_markers else None),
                markersize=(float(marker_size_target) if marker_size_target is not None else 6),
                label=trace_label,
            )
            legend_handles.setdefault(trace_label, h)
            if show_arrows and n_layers_local >= 2:
                _add_segment_arrows_mpl_2d(
                    ax_left,
                    xs=layer_indices.astype(np.float64),
                    ys=traj[:, 0].astype(np.float64),
                    color=color,
                )
        elif n_axes == 2:
            cmap_left = mcolors.LinearSegmentedColormap.from_list(
                "tax_left", ["white", color]
            )
            (h,) = ax_left.plot(
                traj[:, 0],
                traj[:, 1],
                color=color,
                linestyle=linestyle,
                linewidth=1.5,
                label=trace_label,
            )
            scatter_color_kw = (
                dict(c=layer_fracs, cmap=cmap_left, vmin=0.0, vmax=1.0)
                if marker_layer_gradient
                else dict(c=color)
            )
            ax_left.scatter(
                traj[:, 0],
                traj[:, 1],
                **scatter_color_kw,
                marker=marker_2d,
                s=(float(marker_size_target) ** 2 if marker_size_target is not None else 49),
                edgecolors="darkslategrey",
                linewidths=0.5,
                zorder=h.get_zorder() + 1,
            )
            legend_handles.setdefault(trace_label, h)
            if show_arrows and n_layers_local >= 2:
                _add_segment_arrows_mpl_2d(
                    ax_left,
                    xs=traj[:, 0].astype(np.float64),
                    ys=traj[:, 1].astype(np.float64),
                    color=color,
                )
        else:  # 3D
            cmap_left = mcolors.LinearSegmentedColormap.from_list(
                "tax_left", ["white", color]
            )
            (h,) = ax_left.plot(
                traj[:, 0],
                traj[:, 1],
                traj[:, 2],
                color=color,
                linewidth=2.0,
                label=trace_label,
            )
            scatter_color_kw = (
                dict(c=layer_fracs, cmap=cmap_left, vmin=0.0, vmax=1.0)
                if marker_layer_gradient
                else dict(c=color)
            )
            ax_left.scatter(
                traj[:, 0],
                traj[:, 1],
                traj[:, 2],
                **scatter_color_kw,
                marker=marker_3d,
                s=(float(marker_size_target) ** 2 if marker_size_target is not None else 20),
            )
            legend_handles.setdefault(trace_label, h)
            if show_arrows and n_layers_local >= 2:
                _add_segment_arrows_mpl_3d(
                    ax_left,
                    pts=traj.astype(np.float64),
                    color=color,
                )

    # ---- Right column: per-class individuals + bold mean -----------------
    rng = np.random.default_rng(int(sample_seed))
    for group_key in target_groups_in_order:
        proj_mean_src = per_group_proj_mean.get(group_key)
        proj_indiv_src = per_group_proj_indiv.get(group_key)
        if proj_mean_src is None and proj_indiv_src is None:
            continue
        style = _style_for(group_key)
        color = style["color"]
        linestyle = _MPL_DASH.get(style["dash"], "-")
        marker_2d = _MPL_MARKER.get(style["marker"], "o")
        line_mode = style["mode"]
        marker_3d = "o" if group_key.endswith("_GT_1") else "D"
        band_label = _legend_label(group_key, gt_0_name, gt_1_name)

        # Mean trace comes from ``groups_mean_eff``; individuals subsample
        # comes from ``groups_indiv_eff``.
        if proj_mean_src is not None and proj_mean_src.shape[0] > 0:
            mean = proj_mean_src.mean(axis=0)  # (n_layers_local, n_axes)
        else:
            mean = None

        if proj_indiv_src is not None and proj_indiv_src.shape[0] > 0:
            k_total = proj_indiv_src.shape[0]
            if int(max_target_per_class) < 0 or k_total <= int(max_target_per_class):
                sample_idx = np.arange(k_total, dtype=np.int64)
            else:
                sample_idx = np.sort(
                    rng.choice(
                        k_total, size=int(max_target_per_class), replace=False
                    )
                )
        else:
            sample_idx = np.empty((0,), dtype=np.int64)

        # Faint per-sample overlay traces.
        for k in sample_idx:
            if n_axes == 1:
                ax_right.plot(
                    layer_indices_indiv,
                    proj_indiv_src[k, :, 0],
                    color=color,
                    linewidth=float(line_width_individual),
                    alpha=float(individual_line_alpha),
                )
            elif n_axes == 2:
                ax_right.plot(
                    proj_indiv_src[k, :, 0],
                    proj_indiv_src[k, :, 1],
                    color=color,
                    linewidth=float(line_width_individual),
                    alpha=float(individual_line_alpha),
                )
            else:
                ax_right.plot(
                    proj_indiv_src[k, :, 0],
                    proj_indiv_src[k, :, 1],
                    proj_indiv_src[k, :, 2],
                    color=color,
                    linewidth=float(line_width_individual),
                    alpha=float(individual_line_alpha),
                )

        if mean is None:
            continue

        # Bold per-class mean. Marker gradient starts at a tinted version
        # of the class color (matches the Plotly white→color scheme
        # offset by ``marker_gradient_start``). _mix_with_white returns
        # a CSS ``rgb(R,G,B)`` literal which matplotlib's ``to_rgb``
        # can't parse — go through _parse_color_to_rgb01 instead.
        gradient_start_rgb = _parse_color_to_rgb01(
            _mix_with_white(color, float(marker_gradient_start))
        )
        cmap_mean = mcolors.LinearSegmentedColormap.from_list(
            "tax_mean", [gradient_start_rgb, _parse_color_to_rgb01(color)]
        )

        if n_axes == 1:
            include_markers = (line_mode == "lines+markers")
            (h_m,) = ax_right.plot(
                layer_indices_mean,
                mean[:, 0],
                color=color,
                linestyle=linestyle,
                linewidth=float(line_width_mean),
                marker=(marker_2d if include_markers else None),
                markersize=(float(marker_size_mean) if marker_size_mean is not None else 6),
                label=band_label,
            )
            legend_handles.setdefault(band_label, h_m)
            if show_arrows and n_layers_local_mean >= 2:
                _add_segment_arrows_mpl_2d(
                    ax_right,
                    xs=layer_indices_mean.astype(np.float64),
                    ys=mean[:, 0].astype(np.float64),
                    color=color,
                )
        elif n_axes == 2:
            (h_m,) = ax_right.plot(
                mean[:, 0],
                mean[:, 1],
                color=color,
                linestyle=linestyle,
                linewidth=float(line_width_mean),
                label=band_label,
            )
            scatter_color_kw = (
                dict(c=layer_fracs_mean, cmap=cmap_mean, vmin=0.0, vmax=1.0)
                if marker_layer_gradient
                else dict(c=color)
            )
            ax_right.scatter(
                mean[:, 0],
                mean[:, 1],
                **scatter_color_kw,
                marker=marker_2d,
                s=(float(marker_size_mean) ** 2 if marker_size_mean is not None else 64),
                edgecolors="darkslategrey",
                linewidths=0.6,
                zorder=h_m.get_zorder() + 1,
            )
            legend_handles.setdefault(band_label, h_m)
            if show_arrows and n_layers_local_mean >= 2:
                _add_segment_arrows_mpl_2d(
                    ax_right,
                    xs=mean[:, 0].astype(np.float64),
                    ys=mean[:, 1].astype(np.float64),
                    color=color,
                )
        else:  # 3D
            (h_m,) = ax_right.plot(
                mean[:, 0],
                mean[:, 1],
                mean[:, 2],
                color=color,
                linewidth=float(line_width_mean),
                label=band_label,
            )
            scatter_color_kw = (
                dict(c=layer_fracs_mean, cmap=cmap_mean, vmin=0.0, vmax=1.0)
                if marker_layer_gradient
                else dict(c=color)
            )
            ax_right.scatter(
                mean[:, 0],
                mean[:, 1],
                mean[:, 2],
                **scatter_color_kw,
                marker=marker_3d,
                s=(float(marker_size_mean) ** 2 if marker_size_mean is not None else 30),
            )
            legend_handles.setdefault(band_label, h_m)
            if show_arrows and n_layers_local_mean >= 2:
                _add_segment_arrows_mpl_3d(
                    ax_right,
                    pts=mean.astype(np.float64),
                    color=color,
                )

    # ---- Axis titles / decorations --------------------------------------
    range_map = axis_ranges or {}

    def _range_for(name: str) -> Optional[Tuple[float, float]]:
        rng = range_map.get(name)
        if rng is None:
            return None
        return (float(rng[0]), float(rng[1]))

    if n_axes == 1:
        y_rng = _range_for(axis_names[0])
        for ax in (ax_left, ax_right):
            ax.set_xlabel("Layer")
            ax.set_ylabel(f"proj on {axis_names[0]}")
            ax.axhline(0, color="lightgray", linewidth=1, linestyle=":")
            if y_rng is not None:
                ax.set_ylim(y_rng)
    elif n_axes == 2:
        x_rng = _range_for(axis_names[0])
        y_rng = _range_for(axis_names[1])
        for ax in (ax_left, ax_right):
            ax.set_xlabel(f"proj on {axis_names[0]}")
            ax.set_ylabel(f"proj on {axis_names[1]}")
            if x_rng is not None:
                ax.set_xlim(x_rng)
            if y_rng is not None:
                ax.set_ylim(y_rng)
    else:
        x_rng = _range_for(axis_names[0])
        y_rng = _range_for(axis_names[1])
        z_rng = _range_for(axis_names[2])
        for ax in (ax_left, ax_right):
            ax.set_xlabel(f"proj on {axis_names[0]}")
            ax.set_ylabel(f"proj on {axis_names[1]}")
            ax.set_zlabel(f"proj on {axis_names[2]}")
            if x_rng is not None:
                ax.set_xlim(x_rng)
            if y_rng is not None:
                ax.set_ylim(y_rng)
            if z_rng is not None:
                ax.set_zlim(z_rng)

    n_targets = sum(
        1 for g, idx in targets
        if g in groups and 0 <= idx < groups[g].shape[0]
    )
    mean_label_eff = (
        str(target_position_label_mean)
        if target_position_label_mean is not None
        else str(target_position_label)
    )
    indiv_label_eff = (
        str(target_position_label_individual)
        if target_position_label_individual is not None
        else str(target_position_label)
    )
    if mean_label_eff == str(target_position_label) and indiv_label_eff == str(
        target_position_label
    ):
        pos_text = f"pos={target_position_label}"
    else:
        pos_text = (
            f"pos(left)={target_position_label} | "
            f"pos(mean)={mean_label_eff} | pos(indiv)={indiv_label_eff}"
        )
    if (abs_lo_mean, abs_hi_mean) == (abs_lo, abs_hi) and (
        abs_lo_indiv, abs_hi_indiv
    ) == (abs_lo, abs_hi):
        layer_text = f"L{abs_lo}-L{abs_hi}"
    else:
        layer_text = (
            f"L(left)={abs_lo}-{abs_hi} | "
            f"L(mean)={abs_lo_mean}-{abs_hi_mean} | "
            f"L(indiv)={abs_lo_indiv}-{abs_hi_indiv}"
        )
    figure_title = (
        f"{title} | {pos_text} | "
        f"{layer_text} | "
        f"axes={list(axis_names)} | n_targets={n_targets}"
    )
    fig.suptitle(figure_title, fontsize=11)

    if legend_handles:
        ncol = min(len(legend_handles), 4)
        fig.legend(
            list(legend_handles.values()),
            list(legend_handles.keys()),
            loc="lower center",
            ncol=ncol,
            fontsize=9,
            bbox_to_anchor=(0.5, 0.0),
            frameon=False,
        )

    # Leave room for suptitle (top) and shared legend (bottom).
    fig.tight_layout(rect=[0.02, 0.10, 0.98, 0.94])

    _write_mpl_figure(fig, basepath, formats)
    plt.close(fig)
