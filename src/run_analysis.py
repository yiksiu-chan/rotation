"""
CLI shim for the **analysis** stage.

The implementation lives in :mod:`src.analysis.run_analysis`; this file exists
so that ``python src/run_analysis.py`` works directly from a checkout (it
prepends the repo root to ``sys.path`` before importing the package).

Usage
-----
::

    python src/run_analysis.py [--config PATH] [--config_override PATH]

Summary
------------------------------
For each cached ``(SysPrompt, seed)`` directory with all four activation
pickles, do:

- Load the four pickles (2 conditions × 2 GT labels) as NumPy arrays of shape
  ``(k_per_class, n_positions, n_layers, d_model)`` (``k_per_class`` is the
  achieved per-class count for the configured ``cache_sample_method``).
- For each ``(layer, position)``, compute the cosine between
  ``mean(GT_1) − mean(GT_0)`` under control and the same under experiment.
  Returns ``cos_sim`` of shape ``(n_layers, n_positions)``.
- Apply the rotation-metric plugin to the per-layer curve at each position;
  rank positions by ``rotation_r`` descending — the top entry is also
  surfaced as ``best_token_position_name`` for backward compatibility.
- Persist under ``stage_stats/``:
  - ``rotation_stats.json``::

        {
          "<position_label>": {
            "rotation_r": float,
            "cos_sim_across_layers": [float, ...]
          },
          ...,
          "ranked_token_positions": ["<best>", "<2nd>", ...],   // sorted by rotation_r desc
          "best_token_position_name": "<one of the position labels>"   // == ranked_token_positions[0]
        }

  - ``cos_sim_heatmaps_across_tokens.<fmt>``
    y = token positions,
    x = layers
    (formats controlled by ``cfg.cos_sim_heatmap_formats``; default
    ``["svg", "png"]``)

- Persist under ``PCA_plots/``:

  - ``pca_<position_label>.{pdf,svg}`` — per-layer PCA scatter grid for that
    token position; the four ``(condition, GT)`` groups are distinguished by
    marker color and symbol, with one legend entry per group (drawn on the
    first layer subplot only). Legend labels remap GT_0 / GT_1 to the
    task-specific names from ``cfg.gt_0_label_map`` / ``cfg.gt_1_label_map``
    — e.g. ``control / False``, ``control / True`` for lying/sycophancy or
    ``experiment / Left``, ``experiment / Right`` for perspective.
    Title is
    ``"{task}/{dataset} | SysPrompt{id:03d} | seed={n} | token={label} | sampled_size_per_GT_class={k} | method={cache_sample_method}"``
    — ``k`` reflects ``sampled_size_GT_0`` / ``sampled_size_GT_1`` from the
    matching ``completions_*.json`` (``"{n}"`` when both halves match;
    ``"GT0={n0}, GT1={n1}"`` for the ``"maximum"`` method when they differ).
  - **Position selection** — two complementary controls:

    1. ``cfg.pca_token_positions`` (or the CLI flag
       ``--pca_token_positions 'prompt,prompt-1'``) — explicit list of token
       labels to plot. When non-empty, **this overrides everything else**;
       unknown labels are skipped with a warning.
    2. ``cfg.save_top_n_tokens_4_PCAplots`` — fallback. When the explicit
       list is empty, the top ``N`` positions by ``rotation_r`` (i.e. the
       prefix of ``ranked_token_positions``) get a PCA plot. Default ``N=1``
       → only the best position.

  - Aesthetics (``color``, ``symbol``, ``marker_alpha``, ``marker_size``,
    ``n_components``) come from ``cfg.pca_plot_style``.

The variant loop honors ``cfg.start_variant`` / ``cfg.end_variant`` (or the
matching CLI flags ``--start_variant`` / ``--end_variant``).

Rotation metric
---------------
Pluggable via :mod:`src.analysis.rotation_plugins`. ``DEFAULT_ROTATION_METRIC``
is currently :func:`~src.analysis.rotation_plugins.late_layer_anti_alignment`
(``1 − mean(cos_sim across the trailing half of layers)``; range ``[0, 2]``;
larger = more rotation). Override per call by passing
``rotation_metric=<callable>`` to
:func:`src.analysis.run_analysis.run_analysis_for_config`, or globally by
reassigning ``DEFAULT_ROTATION_METRIC``. Any callable matching the
``RotationMetricFn`` protocol (per-layer cosine sequence → scalar) is
accepted.

Note
---------------
``(SysPrompt, seed)`` pairs with any missing activation pickle are skipped
with a printed warning so partially-cached sweeps can be analyzed
incrementally.


"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analysis.run_analysis import main

if __name__ == "__main__":
    main()
