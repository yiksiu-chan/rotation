"""
CLI shim for the **paper plots** stage.

The implementation lives in :mod:`src.paper_plots.generate_paper_plots`; this
file exists so that ``python src/generate_paper_plots.py`` works directly from
a checkout (it prepends the repo root to ``sys.path`` before importing the
package).

Usage
-----
::

    python src/generate_paper_plots.py [--config PATH] [--config_override PATH]

Summary
------------------------------
Aggregates the per-``(SysPrompt, seed)`` artifacts produced by the analysis
stage (``rotation_stats.json``) and the behavior stage
(``behavior_results_*.json``) into paper figures under
``{plot_root}/{model_alias}/``.

Per seed in ``cfg.seed_list_cache``:

- ``cos_sim_heatmaps/cos_heatmap_{task}_{dataset}_seed{n}.{pdf,svg}`` —
  heatmap with one row per system prompt (``y``), columns = layers (``x``),
  color = cosine similarity. **Each row uses that variant's own**
  ``best_token_position_name`` (read from the variant's
  ``rotation_stats.json``); the position label is appended to the row title
  so the heatmap stays interpretable when variants disagree.
- ``best_token_position/best_token_{task}_{dataset}_seed{n}.{pdf,svg}`` —
  bar chart over the canonical token positions
  (``position_labels(n_pre, n_gen)``); height = number of variants whose best
  token landed on that position. Positions with zero counts still appear so
  the x-axis is stable across runs.
- ``rotation_distribution/rotation_dist_{task}_{dataset}_seed{n}.{pdf,svg}``
  — box+points distribution of ``rotation_r`` across system prompts (taken
  at each variant's own best position).
- ``cos_sim_curves/cos_sim_curves_{task}_{dataset}_seed{n}.{pdf,svg}`` —
  mean ± variance-band of ``cos_sim_across_layers`` across system prompts;
  each variant contributes its curve at its own best token position. Band
  kind is set by ``cfg.cos_sim_curves_band_kind`` (``std`` / ``sem`` /
  ``ci95``).

Across seeds (single, non-seed-keyed file):

- ``behavior/behavior_BarBoxplot_{task}_{dataset}.{pdf,svg}`` — bar at the
  control accuracy plus a box of ``Accuracy_experiment`` across variants. The
  filename has no seed suffix because behavior accuracies don't depend on the
  cache seed.

Tolerance for partial sweeps
----------------------------
Variants whose ``rotation_stats.json`` is absent are silently skipped from the
per-seed figures; if a whole seed has nothing to plot, that seed is skipped
with a printed message but the orchestrator continues to the next. The
behavior figure tolerates a missing control file or an empty experiment list
on either axis independently.

All figures use Plotly and are exported as both ``.pdf`` and ``.svg``
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.paper_plots.generate_paper_plots import main

if __name__ == "__main__":
    main()
