"""Plot the attention head sweep and cumulative patching.

Example:
  python src/activation_patching/step4_plot_heads_cumulative.py
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DIRECTION = 'lying_to_honest'
METRIC_LABELS = {
    'rotation_normalized': 'Normalized rotation change',
    'logit_shift_normalized': 'Normalized logit shift',
}
# Short names for legend entries, e.g. "Rotation (top-k)" instead of the
# longer axis label "Normalized rotation change (top-k)". The axis label keeps
# "Normalized" (read once, in isolation); the legend drops it since it would
# otherwise repeat across all four (top-k)/(random) entries.
LEGEND_NAMES = {'rotation_normalized': 'Rotation', 'logit_shift_normalized': 'Logit shift'}
# Zero-centered metrics ("no effect" = 0): keep 0 comfortably inside the
# auto-scaled y-range.
SHIFT_METRICS = {'logit_shift_normalized', 'rotation_normalized'}


def load_rows(path):
    """Load per_prompt_k_metrics.csv rows for the lying_to_honest direction.

    step4_patch_heads_cumulative.py computes normalized_logit_shift/
    normalized_rotation_change directly (the paper's two metrics); this just
    reads them back under the plot's own column names.
    """
    with Path(path).open() as f:
        rows = [r for r in csv.DictReader(f) if r['direction'] == DIRECTION]
    for row in rows:
        row['k'] = int(row['k'])
        row['logit_shift_normalized'] = float(row['normalized_logit_shift'])
        row['rotation_normalized'] = float(row['normalized_rotation_change'])
    return rows


def axis_bounds(values):
    values = [v for arr in values for v in arr]
    return (min(-.03, min(values) - .02), max(1.03, max(values) + .04))


def series_by_k(by_prompt, n_prompts, left_key, right_key):
    per_k = defaultdict(list)
    for rows in by_prompt.values():
        for r in rows:
            per_k[r['k']].append(r)
    k_values = sorted(per_k)
    if k_values != list(range(1, len(k_values) + 1)):
        raise ValueError('Aggregate is missing a contiguous 1-based k sweep')
    for rows in per_k.values():
        if len(rows) != n_prompts:
            raise ValueError(f'Aggregate does not have all {n_prompts} prompts at every k')
    import numpy as np
    return {
        'k_values': k_values,
        'left_mean': np.array([np.mean([r[left_key] for r in per_k[k]]) for k in k_values]),
        'left_std': np.array([np.std([r[left_key] for r in per_k[k]]) for k in k_values]),
        'right_mean': np.array([np.mean([r[right_key] for r in per_k[k]]) for k in k_values]),
        'right_std': np.array([np.std([r[right_key] for r in per_k[k]]) for k in k_values]),
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--line-input', type=Path,
                   default=ROOT/'results/patching/mean-head-best-position-noise-ranked-k30-selfrot/per_prompt_k_metrics.csv',
                   help='per_prompt_k_metrics.csv for the lying-to-honest line panel (a noise-ranked run)')
    p.add_argument('--random-input', type=Path,
                   default=ROOT/'results/patching/mean-head-best-position-random-baseline-k30-selfrot/per_prompt_k_metrics.csv',
                   help='per_prompt_k_metrics.csv from a --selection random run, overlaid as a control')
    p.add_argument('--head-screen', type=Path,
                   default=ROOT/'results/patching/mean-head-best-position-20prompts-100-fast/results.json',
                   help='Schema-3 single-head screen results.json (source of the head_screen.png heatmap)')
    p.add_argument('--logit-metric', default='logit_shift_normalized')
    p.add_argument('--rotation-metric', default='rotation_normalized')
    p.add_argument('--output', type=Path, default=ROOT/'results/plots/mean_patching_head_k30')
    p.add_argument('--line-filename-stem', default='lying_to_honest_top_heads_selfrotation')
    p.add_argument('--heatmap-filename-stem', default='lying_to_honest_single_heads')
    p.add_argument('--line-title', default='Mean patching on top attention heads')
    p.add_argument('--heatmap-title', default='Mean patching on single attention heads')
    p.add_argument('--figsize', default='5.2,3.9', help='width,height in inches, shared by both figures')
    p.add_argument('--formats', default='png,pdf')
    args = p.parse_args(argv)
    args.formats = args.formats.split(',')
    return args


def load_line_panel_data(line_input, random_input, left_key, right_key):
    rows = load_rows(line_input)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[row['system_prompt_id']].append(row)

    random_by_prompt = None
    if random_input:
        random_rows = load_rows(random_input)
        random_by_prompt = defaultdict(list)
        for row in random_rows:
            random_by_prompt[row['system_prompt_id']].append(row)
        if sorted(random_by_prompt, key=int) != sorted(by_prompt, key=int):
            raise ValueError('Random-input run must cover the same prompts as --line-input')

    d = series_by_k(by_prompt, len(by_prompt), left_key, right_key)
    rd = series_by_k(random_by_prompt, len(random_by_prompt), left_key, right_key) if random_by_prompt else None
    if rd and rd['k_values'] != d['k_values']:
        raise ValueError('Ranked and random runs must sweep the same k values for an overlay')

    def bounds(key, metric_key):
        arrs = [d[f'{key}_mean'] - d[f'{key}_std'], d[f'{key}_mean'] + d[f'{key}_std']]
        if rd:
            arrs += [rd[f'{key}_mean'] - rd[f'{key}_std'], rd[f'{key}_mean'] + rd[f'{key}_std']]
        if metric_key in SHIFT_METRICS:
            import numpy as np
            arrs.append(np.array([0.0]))
        return axis_bounds(arrs)

    return d, rd, bounds('left', left_key), bounds('right', right_key)


def load_heatmap_matrix(head_screen_path):
    """Recomputes the lying-to-honest single-head normalized-logit-shift
    matrix from the head screen.

    Uses the 'noise' key (donor=lying/experiment, recipient=honest/control),
    matching the same donor/recipient pairing as the 'lying_to_honest' rows
    used by the line panel -- NOT step3_head_sweep.py's own head_screen
    plot's 'denoise' key, even though that plot titles this pairing
    "Lying -> honest". That plot titles panels by the recipient's resulting
    behavior, the opposite convention from 'lying_to_honest' as used
    everywhere else in this codebase (donor's condition -> recipient's
    original condition); see PATCHING_WALKTHROUGH.md's note on this exact
    ambiguity.

    Same normalization as logit_shift_normalized on the line panel: each
    variant's shift is flipped (positive = toward target) and divided by
    that variant's own unpatched baseline magnitude BEFORE averaging across
    the 20 variants, not the other way around -- otherwise a variant with a
    larger baseline would implicitly outweigh the others in raw logit units.
    """
    import numpy as np
    data = json.loads(head_screen_path.read_text())
    matrix = np.full((data['n_layers'], data['n_heads']), np.nan)
    for i, item in enumerate(data['interventions']):
        normalized = []
        for v in data['variants']:
            e = v['effects'][i]['noise']
            baseline = e['logit_diff'] - e['logit_shift']
            if abs(baseline) < 1e-6:
                raise ValueError('Heatmap normalization undefined: baseline logit_diff is ~0 for '
                                  f"variant {v['system_prompt_id']}, intervention {i}")
            normalized.append(-e['logit_shift'] / abs(baseline))
        matrix[item['layer'], item['head']] = float(np.mean(normalized))
    return matrix, data


def main(argv=None):
    args = parse_args(argv)
    figsize = tuple(float(x) for x in args.figsize.split(','))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.activation_patching.utils.plotting import (
        draw_cumulative_heads_line_panel, draw_head_single_heatmap_panel, save_titled_figure)
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 13,
                         'xtick.labelsize': 11, 'ytick.labelsize': 11,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    args.output.mkdir(parents=True, exist_ok=True)

    d, rd, left_ylim, right_ylim = load_line_panel_data(
        args.line_input, args.random_input, args.logit_metric, args.rotation_metric)
    matrix, screen_data = load_heatmap_matrix(args.head_screen)

    fig1, ax_line = plt.subplots(figsize=figsize, constrained_layout=True)
    right = ax_line.twinx()
    draw_cumulative_heads_line_panel(ax_line, right, d, rd, METRIC_LABELS[args.logit_metric], METRIC_LABELS[args.rotation_metric],
                     LEGEND_NAMES[args.logit_metric], LEGEND_NAMES[args.rotation_metric],
                     left_ylim, right_ylim, legend_fontsize=10)
    save_titled_figure(fig1, args.line_title, args.output, args.line_filename_stem, args.formats, figsize)
    plt.close(fig1)

    fig2, ax_heat = plt.subplots(figsize=figsize, constrained_layout=True)
    draw_head_single_heatmap_panel(ax_heat, fig2, matrix, screen_data['n_layers'], screen_data['n_heads'])
    save_titled_figure(fig2, args.heatmap_title, args.output, args.heatmap_filename_stem, args.formats, figsize)
    plt.close(fig2)


if __name__ == '__main__':
    main()
