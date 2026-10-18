"""Shared figure-drawing functions for the activation-patching pipeline.

Every ``stepN_*.py`` script calls into this module at the end of its own
``main()`` to produce its figures directly -- there are no separate
``plot_*``/``summarize_*`` scripts to run afterward, with one exception:
``step4_plot_heads_cumulative.py`` stays a separate script because its
figure combines two independently-completed runs (a ranked selection and its
random-baseline control), which no single step's ``main()`` has both of.

This module only draws figures (matplotlib calls); it does not load files or
compute metrics -- callers pass in already-loaded rows/arrays.
"""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BLUE, RED = '#2563eb', '#b91c1c'
BLUE_LIGHT, RED_LIGHT = '#93c5fd', '#fca5a5'


def fitting_title_fontsize(title, max_fontsize=16, min_fontsize=10, chars_per_inch_at_10pt=10.5, fig_width=5.0):
    """Largest fontsize (down to min_fontsize) that keeps `title` on one line
    for a figure of width `fig_width` inches, by a simple character-budget
    estimate (matplotlib has no layout-free way to measure text extent)."""
    fontsize = max_fontsize
    while fontsize > min_fontsize:
        budget = chars_per_inch_at_10pt * fig_width * (10 / fontsize)
        if len(title) <= budget:
            return fontsize
        fontsize -= 0.5
    return min_fontsize


def save_titled_figure(fig, title, output, filename_stem, formats, figsize):
    fontsize = fitting_title_fontsize(title, fig_width=figsize[0])
    fig.suptitle(title, fontsize=fontsize)
    for fmt in formats:
        fig.savefig(output/f'{filename_stem}.{fmt}', dpi=300)
    print(f'Saved {output}/{filename_stem}.{{{",".join(formats)}}} (title fontsize {fontsize})')


def plot_position_heatmap(result, output_dir):
    """Step 1: per-prompt token rotation scores, highlighting each row's
    maximum (its frozen "best position"). `result` is step1's own in-memory
    results dict (same schema as its results.json)."""
    variants = result['variants']
    labels = result['position_labels']
    scores = np.array([v['rotation_by_position'] for v in variants])
    best = scores.argmax(axis=1)
    fig, ax = plt.subplots(figsize=(13, 10))
    fig.subplots_adjust(left=.17, right=.78, bottom=.27, top=.87)
    heatmap = ax.imshow(scores, cmap='viridis', vmin=0, vmax=2, aspect='auto')
    ax.set_xticks(range(len(labels)), labels, rotation=55, ha='right', fontsize=9)
    ax.set_yticks(range(len(variants)), [f"ID {v['system_prompt_id']:02d}   (n={v['n_rows_kept']})" for v in variants], fontsize=10)
    ax.set_xlabel('Token position relative to the end of the core prompt', labelpad=12)
    ax.set_ylabel('System prompt ID · retained examples', labelpad=12)
    ax.set_xticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(variants), 1), minor=True)
    ax.grid(which='minor', color='white', alpha=.13, linewidth=.6)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.scatter(best, np.arange(len(variants)), marker='*', s=125, color='white', edgecolor='black', linewidth=.65)
    ax.text(1.02, 1.015, 'Best position · score', transform=ax.transAxes, fontsize=10, fontweight='bold')
    for row, v in enumerate(variants):
        ax.text(len(labels)-.1, row, f"{v['best_position']:>6}  ·  {v['best_rotation']:.3f}",
                va='center', fontsize=10, family='monospace', clip_on=False)
    color_ax = fig.add_axes([.94, .24, .016, .54])
    fig.colorbar(heatmap, cax=color_ax, label='Rotation score (cosine drop)')
    fig.suptitle('Where does rotation peak for each system prompt?', x=.5, y=.97, fontsize=18, fontweight='bold')
    fig.text(.5, .925, f'{result["config"]["model_path"]} · {result["n_rows_total"]} shared examples before filtering · ★ maximum within each prompt', ha='center', fontsize=11)
    fig.text(.17, .045, 'Strict half-layer metric: max(0, max cosine in layers 0–15 − min cosine in layers 16–31).\n'
             'Retain examples answered correctly under the honest instruction and incorrectly under the lying instruction.\n'
             '“prompt” precedes the answer prefix; “gen+i” includes answer-prefix tokens.', fontsize=9, linespacing=1.6)
    paths = []
    for ext in ('png', 'pdf'):
        path = output_dir / f'best_token_position_heatmap.{ext}'
        fig.savefig(path, dpi=200, bbox_inches='tight')
        paths.append(path)
    plt.close(fig)
    return paths


def plot_diagnostic_heatmaps(rows, variant_ids, selection, x_values, x_key, x_label, output_dir, suptitle, xtick_step=1):
    """2-panel heatmap (normalized_logit_shift / normalized_rotation_change),
    rows=variants, columns=x_values (residual layers for step 2, cumulative-k
    for step 4). `rows` is the flat per-(variant, x) metrics list; `selection`
    maps variant_id -> its step-1 baseline entry (for the y-tick label).
    `xtick_step`: label every Nth column (step 2 uses 2, to match every-other
    -layer ticks; step 4 uses the default 1, labeling every k)."""
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 11), sharex=True, layout='constrained')
    for row, (metric, title) in enumerate((('normalized_logit_shift', 'normalized logit shift'),
                                            ('normalized_rotation_change', 'normalized rotation change'))):
        matrix = np.array([[next(r[metric] for r in rows if r['system_prompt_id'] == v and r[x_key] == x)
                            for x in x_values] for v in variant_ids])
        ax = axes[row]
        scale = np.abs(matrix).max() or 1.0
        im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r', vmin=-scale, vmax=scale)
        ax.set_title(f'Lying → honest · {title}')
        ax.set_yticks(range(len(variant_ids)), [f'{v} ({selection[v]["best_position"]})' for v in variant_ids], fontsize=8)
        ax.set_xticks(range(0, len(x_values), xtick_step), x_values[::xtick_step])
        ax.set_ylabel('System prompt ID (frozen token)')
        if row == 1:
            ax.set_xlabel(x_label)
        fig.colorbar(im, ax=ax, shrink=.8)
    fig.suptitle(suptitle, fontsize=14 if len(suptitle) > 60 else 15)
    paths = []
    for ext in ('png', 'pdf'):
        path = output_dir / f'per_prompt_heatmaps.{ext}'
        fig.savefig(path, dpi=180)
        paths.append(path)
    plt.close(fig)
    return paths


def plot_diagnostic_lines(aggregates, x_key, x_label, output_dir, suptitle, filename_stem='mean_effects'):
    """2-panel line plot (mean normalized_logit_shift / normalized_rotation_change
    across prompts) over x_values (residual layers for step 2, cumulative-k for
    step 4). `aggregates` is a list of per-x mean-metric dicts."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for ax, metric, label in zip(axes, ('normalized_logit_shift', 'normalized_rotation_change'),
                                  ('Normalized logit shift', 'Normalized rotation change')):
        ax.plot([r[x_key] for r in aggregates], [r[metric] for r in aggregates], marker='o', markersize=3, label='Lying → honest')
        ax.set(xlabel=x_label, ylabel=label)
        ax.axhline(0, color='gray', linewidth=.7)
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle(suptitle)
    paths = []
    for ext in ('png', 'pdf'):
        path = output_dir / f'{filename_stem}.{ext}'
        fig.savefig(path, dpi=180)
        paths.append(path)
    plt.close(fig)
    return paths


def plot_head_screen_heatmap(rows, data, output_dir):
    """Step 3: 2-panel (denoise / noise direction) single-head logit-shift
    heatmap over (layer, head), plus the status/coverage suptitle."""
    from matplotlib.colors import TwoSlopeNorm
    specs = [('denoise', 'Lying → honest'), ('noise', 'Honest → lying')]
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)
    scale = max(abs(r['target_logit_shift']) for r in rows) or 1
    for ax, (direction, title) in zip(axes, reversed(specs)):
        group = [r for r in rows if r['direction'] == direction]
        matrix = np.full((data['n_layers'], data['n_heads']), np.nan)
        for r in group:
            matrix[r['layer'], r['head']] = r['target_logit_shift']
        im = ax.imshow(matrix, origin='lower', aspect='auto', cmap='RdBu_r',
                        norm=TwoSlopeNorm(vmin=-scale, vcenter=0, vmax=scale))
        ax.set(xlabel='Head', ylabel='Layer', xticks=range(0, data['n_heads'], 4), yticks=range(0, data['n_layers'], 4))
        ax.set_title(title + '\nLogit shift toward target')
        fig.colorbar(im, ax=ax, shrink=.85)
    completed = len({r['layer'] for r in rows})
    variant_count = len(data['variants'])
    requested_count = len(data.get('requested_variants', data['variants']))
    status_label = ' · partial' if data.get('status') != 'complete' else ''
    fig.suptitle('Mean patching (attention heads)\n'
                 f'{variant_count}/{requested_count} prompts · {data["n_rows_total"]} examples per prompt{status_label}',
                 fontsize=13)
    paths = []
    for fmt in ('png', 'pdf'):
        path = output_dir / f'head_screen.{fmt}'
        fig.savefig(path, dpi=180)
        paths.append(path)
    plt.close(fig)
    return paths


def _paper_panel_bounds(mean, std):
    arr = np.concatenate([mean - std, mean + std, [0.0]])
    return (min(-.03, arr.min() - .02), max(1.03, arr.max() + .04))


def plot_residual_paper_panel(x_values, left_mean, left_std, right_mean, right_std,
                               output_dir, title, filename_stem='lying_to_honest_residual',
                               figsize=(5.0, 3.9), formats=('png', 'pdf')):
    """Step 2's clean paper-appendix figure: normalized logit shift (left
    axis) + normalized rotation change (right axis) vs. residual layer, no
    random control (step 2 exhaustively patches every layer, so there's no
    top-k selection to compare against a baseline)."""
    left_ylim, right_ylim = _paper_panel_bounds(left_mean, left_std), _paper_panel_bounds(right_mean, right_std)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    right = ax.twinx()
    ax.fill_between(x_values, left_mean - left_std, left_mean + left_std, color=BLUE, alpha=.22, linewidth=0)
    ax.plot(x_values, left_mean, color=BLUE, linestyle='-', marker='o', markersize=1.7, markerfacecolor=BLUE, linewidth=1.8, label='Logit shift')
    right.fill_between(x_values, right_mean - right_std, right_mean + right_std, color=RED, alpha=.22, linewidth=0)
    right.plot(x_values, right_mean, color=RED, linestyle='-', marker='o', markersize=1.6, markerfacecolor=RED, linewidth=1.8, label='Rotation')
    ax.set(xlabel='Layer', ylabel='Normalized logit shift', xticks=list(range(0, len(x_values), 4)), ylim=left_ylim)
    ax.tick_params(axis='y', labelcolor=BLUE)
    ax.yaxis.label.set_color(BLUE)
    right.set(ylabel='Normalized rotation change', ylim=right_ylim)
    right.tick_params(axis='y', labelcolor=RED)
    right.yaxis.label.set_color(RED)
    ax.grid(axis='y', color='#d7dce2', linewidth=.8, alpha=.8)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    right.spines['top'].set_visible(False)
    left_handles, left_labels = ax.get_legend_handles_labels()
    right_handles, right_labels = right.get_legend_handles_labels()
    ax.legend(left_handles + right_handles, left_labels + right_labels, fontsize=10, loc='upper left', framealpha=.85)
    paths = []
    for fmt in formats:
        path = output_dir / f'{filename_stem}.{fmt}'
        fig.savefig(path, dpi=300)
        paths.append(path)
    plt.close(fig)
    print(f'Saved {output_dir}/{filename_stem}.{{{",".join(formats)}}}')
    return paths


def tick_subset(k_values, max_ticks=12):
    if len(k_values) <= max_ticks:
        return k_values
    step = -(-len(k_values) // max_ticks)  # ceil division
    ticks = list(k_values[::step])
    if ticks[-1] != k_values[-1]:
        ticks.append(k_values[-1])
    return ticks


def style_dual_axis_panel(ax, right, k_values, left_axis_label, right_axis_label, left_ylim, right_ylim):
    ax.set(xlabel='Cumulative top-k heads', ylabel=left_axis_label, xticks=tick_subset(k_values), ylim=left_ylim)
    ax.tick_params(axis='y', labelcolor=BLUE)
    ax.yaxis.label.set_color(BLUE)
    right.set(ylabel=right_axis_label, ylim=right_ylim)
    right.tick_params(axis='y', labelcolor=RED)
    right.yaxis.label.set_color(RED)
    ax.grid(axis='y', color='#d7dce2', linewidth=.8, alpha=.8)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    right.spines['top'].set_visible(False)


def draw_cumulative_heads_line_panel(ax, right, d, rd, left_axis_label, right_axis_label,
                                      left_legend_name, right_legend_name, left_ylim, right_ylim, legend_fontsize=6.5):
    """Draw the dual-axis cumulative-top-k (+ optional random control `rd`)
    lines onto an existing (ax, right) pair, including the legend. `d`/`rd`
    are dicts with k_values/left_mean/left_std/right_mean/right_std (see
    step4_plot_heads_cumulative.py's series_by_k). Axis labels (e.g.
    "Normalized logit shift") and legend names (e.g. "Logit shift") are
    passed separately since the paper panel uses the longer form for axes
    and the shorter form for the repeated per-series legend entries."""
    if rd:
        ax.fill_between(rd['k_values'], rd['left_mean'] - rd['left_std'], rd['left_mean'] + rd['left_std'],
                        color=BLUE_LIGHT, alpha=.25, linewidth=0)
        ax.plot(rd['k_values'], rd['left_mean'], color=BLUE_LIGHT, linestyle='--',
               marker='o', markersize=1.4, markerfacecolor=BLUE_LIGHT, linewidth=1.4,
               label=f'{left_legend_name} (random)')
        right.fill_between(rd['k_values'], rd['right_mean'] - rd['right_std'], rd['right_mean'] + rd['right_std'],
                           color=RED_LIGHT, alpha=.25, linewidth=0)
        right.plot(rd['k_values'], rd['right_mean'], color=RED_LIGHT, linestyle='--',
                  marker='o', markersize=1.3, markerfacecolor=RED_LIGHT, linewidth=1.4,
                  label=f'{right_legend_name} (random)')
    ax.fill_between(d['k_values'], d['left_mean'] - d['left_std'], d['left_mean'] + d['left_std'],
                    color=BLUE, alpha=.22, linewidth=0)
    ax.plot(d['k_values'], d['left_mean'], color=BLUE, linestyle='-',
           marker='o', markersize=1.7, markerfacecolor=BLUE, linewidth=1.8,
           label=f'{left_legend_name} (top-k)')
    right.fill_between(d['k_values'], d['right_mean'] - d['right_std'], d['right_mean'] + d['right_std'],
                       color=RED, alpha=.22, linewidth=0)
    right.plot(d['k_values'], d['right_mean'], color=RED, linestyle='-',
              marker='o', markersize=1.6, markerfacecolor=RED, linewidth=1.8,
              label=f'{right_legend_name} (top-k)')
    style_dual_axis_panel(ax, right, d['k_values'], left_axis_label, right_axis_label, left_ylim, right_ylim)
    # Plot order is (random, top-k) per axis; reverse each pair so the
    # legend reads top-k above random, grouped by metric.
    left_handles, left_labels = ax.get_legend_handles_labels()
    right_handles, right_labels = right.get_legend_handles_labels()
    handles = left_handles[::-1] + right_handles[::-1]
    labels = left_labels[::-1] + right_labels[::-1]
    if handles:
        ax.legend(handles, labels, fontsize=legend_fontsize, loc='upper left', framealpha=.85)


def draw_head_single_heatmap_panel(ax, fig, matrix, n_layers, n_heads):
    from matplotlib.colors import TwoSlopeNorm
    scale = np.nanmax(np.abs(matrix)) or 1.0
    im = ax.imshow(matrix, origin='lower', aspect='auto', cmap='RdBu_r',
                    norm=TwoSlopeNorm(vmin=-scale, vcenter=0, vmax=scale))
    ax.set(xlabel='Head', ylabel='Layer', xticks=range(0, n_heads, 4), yticks=range(0, n_layers, 4))
    fig.colorbar(im, ax=ax, shrink=.85, label='Normalized logit shift')
