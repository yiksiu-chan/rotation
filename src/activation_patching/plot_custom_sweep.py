"""Plot and summarize run_custom_sweep.py results (new or historical) with
one command. The paper's own step1-4 pipeline plots itself automatically at
the end of each step's main() (see src/activation_patching/utils/plotting.py
for the shared drawing functions); step4_plot_heads_cumulative.py is the one
exception, since it combines two separate runs (ranked heads + random
control) into one figure.

Examples:
  python src/activation_patching/plot_custom_sweep.py --inputs results/patching/run/results.json --output figures
  python src/activation_patching/plot_custom_sweep.py --inputs results/patching/rebuttal --metric all --output figures
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

METRIC_LABELS = {'recovery': 'Normalized effect (recipient=0, donor=1)',
                 'flip_rate': 'Post-patch answer sign rate', 'logit_diff': 'Honesty logit difference',
                 'logit_shift': 'Signed change from recipient baseline',
                 'patched_rotation': 'Patched rotation', 'rotation_delta': 'Baseline − patched rotation'}
DIRECTION_LABELS = {'denoise': 'Control → experiment (denoise)', 'noise': 'Experiment → control (noise)'}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--inputs', nargs='+', type=Path, required=True, help='JSON files or directories (including recipe directories)')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--metric', choices=['all', *METRIC_LABELS], default='recovery')
    p.add_argument('--direction', choices=['both', 'denoise', 'noise'], default='both')
    p.add_argument('--variants', help='Variant IDs/list/range; default aggregates all available variants equally')
    p.add_argument('--aggregate', choices=['mean', 'per-variant'], default='mean')
    p.add_argument('--kind', choices=['auto', 'curve', 'heatmap', 'rotation-behavior'], default='auto')
    p.add_argument('--layer', type=int, help='Slice a layer before plotting')
    p.add_argument('--head', type=int, help='Slice a head before plotting')
    p.add_argument('--position', type=int, help='Slice one negative prompt-end offset')
    p.add_argument('--rotation-position', help='Measure rotation at this cached token label, e.g. prompt-1; requires per-position scores or an already matching measurement')
    p.add_argument('--best-position', action='store_true',
                   help='Keep the highest-valued position independently for each layer (one layer curve)')
    p.add_argument('--reduce', choices=['mean', 'max'], default='max', help='Explicit reduction over hidden positions in a layer×head×position heatmap')
    p.add_argument('--formats', default='png,pdf', help='Comma list from png,pdf,svg')
    p.add_argument('--vmin', type=float)
    p.add_argument('--vmax', type=float)
    p.add_argument('--top-n', type=int, default=0, help='Annotate the highest-valued cells in heatmaps')
    p.add_argument('--tables-only', action='store_true', help='Write CSV/Markdown without importing matplotlib')
    args = p.parse_args(argv)
    if args.top_n < 0:
        p.error('--top-n must be nonnegative')
    args.formats = args.formats.split(',')
    if not args.formats or any(f not in ('png', 'pdf', 'svg') for f in args.formats):
        p.error('--formats accepts png,pdf,svg')
    return args


def discover_inputs(paths):
    found = []
    for path in paths:
        if path.is_file():
            found.append(path)
        elif path.is_dir():
            for candidate in sorted(path.rglob('*.json')):
                if candidate.name.endswith('.config.json') or candidate.name in ('config.json', 'rows.json'):
                    continue
                data = json.loads(candidate.read_text())
                if isinstance(data, dict) and 'variants' in data and ('interventions' in data or 'component' in data):
                    # Skip duplicate shard sets if a merged results file exists at the run root.
                    if 'shards' in candidate.relative_to(path).parts and (path/'results.json').exists():
                        continue
                    found.append(candidate)
        else:
            raise FileNotFoundError(path)
    found = list(dict.fromkeys(p.resolve() for p in found))
    if not found:
        raise ValueError('No patching result files found')
    return found


def table_rows(data, args):
    import numpy as np
    from src.activation_patching.utils.core import parse_indices
    from src.activation_patching.utils.results import summary_rows
    variants = parse_indices(args.variants) if args.variants else None
    directions = ['denoise', 'noise'] if args.direction == 'both' else [args.direction]
    subsets = [variants] if args.aggregate == 'mean' else [[v['system_prompt_id']] for v in data['variants'] if variants is None or v['system_prompt_id'] in variants]
    rows = []
    for subset in subsets:
        for row in summary_rows(data, directions, subset):
            if any(getattr(args, key) is not None and row[key] != getattr(args, key) for key in ('layer', 'head', 'position')):
                continue
            if args.metric != 'all' and row['metric'] != args.metric and args.kind != 'rotation-behavior':
                continue
            row['variant'] = subset[0] if args.aggregate == 'per-variant' else 'mean'
            rows.append(row)
    if not rows:
        raise ValueError('No results match the requested metric/slice. Old head-position files, for example, do not contain flip rates.')
    if args.best_position:
        if args.position is not None:
            raise ValueError('--best-position cannot be combined with --position')
        grouped = {}
        for row in rows:
            if row['layer'] is None or row['position'] is None:
                raise ValueError('--best-position requires results with explicit layer and position coordinates')
            # A dual-axis figure must use one shared position per layer. Select
            # it by rotation change when available, then retain both metrics.
            if args.kind == 'rotation-behavior':
                key = (row['component'], row['direction'], row['variant'], row['layer'])
            else:
                key = (row['component'], row['direction'], row['variant'], row['metric'], row['layer'])
            grouped.setdefault(key, []).append(row)
        selected = []
        for group in grouped.values():
            if args.kind == 'rotation-behavior':
                candidates = [r for r in group if r['metric'] == 'rotation_delta' and r['mean'] is not None] or group
            else:
                candidates = group
            best = max(candidates, key=lambda row: (float('-inf') if row['mean'] is None else row['mean'],
                                                     -(row['position'] if row['position'] is not None else 0)))
            selected.extend(r for r in group if r['position'] == best['position'])
        rows = selected
        rows.sort(key=lambda row: (row['component'], row['direction'], row['variant'], row['metric'], row['layer']))
    # Random-head controls contain repeated draws for each cumulative k. Treat
    # draws as the replicate dimension so curves have one point per k.
    if data.get('settings', {}).get('head_control') == 'random':
        grouped = {}
        for row in rows:
            if row['k'] is not None:
                grouped.setdefault((row['component'], row['direction'], row['variant'], row['metric'], row['k']), []).append(row)
        if any(len(group) > 1 for group in grouped.values()):
            collapsed = []
            for group in grouped.values():
                values = [r['mean'] for r in group if r['mean'] is not None]
                row = dict(group[0])
                row['mean'] = float(np.mean(values)) if values else None
                row['std'] = float(np.std(values)) if len(values) > 1 else 0.0
                row['n_valid'] = len(values)
                row['n_variants'] = len(group)
                collapsed.append(row)
            rows = sorted(collapsed, key=lambda row: (row['component'], row['direction'], row['variant'], row['metric'], row['k']))
    return rows


def write_tables(rows, output):
    output.mkdir(parents=True, exist_ok=True)
    with (output/'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({**r, 'sites': json.dumps(r['sites'], sort_keys=True)} for r in rows)
    lines = ['# Activation patching summary', '',
             'Mean and population standard deviation across variants. Missing/nonfinite values are excluded; counts are explicit.', '',
             '| Component | Direction | Layer | Head | Position | k | Metric | Mean | SD | Valid variants |',
             '|---|---|---:|---:|---:|---:|---|---:|---:|---:|']
    for r in rows:
        fmt = lambda x: '—' if x is None else f'{x:.6g}' if isinstance(x, float) else str(x)
        cells = [r[k] for k in ('component', 'direction', 'layer', 'head', 'position', 'k', 'metric', 'mean', 'std')]
        lines.append('| ' + ' | '.join(map(fmt, cells)) + f' | {r["n_valid"]}/{r["n_variants"]} |')
    (output/'summary.md').write_text('\n'.join(lines) + '\n')


def _save(fig, output, stem, formats):
    import matplotlib.pyplot as plt
    for fmt in formats:
        fig.savefig(output/f'{stem}.{fmt}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def _coordinates(rows):
    # Top-k is the independent variable even when k=1 has layer/head coordinates.
    if all(r['k'] is not None for r in rows):
        return 'k', None
    if all(r['head'] is not None for r in rows):
        if len({r['layer'] for r in rows}) == 1 and len({r['position'] for r in rows}) > 1 and all(r['position'] is not None for r in rows):
            return 'position', 'head'
        return 'head', 'layer'
    if all(r['position'] is not None for r in rows) and len({r['position'] for r in rows}) > 1:
        return 'position', 'layer'
    return 'layer', None


def plot_group(rows, args, output, title, stem):
    import numpy as np
    import matplotlib.pyplot as plt
    xkey, ykey = ('layer', None) if args.best_position else _coordinates(rows)
    varying_y = ykey and len({r[ykey] for r in rows}) > 1
    heatmap = args.kind == 'heatmap' or (args.kind == 'auto' and varying_y)
    metric = rows[0]['metric']
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    if heatmap:
        ykey = ykey or ('layer' if xkey != 'layer' else 'component')
        xs = sorted({r[xkey] for r in rows})
        ys = sorted({r[ykey] for r in rows})
        matrix = np.full((len(ys), len(xs)), np.nan)
        reduced = False
        for yi, y in enumerate(ys):
            for xi, x in enumerate(xs):
                values = [r['mean'] for r in rows if r[xkey] == x and r[ykey] == y and r['mean'] is not None]
                if values:
                    matrix[yi, xi] = np.max(values) if args.reduce == 'max' else np.mean(values)
                reduced |= len(values) > 1
        im = ax.imshow(matrix, aspect='auto', origin='lower', cmap='RdBu_r', vmin=args.vmin, vmax=args.vmax)
        ax.set_xticks(range(len(xs)), labels=xs)
        ax.set_yticks(range(len(ys)), labels=ys)
        ax.set_xlabel(xkey)
        ax.set_ylabel(ykey)
        fig.colorbar(im, ax=ax, label=METRIC_LABELS[metric])
        if reduced:
            title += f'\n{args.reduce} over remaining sites, after averaging variants'
        for flat in np.argsort(np.nan_to_num(matrix, nan=-np.inf), axis=None)[::-1][:args.top_n]:
            yi, xi = np.unravel_index(flat, matrix.shape)
            if np.isfinite(matrix[yi, xi]):
                ax.text(xi, yi, f'{matrix[yi, xi]:.2f}', ha='center', va='center', fontsize=8)
    else:
        # Retain separate layer curves when plotting a two-dimensional sweep.
        series_key = ykey if ykey is not None else None
        series = sorted({r[series_key] for r in rows}) if series_key else [None]
        for series_value in series:
            selected = [r for r in rows if series_key is None or r[series_key] == series_value]
            xs = sorted({r[xkey] for r in selected})
            means, deviations = [], []
            for x in xs:
                group = [r for r in selected if r[xkey] == x]
                if len(group) > 1:
                    raise ValueError('Curve has multiple sites per x value; choose a --position/--head slice or heatmap')
                means.append(group[0]['mean'] if group[0]['mean'] is not None else np.nan)
                deviations.append(group[0]['std'] if group[0]['std'] is not None else 0)
            ax.plot(xs, means, marker='o', label=f'{series_key} {series_value}' if series_key else None)
            ax.fill_between(xs, np.array(means)-deviations, np.array(means)+deviations, alpha=.15)
            if args.best_position:
                selected = {r['layer']: r['position'] for r in selected}
                for x, mean in zip(xs, means):
                    if not np.isnan(mean):
                        ax.annotate(str(selected[x]), (x, mean), xytext=(0, 6),
                                    textcoords='offset points', ha='center', fontsize=7)
        if series_key:
            ax.legend()
        ax.set_xlabel(xkey)
        ax.set_ylabel(METRIC_LABELS[metric])
        if args.vmin is not None or args.vmax is not None:
            ax.set_ylim(args.vmin, args.vmax)
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, output, stem, args.formats)


def plot_rotation_behavior(rows, args, output, title, stem):
    import matplotlib.pyplot as plt
    rotation_metric = 'rotation_delta' if any(r['metric'] == 'rotation_delta' for r in rows) else 'patched_rotation'
    by_metric = {metric: [r for r in rows if r['metric'] == metric]
                 for metric in ('flip_rate', rotation_metric)}
    if not all(by_metric.values()):
        raise ValueError('rotation-behavior requires flip_rate and rotation_delta (or patched_rotation)')
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    other = ax.twinx()
    ids = sorted(set(r['intervention'] for r in rows))
    for metric, axis, color in [('flip_rate', ax, 'tab:blue'), (rotation_metric, other, 'tab:red')]:
        values = {r['intervention']: r['mean'] for r in by_metric[metric]}
        axis.plot(range(len(ids)), [values.get(i) for i in ids], marker='o', color=color,
                  linewidth=2, markersize=4)
        axis.set_ylabel('Change in rotation' if metric == 'rotation_delta' else METRIC_LABELS[metric], color=color)
        axis.tick_params(axis='y', colors=color)
    coords = {r['intervention']: r for r in rows}
    labels = []
    for i in ids:
        row = coords[i]
        if row['k'] is not None:
            labels.append(f'k={row["k"]}')
        elif args.best_position and row['position'] is not None:
            labels.append(f'L{row["layer"]}\n p{row["position"]}')
        else:
            labels.append(f'L{row["layer"]}')
    ax.set_xticks(range(len(ids)), labels=labels, rotation=45)
    ax.grid(axis='y', color='lightgray', linewidth=.6)
    ax.set_axisbelow(True)
    ax.set_title(title)
    fig.tight_layout(pad=.4)
    _save(fig, output, stem, args.formats)


def main(argv=None):
    args = parse_args(argv)
    from src.activation_patching.utils.results import load_results, at_rotation_position
    paths = discover_inputs(args.inputs)
    for index, path in enumerate(paths):
        data = load_results(path)
        if args.rotation_position:
            data = at_rotation_position(data, args.rotation_position)
        if data.get('status') == 'partial':
            print(f'Warning: plotting completed variants from partial run {path}', file=sys.stderr)
        rows = table_rows(data, args)
        output = args.output if len(paths) == 1 else args.output/f'{index:02d}_{path.parent.name}_{path.stem}'
        write_tables(rows, output)
        if not args.tables_only:
            import matplotlib
            matplotlib.use('Agg')
            groups = {}
            for row in rows:
                key = (row['component'], row['direction'], row['variant'],
                       'joint' if args.kind == 'rotation-behavior' else row['metric'])
                groups.setdefault(key, []).append(row)
            for (component, direction, variant, metric), group in groups.items():
                title = f'{component} · {DIRECTION_LABELS[direction]} · variant {variant}'
                stem = f'{component}_{direction}_{variant}_{metric}'
                if args.kind == 'rotation-behavior':
                    plot_rotation_behavior(group, args, output, title, stem)
                else:
                    plot_group(group, args, output, title, stem)
        print(f'Saved tables' + (' and figures' if not args.tables_only else '') + f' to {output}')


if __name__ == '__main__':
    main()
