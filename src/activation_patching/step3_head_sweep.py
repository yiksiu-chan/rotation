"""Step 3: activation patching on each attention head.
"""
from dataclasses import asdict
from pathlib import Path
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.activation_patching.run_custom_sweep import parse_args, resolved_config
from src.activation_patching.utils.core import PatchingEngine, balanced_sample, build_interventions, parse_indices
from src.activation_patching.utils.fast_heads import FastHeadEngine
from src.activation_patching.utils.results import save_config, write_csv, write_json
from src.task_corpus.dispatch import load_statements_for_config as _load_statements
from src.model.inference import load_model_and_tokenizer


def summarize_and_plot(data, output):
    """Tabulate (CSV) and plot a completed (or partial) head-screen result.

    `data` is the same dict main() just wrote to results.json.
    """
    import numpy as np
    from src.activation_patching.utils.plotting import plot_head_screen_heatmap

    specs = [('denoise', 'Lying → honest', 'Fraction favoring correct answer'),
             ('noise', 'Honest → lying', 'Fraction favoring incorrect answer')]
    rows = []
    for direction, title, label in specs:
        for i, item in enumerate(data['interventions']):
            values = [v['effects'][i][direction] for v in data['variants']]
            shift = float(np.mean([v['logit_shift'] for v in values]))
            rows.append(dict(direction=direction, layer=item['layer'], head=item['head'],
                             position=('per-prompt' if data.get('position_selection') else item['sites'][0]['positions'][0]),
                             target_logit_shift=shift if direction == 'denoise' else -shift,
                             target_answer_rate=float(np.mean([v['flip_rate'] for v in values])),
                             recovery=float(np.mean([v['recovery'] for v in values]))))
    write_csv(output/'all_heads.csv', rows)
    best = []
    for direction, _, _ in specs:
        group = sorted([r for r in rows if r['direction'] == direction], key=lambda r: r['target_logit_shift'], reverse=True)
        best.extend(dict(rank=i+1, **r) for i, r in enumerate(group[:10]))
    write_csv(output/'top_heads.csv', best)
    plot_head_screen_heatmap(rows, data, output)
    variants = [v['system_prompt_id'] for v in data['variants']]
    positions = sorted({r['position'] for r in rows})
    settings = data.get('settings', {})
    site_description = (f'the frozen per-prompt positions from `{data["position_selection"]["source_path"]}`'
                        if data.get('position_selection') else f'full-input offsets {positions}')
    if data.get('position_selection'):
        fields = ['system_prompt_id', 'position_label', 'input_offset', 'token_id', 'token_text']
        write_csv(output/'positions_by_prompt.csv',
            [{k: v[k] for k in fields} for v in (data['position_selection']['variants'][str(v)] for v in variants)])
    completed = len({r['layer'] for r in rows})
    lines = ['# Mean-head screen', '', f'Status: **{data.get("status")}**; {completed}/{data["n_layers"]} layers; {len(data["interventions"])} individual heads.',
           '', f'Each head is replaced independently at {site_description} with the opposite condition’s mean over {data.get("mean_reference_count")} reference examples, computed separately per variant. Results average {len(variants)} variants equally: {variants}. Evaluation uses {data["n_rows_total"]} examples per variant, seed {settings.get("row_seed", "unknown")}, row filter `{data.get("row_filter")}`. Positive target logit shift means the intervention moved toward the intended behavioral change. Target-answer rate is an absolute rate, not the fraction newly flipped. This screen does not include held-out confirmation or same-condition controls.',
           '', 'Missing layers in partial results are blank.', '']
    for direction, title, label in specs:
        lines += ['## '+title, '', '| Rank | Layer | Head | Target logit shift | Target-answer rate |', '|---:|---:|---:|---:|---:|']
        for r in best:
            if r['direction'] == direction:
                lines.append(f'| {r["rank"]} | {r["layer"]} | {r["head"]} | {r["target_logit_shift"]:.4f} | {r["target_answer_rate"]:.0%} |')
        lines.append('')
    (output/'summary.md').write_text('\n'.join(lines)+'\n')
    print(f'Saved {output}; {completed} layers')


def main():
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument('--head-batch-size', type=int, default=8)
    extra.add_argument('--benchmark', action='store_true')
    extra.add_argument('--positions-from', type=Path,
                       help='Frozen best_positions.json; resolve one patch token per prompt')
    extra.add_argument('--verify-top-heads', type=int, default=0,
                       help='After the sweep, verify the top N heads per direction with full forwards')
    opt, rest = extra.parse_known_args()
    args = parse_args(rest)
    if opt.positions_from and any(x == '--positions' or x.startswith('--positions=') for x in rest):
        raise ValueError('Use either --positions-from or --positions, not both')
    if (args.component, args.sweep, args.patch_source, args.mean_reference, args.metrics) != ('head', 'head', 'mean', 'evaluation', 'behavior'):
        raise ValueError('Requires head sweep, mean evaluation reference, behavior metrics')
    if args.recipe or args.merge or args.rank_from or args.residual_from:
        raise ValueError('Plain independent head sweep required')
    if opt.head_batch_size < 1: raise ValueError('head batch size must be positive')
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise ValueError('Output already exists; use a fresh directory')
    cfg = resolved_config(args)
    args.head_batch_size = opt.head_batch_size
    args.benchmark = opt.benchmark
    args.verify_top_heads = opt.verify_top_heads
    args.positions_from = opt.positions_from
    source = None
    if opt.positions_from:
        from src.activation_patching.utils.variant_positions import load_position_selection
        source = load_position_selection(opt.positions_from, args.variants)
        args.positions = None
    if args.dry_run:
        print(vars(args)); return
    import torch
    torch.manual_seed(args.seed)
    rows = balanced_sample(_load_statements(cfg), args.n_rows, args.seed)
    model, tokenizer = load_model_and_tokenizer(cfg.model_path, dtype=cfg.model_dtype)
    engine = FastHeadEngine(model, tokenizer)
    selection = None
    if source is not None:
        from src.activation_patching.utils.variant_positions import resolve_selections, items_at_position, combine_position_results
        selection = resolve_selections(cfg, rows, tokenizer, source, opt.positions_from)
        write_json(args.output/'position_selection.json', selection)
        positions = [selection['variants'][str(args.variants[0])]['input_offset']]
    else:
        positions = args.positions
    items = build_interventions('head', parse_indices(args.layers, len(engine.layers)), positions,
                               parse_indices(args.heads, engine.n_heads), 'head')
    save_config(args.output, cfg, args)
    write_json(args.output/'rows.json', [asdict(r) for r in rows])
    started = time.monotonic()
    def checkpoint(result):
        result.update(requested_variants=args.variants,
            status='complete' if len(result['variants']) == len(args.variants) else 'partial',
            elapsed_seconds=time.monotonic()-started,
            settings=dict(row_seed=args.seed, direction=args.direction, metrics='behavior',
                model_dtype=cfg.model_dtype, patch_source='mean', mean_reference='evaluation',
                implementation='cached-prefix-batched-heads', head_batch_size=opt.head_batch_size,
                baseline_execution='same cached suffix and batch shape as interventions',
                batch_size=args.batch_size, config=asdict(cfg)))
        write_json(args.output/'results.json', result)
    if selection is None:
        data = engine.run(cfg, rows, args.variants, items, batch_size=args.batch_size,
            head_batch_size=opt.head_batch_size, direction=args.direction, row_filter=args.filter,
            on_variant=checkpoint)
    else:
        chunks = []
        for variant in args.variants:
            site = selection['variants'][str(variant)]
            local_items = items_at_position(items, site['input_offset'])
            print(f'POSITION variant={variant} label={site["position_label"]} offset={site["input_offset"]} token={site["token_text"]!r}', flush=True)
            chunk = engine.run(cfg, rows, [variant], local_items, batch_size=args.batch_size,
                head_batch_size=opt.head_batch_size, direction=args.direction, row_filter=args.filter)
            chunk.update(status='complete', requested_variants=[variant], patch_position=site)
            write_json(args.output/'variants'/f'variant{variant:03d}'/'results.json', chunk)
            chunks.append(chunk)
            data = combine_position_results(chunks, selection)
            checkpoint(data)
    if opt.benchmark or opt.verify_top_heads:
        indices = list(range(len(items)))
        if opt.verify_top_heads and not opt.benchmark:
            selected = set()
            for direction in data['variants'][0]['effects'][0]:
                sign = 1 if direction == 'denoise' else -1
                selected.update(sorted(indices, key=lambda i: sign * sum(
                    v['effects'][i][direction]['logit_shift'] for v in data['variants']),
                    reverse=True)[:opt.verify_top_heads])
            indices = sorted(selected)
        selected_items = [items[i] for i in indices]
        # Compare identical examples/heads, using the original full-prompt path.
        old_start = time.monotonic()
        def reference_checkpoint(result):
            result.update(requested_variants=args.variants,
                status='complete' if len(result['variants']) == len(args.variants) else 'partial',
                settings=data['settings'] | {'implementation': 'original-full-forward',
                                            'baseline_execution': 'full-prompt forward'})
            write_json(args.output/'reference.json', result)
        reference_engine = PatchingEngine(model, tokenizer)
        if selection is None:
            old = reference_engine.run(cfg, rows, args.variants, selected_items,
                batch_size=args.batch_size, direction=args.direction, row_filter=args.filter,
                patch_source='mean', mean_rows=rows, on_variant=reference_checkpoint)
        else:
            chunks = []
            for variant in args.variants:
                local_items = items_at_position(selected_items, selection['variants'][str(variant)]['input_offset'])
                chunk = reference_engine.run(cfg, rows, [variant], local_items,
                    batch_size=args.batch_size, direction=args.direction, row_filter=args.filter,
                    patch_source='mean', mean_rows=rows)
                chunk.update(status='complete', requested_variants=[variant],
                             patch_position=selection['variants'][str(variant)])
                write_json(args.output/'variants'/f'variant{variant:03d}'/'reference.json', chunk)
                chunks.append(chunk)
                old = combine_position_results(chunks, selection)
                reference_checkpoint(old)
        errors = {k: [] for k in ('logit_diff', 'logit_shift', 'flip_rate', 'recovery')}
        for a,b in zip(old['variants'], data['variants']):
            for ae,be in zip(a['effects'],[b['effects'][i] for i in indices]):
                for d in ae:
                    for k in errors:
                        if ae[d][k] is not None and be[d][k] is not None:
                            errors[k].append(abs(ae[d][k]-be[d][k]))
        report = dict(fast_seconds=data['elapsed_seconds'], reference_seconds=time.monotonic()-old_start,
                      verified_head_count=len(indices),
                      max_absolute_error={k:max(v) for k,v in errors.items()},
                      mean_absolute_error={k:sum(v)/len(v) for k,v in errors.items()},
                      peak_gpu_gb=torch.cuda.max_memory_allocated()/1e9)
        write_json(args.output/'benchmark.json', report)
        print(json.dumps(report), flush=True)
    summarize_and_plot(data, args.output)


if __name__ == '__main__': main()
