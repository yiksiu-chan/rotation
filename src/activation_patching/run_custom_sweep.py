"""General-purpose activation-patching sweep runner (not a numbered pipeline
step): residual/attention/MLP/head windows, position sweeps, or custom
cumulative sweeps, plus a complete rebuttal recipe or shard-merging mode. For
the paper's actual step-by-step pipeline (find best position -> patch
residual -> screen heads -> patch cumulative heads), see
src/activation_patching/step{1,2,3,4}_*.py instead.

Examples:
  python src/activation_patching/run_custom_sweep.py --component residual --layers 8-20 --positions=-12:-1 --sweep position --output results/patching/residual
  python src/activation_patching/run_custom_sweep.py --recipe lying-rebuttal --output results/patching/rebuttal
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from pathlib import Path
import sys

# Direct script execution from any working directory.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return value


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, default=ROOT/'config/patching_lying.yaml', help='YAML override layered over config/default.yaml')
    p.add_argument('--set', action='append', default=[], metavar='FIELD=VALUE', help='Repeatable flat/dotted pipeline config override')
    p.add_argument('--model-path')
    p.add_argument('--task')
    p.add_argument('--dataset')
    p.add_argument('--component', choices=['residual', 'attention', 'mlp', 'head'], default='residual')
    p.add_argument('--layers', default='all', help='Zero-based: 8, 8,12,16, 8-20, or all; one layer at a time')
    p.add_argument('--positions', default=None, help='Negative offsets; use --positions=-12:-1 for an inclusive range (default: -1, or -12:-1 for recipes)')
    p.add_argument('--heads', default='all', help='Selected head IDs/range, or all')
    p.add_argument('--sweep', choices=['window', 'position', 'head', 'head-position', 'cumulative'], default='window', help='window: selected positions/heads together at each layer; other modes vary one site at a time')
    p.add_argument('--direction', choices=['denoise', 'noise', 'both'], default='both')
    p.add_argument('--patch-source', choices=['donor', 'mean'], default='donor',
                   help='donor patches the corresponding example; mean broadcasts the reference-condition mean')
    p.add_argument('--mean-reference', choices=['evaluation', 'dataset'], default='evaluation',
                   help='rows used to compute mean activations (evaluation sample or full loaded dataset)')
    p.add_argument('--metrics', choices=['behavior', 'both'], default='behavior', help='both adds patched generation/rotation measurement')
    p.add_argument('--rotation-position', default='best',
                   help='Position at which rotation is measured: best (historical default), prompt-1, prompt, gen+1, etc. Independent of --positions, which selects patch sites.')
    p.add_argument('--variants', default=None, help='Comma list/range; default 0, or the 20 rebuttal variants for recipes')
    p.add_argument('--n-rows', type=positive_int, default=None)
    p.add_argument('--batch-size', type=positive_int, default=8)
    p.add_argument('--seed', type=int, default=0, help='Historical row sampling seed; also seeds model randomness')
    p.add_argument('--filter', choices=['none', 'logitdiff'], default='logitdiff')
    p.add_argument('--rank-from', type=Path, help='Head sweep result, required for cumulative runs')
    p.add_argument('--rank-metric', choices=['recovery', 'flip_rate', 'logit_shift'], default='recovery')
    p.add_argument('--rank-direction', choices=['denoise', 'noise', 'bidirectional'], default='denoise')
    p.add_argument('--top-k', default='1-10', help='Inclusive list/range of cumulative head counts')
    p.add_argument('--head-control', choices=['ranked', 'random'], default='ranked',
                   help='Cumulative head source; random uses seeded controls from --rank-from')
    p.add_argument('--random-repeats', type=positive_int, default=1)
    p.add_argument('--random-seed', type=int, default=0)
    p.add_argument('--random-match', choices=['layer', 'global'], default='layer')
    p.add_argument('--residual-from', type=Path, help='Residual-position sweep; with --head-from selects the historical rotation interventions')
    p.add_argument('--head-from', type=Path)
    p.add_argument('--residual-mode', choices=['best', 'all_layers', 'all_layer_positions', 'none'], default='best')
    p.add_argument('--include-final-layer', action='store_true', help='Include final residual layer in ranked rotation interventions')
    p.add_argument('--recipe', choices=['lying-rebuttal', 'lying-rotation-grid'], help='Run residual sweep → head sweep → cumulative heads → joint rotation with one loaded model')
    p.add_argument('--merge', nargs='+', type=Path, help='Merge disjoint variant shards; no model is loaded')
    p.add_argument('--output', type=Path, required=True, help='Run directory; paths are relative to your working directory')
    p.add_argument('--overwrite', action='store_true', help='Allow replacing results/config at this output path')
    p.add_argument('--dry-run', action='store_true', help='Validate arguments/config and print the plan without loading model/data')
    args = p.parse_args(argv)
    from src.activation_patching.utils.core import parse_indices
    try:
        args.positions = parse_indices(args.positions or ('-12:-1' if args.recipe else '-1'), negative=True)
        args.variants = parse_indices(args.variants or ('45,26,42,14,33,76,47,22,43,23,4,19,27,17,32,36,64,48,69,66' if args.recipe else '0'))
        args.top_k = parse_indices(args.top_k)
        if min(args.top_k) < 1:
            raise ValueError('--top-k must be positive')
        args.n_rows = args.n_rows or (100 if args.recipe else 48)
        for selection in (args.layers, args.heads):
            if selection != 'all':
                parse_indices(selection)
        if not args.merge:
            if args.sweep in ('head', 'head-position', 'cumulative') and args.component != 'head':
                raise ValueError('Head sweeps require --component head')
            if args.sweep == 'cumulative' and args.rank_from is None:
                raise ValueError('Cumulative patching requires --rank-from')
            if args.head_control == 'random' and args.sweep != 'cumulative':
                raise ValueError('--head-control random requires --sweep cumulative')
            if bool(args.residual_from) != bool(args.head_from):
                raise ValueError('Use --residual-from and --head-from together')
            if args.residual_from and args.metrics != 'both':
                raise ValueError('Ranked rotation interventions require --metrics both')
            if args.recipe and (args.direction != 'both' or args.rank_from or args.residual_from or args.sweep != 'window' or args.component != 'residual'):
                raise ValueError('Recipes define components/sweeps/ranking and require --direction both; use individual runs for custom selections')
    except ValueError as exc:
        p.error(str(exc))
    return args


def resolved_config(args):
    from src.config.cli import parse_set_overrides
    from src.config.loader import load_yaml_config, default_config_path
    overrides = parse_set_overrides(args.set)
    overrides.update({k: v for k, v in {'model_path': args.model_path, 'task': args.task, 'dataset': args.dataset}.items() if v is not None})
    cfg = load_yaml_config(default_config_path(), override_path=args.config, override_dict=overrides)
    if args.recipe and cfg.task != 'lying':
        raise ValueError('The lying recipes require task=lying')
    return cfg


def validate_source(source, cfg):
    for key in ('task', 'dataset', 'model_alias'):
        if source.get(key) != getattr(cfg, key):
            raise ValueError(f'Ranking source {key}={source.get(key)!r} differs from current configuration')
    if source.get('model_path') is not None and source['model_path'] != cfg.model_path:
        raise ValueError('Ranking source model_path differs from the requested model')
    if source.get('status') not in (None, 'complete'):
        raise ValueError('Ranking source is incomplete; finish or merge its sweep first')


def execute(args, cfg, *, model=None, tokenizer=None, records=None):
    """Injectable model/data for download-free regression tests."""
    import torch
    from src.activation_patching.utils.core import PatchingEngine, balanced_sample, parse_indices, build_interventions
    from src.activation_patching.utils.results import (load_results, cumulative_interventions,
                                                random_head_interventions, select_rotation_interventions,
                                                save_config, write_json)
    from src.task_corpus.dispatch import load_statements_for_config as _load_statements
    from src.model.inference import load_model_and_tokenizer
    torch.manual_seed(args.seed)
    all_records = _load_statements(cfg) if records is None else records
    rows = balanced_sample(all_records, args.n_rows, args.seed)
    if model is None:
        model, tokenizer = load_model_and_tokenizer(cfg.model_path, dtype=cfg.model_dtype)
    engine = PatchingEngine(model, tokenizer)
    layers = parse_indices(args.layers, len(engine.layers))
    heads = parse_indices(args.heads, engine.n_heads)
    save_config(args.output, cfg, args)
    write_json(args.output/'rows.json', [asdict(r) for r in rows])

    def run(items, name=None, rotation=False):
        output = args.output/name if name else args.output
        settings = {'row_seed': args.seed, 'metrics': 'both' if rotation else 'behavior',
                    'rotation_position': args.rotation_position if rotation else None,
                    'direction': args.direction, 'model_dtype': cfg.model_dtype,
                    'patch_source': args.patch_source,
                    'head_control': args.head_control,
                    'random_repeats': args.random_repeats if args.head_control == 'random' else None,
                    'random_seed': args.random_seed if args.head_control == 'random' else None,
                    'random_match': args.random_match if args.head_control == 'random' else None,
                    'mean_reference': args.mean_reference if args.patch_source == 'mean' else None,
                    'config': asdict(cfg)}
        def checkpoint(result):
            result.update(settings=settings, requested_variants=args.variants,
                          status='complete' if len(result['variants']) == len(args.variants) else 'partial')
            write_json(output/'results.json', result)
        data = engine.run(cfg, rows, args.variants, items, batch_size=args.batch_size,
                          direction=args.direction, row_filter=args.filter, rotation=rotation,
                          rotation_position=args.rotation_position,
                          patch_source=args.patch_source,
                          mean_rows=(all_records if args.mean_reference == 'dataset' else rows),
                          on_variant=checkpoint)
        print(f'Saved {output / "results.json"}', flush=True)
        return data

    if args.recipe:
        # Historical workflow: map residual layer/token sites and individual
        # heads first, then choose interventions and measure behavior + rotation.
        # The final rotation stage uses direction-specific flip-rate rankings;
        # the separate 'cumulative' stage below uses the CLI ranking options.
        residual = run(build_interventions('residual', layers, args.positions, heads, 'position'), 'residual')
        head = run(build_interventions('head', layers, args.positions, heads, 'head'), 'heads')
        run(cumulative_interventions(head, args.top_k, args.positions, metric=args.rank_metric,
                                     direction=args.rank_direction), 'cumulative')
        grid = args.recipe == 'lying-rotation-grid'
        # The compact rebuttal uses top-2/top-10; a restricted --top-k also works for smoke runs.
        k_values = args.top_k if grid else [k for k in (2, 10) if k <= max(args.top_k)]
        if not k_values:
            k_values = [min(args.top_k)]
        items = select_rotation_interventions(residual, head, 'all_layers' if grid else 'best',
                                             k_values, args.positions,
                                             include_final=grid or args.include_final_layer)
        run(items, 'rotation', rotation=True)
    elif args.residual_from:
        residual, head = load_results(args.residual_from), load_results(args.head_from)
        for source in (residual, head):
            validate_source(source, cfg)
        items = select_rotation_interventions(residual, head, args.residual_mode, args.top_k,
                                             args.positions, include_final=args.include_final_layer,
                                             layers=layers, heads=heads)
        run(items, rotation=True)
    elif args.sweep == 'cumulative':
        # General ranked/legacy-random CLI. For the current one-draw uniform
        # baseline, use src/activation_patching/step4_patch_heads_cumulative.py --selection random instead.
        source = load_results(args.rank_from)
        validate_source(source, cfg)
        if args.head_control == 'random':
            items = random_head_interventions(source, args.top_k, args.positions,
                                               repeats=args.random_repeats, seed=args.random_seed,
                                               match=args.random_match, metric=args.rank_metric,
                                               direction=args.rank_direction, layers=layers, heads=heads)
        else:
            items = cumulative_interventions(source, args.top_k, args.positions, metric=args.rank_metric,
                                             direction=args.rank_direction, layers=layers, heads=heads)
        run(items, rotation=args.metrics == 'both')
    else:
        run(build_interventions(args.component, layers, args.positions, heads, args.sweep), rotation=args.metrics == 'both')


def main(argv=None):
    args = parse_args(argv)
    from src.activation_patching.utils.results import write_json, merge_results
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite and not args.dry_run:
        raise ValueError(f'Output directory is not empty: {args.output}. Choose a new directory or --overwrite.')
    if args.merge:
        data = merge_results(args.merge)
        if args.dry_run:
            print(f'Would merge {len(data["variants"])} variants into {args.output}')
        else:
            write_json(args.output/'results.json', data)
        return
    cfg = resolved_config(args)
    if args.dry_run:
        import json
        print(json.dumps({'arguments': vars(args), 'config': asdict(cfg),
                          'note': 'Model dimensions, memory and dataset access are checked on a real run.'}, default=str, indent=2))
        return
    execute(args, cfg)


if __name__ == '__main__':
    main()
