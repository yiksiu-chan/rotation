"""Step 4: cumulative top-k mean head patching at each system prompt's best position.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


DIRECTION = 'lying_to_honest'


def summarize_and_plot(out):
    """Validate, tabulate (CSV) and plot a completed run directory.

    Re-reads the just-written per-k JSON cells from disk (rather than
    reusing in-memory state) so this also works standalone against an
    already-completed --output directory, and so a resumed run always
    reflects every variant/k on disk, not just the ones this particular
    invocation computed.
    """
    import numpy as np
    from src.activation_patching.utils.metrics import early_to_late_drop
    from src.activation_patching.utils.plotting import plot_diagnostic_heatmaps, plot_diagnostic_lines
    from src.activation_patching.utils.results import write_csv

    manifest = json.loads((out/'manifest.json').read_text())
    source = json.loads((Path(manifest['baseline_path'])/'results.json').read_text())
    selection = {v['system_prompt_id']: v for v in source['variants']}
    ranked = manifest['ranked_heads']
    max_k = max(manifest['top_k_values'])
    n_layers = json.loads((out/'model.json').read_text())['n_layers']
    rows, rechecks, source_differences = [], {}, {}
    for variant in manifest['variants']:
        folder = out/'variants'/f'variant{variant:03d}'
        baseline = json.loads((folder/'baseline.json').read_text())
        original = selection[variant]
        assert baseline['kept_row_indices'] == original['kept_row_indices']
        assert baseline['position_label'] == original['best_position']
        assert baseline['n_rows_kept'] == original['n_rows_kept']
        assert len(baseline['baseline_cosine_by_layer']) == n_layers
        np.testing.assert_allclose(early_to_late_drop(baseline['baseline_cosine_by_layer']), baseline['baseline_rotation'], atol=1e-10)
        control = np.array(baseline['baseline_logit_differences']['control'])
        experiment = np.array(baseline['baseline_logit_differences']['experiment'])
        n_rows = baseline['n_rows_kept']
        assert len(control) == len(experiment) == n_rows
        rechecks[variant] = int(((control>0)&(experiment<0)).sum())
        source_differences[variant] = baseline['baseline_rotation']-original['best_rotation']
        for k in manifest['top_k_values']:
            cell = json.loads((folder/f'k{k:02d}.json').read_text())
            assert cell['system_prompt_id'] == variant and cell['k'] == k
            earliest_layer = min(h['layer'] for h in cell['heads'])
            effect = cell['effects'][DIRECTION]
            patched = np.array(effect['patched_logit_differences'])
            self_cosine = np.array(effect['self_cosine_by_layer'])
            assert len(patched)==n_rows and len(self_cosine)==n_layers
            assert np.isfinite(patched).all() and np.isfinite(self_cosine).all()
            np.testing.assert_allclose(early_to_late_drop(self_cosine), effect['self_rotation'], atol=1e-10)
            pre_patch_logit_diff = effect['logit_diff'] - effect['logit_shift']
            np.testing.assert_allclose(-effect['logit_shift']/abs(pre_patch_logit_diff), effect['normalized_logit_shift'], atol=1e-8)
            np.testing.assert_allclose(effect['self_rotation']/effect['baseline_rotation'], effect['normalized_rotation_change'], atol=1e-8)
            # Causality: before the earliest patched head's layer, the patched
            # run is byte-identical to its own unpatched self, so self-cosine
            # there must be exactly 1.
            np.testing.assert_allclose(self_cosine[:earliest_layer], 1.0, atol=1e-6, rtol=0)
            rows.append({'system_prompt_id': variant, 'position_label': baseline['position_label'],
                'input_offset': baseline['input_offset'], 'n_rows_kept': n_rows,
                'k': k, 'direction': DIRECTION,
                'normalized_logit_shift': effect['normalized_logit_shift'],
                'normalized_rotation_change': effect['normalized_rotation_change']})
    write_csv(out/'per_prompt_k_metrics.csv', rows)
    variant_ids = manifest['variants']
    k_values = manifest['top_k_values']
    aggregates = []
    for k in k_values:
        group = [r for r in rows if r['k']==k]
        aggregates.append({'direction':DIRECTION, 'k':k, 'n_prompts':len(group),
            **{key:float(np.mean([r[key] for r in group])) for key in (
                'normalized_logit_shift', 'normalized_rotation_change')}})
    write_csv(out/'mean_by_k.csv', aggregates)
    best_rows = []
    for variant in variant_ids:
        group = [r for r in rows if r['system_prompt_id']==variant]
        best_logit_shift = max(group, key=lambda r:r['normalized_logit_shift'])
        best_rotation = max(group, key=lambda r:r['normalized_rotation_change'])
        best_rows.append({'system_prompt_id':variant, 'direction':DIRECTION,
            'position_label':best_logit_shift['position_label'], 'n_rows_kept':best_logit_shift['n_rows_kept'],
            'max_normalized_logit_shift_k':best_logit_shift['k'], 'max_normalized_logit_shift':best_logit_shift['normalized_logit_shift'],
            'max_normalized_rotation_change_k':best_rotation['k'], 'max_normalized_rotation_change':best_rotation['normalized_rotation_change']})
    write_csv(out/'best_k_per_prompt.csv', best_rows)
    write_csv(out/'ranked_heads.csv', ranked)
    report = {'status':'passed', 'n_prompts':len(variant_ids), 'max_k':max_k, 'n_directional_cells':len(rows),
        'selection': manifest.get('selection', 'ranked'),
        'rank_metric': manifest.get('rank_metric'),
        'random_seed': manifest.get('random_seed'),
        'checks':['exact frozen rows/positions', 'saved cosine curves reproduce half-layer scores and changes',
                  'per-example logits reproduce normalized logit shift/rotation change', 'block outputs before the earliest patched layer are unchanged'],
        'baseline_filter_recheck_counts':rechecks, 'baseline_rotation_difference_from_selection_run':source_differences}
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    print(DIRECTION, 'max mean normalized logit shift:',max(aggregates,key=lambda r:r['normalized_logit_shift']),flush=True)

    selection_desc = (f'ranked by {manifest["rank_metric"]}' if manifest.get('selection', 'ranked') == 'ranked'
                      else f'uniform random draw (seed {manifest.get("random_seed")})')
    plot_diagnostic_heatmaps(rows, variant_ids, selection, k_values, 'k', 'Cumulative top-k heads', out,
        f'Cumulative mean head patching at each prompt’s frozen best token\nTop-{max_k} heads {selection_desc}')
    plot_diagnostic_lines(aggregates, 'k', 'Cumulative top-k heads', out,
        f'Mean across {len(variant_ids)} prompts · each prompt weighted equally', filename_stem='mean_k_effects')


def random_heads(seed, n_layers, n_heads, k):
    """Uniform draw of k distinct (layer, head) pairs, no ranking score.

    Matches results.uniform_cumulative_head_interventions's sampling: one
    seeded permutation of the full pool, taken as a prefix, so k=1..max_k are
    nested subsets exactly like the ranked selection.
    """
    import random
    pool = [(layer, head) for layer in range(n_layers) for head in range(n_heads)]
    if k > len(pool):
        raise ValueError(f'Requested {k} random heads exceeds {len(pool)} available')
    drawn = random.Random(seed).sample(pool, k)
    return [(layer, head, None) for layer, head in drawn]


def rank_heads_from_screen(data, k, *, metric='recovery'):
    """Global (layer, head) ranking from a per-prompt-position head screen (schema 3),
    by the 'noise' (lying-to-honest) direction's screened effects.

    Unlike results.rank_heads, this does not require a single shared patch
    position across variants: each variant's effects were already measured at
    its own frozen site.
    """
    import numpy as np
    scored = []
    for i, item in enumerate(data['interventions']):
        values = [v['effects'][i]['noise'][metric] for v in data['variants']]
        if any(x is None or not np.isfinite(x) for x in values):
            raise ValueError(f'Head screen missing finite {metric} for intervention {i}')
        scored.append((item['layer'], item['head'], float(np.mean(values))))
    scored.sort(key=lambda row: row[2], reverse=True)
    if k > len(scored):
        raise ValueError(f'Requested top-{k} exceeds {len(scored)} screened heads')
    return scored[:k]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, default=ROOT/'results/patching/unpatched-half-20prompts-100')
    p.add_argument('--selection', choices=['ranked', 'random'], default='ranked',
                   help='ranked: top-k from --head-screen; random: uniform draw, a baseline control')
    p.add_argument('--head-screen', type=Path, help='schema-3 results.json from the fast per-prompt-position head sweep; required for --selection ranked')
    p.add_argument('--rank-metric', choices=['recovery', 'flip_rate', 'logit_shift'], default='recovery')
    p.add_argument('--random-seed', type=int, default=0, help='Selection --selection random: seed for the single uniform draw')
    p.add_argument('--top-k', default='30', help='Maximum cumulative head count; sweeps 1..this value')
    p.add_argument('--output', type=Path, required=True,
                   help='Each --selection needs its own output dir, e.g. '
                        'results/patching/mean-head-best-position-noise-ranked-k30 '
                        'or mean-head-best-position-random-baseline-k30')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--variants', help='Optional comma-separated subset for a pilot')
    args = p.parse_args()
    if args.selection == 'ranked' and args.head_screen is None:
        p.error('--selection ranked requires --head-screen')
    import numpy as np
    import torch
    from src.activation_patching.run_custom_sweep import parse_args as parse_patching_args, resolved_config
    from src.activation_patching.utils.core import (PatchingEngine, make_intervention,
                                              first_answer_token_id, parse_indices)
    from src.activation_patching.utils.metrics import ContrastAccumulator, contrast_cosine, early_to_late_drop, behavior_scores
    from src.activation_patching.utils.prefix_rotation import score_and_capture
    from src.activation_patching.utils.results import write_json
    from src.activation_patching.utils.variant_positions import resolve_input_position
    from src.model.inference import load_model_and_tokenizer
    from src.prompt.chat_template import apply_chat_template_split, load_experiment_system_prompts
    from src.task_corpus.records import StatementRecord

    max_k = int(args.top_k)
    if max_k < 1:
        raise ValueError('--top-k must be positive')
    if args.batch_size < 1 or not torch.cuda.is_available():
        raise RuntimeError('CUDA and a positive batch size are required')

    baseline = json.loads((args.baseline/'results.json').read_text())
    records = [StatementRecord(**r) for r in json.loads((args.baseline/'rows.json').read_text())]
    if baseline['status'] != 'complete':
        raise ValueError('Baseline must be complete')
    screen = None
    if args.selection == 'ranked':
        screen = json.loads(args.head_screen.read_text())
        if screen.get('status') != 'complete':
            raise ValueError('Head screen must be complete')
        if screen.get('schema_version') != 3 or not screen.get('position_selection'):
            raise ValueError('Expected a per-prompt-position (schema 3) head screen from step3_head_sweep.py --positions-from')

    cfg = resolved_config(parse_patching_args(['--output', str(args.output), '--model-path', baseline['config']['model_path']]))
    current = json.loads(json.dumps(asdict(cfg), default=str))
    for key in ('model_path', 'model_dtype', 'task', 'dataset', 'user_prompt', 'answer_prefix',
                'control_system_prompt_map', 'gt_0_label_map', 'gt_1_label_map'):
        if current[key] != baseline['config'][key]:
            raise ValueError(f'Configuration differs from baseline: {key}')
    if screen is not None and screen['model_path'] != cfg.model_path:
        raise ValueError('Head screen model differs from the requested model')

    prompts = load_experiment_system_prompts(cfg)
    variants = parse_indices(args.variants) if args.variants else baseline['requested_variants']
    source = {v['system_prompt_id']: v for v in baseline['variants']}
    if not set(variants) <= set(source):
        raise ValueError('Requested prompt is absent from baseline')
    if screen is not None and not set(variants) <= {int(v) for v in screen['position_selection']['variants']}:
        raise ValueError('Requested prompt is absent from the head screen')

    torch.manual_seed(baseline['seed'])
    model, tokenizer = load_model_and_tokenizer(cfg.model_path, dtype=cfg.model_dtype)
    engine = PatchingEngine(model, tokenizer)

    if args.selection == 'ranked':
        ranked = rank_heads_from_screen(screen, max_k, metric=args.rank_metric)
    else:
        ranked = random_heads(args.random_seed, len(engine.layers), engine.n_heads, max_k)
    ranked_layers = sorted({layer for layer, _, _ in ranked})

    mapping = []
    for variant in variants:
        v = source[variant]
        if prompts[variant] != v['system_prompt']:
            raise ValueError('Prompt text changed since baseline')
        mapping.append({'system_prompt_id': variant, 'system_prompt': v['system_prompt'],
                        'position_label': v['best_position'], 'selection_score': v['best_rotation'],
                        'kept_row_indices': v['kept_row_indices'], 'n_rows_kept': v['n_rows_kept']})
    manifest = {
        'schema_version': 1,
        'experiment': 'mean_head_frozen_best_position_cumulative' if args.selection == 'ranked' else 'mean_head_frozen_best_position_random_baseline',
        'baseline_path': str(args.baseline.resolve()), 'baseline_sha256': sha256(args.baseline/'results.json'),
        'rows_sha256': sha256(args.baseline/'rows.json'),
        'selection': args.selection,
        'head_screen_path': str(args.head_screen.resolve()) if args.selection == 'ranked' else None,
        'rank_metric': args.rank_metric if args.selection == 'ranked' else None,
        'random_seed': args.random_seed if args.selection == 'random' else None,
        'random_sampling': 'uniform without replacement over all (layer, head) pairs, one draw' if args.selection == 'random' else None,
        'ranked_heads': [{'rank': i + 1, 'layer': l, 'head': h, 'score': s} for i, (l, h, s) in enumerate(ranked)],
        'top_k_values': list(range(1, max_k + 1)),
        'batch_size': args.batch_size, 'variants': variants, 'config': current,
        'rotation_metric': 'cross_half_cosine_drop_v1',
        'self_rotation': 'cross_half_cosine_drop_v1 between the patched recipient and its own unpatched self',
        'filter': 'exact saved kept_row_indices from the unpatched baseline, never re-filtered',
        'mean_reference': 'same filtered rows, GT classes pooled, separate mean per condition/layer/prompt',
        'site': 'o_proj input slice for each cumulative head; one fixed token per prompt',
        'direction': 'lying_to_honest',
        'source_hashes': {name: sha256(ROOT/name) for name in (
            'src/activation_patching/step4_patch_heads_cumulative.py', 'src/activation_patching/utils/core.py',
            'src/activation_patching/utils/prefix_rotation.py', 'src/activation_patching/utils/metrics.py')},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output/'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Output manifest differs; use a fresh output directory')
    write_json(path, manifest)
    write_json(args.output/'best_positions.json', mapping)
    write_json(args.output/'ranked_heads.json', manifest['ranked_heads'])
    write_json(args.output/'rows.json', [asdict(r) for r in records])
    write_json(args.output/'model.json', {'commit': getattr(model.config, '_commit_hash', None),
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'n_layers': len(engine.layers),
        'n_heads': engine.n_heads, 'attention_implementation': getattr(model.config, '_attn_implementation', None)})

    with torch.no_grad():
        for variant in variants:
            start = time.monotonic()
            out = args.output/'variants'/f'variant{variant:03d}'
            kept = source[variant]['kept_row_indices']
            by_id = {r.index: r for r in records}
            rows = [by_id[i] for i in kept]
            labels = np.array([r.label for r in rows])
            if len(rows) != source[variant]['n_rows_kept'] or set(labels) != {0, 1}:
                raise ValueError('Invalid saved retained sample')
            pairs = {c: apply_chat_template_split(cfg, rows, tokenizer, condition_type=c, system_prompt_id=variant)
                     for c in ('control', 'experiment')}
            position, token_id, token_text = resolve_input_position(tokenizer, pairs['control']+pairs['experiment'], source[variant]['best_position'])
            probe = pairs['control'][0]
            answer_ids = [first_answer_token_id(tokenizer, ''.join(probe), probe[1], m[cfg.task])
                          for m in (cfg.gt_0_label_map, cfg.gt_1_label_map)]
            if answer_ids[0] == answer_ids[1]:
                raise ValueError('Answers must have different token IDs')
            print(f'[start {variant}] n={len(rows)} site={source[variant]["best_position"]} offset={position}', flush=True)
            encs = {c: [engine.encode([''.join(p) for p in ps[lo:lo+args.batch_size]], [position])
                        for lo in range(0, len(rows), args.batch_size)] for c, ps in pairs.items()}

            scores, accumulators, head_means = {}, {}, {}
            requests = {('head', layer): [position] for layer in ranked_layers}
            for condition in pairs:
                chunks = []
                accumulator = ContrastAccumulator()
                sums = {key: None for key in requests}
                for batch, enc in enumerate(encs[condition]):
                    lo = batch*args.batch_size
                    labs = labels[lo:lo+args.batch_size]
                    logits, cache = score_and_capture(engine, enc, labs, answer_ids, position)
                    chunks.append(logits)
                    accumulator.add(cache, labs, np.ones(len(labs), dtype=bool))
                    captured = engine.capture(enc, requests)
                    for key, (values, _positions) in captured.items():
                        # keepdim=True preserves a size-1 batch axis so the mean
                        # broadcasts against the recipient batch in engine.patch(),
                        # matching source[:, indices]'s expected [1, n_pos, feat] shape.
                        values = values.float().sum(dim=0, keepdim=True)
                        sums[key] = values if sums[key] is None else sums[key] + values
                scores[condition] = torch.cat(chunks)
                accumulators[condition] = accumulator
                head_means[condition] = {key: (total/len(rows), [position]) for key, total in sums.items()}
            baseline_cosine = contrast_cosine(accumulators['control'], accumulators['experiment'])[:, 0]
            baseline_rotation = early_to_late_drop(baseline_cosine)
            if abs(baseline_rotation) < 1e-6:
                raise ValueError(f'normalized_rotation_change undefined: baseline_rotation is ~0 for prompt {variant}')
            mask = torch.ones(len(rows), dtype=torch.bool)
            vbase = {
                'system_prompt_id': variant, 'system_prompt': prompts[variant], 'n_rows_kept': len(rows),
                'kept_row_indices': kept, 'kept_labels': labels.tolist(),
                'kept_class_counts': accumulators['control'].counts,
                'position_label': source[variant]['best_position'], 'input_offset': position,
                'token_id': token_id, 'token_text': token_text, 'answer_token_ids': answer_ids,
                'baseline_rotation': baseline_rotation, 'selection_baseline_rotation': source[variant]['best_rotation'],
                'baseline_cosine_by_layer': baseline_cosine.tolist(),
                'baseline_logit_differences': {c: s.tolist() for c, s in scores.items()},
                'baseline_filter_recheck_count': int(((scores['control']>0)&(scores['experiment']<0)).sum()),
                'n_layers': len(engine.layers), 'ranked_heads': manifest['ranked_heads'],
            }
            write_json(out/'baseline.json', vbase)
            np.savez_compressed(out/'donor_head_means.npz', **{
                f'{c}_layer{layer:02d}': t.cpu().numpy() for c, means in head_means.items() for (comp, layer), (t, _p) in means.items()})
            for k in range(1, max_k + 1):
                cell = out/f'k{k:02d}.json'
                if cell.exists():
                    continue
                selected = ranked[:k]
                sites = [{'layer': l, 'heads': [h], 'positions': [position]} for l, h, _ in selected]
                item = make_intervention('head', sites, k=k)
                donor, recipient = 'experiment', 'control'
                accumulator, chunks = ContrastAccumulator(), []
                donor_cache = head_means[donor]
                for batch, enc in enumerate(encs[recipient]):
                    lo = batch*args.batch_size
                    labs = labels[lo:lo+args.batch_size]
                    with engine.patch(item, 'noise', donor_cache):
                        logits, cache = score_and_capture(engine, enc, labs, answer_ids, position)
                    chunks.append(logits)
                    accumulator.add(cache, labs, np.ones(len(labs), dtype=bool))
                patched = torch.cat(chunks)
                # Rotation of the patched recipient AWAY FROM ITS OWN unpatched
                # self. This is 0 at k=0 (patched == unpatched self) and should
                # approach baseline_rotation if patching fully converts the
                # recipient into the donor condition, since patched would then
                # equal the untouched donor exactly.
                self_cosine = contrast_cosine(accumulators[recipient], accumulator)[:, 0]
                self_rotation = early_to_late_drop(self_cosine)
                scores_dict = behavior_scores(scores['control'], scores['experiment'], patched, mask, 'noise')
                # Paper metrics only; flip_rate/recovery are not part of the
                # paper's normalized-logit-shift / normalized-rotation-change
                # definitions and are deliberately not computed here.
                pre_patch_logit_diff = scores_dict['logit_diff'] - scores_dict['logit_shift']
                if abs(pre_patch_logit_diff) < 1e-6:
                    raise ValueError(f'normalized_logit_shift undefined: pre-patch logit_diff is ~0 for prompt {variant} k={k}')
                normalized_logit_shift = -scores_dict['logit_shift'] / abs(pre_patch_logit_diff)
                normalized_rotation_change = self_rotation / baseline_rotation
                metrics = {
                    'normalized_logit_shift': normalized_logit_shift,
                    'normalized_rotation_change': normalized_rotation_change,
                    'logit_diff': scores_dict['logit_diff'], 'logit_shift': scores_dict['logit_shift'],
                    'baseline_rotation': baseline_rotation, 'self_rotation': self_rotation,
                    'self_cosine_by_layer': self_cosine.tolist(),
                    'patched_logit_differences': patched.tolist(),
                }
                effects = {'lying_to_honest': metrics}
                write_json(cell, {'system_prompt_id': variant, 'k': k,
                                 'heads': [{'rank': i+1, 'layer': l, 'head': h, 'score': s} for i, (l, h, s) in enumerate(selected)],
                                 'position_label': vbase['position_label'], 'n_rows_kept': len(rows), 'effects': effects})
                print(f'prompt {variant} k={k} lying_to_honest: normalized_logit_shift={normalized_logit_shift:.3f} normalized_rotation_change={normalized_rotation_change:.3f}', flush=True)
            write_json(out/'status.json', {'status':'complete', 'elapsed_seconds':time.monotonic()-start})
            print(f'[done {variant}] seconds={time.monotonic()-start:.1f}', flush=True)
    write_json(args.output/'status.json', {'status':'complete', 'n_variants':len(variants), 'max_k':max_k})
    summarize_and_plot(args.output)


if __name__ == '__main__':
    main()
