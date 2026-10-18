"""Step 2: run residual stream activation patching, from lying to honest. """
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

DIRECTION = 'lying_to_honest' ## change to honest_to_lying if needed

def summarize_and_plot(out):
    """Validate, tabulate (CSV) and plot a completed run directory.

    Re-reads the just-written per-layer JSON cells from disk (rather than
    reusing in-memory state) so this also works standalone against an
    already-completed --output directory, and so a resumed run always
    reflects every variant/layer on disk, not just the ones this particular
    invocation computed.
    """
    import numpy as np
    from src.activation_patching.utils.metrics import early_to_late_drop
    from src.activation_patching.utils.plotting import plot_diagnostic_heatmaps, plot_diagnostic_lines, plot_residual_paper_panel
    from src.activation_patching.utils.results import write_csv

    manifest = json.loads((out/'manifest.json').read_text())
    source = json.loads((Path(manifest['baseline_path'])/'results.json').read_text())
    selection = {v['system_prompt_id']: v for v in source['variants']}
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
        with np.load(folder/'donor_means.npz') as means:
            for condition in ('control', 'experiment'):
                assert means[condition].shape[:3] == (n_layers, 1, 1)
                assert np.isfinite(means[condition]).all()
        rechecks[variant] = int(((control>0)&(experiment<0)).sum())
        source_differences[variant] = baseline['baseline_rotation']-original['best_rotation']
        for layer in range(n_layers):
            cell = json.loads((folder/f'layer{layer:02d}.json').read_text())
            assert cell['system_prompt_id'] == variant and cell['layer'] == layer
            effect = cell['effects'][DIRECTION]
            patched = np.array(effect['patched_logit_differences'])
            self_cosine = np.array(effect['self_cosine_by_layer'])
            assert len(patched)==n_rows and len(self_cosine)==n_layers
            assert np.isfinite(patched).all() and np.isfinite(self_cosine).all()
            np.testing.assert_allclose(early_to_late_drop(self_cosine), effect['self_rotation'], atol=1e-10)
            pre_patch_logit_diff = effect['logit_diff'] - effect['logit_shift']
            np.testing.assert_allclose(-effect['logit_shift']/abs(pre_patch_logit_diff), effect['normalized_logit_shift'], atol=1e-8)
            np.testing.assert_allclose(effect['self_rotation']/effect['baseline_rotation'], effect['normalized_rotation_change'], atol=1e-8)
            # Causality: replacing layer L's block output cannot affect earlier blocks.
            np.testing.assert_allclose(self_cosine[:layer], 1.0, atol=1e-6, rtol=0)
            # A single pooled donor mean erases GT contrast at the patched token/layer.
            np.testing.assert_allclose(self_cosine[layer], 0, atol=1e-7)
            rows.append({'system_prompt_id': variant, 'position_label': baseline['position_label'],
                'input_offset': baseline['input_offset'], 'n_rows_kept': n_rows,
                'layer': layer, 'direction': DIRECTION,
                'normalized_logit_shift': effect['normalized_logit_shift'],
                'normalized_rotation_change': effect['normalized_rotation_change']})
    write_csv(out/'per_prompt_layer_metrics.csv', rows)
    variant_ids = manifest['variants']
    layers = list(range(n_layers))
    aggregates = []
    for layer in layers:
        group = [r for r in rows if r['layer']==layer]
        aggregates.append({'direction':DIRECTION, 'layer':layer, 'n_prompts':len(group),
            **{key:float(np.mean([r[key] for r in group])) for key in (
                'normalized_logit_shift', 'normalized_rotation_change')}})
    write_csv(out/'mean_by_layer.csv', aggregates)
    best_rows = []
    for variant in variant_ids:
        group = [r for r in rows if r['system_prompt_id']==variant]
        best_logit_shift = max(group, key=lambda r:r['normalized_logit_shift'])
        best_rotation = max(group, key=lambda r:r['normalized_rotation_change'])
        best_rows.append({'system_prompt_id':variant, 'direction':DIRECTION,
            'position_label':best_logit_shift['position_label'], 'n_rows_kept':best_logit_shift['n_rows_kept'],
            'max_normalized_logit_shift_layer':best_logit_shift['layer'], 'max_normalized_logit_shift':best_logit_shift['normalized_logit_shift'],
            'max_normalized_rotation_change_layer':best_rotation['layer'], 'max_normalized_rotation_change':best_rotation['normalized_rotation_change']})
    write_csv(out/'best_layers_per_prompt.csv', best_rows)
    report = {'status':'passed', 'n_prompts':len(variant_ids), 'n_layers':n_layers, 'n_directional_cells':len(rows),
        'checks':['exact frozen rows/positions', 'finite saved donor means', 'saved cosine curves reproduce half-layer scores',
                  'per-example logits reproduce normalized logit shift/rotation change', 'upstream layers unchanged', 'pooled mean removes GT contrast at patched layer'],
        'baseline_filter_recheck_counts':rechecks, 'baseline_rotation_difference_from_selection_run':source_differences}
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    print(DIRECTION, 'max mean normalized logit shift:',max(aggregates,key=lambda r:r['normalized_logit_shift']),flush=True)

    plot_diagnostic_heatmaps(rows, variant_ids, selection, layers, 'layer', 'Patched residual layer (zero-based)', out,
        'Mean residual patching at each prompt’s frozen best token\nSaved filtered rows', xtick_step=2)
    plot_diagnostic_lines(aggregates, 'layer', 'Patched residual layer (zero-based)', out,
        f'Mean across {len(variant_ids)} prompts · each prompt weighted equally', filename_stem='mean_layer_effects')
    left_mean = np.array([r['normalized_logit_shift'] for r in aggregates])
    left_std = np.array([np.std([r['normalized_logit_shift'] for r in rows if r['layer']==l]) for l in layers])
    right_mean = np.array([r['normalized_rotation_change'] for r in aggregates])
    right_std = np.array([np.std([r['normalized_rotation_change'] for r in rows if r['layer']==l]) for l in layers])
    plot_residual_paper_panel(layers, left_mean, left_std, right_mean, right_std, out,
        title='Mean patching on the residual stream')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, default=ROOT/'results/patching/unpatched-half-20prompts-100')
    p.add_argument('--output', type=Path, default=ROOT/'results/patching/mean-residual-best-position-filtered')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--variants', help='Optional comma-separated subset for a pilot')
    args = p.parse_args()
    import numpy as np
    import torch
    from src.activation_patching.run_custom_sweep import parse_args, resolved_config
    from src.activation_patching.utils.core import PatchingEngine, build_interventions, first_answer_token_id, parse_indices
    from src.activation_patching.utils.metrics import ContrastAccumulator, contrast_cosine, early_to_late_drop, behavior_scores
    from src.activation_patching.utils.prefix_rotation import score_and_capture, generate_from_enc
    from src.activation_patching.utils.results import write_json
    from src.activation_patching.utils.variant_positions import resolve_input_position
    from src.model.inference import load_model_and_tokenizer
    from src.prompt.chat_template import apply_chat_template_split, load_experiment_system_prompts
    from src.task_corpus.records import StatementRecord

    if args.batch_size < 1 or not torch.cuda.is_available():
        raise RuntimeError('CUDA and a positive batch size are required')
    baseline = json.loads((args.baseline/'results.json').read_text())
    records = [StatementRecord(**r) for r in json.loads((args.baseline/'rows.json').read_text())]
    if baseline['status'] != 'complete':
        raise ValueError('Baseline must be complete')
    cfg = resolved_config(parse_args(['--output', str(args.output), '--model-path', baseline['config']['model_path']]))
    current = json.loads(json.dumps(asdict(cfg), default=str))
    for key in ('model_path', 'model_dtype', 'task', 'dataset', 'user_prompt', 'answer_prefix',
                'control_system_prompt_map', 'gt_0_label_map', 'gt_1_label_map'):
        if current[key] != baseline['config'][key]:
            raise ValueError(f'Configuration differs from baseline: {key}')
    prompts = load_experiment_system_prompts(cfg)
    variants = parse_indices(args.variants) if args.variants else baseline['requested_variants']
    source = {v['system_prompt_id']: v for v in baseline['variants']}
    if not set(variants) <= set(source):
        raise ValueError('Requested prompt is absent from baseline')
    mapping = []
    for variant in variants:
        v = source[variant]
        if prompts[variant] != v['system_prompt']:
            raise ValueError('Prompt text changed since baseline')
        mapping.append({'system_prompt_id': variant, 'system_prompt': v['system_prompt'],
                        'position_label': v['best_position'], 'selection_score': v['best_rotation'],
                        'kept_row_indices': v['kept_row_indices'], 'n_rows_kept': v['n_rows_kept']})
    manifest = {
        'schema_version': 1, 'experiment': 'mean_residual_frozen_best_position',
        'baseline_path': str(args.baseline.resolve()),
        'baseline_sha256': sha256(args.baseline/'results.json'),
        'rows_sha256': sha256(args.baseline/'rows.json'),
        'batch_size': args.batch_size, 'variants': variants, 'config': current,
        'rotation_metric': 'cross_half_cosine_drop_v1',
        'self_rotation': 'cross_half_cosine_drop_v1 between the patched recipient and its own unpatched self',
        'filter': 'exact saved kept_row_indices, never re-filtered',
        'mean_reference': 'same filtered rows, GT classes pooled, separate mean per condition/layer/prompt',
        'site': 'full transformer block output; one layer at a time; fixed token per prompt',
        'direction': 'lying_to_honest',
        'source_hashes': {name: sha256(ROOT/name) for name in (
            'src/activation_patching/step2_patch_residual.py', 'src/activation_patching/utils/core.py',
            'src/activation_patching/utils/prefix_rotation.py', 'src/activation_patching/utils/metrics.py')},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output/'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Output manifest differs; use a fresh output directory')
    write_json(path, manifest)
    write_json(args.output/'best_positions.json', mapping)
    write_json(args.output/'rows.json', [asdict(r) for r in records])
    torch.manual_seed(baseline['seed'])
    model, tokenizer = load_model_and_tokenizer(cfg.model_path, dtype=cfg.model_dtype)
    engine = PatchingEngine(model, tokenizer)
    write_json(args.output/'model.json', {'commit': getattr(model.config, '_commit_hash', None),
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'n_layers': len(engine.layers),
        'attention_implementation': getattr(model.config, '_attn_implementation', None)})
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
            site = resolve_input_position(tokenizer, pairs['control']+pairs['experiment'], source[variant]['best_position'])
            position, token_id, token_text = site
            probe = pairs['control'][0]
            answer_ids = [first_answer_token_id(tokenizer, ''.join(probe), probe[1], m[cfg.task])
                          for m in (cfg.gt_0_label_map, cfg.gt_1_label_map)]
            if answer_ids[0] == answer_ids[1]:
                raise ValueError('Answers must have different token IDs')
            print(f'[start {variant}] n={len(rows)} site={source[variant]["best_position"]} offset={position}', flush=True)
            encs = {c: [engine.encode([''.join(p) for p in ps[lo:lo+args.batch_size]], [position])
                        for lo in range(0, len(rows), args.batch_size)] for c, ps in pairs.items()}
            scores, accumulators, means, unpatched_completions = {}, {}, {}, {}
            row_indices = [r.index for r in rows]
            for condition in pairs:
                chunks, total, responses = [], None, []
                accumulator = ContrastAccumulator()
                for batch, enc in enumerate(encs[condition]):
                    lo = batch*args.batch_size
                    labs = labels[lo:lo+args.batch_size]
                    logits, cache = score_and_capture(engine, enc, labs, answer_ids, position)
                    chunks.append(logits)
                    accumulator.add(cache, labs, np.ones(len(labs), dtype=bool))
                    batch_sum = cache.float().sum(dim=1, keepdim=True)
                    total = batch_sum if total is None else total+batch_sum
                    responses.extend(generate_from_enc(engine, enc, tokenizer, cfg.max_new_tokens))
                scores[condition] = torch.cat(chunks)
                accumulators[condition] = accumulator
                means[condition] = total / len(rows)
                unpatched_completions[condition] = [{'row_index': i, 'response': r} for i, r in zip(row_indices, responses)]
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
                'n_layers': len(engine.layers), 'completions': unpatched_completions,
            }
            write_json(out/'baseline.json', vbase)
            np.savez_compressed(out/'donor_means.npz', **{c: t.cpu().numpy() for c,t in means.items()})
            for layer in range(len(engine.layers)):
                cell = out/f'layer{layer:02d}.json'
                if cell.exists():
                    continue
                item = build_interventions('residual', [layer], [position], [], 'window')[0]
                donor, recipient = 'experiment', 'control'
                accumulator, chunks, responses = ContrastAccumulator(), [], []
                donor_cache = {('residual', layer): (means[donor][layer], [position])}
                for batch, enc in enumerate(encs[recipient]):
                    lo = batch*args.batch_size
                    labs = labels[lo:lo+args.batch_size]
                    with engine.patch(item, 'noise', donor_cache):
                        logits, cache = score_and_capture(engine, enc, labs, answer_ids, position)
                    chunks.append(logits)
                    accumulator.add(cache, labs, np.ones(len(labs), dtype=bool))
                    with engine.patch(item, 'noise', donor_cache):
                        responses.extend(generate_from_enc(engine, enc, tokenizer, cfg.max_new_tokens))
                patched = torch.cat(chunks)
                # Rotation of the patched recipient away from its own unpatched self.
                self_cosine = contrast_cosine(accumulators[recipient], accumulator)[:, 0]
                self_rotation = early_to_late_drop(self_cosine)
                scores_dict = behavior_scores(scores['control'], scores['experiment'], patched, mask, 'noise')
                # Paper metrics only. logit_diff/logit_shift here is a recipient
                # baseline recomputed algebraically as logit_diff - logit_shift
                # (pre-patch value); flip_rate/recovery are not part of the
                # paper's normalized-logit-shift / normalized-rotation-change
                # definitions and are deliberately not computed here.
                pre_patch_logit_diff = scores_dict['logit_diff'] - scores_dict['logit_shift']
                if abs(pre_patch_logit_diff) < 1e-6:
                    raise ValueError(f'normalized_logit_shift undefined: pre-patch logit_diff is ~0 for prompt {variant} layer {layer}')
                normalized_logit_shift = -scores_dict['logit_shift'] / abs(pre_patch_logit_diff)
                normalized_rotation_change = self_rotation / baseline_rotation
                metrics = {
                    'normalized_logit_shift': normalized_logit_shift,
                    'normalized_rotation_change': normalized_rotation_change,
                    'logit_diff': scores_dict['logit_diff'], 'logit_shift': scores_dict['logit_shift'],
                    'baseline_rotation': baseline_rotation, 'self_rotation': self_rotation,
                    'self_cosine_by_layer': self_cosine.tolist(),
                    'patched_logit_differences': patched.tolist(),
                    'completions': [{'row_index': i, 'response': r} for i, r in zip(row_indices, responses)],
                }
                effects = {'lying_to_honest': metrics}
                write_json(cell, {'system_prompt_id': variant, 'layer': layer, 'position_label': vbase['position_label'],
                                 'n_rows_kept': len(rows), 'effects': effects})
                print(f'prompt {variant} layer {layer} lying_to_honest: normalized_logit_shift={normalized_logit_shift:.3f} normalized_rotation_change={normalized_rotation_change:.3f}', flush=True)
            write_json(out/'status.json', {'status':'complete', 'elapsed_seconds':time.monotonic()-start})
            print(f'[done {variant}] seconds={time.monotonic()-start:.1f}', flush=True)
    write_json(args.output/'status.json', {'status':'complete', 'n_variants':len(variants), 'n_layers':len(engine.layers)})
    summarize_and_plot(args.output)


if __name__ == '__main__':
    main()
