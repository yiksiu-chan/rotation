"""Step 1: For each system prompt, finds which token position shows the
strongest rotation, which we will use for the patching experiments.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-path', default=None, help='Defaults to config/default.yaml\'s model_path')
    parser.add_argument('--variants', default=None, help='Defaults to config/patching_lying.yaml\'s default_patching_variants')
    parser.add_argument('--n-rows', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    import numpy as np
    import torch
    from src.activation_patching.run_custom_sweep import parse_args, resolved_config
    from src.activation_patching.utils.core import balanced_sample, first_answer_token_id, parse_indices
    from src.activation_patching.utils.metrics import honesty, argmax_matches_truth, ContrastAccumulator, contrast_cosine, early_to_late_drop
    from src.activation_patching.utils.results import write_json
    from src.activation_patching.utils.plotting import plot_position_heatmap
    from src.task_corpus.dispatch import load_statements_for_config as _load_statements
    from src.cache_activations.hooks import generate_with_multi_pos_cache, position_labels
    from src.model.inference import load_model_and_tokenizer, tokenize_prompt_batch
    from src.prompt.chat_template import apply_chat_template_split, load_experiment_system_prompts

    if args.batch_size < 1 or not torch.cuda.is_available():
        raise RuntimeError('A positive batch size and CUDA GPU are required')
    if args.output.exists() and any(args.output.glob('*.json')):
        raise ValueError('Use a fresh output directory to preserve existing results')
    patching_args = ['--output', str(args.output)]
    if args.model_path:
        patching_args += ['--model-path', args.model_path]
    cfg = resolved_config(parse_args(patching_args))
    if not (args.variants or cfg.default_patching_variants):
        raise ValueError('Pass --variants or set default_patching_variants in the loaded config')


    variants = parse_indices(args.variants or cfg.default_patching_variants)
    prompts = load_experiment_system_prompts(cfg)
    torch.manual_seed(args.seed)
    rows = balanced_sample(_load_statements(cfg), args.n_rows, args.seed)
    if len(rows) != args.n_rows:
        raise ValueError('Could not obtain the requested balanced sample')
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output/'rows.json', [asdict(r) for r in rows])
    model, tokenizer = load_model_and_tokenizer(cfg.model_path, dtype=cfg.model_dtype)
    labels = np.array([int(r.label) for r in rows])
    positions = position_labels(cfg.n_pre_prompt_tokens, cfg.n_generated_tokens)

    result = {
        'status': 'running', 'rotation_metric': 'cross_half_cosine_drop_v1',
        'metric_definition': 'max(0, max(cosine[:n_layers//2]) - min(cosine[n_layers//2:]))',
        'config': asdict(cfg), 'seed': args.seed, 'batch_size': args.batch_size,
        'n_rows_total': len(rows), 'requested_variants': variants,
        'row_indices': [r.index for r in rows], 'position_labels': positions,
        'filter': 'argmax(full-vocab logits) is the ground-truth answer token in control '
                  'AND the opposite (lying) answer token in experiment; no fallback',
        'gpu': torch.cuda.get_device_name(), 'torch_version': torch.__version__,
        'model_commit': getattr(model.config, '_commit_hash', None),
        'metric_source_sha256': hashlib.sha256((ROOT/'src/activation_patching/utils/metrics.py').read_bytes()).hexdigest(),
        'variants': [],
    }

    write_json(args.output/'results.json', result)

    with torch.no_grad():
        for variant in variants:
            start = time.monotonic()
            pairs = {c: apply_chat_template_split(cfg, rows, tokenizer, condition_type=c, system_prompt_id=variant)
                     for c in ('control', 'experiment')}
            probe = pairs['control'][0]
            ids = [first_answer_token_id(tokenizer, ''.join(probe), probe[1], m[cfg.task])
                   for m in (cfg.gt_0_label_map, cfg.gt_1_label_map)]
            if ids[0] == ids[1]:
                raise ValueError('Answer token IDs must differ')
            scores, matches = {}, {}
            for condition, ps in pairs.items():
                # Expected argmax: the ground-truth answer in control, its
                # opposite (the lying answer) in experiment
                expect_flip = 1 if condition == 'experiment' else 0
                diff_chunks, match_chunks = [], []
                for lo in range(0, len(rows), args.batch_size):
                    enc = tokenize_prompt_batch(tokenizer, [''.join(p) for p in ps[lo:lo+args.batch_size]], model.device)
                    logits = model(**enc, use_cache=False).logits[:, -1]
                    batch_labels = labels[lo:lo+args.batch_size]
                    diff_chunks.append(honesty(logits, batch_labels, *ids))
                    expected = (1 - batch_labels) if expect_flip else batch_labels
                    match_chunks.append(argmax_matches_truth(logits, expected, *ids))
                scores[condition] = torch.cat(diff_chunks).numpy()
                matches[condition] = torch.cat(match_chunks).numpy()
            # Success = the model's top predicted token is exactly the expected
            # answer in both conditions (see argmax_matches_truth): the correct
            # answer under control, the specific opposite (lying) answer under
            # experiment.
            keep = matches['control'] & matches['experiment']
            kept_labels = labels[keep]
            if set(kept_labels.tolist()) != {0, 1}:
                raise ValueError(f'Variant {variant}: filter retained insufficient classes for rotation')
            print(f'[variant {variant}] retained {int(keep.sum())}/{len(rows)}', flush=True)
            accumulators = {}
            for condition, ps in pairs.items():
                retained = [p for p, k in zip(ps, keep) if k]
                accumulator = ContrastAccumulator()
                for lo in range(0, len(retained), args.batch_size):
                    batch = retained[lo:lo+args.batch_size]
                    _, cache = generate_with_multi_pos_cache(
                        model, tokenizer, batch, n_pre=cfg.n_pre_prompt_tokens,
                        n_gen=cfg.n_generated_tokens, batch_size=len(batch),
                        max_new_tokens=cfg.max_new_tokens, desc=f'{variant} {condition}')
                    accumulator.add(cache, kept_labels[lo:lo+len(batch)], np.ones(len(batch), dtype=bool))
                accumulators[condition] = accumulator
            cosine = contrast_cosine(accumulators['control'], accumulators['experiment'])
            values = [early_to_late_drop(cosine[:, p]) for p in range(cosine.shape[1])]
            order = np.argsort(-np.asarray(values), kind='stable')
            best, second = map(int, order[:2])
            entry = {
                'system_prompt_id': variant, 'system_prompt': prompts[variant],
                'n_rows_kept': int(keep.sum()), 'kept_class_counts': accumulators['control'].counts,
                'kept_row_indices': [r.index for r, k in zip(rows, keep) if k],
                'behavior_logit_differences': {c: v.tolist() for c, v in scores.items()},
                'best_position': positions[best], 'best_rotation': values[best],
                'runner_up_position': positions[second], 'runner_up_rotation': values[second],
                'rotation_by_position': values, 'cosine_by_layer_position': cosine.tolist(),
                'early_layers': list(range(cosine.shape[0]//2)),
                'late_layers': list(range(cosine.shape[0]//2, cosine.shape[0])),
                'elapsed_seconds': time.monotonic()-start,
            }
            write_json(args.output/'variants'/f'variant{variant:03d}.json', entry)
            result['variants'].append(entry)
            result['status'] = 'complete' if len(result['variants']) == len(variants) else 'running'
            write_json(args.output/'results.json', result)
            fields = ['system_prompt_id', 'n_rows_kept', 'best_position', 'best_rotation', 'runner_up_position', 'runner_up_rotation']
            with (args.output/'summary.csv').open('w') as f:
                writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(result['variants'])
            print(f'[done {variant}] best={positions[best]} rotation={values[best]:.6f}', flush=True)

    for path in plot_position_heatmap(result, args.output):
        print(f'Saved {path}')


if __name__ == '__main__':
    main()
