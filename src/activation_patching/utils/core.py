"""One execution engine for residual, attention, MLP, and head interventions.

Only this module handles model hooks. CLI, serialization, and plotting stay
outside the execution loop. All source/target layers are identical; only the
prompt condition changes in an interchange intervention.
"""
from __future__ import annotations
from contextlib import contextmanager
import random
import re

from .metrics import (honesty, select_rows, behavior_scores, ContrastAccumulator,
                      contrast_cosine, best_rotation, early_to_late_drop)


def parse_indices(spec, size=None, *, negative=False):
    """Inclusive ranges: '8-20' for layers, '-12:-1' for token offsets."""
    spec = str(spec).strip()
    if spec == 'all':
        if size is None:
            raise ValueError("'all' requires a model dimension")
        return list(range(size))
    if negative and spec == 'last':
        return [-1]
    values = []
    for part in spec.split(','):
        match = re.fullmatch(r'(-?\d+)(?::|\.\.|-)(-?\d+)', part.strip())
        if match:
            lo, hi = map(int, match.groups())
            if hi < lo:
                raise ValueError(f'Reversed range: {part}')
            values.extend(range(lo, hi + 1))
        else:
            values.append(int(part))
    if not values or len(set(values)) != len(values):
        raise ValueError('Indices must be nonempty and unique')
    if negative and any(v >= 0 for v in values):
        raise ValueError('Positions must be negative prompt-end offsets, e.g. --positions=-12:-1')
    if not negative and any(v < 0 or (size is not None and v >= size) for v in values):
        raise ValueError(f'Indices out of range (size={size})')
    return values


def balanced_sample(records, n_rows, seed):
    """Historical balanced sampling, including even rounding and scarce-class fallback."""
    if n_rows < 1 or not records:
        raise ValueError('Need nonempty data and a positive row count')
    rng = random.Random(seed)
    by_label = {label: [r for r in records if int(r.label) == label] for label in (0, 1)}
    k = min(n_rows // 2, len(by_label[0]), len(by_label[1]))
    if k == 0:
        pool = list(records)
        rng.shuffle(pool)
        return pool[:n_rows]
    rows = rng.sample(by_label[0], k) + rng.sample(by_label[1], k)
    rng.shuffle(rows)
    return rows


def first_answer_token_id(tokenizer, context, prefix, word):
    # Preserve the original answer-boundary convention used in the rebuttal.
    rest = word if prefix.endswith((' ', '\n', '\t')) else ' ' + word
    base = tokenizer(context, add_special_tokens=False)['input_ids']
    full = tokenizer(context + rest, add_special_tokens=False)['input_ids']
    i = 0
    while i < len(base) and i < len(full) and base[i] == full[i]:
        i += 1
    if i >= len(full):
        raise ValueError(f'Cannot isolate answer token for {word!r}')
    return int(full[i])


def make_intervention(component, sites, *, direction_sites=None, **coordinates):
    item = {'component': component, 'sites': sites, **coordinates}
    if direction_sites is not None:
        item['direction_sites'] = direction_sites
    return item


def build_interventions(component, layers, positions, heads, sweep):
    # An intervention is one independent recipient run. A residual 'position'
    # sweep changes one (layer, token); 'window' changes all selected tokens at
    # one layer. A 'head' sweep changes one head over the selected token window.
    if sweep in ('head', 'head-position') and component != 'head':
        raise ValueError(f'--sweep {sweep} requires --component head')
    result = []
    for layer in layers:
        position_sets = [[p] for p in positions] if sweep in ('position', 'head-position') else [positions]
        head_sets = [[h] for h in heads] if component == 'head' and sweep in ('head', 'head-position') else [heads if component == 'head' else None]
        for selected_heads in head_sets:
            for selected_positions in position_sets:
                coords = {'layer': layer}
                if len(selected_positions) == 1:
                    coords['position'] = selected_positions[0]
                if selected_heads is not None and len(selected_heads) == 1:
                    coords['head'] = selected_heads[0]
                site = {'layer': layer, 'positions': selected_positions, 'heads': selected_heads}
                result.append(make_intervention(component, [site], **coords))
    return result


class PatchingEngine:
    def __init__(self, model, tokenizer):
        self.model, self.tokenizer = model, tokenizer
        self.model.eval()
        try:
            self.layers = list(model.model.layers)
        except AttributeError as exc:
            raise ValueError('Patching requires Llama-style model.model.layers') from exc
        self.n_heads = int(model.config.num_attention_heads)
        self.head_dim = self.layers[0].self_attn.o_proj.in_features // self.n_heads
        if self.head_dim * self.n_heads != self.layers[0].self_attn.o_proj.in_features:
            raise ValueError('Attention output dimension is not divisible by head count')

    def module(self, component, layer):
        block = self.layers[layer]
        # Residual patching replaces the full block OUTPUT (after attention +
        # MLP residual additions). Head patching replaces slices of o_proj INPUT,
        # where head outputs are still concatenated, before being mixed by W_O.
        return {'residual': block, 'attention': block.self_attn,
                'mlp': block.mlp, 'head': block.self_attn.o_proj}[component]

    @staticmethod
    def sites(item, direction):
        return item.get('direction_sites', {}).get(direction, item['sites'])

    def validate(self, interventions):
        for item in interventions:
            for direction in ('denoise', 'noise'):
                for site in self.sites(item, direction):
                    if not 0 <= site['layer'] < len(self.layers):
                        raise ValueError(f'Invalid layer {site["layer"]}')
                    if not site['positions'] or any(p >= 0 for p in site['positions']):
                        raise ValueError('Patch sites need negative token offsets')
                    if item['component'] == 'head':
                        if not site['heads'] or any(h < 0 or h >= self.n_heads for h in site['heads']):
                            raise ValueError(f'Invalid head selection: {site["heads"]}')

    def encode(self, prompts, positions):
        from src.model.inference import tokenize_prompt_batch
        enc = tokenize_prompt_batch(self.tokenizer, prompts, self.model.device)
        # Check real lengths, not padded width; silently patching padding is invalid.
        if min(enc['attention_mask'].sum(dim=1)).item() < max(abs(p) for p in positions):
            raise ValueError('Patch window reaches padding in at least one prompt; shorten --positions')
        return enc

    @contextmanager
    def patch(self, item, direction, donor):
        handles = []
        try:
            # One hook per module so cumulative sites in the same layer compose.
            grouped = {}
            for site in self.sites(item, direction):
                grouped.setdefault(site['layer'], []).append(site)
            for layer, sites in grouped.items():
                component = item['component']
                source, cached_positions = donor[(component, layer)]
                def replace(x, sites=sites, source=source, cached_positions=cached_positions, component=component):
                    for site in sites:
                        indices = [cached_positions.index(p) for p in site['positions']]
                        value = source[:, indices].to(device=x.device, dtype=x.dtype)
                        if component == 'head':
                            # x is [batch, token, n_heads * head_dim]. Replace
                            # only this head's feature slice, not attention
                            # probabilities, Q/K/V tensors, or model weights.
                            for head in site['heads']:
                                cols = slice(head * self.head_dim, (head + 1) * self.head_dim)
                                x[:, site['positions'], cols] = value[:, :, cols]
                        else:
                            # For residual interventions this is the entire
                            # d_model vector at each selected token position.
                            x[:, site['positions'], :] = value
                # Fire on the first call only. During generation this is the prompt
                # pass, never a later KV-cache token or repeated full-prefix pass.
                fired = [False]
                if component == 'head':
                    def pre(_module, inputs, replace=replace, fired=fired):
                        if not fired[0]:
                            replace(inputs[0])
                            fired[0] = True
                    handle = self.module(component, layer).register_forward_pre_hook(pre)
                else:
                    def post(_module, inputs, output, replace=replace, fired=fired):
                        if not fired[0]:
                            replace(output[0] if isinstance(output, tuple) else output)
                            fired[0] = True
                        return output
                    handle = self.module(component, layer).register_forward_hook(post)
                handles.append(handle)
            yield
        finally:
            for handle in handles:
                handle.remove()

    def capture(self, enc, requests):
        cache, handles = {}, []
        try:
            for (component, layer), positions in requests.items():
                def save(x, key=(component, layer), positions=positions):
                    cache[key] = (x[:, positions].detach().clone(), positions)
                if component == 'head':
                    def pre(_module, inputs, save=save):
                        save(inputs[0])
                    handles.append(self.module(component, layer).register_forward_pre_hook(pre))
                else:
                    def post(_module, inputs, output, save=save):
                        save(output[0] if isinstance(output, tuple) else output)
                        return output
                    handles.append(self.module(component, layer).register_forward_hook(post))
            self.model(**enc, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        return cache

    def run(self, cfg, rows, variants, interventions, *, batch_size=8,
            direction='both', row_filter='logitdiff', rotation=False,
            patch_source='donor', mean_rows=None, on_variant=None,
            rotation_position='best'):
        import torch
        from src.prompt.chat_template import apply_chat_template_split
        self.validate(interventions)
        if patch_source not in ('donor', 'mean'):
            raise ValueError("patch_source must be 'donor' or 'mean'")
        if patch_source == 'mean' and not mean_rows:
            raise ValueError('Mean patching requires a nonempty reference dataset')
        directions = ['denoise', 'noise'] if direction == 'both' else [direction]
        labels = [int(row.label) for row in rows]
        requests = {}
        for item in interventions:
            for d in directions:
                for site in self.sites(item, d):
                    key = (item['component'], site['layer'])
                    requests[key] = sorted(set(requests.get(key, []) + site['positions']))
        positions = sorted({p for ps in requests.values() for p in ps})
        if not positions or not rows or not variants or batch_size < 1:
            raise ValueError('Run requires rows, variants, interventions, and a positive batch size')
        probe = apply_chat_template_split(cfg, rows[:1], self.tokenizer, condition_type='control', system_prompt_id=variants[0])[0]
        answer_ids = [first_answer_token_id(self.tokenizer, ''.join(probe), probe[1], mapping[cfg.task])
                      for mapping in (cfg.gt_0_label_map, cfg.gt_1_label_map)]
        if answer_ids[0] == answer_ids[1]:
            raise ValueError('Both answers map to the same token')
        result = {'schema_version': 2, 'task': cfg.task, 'dataset': cfg.dataset,
                  'model_alias': cfg.model_alias, 'model_path': cfg.model_path,
                  'n_layers': len(self.layers), 'n_heads': self.n_heads,
                  'row_indices': [r.index for r in rows], 'n_rows_total': len(rows),
                  'row_filter': row_filter, 'answer_token_ids': answer_ids,
                  'interventions': interventions, 'variants': [],
                  'patch_source': patch_source,
                  'mean_reference_count': len(mean_rows) if patch_source == 'mean' else None}
        if rotation:
            from src.cache_activations.hooks import generate_with_multi_pos_cache, position_labels
            result['rotation_metric'] = 'largest_drop_any_early_to_late'
            result['position_labels'] = position_labels(cfg.n_pre_prompt_tokens, cfg.n_generated_tokens)
            if rotation_position != 'best' and rotation_position not in result['position_labels']:
                raise ValueError(f'Unknown rotation position {rotation_position!r}; choose best or one of {result["position_labels"]}')
            result['rotation_position'] = rotation_position

        def score(enc, labs):
            logits = self.model(**enc, use_cache=False).logits[:, -1]
            return honesty(logits, labs, *answer_ids)

        def generated_cache(pairs):
            # Also returns the decoded completions so callers can archive
            # actual generated text alongside the hidden-state cache.
            return generate_with_multi_pos_cache(
                self.model, self.tokenizer, pairs, n_pre=cfg.n_pre_prompt_tokens,
                n_gen=cfg.n_generated_tokens, batch_size=len(pairs),
                max_new_tokens=cfg.max_new_tokens, desc='patched rotation')

        with torch.no_grad():
            for variant in variants:
                pairs = {c: apply_chat_template_split(cfg, rows, self.tokenizer, condition_type=c, system_prompt_id=variant)
                         for c in ('control', 'experiment')}
                base = {c: [] for c in pairs}
                # First pass establishes one global row mask, independent of batching.
                for start in range(0, len(rows), batch_size):
                    sl = slice(start, start + batch_size)
                    for condition in pairs:
                        enc = self.encode([''.join(p) for p in pairs[condition][sl]], positions)
                        base[condition].append(score(enc, labels[sl]))
                control, experiment = [torch.cat(base[c]) for c in ('control', 'experiment')]
                mask, fallback = select_rows(control, experiment, row_filter)
                print(f'[variant {variant}] kept={int(mask.sum())}/{len(rows)} fallback={fallback}', flush=True)
                patched = {(i, d): [] for i in range(len(interventions)) for d in directions}
                base_rotation = {c: ContrastAccumulator() for c in pairs} if rotation else {}
                patch_rotation = {(i, d): ContrastAccumulator() for i, d in patched} if rotation else {}
                # Completions only exist where generation already runs (rotation
                # requires greedy-generating to build the hidden-state cache);
                # behavior-only runs never generate text.
                completions = {'unpatched': {c: [] for c in pairs},
                               'patched': {key: [] for key in patched}} if rotation else None
                if patch_source == 'mean':
                    # Build a fixed reference tensor before evaluating any
                    # intervention. The mean is over examples only, separately
                    # for every condition/component/layer/position request.
                    # The reference includes BOTH GT classes, without the
                    # behavioral row filter. Recompute for each prompt variant;
                    # never average over variants, heads, or feature dimensions.
                    mean_sums, mean_counts, mean_positions = {}, {}, {}
                    reference_pairs = {c: apply_chat_template_split(
                        cfg, mean_rows, self.tokenizer, condition_type=c,
                        system_prompt_id=variant) for c in ('control', 'experiment')}
                    for start in range(0, len(mean_rows), batch_size):
                        sl = slice(start, start + batch_size)
                        encs_mean = {c: self.encode([''.join(p) for p in ps[sl]], positions)
                                     for c, ps in reference_pairs.items()}
                        captures = {c: self.capture(enc, requests) for c, enc in encs_mean.items()}
                        for condition, cache in captures.items():
                            for key, (values, cached_positions) in cache.items():
                                values = values.float()
                                mean_sums[condition, key] = mean_sums.get((condition, key), torch.zeros_like(values[0])) + values.sum(dim=0)
                                mean_counts[condition, key] = mean_counts.get((condition, key), 0) + values.shape[0]
                                mean_positions[condition, key] = cached_positions
                    mean_donors = {
                        c: {key: (total.div(mean_counts[c, key]).unsqueeze(0), mean_positions[c, key])
                            for (cc, key), total in mean_sums.items() if cc == c}
                        for c in pairs
                    }
                    # Leading dimension 1 broadcasts the same reference mean
                    # to every recipient example. Standard activation patching preserves
                    # one-to-one correspondence between statement rows.
                for start in range(0, len(rows), batch_size):
                    sl = slice(start, start + batch_size)
                    encs = {c: self.encode([''.join(p) for p in ps[sl]], positions) for c, ps in pairs.items()}
                    donors = {c: self.capture(enc, requests) for c, enc in encs.items()} if patch_source == 'donor' else None
                    active_donors = donors if patch_source == 'donor' else mean_donors
                    if rotation:
                        for c in pairs:
                            responses, cache = generated_cache(pairs[c][sl])
                            base_rotation[c].add(cache, labels[sl], mask[sl])
                            completions['unpatched'][c].extend(
                                {'row_index': row.index, 'response': response}
                                for row, response in zip(rows[sl], responses))
                    for i, item in enumerate(interventions):
                        for d in directions:
                            # denoise: honest activations -> lying run;
                            # noise: lying activations -> honest run.
                            donor, recipient = ('control', 'experiment') if d == 'denoise' else ('experiment', 'control')
                            with self.patch(item, d, active_donors[donor]):
                                patched[i, d].append(score(encs[recipient], labels[sl]))
                            if rotation:
                                with self.patch(item, d, active_donors[donor]):
                                    responses, cache = generated_cache(pairs[recipient][sl])
                                patch_rotation[i, d].add(cache, labels[sl], mask[sl])
                                completions['patched'][i, d].extend(
                                    {'row_index': row.index, 'response': response}
                                    for row, response in zip(rows[sl], responses))
                    del donors, encs
                entry = {'system_prompt_id': variant, 'n_rows_kept': int(mask.sum()),
                         'filter_fallback': fallback, 'kept_row_indices': [r.index for r, keep in zip(rows, mask) if keep],
                         'logit_diff_control': float(control[mask].mean()),
                         'logit_diff_experiment': float(experiment[mask].mean()), 'effects': []}
                if rotation:
                    baseline_cosine = contrast_cosine(base_rotation['control'], base_rotation['experiment'])
                    # Historical default: choose the largest-drop token using
                    # only UNPATCHED runs. Keep that measurement token fixed for
                    # every intervention and both directions within this variant.
                    # An explicit rotation_position overrides this selection;
                    # neither choice changes which tokens receive the patch.
                    best_value, best_pos = best_rotation(baseline_cosine)
                    measured_pos = best_pos if rotation_position == 'best' else result['position_labels'].index(rotation_position)
                    baseline = early_to_late_drop(baseline_cosine[:, measured_pos])
                    entry.update(baseline_rotation=baseline, baseline_best_rotation=best_value,
                                 baseline_best_position_index=best_pos,
                                 baseline_best_position_label=result['position_labels'][best_pos],
                                 rotation_position_index=measured_pos,
                                 rotation_position_label=result['position_labels'][measured_pos],
                                 baseline_rotation_by_position=[early_to_late_drop(baseline_cosine[:, p]) for p in range(baseline_cosine.shape[1])])
                    # Decoded completions, indexed the same way as `effects`:
                    # completions['patched'][i][direction] <-> interventions[i].
                    entry['completions'] = {
                        'unpatched': completions['unpatched'],
                        'patched': [{d: completions['patched'][i, d] for d in directions}
                                    for i in range(len(interventions))],
                    }
                for i, item in enumerate(interventions):
                    effect = {}
                    for d in directions:
                        scores = behavior_scores(control, experiment, torch.cat(patched[i, d]), mask, d)
                        if rotation:
                            left, right = (base_rotation['control'], patch_rotation[i, d]) if d == 'denoise' else (patch_rotation[i, d], base_rotation['experiment'])
                            cosine = contrast_cosine(left, right)
                            by_position = [early_to_late_drop(cosine[:, p]) for p in range(cosine.shape[1])]
                            value = by_position[measured_pos]
                            # Positive delta = LESS rotation after patching.
                            # The earlier/later layer pair is re-maximized, but
                            # the measurement token stays the same as baseline.
                            scores.update(patched_rotation=value, rotation_delta=baseline - value,
                                          patched_rotation_by_position=by_position)
                        effect[d] = scores
                    entry['effects'].append(effect)
                result['variants'].append(entry)
                if on_variant:
                    on_variant(result)
        return result
