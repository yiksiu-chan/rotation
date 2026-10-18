"""Behavior-only single-head mean sweeps with immutable-prefix KV reuse.

Each intervention has its own batch slice and cache. Only tokens at or after
its patch position are recomputed; no changed KV state is reused between heads.
"""
from contextlib import contextmanager
import time
import torch
from transformers import DynamicCache
from .core import PatchingEngine, first_answer_token_id
from .metrics import honesty, select_rows, behavior_scores


class FastHeadEngine(PatchingEngine):
    @contextmanager
    def batched_patch(self, items, mean, batch):
        handles = []
        grouped = {}
        for j, item in enumerate(items):
            site = item['sites'][0]
            grouped.setdefault(site['layer'], []).append((j, site))
        try:
            for layer, sites in grouped.items():
                def patch(_module, inputs, sites=sites, layer=layer):
                    x = inputs[0]
                    source, positions = mean['head', layer]
                    for j, site in sites:
                        h = site['heads'][0]
                        cols = slice(h*self.head_dim, (h+1)*self.head_dim)
                        for pos in site['positions']:
                            x[j*batch:(j+1)*batch, pos, cols] = source[0, positions.index(pos), cols].to(x.dtype)
                handles.append(self.module('head', layer).register_forward_pre_hook(patch))
            yield
        finally:
            for handle in handles:
                handle.remove()

    @torch.no_grad()
    def run(self, cfg, rows, variants, interventions, *, batch_size=20,
            head_batch_size=8, direction='both', row_filter='none', on_variant=None):
        from src.prompt.chat_template import apply_chat_template_split
        self.validate(interventions)
        if not rows or not interventions or batch_size < 1 or head_batch_size < 1:
            raise ValueError('Nonempty rows/interventions and positive batch sizes required')
        for item in interventions:
            if item['component'] != 'head' or len(item['sites']) != 1 or item.get('direction_sites'):
                raise ValueError('Fast path supports plain single-site head interventions only')
            if len(item['sites'][0]['heads']) != 1:
                raise ValueError('Each intervention must patch exactly one head')
        requests = {}
        for item in interventions:
            s = item['sites'][0]
            key = ('head', s['layer'])
            requests[key] = sorted(set(requests.get(key, []) + s['positions']))
        positions = sorted({p for ps in requests.values() for p in ps})
        suffix = -min(positions)
        directions = ['denoise', 'noise'] if direction == 'both' else [direction]
        labels = [int(r.label) for r in rows]
        probe = apply_chat_template_split(cfg, rows[:1], self.tokenizer, condition_type='control', system_prompt_id=variants[0])[0]
        ids = [first_answer_token_id(self.tokenizer, ''.join(probe), probe[1], m[cfg.task])
               for m in (cfg.gt_0_label_map, cfg.gt_1_label_map)]
        if ids[0] == ids[1]:
            raise ValueError('Answer tokens must differ')
        result = dict(schema_version=2, task=cfg.task, dataset=cfg.dataset,
                      model_alias=cfg.model_alias, model_path=cfg.model_path,
                      n_layers=len(self.layers), n_heads=self.n_heads,
                      row_indices=[r.index for r in rows], n_rows_total=len(rows),
                      row_filter=row_filter, answer_token_ids=ids, interventions=interventions,
                      patch_source='mean', mean_reference_count=len(rows), variants=[])
        for variant in variants:
            started = time.monotonic()
            pairs = {c: apply_chat_template_split(cfg, rows, self.tokenizer, condition_type=c,
                                                 system_prompt_id=variant)
                     for c in ('control', 'experiment')}
            means, baseline = {}, {}
            # Capture all requested layer means together; no vocabulary projection needed.
            for condition in pairs:
                sums = {}
                for start in range(0, len(rows), batch_size):
                    enc = self.encode([''.join(p) for p in pairs[condition][start:start+batch_size]], positions)
                    handles = []
                    try:
                        for key, ps in requests.items():
                            def save(_module, inputs, key=key, ps=ps):
                                values = inputs[0][:, ps].float().sum(0, keepdim=True)
                                sums[key] = sums.get(key, 0) + values
                            handles.append(self.module(*key).register_forward_pre_hook(save))
                        self.model.model(**enc, use_cache=False)
                    finally:
                        for h in handles: h.remove()
                means[condition] = {key: (value/len(rows), requests[key]) for key, value in sums.items()}
                baseline[condition] = []
            patched = {(i, d): [] for i in range(len(interventions)) for d in directions}
            for start in range(0, len(rows), batch_size):
                labs = labels[start:start+batch_size]
                batch = len(labs)
                for recipient in pairs:
                    d = 'noise' if recipient == 'control' else 'denoise'
                    donor = 'experiment' if recipient == 'control' else 'control'
                    enc = self.encode([''.join(p) for p in pairs[recipient][start:start+batch_size]], positions)
                    width = enc['input_ids'].shape[1]
                    prefix_len = width - suffix
                    # Match the original full-forward absolute padded positions.
                    prefix = None
                    if prefix_len:
                        prefix = self.model.model(input_ids=enc['input_ids'][:, :prefix_len],
                            attention_mask=enc['attention_mask'][:, :prefix_len], use_cache=True).past_key_values
                    count = min(head_batch_size, len(interventions))
                    cache = DynamicCache() if prefix is None else DynamicCache(
                        [(k.repeat(count, 1, 1, 1), v.repeat(count, 1, 1, 1)) for k, v in prefix])
                    inputs = dict(
                        input_ids=enc['input_ids'][:, prefix_len:].repeat(count, 1),
                        attention_mask=enc['attention_mask'].repeat(count, 1),
                        past_key_values=cache, use_cache=True, logits_to_keep=1,
                        cache_position=torch.arange(prefix_len, width, device=self.model.device))
                    # Match the baseline and intervention kernel shapes exactly.
                    # Otherwise BF16 rounding creates apparent effects even for
                    # causally irrelevant last-layer heads at position -2.
                    logits = self.model(**inputs).logits[:, -1]
                    baseline[recipient].append(honesty(logits[:batch], labs, *ids))
                    if d in directions:
                        for offset in range(0, len(interventions), head_batch_size):
                            items = interventions[offset:offset+head_batch_size]
                            padded_items = items + [items[-1]] * (count-len(items))
                            cache.crop(prefix_len)
                            with self.batched_patch(padded_items, means[donor], batch):
                                logits = self.model(**inputs).logits[:, -1]
                            scores = honesty(logits, labs*count, *ids).reshape(count, batch)
                            for j in range(len(items)): patched[offset+j, d].append(scores[j])
                    del cache, prefix, inputs
            baseline = {c: torch.cat(v) for c, v in baseline.items()}
            control, experiment = baseline['control'], baseline['experiment']
            mask, fallback = select_rows(control, experiment, row_filter)
            entry = dict(system_prompt_id=variant, n_rows_kept=int(mask.sum()), filter_fallback=fallback,
                         kept_row_indices=[r.index for r, keep in zip(rows, mask) if keep],
                         logit_diff_control=float(control[mask].mean()),
                         logit_diff_experiment=float(experiment[mask].mean()), effects=[],
                         elapsed_seconds=time.monotonic()-started)
            for i in range(len(interventions)):
                entry['effects'].append({d: behavior_scores(control, experiment,
                    torch.cat(patched[i,d]), mask, d) for d in directions})
            result['variants'].append(entry)
            print(f'FAST variant={variant} rows={int(mask.sum())} heads={len(interventions)} seconds={entry["elapsed_seconds"]:.2f}', flush=True)
            if on_variant: on_variant(result)
        return result
