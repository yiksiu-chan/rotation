"""Versioned results, legacy readers, site ranking, and table aggregation.

Every result uses an intervention list and a matching effects list per variant.
Coordinates remain explicit, including direction-specific ranked sites. Old
rebuttal JSONs are converted in memory; source files are never rewritten.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import importlib.metadata
import json
import math
import random
import subprocess
import numpy as np

from .core import make_intervention


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    """Atomic replacement so interrupted runs leave the previous checkpoint readable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(_json_safe(value), indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def write_csv(path, rows):
    """Write a list of flat dicts as CSV, columns taken from the first row."""
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_config(output, cfg, arguments):
    from src.paths import PROJECT_ROOT
    versions = {}
    for name in ('torch', 'transformers', 'accelerate', 'datasets', 'numpy', 'matplotlib'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT, text=True, capture_output=True).stdout.strip()
        dirty = bool(subprocess.run(['git', 'status', '--porcelain'], cwd=PROJECT_ROOT, text=True, capture_output=True).stdout.strip())
    except OSError:
        revision, dirty = None, None
    write_json(Path(output) / 'config.json', {'config': asdict(cfg), 'arguments': vars(arguments),
                                             'versions': versions, 'git_commit': revision, 'git_dirty': dirty})


def load_results(path):
    data = json.loads(Path(path).read_text())
    if data.get('schema_version') == 2:
        return data
    if 'schema_version' in data:
        raise ValueError(f'Unsupported result schema in {path}: {data["schema_version"]}')
    return convert_legacy(data)


def convert_legacy(data):
    """Convert all six historical experiment schemas, preserving metric meanings."""
    if not data.get('variants'):
        raise ValueError('Expected patching results with at least one variant')
    result = {k: deepcopy(v) for k, v in data.items() if k != 'variants'}
    result.update(schema_version=2, variants=[], interventions=[], legacy=True)
    component = data.get('component', 'resid')
    positions = data.get('patch_positions', data.get('window_positions', [-1]))
    first = data['variants'][0]
    indices = []
    if component == 'patching_rotation_metrics':
        for row in first['interventions']:
            direction_sites = {}
            comp = 'head' if row['kind'] == 'head_sites' else 'residual'
            for d in ('denoise', 'noise'):
                payload = row[d]
                direction_sites[d] = ([{'layer': s['layer'], 'heads': [s['head']], 'positions': positions} for s in payload['sites']]
                                      if comp == 'head' else [{'layer': payload['layer'], 'positions': [payload['position']], 'heads': None}])
            coords = {'k': len(direction_sites['denoise'])} if comp == 'head' else {'layer': row['denoise']['layer']}
            result['interventions'].append(make_intervention(comp, direction_sites['denoise'], direction_sites=direction_sites, name=row['name'], **coords))
    elif component == 'head_cumulative':
        for row in first['cumulative']:
            selected = row.get('sites', data['sites'][:row['k']])
            sites = [{'layer': s['layer'], 'heads': [s['head']], 'positions': positions} for s in selected]
            result['interventions'].append(make_intervention('head', sites, k=row['k']))
    elif 'curves' in first:
        direction = next(iter(first['curves']))
        for row in first['curves'][direction]:
            comp = {'resid': 'residual', 'attn': 'attention'}.get(component, component)
            result['interventions'].append(make_intervention(comp, [{'layer': row['layer'], 'heads': None, 'positions': positions}], layer=row['layer']))
    else:
        shape = np.asarray(first['denoise']).shape
        for idx in np.ndindex(shape):
            layer = data.get('layers', list(range(shape[0])))[idx[0]]
            head = idx[1] if component in ('head', 'head_position') else None
            selected_positions = ([positions[idx[-1]]] if component in ('resid_position', 'head_position') else positions)
            comp = 'head' if head is not None else 'residual'
            coords = {'layer': layer}
            if head is not None:
                coords['head'] = head
            if len(selected_positions) == 1:
                coords['position'] = selected_positions[0]
            result['interventions'].append(make_intervention(comp, [{'layer': layer, 'heads': [head] if head is not None else None, 'positions': selected_positions}], **coords))
            indices.append(idx)
    for variant in data['variants']:
        entry = {k: deepcopy(v) for k, v in variant.items() if k in ('system_prompt_id', 'n_rows_kept', 'logit_diff_control', 'logit_diff_experiment', 'baseline_rotation', 'baseline_best_position_index', 'baseline_best_position_label')}
        entry['effects'] = []
        for i, item in enumerate(result['interventions']):
            effect = {}
            for d in ('denoise', 'noise'):
                metrics = {}
                if component == 'patching_rotation_metrics':
                    payload = variant['interventions'][i][d]
                    metrics = {k: payload[k] for k in ('flip_rate', 'patched_rotation', 'rotation_delta') if k in payload}
                elif component == 'head_cumulative':
                    payload = variant['cumulative'][i]
                    mapping = {'recovery': f'{d}_recovery_fraction' if d == 'denoise' else 'noise_flip_fraction',
                               'logit_diff': f'{d}_logit_diff', 'flip_rate': f'{d}_flip_rate'}
                    metrics = {name: payload[key] for name, key in mapping.items() if key in payload}
                elif 'curves' in variant:
                    if d not in variant['curves']:
                        continue
                    payload = variant['curves'][d][i]
                    metrics = {'logit_diff': payload['patched_logit_diff'],
                               'recovery': payload['recovery_fraction' if d == 'denoise' else 'flip_fraction']}
                else:
                    for name, key in [('recovery', d), ('logit_diff', d+'_logit_diff'), ('flip_rate', d+'_flip_rate')]:
                        if key in variant:
                            metrics[name] = float(np.asarray(variant[key])[indices[i]])
                start = variant.get('logit_diff_experiment' if d == 'denoise' else 'logit_diff_control')
                end = variant.get('logit_diff_control' if d == 'denoise' else 'logit_diff_experiment')
                if 'logit_diff' not in metrics and 'recovery' in metrics and start is not None:
                    metrics['logit_diff'] = start + metrics['recovery'] * (end - start)
                if 'logit_diff' in metrics and start is not None:
                    metrics['logit_shift'] = metrics['logit_diff'] - start
                effect[d] = metrics
            entry['effects'].append(effect)
        result['variants'].append(entry)
    return _json_safe(result)


def rank_heads(data, metric='recovery', direction='denoise', *, layers=None, heads=None):
    """Mean across variants, with the original descending numpy argsort tie order."""
    # One global head ordering across source variants, not a separate ordering
    # per statement or per variant. The old rotation figures used flip_rate,
    # ranked independently for noise and denoise (not this function's defaults).
    candidates, scores = [], []
    for i, item in enumerate(data['interventions']):
        if item['component'] != 'head' or 'k' in item or len(item['sites']) != 1:
            continue
        site = item['sites'][0]
        if not site['heads'] or len(site['heads']) != 1:
            continue
        if layers is not None and site['layer'] not in layers:
            continue
        if heads is not None and site['heads'][0] not in heads:
            continue
        directions = ['denoise', 'noise'] if direction == 'bidirectional' else [direction]
        values = [v['effects'][i].get(d, {}).get(metric) for v in data['variants'] for d in directions]
        if not values or any(x is None or not np.isfinite(x) for x in values):
            raise ValueError(f'Ranking needs finite {metric} for every selected variant/direction')
        candidates.append(site)
        scores.append(float(np.mean(values)))
    keys = [(s['layer'], s['heads'][0]) for s in candidates]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError('Rank from a single-head window sweep, with one entry per layer/head')
    order = np.argsort(scores)[::-1]
    return [deepcopy(candidates[i]) for i in order]


def cumulative_interventions(data, k_values, positions, *, metric='recovery', direction='denoise', layers=None, heads=None):
    # Top-k heads are patched SIMULTANEOUSLY in one recipient run. Their joint
    # effect is measured directly, not the sum of their single-head effects.
    ranked = rank_heads(data, metric, direction, layers=layers, heads=heads)
    if max(k_values) > len(ranked):
        raise ValueError(f'Requested top-k exceeds {len(ranked)} ranked heads')
    for site in ranked:
        if site['positions'] != positions:
            raise ValueError('Rank-source token positions differ from --positions')
    return [make_intervention('head', deepcopy(ranked[:k]), k=k) for k in k_values]


def uniform_cumulative_head_interventions(layers, heads, k_values, positions, *, repeats=1, seed=0):
    """Uniform nested random sets, independent of ranking scores or k ordering.

    A repeat uses prefixes of one permutation of the canonical layer/head pool.
    Ranked heads remain eligible; sampling is without replacement within a set.
    """
    pool = [(layer, head) for layer in sorted(set(layers)) for head in sorted(set(heads))]
    if repeats < 1 or not k_values or min(k_values) < 1 or max(k_values) > len(pool):
        raise ValueError('Need positive repeats and k values within the eligible head count')
    result = []
    for draw in range(repeats):
        permutation = random.Random(seed + draw).sample(pool, len(pool))
        # With repeats=1 and k=1..10, select ten distinct heads ONCE, then patch
        # successive prefixes. Their layers are unconstrained. The caller reuses
        # these exact sites for all examples and all prompt variants.
        for k in k_values:
            sites = [{'layer': layer, 'heads': [head], 'positions': list(positions)}
                     for layer, head in permutation[:k]]
            result.append(make_intervention('head', sites, k=k, random_draw=draw,
                                           random_seed=seed + draw, random_match='global',
                                           random_nested=True, positions=list(positions)))
    return result


def random_head_interventions(data, k_values, positions, *, repeats=1, seed=0,
                              match='layer', metric='recovery', direction='denoise',
                              layers=None, heads=None):
    """Build seeded random head sets using the same pool as a ranked sweep.

    ``match='layer'`` preserves the layer histogram of the ranked top-k set;
    ``global`` samples uniformly from all eligible heads. Each draw is fixed
    across variants, and heads are sampled without replacement.
    """
    # Older configurable control: independently sample at each k and optionally
    # match layer counts. The current one-draw comparison instead calls
    # uniform_cumulative_head_interventions above (uniform, nested, no ranking).
    if repeats < 1 or match not in ('layer', 'global'):
        raise ValueError("repeats must be positive and match must be 'layer' or 'global'")
    ranked = rank_heads(data, metric, direction, layers=layers, heads=heads)
    candidates = [deepcopy(site) for site in ranked]
    if not candidates:
        raise ValueError('Random head control needs a single-head sweep source')
    if any(site['positions'] != positions for site in candidates):
        raise ValueError('Rank source token positions differ from the requested positions')
    result = []
    for draw in range(repeats):
        rng = random.Random(seed + draw)
        for k in k_values:
            if k > len(candidates):
                raise ValueError(f'Requested random top-k {k} exceeds {len(candidates)} eligible heads')
            if match == 'global':
                selected = rng.sample(candidates, k)
            else:
                target = ranked[:k]
                counts = {}
                for site in target:
                    counts[site['layer']] = counts.get(site['layer'], 0) + 1
                by_layer = {}
                for site in candidates:
                    by_layer.setdefault(site['layer'], []).append(site)
                selected = []
                for layer, count in counts.items():
                    pool = by_layer.get(layer, [])
                    if count > len(pool):
                        raise ValueError(f'Not enough candidate heads in layer {layer} for matched random draw')
                    selected.extend(rng.sample(pool, count))
                selected.sort(key=lambda s: (s['layer'], s['heads'][0]))
            result.append(make_intervention(
                'head', selected, k=k, random_draw=draw,
                random_seed=seed + draw, random_match=match,
                positions=positions))
    return result


def select_rotation_interventions(residual, head, mode, k_values, positions, *, include_final=False, layers=None, heads=None):
    """Historical direction-specific flip-rate ranking for the joint experiment."""
    selected = []
    candidates = [(i, item) for i, item in enumerate(residual['interventions'])
                  if item['component'] == 'residual' and len(item['sites']) == 1
                  and len(item['sites'][0]['positions']) == 1
                  and item['sites'][0]['positions'][0] in positions
                  and (include_final or item['sites'][0]['layer'] != residual['n_layers'] - 1)
                  and (layers is None or item['sites'][0]['layer'] in layers)]
    if mode != 'none' and not candidates:
        raise ValueError('No eligible residual sites in ranking source')
    if mode == 'best':
        groups = [candidates]
    elif mode == 'all_layers':
        groups = [[c for c in candidates if c[1]['sites'][0]['layer'] == layer]
                  for layer in sorted({c[1]['sites'][0]['layer'] for c in candidates})]
    elif mode == 'all_layer_positions':
        groups = [[c] for c in candidates]
    elif mode == 'none':
        groups = []
    else:
        raise ValueError(f'Unknown residual mode {mode}')
    for group in groups:
        direction_sites = {}
        for direction in ('denoise', 'noise'):
            values = []
            for i, _ in group:
                scores = [v['effects'][i].get(direction, {}).get('flip_rate') for v in residual['variants']]
                if any(x is None or not np.isfinite(x) for x in scores):
                    raise ValueError('Residual ranking requires finite flip_rate in both directions')
                values.append(np.mean(scores))
            item = group[int(np.argmax(values))][1]
            direction_sites[direction] = deepcopy(item['sites'])
        coords = {'layer': direction_sites['denoise'][0]['layer']} if mode != 'best' else {}
        if mode == 'all_layer_positions':
            coords['position'] = direction_sites['denoise'][0]['positions'][0]
        selected.append(make_intervention('residual', direction_sites['denoise'], direction_sites=direction_sites, **coords))
    if k_values:
        ranks = {d: rank_heads(head, 'flip_rate', d, layers=layers, heads=heads) for d in ('denoise', 'noise')}
        if max(k_values) > min(map(len, ranks.values())):
            raise ValueError('Requested top-k exceeds ranked heads')
        if any(s['positions'] != positions for sites in ranks.values() for s in sites):
            raise ValueError('Head ranking positions differ from --positions')
        for k in k_values:
            sites = {d: deepcopy(ranks[d][:k]) for d in ranks}
            selected.append(make_intervention('head', sites['denoise'], direction_sites=sites, k=k))
    return selected


def at_rotation_position(data, label):
    """Select a measured token without rerunning, when per-position scores exist.

    Historical scalar-only results cannot be relabeled as a different token.
    Patch sites and behavior scores are unaffected by this selection.
    """
    result = deepcopy(data)
    if label not in data.get('position_labels', []):
        raise ValueError(f'Rotation position {label!r} is not present in these results')
    pos = data['position_labels'].index(label)
    for variant in result['variants']:
        baseline = variant.get('baseline_rotation_by_position')
        if baseline is None:
            recorded = variant.get('rotation_position_label', variant.get('baseline_best_position_label'))
            if recorded != label:
                raise ValueError(f'Results only store rotation at {recorded!r}; rerun to measure {label!r}')
        else:
            variant['baseline_rotation'] = baseline[pos]
            for effect in variant['effects']:
                for scores in effect.values():
                    if 'patched_rotation_by_position' not in scores:
                        raise ValueError('Missing per-position patched rotation; rerun this experiment')
                    scores['patched_rotation'] = scores['patched_rotation_by_position'][pos]
                    scores['rotation_delta'] = baseline[pos] - scores['patched_rotation']
        variant.update(rotation_position_index=pos, rotation_position_label=label)
    result['rotation_position'] = label
    return result


def summary_rows(data, directions=('denoise', 'noise'), variants=None):
    rows = []
    chosen = [v for v in data['variants'] if variants is None or v['system_prompt_id'] in variants]
    if not chosen:
        raise ValueError('No matching variants')
    for i, item in enumerate(data['interventions']):
        for direction in directions:
            metrics = {key for v in chosen for key in v['effects'][i].get(direction, {})}
            metrics.discard('patched_rotation_by_position')
            for metric in sorted(metrics):
                values = [v['effects'][i].get(direction, {}).get(metric) for v in chosen]
                finite = [x for x in values if x is not None and np.isfinite(x)]
                sites = item.get('direction_sites', {}).get(direction, item['sites'])
                coords = {k: item.get(k) for k in ('layer', 'head', 'position', 'k')}
                # Direction-specific selections may use different layers/positions.
                if len(sites) == 1:
                    coords['layer'] = sites[0]['layer']
                    if len(sites[0]['positions']) == 1:
                        coords['position'] = sites[0]['positions'][0]
                rows.append(dict(intervention=i, component=item['component'], direction=direction,
                                 metric=metric, **coords, sites=sites,
                                 mean=float(np.mean(finite)) if finite else None,
                                 std=float(np.std(finite)) if finite else None,
                                 n_valid=len(finite), n_variants=len(chosen)))
    return rows


def merge_results(paths):
    blobs = [load_results(path) for path in paths]
    merged = deepcopy(blobs[0])
    if any(b.get('status') == 'partial' for b in blobs):
        raise ValueError('Cannot merge an incomplete shard')
    for key in ('task', 'dataset', 'model_alias', 'model_path', 'n_layers', 'n_heads', 'interventions', 'row_indices', 'row_seed', 'n_rows_total', 'row_filter', 'settings', 'rotation_position', 'position_labels'):
        if any(b.get(key) != merged.get(key) for b in blobs[1:]):
            raise ValueError(f'Shards disagree on {key}')
    seen, variants = set(), []
    for blob in blobs:
        for variant in blob['variants']:
            index = variant['system_prompt_id']
            if index in seen:
                raise ValueError(f'Duplicate variant in shards: {index}')
            seen.add(index)
            variants.append(variant)
    merged.update(variants=sorted(variants, key=lambda v: v['system_prompt_id']),
                  requested_variants=sorted(seen), merged_from=[str(p) for p in paths], status='complete')
    return merged
