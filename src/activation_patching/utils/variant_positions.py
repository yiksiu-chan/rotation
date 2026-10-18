"""Frozen per-prompt patch positions and explicit heterogeneous sweep results."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path


def resolve_input_position(tokenizer, pairs, label):
    """Verify that the frozen label identifies the same input offset and token.

    Shared by step2_patch_residual.py and step4_patch_heads_cumulative.py,
    which both freeze one input token per prompt (from step1's baseline) and
    need to re-locate that exact token/offset under the current tokenizer.
    """
    relative = 0 if label == 'prompt' else int(label.removeprefix('gen+'))
    if label != 'prompt' and not label.startswith('gen+'):
        raise ValueError(f'Unsupported frozen label: {label}')
    offsets, tokens = set(), set()
    for core, prefix in pairs:
        core_ids = tokenizer(core, add_special_tokens=False)['input_ids']
        full_ids = tokenizer(core + prefix, add_special_tokens=False)['input_ids']
        if full_ids[:len(core_ids)] != core_ids:
            raise ValueError('Tokenization changed at the core/prefix boundary')
        index = len(core_ids) - 1 + relative
        if not 0 <= index < len(full_ids):
            raise ValueError('Selected rotation position is not an input token')
        offsets.add(index - len(full_ids))
        tokens.add(full_ids[index])
    if len(offsets) != 1 or len(tokens) != 1:
        raise ValueError('Position offset/token varies across rows; a per-row hook is needed')
    token_id = tokens.pop()
    return offsets.pop(), token_id, tokenizer.decode([token_id])


def load_position_selection(path, variants):
    entries = json.loads(Path(path).read_text())
    if not isinstance(entries, list):
        raise ValueError('Position selection must be a list of prompt records')
    by_id = {}
    for entry in entries:
        key = entry['system_prompt_id']
        if key in by_id:
            raise ValueError(f'Duplicate position selection for prompt {key}')
        by_id[key] = entry
    missing = set(variants) - set(by_id)
    if missing:
        raise ValueError(f'Missing position selections for prompts {sorted(missing)}')
    return {v: by_id[v] for v in variants}


def resolve_selections(cfg, rows, tokenizer, source, path):
    from src.prompt.chat_template import apply_chat_template_split, load_experiment_system_prompts
    prompts = load_experiment_system_prompts(cfg)
    resolved = {}
    for variant, entry in source.items():
        if entry['system_prompt'] != prompts[variant]:
            raise ValueError(f'Prompt {variant} text changed since position selection')
        pairs = [p for condition in ('control', 'experiment') for p in apply_chat_template_split(
            cfg, rows, tokenizer, condition_type=condition, system_prompt_id=variant)]
        offset, token_id, token_text = resolve_input_position(tokenizer, pairs, entry['position_label'])
        resolved[str(variant)] = dict(system_prompt_id=variant,
            system_prompt=entry['system_prompt'], position_label=entry['position_label'],
            input_offset=offset, token_id=token_id, token_text=token_text,
            selection_score=entry.get('selection_score'))
    return dict(source_path=str(Path(path).resolve()),
                source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                variants=resolved, row_policy='Positions only; evaluation sample follows run arguments')


def items_at_position(items, position):
    result = deepcopy(items)
    for item in result:
        item['position'] = position
        item['sites'][0]['positions'] = [position]
    return result


def combine_position_results(chunks, selection):
    """Schema 3 uses head identities plus explicit per-variant site lists.

    Individual variant artifacts remain schema 2. Do not represent mixed-token
    sweeps as schema 2: its consumers assume one shared site list for all prompts.
    """
    first = chunks[0]
    identities = [{k: item[k] for k in ('component', 'layer', 'head')}
                  for item in first['interventions']]
    result = {k: deepcopy(v) for k, v in first.items() if k not in ('variants', 'interventions')}
    result.update(schema_version=3, interventions=identities, variants=[],
                  interventions_by_variant={}, position_selection=selection)
    for chunk in chunks:
        if chunk['row_indices'] != first['row_indices']:
            raise ValueError('Evaluation rows changed between prompts')
        actual = [{k: item[k] for k in ('component', 'layer', 'head')} for item in chunk['interventions']]
        if actual != identities:
            raise ValueError('Head identities/order changed between prompts')
        for variant in chunk['variants']:
            key = str(variant['system_prompt_id'])
            if key in result['interventions_by_variant']:
                raise ValueError(f'Duplicate prompt result {key}')
            result['interventions_by_variant'][key] = chunk['interventions']
            entry = dict(variant, patch_position=selection['variants'][key])
            result['variants'].append(entry)
    return result
