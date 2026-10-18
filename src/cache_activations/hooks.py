"""
Multi-position activation cache helper.

Generates greedy completions while recording, for every transformer block,
hidden states at the last ``n_pre`` core-prompt tokens, the final core-prompt
token (label ``"prompt"`` — the token immediately *before* ``answer_prefix``),
and the next ``n_gen`` post-prompt tokens (labels ``"gen+1"`` … ``"gen+{n_gen}"``).
``n_positions = n_pre + 1 + n_gen`` columns per row.

The "prompt" position is the **last token of the chat-template body** (i.e. of
``core_prompt = body + inner_suffix``), *before* ``answer_prefix`` is appended.
The post-prompt window (``gen+i``) is uniform: the leading ``n_ap`` slots are
populated by the ``answer_prefix`` tokens (read from the prompt forward pass at
indices ``-n_ap .. -1``), and the remaining ``n_gen - n_ap`` slots are
populated by model-generated tokens (one per generation step). For rows where
``n_ap >= n_gen`` the entire post-prompt window is answer_prefix and no
generated activations are stored.

``answer_prefix`` may vary per row (``cfg.answer_prefix`` for most tasks;
``r.meta["answer_prefix"]`` for ``lying_to_achieve_goals``), so the "prompt"
index is computed per row from the difference between the tokenized
``core_prompt`` and the tokenized ``core_prompt + answer_prefix``. The label
``"gen+i"`` is purely positional from the prompt anchor; it does **not** mean
"i-th model-generated token" — for rows where ``n_ap`` differs, the same
``"gen+i"`` label may correspond to an answer_prefix token in one row and a
model-generated token in another.

Token-position layout (column index → meaning):

- ``0 .. n_pre - 1``                "prompt-{n_pre}" .. "prompt-1"
- ``n_pre``                         "prompt"   (last core_prompt token)
- ``n_pre + 1 .. n_pre + n_gen``    "gen+1" .. "gen+{n_gen}"   (answer_prefix
                                    tokens followed by model-generated tokens)

Left padding (set in :func:`src.model.inference.load_model_and_tokenizer`) is
required so per-row indexing from the right edge aligns to real prompt tokens
for every row in the batch, regardless of original prompt length.

Block-module discovery currently assumes a Llama-style ``model.model.layers``
attribute; extend ``_get_block_modules`` for other families when needed.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
from tqdm import tqdm

from src.model.inference import greedy_generate_kwargs, tokenize_prompt_batch


def _get_block_modules(model) -> list:
    """
    Return the list of transformer blocks for hook registration.

    Assumes a Llama-style ``model.model.layers`` attribute. Other model
    families (e.g. GPT-2-style ``model.transformer.h``) are not supported;
    extend this function to add them.
    """
    return list(model.model.layers)


def _count_answer_prefix_tokens(tokenizer, core: str, full: str) -> int:
    """
    Number of tokens that ``answer_prefix`` contributes to ``full = core + ap``.

    Computed as ``len(tokenize(full)) - len(tokenize(core))`` with
    ``add_special_tokens=False`` so the BOS is not double-counted. Clamped to
    0 in the (extremely unlikely) case the tokenizer produces fewer total
    tokens for ``full`` than for ``core``.
    """
    ids_core = tokenizer(core, add_special_tokens=False)["input_ids"]
    ids_full = tokenizer(full, add_special_tokens=False)["input_ids"]
    return max(0, len(ids_full) - len(ids_core))


@torch.no_grad()
def generate_with_multi_pos_cache(
    model,
    tokenizer,
    prompts: Sequence[Tuple[str, str]],
    n_pre: int,
    n_gen: int,
    batch_size: int = 64,
    max_new_tokens: int = 10,
    desc: str = "cache",
) -> Tuple[List[str], torch.Tensor]:
    """
    Greedy-generate one continuation per prompt while caching layer-wise
    hidden states across the ``(n_pre, prompt, n_gen)`` token grid.

    Args:
        model: Causal LM (Llama-3-style block layout).
        tokenizer: Matching tokenizer; must left-pad (caller's responsibility).
        prompts: ``(core_prompt, answer_prefix)`` tuples per row, as produced
            by :func:`src.prompt.chat_template.apply_chat_template_split`. The
            full prompt sent to the model is ``core_prompt + answer_prefix``;
            the split lets the hook locate the "prompt" position (last token
            of ``core_prompt``).
        n_pre: Number of pre-prompt tokens to retain (positions ``0 .. n_pre-1``).
        n_gen: Number of post-prompt tokens to retain (positions
            ``n_pre+1 .. n_pre+n_gen``). The leading ``n_ap`` slots come from
            the answer_prefix; the rest come from generation steps.
        batch_size: Micro-batch size for ``model.generate``.
        max_new_tokens: Greedy generation budget (≥ ``n_gen``).
        desc: tqdm progress label.

    Returns:
        ``(responses, cache)`` where:

        - ``responses`` — decoded continuations (no prompt), one per row.
        - ``cache`` — float16 tensor of shape
          ``(n_layers, len(prompts), n_pre + 1 + n_gen, d_model)``
          on ``model.device``. The caller is responsible for moving /
          permuting / slicing it before persisting.
    """
    blocks = _get_block_modules(model)
    n_layers = len(blocks)
    d_model = model.config.hidden_size
    n_positions = n_pre + 1 + n_gen
    n_total = len(prompts)

    # Per-row answer_prefix token counts and the concatenated prompt strings.
    full_prompt_strs: List[str] = []
    n_ap_per_row: List[int] = []
    for core, ap in prompts:
        full_prompt_strs.append(core + ap)
        n_ap_per_row.append(_count_answer_prefix_tokens(tokenizer, core, core + ap))

    cache = torch.zeros(
        (n_layers, n_total, n_positions, d_model),
        dtype=torch.float16,
        device=model.device,
    )
    responses: List[str] = []
    sample_offset = 0

    for batch_start in tqdm(range(0, n_total, batch_size), desc=desc):
        batch_full = full_prompt_strs[batch_start : batch_start + batch_size]
        batch_n_ap = n_ap_per_row[batch_start : batch_start + batch_size]
        enc = tokenize_prompt_batch(tokenizer, batch_full, model.device)
        n_in_batch = enc["input_ids"].shape[0]
        len_prompt = enc["input_ids"].shape[1]

        # Counter is incremented by an embedding-layer hook on every forward pass,
        # so block hooks can tell prompt-pass (counter == 0) from generation-step
        # forwards (counter == 1, 2, ..., max_new_tokens).
        counter = [-1]

        def make_block_hook(layer_idx: int):
            def hook(module, _input, output):
                act = output[0] if isinstance(output, tuple) else output
                if act.shape[1] == len_prompt:
                    # Prompt forward pass. answer_prefix length varies per row,
                    # so we index per row. With left padding the right edge is
                    # aligned: the "prompt" token for row r sits at offset
                    # ``-1 - n_ap[r]`` from the end (the last core_prompt
                    # token, immediately before answer_prefix). Walk j over
                    # [-n_pre, n_ap_r] (capped at n_gen on the forward side)
                    # to fill prompt-{n_pre..1}, prompt, gen+{1..n_ap_r}.
                    for r_local in range(n_in_batch):
                        n_ap_r = batch_n_ap[r_local]
                        global_r = sample_offset + r_local
                        forward_cap = min(n_ap_r, n_gen)
                        for j in range(-n_pre, forward_cap + 1):
                            src_idx = -1 - n_ap_r + j
                            if src_idx < -len_prompt or src_idx > -1:
                                # Past start of sequence (left side) or past
                                # end of prompt (would overlap generation):
                                # leave the cache slot zero.
                                continue
                            cache[
                                layer_idx,
                                global_r,
                                n_pre + j,
                                :,
                            ] = act[r_local, src_idx, :].detach().to(torch.float16)
                elif act.shape[1] == 1 and counter[0] >= 1:
                    # Generation step (KV cache): single new model-generated
                    # token. For row r, the destination column is
                    # ``n_pre + n_ap_r + counter[0]`` (= "gen+{n_ap_r + counter[0]}").
                    # Skip rows whose destination would fall outside the
                    # n_gen budget (their post-prompt window was fully
                    # consumed by the answer_prefix).
                    for r_local in range(n_in_batch):
                        n_ap_r = batch_n_ap[r_local]
                        col = n_pre + n_ap_r + counter[0]
                        if col >= n_pre + 1 + n_gen:
                            continue
                        cache[
                            layer_idx,
                            sample_offset + r_local,
                            col,
                            :,
                        ] = act[r_local, -1, :].detach().to(torch.float16)
                return output

            return hook

        block_handles = [
            b.register_forward_hook(make_block_hook(i)) for i, b in enumerate(blocks)
        ]
        emb = model.get_input_embeddings()

        def emb_hook(_module, _input, output):
            counter[0] += 1
            return output

        emb_handle = emb.register_forward_hook(emb_hook)

        try:
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                **greedy_generate_kwargs(tokenizer),
            )
        finally:
            for h in block_handles:
                h.remove()
            emb_handle.remove()

        cont = gen[:, len_prompt:]
        responses.extend(tokenizer.batch_decode(cont, skip_special_tokens=True))
        sample_offset += n_in_batch

    return responses, cache


def position_labels(n_pre: int, n_gen: int) -> List[str]:
    """
    Human-readable labels for each cached column in :func:`generate_with_multi_pos_cache`.

    Returns
    ``["prompt-{n_pre}", ..., "prompt-1", "prompt", "gen+1", ..., "gen+{n_gen}"]``.
    """
    pre = [f"prompt-{n_pre - i}" for i in range(n_pre)]
    gen = [f"gen+{i}" for i in range(1, n_gen + 1)]
    return pre + ["prompt"] + gen
