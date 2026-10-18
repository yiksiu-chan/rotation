"""Capture residuals and answer logits together at an input-only rotation site."""
import torch

from .metrics import honesty


def score_and_capture(engine, enc, labels, answer_ids, position):
    """Return correctness logits and [layer, row, 1, feature] residuals.

    Register these hooks inside the patch context so they observe patched block
    outputs. All selected tokens must already exist in the full input. Capture
    precision matches generate_with_multi_pos_cache's float16 cache.
    """
    cache, handles = {}, []
    try:
        for layer, block in enumerate(engine.layers):
            def capture(_module, _inputs, output, layer=layer):
                x = output[0] if isinstance(output, tuple) else output
                cache[layer] = x[:, [position], :].detach().to(torch.float16).clone()
            handles.append(block.register_forward_hook(capture))
        logits = engine.model(**enc, use_cache=False).logits[:, -1]
        scores = honesty(logits, labels, *answer_ids)
        return scores, torch.stack([cache[layer] for layer in range(len(engine.layers))])
    finally:
        for handle in handles:
            handle.remove()


def generate_from_enc(engine, enc, tokenizer, max_new_tokens):
    """Greedy-decode a continuation for an already-encoded, left-padded batch.

    Call this inside the same patch context (a fresh ``with engine.patch(...)``
    block, separate from the one wrapping score_and_capture) so the hook fires
    once on this call's own prompt pass. A second full pass beyond
    score_and_capture's is unavoidable: the patch hook must not fire again on
    later KV-cache generation steps, and generate() does its own prompt pass.
    """
    from src.model.inference import greedy_generate_kwargs
    prompt_len = enc['input_ids'].shape[1]
    gen = engine.model.generate(**enc, max_new_tokens=max_new_tokens, **greedy_generate_kwargs(tokenizer))
    return tokenizer.batch_decode(gen[:, prompt_len:], skip_special_tokens=True)
