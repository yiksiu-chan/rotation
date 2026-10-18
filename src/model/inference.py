"""
Minimal Hugging Face generation helpers for the new pipeline.

Implemented separately from legacy scripts so this package stays self-contained.
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reduce TF log noise on clusters (same convention as other project scripts).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def load_model_and_tokenizer(
    model_path: str,
    dtype: str = "bfloat16",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load causal LM + tokenizer with left padding for batched generation.

    Args:
        model_path: Hugging Face id or local checkpoint path.
        dtype: ``"bfloat16"`` or ``"float16"`` for ``torch_dtype``.

    Returns:
        ``(model, tokenizer)`` on the default device / ``device_map="auto"``.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def tokenize_prompt_batch(
    tokenizer: AutoTokenizer,
    prompts: Sequence[str],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Left-pad and tokenize full prompt strings for batched ``model.generate``.

    Uses ``add_special_tokens=False`` because chat templates from
    :func:`src.prompt.chat_template.apply_chat_template` already include the
    leading BOS / header tokens; the default ``True`` would double-insert BOS
    on Llama 3.x and change greedy decoding relative to the behavior stage.
    """
    enc = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=False,
    )
    return {k: v.to(device) for k, v in enc.items()}


def greedy_generate_kwargs(tokenizer: AutoTokenizer) -> Dict[str, object]:
    """Shared greedy-decoding flags for behavior and cache generation."""
    return {
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
    }


@torch.no_grad()
def generate_completions(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    batch_size: int = 128,
    max_new_tokens: int = 10,
    desc: str = "generate",
) -> List[str]:
    """
    Greedy-decode one continuation per prompt (no KV-cache hooks).

    Args:
        model: Causal language model.
        tokenizer: Matching tokenizer (left-padded batches).
        prompts: Full prompt strings including assistant prefix stem.
        batch_size: Micro-batch size for ``model.generate``.
        max_new_tokens: Maximum generated tokens after the prompt.
        desc: Progress bar label.

    Returns:
        Decoded generated segments (excluding prompt tokens), one string per prompt.
    """
    out: List[str] = []
    device = model.device
    gen_kwargs = greedy_generate_kwargs(tokenizer)
    for i in tqdm(range(0, len(prompts), batch_size), desc=desc):
        batch = prompts[i : i + batch_size]
        enc = tokenize_prompt_batch(tokenizer, batch, device)
        prompt_len = enc["input_ids"].shape[1]
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            **gen_kwargs,
        )
        cont = gen[:, prompt_len:]
        out.extend(tokenizer.batch_decode(cont, skip_special_tokens=True))
    return out
