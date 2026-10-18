"""
CLI shim for the **cache activations** stage.

The implementation lives in :mod:`src.cache_activations.cache_activations`;
this file exists so that ``python src/run_cache_activations.py`` works
directly from a checkout (it prepends the repo root to ``sys.path`` before
importing the package).

Usage
-----
::

    python src/run_cache_activations.py [--config PATH] [--config_override PATH]

Summary
------------------------------
For each ``system_prompt_id`` for which the behavior stage produced a result,
and each ``seed`` in ``cfg.seed_list_cache``:

1. Load two GT-split pools of statements successful (followed the instruction)
   in both conditions:
   - Read ``S_success_idx_both_conditions_GT_0`` and
   ``S_success_idx_both_conditions_GT_1`` from the behavior stage's per-prompt
   ``behavior_results_experiment.json`` and treat them as separate sampling
   pools.

2. Sample from each GT pool separately (governed by ``cfg.cache_sample_method``):
   - ``"balance_set"`` — ``n_sample_cache`` GT_0 + ``n_sample_cache`` GT_1
     (both halves capped to ``min(n_sample_cache, |pool_GT_0|, |pool_GT_1|)``).
   - ``"maximum"`` — up to ``n_sample_cache`` per GT class, each cap independent.
   - if n_sample_cache=-1, use the full pools (no sampling)

3. For each condition (``control``, ``experiment``):
   - render prompts via
   :func:`src.prompt.chat_template.apply_chat_template_split`
   (returns ``(core_prompt, answer_prefix)`` per row so the cache hook can
   anchor the "prompt" position at the last token *before* answer_prefix),
   - generate with hooks that cache hidden states across ``n_pre + 1 + n_gen``
   token positions; the leading ``n_ap`` "gen+i" slots are filled from the
   answer_prefix tokens of the prompt forward pass, and the remaining slots
   from model-generated tokens
   (:func:`src.cache_activations.hooks.generate_with_multi_pos_cache`), and
   - score ``output_label``, 1 means aligned with the instruction
     - For control condition, 1 means model's output matches the GT label
     - For experiment condition, 1 means model's output is the opposite of the GT

4. **Experiment-fail subset.** Read the GT-split pools
   ``S_only_fail_in_experiment_GT_0`` and ``S_only_fail_in_experiment_GT_1``
   from the experiment behavior JSON (statements that **succeeded in control**
   ``output_label == 1`` **and failed in experiment** ``output_label == 0``
   for this variant — i.e. the intersection, not "any failure in
   experiment"). Sample from each GT pool separately with the same
   ``cache_sample_method`` / ``n_sample_cache`` budget, run the
   **experiment** condition only, and persist under the ``_fail`` filename
   suffix. Skipped silently when both pools are empty for a given
   ``(SysPrompt, seed)``.

5. Persist completions and per-GT activation pickles under
   ``{output_root}/{model_alias}/{task}_{dataset}/SysPrompt{id:03d}/{seed}_pre{n}_gen{m}/``

Output schema
-------------
All artifacts for one ``(SysPrompt, seed)`` live under::

    {output_root}/{model_alias}/{task}_{dataset}/SysPrompt{id:03d}/{seed}_pre{n_pre}_gen{n_gen}/
    ├── completions/
    │   ├── completions_control_{cache_sample_method}.json
    │   ├── completions_experiment_{cache_sample_method}.json
    │   └── completions_experiment_{cache_sample_method}_fail.json   # experiment-fail subset (optional)
    └── activations/
        ├── activations_control_{cache_sample_method}_GT_0.pkl
        ├── activations_control_{cache_sample_method}_GT_1.pkl
        ├── activations_experiment_{cache_sample_method}_GT_0.pkl
        ├── activations_experiment_{cache_sample_method}_GT_1.pkl
        ├── activations_experiment_{cache_sample_method}_GT_0_fail.pkl   # experiment-fail subset (optional)
        └── activations_experiment_{cache_sample_method}_GT_1_fail.pkl   # experiment-fail subset (optional)

The ``_fail`` files contain the **experiment-fail subset** described in
step 4 above: statements that succeeded in control (``output_label == 1``)
**and** failed in experiment (``output_label == 0``). They are written
only when that intersection is non-empty for a given ``(SysPrompt, seed)``
and may be missing for one or both GT classes.

``completions/completions_{condition}_{cache_sample_method}.json``::

    {
      "system_prompt":         str,                        // exact system prompt rendered for this condition
      "condition":             "control" | "experiment",
      "system_prompt_id":      int,                        // variant index from the experiment bank
      "seed":                  int,                        // seed used to draw the subset
      "n_pre_prompt_tokens":   int,                        // number of tokens cached before the "prompt" anchor
      "n_generated_tokens":    int,                        // number of post-prompt tokens cached as "gen+1", "gen+2", ...
                                                           //   (covers answer_prefix tokens then model-generated tokens)
      "cache_sample_method":   "balance_set" | "maximum",
      "sampled_size_GT_0":     int,                        // achieved GT_0 count
      "sampled_size_GT_1":     int,                        // achieved GT_1 count
                                                           //   - "balance_set": both equal to
                                                           //       min(n_sample_cache, |pool ∩ GT_0|, |pool ∩ GT_1|)
                                                           //   - "maximum":     each = min(n_sample_cache, |pool ∩ GT_c|)
      "full_prompt_example":   str,                        // one fully-rendered prompt for traceability
      "outputs": [
        {
          "statement_index":   int,                        // global index into the filtered corpus
          "statement":         str,                        // raw claim text
          "GT_label":          0 | 1,                      // ground-truth label for this statement
          "model_output":      str,                        // greedy continuation (no prompt)
          "output_label":      0 | 1                       // 1 = aligned with the instruction (aligned with GT in
                                                           //     control, opposite of GT in experiment)
        },
        ...                                                // sampled_size_GT_0 GT_0 rows then sampled_size_GT_1 GT_1 rows
      ]
    }

``activations/activations_{condition}_{cache_sample_method}_GT_{label}.pkl`` —
pickled torch tensor (one file per GT split)::

    shape = (sampled_size_GT_{label}, n_positions, n_layers, d_model)   // asserted before writing
    dtype = torch.float16
    device = CPU                                                        // moved off-device before pickling

    Axis layout:
      axis 0  ──  sample (within one GT class for this condition)
      axis 1  ──  token position
                   - 0 .. n_pre - 1            "prompt-{n_pre}" .. "prompt-1"
                   - n_pre                     "prompt"   (last core_prompt token,
                                                            i.e. before answer_prefix)
                   - n_pre + 1 .. n_pre + n_gen "gen+1" .. "gen+{n_gen}"
                                                   The leading n_ap (per row)
                                                   slots come from the
                                                   answer_prefix tokens; the
                                                   rest come from model
                                                   generation steps.
      axis 2  ──  transformer layer
      axis 3  ──  hidden dimension

Variant range
-------------
The variant loop honors ``cfg.start_variant`` / ``cfg.end_variant`` (or the
matching CLI flags ``--start_variant`` / ``--end_variant``). Only variants in
the inclusive range are cached; the rest are silently skipped.

Hook block-module discovery
----------------------------
``hooks._get_block_modules`` currently assumes a Llama-style
``model.model.layers`` attribute. To run on Gemma / Qwen / Yi / Mistral, add
the right block-list lookup there.

Atomicity
---------
Both JSON and pickle writes go through a ``*.tmp`` file followed by
``os.replace``, so a crashed run never leaves a half-written artifact for the
analysis stage to consume.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.cache_activations.cache_activations import main

if __name__ == "__main__":
    main()
