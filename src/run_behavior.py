"""
CLI shim for the **behavior** stage.

The implementation lives in :mod:`src.behavior.run_behavior`; this file exists
so that ``python src/run_behavior.py`` works directly from a checkout (it
prepends the repo root to ``sys.path`` before importing the package).

Usage
-----
::

    python src/run_behavior.py [--config PATH] [--config_override PATH]

Summary
------------------------------
For the configured ``task`` and ``dataset``:

1. **Load statements** via :mod:`src.task_corpus` (e.g. TriviaQA JSON, Azaria HF)
   and pick a class-balanced subset: ``balanced_set_size_behavior`` GT_0 +
   ``balanced_set_size_behavior`` GT_1, drawn with ``behavior_seed``.
   The selected indices are positions in the filtered list returned by the
   loader and are reused as the global index space throughout the rest of the
   pipeline.
2. **Control pass**
   - model forward pass with the per-task control system prompt
   (``cfg.control_system_prompt_map[task]``) via
   :func:`src.prompt.chat_template.apply_chat_template`, generate greedily,
   - score each generation judging if model's output is aligned with the instruction,
     with :func:`src.behavior.scoring.score_control_alignment` (1 = output matches GT).
     (1 = output matches GT).
   - output
   ``{output_root}/{model_alias}/{task}_{dataset}/behavior_results_control.json``.
3. **Experiment pass** (per ``system_prompt_id``)
   - model forward pass with each system prompt variant, generate greedily
   - score each generation for judging if model's output is aligned with the instruction,
     with :func:`src.behavior.scoring.score_experiment_alignment` (1 = output is the
     instructed *opposite* of GT).
   - output
   ``SysPrompt{id:03d}/behavior_results_experiment.json``.

Output schema
-------------
``behavior_results_control.json``::

    {
      "S_success_idx_control_GT_0": [int, ...],   // sorted global indices (GT == 0) that succeeded
      "S_success_idx_control_GT_1": [int, ...],   // sorted global indices (GT == 1) that succeeded
      "Accuracy_control": float,             // |successes| / |evaluated|
      "n_evaluated": int,                    // sample size used for control evaluation
      "n_evaluated_GT_0":     int,           // count of evaluated statements with GT == 0
      "n_evaluated_GT_1":     int,
      "n_evaluated_output_0": int,           // count of evaluated statements with output_label == 0
      "n_evaluated_output_1": int,
      "n_GT_0_output_0":      int,           // GT × output cross-tab counts
      "n_GT_0_output_1":      int,
      "n_GT_1_output_0":      int,
      "n_GT_1_output_1":      int            // (output_label == 1 ⇔ aligned with GT under control)
    }

``SysPrompt{id:03d}/behavior_results_experiment.json``::

    {
      "S_success_idx_both_conditions_GT_0": [int, ...],   // sorted GT_0 indices successful in BOTH conditions
      "S_success_idx_both_conditions_GT_1": [int, ...],   // sorted GT_1 indices successful in BOTH conditions
      "S_only_fail_in_experiment_GT_0":     [int, ...],   // sorted GT_0 indices: succeeded in control AND failed in experiment
      "S_only_fail_in_experiment_GT_1":     [int, ...],   // sorted GT_1 indices: succeeded in control AND failed in experiment
      "Accuracy_control":    float,
      "Accuracy_experiment": float,
      "n_evaluated":         int,
      "system_prompt_id":    int,
      "n_evaluated_GT_0":     int,
      "n_evaluated_GT_1":     int,
      "n_evaluated_output_0": int,
      "n_evaluated_output_1": int,
      "n_GT_0_output_0":      int,
      "n_GT_0_output_1":      int,
      "n_GT_1_output_0":      int,
      "n_GT_1_output_1":      int            // (output_label == 1 ⇔ output is the instructed opposite of GT)
    }

The experiment loop honors ``cfg.start_variant`` / ``cfg.end_variant`` (or
the matching CLI flags ``--start_variant`` / ``--end_variant``); only
variants in that inclusive range are evaluated.

Indices invariant
-----------------
``S_success_idx_*`` are global positions in the list returned by
``_load_statements(cfg)`` (i.e. the **filtered** corpus). The cache stage
re-loads the same list and indexes into it directly, so this ordering must
stay reproducible across runs.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.behavior.run_behavior import main

if __name__ == "__main__":
    main()
