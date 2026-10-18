# Representations in Motion

## Contents

- [Setup](#setup)
- [Part 1 — The core pipeline (behavior + rotation, per task)](#part-1--the-core-pipeline-behavior--rotation-per-task)
- [Part 2 — Causal experiments (activation patching)](#part-2--causal-validation-activation-patching)
- [Repository layout](#repository-layout)
- [Datasets](#datasets)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Part 1 — The core pipeline (behavior + rotation, per task)

The paper studies four tasks, plus a follow-up trajectory analysis and an
implicit/role-play variant. Each defines a control and an experiment
condition and measures both behavior and how the contrast direction between
them rotates through the layers.

| Task | Config | Data |
| --- | --- | --- |
| Lying / deception | `config/lying.yaml` | Azaria–Mitchell (HF) or TriviaQA |
| Sycophancy | `config/sycophancy.yaml` | Azaria–Mitchell / TriviaQA + sycophancy prompt variants |
| Spatial reasoning | `config/spatial_easy.yaml` | `datasets/spatial_reasoning/` |
| Hallucination (control) | `config/hallucination.yaml` | `datasets/triviaqa_hallucination/` |
| Forward-pass trajectories | `config/dynamical_trajectory.yaml` | cache-stage activations from any of the above |
| Implicit / role-play lying | `config/lying_to_achieve_goals.yaml`, `config/lying_situation_induced.yaml` | `datasets/real_world_lying/`, MASK (HF) |

### Quick start

The whole pipeline (behavior → activation cache → analysis → plots) runs
through one parameterized launcher:

```bash
# Lying / Azaria-Mitchell
sbatch scripts/run_pipeline.sh --task lying --model allenai/Olmo-3-7B-Instruct

# Spatial reasoning
sbatch scripts/run_pipeline.sh --task spatial-easy --model Qwen/Qwen3-4B-Instruct-2507

# Sycophancy (choose azaria-mitchell or trivia-qa)
sbatch scripts/run_pipeline.sh --task sycophancy --dataset trivia-qa --model Qwen/Qwen3-4B-Instruct-2507

# Just some stages
sbatch scripts/run_pipeline.sh --task lying --model allenai/Olmo-3-7B-Instruct --steps behavior,cache
```

`--task` picks the default config and dataset; override with `--config`,
`--dataset`, `--batch-generation`, `--batch-cache`. Anything else you pass is
forwarded straight to `src/run_pipeline.py` (variant ranges, sample sizes,
output overrides). Override cluster resources at submission time, e.g.
`sbatch --mem=96G --time=12:00:00 scripts/run_pipeline.sh --task lying --model allenai/Olmo-3-7B-Instruct`.

Hallucination needs per-model consistency labels before (or alongside) the
main run:

```bash
# Compute labels and run the full pipeline in one job
sbatch scripts/run_pipeline.sh --task hallucination --model allenai/Olmo-3-7B-Instruct --full

# Or generate labels separately, then run the pipeline
sbatch scripts/run_hallucination_consistency.sh \
  --model meta-llama/Llama-3.3-70B-Instruct --alias Llama-3.3-70B-Instruct --batch-size 64
sbatch scripts/run_pipeline.sh --task hallucination --model meta-llama/Llama-3.3-70B-Instruct
```

### Running one task directly, without the launcher

```bash
python src/run_pipeline.py --steps behavior,cache,analysis,plots \
  --config_override config/lying.yaml --dataset azaria-mitchell
```

Swap `--config_override` for any of the configs in the table above.

### Overriding config without editing YAML

`config/default.yaml` is the base; each task config is a diff-only override,
deep-merged on top.

```bash
# Shrink a run for a smoke test
python src/run_pipeline.py --steps behavior,cache --set balanced_set_size_behavior=25 --set task=lying

# Restrict to a contiguous range of system-prompt variants
python src/run_pipeline.py --steps cache,analysis,plots --start_variant 10 --end_variant 25

# Sweep multiple cache seeds
python src/run_cache_activations.py --set 'seed_list_cache=[0, 1, 2, 3, 4]'
```

`balanced_set_size_behavior` (behavior stage) and `n_sample_cache` (cache
stage) are **per ground-truth class**, not a total — the value you set gets
roughly doubled. A worked example, "rotation for spatial reasoning, ~100
examples, one variant, right now":

```bash
python src/run_pipeline.py --steps behavior,cache,analysis,plots \
  --config_override config/spatial_easy.yaml --dataset spatial-easy \
  --set balanced_set_size_behavior=50 --end_variant 0
```

This writes into `results/runs/` and `results/plots/` under the repo root by
default (no path overrides needed on a fresh clone). When it finishes, read
`results/plots/{model_alias}/summary_spatial_easy_spatial-easy.json` first —
it's also printed to stdout — for the plain-language answer (control
accuracy, mean flip rate, mean rotation) before digging into any figure.

### Running a single stage

Each stage reads whatever upstream artifacts already exist on disk, so you
can rerun any one of them in isolation:

```bash
python src/run_behavior.py          --config_override config/lying.yaml   # 1. behavior (GPU)
python src/run_cache_activations.py --config_override config/lying.yaml   # 2. cache (GPU)
python src/run_analysis.py          --config_override config/lying.yaml   # 3. analysis (CPU)
python src/generate_paper_plots.py  --config_override config/lying.yaml   # 4. plots (CPU)
```

### Forward-pass trajectory analysis

After a cache run, you can dig into individual-trial trajectories through
the layers:

```bash
# Train predictors of the microscopic class from trajectory descriptors
python -m src.run_dynamical_trajectory_analysis \
  --config config/dynamical_trajectory.yaml \
  --data_folder results/runs/<model_alias>/<task_dataset>/SysPrompt000/0_pre10_gen10

# Per-trial projection + 2D/3D trajectory plots
python src/run_microscopic.py \
  --task_vector_data_folder <seed_folder> --targets_data_folder <seed_folder> \
  --targets control_GT_0:0,5 experiment_GT_1:0,2
```

### Where things land

```
results/runs/{model_alias}/{task}_{dataset}/
├── behavior_results_control.json
└── SysPrompt{id:03d}/
    ├── behavior_results_experiment.json
    └── {seed}_pre{n_pre}_gen{n_gen}/
        ├── completions/       # generated text per condition
        ├── activations/       # cached residual-stream activations (.pkl)
        ├── stage_stats/       # rotation_stats.json + heatmaps
        └── PCA_plots/         # per-position PCA grids

results/plots/{model_alias}/    # aggregated publication figures
```

See the docstrings in each `src/run_*.py` shim for the exact field-by-field
schema of every artifact and the definition of each rotation metric.

## Part 2 — Causal validation (activation patching)

The pipeline above shows *correlation*: behavior and rotation change
together. Activation patching tries to show *causation*: if we copy an
activation from one condition into the other during the forward pass, does the
model's behavior/representation follow it?

All patching code are in `src/activation_patching/`. 
For each step, there is a `stepN_*.py` file and a bash script for submitting to slurm in `scripts/patching/`.
The `run_custom_sweep.py`/`plot_custom_sweep.py` are exploratory scripts in which you can customisze patching experiments. 


| Step | What it does | Run it |
| --- | --- | --- |
| 1. Unpatched rotation baseline | For each of 20 system prompt variants and 100 examples, finds the token position that has the strongest rotation. | `bash scripts/patching/step1_find_best_position.sh` |
| 2. Mean residual stream patching | At each prompt's best token position, replace the entire residual stream vector at each layer with the lying condition's mean activation (averaged over the dataset). | `bash scripts/patching/step2_patch_residual.sh` |
| 3. Attention head sweep | Patch every individual attention head at each system prompt's best token, then rank the heads by patching effect. | `bash scripts/patching/step3_head_sweep.sh` |
| 4. Cumulative top-k head patching | Patch the top-k heads together (k = 1..30). | `SELECTION=ranked HEAD_SCREEN=<step-3 results.json> bash scripts/patching/step4_patch_heads_cumulative.sh` and `SELECTION=random bash scripts/patching/step4_patch_heads_cumulative.sh`, then `python src/activation_patching/step4_plot_heads_cumulative.py` |
| General-purpose sweeps | A flexible runner for residual/attention/MLP/head windows, position sweeps, or custom cumulative sweeps, if you need something the numbered steps above don't cover. Reproduces the lying-to-honest direction by default; can also run the reverse direction or both at once for exploration beyond what the paper reports. | `bash scripts/patching/run_custom_sweep.sh run --config config/patching_lying.yaml --component residual --layers 8,12 --positions=-12:-1 --sweep window --direction noise --output results/patching/residual` |

The `default_patching_variants` field in `config/patching_lying.yaml` is the
20 variant subset used for our main experiments.
You can override with each script's `VARIANTS` env var (steps 1-4) or `--variants`.

A few useful flags on `run_custom_sweep.sh run` (i.e.
`src/activation_patching/run_custom_sweep.py`): `--component
residual|attention|mlp|head`, `--sweep window|position|head|head-position|cumulative`,
`--direction noise` (lying-to-honest, what the paper reports; `denoise` and
`both` are also accepted, for exploration beyond the paper), `--patch-source
donor|mean` (donor = the matched statement's own activation, mean = the
reference-average version), and `--dry-run` to validate settings without
loading a model. `--recipe lying-rebuttal` / `--recipe lying-rotation-grid`
bundle the full rebuttal sweeps. Plot any of its outputs with:

```bash
bash scripts/patching/run_custom_sweep.sh plot --inputs results/patching/residual --output results/patching/plots --metric all
```


## Repository layout

```
.
├── README.md                 # this file
├── requirements.txt
├── config/                   # YAML configs (default.yaml + per-task overrides)
├── datasets/                 # prompt-variant banks + task datasets (small)
├── scripts/                  # SLURM launchers + data-prep helpers
├── supplementary/            # a separate, trimmed release bundle of this
│                              # code meant to accompany the paper submission
│                              # (its own README, config, src, datasets)
└── src/                      # the package
    ├── run_pipeline.py             # orchestrator (behavior→cache→analysis→plots)
    ├── run_behavior.py             # stage 1 shim
    ├── run_cache_activations.py    # stage 2 shim (GPU)
    ├── run_analysis.py             # stage 3 shim
    ├── generate_paper_plots.py     # stage 4 shim
    ├── run_dynamical_trajectory_analysis.py, run_microscopic.py  # trajectory analysis
    ├── activation_patching/        # causal patching (Part 2)
    │   ├── step1_find_best_position.py    # + plots automatically
    │   ├── step2_patch_residual.py        # + validates/tabulates/plots automatically
    │   ├── step3_head_sweep.py          # + tabulates/plots automatically
    │   ├── step4_patch_heads_cumulative.py  # + validates/tabulates/plots automatically
    │   │   step4_plot_heads_cumulative.py   # combines a ranked + random run (only standalone plot script left)
    │   ├── run_custom_sweep.py, plot_custom_sweep.py  # general-purpose tool
    │   └── utils/                  # shared engine (not needed to run the tool)
    │       ├── core.py, fast_heads.py, metrics.py, prefix_rotation.py,
    │       │   results.py, variant_positions.py, plotting.py
    ├── behavior/  cache_activations/  analysis/  paper_plots/
    ├── config/    model/   prompt/   task_corpus/
    └── paths.py                    # PROJECT_ROOT resolution
```

All artifacts are written under `<repo>/results/` by default (configurable
via the `paths:` block in each config).

## Datasets

Bundled under `datasets/`:

- `datasets/system_prompts_all/` — system-prompt variant banks (lying: 156,
  sycophancy: 100, hallucination: 20, spatial: 20).
- `datasets/triviaqa_tf/`, `datasets/triviaqa_hallucination/` — TriviaQA
  true/false statements and per-variant hallucination labels. TriviaQA files
  used by lying and sycophancy are pre-filtered toward statements the model
  already knows, so their control accuracy is intentionally high — don't
  read it as an unbiased accuracy estimate.
- `datasets/spatial_reasoning/` — spatial perspective-taking datasets.
- `datasets/real_world_lying/` — role-play / implicit-lying scenarios.

Loaded from the Hugging Face Hub at runtime instead of bundled:

- **Azaria–Mitchell true/false statements** (Azaria & Mitchell, 2023, *"The
  Internal State of an LLM Knows When It's Lying"*, Findings of EMNLP).
  Canonical data:
  [azariaa.com/Content/Datasets/true-false-dataset.zip](https://azariaa.com/Content/Datasets/true-false-dataset.zip).
  The loader (`src/task_corpus/azaria_loader.py`) expects a dataset exposing
  a `combined` split with `claim` (str) and `label` (0/1) columns — set
  `azaria_hf_dataset` in `config/default.yaml` to your own HF mirror or local
  copy (the shipped value is a placeholder).
- **MASK** (`cais/MASK`), used only by the implicit-lying task, loaded
  directly from the Hub.

Hallucination uses a separate per-model artifact: `precompute_per_variant_consistency.py`
runs repeated generations to estimate whether the model already "knows"
each statement, and those labels define the control/hallucination pools. If
you change the thresholds, regenerate this artifact before running
downstream stages — and don't substitute the pre-filtered, known-answer
TriviaQA file for the unfiltered input this step expects.
