"""
Typed experiment configuration loaded from ``config/default.yaml`` (and overrides).

All path-like settings stored in the YAML config are either absolute or
relative to :class:`PROJECT_ROOT` after substitution of ``${PROJECT_ROOT}``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence


def _substitute_project_root(obj: Any, root: Path) -> Any:
    """Recursively replace ``${PROJECT_ROOT}`` string markers with ``root``."""
    if isinstance(obj, str):
        return obj.replace("${PROJECT_ROOT}", str(root))
    if isinstance(obj, list):
        return [_substitute_project_root(x, root) for x in obj]
    if isinstance(obj, dict):
        return {k: _substitute_project_root(v, root) for k, v in obj.items()}
    return obj


@dataclass
class ExperimentSystemPromptSpec:
    """Maps a task name to a JSON file under the system-prompt directory."""

    filename: str
    variants_json_key: str


@dataclass
class ExperimentConfig:
    """
    Global knobs for the rotation pipeline (behavior → cache → analysis → plots).

    Attributes mirror ``config/default.yaml``; see that file for semantic
    defaults and the cosmetic section grouping (loader flattens it before
    construction).
    """

    model_path: str
    task: str
    dataset: str
    n_pre_prompt_tokens: int
    n_generated_tokens: int
    max_new_tokens: int
    batch_size_generation: int
    batch_size_cache: int
    user_prompt: str
    # Assistant-side text appended after the chat-template body. The cache
    # stage's "prompt" position anchors at the **last token of the body** (i.e.
    # the token right before this string); the answer_prefix tokens occupy the
    # leading ``gen+i`` slots of the post-prompt cache window, with
    # model-generated tokens filling the rest. For ``lying_to_achieve_goals``
    # a per-row ``meta["answer_prefix"]`` overrides this config value.
    answer_prefix: str
    control_system_prompt_map: Dict[str, str]
    gt_0_label_map: Dict[str, str]
    gt_1_label_map: Dict[str, str]
    sys_prompt_dir_relative: str
    experiment_system_prompts: Dict[str, ExperimentSystemPromptSpec]
    output_root_relative: str
    plot_root_relative: str
    balanced_set_size_behavior: int
    n_sample_cache: int
    cache_sample_method: str
    behavior_seed: int
    seed_list_cache: List[int]
    triviaqa_dataset_glob: str
    triviaqa_hallucination_dataset_glob: str
    azaria_hf_dataset: str
    azaria_hf_split: str
    real_world_lying_dataset_paths: Dict[str, str]
    mask_hf_dataset: str
    mask_hf_split: str
    mask_dataset_configs: List[str]
    spatial_easy_dataset_path: str
    model_dtype: str
    pca_plot_style: Dict[str, Any] = field(default_factory=dict)
    # Aesthetics for the per-position 2D/3D manifold plots (PCA / t-SNE / UMAP).
    # Same shape as ``pca_plot_style`` plus a ``layer_shade_min``. Highlighted
    # target traversals come from the ``run_microscopic --targets`` CLI; they
    # are no longer part of this style dict. Empty dict = use module defaults.
    manifold_plot_style: Dict[str, Any] = field(default_factory=dict)
    # Inclusive variant range (None = no bound). Lets the user run a subset of
    # system prompts without editing the bank or moving artifacts around.
    # Honored by **every** stage's variant loop (behavior, cache, analysis,
    # paper plots) via :meth:`variant_indices`.
    start_variant: Optional[int] = None
    end_variant: Optional[int] = None
    # Optional explicit list of variant indices. When non-None, this overrides
    # ``start_variant`` / ``end_variant`` so the user can run a non-contiguous
    # subset (e.g. the variants that produced clean behavioral sycophancy on
    # both datasets). Out-of-range entries are dropped silently.
    variant_subset: Optional[List[int]] = None
    # Analysis-stage cache toggle. ``False`` (default) = read existing
    # ``activations_*.pkl`` / ``completions_*.json`` from the cache stage;
    # variants whose pickles are missing are silently skipped (current
    # behavior). ``True`` = invoke :func:`run_cache_activations_for_config`
    # **first**, regenerating the pickles for the same
    # ``[start_variant, end_variant]`` range, then run the analysis loop.
    # Useful for one-shot "rebuild + plot" runs from a single CLI invocation
    # without remembering to call the cache stage separately.
    rerun_cache_in_analysis: bool = False
    # How many top-ranked token positions (per rotation_r) get a PCA plot
    # written by the analysis stage. Default 1 = only the best position.
    # Ignored when ``pca_token_positions`` is non-empty.
    save_top_n_tokens_4_PCAplots: int = 1
    # Master toggle for the per-layer PCA grid (``PCA_plots/pca_<pos>.{pdf,svg}``).
    # Set ``false`` to skip the PCA-grid loop (useful when iterating only on
    # heatmaps / manifolds). Default ``true``.
    save_PCA_plots: bool = True
    # Output file formats for the per-position PCA grid plots. Any combination
    # of ``pdf``, ``svg``, ``png``, ``jpg`` / ``jpeg``, ``webp``, ``eps`` (all
    # rendered via Plotly's kaleido backend) and ``html`` (interactive
    # standalone). Default ``["pdf", "svg"]`` matches the project convention
    # for non-heatmap figures.
    PCA_plot_formats: List[str] = field(default_factory=lambda: ["pdf", "svg"])
    # Optional override for the PCA-grid output root. When set, PCA grids
    # are written under
    # ``{this_root}/{model_alias}/{task}_{dataset}/SysPrompt{id:03d}/{seed}_pre{n}_gen{m}/PCA_plots/``
    # instead of next to the cached pickles under ``output_root``. Use this to
    # re-emit PCA plots when the original cache_seed_dir is read-only (cached
    # pickles are still read from the original location). ``None`` (default)
    # keeps the historical co-located layout.
    pca_plots_output_root: Optional[str] = None
    # How many top-ranked token positions (per rotation_r) get a manifold
    # plot bundle (2D + 3D × ``manifold_type``). Same selection logic as
    # ``save_top_n_tokens_4_PCAplots``: explicit ``pca_token_positions`` wins;
    # otherwise the top-N positions by the driving metric. Default 1.
    save_top_n_tokens_4_manifolds: int = 1
    # Master toggle for manifold plot generation. Set ``false`` to skip the
    # manifold loop entirely (useful when iterating only on PCA / heatmaps).
    save_manifold_plots: bool = True
    # Output file formats for both manifold subspaces (across-layer + set-layer,
    # 2D + 3D). Same accepted values as :attr:`PCA_plot_formats`. Default
    # ``["pdf", "svg", "html"]`` writes static + interactive — drop ``"html"``
    # if the extra files aren't useful.
    manifold_plot_formats: List[str] = field(
        default_factory=lambda: ["pdf", "svg", "html"]
    )
    # Which manifold methods are emitted by the manifold loop. Each entry must
    # be one of ``"PCA"``, ``"t-SNE"``, ``"UMAP"``; an empty list disables
    # manifold plots even when ``save_manifold_plots`` is true.
    manifold_type: List[str] = field(
        default_factory=lambda: ["PCA", "t-SNE", "UMAP"]
    )
    # Per-GT-class subsample size for the manifold loop (taken as the leading
    # rows of each ``(condition, GT)`` activation pickle). ``0`` or negative
    # means "use all available samples". Manifold plots typically render
    # cleaner with a modest cap (e.g. 30) than with the full cache.
    n_sample_manifold: int = 30
    # Inclusive layer-range bounds for the **across-layer** manifold subspace
    # (the reducer fits on samples × layers across the chosen range, encoding
    # the layer index as a colorscale gradient). ``-1`` on either end means
    # "no bound" (``(-1, -1)`` = all layers). The filename always reflects
    # the resolved absolute range (e.g. ``manifold_prompt_L4-L15_…``);
    # per-trace hover labels and the right-margin colorbars also tick at
    # absolute layer indices.
    manifold_across_start_layer: int = -1
    manifold_across_end_layer: int = -1
    # Which **PCA subspaces** are emitted by the manifold loop. The two
    # supported subspaces are:
    #   - ``"across_layer"`` (default behavior): one reducer fit per
    #     ``(position, layer-range)``, encoding layer index as a colorscale
    #     gradient. Saved under ``manifolds/{2D,3D}/{method}/…``.
    #   - ``"set_layer"``: one reducer fit per ``(position, layer)`` listed
    #     in :attr:`manifold_PCA_set_layer`; all groups' activations at the
    #     fixed layer are projected into the PC plane, and target traversals
    #     project later layers' activations through the **same** PCA so the
    #     reader can see where target paths land in the set-layer subspace.
    #     Saved under ``manifolds_set_layer/{2D,3D}/PCA/…``. Applies only
    #     to the ``"PCA"`` method (t-SNE / UMAP have no fixed-layer subspace).
    manifold_PCA_subspace: List[str] = field(
        default_factory=lambda: ["set_layer", "across_layer"]
    )
    # Layers (absolute indices) at which the ``"set_layer"`` PCA subspace
    # is fit and plotted. Empty list = no set-layer plots are emitted even
    # when ``"set_layer"`` ∈ :attr:`manifold_PCA_subspace`.
    manifold_PCA_set_layer: List[int] = field(default_factory=list)
    # Absolute layer indices anchoring the trajectory plot's background
    # scatter. Mirrors :attr:`manifold_PCA_set_layer` — each entry produces a
    # separate trajectory figure (filename encodes ``setL{L}``). Empty list =
    # trajectory plots are skipped. Consumed only by ``src.run_microscopic``
    # (no analysis-stage path reads this).
    trajectory_set_layer: List[int] = field(default_factory=list)
    # When ``True``, every background scatter (manifold + trajectory) renders
    # as a uniform light-grey instead of per-group color/colorscale. Per-target
    # trajectory markers and target traversals keep their colors.
    bg_grey_scale: bool = False
    # Inclusive layer range projected through the set-layer PCA for each
    # ``run_microscopic --targets`` entry (i.e. how far back / forward each
    # target traversal extends from the set layer). ``-1`` on either bound
    # mirrors the corresponding ``manifold_across_*_layer`` bound.
    manifold_set_start_layer: int = -1
    manifold_set_end_layer: int = -1
    # Which group sets are plotted. Each entry generates one set of files
    # whose filename suffix encodes the choice:
    #   - ``"with_experiment_fail"``: include the
    #     ``experiment_GT_*_fail`` groups (when their pickles exist) →
    #     filename suffix ``_with_fail``.
    #   - ``"without_experiment_fail"``: only the four base groups →
    #     filename suffix ``_without_fail``.
    manifold_group: List[str] = field(
        default_factory=lambda: ["with_experiment_fail", "without_experiment_fail"]
    )
    # Explicit token-position labels (e.g. ``["prompt", "prompt-1"]``) to
    # plot in PCA. When non-empty this overrides ``save_top_n_tokens_4_PCAplots``;
    # unknown labels are skipped with a warning. Default ``[]`` = use top-N.
    pca_token_positions: List[str] = field(default_factory=list)
    # Name (in ``ROTATION_METRIC_REGISTRY``) of the rotation metric used to
    # pick the "best token position" for analysis-stage PCA selection and for
    # row ordering / labels in the paper-plot cross-prompt heatmap. Every
    # registered metric is computed regardless; this only selects the driver.
    rotation_metric: str = "late_layer_anti_alignment"
    # Which metrics are shown as the trailing ``r1, r2, …`` block in the
    # cosine-similarity heatmaps (analysis stage and paper-plot cross-prompt
    # heatmap). One of:
    #   - "all"            → one column per metric in ROTATION_METRIC_ORDER
    #   - "chosen_metric"  → only ``rotation_metric``'s column
    heatmap_r_metric_display: str = "all"
    # Whether to render the per-row text label on the left side of the
    # paper-plot cross-prompt cosine heatmap (system-prompt index, best token
    # position, raw prompt content). True (default) keeps the labels; False
    # hides them and shrinks the figure's left margin.
    heatmap_label_display: bool = True
    # Output file formats for the analysis-stage cosine-similarity heatmap
    # (``stage_stats/cos_sim_heatmaps_across_tokens.<fmt>``). Same accepted
    # values as :attr:`PCA_plot_formats` (any combo of ``pdf`` / ``svg`` /
    # ``eps`` / ``png`` / ``jpg`` / ``webp`` / ``html``). Default
    # ``["svg", "png"]`` matches the project's prior on-disk convention for
    # heatmaps (vector + high-DPI raster). Distinct from the paper-plot
    # cross-prompt heatmap, which is controlled by
    # ``paper_plot_formats["cos_sim_heatmap"]``.
    cos_sim_heatmap_formats: List[str] = field(
        default_factory=lambda: ["svg", "png"]
    )
    # Names of additional datasets (same task, same model, same seed list) to
    # cross-compare against ``cfg.dataset`` in the paper-plot stage. For each
    # entry, a per-seed scatter is written under ``no_rotation_analysis/``
    # pairing each system prompt's ``rotation_r`` (at its own best position
    # under ``cfg.rotation_metric``) on ``cfg.dataset`` (x) vs the other
    # dataset (y). Empty list = no cross-dataset scatter.
    compare_datasets: List[str] = field(default_factory=list)

    # Comma/range spec (parsed by ``activation_patching.utils.core.parse_indices``)
    # of the curated system-prompt variants used as the single source of
    # truth by every activation-patching step script
    # (``src/activation_patching/step{1,3}_*.py`` and
    # ``run_custom_sweep.py --recipe``). Only meaningful for
    # ``config/patching_lying.yaml``; empty for every other task config.
    default_patching_variants: str = ""

    # --- 4-class classifier accuracy curve (analysis + paper-plot stages) ---
    # Master toggle for the per-(SysPrompt, seed) 4-class classifier accuracy
    # curve. When ``True``, the analysis stage trains a per-layer multinomial
    # logistic-regression classifier (in a PCA-reduced subspace fit per CV
    # fold to avoid leakage) on the four ``(condition, GT)`` groups and writes
    # the resulting accuracy curve to ``stage_stats/``. The paper-plot stage
    # aggregates these curves into a mean ± std band across system prompts.
    save_class_separation_curve: bool = True
    # Token-position label feeding the classifier. ``None`` (the YAML ``null``)
    # = use each variant's best position under ``rotation_metric``; otherwise
    # a canonical label like ``"prompt-1"``. Unknown labels fall back to the
    # variant's best position with a warning.
    class_separation_token_position: Optional[str] = None
    # Number of principal components kept before the logistic-regression step.
    # PCA is fit per training fold (inside the CV pipeline) so the reported
    # accuracy is leakage-free. Filenames include this value so reruns with
    # different ``n_PC_classifier`` don't overwrite each other.
    n_PC_classifier: int = 3
    # Stratified k-fold cross-validation parameters.
    cv_n_splits: int = 5
    cv_shuffle: bool = True
    cv_random_state: int = 0
    # Multinomial logistic-regression hyperparameters (sklearn defaults +
    # explicit ``solver="lbfgs"``). L2 regularization strength = ``1/C``.
    classifier_C: float = 1.0
    classifier_max_iter: int = 1000
    # Output file formats for the per-seed ``classifier_acc_curves_nPC{n}``
    # plot and the paper-plot ``classifier_acc_curves_nPC{n}_{task}_…`` mean
    # ± std band figure. Same accepted values as :attr:`PCA_plot_formats`.
    class_separation_plot_formats: List[str] = field(
        default_factory=lambda: ["png", "svg"]
    )
    # Line + band aesthetics shared by both class-separation figures (the
    # per-seed curve in the analysis stage and the mean-band paper figure).
    # Line styling applies to both plots; band styling is consumed only by
    # the mean-band paper figure. ``*_color`` accepts hex (``#b22222``),
    # ``rgb()``/``rgba()`` strings, or a common CSS name handled by the
    # ``_to_rgba`` helper in :mod:`src.paper_plots.figures`; ``band_alpha``
    # is a float in [0, 1].
    class_separation_line_color: str = "firebrick"
    class_separation_line_width: float = 2.0
    class_separation_band_color: str = "firebrick"
    class_separation_band_alpha: float = 0.2
    # Y-axis range ``[lo, hi]`` for both class-separation figures (per-seed
    # curve in the analysis stage and the mean-band paper figure).
    class_separation_y_range: List[float] = field(
        default_factory=lambda: [0.0, 1.05]
    )

    # Variance-band kind for the paper-plot ``cos_sim_curves`` figure. One of
    # ``"std"`` (1-σ, default), ``"sem"``, or ``"ci95"`` (1.96 × SEM). Drives
    # how the per-seed mean curve's shaded band is computed across system
    # prompts (each variant contributes its ``cos_sim_across_layers`` at its
    # own best position under ``cfg.rotation_metric``).
    cos_sim_curves_band_kind: str = "std"
    # Line + band aesthetics for the paper-plot ``cos_sim_curves`` figure.
    # Same color/alpha conventions as the class-separation styling above.
    cos_sim_curves_line_color: str = "steelblue"
    cos_sim_curves_line_width: float = 2.0
    cos_sim_curves_band_color: str = "steelblue"
    cos_sim_curves_band_alpha: float = 0.2
    # Y-axis range ``[lo, hi]`` for the paper-plot ``cos_sim_curves`` figure.
    cos_sim_curves_y_range: List[float] = field(
        default_factory=lambda: [-1.05, 1.05]
    )

    # Per-figure on/off for the paper-plot stage. Keys must match the
    # canonical figure slugs in :mod:`src.paper_plots.generate_paper_plots`
    # (``cos_sim_heatmap``, ``best_token_position``, ``rotation_distribution``,
    # ``accuracy_vs_rotation_scatter``, ``cross_dataset_scatter``,
    # ``behavior_bar_box``, ``class_separation``, ``cos_sim_curves``). Missing
    # keys default to ``True`` so dropping a key from the YAML never silently
    # disables a figure — set the value to ``false`` explicitly.
    paper_plots_to_generate: Dict[str, bool] = field(
        default_factory=lambda: {
            "cos_sim_heatmap": True,
            "best_token_position": True,
            "rotation_distribution": True,
            "accuracy_vs_rotation_scatter": True,
            "cross_dataset_scatter": True,
            "behavior_bar_box": True,
            "class_separation": True,
            "cos_sim_curves": True,
        }
    )
    # Per-figure output formats for the paper-plot stage. Same keys as
    # :attr:`paper_plots_to_generate`; values are lists of formats accepted by
    # :func:`src.paper_plots.figures._write_figure` (any combo of ``pdf`` /
    # ``svg`` / ``eps`` / ``png`` / ``jpg`` / ``webp`` / ``html``). Missing
    # or empty entries fall back to each figure function's built-in default
    # (``["svg", "png"]`` for the cos-sim heatmap, ``["pdf", "svg"]`` for the
    # rest), preserving the project's prior on-disk convention. The
    # ``class_separation`` entry falls back to
    # :attr:`class_separation_plot_formats` when missing.
    paper_plot_formats: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "cos_sim_heatmap": ["svg", "png"],
            "best_token_position": ["pdf", "svg"],
            "rotation_distribution": ["pdf", "svg"],
            "accuracy_vs_rotation_scatter": ["pdf", "svg"],
            "cross_dataset_scatter": ["pdf", "svg"],
            "behavior_bar_box": ["pdf", "svg"],
            "class_separation": ["png", "svg"],
            "cos_sim_curves": ["pdf", "svg"],
        }
    )
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])

    @property
    def model_alias(self) -> str:
        """Filesystem-safe alias derived from ``model_path`` (last path segment)."""
        return self.model_path.rstrip("/").split("/")[-1]

    @property
    def sys_prompt_dir_path(self) -> Path:
        """Absolute directory containing ``*_system_prompts.json`` / variant banks."""
        p = Path(self.sys_prompt_dir_relative)
        return p if p.is_absolute() else (self.project_root / p).resolve()

    @property
    def output_root(self) -> Path:
        p = Path(self.output_root_relative)
        return p if p.is_absolute() else (self.project_root / p).resolve()

    @property
    def plot_root(self) -> Path:
        p = Path(self.plot_root_relative)
        return p if p.is_absolute() else (self.project_root / p).resolve()

    def task_dataset_slug(self) -> str:
        """Directory segment ``{task}_{dataset}`` for artifact grouping."""
        return f"{self.task}_{self.dataset}"

    def variant_indices(self, n_variants: int) -> "Sequence[int]":
        """
        Variant indices to iterate this run.

        Precedence: ``variant_subset`` (explicit list, possibly non-contiguous)
        wins if non-None and non-empty; otherwise falls back to the inclusive
        ``[start_variant, end_variant]`` range clamped to ``[0, n_variants-1]``.
        Both range endpoints default to "no bound", giving every variant.

        Args:
            n_variants: Total number of variants in the experiment-prompt bank.

        Returns:
            Sequence (``range`` or ``list``) of system-prompt indices.
        """
        if self.variant_subset:
            return sorted({i for i in self.variant_subset if 0 <= int(i) < n_variants})
        s = 0 if self.start_variant is None else max(0, int(self.start_variant))
        e = (
            n_variants - 1
            if self.end_variant is None
            else min(n_variants - 1, int(self.end_variant))
        )
        if s > e:
            return range(0)
        return range(s, e + 1)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], project_root: Optional[Path] = None) -> ExperimentConfig:
        """
        Build :class:`ExperimentConfig` from a flat config dict.

        Args:
            raw: Flat mapping of dataclass field name -> value (typically the
                output of :func:`src.config.loader._flatten_section_groups`).
                A leftover ``_comment`` top-level key, if present, is dropped.
            project_root: Override repository root (defaults to ``src.paths.PROJECT_ROOT``).

        Returns:
            Frozen logical config with nested experiment-system-prompt specs.
        """
        root = project_root or Path(__file__).resolve().parents[2]
        data: MutableMapping[str, Any] = dict(raw)
        data.pop("_comment", None)
        substituted = _substitute_project_root(dict(data), root)

        esp_raw = substituted.pop("experiment_system_prompts", {})
        experiment_system_prompts: Dict[str, ExperimentSystemPromptSpec] = {}
        for task_name, spec in esp_raw.items():
            experiment_system_prompts[task_name] = ExperimentSystemPromptSpec(
                filename=str(spec["filename"]),
                variants_json_key=str(spec["variants_json_key"]),
            )

        substituted["experiment_system_prompts"] = experiment_system_prompts
        substituted["project_root"] = root
        return cls(**substituted)


def merge_dict(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Deep-merge ``override`` into ``base`` (dict values recurse)."""
    for k, v in override.items():
        if (
            k in base
            and isinstance(base[k], dict)
            and isinstance(v, dict)
        ):
            merge_dict(base[k], v)  # type: ignore[arg-type]
        else:
            base[k] = v
    return base
