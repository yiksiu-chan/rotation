"""Filesystem locations for cache-stage artifacts."""

from __future__ import annotations

from pathlib import Path

from src.behavior.paths import behavior_run_dir
from src.config.schema import ExperimentConfig


def cache_seed_dir(cfg: ExperimentConfig, system_prompt_id: int, seed: int) -> Path:
    """
    Return ``{output_root}/{model_alias}/{task}_{dataset}/SysPrompt{id:03d}/{seed}_pre{n}_gen{m}``.

    The seed-named subfolder fully namespaces one cache-stage run; multiple
    seeds for the same system prompt live side-by-side. The
    ``cache_sample_method`` is encoded into the per-file names rather than the
    directory so the two methods can coexist in the same seed folder.
    """
    name = f"{seed}_pre{cfg.n_pre_prompt_tokens}_gen{cfg.n_generated_tokens}"
    return behavior_run_dir(cfg) / f"SysPrompt{system_prompt_id:03d}" / name


def _method_tag(cfg: ExperimentConfig) -> str:
    """
    File-name infix encoding ``cache_sample_method``.

    Returns ``"_{method}"`` for the standard sampling modes, or an empty
    string when ``cfg.n_sample_cache == -1`` — sampling is disabled in that
    case so the method label would be misleading.
    """
    if cfg.n_sample_cache == -1:
        return ""
    return f"_{cfg.cache_sample_method}"


def completions_path(
    cfg: ExperimentConfig,
    system_prompt_id: int,
    seed: int,
    condition: str,
    fail: bool = False,
) -> Path:
    """
    Path to ``completions/completions_{condition}_{cache_sample_method}[_fail].json``.

    When ``cfg.n_sample_cache == -1`` the ``_{cache_sample_method}`` tag is
    omitted (sampling is disabled, so the method is irrelevant) and the
    file becomes ``completions_{condition}[_fail].json``.

    The ``_fail`` suffix selects the experiment-only subset cached for
    statements that succeeded in control but **failed** the instruction in
    experiment (``output_label == 0``). The base file is always written; the
    ``_fail`` companion exists only when ``S_only_fail_in_experiment`` was
    non-empty for that ``(SysPrompt, seed)``.
    """
    suffix = "_fail" if fail else ""
    return (
        cache_seed_dir(cfg, system_prompt_id, seed)
        / "completions"
        / f"completions_{condition}{_method_tag(cfg)}{suffix}.json"
    )


def activations_path(
    cfg: ExperimentConfig,
    system_prompt_id: int,
    seed: int,
    condition: str,
    gt_label: int,
    fail: bool = False,
) -> Path:
    """
    Path to ``activations/activations_{condition}_{cache_sample_method}_GT_{label}[_fail].pkl``.

    When ``cfg.n_sample_cache == -1`` the ``_{cache_sample_method}`` tag is
    omitted (see :func:`completions_path`) and the file becomes
    ``activations_{condition}_GT_{label}[_fail].pkl``.

    See :func:`completions_path` for the ``fail`` semantics.
    """
    suffix = "_fail" if fail else ""
    return (
        cache_seed_dir(cfg, system_prompt_id, seed)
        / "activations"
        / f"activations_{condition}{_method_tag(cfg)}_GT_{gt_label}{suffix}.pkl"
    )
