"""Load and merge YAML experiment configuration from disk."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, FrozenSet, Mapping, MutableMapping, Optional

import yaml

from src.config.schema import ExperimentConfig, merge_dict
from src.paths import PROJECT_ROOT


# Top-level YAML keys treated as cosmetic groups: their nested keys get
# flattened up to the top level before constructing ExperimentConfig.
# These names are deliberately distinct from every dataclass field name in
# ExperimentConfig, so the flatten step never collides with a real field.
# Files may also be fully flat (no group wrappers) — non-section keys pass
# through unchanged.
_SECTION_GROUPS: FrozenSet[str] = frozenset({
    "model",
    "task_setup",
    "prompt",
    "tokens",
    "batching",
    "behavior",
    "cache",
    "variants",
    "analysis",
    "pca_plots",
    "manifold_plots",
    "class_separation",
    "paper_plots",
    "gt_labels",
    "system_prompts",
    "datasets",
    "paths",
})


def _read_yaml(path: Path) -> MutableMapping[str, Any]:
    """Read a YAML file and return its top-level mapping (or ``{}`` if empty)."""
    with open(path, encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(
            f"Top-level YAML in {path} must be a mapping; got {type(loaded).__name__}."
        )
    return loaded


def _flatten_section_groups(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Inline the contents of any cosmetic top-level group (see
    :data:`_SECTION_GROUPS`) into a single flat dict suitable for
    :meth:`ExperimentConfig.from_dict`.

    Keys outside :data:`_SECTION_GROUPS` are passed through unchanged, so a
    fully-flat YAML file still works. Raises :class:`ValueError` if the same
    flat field name would be set both inside a section and elsewhere in the
    file (a likely authoring mistake).
    """
    flat: Dict[str, Any] = {}
    for key, value in raw.items():
        if key in _SECTION_GROUPS and isinstance(value, dict):
            for sub_key, sub_val in value.items():
                if sub_key in flat:
                    raise ValueError(
                        f"Duplicate config field {sub_key!r}: defined under "
                        f"section {key!r} and also elsewhere in the same file."
                    )
                flat[sub_key] = sub_val
        else:
            if key in flat:
                raise ValueError(
                    f"Duplicate config field {key!r} in YAML file."
                )
            flat[key] = value
    return flat


def load_yaml_config(
    path: Path,
    override_path: Optional[Path] = None,
    override_dict: Optional[Mapping[str, Any]] = None,
    project_root: Optional[Path] = None,
) -> ExperimentConfig:
    """
    Load a YAML config and apply optional layered overrides.

    Both the base file and any override file may use cosmetic section
    groups (model, prompt, pca_plots, manifold_plots, ...) — each file is
    flattened independently before merging, so files can mix grouped and
    flat forms freely.

    Args:
        path: Primary YAML file (e.g. ``PROJECT_ROOT / "config/default.yaml"``).
        override_path: Optional second YAML merged on top (CLI ``--config_override``).
        override_dict: Optional in-memory overrides keyed by *flat* field
            name (typically built by :mod:`src.config.cli` from ``--set``).
        project_root: Repository root for substitution and path resolution.

    Returns:
        Parsed :class:`ExperimentConfig`.
    """
    root = project_root or PROJECT_ROOT
    base_flat = _flatten_section_groups(_read_yaml(path))

    if override_path is not None and override_path.exists():
        merge_dict(base_flat, _flatten_section_groups(_read_yaml(override_path)))

    if override_dict:
        merge_dict(base_flat, dict(override_dict))

    return ExperimentConfig.from_dict(base_flat, project_root=root)


def default_config_path() -> Path:
    """Return the canonical path to ``config/default.yaml``."""
    return PROJECT_ROOT / "config" / "default.yaml"


# Backwards-compat alias for the prior JSON-based loader name. New code
# should call :func:`load_yaml_config` directly.
load_json_config = load_yaml_config
