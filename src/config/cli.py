"""
Shared CLI helpers for the per-stage entry points.

All entry points use :func:`load_config_from_cli` so the same flag surface is
available everywhere:

- ``--config`` / ``--config_override`` — base + layered YAML paths.
- An auto-generated ``--<field>`` flag for **every** :class:`ExperimentConfig`
  field, derived directly from the dataclass annotations (``--task``,
  ``--dataset``, ``--n_pre_prompt_tokens``, ``--seed_list_cache``, …).
  Scalars (``int`` / ``float`` / ``str`` / ``bool``) parse natively; lists
  and dicts JSON-parse (e.g. ``--seed_list_cache '[0,1,2]'``).
- A few **typed convenience flags** that override their auto-generated
  counterparts with a friendlier surface (``--pca_token_positions`` accepts
  comma-separated labels; ``--heatmap_r_metric_display`` is constrained to
  ``{all, chosen_metric}``). When a typed flag exists for a field, the
  auto-generated one is suppressed.
- A generic, repeatable ``--set FIELD=VALUE`` that also reaches **nested**
  fields via dotted paths (e.g. ``--set pca_plot_style.marker_alpha=0.4``).
  Values are JSON-parsed when possible and fall back to a string literal
  otherwise. Use this for nested overrides; for top-level fields the
  ``--<field>`` form is more discoverable.

Precedence (later wins): base YAML → ``--config_override`` YAML → ``--set``
deep-merge → auto-generated ``--<field>`` flags → typed convenience flags.
So ``--set start_variant=5`` is overridden by ``--start_variant 10``.

``--set`` and ``--<field>`` flags target *flat* dataclass field names
(e.g. ``--set start_variant=5``), independent of whether the YAML files
use cosmetic section groups (``variants:`` / ``pca_plots:`` / ...).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import typing
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, MutableMapping, Sequence

from src.config.loader import default_config_path, load_yaml_config
from src.config.schema import ExperimentConfig, merge_dict


# --------------------------------------------------------------------------- #
# Auto-generated --<field> flags                                              #
# --------------------------------------------------------------------------- #


# Field names handled below by typed convenience flags or that should never
# be exposed via auto-generated CLI args (internal / overly complex types).
# Auto-generation skips these so we don't double-register their flag names.
_AUTO_FIELD_SKIP: FrozenSet[str] = frozenset({
    # Internal / derived
    "project_root",
    # Nested dataclass-of-dataclass; use --config_override or --set for these.
    "experiment_system_prompts",
    # Have typed convenience flags below with friendlier parsing:
    "pca_token_positions",       # comma-separated string instead of JSON list
    "heatmap_r_metric_display",  # restricted to choices=…
    "rotation_metric",           # plain string today, but callout in --help
})


def _argparse_type_for(annotation: Any) -> Callable[[str], Any]:
    """
    Pick an argparse ``type=`` coercer from a config-field annotation.

    Scalars map to themselves (``int``/``float``/``str``); ``bool`` accepts
    JSON-style ``true``/``false`` (case-insensitive); ``Optional[X]`` peels
    off the ``None`` and recurses; everything else (lists, dicts, complex)
    JSON-parses.
    """
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if origin is typing.Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _argparse_type_for(non_none[0])

    if annotation is int:
        return int
    if annotation is float:
        return float
    if annotation is str:
        return str
    if annotation is bool:
        return lambda s: bool(json.loads(s.lower()))
    return json.loads


def _add_auto_field_flags(parser: argparse.ArgumentParser) -> None:
    """
    Register a ``--<field>`` flag for every overridable
    :class:`ExperimentConfig` dataclass field (skipping
    :data:`_AUTO_FIELD_SKIP`).

    Annotations are resolved via :func:`typing.get_type_hints` so the
    ``from __future__ import annotations``-style stringified types still
    map to real classes.
    """
    hints = typing.get_type_hints(ExperimentConfig)
    for f in dataclasses.fields(ExperimentConfig):
        if f.name in _AUTO_FIELD_SKIP:
            continue
        annot = hints.get(f.name, str)
        coercer = _argparse_type_for(annot)
        # Friendly type label in --help; falls back to repr() for complex types.
        type_label = getattr(annot, "__name__", None) or str(annot).replace("typing.", "")
        parser.add_argument(
            f"--{f.name}",
            type=coercer,
            default=None,
            metavar=type_label.upper(),
            help=(
                f"Override cfg.{f.name} ({type_label}). "
                "Auto-generated from the ExperimentConfig dataclass."
            ),
        )


# --------------------------------------------------------------------------- #
# Argparse setup                                                              #
# --------------------------------------------------------------------------- #


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """
    Register the shared flags on ``parser``.

    See module docstring for the precedence rules between flags.
    """
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path(),
        help="Path to YAML config (default: config/default.yaml).",
    )
    parser.add_argument(
        "--config_override",
        type=Path,
        default=None,
        help="Optional second YAML merged on top of --config.",
    )
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help=(
            "Override any config field, including nested ones. Repeatable. "
            "FIELD may be dotted to reach nested keys (e.g. "
            "'pca_plot_style.marker_alpha=0.4'). VALUE is JSON-parsed when "
            "possible (so [0,1,2], true, null, 42 keep their types); "
            "otherwise treated as a raw string. Applied before "
            "auto-generated --<field> flags."
        ),
    )

    # Auto-generated per-field overrides (covers every ExperimentConfig
    # field except those in _AUTO_FIELD_SKIP).
    _add_auto_field_flags(parser)

    # Typed convenience flags for the few fields where a friendlier surface
    # is worth the extra registration. These win over the auto-generated
    # flag with the same target field — the auto-generated version is
    # suppressed for these names via _AUTO_FIELD_SKIP.
    parser.add_argument(
        "--pca_token_positions",
        type=str,
        default=None,
        help=(
            "Comma-separated token-position labels (e.g. 'prompt,prompt-1') "
            "to plot PCA grids for in the analysis stage. Overrides "
            "cfg.pca_token_positions and takes precedence over "
            "cfg.save_top_n_tokens_4_PCAplots."
        ),
    )
    parser.add_argument(
        "--rotation_metric",
        type=str,
        default=None,
        help=(
            "Name of the rotation metric used to choose the 'best token "
            "position' (must match a key in ROTATION_METRIC_REGISTRY). "
            "Overrides cfg.rotation_metric. Every registered metric is "
            "still computed; this only selects which one drives PCA "
            "selection and paper-plot row ordering."
        ),
    )
    parser.add_argument(
        "--heatmap_r_metric_display",
        type=str,
        default=None,
        choices=["all", "chosen_metric"],
        help=(
            "Which metric columns appear in the trailing r-block of the "
            "cosine-similarity heatmaps. 'all' = every registered metric; "
            "'chosen_metric' = only cfg.rotation_metric. Overrides "
            "cfg.heatmap_r_metric_display."
        ),
    )


# --------------------------------------------------------------------------- #
# --set parsing                                                               #
# --------------------------------------------------------------------------- #


def _parse_set_value(raw: str) -> Any:
    """Parse a ``--set`` value as JSON, falling back to a raw string literal."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


def _insert_dotted(target: MutableMapping[str, Any], key_path: str, value: Any) -> None:
    """Insert ``value`` at ``a.b.c`` inside ``target``, creating dicts as needed."""
    if not key_path:
        raise ValueError("Empty FIELD in --set; expected non-empty key (or dotted path).")
    parts = key_path.split(".")
    cursor: MutableMapping[str, Any] = target
    for part in parts[:-1]:
        nxt = cursor.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cursor[part] = nxt
        cursor = nxt
    cursor[parts[-1]] = value


def parse_set_overrides(set_args: Sequence[str] | None) -> Dict[str, Any]:
    """
    Build a deep-merge dict from a sequence of ``"FIELD=VALUE"`` strings.

    Args:
        set_args: Output of ``argparse``'s ``--set`` collection (each entry is
            a single ``FIELD=VALUE`` string). ``None`` and ``[]`` both yield
            an empty dict.

    Returns:
        Nested dict suitable for :func:`merge_dict`. Later entries override
        earlier ones; conflicts at non-dict leaves overwrite.

    Raises:
        ValueError: On ``--set`` strings missing ``=`` or with empty keys.
    """
    out: Dict[str, Any] = {}
    for entry in set_args or ():
        if "=" not in entry:
            raise ValueError(
                f"Invalid --set entry {entry!r}: expected FIELD=VALUE."
            )
        key, _, raw_val = entry.partition("=")
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --set entry {entry!r}: empty FIELD before '='.")
        value = _parse_set_value(raw_val)
        # Build a one-key nested dict for this entry, then deep-merge it in
        # so multiple --set entries that share a prefix combine correctly.
        leaf: Dict[str, Any] = {}
        _insert_dotted(leaf, key, value)
        merge_dict(out, leaf)
    return out


# --------------------------------------------------------------------------- #
# Typed convenience override application                                      #
# --------------------------------------------------------------------------- #


def _parse_position_list(s: str) -> List[str]:
    """Split a comma-separated position-label string into a clean list."""
    return [tok.strip() for tok in s.split(",") if tok.strip()]


def _apply_auto_field_overrides(
    cfg: ExperimentConfig, args: argparse.Namespace
) -> None:
    """
    Apply every auto-generated ``--<field>`` value (skipping unset ``None``s
    and the fields in :data:`_AUTO_FIELD_SKIP`).
    """
    for f in dataclasses.fields(ExperimentConfig):
        if f.name in _AUTO_FIELD_SKIP:
            continue
        val = getattr(args, f.name, None)
        if val is not None:
            setattr(cfg, f.name, val)


def apply_cli_overrides(cfg: ExperimentConfig, args: argparse.Namespace) -> None:
    """
    Mutate ``cfg`` in place from parsed CLI args.

    Order:
        1. Auto-generated ``--<field>`` overrides for every dataclass field
           that isn't in :data:`_AUTO_FIELD_SKIP`.
        2. Typed convenience flags below (they win over the auto-generated
           override for their respective fields):

           - ``--pca_token_positions`` (comma-separated parsing)
           - ``--rotation_metric``
           - ``--heatmap_r_metric_display``

    Flags left at their argparse default (``None``) leave ``cfg`` untouched.
    Generic ``--set`` overrides are applied earlier — at JSON-load time
    inside :func:`load_config_from_cli` — so the per-field flags here always
    win over ``--set``.
    """
    _apply_auto_field_overrides(cfg, args)

    raw = getattr(args, "pca_token_positions", None)
    if raw is not None:
        cfg.pca_token_positions = _parse_position_list(raw)
    rot = getattr(args, "rotation_metric", None)
    if rot is not None:
        cfg.rotation_metric = str(rot)
    disp = getattr(args, "heatmap_r_metric_display", None)
    if disp is not None:
        cfg.heatmap_r_metric_display = str(disp)


# --------------------------------------------------------------------------- #
# Unified config-from-args entry                                              #
# --------------------------------------------------------------------------- #


def load_config_from_cli(args: argparse.Namespace) -> ExperimentConfig:
    """
    Build a fully-resolved :class:`ExperimentConfig` from parsed CLI args.

    Loads ``--config`` and ``--config_override``, deep-merges any ``--set``
    overrides, constructs the dataclass, then applies the typed convenience
    flags (so they win over ``--set``). All entry-point ``main`` functions
    funnel through this helper so the precedence rules in this module's
    docstring are enforced uniformly.
    """
    set_overrides = parse_set_overrides(getattr(args, "set_overrides", None))
    cfg = load_yaml_config(
        args.config,
        override_path=args.config_override,
        override_dict=set_overrides or None,
    )
    apply_cli_overrides(cfg, args)
    return cfg


# Backwards-compatible alias — earlier code imported `apply_variant_range_overrides`.
# New code should call `apply_cli_overrides` directly.
apply_variant_range_overrides = apply_cli_overrides
