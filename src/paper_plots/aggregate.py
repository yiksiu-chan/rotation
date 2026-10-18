"""
Cross-system-prompt aggregation helpers for the paper-plot stage.

Reads upstream artifacts produced by the analysis stage
(``rotation_stats.json``) and the behavior stage
(``behavior_results_{control,experiment}.json``), then reshapes them into
arrays keyed by system-prompt index. The Plotly figure builders in
:mod:`src.paper_plots.figures` consume these aggregates directly.

Each :class:`VariantRotation` is keyed at *that variant's* best token position
under ``cfg.rotation_metric`` — i.e. the chosen rotation metric drives both
the displayed cosine curve and downstream row ordering. Per-metric scores at
the same position are also retained so the cross-prompt heatmap can show an
``r1, r2, …`` value block.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from src.analysis.paths import (
    class_separation_binary_stats_path,
    class_separation_per_class_stats_path,
    class_separation_stats_path,
    rotation_stats_path,
)
from src.analysis.rotation_plugins import (
    ROTATION_METRIC_ORDER,
    resolve_rotation_metric,
)
from src.behavior.paths import control_behavior_path, experiment_behavior_path
from src.config.schema import ExperimentConfig


@dataclass
class VariantRotation:
    """Per-variant slice of one ``rotation_stats.json``, keyed at the variant's best position."""

    system_prompt_id: int
    best_position_label: str
    rotation_r: float                         # value of cfg.rotation_metric at best position
    rotation_r_by_metric: Dict[str, float]    # all registered metrics at that position
    cos_sim_across_layers: List[float]
    system_prompt_text: str = ""              # raw content from the experiment-prompt JSON bank


@dataclass
class SeedAggregate:
    """All variants' rotation summaries for one seed (only those with `rotation_stats.json`)."""

    seed: int
    driving_metric_name: str
    variants: List[VariantRotation] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.variants


@dataclass
class ClassSeparationVariant:
    """Per-variant slice of one ``class_separation_stats_nPC{n}.json``."""

    system_prompt_id: int
    token_position: str
    cv_accuracy_across_layers: List[float]


@dataclass
class ClassSeparationAggregate:
    """All variants' 4-class CV accuracy curves for one seed."""

    seed: int
    n_PC_classifier: int
    variants: List[ClassSeparationVariant] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.variants


@dataclass
class ClassSeparationPerClassVariant:
    """Per-variant slice of one ``class_separation_per_class_stats_nPC{n}.json``."""

    system_prompt_id: int
    token_position: str
    # class name (control_GT_0 / control_GT_1 / experiment_GT_0 / experiment_GT_1)
    # -> length-n_layers list of per-class CV recalls.
    cv_recall_per_class_across_layers: Dict[str, List[float]] = field(
        default_factory=dict
    )


@dataclass
class ClassSeparationPerClassAggregate:
    """All variants' per-class CV recall curves for one seed."""

    seed: int
    n_PC_classifier: int
    variants: List[ClassSeparationPerClassVariant] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.variants


@dataclass
class ClassSeparationBinaryVariant:
    """Per-variant slice of one ``class_separation_binary_stats_nPC{n}.json``."""

    system_prompt_id: int
    token_position: str
    # "control" / "experiment" -> length-n_layers list of binary CV accuracies.
    cv_accuracy_per_condition_across_layers: Dict[str, List[float]] = field(
        default_factory=dict
    )


@dataclass
class ClassSeparationBinaryAggregate:
    """All variants' per-condition binary CV accuracy curves for one seed."""

    seed: int
    n_PC_classifier: int
    variants: List[ClassSeparationBinaryVariant] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.variants


@dataclass
class BehaviorAggregate:
    """Behavior accuracies aggregated across system prompts (independent of cache seed)."""

    accuracy_control: Optional[float]
    accuracy_experiments: List[float]
    variant_ids_with_experiment: List[int]

    @property
    def has_data(self) -> bool:
        return self.accuracy_control is not None or bool(self.accuracy_experiments)


# --------------------------------------------------------------------------- #
# Loaders                                                                     #
# --------------------------------------------------------------------------- #


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_rotation_aggregate_for_seed(
    cfg: ExperimentConfig,
    seed: int,
    system_prompts: List[str],
) -> SeedAggregate:
    """
    Walk the configured variant range for one seed and collect available
    rotation summaries, keyed at each variant's ``cfg.rotation_metric`` best
    token position.

    Args:
        cfg: Experiment config. Honors ``cfg.start_variant`` / ``cfg.end_variant``.
        seed: Cache/analysis seed to aggregate over.
        system_prompts: Ordered list of experiment system prompts (the JSON
            bank content). Length defines the variant index range; each
            ``VariantRotation.system_prompt_text`` is populated from this list.

    Returns:
        :class:`SeedAggregate`. Variants without a ``rotation_stats.json`` are
        silently skipped — the aggregate may end up empty for partially
        analyzed sweeps, in which case no figure is written by the orchestrator.
    """
    chosen_name, _ = resolve_rotation_metric(cfg.rotation_metric)
    best_key = f"best_token_pos_{chosen_name}"

    out = SeedAggregate(seed=seed, driving_metric_name=chosen_name)
    for vid in cfg.variant_indices(len(system_prompts)):
        path = rotation_stats_path(cfg, vid, seed)
        if not path.exists():
            continue
        stats = _load_json(path)
        best = stats.get(best_key)
        if best is None or best not in stats:
            continue
        bucket = stats[best]
        rot_r_by_metric_raw = bucket.get("rotation_r", {})
        # Per-position rotation_r is a dict {metric_name: float} in the new
        # schema. Keep only registered metrics so downstream display order is
        # deterministic.
        rot_r_by_metric = {
            m: float(rot_r_by_metric_raw[m])
            for m in ROTATION_METRIC_ORDER
            if m in rot_r_by_metric_raw
        }
        out.variants.append(
            VariantRotation(
                system_prompt_id=vid,
                best_position_label=str(best),
                rotation_r=float(rot_r_by_metric.get(chosen_name, 0.0)),
                rotation_r_by_metric=rot_r_by_metric,
                cos_sim_across_layers=[float(x) for x in bucket["cos_sim_across_layers"]],
                system_prompt_text=str(system_prompts[vid]),
            )
        )
    return out


def load_class_separation_aggregate_for_seed(
    cfg: ExperimentConfig,
    seed: int,
    system_prompts: List[str],
) -> ClassSeparationAggregate:
    """
    Walk the configured variant range for one seed and collect available
    classifier-accuracy curves from
    ``stage_stats/class_separation_stats_nPC{n}.json`` (the ``n_PC`` suffix
    follows ``cfg.n_PC_classifier``).

    Args:
        cfg: Experiment config. Honors ``cfg.start_variant`` /
            ``cfg.end_variant``; the ``n_PC_classifier`` field picks which
            stats file is read.
        seed: Cache/analysis seed to aggregate over.
        system_prompts: Ordered list of experiment system prompts; only its
            length is consulted (to drive ``variant_indices``).

    Returns:
        :class:`ClassSeparationAggregate`. Variants without a stats file are
        silently skipped — the aggregate may be empty for partially-analyzed
        sweeps, in which case the paper-plot orchestrator skips the figure.
    """
    out = ClassSeparationAggregate(
        seed=seed, n_PC_classifier=int(cfg.n_PC_classifier)
    )
    for vid in cfg.variant_indices(len(system_prompts)):
        path = class_separation_stats_path(cfg, vid, seed)
        if not path.exists():
            continue
        d = _load_json(path)
        out.variants.append(
            ClassSeparationVariant(
                system_prompt_id=vid,
                token_position=str(d.get("token_position", "")),
                cv_accuracy_across_layers=[
                    float(x) for x in d.get("cv_accuracy_across_layers", [])
                ],
            )
        )
    return out


def load_class_separation_per_class_aggregate_for_seed(
    cfg: ExperimentConfig,
    seed: int,
    system_prompts: List[str],
) -> ClassSeparationPerClassAggregate:
    """
    Parallel to :func:`load_class_separation_aggregate_for_seed` but reads
    ``class_separation_per_class_stats_nPC{n}.json`` and keeps the per-class
    recall dict for each variant.
    """
    out = ClassSeparationPerClassAggregate(
        seed=seed, n_PC_classifier=int(cfg.n_PC_classifier)
    )
    for vid in cfg.variant_indices(len(system_prompts)):
        path = class_separation_per_class_stats_path(cfg, vid, seed)
        if not path.exists():
            continue
        d = _load_json(path)
        per_class_raw = d.get("cv_recall_per_class_across_layers", {}) or {}
        per_class: Dict[str, List[float]] = {
            str(cls_name): [float(x) for x in vals]
            for cls_name, vals in per_class_raw.items()
        }
        out.variants.append(
            ClassSeparationPerClassVariant(
                system_prompt_id=vid,
                token_position=str(d.get("token_position", "")),
                cv_recall_per_class_across_layers=per_class,
            )
        )
    return out


def load_class_separation_binary_aggregate_for_seed(
    cfg: ExperimentConfig,
    seed: int,
    system_prompts: List[str],
) -> ClassSeparationBinaryAggregate:
    """
    Parallel loader for the per-condition binary classifier stats — reads
    ``class_separation_binary_stats_nPC{n}.json`` and keeps the
    ``"control"`` / ``"experiment"`` arrays for each variant.
    """
    out = ClassSeparationBinaryAggregate(
        seed=seed, n_PC_classifier=int(cfg.n_PC_classifier)
    )
    for vid in cfg.variant_indices(len(system_prompts)):
        path = class_separation_binary_stats_path(cfg, vid, seed)
        if not path.exists():
            continue
        d = _load_json(path)
        per_cond_raw = d.get("cv_accuracy_per_condition_across_layers", {}) or {}
        per_cond: Dict[str, List[float]] = {
            str(cond): [float(x) for x in vals]
            for cond, vals in per_cond_raw.items()
        }
        out.variants.append(
            ClassSeparationBinaryVariant(
                system_prompt_id=vid,
                token_position=str(d.get("token_position", "")),
                cv_accuracy_per_condition_across_layers=per_cond,
            )
        )
    return out


def load_behavior_aggregate(
    cfg: ExperimentConfig, n_variants: int
) -> BehaviorAggregate:
    """
    Read the behavior stage's control + per-variant experiment accuracies.

    Honors ``cfg.start_variant`` / ``cfg.end_variant`` when iterating over
    experiment variants. Missing files are tolerated: ``accuracy_control`` is
    ``None`` if the control JSON is absent, and ``accuracy_experiments``
    contains only the variants whose experiment JSON exists (with
    ``variant_ids_with_experiment`` holding the matching indices).
    """
    ctrl_path = control_behavior_path(cfg)
    accuracy_control: Optional[float] = None
    if ctrl_path.exists():
        accuracy_control = float(_load_json(ctrl_path).get("Accuracy_control"))

    accuracy_experiments: List[float] = []
    variant_ids: List[int] = []
    per_variant_controls: List[float] = []
    for vid in cfg.variant_indices(n_variants):
        ep = experiment_behavior_path(cfg, vid)
        if not ep.exists():
            continue
        d = _load_json(ep)
        if "Accuracy_experiment" in d:
            accuracy_experiments.append(float(d["Accuracy_experiment"]))
            variant_ids.append(vid)
        if accuracy_control is None:
            cp = control_behavior_path(cfg, system_prompt_id=vid)
            if cp.exists():
                cd = _load_json(cp)
                if "Accuracy_control" in cd:
                    per_variant_controls.append(float(cd["Accuracy_control"]))

    # Fallback: tasks like hallucination keep control per-variant rather than at
    # the run root. Average across variants so the figure has a control bar.
    if accuracy_control is None and per_variant_controls:
        accuracy_control = sum(per_variant_controls) / len(per_variant_controls)

    return BehaviorAggregate(
        accuracy_control=accuracy_control,
        accuracy_experiments=accuracy_experiments,
        variant_ids_with_experiment=variant_ids,
    )
