"""Activation analysis: cosine matrices, PCA scatter grids, rotation metrics."""

from src.analysis.rotation_plugins import (
    DEFAULT_ROTATION_METRIC_NAME,
    ROTATION_METRIC_ORDER,
    ROTATION_METRIC_REGISTRY,
    largest_drop_any_early_to_late,
    largest_drop_any_early_to_late_signed,
    late_layer_anti_alignment,
    metric_short_label,
    resolve_rotation_metric,
)

def __getattr__(name):
    # Importing numerical metrics must not load the GPU/plotting pipeline.
    if name in {"analysis_main", "run_analysis_for_config"}:
        from src.analysis import run_analysis
        return run_analysis.main if name == "analysis_main" else run_analysis.run_analysis_for_config
    raise AttributeError(name)


__all__ = [
    "DEFAULT_ROTATION_METRIC_NAME",
    "ROTATION_METRIC_ORDER",
    "ROTATION_METRIC_REGISTRY",
    "largest_drop_any_early_to_late",
    "largest_drop_any_early_to_late_signed",
    "late_layer_anti_alignment",
    "metric_short_label",
    "resolve_rotation_metric",
    "run_analysis_for_config",
    "analysis_main",
]
