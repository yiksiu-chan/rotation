"""Configuration loading for the rotation pipeline."""

from src.config.cli import (
    add_common_args,
    apply_cli_overrides,
    apply_variant_range_overrides,
)
from src.config.loader import default_config_path, load_json_config, load_yaml_config
from src.config.schema import ExperimentConfig, ExperimentSystemPromptSpec

__all__ = [
    "ExperimentConfig",
    "ExperimentSystemPromptSpec",
    "load_yaml_config",
    "load_json_config",  # backwards-compat alias for load_yaml_config
    "default_config_path",
    "add_common_args",
    "apply_cli_overrides",
    "apply_variant_range_overrides",  # alias for backwards compat
]
