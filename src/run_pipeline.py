"""
End-to-end pipeline orchestrator: behavior → cache → analysis → paper plots.

This is the top-level CLI for running one or more stages back-to-back against
a single :class:`~src.config.schema.ExperimentConfig`. Each stage is a thin
dispatch into the package that implements it. 
Stages are executed sequentially in the order given on the
command line, and a failing stage aborts the rest of the run.

Stages
------
- ``behavior`` — control + experiment generations (:mod:`src.behavior.run_behavior`).
- ``cache``    — generate balanced-set activation pickles (:mod:`src.cache_activations`).
- ``analysis`` — PCA, cosine similarity heatmaps & rotation metrics
  (:mod:`src.analysis.run_analysis`).
- ``plots``    — generate paper figures 
  (:mod:`src.paper_plots.generate_paper_plots`).

Examples
--------
Run only the behavior stage with the default config::

    python src/run_pipeline.py --steps behavior

Run the full pipeline with a config override applied on top of the default::

    python src/run_pipeline.py \\
        --steps behavior,cache,analysis,plots \\
        --config_override config/sycophancy_llama8b.json

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analysis.run_analysis import run_analysis_for_config
from src.behavior.paths import behavior_run_dir
from src.behavior.run_behavior import run_behavior_for_config
from src.cache_activations.cache_activations import run_cache_activations_for_config
from src.config.cli import add_common_args, load_config_from_cli
from src.config.run_record import dump_run_record
from src.paper_plots.generate_paper_plots import generate_paper_plots_for_config

_STAGES = {
    "behavior": run_behavior_for_config,
    "cache": run_cache_activations_for_config,
    "analysis": run_analysis_for_config,
    "plots": generate_paper_plots_for_config,
}


def _run_steps(cfg, steps: Iterable[str]) -> None:
    """Dispatch the requested pipeline stages in order; raise on unknown names."""
    for step in steps:
        fn = _STAGES.get(step)
        if fn is None:
            raise ValueError(
                f"Unknown pipeline step: {step!r}. "
                f"Valid steps: {sorted(_STAGES)}."
            )
        fn(cfg)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse pipeline CLI arguments (config paths + ordered stage list)."""
    p = argparse.ArgumentParser(description="Run rotation pipeline stages end-to-end.")
    add_common_args(p)
    p.add_argument(
        "--steps",
        type=str,
        default="behavior",
        help=(
            "Comma-separated subset of "
            f"{','.join(_STAGES)}; executed in the given order."
        ),
    )
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    """CLI entry point: load (and merge) config, then run the requested stages."""
    args = parse_args(argv)
    cfg = load_config_from_cli(args)
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    if not steps:
        raise ValueError("--steps must list at least one stage.")
    dump_run_record(
        cfg,
        out_dir=behavior_run_dir(cfg),
        argv=sys.argv,
        stage="pipeline",
        steps=steps,
    )
    _run_steps(cfg, steps)


if __name__ == "__main__":
    main()
