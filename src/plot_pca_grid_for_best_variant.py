"""Render the per-layer PCA grid for one chosen (config, SysPrompt, position).

Standalone — does not load a model. Reads activations cached by the
cache-activations stage and calls :func:`src.analysis.plots.plot_pca_grid_for_position`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pickle

import numpy as np

from src.analysis.paths import pca_plot_basepath
from src.analysis.plots import plot_pca_grid_for_position, resolve_pca_style
from src.cache_activations.hooks import position_labels
from src.cache_activations.paths import activations_path
from src.config.loader import default_config_path, load_yaml_config


def _load_slab_for_position(pkl_path: Path, position_index: int) -> np.ndarray:
    """Return ``(k, 1, n_layers, d)`` fp32 slab — only the chosen position."""
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    sliced = obj[:, position_index : position_index + 1]  # keep position axis
    arr = sliced.detach().cpu().numpy() if hasattr(sliced, "numpy") else np.asarray(sliced)
    return arr.astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_override", type=Path, required=True,
                        help="Task YAML layered onto config/default.yaml "
                             "(e.g. config/sycophancy.yaml).")
    parser.add_argument("--system_prompt_id", type=int, required=True)
    parser.add_argument("--position", type=str, required=True,
                        help="Token-position label, e.g. 'gen+3'.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_yaml_config(default_config_path(), override_path=args.config_override)
    pos_labels = position_labels(cfg.n_pre_prompt_tokens, cfg.n_generated_tokens)
    if args.position not in pos_labels:
        raise SystemExit(
            f"position {args.position!r} not in pos_labels (n_pre={cfg.n_pre_prompt_tokens}, "
            f"n_gen={cfg.n_generated_tokens}); choices: {pos_labels}"
        )
    position_index = pos_labels.index(args.position)

    activations: dict[str, np.ndarray] = {}
    for cond in ("control", "experiment"):
        for gt in (0, 1):
            p = activations_path(cfg, args.system_prompt_id, args.seed, cond, gt)
            if not p.exists():
                raise SystemExit(f"missing {p}")
            activations[f"{cond}_GT_{gt}"] = _load_slab_for_position(p, position_index)
    # We sliced to a single position, so re-anchor the position index.
    position_index = 0

    style = resolve_pca_style(getattr(cfg, "pca_plot_style", None))
    gt0_name = cfg.gt_0_label_map.get(cfg.task, "GT_0")
    gt1_name = cfg.gt_1_label_map.get(cfg.task, "GT_1")
    legend_label_map = {
        "control_GT_0":    f"control / {gt0_name}",
        "control_GT_1":    f"control / {gt1_name}",
        "experiment_GT_0": f"experiment / {gt0_name}",
        "experiment_GT_1": f"experiment / {gt1_name}",
    }
    title = (
        f"{cfg.task}/{cfg.dataset} | SysPrompt{args.system_prompt_id:03d} | "
        f"seed={args.seed} | token={args.position} | method={cfg.cache_sample_method}"
    )

    basepath = pca_plot_basepath(cfg, args.system_prompt_id, args.seed, args.position)
    plot_pca_grid_for_position(
        activations_by_group=activations,
        position_index=position_index,
        position_label=args.position,
        basepath=basepath,
        style=style,
        title=title,
        legend_label_map=legend_label_map,
        formats=("svg",),
    )
    print(f"wrote {basepath.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
