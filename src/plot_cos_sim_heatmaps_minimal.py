"""Re-render the cross-prompt cosine heatmaps in a minimal style.

Loads the same rotation aggregates as the paper-plot stage and calls
:func:`src.paper_plots.figures.plot_cos_sim_heatmap_across_prompts` with:

  - ``label_format="prompt_only"``  → y-tick = raw system-prompt text only
    (no leading variant id, no ``[best_position]`` chip)
  - ``show_metric_block=False``     → drop the trailing ``r`` column
  - ``show_title=False``            → drop the figure title

Output goes to the same directory as the originals, but with a ``_minimal``
suffix on the basename so existing PNG/SVG files are not overwritten.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.cli import add_common_args, load_config_from_cli
from src.paper_plots.aggregate import load_rotation_aggregate_for_seed
from src.paper_plots.figures import plot_cos_sim_heatmap_across_prompts
from src.paper_plots.paths import cos_sim_heatmap_basepath
from src.prompt.chat_template import load_experiment_system_prompts


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument(
        "--suffix",
        type=str,
        default="_minimal",
        help="Filename suffix appended to each basepath (default: '_minimal').",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory override. When set, the figure is "
            "written to '{out_dir}/{basename}{suffix}.<ext>' instead of the "
            "default cos_sim_heatmaps subdir under cfg.plot_root (use this "
            "when the cfg.plot_root location is read-only)."
        ),
    )
    p.add_argument(
        "--png_scale",
        type=float,
        default=2.0,
        help="Kaleido scale factor for the PNG (default 2.0; the paper-plot "
             "default is 4.0, which produces ~5-7 MB files for large heatmaps).",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = load_config_from_cli(args)

    seeds = list(cfg.seed_list_cache or [])
    if not seeds:
        raise SystemExit("seed_list_cache is empty; nothing to plot.")

    variants = load_experiment_system_prompts(cfg)
    fmt_map = getattr(cfg, "paper_plot_formats", None) or {}
    formats = list(fmt_map.get("cos_sim_heatmap") or []) or None

    for seed in seeds:
        seed_data = load_rotation_aggregate_for_seed(cfg, seed, variants)
        if seed_data.is_empty:
            print(f"  skip seed={seed}: no rotation_stats.json across variants")
            continue
        base = cos_sim_heatmap_basepath(cfg, seed)
        if args.out_dir is not None:
            out_base = args.out_dir / (base.name + args.suffix)
        else:
            out_base = base.with_name(base.name + args.suffix)
        out_base.parent.mkdir(parents=True, exist_ok=True)
        plot_cos_sim_heatmap_across_prompts(
            seed_data,
            basepath=out_base,
            title="",  # unused (show_title=False) but required by the signature
            metric_display_mode=cfg.heatmap_r_metric_display,
            label_display=cfg.heatmap_label_display,
            formats=formats,
            label_format="prompt_only",
            show_metric_block=False,
            show_title=False,
            label_px_per_char=3.0,
            label_px_pad=0,
            png_scale=args.png_scale,
        )
        print(f"  wrote {out_base}.* ({len(seed_data.variants)} variants)")


if __name__ == "__main__":
    main()
