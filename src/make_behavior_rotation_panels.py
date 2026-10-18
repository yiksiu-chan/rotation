"""Combine behavior bar/boxplot PNGs with cosine-sim curve SVGs side-by-side.

For each task there is a matching pair:
  behavior/behavior_BarBoxplot_{TAG}.png
  cos_sim_curves/cos_sim_curves_{TAG}_seed{SEED}.svg

Outputs:
  behavior_rotation/behavior_rotation_{TAG}.png
"""
from __future__ import annotations

import argparse
import html
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BEHAVIOR_RE = re.compile(r"^behavior_BarBoxplot_(?P<tag>.+)\.png$")
COSSIM_RE = re.compile(r"^cos_sim_curves_(?P<tag>.+)_seed(?P<seed>\d+)\.svg$")

# Map known task prefixes (the part of the tag before the dataset) to the
# display string used in the behavior title.
TASK_DISPLAY: dict[str, str] = {
    "hallucination": "Hallucination",
    "spatial_easy": "Spatial Easy",
    "sycophancy": "Sycophancy",
    "lying": "Lying",
}

FONT_CANDIDATES = (
    "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSansCondensed.ttf",
)


def _task_display_from_tag(tag: str) -> str:
    """Resolve the display name from a tag like ``sycophancy_azaria-mitchell``."""
    for key in sorted(TASK_DISPLAY, key=len, reverse=True):
        if tag == key or tag.startswith(key + "_"):
            return TASK_DISPLAY[key]
    return tag.split("_", 1)[0].capitalize()


def _load_font(size: int) -> ImageFont.ImageFont:
    for p in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(p, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def patch_cos_sim_svg(
    svg_text: str,
    *,
    title: str,
    ylabel: str,
    title_font_size_px: int = 14,
) -> str:
    """Rewrite the Plotly ``gtitle`` and ``ytitle`` text content in-place.

    Also re-centers the title (Plotly left-anchors long titles, which collides
    with the rotated y-axis label once we replace the original verbose title
    with a short one) and rewrites the title font-size so it matches the
    14 px axis labels.
    """
    title_esc = html.escape(title, quote=False)
    ylabel_esc = html.escape(ylabel, quote=False)

    m = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg_text)
    center_x = float(m.group(1)) / 2.0 if m else 380.0

    def _patch_gtitle(match: re.Match[str]) -> str:
        el = match.group(0)
        el = re.sub(r'\bx="[^"]*"', f'x="{center_x:g}"', el, count=1)
        el = re.sub(
            r'text-anchor="[^"]*"', 'text-anchor="middle"', el, count=1
        )
        el = re.sub(
            r'font-size:\s*[\d.]+px',
            f'font-size: {title_font_size_px}px',
            el,
            count=1,
        )
        el = re.sub(r'(>)[^<]*(</text>)', rf'\g<1>{title_esc}\g<2>', el, count=1)
        return el

    svg_text, n_title = re.subn(
        r'<text class="gtitle"[^>]*>[^<]*</text>',
        _patch_gtitle,
        svg_text,
        count=1,
    )
    svg_text, n_y = re.subn(
        r'(<text class="ytitle"[^>]*>)[^<]*(</text>)',
        lambda m: m.group(1) + ylabel_esc + m.group(2),
        svg_text,
        count=1,
    )
    if not n_title:
        print("  [warn] gtitle not found in SVG; title not patched")
    if not n_y:
        print("  [warn] ytitle not found in SVG; y-label not patched")
    return svg_text


def overlay_title_on_png(
    im: Image.Image,
    title: str,
    *,
    font_size_px: int = 48,
    cover_height_frac: float = 0.13,
    text_color: tuple[int, int, int] = (42, 63, 95),
) -> Image.Image:
    """Paint a white strip over the top of ``im`` and render ``title`` centered.

    The white strip is sized by ``cover_height_frac`` (large enough to fully
    erase Plotly's original title region); the title is rendered at
    ``font_size_px``, shrunk only if it doesn't fit the image width.
    """
    out = im.convert("RGB").copy()
    W, H = out.size
    cover_h = int(round(H * cover_height_frac))
    draw = ImageDraw.Draw(out)
    draw.rectangle([(0, 0), (W, cover_h)], fill=(255, 255, 255))

    max_w = int(W * 0.92)
    font_size = max(12, int(font_size_px))
    while font_size > 12:
        font = _load_font(font_size)
        bbox = draw.textbbox((0, 0), title, font=font)
        if bbox[2] - bbox[0] <= max_w:
            break
        font_size -= 4
    else:
        font = _load_font(font_size)
        bbox = draw.textbbox((0, 0), title, font=font)

    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (W - tw) // 2 - bbox[0]
    y = (cover_h - th) // 2 - bbox[1]
    draw.text((x, y), title, fill=text_color, font=font)
    return out


def find_pairs(plot_root: Path, seed: int) -> list[tuple[str, Path, Path]]:
    behavior_dir = plot_root / "behavior"
    cossim_dir = plot_root / "cos_sim_curves"

    behavior_by_tag: dict[str, Path] = {}
    for p in sorted(behavior_dir.glob("behavior_BarBoxplot_*.png")):
        m = BEHAVIOR_RE.match(p.name)
        if m:
            behavior_by_tag[m.group("tag")] = p

    cossim_by_tag: dict[str, Path] = {}
    for p in sorted(cossim_dir.glob(f"cos_sim_curves_*_seed{seed}.svg")):
        m = COSSIM_RE.match(p.name)
        if m and int(m.group("seed")) == seed:
            cossim_by_tag[m.group("tag")] = p

    common = sorted(set(behavior_by_tag) & set(cossim_by_tag))
    missing_b = sorted(set(cossim_by_tag) - set(behavior_by_tag))
    missing_c = sorted(set(behavior_by_tag) - set(cossim_by_tag))
    for tag in missing_b:
        print(f"  [warn] no behavior PNG for tag={tag!r}")
    for tag in missing_c:
        print(f"  [warn] no cos_sim SVG for tag={tag!r}")
    return [(tag, behavior_by_tag[tag], cossim_by_tag[tag]) for tag in common]


def svg_to_png(svg_path: Path, png_path: Path, target_height: int) -> None:
    """Rasterize SVG to PNG at the requested pixel height using rsvg-convert."""
    if not shutil.which("rsvg-convert"):
        raise RuntimeError("rsvg-convert not found on PATH")
    subprocess.run(
        [
            "rsvg-convert",
            "--height", str(target_height),
            "--format", "png",
            "--output", str(png_path),
            str(svg_path),
        ],
        check=True,
    )


def hstack(left: Image.Image, right: Image.Image, gap_px: int, bg=(255, 255, 255)) -> Image.Image:
    h = max(left.height, right.height)

    def fit(im: Image.Image) -> Image.Image:
        if im.height == h:
            return im
        new_w = round(im.width * h / im.height)
        return im.resize((new_w, h), Image.LANCZOS)

    L = fit(left)
    R = fit(right)
    out = Image.new("RGB", (L.width + gap_px + R.width, h), bg)
    out.paste(L.convert("RGB"), (0, 0))
    out.paste(R.convert("RGB"), (L.width + gap_px, 0))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--plot-root",
        required=True,
        type=Path,
        help="Plots root, e.g. results/plots/Llama-3.1-8B-Instruct",
    )
    ap.add_argument(
        "--out-subdir",
        default="behavior_rotation",
        help="Subdirectory under --plot-root for outputs.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Absolute output directory. If set, overrides --plot-root/--out-subdir "
             "(use when --plot-root is read-only, e.g. a shared lab results dir).",
    )
    ap.add_argument("--seed", type=int, default=0, help="Cosine-sim curve seed to pair.")
    ap.add_argument(
        "--target-height",
        type=int,
        default=1920,
        help="Target panel height in pixels before the 2x scale.",
    )
    ap.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="Final upscale factor for the combined PNG (default 2.0).",
    )
    ap.add_argument(
        "--gap-frac",
        type=float,
        default=0.02,
        help="Gap between the two panels as a fraction of panel height.",
    )
    ap.add_argument(
        "--quantize",
        action="store_true",
        help="Convert to 8-bit palette PNG (smaller file, fine for plots).",
    )
    ap.add_argument(
        "--colors",
        type=int,
        default=256,
        help="Palette size when --quantize is set (max 256).",
    )
    ap.add_argument(
        "--behavior-title-template",
        default="{task} - Instruction Following Rate",
        help="Format string for the behavior panel title. {task} is replaced "
             "with the display name resolved from TASK_DISPLAY.",
    )
    ap.add_argument(
        "--cos-sim-title",
        default="Cosine Similarity Across Layers",
        help="Title text for the cos-sim curve panel.",
    )
    ap.add_argument(
        "--cos-sim-ylabel",
        default="Cosine similarity",
        help="Y-axis label for the cos-sim curve panel.",
    )
    ap.add_argument(
        "--behavior-title-px",
        type=int,
        default=48,
        help="Behavior-panel title font size in output pixels. Default 48 matches "
             "Plotly's 12 pt y-axis label rendered at the 4x PNG scale.",
    )
    args = ap.parse_args()

    plot_root: Path = args.plot_root.resolve()
    out_dir = (args.out_dir if args.out_dir is not None else plot_root / args.out_subdir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(plot_root, args.seed)
    if not pairs:
        raise SystemExit(f"no matching (behavior, cos_sim) pairs under {plot_root}")

    H = args.target_height
    gap_px = max(1, int(round(H * args.gap_frac)))

    print(f"plot_root: {plot_root}")
    print(f"out_dir:   {out_dir}")
    print(f"target_height={H}px, gap={gap_px}px, scale={args.scale}x  ({len(pairs)} pairs)")

    for tag, behavior_png, cossim_svg in pairs:
        print(f"\n[{tag}]")
        print(f"  behavior: {behavior_png.relative_to(plot_root)}")
        print(f"  cos_sim:  {cossim_svg.relative_to(plot_root)}")

        # Patch the cos-sim SVG (title + y-axis label) before rasterizing.
        patched_svg = patch_cos_sim_svg(
            cossim_svg.read_text(encoding="utf-8"),
            title=args.cos_sim_title,
            ylabel=args.cos_sim_ylabel,
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".svg", delete=False, encoding="utf-8"
        ) as f:
            f.write(patched_svg)
            patched_svg_path = Path(f.name)
        tmp_png = out_dir / f"_tmp_cos_sim_{tag}.png"
        try:
            svg_to_png(patched_svg_path, tmp_png, target_height=H)
            task_display = _task_display_from_tag(tag)
            behavior_title = args.behavior_title_template.format(task=task_display)
            print(f"  behavior title: {behavior_title!r}")
            left = overlay_title_on_png(
                Image.open(behavior_png),
                behavior_title,
                font_size_px=args.behavior_title_px,
            )
            right = Image.open(tmp_png)
            combined = hstack(left, right, gap_px=gap_px)
        finally:
            tmp_png.unlink(missing_ok=True)
            patched_svg_path.unlink(missing_ok=True)

        if args.scale != 1.0:
            new_size = (
                round(combined.width * args.scale),
                round(combined.height * args.scale),
            )
            combined = combined.resize(new_size, Image.LANCZOS)

        if args.quantize:
            combined = combined.convert("RGB").quantize(
                colors=max(2, min(256, args.colors)),
                method=Image.Quantize.MEDIANCUT,
                dither=Image.Dither.NONE,
            )

        out_path = out_dir / f"behavior_rotation_{tag}.png"
        combined.save(out_path, format="PNG", optimize=True)
        kb = out_path.stat().st_size / 1024
        try:
            disp = out_path.relative_to(plot_root)
        except ValueError:
            disp = out_path
        print(
            f"  wrote: {disp}  "
            f"({combined.size[0]}x{combined.size[1]}, {kb:.0f} KB)"
        )


if __name__ == "__main__":
    main()
