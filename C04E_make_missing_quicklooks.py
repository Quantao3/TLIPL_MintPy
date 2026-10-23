#!/usr/bin/env python3

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from rasterio.enums import Resampling


ROOT = Path("/scratch4/workspace/quantao_zhu_uri_edu-ri_insar/TLIPL_modfilter")
IFG_ROOT = ROOT / "merge" / "interferograms"

WATER_FILE = Path(
    "/scratch4/workspace/quantao_zhu_uri_edu-ri_insar/"
    "dol_modified/20170224_20170823/geometry/waterMask_lk.tif"
)

SHADOW_FILE = Path(
    "/scratch4/workspace/quantao_zhu_uri_edu-ri_insar/"
    "dol_modified/20170224_20170823/geometry/shadowMask_lk.tif"
)

WIDTH = 15889
HEIGHT = 6951
DISPLAY_STRIDE = 4


def make_quicklook(pair, mode, raw_file, unw_file, cc_file, out_png, geometry_valid):
    # Full-resolution coverage calculation
    with rasterio.open(raw_file) as src:
        raw = src.read(1)

    with rasterio.open(cc_file) as src:
        cc_full = src.read(1)

    raw_ifg_valid = (
        np.isfinite(raw.real)
        & np.isfinite(raw.imag)
        & (np.abs(raw) > 0)
    )

    valid_input = geometry_valid & raw_ifg_valid
    successful = valid_input & (cc_full > 0)

    denominator = np.count_nonzero(valid_input)
    numerator = np.count_nonzero(successful)

    coverage = 100.0 * numerator / denominator if denominator > 0 else np.nan

    # Downsample ONLY for plotting
    out_h = (HEIGHT + DISPLAY_STRIDE - 1) // DISPLAY_STRIDE
    out_w = (WIDTH + DISPLAY_STRIDE - 1) // DISPLAY_STRIDE

    with rasterio.open(unw_file) as src:
        unw = src.read(1, out_shape=(out_h, out_w), resampling=Resampling.nearest)

    with rasterio.open(cc_file) as src:
        cc = src.read(1, out_shape=(out_h, out_w), resampling=Resampling.nearest)

    valid = np.isfinite(unw) & (cc > 0)

    if np.any(valid):
        vals = unw[valid]
        vmin = float(np.nanpercentile(vals, 1.0))
        vmax = float(np.nanpercentile(vals, 99.0))
        if vmax <= vmin:
            vmax = vmin + 1.0
    else:
        vmin, vmax = -1.0, 1.0

    unw_masked = np.ma.masked_where(~valid, unw)

    labels = np.unique(cc)
    labels = labels[labels > 0]
    n_components = len(labels)

    max_label = int(np.max(cc)) if cc.size > 0 else 1
    color_count = max(max_label + 1, 2)

    colors = plt.cm.tab20(np.linspace(0.0, 1.0, color_count))
    colors[0] = [0.0, 0.0, 0.0, 1.0]

    comp_cmap = ListedColormap(colors)
    boundaries = np.arange(-0.5, color_count + 0.5, 1.0)
    comp_norm = BoundaryNorm(boundaries, comp_cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    im1 = axes[0].imshow(unw_masked, cmap="jet", vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[0].set_title("Unwrapped phase")
    axes[0].set_xlabel("Range pixel")
    axes[0].set_ylabel("Azimuth pixel")
    cb1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cb1.set_label("Unwrapped phase (rad)")

    im2 = axes[1].imshow(cc, cmap=comp_cmap, norm=comp_norm, interpolation="nearest")
    axes[1].set_title("Connected components")
    axes[1].set_xlabel("Range pixel")
    axes[1].set_ylabel("Azimuth pixel")
    cb2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cb2.set_label("Connected component label")

    fig.suptitle(
        f"{pair} | {mode} | coverage={coverage:.2f}% | components={n_components}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    with rasterio.open(WATER_FILE) as src:
        water = src.read(1)

    with rasterio.open(SHADOW_FILE) as src:
        shadow = src.read(1)

    geometry_valid = (water == 1) & (shadow == 0)

    pair_dirs = sorted(p for p in IFG_ROOT.iterdir() if p.is_dir())

    n_skip = 0
    n_make = 0

    for pair_dir in pair_dirs:
        pair = pair_dir.name

        success_2x2 = pair_dir / "_SNAPHU207_2X2_MODFILTER_SUCCESS.json"
        success_1x1 = pair_dir / "_SNAPHU207_1X1_MODFILTER_SUCCESS.json"

        if success_2x2.exists():
            mode = "2x2"
            unw = pair_dir / "filt_tc_snaphu207_2x2.unw.tif"
            cc = pair_dir / "filt_tc_snaphu207_2x2.conncomp.tif"
            png = pair_dir / "filt_tc_snaphu207_2x2.quicklook.png"
        elif success_1x1.exists():
            mode = "1x1"
            unw = pair_dir / "filt_tc_snaphu207_1x1.unw.tif"
            cc = pair_dir / "filt_tc_snaphu207_1x1.conncomp.tif"
            png = pair_dir / "filt_tc_snaphu207_1x1.quicklook.png"
        else:
            continue

        if png.exists():
            print(f"SKIP  {pair}  {mode}  quicklook exists")
            n_skip += 1
            continue

        raw = pair_dir / f"{pair}.int.vrt"

        make_quicklook(pair, mode, raw, unw, cc, png, geometry_valid)
        print(f"MAKE  {pair}  {mode}  {png.name}")
        n_make += 1

    print("=" * 72)
    print(f"Generated : {n_make}")
    print(f"Skipped   : {n_skip}")
    print("=" * 72)


if __name__ == "__main__":
    main()
