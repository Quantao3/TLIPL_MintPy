#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


ROOT = Path("/scratch4/workspace/quantao_zhu_uri_edu-ri_insar/TLIPL_modfilter")
IFG_ROOT = ROOT / "merge" / "interferograms"

GEOM = Path("/scratch4/workspace/quantao_zhu_uri_edu-ri_insar/dol_modified/20170224_20170823/geometry")
WATER = GEOM / "waterMask_lk.tif"
SHADOW = GEOM / "shadowMask_lk.tif"

WIDTH = 15889
HEIGHT = 6951
BLOCK_ROWS = 256


pairs = sorted(p for p in IFG_ROOT.iterdir() if p.is_dir())

with rasterio.open(WATER) as water_src, rasterio.open(SHADOW) as shadow_src:

    for i, pair_dir in enumerate(pairs, 1):
        pair = pair_dir.name

        raw_file = pair_dir / f"{pair}.int"
        filt_file = pair_dir / "filt_tc.int"
        cor_file = pair_dir / "filt_tc.int.cor.tif"

        if not raw_file.is_file() or not filt_file.is_file() or not cor_file.is_file():
            raise FileNotFoundError(f"Missing input for {pair}")

        raw = np.memmap(raw_file, dtype=np.complex64, mode="r", shape=(HEIGHT, WIDTH))
        filt = np.memmap(filt_file, dtype=np.complex64, mode="r+", shape=(HEIGHT, WIDTH))

        with rasterio.open(cor_file, "r+") as cor:
            for y0 in range(0, HEIGHT, BLOCK_ROWS):
                rows = min(BLOCK_ROWS, HEIGHT - y0)
                window = Window(0, y0, WIDTH, rows)

                raw_block = raw[y0:y0 + rows, :]
                water = water_src.read(1, window=window)
                shadow = shadow_src.read(1, window=window)
                cor_block = cor.read(1, window=window)

                valid = (
                    (water == 1)
                    & (shadow == 0)
                    & np.isfinite(raw_block.real)
                    & np.isfinite(raw_block.imag)
                    & (np.abs(raw_block) > 0)
                )

                filt_block = filt[y0:y0 + rows, :]
                filt_block[~valid] = 0.0 + 0.0j

                cor_block[~valid] = 0.0
                cor.write(cor_block, 1, window=window)

        filt.flush()
        del filt, raw

        print(f"[{i:03d}/{len(pairs):03d}] {pair}", flush=True)

print(f"DONE: masked {len(pairs)} pairs")
