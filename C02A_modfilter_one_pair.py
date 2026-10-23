#!/usr/bin/env python3

import sys
from pathlib import Path

import isce.components.isceobj as isceobj
import numpy as np
import rasterio
from rasterio.windows import Window
import FilterAndCoherence


ROOT = Path("/scratch4/workspace/quantao_zhu_uri_edu-ri_insar/TLIPL_modfilter")

WIDTH = 15889
HEIGHT = 6951
BLOCK_ROWS = 256
FILTER_STRENGTH = 0.3

PAIR = sys.argv[1]

PAIR_DIR = ROOT / "merge" / "interferograms" / PAIR

RAW = PAIR_DIR / f"{PAIR}.int"
FILTERED = PAIR_DIR / "filt_lk.int"
FINAL = PAIR_DIR / "filt_tc.int"

TC_FILES = list(PAIR_DIR.glob("temporal_coherence_*.tif"))

if not RAW.is_file():
    raise FileNotFoundError(RAW)

if len(TC_FILES) != 1:
    raise RuntimeError(
        f"Expected exactly one temporal coherence file, found {len(TC_FILES)}"
    )

TC_FILE = TC_FILES[0]


def make_int_metadata(path):
    image = isceobj.createIntImage()
    image.filename = str(path)
    image.width = WIDTH
    image.length = HEIGHT
    image.dataType = "CFLOAT"
    image.renderHdr()


# ----------------------------------------------------------------------
# 1. Standard Goldstein-Werner filtering
# ----------------------------------------------------------------------
print("=" * 72)
print(f"Pair             : {PAIR}")
print(f"Raw IFG          : {RAW}")
print(f"Temporal coh.    : {TC_FILE}")
print(f"Filter strength  : {FILTER_STRENGTH}")
print("=" * 72)

print("Running Goldstein-Werner filter...", flush=True)

FilterAndCoherence.runFilter(str(RAW), str(FILTERED), FILTER_STRENGTH)
make_int_metadata(FILTERED)


# ----------------------------------------------------------------------
# 2. Open raw / filtered / TC / final output
# ----------------------------------------------------------------------
raw = np.memmap(RAW, dtype=np.complex64, mode="r", shape=(HEIGHT, WIDTH))
filtered = np.memmap(FILTERED, dtype=np.complex64, mode="r", shape=(HEIGHT, WIDTH))
final = np.memmap(FINAL, dtype=np.complex64, mode="w+", shape=(HEIGHT, WIDTH))

with rasterio.open(TC_FILE) as tc_src:

    if tc_src.shape != (HEIGHT, WIDTH):
        raise RuntimeError(
            f"TC shape {tc_src.shape} != ({HEIGHT}, {WIDTH})"
        )

    # ------------------------------------------------------------------
    # 3. TC-constrained circular phase fusion
    # ------------------------------------------------------------------
    for y0 in range(0, HEIGHT, BLOCK_ROWS):
        rows = min(BLOCK_ROWS, HEIGHT - y0)

        raw_block = raw[y0:y0 + rows, :]
        filt_block = filtered[y0:y0 + rows, :]

        tc = tc_src.read(
            1,
            window=Window(0, y0, WIDTH, rows),
        ).astype(np.float32, copy=False)

        # Non-finite TC means no temporal-coherence protection.
        tc = np.nan_to_num(tc, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(tc, 0.0, 1.0, out=tc)

        raw_phase = np.angle(raw_block)
        filt_phase = np.angle(filt_block)

        # Shortest circular phase change introduced by filtering.
        delta_phi = (filt_phase - raw_phase + np.pi) % (2.0 * np.pi) - np.pi

        # If filtering gives no valid phase, preserve the raw phase.
        bad_filtered = (
            ~np.isfinite(filt_block.real)
            | ~np.isfinite(filt_block.imag)
            | (np.abs(filt_block) == 0.0)
        )
        delta_phi[bad_filtered] = 0.0

        # TC=1 -> raw phase; TC=0 -> filtered phase.
        phase = raw_phase + (1.0 - tc) * delta_phi

        # Preserve raw IFG amplitude; modify phase only.
        final[y0:y0 + rows, :] = (
            np.abs(raw_block) * np.exp(1j * phase)
        ).astype(np.complex64)

        print(
            f"Fusion rows {y0:5d}-{y0 + rows - 1:5d}",
            flush=True,
        )

final.flush()

del final
del raw
del filtered

make_int_metadata(FINAL)

print()
print("=" * 72)
print("DONE")
print(f"Standard filtered : {FILTERED}")
print(f"TC-constrained    : {FINAL}")
print("=" * 72)
