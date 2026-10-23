#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


ROOT = Path("/scratch4/workspace/quantao_zhu_uri_edu-ri_insar/TLIPL_modfilter")

WIDTH = 15889
HEIGHT = 6951
BLOCK_ROWS = 256

PAIR = sys.argv[1]
PAIR_DIR = ROOT / "merge" / "interferograms" / PAIR

RAW = PAIR_DIR / f"{PAIR}.int"
FILTERED = PAIR_DIR / "filt_lk.int"
FINAL = PAIR_DIR / "filt_tc.int"
TC_FILE = list(PAIR_DIR.glob("temporal_coherence_*.tif"))[0]

BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.000001)]

raw = np.memmap(RAW, dtype=np.complex64, mode="r", shape=(HEIGHT, WIDTH))
filtered = np.memmap(FILTERED, dtype=np.complex64, mode="r", shape=(HEIGHT, WIDTH))
final = np.memmap(FINAL, dtype=np.complex64, mode="r", shape=(HEIGHT, WIDTH))

count = np.zeros(len(BINS), dtype=np.int64)
sum_tc = np.zeros(len(BINS), dtype=np.float64)
sum_raw_dist = np.zeros(len(BINS), dtype=np.float64)
sum_filt_dist = np.zeros(len(BINS), dtype=np.float64)
sum_formula_error = np.zeros(len(BINS), dtype=np.float64)
max_formula_error = 0.0

bad_filtered_count = 0
bad_filtered_raw_diff_sum = 0.0
bad_filtered_raw_diff_max = 0.0


def wrap(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


with rasterio.open(TC_FILE) as tc_src:
    for y0 in range(0, HEIGHT, BLOCK_ROWS):
        rows = min(BLOCK_ROWS, HEIGHT - y0)

        r = raw[y0:y0 + rows, :]
        f = filtered[y0:y0 + rows, :]
        m = final[y0:y0 + rows, :]

        tc = tc_src.read(1, window=Window(0, y0, WIDTH, rows)).astype(np.float32)
        tc = np.nan_to_num(tc, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(tc, 0.0, 1.0, out=tc)

        valid_raw = np.isfinite(r.real) & np.isfinite(r.imag) & (np.abs(r) > 0)
        valid_filt = np.isfinite(f.real) & np.isfinite(f.imag) & (np.abs(f) > 0)
        valid_final = np.isfinite(m.real) & np.isfinite(m.imag) & (np.abs(m) > 0)

        raw_phase = np.angle(r)
        filt_phase = np.angle(f)
        final_phase = np.angle(m)

        # Check filtered-invalid safeguard: final should preserve raw phase.
        bad = valid_raw & ~valid_filt & valid_final
        if np.any(bad):
            d = np.abs(wrap(final_phase[bad] - raw_phase[bad]))
            bad_filtered_count += d.size
            bad_filtered_raw_diff_sum += d.sum(dtype=np.float64)
            bad_filtered_raw_diff_max = max(bad_filtered_raw_diff_max, float(d.max()))

        valid = valid_raw & valid_filt & valid_final
        if not np.any(valid):
            continue

        delta = wrap(filt_phase - raw_phase)
        actual_move = wrap(final_phase - raw_phase)
        expected_move = (1.0 - tc) * delta

        formula_error = np.abs(wrap(actual_move - expected_move))
        dist_raw = np.abs(actual_move)
        dist_filt = np.abs(wrap(final_phase - filt_phase))

        max_formula_error = max(max_formula_error, float(formula_error[valid].max()))

        for i, (lo, hi) in enumerate(BINS):
            mask = valid & (tc >= lo) & (tc < hi)
            if not np.any(mask):
                continue

            n = int(np.count_nonzero(mask))
            count[i] += n
            sum_tc[i] += tc[mask].sum(dtype=np.float64)
            sum_raw_dist[i] += dist_raw[mask].sum(dtype=np.float64)
            sum_filt_dist[i] += dist_filt[mask].sum(dtype=np.float64)
            sum_formula_error[i] += formula_error[mask].sum(dtype=np.float64)


print()
print("=" * 90)
print(f"PAIR: {PAIR}")
print("=" * 90)
print("TC bin       N pixels      mean TC   |final-raw|   |final-filter|   formula error")
print("-" * 90)

for i, (lo, hi) in enumerate(BINS):
    if count[i] == 0:
        continue

    print(
        f"{lo:.1f}-{min(hi,1.0):.1f}  "
        f"{count[i]:12,d}   "
        f"{sum_tc[i] / count[i]:7.4f}   "
        f"{sum_raw_dist[i] / count[i]:11.6f}   "
        f"{sum_filt_dist[i] / count[i]:14.6f}   "
        f"{sum_formula_error[i] / count[i]:13.3e}"
    )

print("-" * 90)
print(f"Maximum formula error : {max_formula_error:.3e} rad")

if bad_filtered_count:
    print()
    print(f"Invalid-filter pixels : {bad_filtered_count:,}")
    print(f"Mean |final-raw|      : {bad_filtered_raw_diff_sum / bad_filtered_count:.3e} rad")
    print(f"Max  |final-raw|      : {bad_filtered_raw_diff_max:.3e} rad")

print("=" * 90)
