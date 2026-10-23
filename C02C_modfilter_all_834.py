#!/usr/bin/env python3

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import isce.components.isceobj as isceobj
import numpy as np
import rasterio
from rasterio.windows import Window
import FilterAndCoherence


ROOT = Path("/scratch4/workspace/quantao_zhu_uri_edu-ri_insar/TLIPL_modfilter")
IFG_ROOT = ROOT / "merge" / "interferograms"

WIDTH = 15889
HEIGHT = 6951
BLOCK_ROWS = 256
FILTER_STRENGTH = 0.3
EXPECTED_BYTES = WIDTH * HEIGHT * 8

logging.getLogger("rasterio").setLevel(logging.WARNING)


def make_int_metadata(path):
    image = isceobj.createIntImage()
    image.filename = str(path)
    image.width = WIDTH
    image.length = HEIGHT
    image.dataType = "CFLOAT"
    image.renderHdr()


def remove_product(path):
    for p in [path, Path(str(path) + ".xml"), Path(str(path) + ".vrt"), Path(str(path) + ".hdr")]:
        p.unlink(missing_ok=True)


def process_pair(pair):
    pair_dir = IFG_ROOT / pair

    raw_file = pair_dir / f"{pair}.int"
    temp_filtered = pair_dir / "filt_lk.int"
    final_file = pair_dir / "filt_tc.int"
    marker = pair_dir / "_MODFILTER_SUCCESS"

    if marker.exists() and final_file.exists() and final_file.stat().st_size == EXPECTED_BYTES:
        return pair, "SKIP", ""

    tc_files = list(pair_dir.glob("temporal_coherence_*.tif"))
    if not raw_file.is_file():
        return pair, "FAIL", f"Missing raw IFG: {raw_file}"
    if len(tc_files) != 1:
        return pair, "FAIL", f"Expected 1 TC file, found {len(tc_files)}"

    tc_file = tc_files[0]

    remove_product(temp_filtered)
    remove_product(final_file)
    marker.unlink(missing_ok=True)

    try:
        # 1. Standard Goldstein-Werner filtering
        FilterAndCoherence.runFilter(str(raw_file), str(temp_filtered), FILTER_STRENGTH)

        if temp_filtered.stat().st_size != EXPECTED_BYTES:
            raise RuntimeError(f"Filtered size = {temp_filtered.stat().st_size}, expected {EXPECTED_BYTES}")

        # 2. TC-constrained circular phase fusion
        raw = np.memmap(raw_file, dtype=np.complex64, mode="r", shape=(HEIGHT, WIDTH))
        filtered = np.memmap(temp_filtered, dtype=np.complex64, mode="r", shape=(HEIGHT, WIDTH))
        final = np.memmap(final_file, dtype=np.complex64, mode="w+", shape=(HEIGHT, WIDTH))

        with rasterio.open(tc_file) as tc_src:
            if tc_src.shape != (HEIGHT, WIDTH):
                raise RuntimeError(f"TC shape {tc_src.shape} != ({HEIGHT}, {WIDTH})")

            for y0 in range(0, HEIGHT, BLOCK_ROWS):
                rows = min(BLOCK_ROWS, HEIGHT - y0)

                raw_block = raw[y0:y0 + rows, :]
                filt_block = filtered[y0:y0 + rows, :]

                tc = tc_src.read(1, window=Window(0, y0, WIDTH, rows)).astype(np.float32, copy=False)
                tc = np.nan_to_num(tc, nan=0.0, posinf=0.0, neginf=0.0)
                np.clip(tc, 0.0, 1.0, out=tc)

                raw_phase = np.angle(raw_block)
                filt_phase = np.angle(filt_block)

                delta_phi = (filt_phase - raw_phase + np.pi) % (2.0 * np.pi) - np.pi

                bad_filtered = (
                    ~np.isfinite(filt_block.real)
                    | ~np.isfinite(filt_block.imag)
                    | (np.abs(filt_block) == 0.0)
                )
                delta_phi[bad_filtered] = 0.0

                phase = raw_phase + (1.0 - tc) * delta_phi
                final[y0:y0 + rows, :] = (
                    np.abs(raw_block) * np.exp(1j * phase)
                ).astype(np.complex64)

        final.flush()
        del final
        del filtered
        del raw

        make_int_metadata(final_file)

        # Goldstein result is only an intermediate product.
        remove_product(temp_filtered)

        marker.write_text("SUCCESS\n")
        return pair, "OK", ""

    except Exception as exc:
        remove_product(temp_filtered)
        remove_product(final_file)
        marker.unlink(missing_ok=True)
        return pair, "FAIL", str(exc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    pairs = sorted(p.name for p in IFG_ROOT.iterdir() if p.is_dir())

    print(f"Pairs            : {len(pairs)}")
    print(f"Parallel workers : {args.workers}")
    print(f"Filter strength  : {FILTER_STRENGTH}")
    print()

    failures = []
    completed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_pair, pair): pair for pair in pairs}

        for future in as_completed(futures):
            pair, status, message = future.result()
            completed += 1

            if status == "FAIL":
                failures.append((pair, message))
                print(f"[{completed:3d}/{len(pairs)}] FAIL {pair}: {message}", flush=True)
            else:
                print(f"[{completed:3d}/{len(pairs)}] {status:4s} {pair}", flush=True)

    print()
    print("=" * 72)
    print(f"Total    : {len(pairs)}")
    print(f"Failed   : {len(failures)}")
    print(f"Completed: {len(pairs) - len(failures)}")
    print("=" * 72)

    if failures:
        print("\nFailed pairs:")
        for pair, message in failures:
            print(f"{pair}: {message}")
        raise RuntimeError(f"{len(failures)} pairs failed")


if __name__ == "__main__":
    main()
