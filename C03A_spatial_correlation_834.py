#!/usr/bin/env python3

import argparse
from pathlib import Path

from dolphin.interferogram import estimate_interferometric_correlations


ROOT = Path("/scratch4/workspace/quantao_zhu_uri_edu-ri_insar/TLIPL_modfilter")
IFG_ROOT = ROOT / "merge" / "interferograms"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    ifg_files = sorted(IFG_ROOT.glob("20*_20*/filt_tc.int.vrt"))

    if len(ifg_files) != 834:
        raise RuntimeError(f"Expected 834 filt_tc IFGs, found {len(ifg_files)}")

    print(f"IFGs    : {len(ifg_files)}")
    print(f"Workers : {args.workers}")
    print("Window  : (11, 11)")
    print()

    estimate_interferometric_correlations(
        ifg_files,
        window_size=(11, 11),
        out_suffix=".cor.tif",
        keep_bits=10,
        num_workers=args.workers,
    )

    outputs = list(IFG_ROOT.glob("20*_20*/filt_tc.int.cor.tif"))
    print()
    print(f"DONE: {len(outputs)} correlation files")


if __name__ == "__main__":
    main()
