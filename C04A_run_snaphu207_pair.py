#!/usr/bin/env python3

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import snaphu


ROOT = Path("/scratch4/workspace/quantao_zhu_uri_edu-ri_insar/TLIPL_modfilter")
IFG_ROOT = ROOT / "merge" / "interferograms"

WIDTH = 15889
HEIGHT = 6951

TILE_COST_THRESH = 50
MIN_REGION_SIZE = 10000


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", required=True)
    parser.add_argument("--mode", required=True, choices=["1x1", "2x2"])
    args = parser.parse_args()

    pair = args.pair
    mode = args.mode
    pair_dir = IFG_ROOT / pair

    if mode == "2x2":
        ntiles = (2, 2)
        overlap = (512, 512)
        nproc = 4
    else:
        ntiles = (1, 1)
        overlap = (0, 0)
        nproc = 1

    ifg = pair_dir / "filt_tc.int.vrt"
    cor = pair_dir / "filt_tc.int.cor.tif"

    final_unw = pair_dir / f"filt_tc_snaphu207_{mode}.unw.tif"
    final_cc = pair_dir / f"filt_tc_snaphu207_{mode}.conncomp.tif"

    partial_unw = pair_dir / f".filt_tc_snaphu207_{mode}.unw.partial.tif"
    partial_cc = pair_dir / f".filt_tc_snaphu207_{mode}.conncomp.partial.tif"

    scratch = pair_dir / "tem" / f"snaphu207_{mode}_modfilter"
    success = pair_dir / f"_SNAPHU207_{mode.upper()}_MODFILTER_SUCCESS.json"
    failed = pair_dir / f"_SNAPHU207_{mode.upper()}_MODFILTER_FAILED.json"

    if snaphu.get_snaphu_version() != "2.0.7":
        raise RuntimeError(f"Embedded SNAPHU is {snaphu.get_snaphu_version()}, required 2.0.7")

    if not ifg.is_file():
        raise FileNotFoundError(ifg)
    if not cor.is_file():
        raise FileNotFoundError(cor)

    if success.is_file() and final_unw.is_file() and final_cc.is_file():
        print(f"Already complete: {pair} {mode}")
        return

    if final_unw.exists() or final_cc.exists():
        raise RuntimeError(f"Existing final output found for {pair}; refusing overwrite")

    partial_unw.unlink(missing_ok=True)
    partial_cc.unlink(missing_ok=True)

    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)

    print("=" * 72)
    print(f"Pair       : {pair}")
    print(f"Mode       : {mode}")
    print(f"SNAPHU     : {snaphu.get_snaphu_version()}")
    print(f"Tiles      : {ntiles}")
    print(f"Overlap    : {overlap}")
    print(f"NPROC      : {nproc}")
    print(f"Input      : {ifg}")
    print(f"Correlation: {cor}")
    print("=" * 72, flush=True)

    try:
        with snaphu.io.Raster(ifg) as igram, snaphu.io.Raster(cor) as corr:
            with snaphu.io.Raster.create(
                partial_unw, width=WIDTH, height=HEIGHT, dtype="float32",
                driver="GTiff", nodata=0.0, tiled=True,
                blockxsize=512, blockysize=256, BIGTIFF="IF_SAFER"
            ) as unw:
                with snaphu.io.Raster.create(
                    partial_cc, width=WIDTH, height=HEIGHT, dtype="uint32",
                    driver="GTiff", nodata=0, tiled=True,
                    blockxsize=512, blockysize=256, BIGTIFF="IF_SAFER"
                ) as conncomp:
                    snaphu.unwrap(
                        igram=igram,
                        corr=corr,
                        nlooks=1.0,
                        cost="smooth",
                        init="mcf",
                        mask=None,
                        min_conncomp_frac=0.001,
                        phase_grad_window=(7, 7),
                        ntiles=ntiles,
                        tile_overlap=overlap,
                        nproc=nproc,
                        tile_cost_thresh=TILE_COST_THRESH,
                        min_region_size=MIN_REGION_SIZE,
                        single_tile_reoptimize=False,
                        regrow_conncomps=True,
                        scratchdir=scratch,
                        delete_scratch=True,
                        unw=unw,
                        conncomp=conncomp,
                    )

        partial_unw.rename(final_unw)
        partial_cc.rename(final_cc)

        payload = {
            "completed_utc": utc_now(),
            "pair": pair,
            "embedded_snaphu_version": snaphu.get_snaphu_version(),
            "ntiles": list(ntiles),
            "tile_overlap": list(overlap),
            "nproc": nproc,
            "nlooks": 1.0,
            "cost": "smooth",
            "init": "mcf",
            "min_conncomp_frac": 0.001,
            "phase_grad_window": [7, 7],
            "tile_cost_thresh": TILE_COST_THRESH,
            "min_region_size": MIN_REGION_SIZE,
            "single_tile_reoptimize": False,
            "regrow_conncomps": True,
            "input_ifg": str(ifg),
            "input_correlation": str(cor),
            "unwrapped": str(final_unw),
            "conncomp": str(final_cc),
        }
        success.write_text(json.dumps(payload, indent=2))

        print(f"SUCCESS: {pair} {mode}", flush=True)

    except Exception as exc:
        partial_unw.unlink(missing_ok=True)
        partial_cc.unlink(missing_ok=True)

        failed.write_text(json.dumps({
            "failed_utc": utc_now(),
            "pair": pair,
            "mode": mode,
            "error": f"{type(exc).__name__}: {exc}",
        }, indent=2))
        raise


if __name__ == "__main__":
    main()
