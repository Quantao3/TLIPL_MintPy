#!/usr/bin/env python3

from pathlib import Path
import shutil


ROOT = Path("/scratch4/workspace/quantao_zhu_uri_edu-ri_insar/TLIPL_modfilter")
IFG_ROOT = ROOT / "merge" / "interferograms"

REF_ROOT = Path(
    "/scratch4/workspace/quantao_zhu_uri_edu-ri_insar/"
    "TLIPL_nofilter/merge/interferograms"
)


def make_link(src, dst):
    if dst.is_symlink():
        if dst.resolve() == src.resolve():
            return
        raise RuntimeError(
            f"Existing symlink points elsewhere:\n"
            f"  {dst}\n"
            f"  -> {dst.resolve()}"
        )

    if dst.exists():
        raise RuntimeError(f"Destination already exists and is not a symlink: {dst}")

    dst.symlink_to(src)


def copy_rsc(src, dst):
    if not src.exists():
        raise RuntimeError(f"Missing reference RSC: {src}")

    if dst.exists():
        return

    text = src.read_text()

    if "PROCESSOR" not in text.upper() or "gdal" not in text.lower():
        raise RuntimeError(f"Unexpected PROCESSOR metadata in {src}")

    shutil.copy2(src, dst)


def main():
    pair_dirs = sorted(p for p in IFG_ROOT.iterdir() if p.is_dir())

    n_2x2 = 0
    n_1x1 = 0
    n_done = 0

    for pair_dir in pair_dirs:
        pair = pair_dir.name

        success_2x2 = pair_dir / "_SNAPHU207_2X2_MODFILTER_SUCCESS.json"
        success_1x1 = pair_dir / "_SNAPHU207_1X1_MODFILTER_SUCCESS.json"

        if success_2x2.exists():
            mode = "2x2"
            src_unw = pair_dir / "filt_tc_snaphu207_2x2.unw.tif"
            src_cc = pair_dir / "filt_tc_snaphu207_2x2.conncomp.tif"
            n_2x2 += 1

        elif success_1x1.exists():
            mode = "1x1"
            src_unw = pair_dir / "filt_tc_snaphu207_1x1.unw.tif"
            src_cc = pair_dir / "filt_tc_snaphu207_1x1.conncomp.tif"
            n_1x1 += 1

        else:
            raise RuntimeError(f"No SNAPHU success marker: {pair}")

        src_cor = pair_dir / "filt_tc.int.cor.tif"

        for src in (src_unw, src_cc, src_cor):
            if not src.exists():
                raise RuntimeError(f"Missing source file: {src}")

        dst_unw = pair_dir / f"{pair}.unw.tif"
        dst_cc = pair_dir / f"{pair}.unw.conncomp.tif"
        dst_cor = pair_dir / f"{pair}.int.cor.tif"

        make_link(src_unw, dst_unw)
        make_link(src_cc, dst_cc)
        make_link(src_cor, dst_cor)

        ref_dir = REF_ROOT / pair

        copy_rsc(
            ref_dir / f"{pair}.unw.tif.rsc",
            pair_dir / f"{pair}.unw.tif.rsc",
        )

        copy_rsc(
            ref_dir / f"{pair}.unw.conncomp.tif.rsc",
            pair_dir / f"{pair}.unw.conncomp.tif.rsc",
        )

        copy_rsc(
            ref_dir / f"{pair}.int.cor.tif.rsc",
            pair_dir / f"{pair}.int.cor.tif.rsc",
        )

        print(f"{pair}  {mode}")
        n_done += 1

    print()
    print("=" * 72)
    print(f"Total prepared : {n_done}")
    print(f"2x2            : {n_2x2}")
    print(f"1x1            : {n_1x1}")
    print("=" * 72)


if __name__ == "__main__":
    main()
