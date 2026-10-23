#!/usr/bin/env python3

from pathlib import Path

ROOT = Path(
    "/scratch4/workspace/quantao_zhu_uri_edu-ri_insar/"
    "TLIPL_modfilter"
)

DOL_ROOT = Path(
    "/scratch4/workspace/quantao_zhu_uri_edu-ri_insar/"
    "dol_modified"
)

IFG_ROOT = ROOT / "merge" / "interferograms"

pairs = sorted(
    p for p in IFG_ROOT.iterdir()
    if p.is_dir()
)

for i, pair_dir in enumerate(pairs, 1):
    pair = pair_dir.name

    # Current raw IFG symlink already encodes which ministack
    # was selected, including later-ministack selection for duplicates.
    int_link = pair_dir / f"{pair}.int"
    int_source = int_link.resolve()

    # Example:
    # dol_modified/20170224_20170823/interferograms/PAIR/PAIR.int
    ministack = int_source.relative_to(DOL_ROOT).parts[0]

    tc_source = (
        DOL_ROOT
        / ministack
        / "linked_phase"
        / f"temporal_coherence_{ministack}.tif"
    )

    tc_link = (
        pair_dir
        / f"temporal_coherence_{ministack}.tif"
    )

    if tc_link.exists() or tc_link.is_symlink():
        tc_link.unlink()

    tc_link.symlink_to(tc_source)

    print(
        f"[{i:03d}/{len(pairs):03d}] "
        f"{pair}  ->  {ministack}",
        flush=True,
    )

print()
print("=" * 72)
print("DONE")
print("Pairs:", len(pairs))
print("=" * 72)
