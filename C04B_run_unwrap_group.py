#!/usr/bin/env python3

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path


ROOT = Path("/scratch4/workspace/quantao_zhu_uri_edu-ri_insar/TLIPL_modfilter")
RUNNER = ROOT / "C04A_run_snaphu207_pair.py"
TIMEOUT_SECONDS = 24 * 3600


def kill_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    except ProcessLookupError:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["1x1", "2x2"])
    parser.add_argument("--pair-file", required=True)
    args = parser.parse_args()

    pair_file = Path(args.pair_file)
    pairs = [x.strip() for x in pair_file.read_text().splitlines() if x.strip()]

    print(f"Mode      : {args.mode}")
    print(f"Pair file : {pair_file}")
    print(f"Pairs     : {len(pairs)}")
    print(f"Timeout   : 24 h / pair")
    print()

    failed = []

    for i, pair in enumerate(pairs, 1):
        pair_dir = ROOT / "merge" / "interferograms" / pair
        log = pair_dir / f"snaphu207_{args.mode}_modfilter.log"

        print(f"[{i}/{len(pairs)}] START {pair}", flush=True)

        with log.open("w") as fp:
            proc = subprocess.Popen(
                [sys.executable, "-u", str(RUNNER), "--pair", pair, "--mode", args.mode],
                stdout=fp,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

            try:
                rc = proc.wait(timeout=TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                print(f"[{i}/{len(pairs)}] TIMEOUT {pair}", flush=True)
                kill_group(proc)
                failed.append((pair, "TIMEOUT"))
                continue

        if rc == 0:
            print(f"[{i}/{len(pairs)}] OK      {pair}", flush=True)
        else:
            print(f"[{i}/{len(pairs)}] FAILED  {pair} rc={rc}", flush=True)
            failed.append((pair, f"exit={rc}"))

    print()
    print("=" * 72)
    print(f"Pairs  : {len(pairs)}")
    print(f"Failed : {len(failed)}")
    print("=" * 72)

    if failed:
        for pair, reason in failed:
            print(f"{pair}  {reason}")
        raise RuntimeError(f"{len(failed)} pair(s) failed or timed out")


if __name__ == "__main__":
    main()
