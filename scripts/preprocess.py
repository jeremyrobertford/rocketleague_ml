#!/usr/bin/env python3
"""
Preprocess script: call rrrocket on raw replays and save JSON to data/processed/json/
"""

import os
import json
import argparse
from pathlib import Path

from rocketleague_ml.data.loader import load_replay
from rocketleague_ml.config import RAW_REPLAYS, PROCESSED


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def is_replay_file(fname: str):
    return fname.lower().endswith(".replay")


def main():
    parser = argparse.ArgumentParser(
        description="Parse .replay files with rrrocket and save JSON."
    )
    parser.add_argument(
        "--input", "-i", default=RAW_REPLAYS, help="Folder with .replay files"
    )
    parser.add_argument(
        "--output",
        "-o",
        default=os.path.join(PROCESSED, "json"),
        help="Folder to write rrrocket JSON files",
    )
    parser.add_argument(
        "--bin-path", default=None, help="Optional path to rrrocket executable"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing JSON files"
    )
    args = parser.parse_args()

    in_dir: str = args.input
    out_dir: str = args.output
    ensure_dir(out_dir)

    replay_files = sorted([f for f in os.listdir(in_dir) if is_replay_file(f)])
    if not replay_files:
        print(f"No .replay files found in {in_dir}")
        return

    succeeded = 0
    failed = 0
    for fname in replay_files:
        in_path = os.path.join(in_dir, fname)
        base = os.path.splitext(fname)[0]
        out_path = os.path.join(out_dir, f"{base}.json")

        if os.path.exists(out_path) and not args.overwrite:
            print(f"Skipping {fname} (JSON exists). Use --overwrite to force.")
            continue

        print(f"Parsing {fname} -> {out_path} ...", end=" ", flush=True)
        try:
            replay_json = load_replay(in_path, rrrocket_path=args.bin_path)
            # write JSON nicely for inspection
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(replay_json, fh, indent=2, ensure_ascii=False)
            print("OK")
            succeeded += 1
        except Exception as e:
            print("FAILED")
            print(f"  {type(e).__name__}: {e}")
            failed += 1

    print()
    print(f"Done. succeeded={succeeded}, failed={failed}")
    print(f"Saved JSON to: {out_dir}")


if __name__ == "__main__":
    main()
