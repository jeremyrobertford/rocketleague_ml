"""
Process script: convert preprocessed games into pkl objects for feature extraction
"""

import os
import argparse
import pickle
import json
from pathlib import Path

from rocketleague_ml.data.loader import load_preprocessed_replay
from rocketleague_ml.models.process import process_game
from rocketleague_ml.config import RAW_REPLAYS, PROCESSED


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def is_json_file(fname: str):
    return fname.lower().endswith(".json")


def main():
    parser = argparse.ArgumentParser(description="Parse replay JSON files.")
    parser.add_argument(
        "--input", "-i", default=RAW_REPLAYS, help="Folder with replay JSON files"
    )
    parser.add_argument(
        "--output",
        "-o",
        default=PROCESSED,
        help="Folder to write processed pickle files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force re-run analysis even if it was already done",
    )
    args = parser.parse_args()

    in_dir: str = args.input
    out_dir: str = args.output
    ensure_dir(out_dir)

    replay_files = sorted([f for f in os.listdir(in_dir) if is_json_file(f)])
    if not replay_files:
        print(f"No JSON files found in {in_dir}")
        return

    succeeded = 0
    failed = 0
    for fname in replay_files:
        in_path = os.path.join(in_dir, fname)
        base = os.path.splitext(fname)[0]
        out_path_pkl = os.path.join(out_dir, "pkl", f"{base}.pkl")
        out_path_json = os.path.join(out_dir, "json", f"{base}.json")

        replay_json = None

        if (
            os.path.exists(out_path_pkl)
            and os.path.exists(out_path_json)
            and not args.overwrite
        ):
            print(f"Skipping existing analysis for {fname}")
        else:
            print(f"Parsing {fname} -> {out_dir} ...", end=" ", flush=True)
            try:
                replay_json = load_preprocessed_replay(in_path)
                game = process_game(replay_json)
                game_json = game.to_dict()
                with open(out_path_json, "w+") as f:
                    json.dump(game_json, f, indent=4)
                with open(out_path_pkl, "wb+") as f:
                    pickle.dump(game, f)
                print("OK")
                succeeded += 1
            except Exception as e:
                print("FAILED")
                print(f"  {type(e).__name__}: {e}")
                failed += 1
                continue  # skip analysis if parsing failed

    print()
    print(f"Done. succeeded={succeeded}, failed={failed}")
    print(f"Saved processed pickles to: {out_dir}")


if __name__ == "__main__":
    main()
