"""
Extract features script: extra features from processed game
"""

import os
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict
from rocketleague_ml.data.loader import load_processed_game
from rocketleague_ml.config import RAW_REPLAYS, FEATURES
from rocketleague_ml.models.extract import extract_features_from_game_for_player


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def is_pickle_file(fname: str):
    return fname.lower().endswith(".pkl")


def main():
    parser = argparse.ArgumentParser(
        description="Extract features from .pkl game files."
    )
    parser.add_argument(
        "--input", "-i", default=RAW_REPLAYS, help="Folder with .pkl game files"
    )
    parser.add_argument(
        "--output",
        "-o",
        default=FEATURES,
        help="Folder to write extracted feature files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force re-run feature extraction even if it was already done",
    )
    args = parser.parse_args()

    in_dir: str = args.input
    out_dir: str = args.output
    ensure_dir(out_dir)

    replay_files = sorted([f for f in os.listdir(in_dir) if is_pickle_file(f)])
    if not replay_files:
        print(f"No .pkl files found in {in_dir}")
        return

    succeeded = 0
    failed = 0
    base_file_name = "features"
    out_path = os.path.join(out_dir, f"{base_file_name}.csv")

    all_features: List[Dict[str, float]] = []

    if os.path.exists(out_path) and not args.overwrite:
        print(f"Skipping extracted features already present in {base_file_name}")
    else:
        print(f"Extract to {base_file_name} -> {out_dir} ...", end=" ", flush=True)
        for fname in replay_files:
            in_path = os.path.join(in_dir, fname)
            try:
                game = load_processed_game(in_path)
                features = extract_features_from_game_for_player(game, "Gyhl")
                all_features = all_features + features
                print("OK")
                succeeded += 1
            except Exception as e:
                print("FAILED")
                print(f"  {type(e).__name__}: {e}")
                failed += 1
                continue  # skip analysis if parsing failed

    features_df = pd.DataFrame(all_features)
    features_df.to_csv(out_path)

    print()
    print(f"Done. succeeded={succeeded}, failed={failed}")
    print(f"Saved extracted features to: {out_path}")


if __name__ == "__main__":
    main()
