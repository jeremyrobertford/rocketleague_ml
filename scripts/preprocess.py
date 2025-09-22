from rocketleague_ml.data.loader import load_replay
from rocketleague_ml.data.features import extract_features
import os
import pandas as pd
from rocketleague_ml.config import RAW_REPLAYS, PROCESSED

def main():
    all_X, all_y = [], []
    for fname in os.listdir(RAW_REPLAYS):
        if fname.endswith(".replay"):
            data = load_replay(os.path.join(RAW_REPLAYS, fname))
            feats = extract_features(data)
            if feats["X"] is not None:
                all_X.append(feats["X"])
                all_y.append(feats["y"])

    if all_X:
        X = pd.concat(all_X)
        y = pd.concat(all_y)
        X.to_csv(os.path.join(PROCESSED, "features.csv"), index=False)
        y.to_csv(os.path.join(PROCESSED, "labels.csv"), index=False)
        print("✅ Preprocessing complete.")
    else:
        print("⚠️ No replays found or feature extraction not implemented.")

if __name__ == "__main__":
    main()