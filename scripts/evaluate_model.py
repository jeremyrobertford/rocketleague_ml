import pandas as pd
from rocketleague_ml.models.train import load_model
from rocketleague_ml.models.evaluate import evaluate_model
from rocketleague_ml.config import PROCESSED

def main():
    X = pd.read_csv(f"{PROCESSED}/features.csv")
    y = pd.read_csv(f"{PROCESSED}/labels.csv").values.ravel()

    model = load_model()
    acc, report = evaluate_model(model, X, y)
    print(f"✅ Accuracy: {acc:.2f}")
    print(report)

if __name__ == "__main__":
    main()
