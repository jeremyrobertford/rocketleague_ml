import pandas as pd
from rocketleague_ml.models.train import train_model, save_model
from rocketleague_ml.config import PROCESSED

def main():
    X = pd.read_csv(f"{PROCESSED}/features.csv")
    y = pd.read_csv(f"{PROCESSED}/labels.csv").values.ravel()

    model, (X_test, y_test) = train_model(X, y)
    save_model(model)
    print("✅ Model trained and saved as model.pkl")

if __name__ == "__main__":
    main()
