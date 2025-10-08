import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rocketleague_ml.models.feature_extractor import Rocket_League_Feature_Extractor
from rocketleague_ml.utils.logging import Logger


class Playstyle_Consistency_Model:
    def __init__(
        self,
        extractor: Rocket_League_Feature_Extractor,
        logger: Logger = Logger(),
    ):
        self.extractor = extractor
        self.logger = logger

    def evaluate(
        self,
        features: pd.DataFrame | None = None,
    ):
        self.logger.print(f"Evaluating playstyle consistency...")

        if features is None:
            features = self.extractor.load_features()
        X = features.drop(columns=["id", "round"], errors="ignore")

        means = X.mean()
        stds = X.std()
        coeff_var = stds / means.replace(0, np.nan)  # type: ignore

        summary = pd.DataFrame(
            {"mean": means, "std": stds, "cv": coeff_var}
        ).sort_values("cv")

        self.logger.print(f"Done.")
        self.logger.print()
        return summary

    def plot_feature_consistency(self, summary: pd.DataFrame, top_n=10):
        stable = summary.nsmallest(top_n, "cv")
        variable = summary.nlargest(top_n, "cv")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        stable["cv"].plot(kind="barh", ax=axes[0], title="Most Stable Traits")
        variable["cv"].plot(
            kind="barh", ax=axes[1], title="Most Variable Traits", color="orange"
        )
        plt.tight_layout()
        plt.show()
