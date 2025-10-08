"""
Evaluate script: evaluate model using extracted features
"""

from rocketleague_ml.models.pipeline import Rocket_League_Pipeline
from rocketleague_ml.models.playstyle_consistency_model import (
    Playstyle_Consistency_Model,
)


def main():
    pipeline = Rocket_League_Pipeline()
    pipeline.extractor.extract_features(main_player="RL_LionHeart")
    pipeline.model = Playstyle_Consistency_Model(
        extractor=pipeline.extractor, logger=pipeline.logger
    )
    summary = pipeline.model.evaluate()
    print(summary.head())
    print(summary.tail())
    pipeline.model.plot_feature_consistency(summary, 10)


if __name__ == "__main__":
    main()
