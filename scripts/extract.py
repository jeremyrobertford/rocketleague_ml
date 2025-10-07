"""
Extract script: extra features from processed games
"""

from rocketleague_ml.core.pipeline import Rocket_League_Pipeline


def main():
    pipeline = Rocket_League_Pipeline()
    pipeline.preprocessor.convert_replays()
    pipeline.processor.process_games()
    pipeline.extractor.extract_features(main_player="RL_LionHeart", overwrite=True)


if __name__ == "__main__":
    main()
