"""
Process script: convert preprocessed games into pkl objects for feature extraction
"""

from rocketleague_ml.core.pipeline import Rocket_League_Pipeline


def main():
    pipeline = Rocket_League_Pipeline()
    pipeline.preprocessor.convert_replays()
    pipeline.processor.process_games(overwrite=True)


if __name__ == "__main__":
    main()
