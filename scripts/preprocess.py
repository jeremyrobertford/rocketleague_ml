"""
Preprocess script: call rrrocket on raw replays and save JSON to data/processed/json/
"""

from rocketleague_ml.core.pipeline import Rocket_League_Pipeline


def main():
    pipeline = Rocket_League_Pipeline()
    pipeline.preprocessor.convert_replays(overwrite=True)


if __name__ == "__main__":
    main()
