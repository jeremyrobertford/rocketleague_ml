"""
Extract script: extra features from processed games
"""

from rocketleague_ml.models.pipeline import Rocket_League_Pipeline


def main():
    pipeline = Rocket_League_Pipeline()
    pipeline.extractor.extract_features(overwrite=True)


if __name__ == "__main__":
    main()
