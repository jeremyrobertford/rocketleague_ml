"""
Run script: overwrite everything and run it all
"""

from rocketleague_ml.models.pipeline import Rocket_League_Pipeline


def main():
    pipeline = Rocket_League_Pipeline()
    pipeline.preprocessor.convert_replays(overwrite=False)
    pipeline.wrangler.wrangle_games(overwrite=False)
    pipeline.processor.process_games(overwrite=True)
    pipeline.extractor.extract_features(overwrite=True)


if __name__ == "__main__":
    main()
