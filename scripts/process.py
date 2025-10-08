"""
Process script: convert preprocessed games into csv files for feature extraction
"""

from rocketleague_ml.models.pipeline import Rocket_League_Pipeline


def main():
    pipeline = Rocket_League_Pipeline()
    pipeline.processor.process_games()


if __name__ == "__main__":
    main()
