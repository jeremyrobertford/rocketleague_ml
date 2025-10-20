"""
Debug script: debug specific portions of the analysis process
"""

import os
import pandas as pd
from rocketleague_ml.config import PROCESSED
from rocketleague_ml.models.pipeline import Rocket_League_Pipeline

# from rocketleague_ml.models.principle_component_analyzer import (
#     Principle_Component_Analyzer,
# )


def main():
    pipeline = Rocket_League_Pipeline()

    def process():  # pyright: ignore[reportUnusedFunction]
        debug_id = "2EF689F5462A8F2B981329B15D08402A"
        replay = pipeline.wrangler.load_wrangled_file(debug_id + ".json")
        pipeline.processor.config(
            include_ball_positioning=True,
            include_car_positioning=True,
            # include_simple_vision=True,
            # include_advanced_vision=True,
            include_movement=True,
            include_player_demos=True,
            include_boost_management=True,
            include_scoreboard_metrics=True,
            include_ball_collisions=True,
            # not developed yet
            include_player_collisions=False,
            include_custom_scoreboard_metrics=False,
            include_mechanics=False,
        )
        result = pipeline.processor.process_games(
            game_datas=[replay], save_output=False, overwrite=True
        )

        if result:
            debug_file_path = os.path.join(PROCESSED, "debug.csv")
            r = pd.DataFrame(result[0])
            r.to_csv(debug_file_path, index=False)

        return result

    # process()

    def extract_features():  # pyright: ignore[reportUnusedFunction]
        processed_game = pipeline.processor.load_processed_file("debug.csv")
        features = pipeline.extractor.extract_features(
            main_player="RL_LionHeart",
            games=[processed_game],
            save_output=False,
            overwrite=True,
        )
        if features is not None:
            print(features.describe().T)

    extract_features()
    # features = pipeline.extractor.load_features()
    # Principle_Component_Analyzer().analyze(features)


if __name__ == "__main__":
    main()
