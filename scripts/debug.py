"""
Debug script: debug specific portions of the analysis process
"""

import os
import pandas as pd
from rocketleague_ml.config import PROCESSED
from rocketleague_ml.models.pipeline import Rocket_League_Pipeline
from rocketleague_ml.models.principle_component_analyzer import (
    Principle_Component_Analyzer,
)


def main():
    pipeline = Rocket_League_Pipeline()

    def process():  # pyright: ignore[reportUnusedFunction]
        # debug_id = "2EF689F5462A8F2B981329B15D08402A"
        debug_id = "848245D7461E5618628D6398ADDF2E98"
        replay = pipeline.wrangler.load_wrangled_file(debug_id + ".json")
        pipeline.processor.config(
            include_ball_positioning=False,
            include_car_positioning=False,
            include_simple_vision=False,
            include_advanced_vision=False,
            include_movement=True,
            include_player_demos=False,
            include_boost_management=False,
            include_scoreboard_metrics=False,
            include_ball_collisions=False,
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
            r_cols = [c for c in r.columns if "RL_Lion" in c or "time" in c]
            r = r[r_cols]
            r.to_csv(debug_file_path, index=False)

    process()

    def extract_features():  # pyright: ignore[reportUnusedFunction]
        processed_game = pipeline.processor.load_processed_file(
            "2EF689F5462A8F2B981329B15D08402A.csv"
        )
        pipeline.extractor.extract_features(
            main_player="RL_LionHeart",
            games=[processed_game],
            save_output=False,
            overwrite=True,
        )

    # extract_features()
    features = pipeline.extractor.load_features()
    Principle_Component_Analyzer().analyze(features)


if __name__ == "__main__":
    main()
