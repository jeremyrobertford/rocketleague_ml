"""
Debug script: debug specific portions of the analysis process
"""

import os
import pandas as pd
from rocketleague_ml.models.pipeline import Rocket_League_Pipeline
from rocketleague_ml.config import PROCESSED


def main():
    pipeline = Rocket_League_Pipeline()
    debug_id = "2EF689F5462A8F2B981329B15D08402A"
    replay = pipeline.wrangler.load_wrangled_file(debug_id + ".json")
    pipeline.processor.config(
        include_ball_positioning=False,
        include_car_positioning=False,
        include_simple_vision=False,
        include_advanced_vision=False,
        include_movement=False,
        include_player_demos=False,
        include_boost_management=False,
        include_scoreboard_metrics=True,
        # not developed yet
        include_possession=True,
        include_ball_collisions=False,
        include_player_collisions=False,
        include_custom_scoreboard_metrics=False,
        include_mechanics=False,
    )
    result = pipeline.processor.process_games(
        game_datas=[replay], save_output=False, overwrite=False
    )

    if result:
        debug_file_path = os.path.join(PROCESSED, "debug.csv")
        pd.DataFrame(result[0]).to_csv(debug_file_path, index=False)


if __name__ == "__main__":
    main()
