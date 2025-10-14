import pandas as pd
from typing import Dict
from rocketleague_ml.models.feature_extractor.aggregators.helpers import (
    get_time_percentage_and_stint_for_cols,
)


def aggregate_speed(game: pd.DataFrame, main_player: str) -> Dict[str, float]:
    features: Dict[str, float] = {}

    speed_columns = {
        "Stationary": ["is_stationary"],
        "Slow": ["is_slow"],
        "Semi-Slow": ["is_semi_slow"],
        "Medium-Speed": ["is_medium_speed"],
        "Semi-Fast": ["is_semi_fast"],
        "Fast": ["is_fast"],
        "Drive Speed": ["is_drive_speed"],
        "Boost Speed": ["is_boost_speed"],
        "Supersonic": ["is_supersonic"],
    }

    for label, cols in speed_columns.items():
        percent_time, average_percent_stint = get_time_percentage_and_stint_for_cols(
            game, cols, main_player
        )
        features[f"Percent Time while {label}"] = percent_time
        features[f"Average Stint while {label}"] = average_percent_stint

    features[f"Average Speed"] = (
        game[f"{main_player}_positioning_linear_velocity"].mean() / 100
    )
    return features
