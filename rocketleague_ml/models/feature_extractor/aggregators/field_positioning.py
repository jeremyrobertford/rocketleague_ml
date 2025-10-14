import pandas as pd
from typing import Dict
from rocketleague_ml.models.feature_extractor.aggregators.helpers import (
    get_time_percentage_and_stint_for_cols,
)


def aggregate_field_positioning(
    game: pd.DataFrame, main_player: str
) -> Dict[str, float]:
    features: Dict[str, float] = {}

    positioning_columns = {
        # Halves
        "In Offensive Half": ["offensive_half"],
        "In Defensive Half": ["defensive_half"],
        "In Left Half": ["left_half"],
        "In Right Half": ["right_half"],
        "In Highest Half": ["highest_half"],
        "In Lowest Half": ["lowest_half"],
        # Thirds (longitudinal)
        "In Offensive Third": ["offensive_third"],
        "In Neutral Third": ["neutral_third"],
        "In Defensive Third": ["defensive_third"],
        # Thirds (lateral)
        "In Left Third": ["left_third"],
        "In Middle Third": ["middle_third"],
        "In Right Third": ["right_third"],
        # Third (Vertical)
        "In Highest Third": ["highest_third"],
        "In Middle Aerial Third": ["middle_aerial_third"],
        "In Lowest Third": ["lowest_third"],
        # Half intersections
        "In Offensive Left Half": [
            "offensive_half",
            "left_half",
        ],
        "In Offensive Right Half": [
            "offensive_half",
            "right_half",
        ],
        "In Defensive Left Half": [
            "defensive_half",
            "left_half",
        ],
        "In Defensive Right Half": [
            "defensive_half",
            "right_half",
        ],
        # Half intersections with verticality
        "In Offensive Left Highest Half": [
            "offensive_half",
            "left_half",
            "highest_half",
        ],
        "In Offensive Left Lowest Half": [
            "offensive_half",
            "left_half",
            "lowest_half",
        ],
        "In Offensive Right Highest Half": [
            "offensive_half",
            "right_half",
            "highest_half",
        ],
        "In Offensive Right Lowest Half": [
            "offensive_half",
            "right_half",
            "lowest_half",
        ],
        "In Defensive Left Highest Half": [
            "defensive_half",
            "left_half",
            "highest_half",
        ],
        "In Defensive Left Lowest Half": [
            "defensive_half",
            "left_half",
            "lowest_half",
        ],
        "In Defensive Right Highest Half": [
            "defensive_half",
            "right_half",
            "highest_half",
        ],
        "In Defensive Right Lowest Half": [
            "defensive_half",
            "right_half",
            "lowest_half",
        ],
        # Third intersections
        "In Offensive Left Third": [
            "offensive_third",
            "left_third",
        ],
        "In Offensive Middle Third": [
            "offensive_third",
            "middle_third",
        ],
        "In Offensive Right Third": [
            "offensive_third",
            "right_third",
        ],
        "In Neutral Left Third": [
            "neutral_third",
            "left_third",
        ],
        "In Neutral Middle Third": [
            "neutral_third",
            "middle_third",
        ],
        "In Neutral Right Third": [
            "neutral_third",
            "right_third",
        ],
        "In Defensive Left Third": [
            "defensive_third",
            "left_third",
        ],
        "In Defensive Middle Third": [
            "defensive_third",
            "middle_third",
        ],
        "In Defensive Right Third": [
            "defensive_third",
            "right_third",
        ],
        # Third intersections with verticality
        "In Offensive Left Highest Third": [
            "offensive_third",
            "left_third",
            "highest_third",
        ],
        "In Offensive Middle Highest Third": [
            "offensive_third",
            "middle_third",
            "highest_third",
        ],
        "In Offensive Right Highest Third": [
            "offensive_third",
            "right_third",
            "highest_third",
        ],
        "In Neutral Left Highest Third": [
            "neutral_third",
            "left_third",
            "highest_third",
        ],
        "In Neutral Middle Highest Third": [
            "neutral_third",
            "middle_third",
            "highest_third",
        ],
        "In Neutral Right Highest Third": [
            "neutral_third",
            "right_third",
            "highest_third",
        ],
        "In Defensive Left Highest Third": [
            "defensive_third",
            "left_third",
            "highest_third",
        ],
        "In Defensive Middle Highest Third": [
            "defensive_third",
            "middle_third",
            "highest_third",
        ],
        "In Defensive Right Highest Third": [
            "defensive_third",
            "right_third",
            "highest_third",
        ],
        "In Offensive Left Middle-Aerial Third": [
            "offensive_third",
            "left_third",
            "middle_aerial_third",
        ],
        "In Offensive Middle Middle-Aerial Third": [
            "offensive_third",
            "middle_third",
            "middle_aerial_third",
        ],
        "In Offensive Right Middle-Aerial Third": [
            "offensive_third",
            "right_third",
            "middle_aerial_third",
        ],
        "In Neutral Left Middle-Aerial Third": [
            "neutral_third",
            "left_third",
            "middle_aerial_third",
        ],
        "In Neutral Middle Middle-Aerial Third": [
            "neutral_third",
            "middle_third",
            "middle_aerial_third",
        ],
        "In Neutral Right Middle-Aerial Third": [
            "neutral_third",
            "right_third",
            "middle_aerial_third",
        ],
        "In Defensive Left Middle-Aerial Third": [
            "defensive_third",
            "left_third",
            "middle_aerial_third",
        ],
        "In Defensive Middle Middle-Aerial Third": [
            "defensive_third",
            "middle_third",
            "middle_aerial_third",
        ],
        "In Defensive Right Middle-Aerial Third": [
            "defensive_third",
            "right_third",
            "middle_aerial_third",
        ],
        "In Offensive Left Lowest Third": [
            "offensive_third",
            "left_third",
            "lowest_third",
        ],
        "In Offensive Middle Lowest Third": [
            "offensive_third",
            "middle_third",
            "lowest_third",
        ],
        "In Offensive Right Lowest Third": [
            "offensive_third",
            "right_third",
            "lowest_third",
        ],
        "In Neutral Left Lowest Third": [
            "neutral_third",
            "left_third",
            "lowest_third",
        ],
        "In Neutral Middle Lowest Third": [
            "neutral_third",
            "middle_third",
            "lowest_third",
        ],
        "In Neutral Right Lowest Third": [
            "neutral_third",
            "right_third",
            "lowest_third",
        ],
        "In Defensive Left Lowest Third": [
            "defensive_third",
            "left_third",
            "lowest_third",
        ],
        "In Defensive Middle Lowest Third": [
            "defensive_third",
            "middle_third",
            "lowest_third",
        ],
        "In Defensive Right Lowest Third": [
            "defensive_third",
            "right_third",
            "lowest_third",
        ],
        # Ball-relative positioning
        "In Front of Ball": ["in_front_of_ball"],
        "Behind Ball": ["behind_ball"],
        # Movement state
        "Grounded": ["grounded"],
        "Airborne": ["airborne"],
        # Surfaces
        "On Ceiling": ["on_ceiling"],
        "On Left Wall": ["on_left_wall"],
        "On Right Wall": ["on_right_wall"],
        "On Back Wall": ["on_back_wall"],
        "On Front Wall": ["on_front_wall"],
        # Goal zones
        "In Own Goal": ["in_own_goal"],
        "In Opponents Goal": ["in_opponents_goal"],
    }
    for label, cols in positioning_columns.items():
        percent_time, average_percent_stint = get_time_percentage_and_stint_for_cols(
            game, cols, main_player
        )
        features[f"Percent Time {label}"] = percent_time
        features[f"Average Stint {label}"] = average_percent_stint

    return features
