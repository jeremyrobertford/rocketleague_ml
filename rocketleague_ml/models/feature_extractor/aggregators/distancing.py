import pandas as pd
from typing import Dict, List
from rocketleague_ml.config import TOTAL_FIELD_DISTANCE


def aggregate_distancing(
    game: pd.DataFrame,
    main_player: str,
    teams: Dict[str, List[str]],
) -> Dict[str, float]:
    features: Dict[str, float] = {}
    main_player_team = next(
        (t for t, players in teams.items() if main_player in players), None
    )
    if not main_player_team:
        raise ValueError("No team found")
    opponent_team = "Orange" if main_player_team == "Blue" else "Blue"

    avg_distance_to_ball = game[f"{main_player}_distance_to_ball"].mean()
    features["Average Distance to Ball"] = avg_distance_to_ball / TOTAL_FIELD_DISTANCE
    teammate_distances = [
        f"{main_player}_distance_to_{teammate}"
        for teammate in teams[main_player_team]
        if teammate != main_player
    ]
    avg_distance_to_teammates = game[teammate_distances].mean(axis=1).mean()
    features["Average Distance to Teammates"] = (
        avg_distance_to_teammates / TOTAL_FIELD_DISTANCE
    )
    opponent_distances = [
        f"{main_player}_distance_to_{opponent}" for opponent in teams[opponent_team]
    ]
    avg_distance_to_opponents = game[opponent_distances].mean(axis=1).mean()
    features["Average Distance to Opponents"] = (
        avg_distance_to_opponents / TOTAL_FIELD_DISTANCE
    )

    return features
