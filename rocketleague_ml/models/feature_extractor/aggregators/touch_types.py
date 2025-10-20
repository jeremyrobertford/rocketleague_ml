import pandas as pd
from typing import Dict


def aggregate_demos(game: pd.DataFrame, main_player: str) -> Dict[str, float]:
    features: Dict[str, float] = {}

    features["Touches"] = game[f"{main_player}_touch_total"].sum()
    features["Fifty-Fifty Touches"] = game[f"{main_player}_touch_fifty_fifty"].sum()
    features["Towards Goal Touches"] = game[f"{main_player}_touch_towards_goal"].sum()
    features["Towards Teammate Touches"] = game[
        f"{main_player}_touch_towards_teammate"
    ].sum()
    features["Towards Opponent Touches"] = game[
        f"{main_player}_touch_towards_opponent"
    ].sum()
    features["Towards Open Space Touches"] = game[
        f"{main_player}_touch_towards_open_space"
    ].sum()
    return features
