import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Dict, List

# TODO: Add polar angles to teammates, opponents, and ball
# Adjust the polar angles so it is based off of car normal?


def get_distancing_cols(game: pd.DataFrame, teams: Dict[str, List[str]]):
    distancing_cols: Dict[str, NDArray[np.float64]] = {}
    for team, players in teams.items():
        opponents_team = "Orange" if team == "Blue" else "Blue"
        for player in players:
            distancing_cols[f"{player}_distance_to_ball"] = np.sqrt(
                (game[f"{player}_positioning_x"] - game["ball_positioning_x"]) ** 2
                + (game[f"{player}_positioning_y"] - game["ball_positioning_y"]) ** 2
                + (game[f"{player}_positioning_z"] - game["ball_positioning_z"]) ** 2
            )
            for p in players + teams[opponents_team]:
                if player == p:
                    continue
                distancing_cols[f"{player}_distance_to_{p}"] = np.sqrt(
                    (game[f"{player}_positioning_x"] - game[f"{p}_positioning_x"]) ** 2
                    + (game[f"{player}_positioning_y"] - game[f"{p}_positioning_y"])
                    ** 2
                    + (game[f"{player}_positioning_z"] - game[f"{p}_positioning_z"])
                    ** 2
                )

    return distancing_cols


def get_dependent_distancing_cols(game: pd.DataFrame, teams: Dict[str, List[str]]):
    ball_distance_cols = [c for c in game.columns if c.endswith("_distance_to_ball")]
    dependent_distancing_cols: Dict[str, pd.Series] = {}
    for players in teams.values():
        for player in players:
            dependent_distancing_cols[f"{player}_closest_to_ball"] = game[
                f"{player}_distance_to_ball"
            ] == game[ball_distance_cols].min(axis=1)
            dependent_distancing_cols[f"{player}_farthest_from_ball"] = game[
                f"{player}_distance_to_ball"
            ] == game[ball_distance_cols].max(axis=1)

    return dependent_distancing_cols
