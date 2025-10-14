import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Dict, List


def get_speed_cols(game: pd.DataFrame, teams: Dict[str, List[str]]):
    speed_cols: Dict[str, NDArray[np.float64]] = {}
    for players in teams.values():
        for player in players:
            speed_cols[f"{player}_positioning_linear_velocity"] = np.sqrt(
                (game[f"{player}_positioning_linear_velocity_x"]) ** 2
                + (game[f"{player}_positioning_linear_velocity_y"]) ** 2
                + (game[f"{player}_positioning_linear_velocity_z"]) ** 2
            )

            speed_cols[f"{player}_positioning_angular_velocity"] = np.sqrt(
                (game[f"{player}_positioning_angular_velocity_x"]) ** 2
                + (game[f"{player}_positioning_angular_velocity_y"]) ** 2
                + (game[f"{player}_positioning_angular_velocity_z"]) ** 2
            )

    return speed_cols


def get_dependent_speed_cols(game: pd.DataFrame, teams: Dict[str, List[str]]):
    speed_cols: Dict[str, pd.Series] = {}
    for players in teams.values():
        for player in players:

            speed_cols[f"{player}_is_stationary"] = (
                game[f"{player}_positioning_linear_velocity"] <= 10
            )
            speed_cols[f"{player}_is_slow"] = (
                game[f"{player}_positioning_linear_velocity"] <= 500
            )
            speed_cols[f"{player}_is_semi_slow"] = (
                game[f"{player}_positioning_linear_velocity"] > 500
            ) & (game[f"{player}_positioning_linear_velocity"] <= 1000)
            speed_cols[f"{player}_is_medium_speed"] = (
                game[f"{player}_positioning_linear_velocity"] > 1000
            ) & (game[f"{player}_positioning_linear_velocity"] <= 1500)
            speed_cols[f"{player}_is_semi_fast"] = (
                game[f"{player}_positioning_linear_velocity"] > 1500
            ) & (game[f"{player}_positioning_linear_velocity"] <= 2000)
            speed_cols[f"{player}_is_fast"] = (
                game[f"{player}_positioning_linear_velocity"] > 2000
            )

            speed_cols[f"{player}_is_drive_speed"] = (
                game[f"{player}_positioning_linear_velocity"] <= 1410
            )
            speed_cols[f"{player}_is_boost_speed"] = (
                game[f"{player}_positioning_linear_velocity"] > 1410
            ) & (game[f"{player}_positioning_linear_velocity"] < 2200)
            speed_cols[f"{player}_is_supersonic"] = (
                game[f"{player}_positioning_linear_velocity"] >= 2200
            )

    return speed_cols
