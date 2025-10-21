import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Dict, List
from rocketleague_ml.utils.helpers import series_quat_to_euler


def get_distancing_cols(
    game: pd.DataFrame, teams: Dict[str, List[str]]
) -> Dict[str, NDArray[np.float64]]:
    distancing_cols: Dict[str, NDArray[np.float64]] = {}

    for team, players in teams.items():
        opponents_team = "Orange" if team == "Blue" else "Blue"

        for player in players:
            # --- Distance to ball ---
            dx_ball = game[f"{player}_positioning_x"] - game["ball_positioning_x"]
            dy_ball = game[f"{player}_positioning_y"] - game["ball_positioning_y"]
            dz_ball = game[f"{player}_positioning_z"] - game["ball_positioning_z"]

            distancing_cols[f"{player}_distance_to_ball"] = np.sqrt(
                dx_ball**2 + dy_ball**2 + dz_ball**2
            )

            distancing_cols[f"{player}_xy_distance_to_ball"] = np.sqrt(
                dx_ball**2 + dy_ball**2
            )

            # --- Convert quaternion to yaw/pitch ---
            yaw, pitch, _ = series_quat_to_euler(
                game[f"{player}_positioning_rotation_x"],
                game[f"{player}_positioning_rotation_y"],
                game[f"{player}_positioning_rotation_z"],
                game[f"{player}_positioning_rotation_w"],
            )

            # --- Horizontal (XY) angle to ball ---
            ball_angle_xy = np.arctan2(-dy_ball, -dx_ball)
            angle_diff_xy = np.arctan2(
                np.sin(ball_angle_xy - yaw), np.cos(ball_angle_xy - yaw)
            )
            distancing_cols[f"{player}_angle_to_ball"] = angle_diff_xy

            # --- Vertical (Z) angle to ball (using XZ plane) ---
            # Ball elevation relative to player forward pitch
            horizontal_dist = np.sqrt(dx_ball**2 + dy_ball**2)
            ball_angle_z = np.arctan2(-dz_ball, horizontal_dist)
            angle_diff_z = np.arctan2(
                np.sin(ball_angle_z - pitch), np.cos(ball_angle_z - pitch)
            )
            distancing_cols[f"{player}_pitch_angle_to_ball"] = angle_diff_z

            # --- Distance and angle to other players ---
            for p in players + teams[opponents_team]:
                if player == p:
                    continue

                dx = game[f"{player}_positioning_x"] - game[f"{p}_positioning_x"]
                dy = game[f"{player}_positioning_y"] - game[f"{p}_positioning_y"]
                dz = game[f"{player}_positioning_z"] - game[f"{p}_positioning_z"]

                distancing_cols[f"{player}_distance_to_{p}"] = np.sqrt(
                    dx**2 + dy**2 + dz**2
                )

                # Horizontal angle between facing and other player
                target_angle_xy = np.arctan2(-dy, -dx)
                angle_diff_p_xy = np.arctan2(
                    np.sin(target_angle_xy - yaw), np.cos(target_angle_xy - yaw)
                )
                distancing_cols[f"{player}_angle_to_{p}"] = angle_diff_p_xy

                # Vertical angle between facing and other player
                horizontal_dist_p = np.sqrt(dx**2 + dy**2)
                target_angle_z = np.arctan2(-dz, horizontal_dist_p)
                angle_diff_p_z = np.arctan2(
                    np.sin(target_angle_z - pitch), np.cos(target_angle_z - pitch)
                )
                distancing_cols[f"{player}_pitch_angle_to_{p}"] = angle_diff_p_z

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

            dependent_distancing_cols[f"{player}_is_close_to_ball"] = (
                game[f"{player}_distance_to_ball"] <= 1000
            )
            dependent_distancing_cols[f"{player}_is_semi_close_to_ball"] = (
                game[f"{player}_distance_to_ball"] > 1000
            ) & (game[f"{player}_distance_to_ball"] <= 2000)
            dependent_distancing_cols[f"{player}_is_medium_distance_to_ball"] = (
                game[f"{player}_distance_to_ball"] > 2000
            ) & (game[f"{player}_distance_to_ball"] <= 3000)
            dependent_distancing_cols[f"{player}_is_semi_far_from_ball"] = (
                game[f"{player}_distance_to_ball"] > 3000
            ) & (game[f"{player}_distance_to_ball"] <= 4000)
            dependent_distancing_cols[f"{player}_is_far_from_ball"] = (
                game[f"{player}_distance_to_ball"] > 4000
            )

            # TODO: Calculate alignment with current closest_to_the_ball car trajectory
            # TODO: Adjust for both current cars' speed / possible hit strength
            # TODO: Calculate alignment with current ball trajectory
            # TODO: Adjust for current ball speed

            for p in players:
                if p == player:
                    continue
                dependent_distancing_cols[f"{player}_is_close_to_{p}"] = (
                    game[f"{player}_distance_to_{p}"] <= 1000
                )
                dependent_distancing_cols[f"{player}_is_semi_close_to_{p}"] = (
                    game[f"{player}_distance_to_{p}"] > 1000
                ) & (game[f"{player}_distance_to_{p}"] <= 2000)
                dependent_distancing_cols[f"{player}_is_medium_distance_to_{p}"] = (
                    game[f"{player}_distance_to_{p}"] > 2000
                ) & (game[f"{player}_distance_to_{p}"] <= 3000)
                dependent_distancing_cols[f"{player}_is_semi_far_from_{p}"] = (
                    game[f"{player}_distance_to_{p}"] > 3000
                ) & (game[f"{player}_distance_to_{p}"] <= 4000)
                dependent_distancing_cols[f"{player}_is_far_from_{p}"] = (
                    game[f"{player}_distance_to_{p}"] > 4000
                )

    return dependent_distancing_cols
