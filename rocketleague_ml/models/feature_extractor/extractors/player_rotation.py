import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Dict, List


def get_player_rotation_cols(
    game: pd.DataFrame, teams: Dict[str, List[str]]
) -> Dict[str, NDArray[np.int_]]:
    """
    Assigns rotation positions (1, 2, 3) to each player for every frame.

    Args:
        game: DataFrame with per-frame data (ball, player positions, velocities, etc.)
        teams: {"blue": [player_names...], "orange": [player_names...]}
        last_touch_times: {player_name: last_touch_frame_index}

    Returns:
        game with added columns f"{player_name}_rotation_position" (values 1–3)
    """
    game = game.copy()

    # Pre-calc common fields
    ball_pos = game[
        ["ball_positioning_x", "ball_positioning_y", "ball_positioning_z"]
    ].values

    player_rotation_cols: Dict[str, NDArray[np.int_]] = {}

    for team_name, players in teams.items():
        goal_y = -5120 if team_name == "blue" else 5120
        team_sign = -1 if team_name == "blue" else 1

        # Build per-player scores
        scores: Dict[str, NDArray[np.float64]] = {}
        for player in players:
            # Positions and velocities
            p_pos = game[
                [
                    f"{player}_positioning_x",
                    f"{player}_positioning_y",
                    f"{player}_positioning_z",
                ]
            ].values
            p_vel = game[
                [
                    f"{player}_positioning_linear_velocity_x",
                    f"{player}_positioning_linear_velocity_y",
                    f"{player}_positioning_linear_velocity_z",
                ]
            ].values

            # Distances
            dist_to_ball = np.linalg.norm(p_pos - ball_pos, axis=1)
            dist_to_goal = np.abs(p_pos[:, 1] - goal_y)

            # Ahead or behind ball (positive = ahead, negative = behind)
            ball_forward = (p_pos[:, 1] - ball_pos[:, 1]) * team_sign

            # Velocity direction toward ball (dot product cosine)
            to_ball = ball_pos - p_pos
            norm_vel = np.linalg.norm(p_vel, axis=1)
            norm_tb = np.linalg.norm(to_ball, axis=1)
            dot = np.einsum("ij,ij->i", p_vel, to_ball)
            toward_ball = np.divide(
                dot,
                norm_vel * norm_tb,
                out=np.zeros_like(dot),
                where=(norm_vel * norm_tb) != 0,
            )

            # Time since last touch normalized
            times: NDArray[np.int_] = game["time"].values  # type: ignore
            touches: NDArray[np.bool_] = game[f"{player}_touch_total"].astype(bool).values  # type: ignore
            last_touch_time = np.zeros_like(times)
            last_time = 0.0
            for i in range(len(times)):
                if touches[i]:
                    last_time = times[i]
                last_touch_time[i] = last_time
            time_since_touch = np.clip(times - last_touch_time, 0, 2.0)

            # Combine into rotation score
            score: NDArray[np.float64] = (
                0.35 * dist_to_ball / (np.max(dist_to_ball) + 1e-6)
                + 0.25 * dist_to_goal / (np.max(dist_to_goal) + 1e-6)
                + 0.2 * ball_forward / (np.max(np.abs(ball_forward)) + 1e-6)
                + 0.1 * time_since_touch
                - 0.1 * toward_ball
            )

            # Out of play adjustment (demoed, deep in opponent zone)
            if f"{player}_is_demoed" in game.columns:
                score += game[f"{player}_is_demoed"].astype(float) * 10
            # Kickoff override: if player is kickoff man, set temporary huge negative bias
            if f"{player}_kickoff" in game.columns:
                score[game[f"{player}_kickoff"].astype(bool)] -= 50.0

            score[team_sign * p_pos[:, 1] > 5500] += 2  # overshot

            scores[player] = score

        # Stack + rank players by score per frame
        score_mat = np.stack([scores[p] for p in players], axis=1)
        ranks = np.argsort(np.argsort(score_mat, axis=1), axis=1) + 1  # 1,2,3

        # Assign back to game
        for i, player in enumerate(players):
            rank = ranks[:, i]
            player_rotation_cols[f"{player}_player_rotation_position"] = rank
            player_rotation_cols[f"{player}_player_rotation_first_man"] = rank == 1
            player_rotation_cols[f"{player}_player_rotation_third_man"] = rank == len(
                players
            )
            player_rotation_cols[f"{player}_player_rotation_second_man"] = ~(
                player_rotation_cols[f"{player}_player_rotation_first_man"]
                | player_rotation_cols[f"{player}_player_rotation_third_man"]
            )

    return player_rotation_cols


def get_simple_player_rotation_cols(game: pd.DataFrame, teams: Dict[str, List[str]]):
    simple_player_rotation_cols: Dict[str, pd.Series] = {}
    for players in teams.values():
        distance_cols = [f"{p}_distance_to_ball" for p in players]
        rank_cols = [f"{p}_simple_player_rotation_position" for p in players]

        # Rank distances among teammates (1 = closest, max = farthest)
        ranks = game[distance_cols].rank(axis=1, method="min")
        ranks.columns = rank_cols

        for player in players:
            # Extract main player's rank
            rank = ranks[f"{player}_simple_player_rotation_position"]

            simple_player_rotation_cols[f"{player}_simple_player_rotation_position"] = (
                rank
            )
            simple_player_rotation_cols[
                f"{player}_simple_player_rotation_first_man"
            ] = (rank == 1)
            simple_player_rotation_cols[
                f"{player}_simple_player_rotation_third_man"
            ] = rank == len(players)
            simple_player_rotation_cols[
                f"{player}_simple_player_rotation_second_man"
            ] = ~(
                simple_player_rotation_cols[
                    f"{player}_simple_player_rotation_first_man"
                ]
                | simple_player_rotation_cols[
                    f"{player}_simple_player_rotation_third_man"
                ]
            )
    return simple_player_rotation_cols
