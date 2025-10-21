import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Dict, List, Tuple


def get_touch_type_cols(game: pd.DataFrame, teams: Dict[str, List[str]]):
    toward_goal_tol = 0.7
    toward_teammate_tol = 0.6
    toward_opponent_tol = 0.6
    toward_open_space_tol: float | None = None

    # Define goal directions (Rocket League convention)
    GOAL_BLUE = np.array([0, -5120, 0])
    GOAL_ORANGE = np.array([0, 5120, 0])
    touch_type_cols: Dict[str, pd.Series] = {}

    # Ball position
    ball_pos = game[
        [
            "ball_positioning_x",
            "ball_positioning_y",
            "ball_positioning_z",
        ]
    ].to_numpy()
    # Precompute player positions
    positions: Dict[str, np.ndarray[Tuple[int, int], np.dtype[np.float64]]] = {}
    for p in teams["Blue"] + teams["Orange"]:
        prefix = f"{p}_hit_ball_"
        cols = [
            prefix + "impulse_x",
            prefix + "impulse_y",
            prefix + "impulse_z",
            prefix + "collision_confidence",
        ]
        for col in cols:
            if col not in game:
                game[col] = 0
        positions[p] = game[
            [
                f"{p}_positioning_x",
                f"{p}_positioning_y",
                f"{p}_positioning_z",
            ]
        ].to_numpy()

    for team, players in teams.items():

        for p in players:
            prefix = f"{p}_hit_ball_"
            impulses = game[
                [prefix + "impulse_x", prefix + "impulse_y", prefix + "impulse_z"]
            ].to_numpy()
            confidence = game[prefix + "collision_confidence"].fillna(0).to_numpy()  # type: ignore

            hit_dir = np.nan_to_num(
                impulses / (np.linalg.norm(impulses, axis=1, keepdims=True) + 1e-6)
            )

            # Compute directional vectors
            goal_target = GOAL_ORANGE if team.lower() == "blue" else GOAL_BLUE
            goal_vec = goal_target - ball_pos
            goal_vec /= np.linalg.norm(goal_vec, axis=1, keepdims=True) + 1e-6
            to_goal = np.sum(hit_dir * goal_vec, axis=1)

            # Teammate direction
            teammate_positions = [
                positions[player] for player in players if p != player
            ]
            if teammate_positions:
                team_center = np.mean(np.stack(teammate_positions), axis=0)
                team_vec = team_center - ball_pos
                team_vec /= np.linalg.norm(team_vec, axis=1, keepdims=True) + 1e-6
                to_teammate = np.sum(hit_dir * team_vec, axis=1)
            else:
                to_teammate = np.zeros(len(game))

            # Opponent direction
            opponent_team = [t for t in teams.keys() if t.lower() != team.lower()][0]
            opponent_positions = [
                positions[player] for player in teams[opponent_team] if p != player
            ]
            if opponent_positions:
                opp_center = np.mean(np.stack(opponent_positions), axis=0)
                opp_vec = opp_center - ball_pos
                opp_vec /= np.linalg.norm(opp_vec, axis=1, keepdims=True) + 1e-6
                to_opponent = np.sum(hit_dir * opp_vec, axis=1)
            else:
                to_opponent = np.zeros(len(game))

            # Open space heuristic
            open_space_score = (
                1
                - np.maximum.reduce(
                    [np.abs(to_goal), np.abs(to_teammate), np.abs(to_opponent)]
                )
                if toward_open_space_tol is None
                else toward_open_space_tol
            )

            # Detect multiple collisions (50/50s)
            collision_count = np.sum(
                [
                    ~game[f"{player}_hit_ball_collision_confidence"].isna().to_numpy()
                    for player in positions
                ],
                axis=0,
            )
            fifty_fifty_mask = collision_count > 1

            # Create independent boolean masks for each touch type
            categories: Dict[str, pd.Series | NDArray[np.float64 | np.bool_]] = {
                "fifty_fifty": fifty_fifty_mask,
                "towards_goal": (to_goal > toward_goal_tol),
                "towards_teammate": (to_teammate > toward_teammate_tol),
                "towards_opponent": (to_opponent > toward_opponent_tol),
                "towards_open_space": (
                    (open_space_score > 0.5)
                    & ~(to_goal > toward_goal_tol)
                    & ~(to_teammate > toward_teammate_tol)
                    & ~(to_opponent > toward_opponent_tol)
                    & ~fifty_fifty_mask
                ),
            }

            # Total touch detection
            touch_mask = confidence > 0
            touch_type_cols[f"{p}_touch_total"] = pd.Series(
                touch_mask.astype(int), index=game.index
            )

            # Add direction dot products for analysis
            touch_type_cols[f"{p}_touch_to_goal_dot"] = pd.Series(
                to_goal, index=game.index
            )
            touch_type_cols[f"{p}_touch_to_teammate_dot"] = pd.Series(
                to_teammate, index=game.index
            )
            touch_type_cols[f"{p}_touch_to_opponent_dot"] = pd.Series(
                to_opponent, index=game.index
            )
            touch_type_cols[f"{p}_touch_open_space_score"] = pd.Series(
                open_space_score, index=game.index
            )

            # Add independent category flags
            for cat, mask in categories.items():
                touch_type_cols[f"{p}_touch_{cat}"] = pd.Series(
                    (mask & touch_mask).astype(int), index=game.index  # type: ignore
                )

            # TODO: Calculate hit strength alignment with distance from ball

    return touch_type_cols
