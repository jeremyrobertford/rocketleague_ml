# type: ignore
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d
from numpy.typing import NDArray
from typing import Dict, List, Tuple


def inefficient_get_touch_type_cols(
    game: pd.DataFrame,
    teams: Dict[str, List[str]],
    open_space_radius: float = 200,
    fifty_fifty_window: int = 3,
) -> Dict[str, pd.Series | NDArray[np.float64]]:
    """
    Computes touch type columns for a game DataFrame with boolean *_hit_ball / *_hit_{player} columns.
    Produces:
      - total touches
      - directional touches (to goal, teammate, opponent)
      - open space touches
      - 50/50 touches
      - hit strength
      - hit strength alignment to closest/average opponent
    """
    touch_type_cols = {}
    ball_pos = game[
        ["ball_positioning_x", "ball_positioning_y", "ball_positioning_z"]
    ].to_numpy()
    ball_vel = game[
        ["ball_linear_velocity_x", "ball_linear_velocity_y", "ball_linear_velocity_z"]
    ].to_numpy()

    # Precompute car positions
    positions = {}
    for p in teams["Blue"] + teams["Orange"]:
        positions[p] = game[
            [f"{p}_positioning_x", f"{p}_positioning_y", f"{p}_positioning_z"]
        ].to_numpy()

    # Goal definitions
    GOAL_BLUE = np.array([0, -5120, 0])
    GOAL_ORANGE = np.array([0, 5120, 0])

    # Collect all player hit boolean columns for 50/50 detection
    player_hit_cols = {p: f"{p}_hit_ball" for p in teams["Blue"] + teams["Orange"]}

    for team, players in teams.items():
        for p in players:
            hit_col = f"{p}_hit_ball"
            if hit_col not in game:
                continue

            touch_mask = game[hit_col].fillna(False).to_numpy().astype(bool)
            touch_type_cols[f"{p}_touch_total"] = pd.Series(
                touch_mask.astype(int), index=game.index
            )

            # Hit strength: change in ball velocity
            delta_v = np.zeros(len(game))
            delta_v[1:] = np.linalg.norm(ball_vel[1:] - ball_vel[:-1], axis=1)
            hit_strength = delta_v * touch_mask
            touch_type_cols[f"{p}_touch_strength"] = pd.Series(
                hit_strength, index=game.index
            )

            # Directional touch types
            player_pos = positions[p]
            relative_dir = ball_pos - player_pos
            relative_dir_norm = relative_dir / (
                np.linalg.norm(relative_dir, axis=1, keepdims=True) + 1e-6
            )

            # Goal direction
            goal_target = GOAL_ORANGE if team.lower() == "blue" else GOAL_BLUE
            goal_vec = goal_target - ball_pos
            goal_vec /= np.linalg.norm(goal_vec, axis=1, keepdims=True) + 1e-6
            to_goal = np.sum(relative_dir_norm * goal_vec, axis=1)
            touch_type_cols[f"{p}_touch_towards_goal"] = pd.Series(
                (to_goal > 0.7) & touch_mask, index=game.index
            ).astype(int)

            # Teammate direction
            teammate_positions = [positions[tp] for tp in players if tp != p]
            if teammate_positions:
                team_center = np.mean(np.stack(teammate_positions), axis=0)
                team_vec = team_center - ball_pos
                team_vec /= np.linalg.norm(team_vec, axis=1, keepdims=True) + 1e-6
                to_teammate = np.sum(relative_dir_norm * team_vec, axis=1)
            else:
                to_teammate = np.zeros(len(game))
            touch_type_cols[f"{p}_touch_towards_teammate"] = pd.Series(
                (to_teammate > 0.6) & touch_mask, index=game.index
            ).astype(int)

            # Opponent direction
            opponent_team = [t for t in teams.keys() if t.lower() != team.lower()][0]
            opponent_positions = [positions[op] for op in teams[opponent_team]]
            if opponent_positions:
                opp_center = np.mean(np.stack(opponent_positions), axis=0)
                opp_vec = opp_center - ball_pos
                opp_vec /= np.linalg.norm(opp_vec, axis=1, keepdims=True) + 1e-6
                to_opponent = np.sum(relative_dir_norm * opp_vec, axis=1)
            else:
                to_opponent = np.zeros(len(game))
            touch_type_cols[f"{p}_touch_towards_opponent"] = pd.Series(
                (to_opponent > 0.6) & touch_mask, index=game.index
            ).astype(int)

            # Open space touch: check no cars along projected path within radius
            open_space_mask = np.zeros(len(game), dtype=bool)
            for i, t in enumerate(touch_mask.nonzero()[0]):
                ball_future_pos = ball_pos[i] + relative_dir_norm[i] * open_space_radius
                obstructed = False
                for car_pos in positions.values():
                    if np.linalg.norm(car_pos[i] - ball_future_pos) < open_space_radius:
                        obstructed = True
                        break
                if not obstructed:
                    open_space_mask[i] = True
            touch_type_cols[f"{p}_touch_open_space"] = pd.Series(
                open_space_mask, index=game.index
            ).astype(int)

            # 50/50 detection
            fifty_fifty_mask = np.zeros(len(game), dtype=bool)
            for i, t in enumerate(touch_mask.nonzero()[0]):
                for other_p, other_col in player_hit_cols.items():
                    if other_p == p:
                        continue
                    other_touch = game[other_col].fillna(False).to_numpy()
                    if np.any(
                        other_touch[
                            max(0, t - fifty_fifty_window) : t + fifty_fifty_window + 1
                        ]
                    ):
                        fifty_fifty_mask[t] = True
                        break
            touch_type_cols[f"{p}_touch_fifty_fifty"] = pd.Series(
                fifty_fifty_mask.astype(int), index=game.index
            )

            # Hit strength alignment (non-50/50 only)
            non_5050_mask = touch_mask & ~fifty_fifty_mask
            if np.any(non_5050_mask):
                # closest opponent
                closest_dist = np.min(
                    [
                        np.linalg.norm(ball_pos - opp, axis=1)
                        for opp in opponent_positions
                    ],
                    axis=0,
                )
                hit_strength_closest = hit_strength * (1 / (closest_dist + 1e-6))
                hit_strength_closest[~non_5050_mask] = 0
                touch_type_cols[f"{p}_hit_strength_alignment_closest_opp"] = pd.Series(
                    hit_strength_closest, index=game.index
                )

                # average opponent
                avg_opp_pos = np.mean(np.stack(opponent_positions), axis=0)
                avg_dist = np.linalg.norm(ball_pos - avg_opp_pos, axis=1)
                hit_strength_avg = hit_strength * (1 / (avg_dist + 1e-6))
                hit_strength_avg[~non_5050_mask] = 0
                touch_type_cols[f"{p}_hit_strength_alignment_avg_opp"] = pd.Series(
                    hit_strength_avg, index=game.index
                )
            else:
                touch_type_cols[f"{p}_hit_strength_alignment_closest_opp"] = pd.Series(
                    np.zeros(len(game)), index=game.index
                )
                touch_type_cols[f"{p}_hit_strength_alignment_avg_opp"] = pd.Series(
                    np.zeros(len(game)), index=game.index
                )

    return touch_type_cols


def get_touch_type_cols(
    game: pd.DataFrame,
    teams: Dict[str, List[str]],
    open_space_radius: float = 200,
    fifty_fifty_window: int = 3,
) -> Dict[str, NDArray[np.float64]]:
    """
    Computes touch type columns for a game DataFrame with boolean *_hit_ball columns.
    Vectorized implementation for 50/50 detection and open space touches.
    Produces:
      - total touches
      - directional touches (to goal, teammate, opponent)
      - open space touches
      - 50/50 touches
      - hit strength
      - hit strength alignment to closest/average opponent
    """
    touch_type_cols = {}
    num_frames = len(game)

    # Ball positions and velocity
    ball_pos = game[
        ["ball_positioning_x", "ball_positioning_y", "ball_positioning_z"]
    ].to_numpy()
    ball_vel = game[
        [
            "ball_positioning_linear_velocity_x",
            "ball_positioning_linear_velocity_y",
            "ball_positioning_linear_velocity_z",
        ]
    ].to_numpy()

    # Precompute player positions
    positions = {
        p: game[
            [f"{p}_positioning_x", f"{p}_positioning_y", f"{p}_positioning_z"]
        ].to_numpy()
        for p in teams["Blue"] + teams["Orange"]
    }

    # Goals
    GOAL_BLUE = np.array([0, -5120, 0])
    GOAL_ORANGE = np.array([0, 5120, 0])

    # Stack all hit_ball columns for 50/50 detection
    players = teams["Blue"] + teams["Orange"]
    hit_matrix = np.stack(
        [game[p + "_hit_ball"].fillna(False).astype(int).to_numpy() for p in players],
        axis=1,
    )  # (frames, players)

    # Vectorized 50/50 detection using max filter along time axis
    filtered = maximum_filter1d(
        hit_matrix, size=2 * fifty_fifty_window + 1, axis=0, mode="constant"
    )
    fifty_fifty_matrix = (
        filtered - hit_matrix
    ) > 0  # True where other player touches within window

    # Precompute relative vectors to ball and ball future positions for open space
    future_ball_offsets = {}
    relative_dirs = {}
    for p in players:
        player_pos = positions[p]
        relative_dir = ball_pos - player_pos
        relative_dir_norm = relative_dir / (
            np.linalg.norm(relative_dir, axis=1, keepdims=True) + 1e-6
        )
        relative_dirs[p] = relative_dir_norm
        future_ball_offsets[p] = relative_dir_norm * open_space_radius

    # Stack car positions for open space check
    car_stack = np.stack(list(positions.values()), axis=1)  # (frames, cars, 3)

    # Precompute opponent positions
    opponent_positions = {}
    for team, team_players in teams.items():
        opp_team = [t for t in teams.keys() if t.lower() != team.lower()][0]
        for p in team_players:
            opponent_positions[p] = [positions[op] for op in teams[opp_team]]

    # Main loop over players (vectorized per player)
    for idx_p, p in enumerate(players):
        touch_mask = hit_matrix[:, idx_p].astype(bool)
        touch_type_cols[f"{p}_touch_total"] = pd.Series(
            touch_mask.astype(int), index=game.index
        )

        # Hit strength: ball velocity change magnitude
        delta_v = np.zeros(num_frames)
        delta_v[1:] = np.linalg.norm(ball_vel[1:] - ball_vel[:-1], axis=1)
        hit_strength = delta_v * touch_mask
        touch_type_cols[f"{p}_touch_strength"] = pd.Series(
            hit_strength, index=game.index
        )

        # Directional touch types
        relative_dir_norm = relative_dirs[p]
        # Goal
        team = "Blue" if p in teams["Blue"] else "Orange"
        goal_target = GOAL_ORANGE if team.lower() == "blue" else GOAL_BLUE
        goal_vec = goal_target - ball_pos
        goal_vec /= np.linalg.norm(goal_vec, axis=1, keepdims=True) + 1e-6
        to_goal = np.sum(relative_dir_norm * goal_vec, axis=1)
        touch_type_cols[f"{p}_touch_towards_goal"] = pd.Series(
            (to_goal > 0.7) & touch_mask, index=game.index
        ).astype(int)

        # Teammate
        teammates = [tp for tp in teams[team] if tp != p]
        if teammates:
            team_center = np.mean(np.stack([positions[tp] for tp in teammates]), axis=0)
            team_vec = team_center - ball_pos
            team_vec /= np.linalg.norm(team_vec, axis=1, keepdims=True) + 1e-6
            to_teammate = np.sum(relative_dir_norm * team_vec, axis=1)
        else:
            to_teammate = np.zeros(num_frames)
        touch_type_cols[f"{p}_touch_towards_teammate"] = pd.Series(
            (to_teammate > 0.6) & touch_mask, index=game.index
        ).astype(int)

        # Opponent
        opp_positions = opponent_positions[p]
        if opp_positions:
            opp_center = np.mean(np.stack(opp_positions), axis=0)
            opp_vec = opp_center - ball_pos
            opp_vec /= np.linalg.norm(opp_vec, axis=1, keepdims=True) + 1e-6
            to_opponent = np.sum(relative_dir_norm * opp_vec, axis=1)
        else:
            to_opponent = np.zeros(num_frames)
        touch_type_cols[f"{p}_touch_towards_opponent"] = pd.Series(
            (to_opponent > 0.6) & touch_mask, index=game.index
        ).astype(int)

        # Open space: vectorized
        future_ball_pos = ball_pos + future_ball_offsets[p]
        distances = np.linalg.norm(car_stack - future_ball_pos[:, None, :], axis=2)
        open_space_mask = np.all(distances >= open_space_radius, axis=1) & touch_mask
        touch_type_cols[f"{p}_touch_open_space"] = pd.Series(
            open_space_mask.astype(int), index=game.index
        )

        # 50/50
        fifty_fifty_mask = fifty_fifty_matrix[:, idx_p]
        touch_type_cols[f"{p}_touch_fifty_fifty"] = pd.Series(
            fifty_fifty_mask.astype(int), index=game.index
        )

        # Hit strength alignment for non-50/50 touches
        non_5050_mask = touch_mask & ~fifty_fifty_mask
        hit_strength_closest = np.zeros(num_frames)
        hit_strength_avg = np.zeros(num_frames)
        if np.any(non_5050_mask):
            # Closest opponent
            closest_dist = np.min(
                [np.linalg.norm(ball_pos - opp, axis=1) for opp in opp_positions],
                axis=0,
            )
            hit_strength_closest[non_5050_mask] = hit_strength[non_5050_mask] / (
                closest_dist[non_5050_mask] + 1e-6
            )
            # Average opponent
            avg_opp_pos = np.mean(np.stack(opp_positions), axis=0)
            avg_dist = np.linalg.norm(ball_pos - avg_opp_pos, axis=1)
            hit_strength_avg[non_5050_mask] = hit_strength[non_5050_mask] / (
                avg_dist[non_5050_mask] + 1e-6
            )

        touch_type_cols[f"{p}_hit_strength_alignment_closest_opp"] = pd.Series(
            hit_strength_closest, index=game.index
        )
        touch_type_cols[f"{p}_hit_strength_alignment_avg_opp"] = pd.Series(
            hit_strength_avg, index=game.index
        )

    return touch_type_cols
