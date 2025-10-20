import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from rocketleague_ml.utils.helpers import quat_to_euler


def calculate_distances(
    vx: np.float64,
    vy: np.float64,
    yaw: np.float64,
    base_forward_distance: float = 600,
    base_backward_distance: float = 400,
    velocity_scale: float = 1000,
) -> Tuple[np.float64, np.float64]:
    # World velocity
    vel_world = np.array([vx, vy])

    # Rotation matrix for yaw
    rot_matrix = np.array(  # type: ignore
        [
            [np.cos(yaw), np.sin(yaw)],  # forward direction # type: ignore
            [-np.sin(yaw), np.cos(yaw)],  # right direction # type: ignore
        ]
    )

    # Convert world velocity into local car space
    vel_local = rot_matrix @ vel_world  # type: ignore

    forward_speed = np.clip(vel_local[0], -1400, 2300)  # +forward, -backward

    if forward_speed < 0:
        forward_distance = np.interp(forward_speed, [-1400, 0], [0, 400])  # type: ignore
    else:
        forward_distance = np.interp(forward_speed, [0, 2300], [400, 1000])  # type: ignore

    backward_distance = np.interp(forward_speed, [-1400, 2300], [600, 0])  # type: ignore

    return forward_distance, backward_distance  # type: ignore


def assign_complex_possession(
    game: pd.DataFrame,
    teams: Dict[str, List[str]],
    possession_threshold: float = 1.0,
    possession_distance: float = 400,
    challenge_distance: float = 600,
    forward_angle: float = np.pi / 2,
) -> pd.DataFrame:
    """
    Determine per-frame clean and contested possession states for each player.

    Possession rules:
    -----------------
    Gain possession:
        - Player touches the ball and keeps control >= possession_threshold.
        - OR no one has possession and player stays within possession_distance for >= possession_threshold.

    Lose possession:
        - Player touches and the ball moves forward by > forward_distance (angle small)
        - OR sideways/backwards by > sideways_or_backwards_distance (angle large)
        - OR player moves outside possession_distance.

    Contested possession:
        - Occurs when multiple players are within challenge_distance of the ball.
        - OR player with possession is approached by opponent within challenge_distance.

    Clean → contested:
        - Player with clean possession becomes contested when an opponent enters within challenge_distance.
        - Contested lasts until opponent(s) leave or possession is lost.

    Notes:
        - Only xy-plane considered for angle logic.
        - All times computed using 'delta' column for per-frame step.
    """

    game = game.copy()
    n_frames = len(game)
    dt = game["delta"].to_numpy()
    player_names = teams["Blue"] + teams["Orange"]

    # Outputs
    possession = np.zeros((n_frames, len(player_names)), dtype=int)
    possession_losses_from_distance = np.zeros((n_frames, len(player_names)), dtype=int)
    contested = np.zeros((n_frames, len(player_names)), dtype=int)

    # --- Precompute distances ---
    dist_to_ball = {
        p: game[f"{p}_xy_distance_to_ball"].to_numpy() for p in player_names
    }
    touched_ball = {
        p: game[f"{p}_touch_total"].astype(bool).to_numpy() for p in player_names
    }
    angle_to_ball = {p: game[f"{p}_angle_to_ball"].to_numpy() for p in player_names}

    # --- Process each player separately ---
    for p_idx, player in enumerate(player_names):
        state = "none"
        t_since_touch = 0.0

        for i in range(n_frames):
            vx = game.at[i, f"{player}_positioning_linear_velocity_x"].astype(
                np.float64
            )
            vy = game.at[i, f"{player}_positioning_linear_velocity_y"].astype(
                np.float64
            )
            yaw, _, _ = quat_to_euler(
                game.at[i, f"{player}_positioning_rotation_x"],  # type: ignore
                game.at[i, f"{player}_positioning_rotation_y"],  # type: ignore
                game.at[i, f"{player}_positioning_rotation_z"],  # type: ignore
                game.at[i, f"{player}_positioning_rotation_w"],  # type: ignore
            )
            forward_distance, backward_distance = calculate_distances(
                vx=vx,
                vy=vy,
                yaw=yaw,
            )

            in_possession_range = dist_to_ball[player][i] <= possession_distance
            touched = touched_ball[player][i]
            angle = angle_to_ball[player][i]
            player_dist_to_ball = dist_to_ball[player][i]

            if i >= 639:
                pass

            too_far_forward = (
                abs(angle) <= forward_angle and player_dist_to_ball > forward_distance
            )
            too_far_backward = (
                abs(angle) > forward_angle and player_dist_to_ball > backward_distance
            )

            players_close: List[int] = []
            # Check if an opponent entered challenge range
            for team_name, members in teams.items():
                for m in members:
                    if player == m:
                        continue
                    if game.at[i, f"{m}_distance_to_ball"] <= challenge_distance:  # type: ignore
                        m_idx = player_names.index(m)
                        players_close.append(m_idx)

            # --- Handle touch and directional clears ---
            if touched:
                # reset time counter
                # if state != "clean":
                t_since_touch = 0.0
                state = "clean"

            # --- Non-touch transitions ---
            if state == "none":
                if in_possession_range:
                    t_since_touch += dt[i]
                    if t_since_touch >= possession_threshold:
                        state = "clean"
                else:
                    t_since_touch = 0.0
                    continue

            elif state == "clean":
                # If the ball went far enough away after the touch → possession ends
                if too_far_forward or too_far_backward:
                    state = "none"
                    possession_losses_from_distance[i, p_idx] = 1
                    t_since_touch = 0.0
                    continue
                else:
                    if len(players_close):
                        # Check if an opponent entered challenge range
                        for m_idx in players_close:
                            contested[i, m_idx] = 1
                        state = "contested"

            elif state == "contested":
                # Lose contested if no longer in range or no more opponents near
                if too_far_forward or too_far_backward:
                    possession_losses_from_distance[i, p_idx] = 1
                    state = "none"
                    t_since_touch = 0.0
                    continue
                if len(players_close) == 0:
                    state = "clean"

            # --- Mark states ---
            if state == "clean":
                possession[i, p_idx] = 1
            elif state == "contested":
                contested[i, p_idx] = 1

            # advance time
            t_since_touch += dt[i]

    # --- Write per-player results ---
    for idx, player in enumerate(player_names):
        game[f"{player}_in_clean_possession"] = possession[:, idx]
        game[f"{player}_lost_possession_from_distance"] = (
            possession_losses_from_distance[:, idx]
        )
        game[f"{player}_in_contested_possession"] = contested[:, idx]
        game[f"{player}_in_possession"] = possession[:, idx] | contested[:, idx]

        # --- Pressure based on ball side ---
        team_name = next(t for t, ps in teams.items() if player in ps)
        if team_name == "Blue":
            on_opponent_side = game["ball_positioning_x"] > 0
        else:
            on_opponent_side = game["ball_positioning_x"] < 0

        game[f"{player}_applying_clean_pressure"] = (
            game[f"{player}_in_clean_possession"].astype(bool) & on_opponent_side
        ).astype(int)
        game[f"{player}_applying_contested_pressure"] = (
            game[f"{player}_in_contested_possession"].astype(bool) & on_opponent_side
        ).astype(int)
        game[f"{player}_applying_pressure"] = (
            game[f"{player}_in_possession"].astype(bool) & on_opponent_side
        ).astype(int)

    fps = 30  # or whatever your replay frame rate is
    frame_threshold = int(possession_threshold * fps)
    stints: Dict[str, List[Dict[str, int]]] = {}

    for player in player_names:
        stints[player] = []
        possession_col = f"{player}_in_possession"
        clean_col = f"{player}_in_clean_possession"

        possession: np.ndarray[Tuple[int, int], np.dtype[np.int_]] = game[possession_col].values  # type: ignore
        clean: np.ndarray[Tuple[int, int], np.dtype[np.int_]] = game[clean_col].values  # type: ignore

        # Find start/end indices of possession stints
        in_stint = False
        stint_start = 0
        clean_stint_start = 0

        for i in range(len(possession)):
            if possession[i] == 1 and not in_stint:
                in_stint = True
                stint_start = i
                if clean[i]:
                    clean_stint_start = i
            elif (possession[i] == 0 or i == len(possession) - 1) and in_stint:
                in_stint = False
                stint_end = i if possession[i] == 0 else i + 1
                if clean_stint_start:
                    clean_stint_length = stint_end - clean_stint_start
                    valid_stint = clean_stint_length > frame_threshold
                else:
                    valid_stint = False
                clean_stint_start = 0
                stints[player].append(
                    {
                        "start": stint_start,
                        "end": stint_end,
                        "valid": valid_stint,
                    }
                )

    for player in player_names:
        lost_possession_col = f"{player}_lost_possession_from_distance"
        possession_col = f"{player}_in_possession"
        clean_col = f"{player}_in_clean_possession"
        contested_col = f"{player}_in_contested_possession"

        lost_possession: np.ndarray[Tuple[int, int], np.dtype[np.int_]] = game[lost_possession_col].values  # type: ignore
        possession: np.ndarray[Tuple[int, int], np.dtype[np.int_]] = game[possession_col].values  # type: ignore
        clean: np.ndarray[Tuple[int, int], np.dtype[np.int_]] = game[clean_col].values  # type: ignore
        contested: np.ndarray[Tuple[int, int], np.dtype[np.int_]] = game[contested_col].values  # type: ignore

        for stint in stints[player]:
            if stint["valid"]:
                continue
            valid_stint = False
            for other in player_names:
                if other == player:
                    continue
                for other_stint in stints[player]:
                    if (
                        other_stint["start"] <= stint["start"]
                        and stint["start"] <= other_stint["end"]
                        and other_stint["valid"]
                    ):
                        valid_stint = True
                        break
                    if stint["end"] < other_stint["start"]:
                        break
                if valid_stint:
                    break
            if not valid_stint:
                if stint["end"] + 1 <= len(possession):
                    lost_possession[stint["start"] : stint["end"] + 1] = 0
                possession[stint["start"] : stint["end"]] = 0
                clean[stint["start"] : stint["end"]] = 0
                contested[stint["start"] : stint["end"]] = 0

        game[lost_possession_col] = lost_possession
        game[possession_col] = possession
        game[clean_col] = clean
        game[contested_col] = contested

    poss_cols = [c for c in game.columns if "time" in c or "possession" in c]
    game[poss_cols].to_csv(
        "C:/Users/jerem/OneDrive/Documents/rocketleague_ml/data/features/debug.csv",
    )
    return game


def assign_simple_possession(
    game: pd.DataFrame, teams: Dict[str, List[str]]
) -> pd.DataFrame:
    """
    Track simple ball possession per player.
    Possession switches immediately on touch and persists until another touch.

    Adds columns:
        - {player}_simple_possession
        - blue_simple_possession
        - orange_simple_possession
        - contested_simple_possession
    """
    game = game.copy()
    player_names = teams["Blue"] + teams["Orange"]
    n_frames = len(game)
    simple_possession = np.zeros((n_frames, len(player_names)), dtype=int)

    # Precompute touch masks
    touch_cols = {
        p: game[f"{p}_touch_total"].to_numpy().astype(bool) for p in player_names
    }

    current_owners: set[str] = set()

    for i in range(n_frames):
        # Find who touched the ball this frame
        touched_players = [p for p, mask in touch_cols.items() if mask[i]]

        if touched_players:
            # New owners are whoever touched this frame
            current_owners = set(touched_players)

        # Mark possession for current frame based on current owners
        for owner in current_owners:
            simple_possession[i, player_names.index(owner)] = 1

    # Fill player-level columns
    for idx, player in enumerate(player_names):
        game[f"{player}_simple_possession"] = simple_possession[:, idx]

    # Team-level and contested fields
    blue_players = teams["Blue"]
    orange_players = teams["Orange"]

    game["blue_simple_possession"] = (
        game[[f"{p}_simple_possession" for p in blue_players]].any(axis=1).astype(int)
    )
    game["orange_simple_possession"] = (
        game[[f"{p}_simple_possession" for p in orange_players]].any(axis=1).astype(int)
    )

    # Contested = multiple players in possession this frame
    game["contested_simple_possession"] = (simple_possession.sum(axis=1) > 1).astype(
        int
    )

    return game


def assign_possession(game: pd.DataFrame, teams: Dict[str, List[str]]) -> pd.DataFrame:
    game = assign_simple_possession(game, teams)
    game = assign_complex_possession(game, teams)
    return game
