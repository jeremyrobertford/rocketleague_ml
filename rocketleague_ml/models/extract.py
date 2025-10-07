import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union
from rocketleague_ml.core.game import Game
from rocketleague_ml.core.player import Player
from rocketleague_ml.config import (
    Z_GROUND,
    Z_CEILING,
    X_WALL,
    Y_WALL,
    X_GOAL,
    GOAL_DEPTH,
    TOL,
    FIELD_Y,
)


def extract_base_features(
    game: Game,
    round: int,
    main_player: Player,
    config: Dict[str, Any],
):
    if not game.ball:
        raise ValueError(f"Game does not contain ball {game.id}")

    # --- base ---
    dense = pd.DataFrame(
        {
            "time": [frame["time"] for frame in game.frames],
            "delta": [frame["delta"] for frame in game.frames],
            "match_time": [frame["match_time"] for frame in game.frames],
            "active": [frame["active"] or False for frame in game.frames],
        }
    )
    dense.set_index("time", drop=False, inplace=True)

    # --- ball ---
    ball_ser = pd.DataFrame(
        {
            u["time"]: (
                (-1 if config["flip_x"] else 1) * u["location"]["x"],
                (-1 if config["flip_y"] else 1) * u["location"]["y"],
                u["location"]["z"],
            )
            for u in game.ball.updates_by_round[round]
        }
    ).T.rename(columns={0: "ball_x", 1: "ball_y", 2: "ball_z"})

    # reindex onto master frame list
    ball_ser = ball_ser.reindex(dense.index, method="ffill")
    for col in ball_ser.columns:
        dense[col] = ball_ser[col]

    # --- players ---
    for player in game.players.values():
        if not player.car:
            raise ValueError(
                f"No player car for player {player.name} in game {game.id}"
            )
        prefix = player.name + "_"
        if player.name == main_player.name:
            prefix = ""
        player_ser = pd.DataFrame(
            {
                u["time"]: (
                    (-1 if config["flip_x"] else 1) * u["location"]["x"],
                    (-1 if config["flip_y"] else 1) * u["location"]["y"],
                    u["location"]["z"],
                    (-1 if config["flip_y"] else 1) * u["rotation"]["x"],
                    (-1 if config["flip_y"] else 1) * u["rotation"]["y"],
                    (-1 if config["flip_y"] else 1) * u["rotation"]["z"],
                    u["rotation"]["w"],
                )
                for u in player.car.updates_by_round[round]
            }
        ).T.rename(
            columns={
                0: f"{prefix}x",
                1: f"{prefix}y",
                2: f"{prefix}z",
                3: f"{prefix}rotation_x",
                4: f"{prefix}rotation_y",
                5: f"{prefix}rotation_z",
                6: f"{prefix}rotation_w",
            }
        )

        if round == 4:
            pass

        player_ser = player_ser.reindex(dense.index, method="ffill")
        for col in player_ser.columns:
            dense[col] = player_ser[col]

    # --- drop rows where all position columns are NaN ---
    pos_cols = [c for c in dense.columns if c != "time"]

    # mask for rows that have at least one position
    mask_pos = dense[pos_cols].notna().any(axis=1)

    # mask for rows that are active
    mask_active = dense["active"] == True

    # combined mask
    mask = mask_pos & mask_active

    # find the first and last "valid" indices
    valid_index = mask[mask].index  # Index of rows where mask is True
    first_valid = valid_index[0]
    last_valid = valid_index[-1]

    # slice the DataFrame to keep only that range
    dense = dense.loc[first_valid:last_valid].reset_index(drop=True)
    dense.set_index("time", drop=False, inplace=True)

    # now you can ffill/bfill safely
    dense = dense.ffill().bfill()

    # --- distance to other players ---
    for player in game.players.values():
        if player.name == main_player.name:
            continue
        dense[f"distance_to_{player.name}"] = np.sqrt(
            (dense["x"] - dense[f"{player.name}_x"]) ** 2
            + (dense["y"] - dense[f"{player.name}_y"]) ** 2
            + (dense["z"] - dense[f"{player.name}_z"]) ** 2
        )

    return dense


def extract_distance_features_from_base(
    base: pd.DataFrame, game: Game, main_player: Player, config: Dict[str, Any]
):
    base["distance_to_ball"] = np.sqrt(
        (base["x"] - base["ball_x"]) ** 2
        + (base["y"] - base["ball_y"]) ** 2
        + (base["z"] - base["ball_z"]) ** 2
    )

    for player in game.players.values():
        if player.name == main_player.name:
            continue
        base[f"{player.name}_distance_to_ball"] = np.sqrt(
            (base[f"{player.name}_x"] - base["ball_x"]) ** 2
            + (base[f"{player.name}_y"] - base["ball_y"]) ** 2
            + (base[f"{player.name}_z"] - base["ball_z"]) ** 2
        )

    return base


def extract_field_pos_features(
    features: pd.DataFrame, game: Game, main_player: Player, config: Dict[str, Any]
):
    max_y = abs(FIELD_Y[0])
    third_of_y = max_y / 3
    features["offensive_half"] = features["y"] > 0
    features["defensive_half"] = features["y"] <= 0
    features["offensive_third"] = features["y"] >= third_of_y
    features["neutral_third"] = (features["y"] < third_of_y) & (
        features["y"] > -third_of_y
    )
    features["defensive_third"] = features["y"] <= -third_of_y

    features["left_half"] = features["x"] > 0
    features["right_half"] = features["x"] <= 0
    features["left_third"] = features["x"] >= third_of_y
    features["middle_third"] = (features["x"] < third_of_y) & (
        features["x"] > -third_of_y
    )
    features["right_third"] = features["x"] <= -third_of_y

    distance_cols = [
        c
        for c in features.columns
        if c.endswith("_distance_to_ball") or c == "distance_to_ball"
    ]
    features["closest_to_ball"] = features["distance_to_ball"] == features[
        distance_cols
    ].min(axis=1)
    features["farthest_from_ball"] = features["distance_to_ball"] == features[
        distance_cols
    ].max(axis=1)
    features["in_front_of_ball"] = features["y"] > features["ball_y"]
    features["behind_ball"] = features["y"] < features["ball_y"]

    features["grounded"] = features["z"].between(0, Z_GROUND)  # type: ignore
    features["on_ceiling"] = features["z"] >= Z_CEILING

    features["on_left_wall"] = features["x"] <= -X_WALL + TOL
    features["on_right_wall"] = features["x"] >= X_WALL - TOL

    # Goal regions
    features["in_own_goal"] = (
        (features["x"].between(-X_GOAL, X_GOAL))  # type: ignore
        & (features["y"] <= -(Y_WALL - TOL))
        & (features["y"] >= -(Y_WALL + GOAL_DEPTH + TOL))
    )
    features["in_opponents_goal"] = (
        (features["x"].between(-X_GOAL, X_GOAL))  # type: ignore
        & (features["y"] >= (Y_WALL - TOL))
        & (features["y"] <= (Y_WALL + GOAL_DEPTH + TOL))
    )

    # Wall checks, excluding goal occupancy
    features["on_back_wall"] = (features["y"] <= -(Y_WALL - TOL)) & (
        ~features["in_own_goal"]
    )
    features["on_front_wall"] = (features["y"] >= (Y_WALL - TOL)) & (
        ~features["in_opponents_goal"]
    )

    # airborne = none of the above
    features["airborne"] = ~(
        features["grounded"]
        | features["on_ceiling"]
        | features["on_left_wall"]
        | features["on_right_wall"]
        | features["on_back_wall"]
        | features["on_front_wall"]
    )

    return features


def get_position_stats_for(cols: Union[str, List[str]], features: pd.DataFrame):
    total_time = features["delta"].sum()

    # build mask
    if isinstance(cols, str):
        mask = features[cols]
    else:
        mask = features[cols].all(axis=1)  # AND across provided columns

    # percentage time
    time_in_position = features.loc[mask, "delta"].sum()
    perc_in_position = time_in_position / total_time if total_time > 0 else 0

    # find runs/stints
    run_ids = mask.ne(mask.shift()).cumsum()
    stint_groups = features[mask].groupby(run_ids)  # type: ignore

    # duration per stint
    stint_durations = stint_groups["delta"].sum()

    # average stint duration
    avg_stint = stint_durations.mean() if not stint_durations.empty else 0

    return perc_in_position, avg_stint


def extract_velocity_features(
    features: pd.DataFrame,
    game: Game,
    round: int,
    main_player: Player,
    config: Dict[str, Any],
):
    if not game.ball:
        raise ValueError(f"Game does not contain ball {game.id}")

    # --- ball ---
    ball_ser = pd.DataFrame(
        {
            u["time"]: (
                (
                    (-1 if config["flip_x"] else 1) * u["linear_velocity"]["x"]
                    if u["linear_velocity"]
                    else 0
                ),
                (
                    (-1 if config["flip_y"] else 1) * u["linear_velocity"]["y"]
                    if u["linear_velocity"]
                    else 0
                ),
                (u["linear_velocity"]["z"] if u["linear_velocity"] else 0),
                (
                    (-1 if config["flip_x"] else 1) * u["angular_velocity"]["x"]
                    if u["angular_velocity"]
                    else 0
                ),
                (
                    (-1 if config["flip_y"] else 1) * u["angular_velocity"]["y"]
                    if u["angular_velocity"]
                    else 0
                ),
                (u["angular_velocity"]["z"] if u["angular_velocity"] else 0),
            )
            for u in game.ball.updates_by_round[round]
        }
    ).T.rename(
        columns={
            0: "ball_linear_velocity_x",
            1: "ball_linear_velocity_y",
            2: "ball_linear_velocity_z",
            3: "ball_angular_velocity_x",
            4: "ball_angular_velocity_y",
            5: "ball_angular_velocity_z",
        }
    )

    # reindex onto master frame list
    ball_ser = ball_ser.reindex(features.index, method="ffill")
    for col in ball_ser.columns:
        features[col] = ball_ser[col]

    # --- players ---
    for player in game.players.values():
        if not player.car:
            raise ValueError(
                f"No player car for player {player.name} in game {game.id}"
            )
        prefix = player.name + "_"
        if player.name == main_player.name:
            prefix = ""
        player_ser = pd.DataFrame(
            {
                u["time"]: (
                    (
                        (-1 if config["flip_x"] else 1) * u["linear_velocity"]["x"]
                        if u["linear_velocity"]
                        else 0
                    ),
                    (
                        (-1 if config["flip_y"] else 1) * u["linear_velocity"]["y"]
                        if u["linear_velocity"]
                        else 0
                    ),
                    (u["linear_velocity"]["z"] if u["linear_velocity"] else 0),
                    (
                        (-1 if config["flip_x"] else 1) * u["angular_velocity"]["x"]
                        if u["angular_velocity"]
                        else 0
                    ),
                    (
                        (-1 if config["flip_y"] else 1) * u["angular_velocity"]["y"]
                        if u["angular_velocity"]
                        else 0
                    ),
                    (u["angular_velocity"]["z"] if u["angular_velocity"] else 0),
                )
                for u in player.car.updates_by_round[round]
            }
        ).T.rename(
            columns={
                0: f"{prefix}linear_velocity_x",
                1: f"{prefix}linear_velocity_y",
                2: f"{prefix}linear_velocity_z",
                3: f"{prefix}angular_velocity_x",
                4: f"{prefix}angular_velocity_y",
                5: f"{prefix}angular_velocity_z",
            }
        )

        player_ser = player_ser.reindex(features.index, method="ffill")
        for col in player_ser.columns:
            features[col] = player_ser[col]

    return features


def extract_speed_features(
    features: pd.DataFrame, game: Game, main_player: Player, config: Dict[str, Any]
):
    features["linear_velocity"] = np.sqrt(
        (features["linear_velocity_x"]) ** 2
        + (features["linear_velocity_y"]) ** 2
        + (features["linear_velocity_z"]) ** 2
    )

    features["angular_velocity"] = np.sqrt(
        (features["angular_velocity_x"]) ** 2
        + (features["angular_velocity_y"]) ** 2
        + (features["angular_velocity_z"]) ** 2
    )

    for player in game.players.values():
        if player.name == main_player.name:
            continue
        features[f"{player.name}_linear_velocity"] = np.sqrt(
            (features[f"{player.name}_linear_velocity_x"]) ** 2
            + (features[f"{player.name}_linear_velocity_y"]) ** 2
            + (features[f"{player.name}_linear_velocity_z"]) ** 2
        )

        features[f"{player.name}_angular_velocity"] = np.sqrt(
            (features[f"{player.name}_angular_velocity_x"]) ** 2
            + (features[f"{player.name}_angular_velocity_y"]) ** 2
            + (features[f"{player.name}_angular_velocity_z"]) ** 2
        )

    features["is_still"] = features["linear_velocity"] <= 10
    features["is_slow"] = features["linear_velocity"] <= 500
    features["is_semi_slow"] = (features["linear_velocity"] > 500) & (
        features["linear_velocity"] <= 1000
    )
    features["is_medium_speed"] = (features["linear_velocity"] > 1000) & (
        features["linear_velocity"] <= 1500
    )
    features["is_semi_fast"] = (features["linear_velocity"] > 1500) & (
        features["linear_velocity"] <= 2000
    )
    features["is_fast"] = features["linear_velocity"] > 2000

    features["is_drive_speed"] = features["linear_velocity"] <= 1410
    features["is_boost_speed"] = (features["linear_velocity"] > 1410) & (
        features["linear_velocity"] < 2200
    )
    features["is_supersonic"] = features["linear_velocity"] >= 2200

    return features


def extract_features_from_game_for_player(game: Game, player_name: str):
    main_player = None
    teammates: List[str] = []
    opponents: List[str] = []
    for player in game.players.values():
        if player.name == player_name:
            main_player = player
            break

    if not main_player:
        raise ValueError(f"Could not find player {player_name}")

    for player in game.players.values():
        if player.name == player_name:
            continue
        if player.team == main_player.team:
            teammates.append(player.name)
        else:
            opponents.append(player.name)

    flip_x = True if main_player and main_player.team == "Blue" else False
    flip_y = True if main_player and main_player.team == "Orange" else False
    config = {
        "flip_x": flip_x,
        "flip_y": flip_y,
    }

    all_round_summaries: List[Dict[str, float]] = []

    for r in range(len(game.rounds) - 1):
        round = r + 1
        base_features = extract_base_features(game, round, main_player, config)
        features = extract_distance_features_from_base(
            base_features, game, main_player, config
        )
        features = extract_field_pos_features(features, game, main_player, config)
        features = extract_velocity_features(features, game, round, main_player, config)
        features = extract_speed_features(features, game, main_player, config)

        round_summary: Dict[str, float] = {}

        avg_speed = features["linear_velocity"].mean()
        round_summary["Average Speed"] = avg_speed

        speed_metric_columns = {
            "Stationary": ["is_still"],
            "Slow": ["is_slow"],
            "Semi-Slow": ["is_semi_slow"],
            "Medium Speed": ["is_slow"],
            "Semi-Fast": ["is_semi_fast"],
            "Drive Speed": ["is_drive_speed"],
            "Boost Speed": ["is_supersonic"],
            "Supersonic": ["is_supersonic"],
        }
        for label, cols in speed_metric_columns.items():
            perc, avg_stint = get_position_stats_for(cols, features)
            round_summary[f"Percent Time while {label}"] = perc
            round_summary[f"Average Stint while {label}"] = avg_stint

        avg_angular_speed = features["angular_velocity"].mean()
        round_summary["Average Angular Speed"] = avg_angular_speed

        avg_distance_to_ball = features["distance_to_ball"].mean()
        round_summary["Average Distance to Ball"] = avg_distance_to_ball
        teammate_distances = [f"distance_to_{teammate}" for teammate in teammates]
        avg_distance_to_teammates = features[teammate_distances].mean(axis=1).mean()
        round_summary["Average Distance to Teammates"] = avg_distance_to_teammates
        opponent_distances = [f"distance_to_{opponent}" for opponent in opponents]
        avg_distance_to_opponents = features[opponent_distances].mean(axis=1).mean()
        round_summary["Average Distance to Opponents"] = avg_distance_to_opponents

        metric_columns = {
            # Halves
            "Offensive Half": ["offensive_half"],
            "Defensive Half": ["defensive_half"],
            "Left Half": ["left_half"],
            "Right Half": ["right_half"],
            # Thirds (longitudinal)
            "Offensive Third": ["offensive_third"],
            "Neutral Third": ["neutral_third"],
            "Defensive Third": ["defensive_third"],
            # Thirds (lateral)
            "Left Third": ["left_third"],
            "Middle Third": ["middle_third"],
            "Right Third": ["right_third"],
            # Half intersections
            "Offensive Left Half": ["offensive_half", "left_half"],
            "Offensive Right Half": ["offensive_half", "right_half"],
            "Defensive Left Half": ["defensive_half", "left_half"],
            "Defensive Right Half": ["defensive_half", "right_half"],
            # Third intersections
            "Offensive Left Third": ["offensive_third", "left_third"],
            "Offensive Middle Third": ["offensive_third", "middle_third"],
            "Offensive Right Third": ["offensive_third", "right_third"],
            "Neutral Left Third": ["neutral_third", "left_third"],
            "Neutral Middle Third": ["neutral_third", "middle_third"],
            "Neutral Right Third": ["neutral_third", "right_third"],
            "Defensive Left Third": ["defensive_third", "left_third"],
            "Defensive Middle Third": ["defensive_third", "middle_third"],
            "Defensive Right Third": ["defensive_third", "right_third"],
            # Ball-relative positioning
            "Closest to Ball": ["closest_to_ball"],
            "Farthest from Ball": ["farthest_from_ball"],
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

        for label, cols in metric_columns.items():
            perc, avg_stint = get_position_stats_for(cols, features)
            round_summary[f"Percent Time in {label}"] = perc
            round_summary[f"Average Stint in {label}"] = avg_stint

        all_round_summaries.append(round_summary)

    return all_round_summaries
