import os
import csv
import json
from pathlib import Path
from typing import cast, Any, Dict, List, Union
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from rocketleague_ml.models.frame_by_frame_processor import Frame_By_Frame_Processor
from rocketleague_ml.utils.logging import Logger
from rocketleague_ml.config import (
    DTYPES,
    # FEATURE_LABELS,
    FEATURES,
    PROCESSED,
    FIELD_Y,
    Z_GROUND,
    Z_CEILING,
    X_WALL,
    Y_WALL,
    X_GOAL,
    GOAL_DEPTH,
    TOL,
    TOTAL_FIELD_DISTANCE,
)


class Rocket_League_Feature_Extractor:
    def __init__(self, processor: Frame_By_Frame_Processor, logger: Logger = Logger()):
        self.processor = processor
        self.logger = logger

    def get_player_names_and_teams(self, game: pd.DataFrame):
        cols = game.columns
        loc_cols = [c for c in cols if c.endswith("_positioning_x")]
        player_cols = [c for c in loc_cols if not c.startswith("ball_")]
        player_names = [c.replace("_positioning_x", "") for c in player_cols]
        teams: Dict[str, List[str]] = {"Blue": [], "Orange": []}
        for player_name in player_names:
            team = game[f"{player_name}_team"][
                game[f"{player_name}_team"].notna()
                & (game[f"{player_name}_team"] != "")
            ].iloc[0]
            teams[team].append(player_name)
        return player_names, teams

    def extract_distances(
        self,
        game: pd.DataFrame,
        player_names: List[str],
        main_player: str,
    ):
        distance_cols: Dict[str, pd.DataFrame | NDArray[Any]] = {}
        for player_name in player_names:
            distance_cols[f"{player_name}_distance_to_ball"] = np.sqrt(
                (game[f"{player_name}_positioning_x"] - game["ball_positioning_x"]) ** 2
                + (game[f"{player_name}_positioning_y"] - game["ball_positioning_y"])
                ** 2
                + (game[f"{player_name}_positioning_z"] - game["ball_positioning_z"])
                ** 2
            )

        for player_name in player_names:
            if main_player == player_name:
                continue
            distance_cols[f"{main_player}_distance_to_{player_name}"] = np.sqrt(
                (
                    game[f"{main_player}_positioning_x"]
                    - game[f"{player_name}_positioning_x"]
                )
                ** 2
                + (
                    game[f"{main_player}_positioning_y"]
                    - game[f"{player_name}_positioning_y"]
                )
                ** 2
                + (
                    game[f"{main_player}_positioning_z"]
                    - game[f"{player_name}_positioning_z"]
                )
                ** 2
            )

        return distance_cols

    def extract_field_positioning(
        self, game: pd.DataFrame, player_names: List[str], main_player: str
    ):
        field_positioning_cols: Dict[str, pd.DataFrame | pd.Series] = {}
        max_y = abs(FIELD_Y[0])
        third_of_y = max_y / 3
        half_of_z = Z_CEILING / 2
        third_of_z = Z_CEILING / 3

        field_positioning_cols[f"{main_player}_highest_half"] = (
            game[f"{main_player}_positioning_z"] >= half_of_z
        )
        field_positioning_cols[f"{main_player}_lowest_half"] = (
            game[f"{main_player}_positioning_z"] < half_of_z
        )
        field_positioning_cols[f"{main_player}_highest_third"] = (
            game[f"{main_player}_positioning_z"] >= third_of_z * 2
        )
        field_positioning_cols[f"{main_player}_middle_aerial_third"] = (
            game[f"{main_player}_positioning_z"] > third_of_z
        ) & (game[f"{main_player}_positioning_z"] < third_of_z * 2)
        field_positioning_cols[f"{main_player}_lowest_third"] = (
            game[f"{main_player}_positioning_z"] <= third_of_z
        )

        field_positioning_cols[f"{main_player}_offensive_half"] = (
            game[f"{main_player}_positioning_y"] > 0
        )
        field_positioning_cols[f"{main_player}_defensive_half"] = (
            game[f"{main_player}_positioning_y"] <= 0
        )
        field_positioning_cols[f"{main_player}_offensive_third"] = (
            game[f"{main_player}_positioning_y"] >= third_of_y
        )
        field_positioning_cols[f"{main_player}_neutral_third"] = (
            game[f"{main_player}_positioning_y"] < third_of_y
        ) & (game[f"{main_player}_positioning_y"] > -third_of_y)
        field_positioning_cols[f"{main_player}_defensive_third"] = (
            game[f"{main_player}_positioning_y"] <= -third_of_y
        )

        field_positioning_cols[f"{main_player}_left_half"] = (
            game[f"{main_player}_positioning_x"] > 0
        )
        field_positioning_cols[f"{main_player}_right_half"] = (
            game[f"{main_player}_positioning_x"] <= 0
        )
        field_positioning_cols[f"{main_player}_left_third"] = (
            game[f"{main_player}_positioning_x"] >= third_of_y
        )
        field_positioning_cols[f"{main_player}_middle_third"] = (
            game[f"{main_player}_positioning_x"] < third_of_y
        ) & (game[f"{main_player}_positioning_x"] > -third_of_y)
        field_positioning_cols[f"{main_player}_right_third"] = (
            game[f"{main_player}_positioning_x"] <= -third_of_y
        )

        field_positioning_cols[f"{main_player}_grounded"] = game[f"{main_player}_positioning_z"].between(0, Z_GROUND)  # type: ignore
        field_positioning_cols[f"{main_player}_on_ceiling"] = (
            game[f"{main_player}_positioning_z"] >= Z_CEILING
        )

        field_positioning_cols[f"{main_player}_on_left_wall"] = (
            game[f"{main_player}_positioning_x"] <= -X_WALL + TOL
        )
        field_positioning_cols[f"{main_player}_on_right_wall"] = (
            game[f"{main_player}_positioning_x"] >= X_WALL - TOL
        )

        # Ball
        field_positioning_cols[f"{main_player}_in_front_of_ball"] = (
            game[f"{main_player}_positioning_y"] > game["ball_positioning_y"]
        )
        field_positioning_cols[f"{main_player}_behind_ball"] = (
            game[f"{main_player}_positioning_y"] < game["ball_positioning_y"]
        )

        # Goal regions
        field_positioning_cols[f"{main_player}_in_own_goal"] = (
            (game[f"{main_player}_positioning_x"].between(-X_GOAL, X_GOAL))  # type: ignore
            & (game[f"{main_player}_positioning_y"] <= -(Y_WALL - TOL))
            & (game[f"{main_player}_positioning_y"] >= -(Y_WALL + GOAL_DEPTH + TOL))
        )
        field_positioning_cols[f"{main_player}_in_opponents_goal"] = (
            (game[f"{main_player}_positioning_x"].between(-X_GOAL, X_GOAL))  # type: ignore
            & (game[f"{main_player}_positioning_y"] >= (Y_WALL - TOL))
            & (game[f"{main_player}_positioning_y"] <= (Y_WALL + GOAL_DEPTH + TOL))
        )

        return field_positioning_cols

    def extract_speeds(
        self,
        game: pd.DataFrame,
        player_names: List[str],
        main_player: str | None = None,
    ) -> Dict[str, pd.DataFrame | pd.Series | NDArray[np.float64]]:
        speed_cols: Dict[str, pd.DataFrame | pd.Series | NDArray[Any]] = {}
        for player_name in player_names:
            if main_player and player_name != main_player:
                continue
            speed_cols[f"{player_name}_positioning_linear_velocity"] = np.sqrt(
                (game[f"{player_name}_positioning_linear_velocity_x"]) ** 2
                + (game[f"{player_name}_positioning_linear_velocity_y"]) ** 2
                + (game[f"{player_name}_positioning_linear_velocity_z"]) ** 2
            )

            speed_cols[f"{player_name}_positioning_angular_velocity"] = np.sqrt(
                (game[f"{player_name}_positioning_angular_velocity_x"]) ** 2
                + (game[f"{player_name}_positioning_angular_velocity_y"]) ** 2
                + (game[f"{player_name}_positioning_angular_velocity_z"]) ** 2
            )

        return speed_cols

    def get_position_stats_for(
        self, game: pd.DataFrame, cols: Union[str, List[str]], main_player: str
    ):
        total_time = game["delta"].sum()

        # build mask
        if isinstance(cols, str):
            mask = game[main_player + "_" + cols]
        else:
            cols = [main_player + "_" + col for col in cols]
            mask = game[cols].all(axis=1)  # AND across provided columns

        # percentage time
        time_in_position = game.loc[mask, "delta"].sum()
        perc_in_position = time_in_position / total_time if total_time > 0 else 0

        # find runs/stints
        run_ids = mask.ne(mask.shift()).cumsum()
        stint_groups = game[mask].groupby(run_ids)  # type: ignore

        # duration per stint
        stint_durations = stint_groups["delta"].sum()

        # average stint duration
        avg_stint = stint_durations.mean() if not stint_durations.empty else 0
        perc_avg_stint = avg_stint / total_time if total_time > 0 else 0

        return perc_in_position, perc_avg_stint

    def extract_touch_locations(
        self,
        game: pd.DataFrame,
        player_names: List[str],
        teams: Dict[str, List[str]],
        main_player: str,
    ) -> Dict[str, pd.Series]:
        toward_goal_tol = 0.7
        toward_teammate_tol = 0.6
        toward_opponent_tol = 0.6
        toward_open_space_tol: float | None = None
        result: Dict[str, pd.Series] = {}

        # Define goal directions (Rocket League convention)
        GOAL_BLUE = np.array([0, -5120, 0])
        GOAL_ORANGE = np.array([0, 5120, 0])

        # Precompute player positions
        positions = {
            p: game[
                [
                    f"{p}_positioning_x",
                    f"{p}_positioning_y",
                    f"{p}_positioning_z",
                ]
            ].to_numpy()
            for p in player_names
        }

        # Ball position
        ball_pos = game[
            [
                "ball_positioning_x",
                "ball_positioning_y",
                "ball_positioning_z",
            ]
        ].to_numpy()

        player = main_player
        prefix = f"{player}_hit_ball_"

        try:
            impulses = game[
                [prefix + "impulse_x", prefix + "impulse_y", prefix + "impulse_z"]
            ].to_numpy()
            confidence = game[prefix + "collision_confidence"].fillna(0).to_numpy()  # type: ignore
        except KeyError:
            return {}

        hit_dir = np.nan_to_num(
            impulses / (np.linalg.norm(impulses, axis=1, keepdims=True) + 1e-6)
        )

        # Determine which team player belongs to
        team = next((t for t, players in teams.items() if player in players), None)
        if not team:
            raise ValueError(f"No team found for player {player} in {teams}")

        # Compute directional vectors
        goal_target = GOAL_ORANGE if team.lower() == "blue" else GOAL_BLUE
        goal_vec = goal_target - ball_pos
        goal_vec /= np.linalg.norm(goal_vec, axis=1, keepdims=True) + 1e-6
        to_goal = np.sum(hit_dir * goal_vec, axis=1)

        # Teammate direction
        teammate_positions = [positions[p] for p in teams[team] if p != player]
        if teammate_positions:
            team_center = np.mean(np.stack(teammate_positions), axis=0)
            team_vec = team_center - ball_pos
            team_vec /= np.linalg.norm(team_vec, axis=1, keepdims=True) + 1e-6
            to_teammate = np.sum(hit_dir * team_vec, axis=1)
        else:
            to_teammate = np.zeros(len(game))

        # Opponent direction
        opponent_team = [t for t in teams.keys() if t.lower() != team.lower()][0]
        opponent_positions = [positions[p] for p in teams[opponent_team] if p != player]
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
                ~game[f"{p}_hit_ball_collision_confidence"].isna().to_numpy()
                for p in player_names
            ],
            axis=0,
        )
        fifty_fifty_mask = collision_count > 1

        # Create independent boolean masks for each touch type
        categories: Dict[str, Any] = {
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
        result[f"{player}_touch_total"] = pd.Series(
            touch_mask.astype(int), index=game.index
        )

        # Add direction dot products for analysis
        result[f"{player}_touch_to_goal_dot"] = pd.Series(to_goal, index=game.index)
        result[f"{player}_touch_to_teammate_dot"] = pd.Series(
            to_teammate, index=game.index
        )
        result[f"{player}_touch_to_opponent_dot"] = pd.Series(
            to_opponent, index=game.index
        )
        result[f"{player}_touch_open_space_score"] = pd.Series(
            open_space_score, index=game.index
        )

        # Add independent category flags
        for cat, mask in categories.items():
            result[f"{player}_touch_{cat}"] = pd.Series(
                (mask & touch_mask).astype(int), index=game.index
            )

        return result

    def add_features_to_result(
        self,
        features: Dict[str, Any],
        main_player: str,
        player_names: List[str],
        game: pd.DataFrame,
        column_label: str,
        filter_label: str,
        teams: Dict[str, List[str]],
        main_player_team: str,
        opponents_team: str,
    ):
        distance_cols = {
            column_label + key: value
            for key, value in self.extract_distances(
                game=game, player_names=player_names, main_player=main_player
            ).items()
        }
        field_positioning_cols = {
            column_label + key: value
            for key, value in self.extract_field_positioning(
                game=game, player_names=player_names, main_player=main_player
            ).items()
        }
        speed_cols = {
            column_label + key: value
            for key, value in self.extract_speeds(
                game=game, player_names=player_names, main_player=main_player
            ).items()
        }
        touch_location_cols = {
            column_label + key: value
            for key, value in self.extract_touch_locations(
                game=game,
                player_names=player_names,
                main_player=main_player,
                teams=teams,
            ).items()
        }

        game = pd.concat(
            [
                game,
                pd.DataFrame(
                    distance_cols
                    | field_positioning_cols
                    | speed_cols
                    | touch_location_cols,
                    index=game.index,
                ),
            ],
            axis=1,
        )

        ball_distance_cols = [
            c for c in distance_cols.keys() if c.endswith("_distance_to_ball")
        ]
        dependent_cols: Dict[str, pd.DataFrame | pd.Series] = {}
        dependent_cols[f"{column_label}{main_player}_closest_to_ball"] = game[
            f"{column_label}{main_player}_distance_to_ball"
        ] == game[ball_distance_cols].min(axis=1)
        dependent_cols[f"{column_label}{main_player}_farthest_from_ball"] = game[
            f"{column_label}{main_player}_distance_to_ball"
        ] == game[ball_distance_cols].max(axis=1)
        # Wall checks, excluding goal occupancy
        dependent_cols[f"{column_label}{main_player}_on_back_wall"] = (
            game[f"{column_label}{main_player}_positioning_y"] <= -(Y_WALL - TOL)
        ) & (~game[f"{column_label}{main_player}_in_own_goal"])
        dependent_cols[f"{column_label}{main_player}_on_front_wall"] = (
            game[f"{column_label}{main_player}_positioning_y"] >= (Y_WALL - TOL)
        ) & (~game[f"{column_label}{main_player}_in_opponents_goal"])

        dependent_cols[f"{column_label}{main_player}_is_still"] = (
            game[f"{column_label}{main_player}_positioning_linear_velocity"] <= 10
        )
        dependent_cols[f"{column_label}{main_player}_is_slow"] = (
            game[f"{column_label}{main_player}_positioning_linear_velocity"] <= 500
        )
        dependent_cols[f"{column_label}{main_player}_is_semi_slow"] = (
            game[f"{column_label}{main_player}_positioning_linear_velocity"] > 500
        ) & (game[f"{column_label}{main_player}_positioning_linear_velocity"] <= 1000)
        dependent_cols[f"{column_label}{main_player}_is_medium_speed"] = (
            game[f"{column_label}{main_player}_positioning_linear_velocity"] > 1000
        ) & (game[f"{column_label}{main_player}_positioning_linear_velocity"] <= 1500)
        dependent_cols[f"{column_label}{main_player}_is_semi_fast"] = (
            game[f"{column_label}{main_player}_positioning_linear_velocity"] > 1500
        ) & (game[f"{column_label}{main_player}_positioning_linear_velocity"] <= 2000)
        dependent_cols[f"{column_label}{main_player}_is_fast"] = (
            game[f"{column_label}{main_player}_positioning_linear_velocity"] > 2000
        )

        dependent_cols[f"{column_label}{main_player}_is_drive_speed"] = (
            game[f"{column_label}{main_player}_positioning_linear_velocity"] <= 1410
        )
        dependent_cols[f"{column_label}{main_player}_is_boost_speed"] = (
            game[f"{column_label}{main_player}_positioning_linear_velocity"] > 1410
        ) & (game[f"{column_label}{main_player}_positioning_linear_velocity"] < 2200)
        dependent_cols[f"{column_label}{main_player}_is_supersonic"] = (
            game[f"{column_label}{main_player}_positioning_linear_velocity"] >= 2200
        )

        game = pd.concat(
            [game, pd.DataFrame(dependent_cols, index=game.index)],
            axis=1,
        )
        game.loc[:, f"{column_label}{main_player}_airborne"] = ~(
            game[f"{column_label}{main_player}_grounded"]
            | game[f"{column_label}{main_player}_on_ceiling"]
            | game[f"{column_label}{main_player}_on_left_wall"]
            | game[f"{column_label}{main_player}_on_right_wall"]
            | game[f"{column_label}{main_player}_on_back_wall"]
            | game[f"{column_label}{main_player}_on_front_wall"]
        )

        total_touches = game[f"{column_label}{main_player}_touch_total"].sum()
        features[f"{filter_label}Fifty-Fifty Touch Percentage"] = (
            game[f"{column_label}{main_player}_touch_fifty_fifty"].sum() / total_touches
        )
        features[f"{filter_label}Towards Goal Touch Percentage"] = (
            game[f"{column_label}{main_player}_touch_towards_goal"].sum()
            / total_touches
        )
        features[f"{filter_label}Towards Teammate Touch Percentage"] = (
            game[f"{column_label}{main_player}_touch_towards_teammate"].sum()
            / total_touches
        )
        features[f"{filter_label}Towards Opponent Touch Percentage"] = (
            game[f"{column_label}{main_player}_touch_towards_opponent"].sum()
            / total_touches
        )
        features[f"{filter_label}Towards Open Space Touch Percentage"] = (
            game[f"{column_label}{main_player}_touch_towards_open_space"].sum()
            / total_touches
        )

        movement_columns: Dict[str, List[str]] = {
            # f"{filter_label}Drifting": [f"{column_label}drift_active"]
        }
        for label, cols in movement_columns.items():
            perc, avg_stint = self.get_position_stats_for(game, cols, main_player)
            features[f"{filter_label}Percent Time while {label}"] = perc
            features[f"{filter_label}Average Stint while {label}"] = avg_stint

        distance_columns: Dict[str, List[str]] = {
            f"{filter_label}Closest to Ball": [f"{column_label}closest_to_ball"],
            f"{filter_label}Farthest from Ball": [f"{column_label}farthest_from_ball"],
        }
        for label, cols in distance_columns.items():
            perc, avg_stint = self.get_position_stats_for(game, cols, main_player)
            features[f"{filter_label}Percent Time while {label}"] = perc
            features[f"{filter_label}Average Stint while {label}"] = avg_stint
        avg_distance_to_ball = game[
            f"{column_label}{main_player}_distance_to_ball"
        ].mean()
        features[f"{filter_label}Average Distance to Ball"] = (
            avg_distance_to_ball / TOTAL_FIELD_DISTANCE
        )
        teammate_distances = [
            f"{column_label}{main_player}_distance_to_{teammate}"
            for teammate in teams[main_player_team]
            if teammate != main_player
        ]
        avg_distance_to_teammates = game[teammate_distances].mean(axis=1).mean()
        features[f"{filter_label}Average Distance to Teammates"] = (
            avg_distance_to_teammates / TOTAL_FIELD_DISTANCE
        )
        opponent_distances = [
            f"{column_label}{main_player}_distance_to_{opponent}"
            for opponent in teams[opponents_team]
        ]
        avg_distance_to_opponents = game[opponent_distances].mean(axis=1).mean()
        features[f"{filter_label}Average Distance to Opponents"] = (
            avg_distance_to_opponents / TOTAL_FIELD_DISTANCE
        )

        positioning_columns = {
            # Halves
            f"{filter_label}In Offensive Half": [f"{column_label}offensive_half"],
            f"{filter_label}In Defensive Half": [f"{column_label}defensive_half"],
            f"{filter_label}In Left Half": [f"{column_label}left_half"],
            f"{filter_label}In Right Half": [f"{column_label}right_half"],
            f"{filter_label}In Highest Half": [f"{column_label}highest_half"],
            f"{filter_label}In Lowest Half": [f"{column_label}lowest_half"],
            # Thirds (longitudinal)
            f"{filter_label}In Offensive Third": [f"{column_label}offensive_third"],
            f"{filter_label}In Neutral Third": [f"{column_label}neutral_third"],
            f"{filter_label}In Defensive Third": [f"{column_label}defensive_third"],
            # Thirds (lateral)
            f"{filter_label}In Left Third": [f"{column_label}left_third"],
            f"{filter_label}In Middle Third": [f"{column_label}middle_third"],
            f"{filter_label}In Right Third": [f"{column_label}right_third"],
            # Third (Vertical)
            f"{filter_label}In Highest Third": [f"{column_label}highest_third"],
            f"{filter_label}In Middle Aerial Third": [
                f"{column_label}middle_aerial_third"
            ],
            f"{filter_label}In Lowest Third": [f"{column_label}lowest_third"],
            # Half intersections
            f"{filter_label}In Offensive Left Half": [
                f"{column_label}offensive_half",
                f"{column_label}left_half",
            ],
            f"{filter_label}In Offensive Right Half": [
                f"{column_label}offensive_half",
                f"{column_label}right_half",
            ],
            f"{filter_label}In Defensive Left Half": [
                f"{column_label}defensive_half",
                f"{column_label}left_half",
            ],
            f"{filter_label}In Defensive Right Half": [
                f"{column_label}defensive_half",
                f"{column_label}right_half",
            ],
            # Half intersections with verticality
            f"{filter_label}In Offensive Left Highest Half": [
                f"{column_label}offensive_half",
                f"{column_label}left_half",
                f"{column_label}highest_half",
            ],
            f"{filter_label}In Offensive Left Lowest Half": [
                f"{column_label}offensive_half",
                f"{column_label}left_half",
                f"{column_label}lowest_half",
            ],
            f"{filter_label}In Offensive Right Highest Half": [
                f"{column_label}offensive_half",
                f"{column_label}right_half",
                f"{column_label}highest_half",
            ],
            f"{filter_label}In Offensive Right Lowest Half": [
                f"{column_label}offensive_half",
                f"{column_label}right_half",
                f"{column_label}lowest_half",
            ],
            f"{filter_label}In Defensive Left Highest Half": [
                f"{column_label}defensive_half",
                f"{column_label}left_half",
                f"{column_label}highest_half",
            ],
            f"{filter_label}In Defensive Left Lowest Half": [
                f"{column_label}defensive_half",
                f"{column_label}left_half",
                f"{column_label}lowest_half",
            ],
            f"{filter_label}In Defensive Right Highest Half": [
                f"{column_label}defensive_half",
                f"{column_label}right_half",
                f"{column_label}highest_half",
            ],
            f"{filter_label}In Defensive Right Lowest Half": [
                f"{column_label}defensive_half",
                f"{column_label}right_half",
                f"{column_label}lowest_half",
            ],
            # Third intersections
            f"{filter_label}In Offensive Left Third": [
                f"{column_label}offensive_third",
                f"{column_label}left_third",
            ],
            f"{filter_label}In Offensive Middle Third": [
                f"{column_label}offensive_third",
                f"{column_label}middle_third",
            ],
            f"{filter_label}In Offensive Right Third": [
                f"{column_label}offensive_third",
                f"{column_label}right_third",
            ],
            f"{filter_label}In Neutral Left Third": [
                f"{column_label}neutral_third",
                f"{column_label}left_third",
            ],
            f"{filter_label}In Neutral Middle Third": [
                f"{column_label}neutral_third",
                f"{column_label}middle_third",
            ],
            f"{filter_label}In Neutral Right Third": [
                f"{column_label}neutral_third",
                f"{column_label}right_third",
            ],
            f"{filter_label}In Defensive Left Third": [
                f"{column_label}defensive_third",
                f"{column_label}left_third",
            ],
            f"{filter_label}In Defensive Middle Third": [
                f"{column_label}defensive_third",
                f"{column_label}middle_third",
            ],
            f"{filter_label}In Defensive Right Third": [
                f"{column_label}defensive_third",
                f"{column_label}right_third",
            ],
            # Third intersections with verticality
            f"{filter_label}In Offensive Left Highest Third": [
                f"{column_label}offensive_third",
                f"{column_label}left_third",
                f"{column_label}highest_third",
            ],
            f"{filter_label}In Offensive Middle Highest Third": [
                f"{column_label}offensive_third",
                f"{column_label}middle_third",
                f"{column_label}highest_third",
            ],
            f"{filter_label}In Offensive Right Highest Third": [
                f"{column_label}offensive_third",
                f"{column_label}right_third",
                f"{column_label}highest_third",
            ],
            f"{filter_label}In Neutral Left Highest Third": [
                f"{column_label}neutral_third",
                f"{column_label}left_third",
                f"{column_label}highest_third",
            ],
            f"{filter_label}In Neutral Middle Highest Third": [
                f"{column_label}neutral_third",
                f"{column_label}middle_third",
                f"{column_label}highest_third",
            ],
            f"{filter_label}In Neutral Right Highest Third": [
                f"{column_label}neutral_third",
                f"{column_label}right_third",
                f"{column_label}highest_third",
            ],
            f"{filter_label}In Defensive Left Highest Third": [
                f"{column_label}defensive_third",
                f"{column_label}left_third",
                f"{column_label}highest_third",
            ],
            f"{filter_label}In Defensive Middle Highest Third": [
                f"{column_label}defensive_third",
                f"{column_label}middle_third",
                f"{column_label}highest_third",
            ],
            f"{filter_label}In Defensive Right Highest Third": [
                f"{column_label}defensive_third",
                f"{column_label}right_third",
                f"{column_label}highest_third",
            ],
            f"{filter_label}In Offensive Left Middle-Aerial Third": [
                f"{column_label}offensive_third",
                f"{column_label}left_third",
                f"{column_label}middle_aerial_third",
            ],
            f"{filter_label}In Offensive Middle Middle-Aerial Third": [
                f"{column_label}offensive_third",
                f"{column_label}middle_third",
                f"{column_label}middle_aerial_third",
            ],
            f"{filter_label}In Offensive Right Middle-Aerial Third": [
                f"{column_label}offensive_third",
                f"{column_label}right_third",
                f"{column_label}middle_aerial_third",
            ],
            f"{filter_label}In Neutral Left Middle-Aerial Third": [
                f"{column_label}neutral_third",
                f"{column_label}left_third",
                f"{column_label}middle_aerial_third",
            ],
            f"{filter_label}In Neutral Middle Middle-Aerial Third": [
                f"{column_label}neutral_third",
                f"{column_label}middle_third",
                f"{column_label}middle_aerial_third",
            ],
            f"{filter_label}In Neutral Right Middle-Aerial Third": [
                f"{column_label}neutral_third",
                f"{column_label}right_third",
                f"{column_label}middle_aerial_third",
            ],
            f"{filter_label}In Defensive Left Middle-Aerial Third": [
                f"{column_label}defensive_third",
                f"{column_label}left_third",
                f"{column_label}middle_aerial_third",
            ],
            f"{filter_label}In Defensive Middle Middle-Aerial Third": [
                f"{column_label}defensive_third",
                f"{column_label}middle_third",
                f"{column_label}middle_aerial_third",
            ],
            f"{filter_label}In Defensive Right Middle-Aerial Third": [
                f"{column_label}defensive_third",
                f"{column_label}right_third",
                f"{column_label}middle_aerial_third",
            ],
            f"{filter_label}In Offensive Left Lowest Third": [
                f"{column_label}offensive_third",
                f"{column_label}left_third",
                f"{column_label}lowest_third",
            ],
            f"{filter_label}In Offensive Middle Lowest Third": [
                f"{column_label}offensive_third",
                f"{column_label}middle_third",
                f"{column_label}lowest_third",
            ],
            f"{filter_label}In Offensive Right Lowest Third": [
                f"{column_label}offensive_third",
                f"{column_label}right_third",
                f"{column_label}lowest_third",
            ],
            f"{filter_label}In Neutral Left Lowest Third": [
                f"{column_label}neutral_third",
                f"{column_label}left_third",
                f"{column_label}lowest_third",
            ],
            f"{filter_label}In Neutral Middle Lowest Third": [
                f"{column_label}neutral_third",
                f"{column_label}middle_third",
                f"{column_label}lowest_third",
            ],
            f"{filter_label}In Neutral Right Lowest Third": [
                f"{column_label}neutral_third",
                f"{column_label}right_third",
                f"{column_label}lowest_third",
            ],
            f"{filter_label}In Defensive Left Lowest Third": [
                f"{column_label}defensive_third",
                f"{column_label}left_third",
                f"{column_label}lowest_third",
            ],
            f"{filter_label}In Defensive Middle Lowest Third": [
                f"{column_label}defensive_third",
                f"{column_label}middle_third",
                f"{column_label}lowest_third",
            ],
            f"{filter_label}In Defensive Right Lowest Third": [
                f"{column_label}defensive_third",
                f"{column_label}right_third",
                f"{column_label}lowest_third",
            ],
            # Ball-relative positioning
            f"{filter_label}In Front of Ball": [f"{column_label}in_front_of_ball"],
            f"{filter_label}Behind Ball": [f"{column_label}behind_ball"],
            # Movement state
            f"{filter_label}Grounded": [f"{column_label}grounded"],
            f"{filter_label}Airborne": [f"{column_label}airborne"],
            # Surfaces
            f"{filter_label}On Ceiling": [f"{column_label}on_ceiling"],
            f"{filter_label}On Left Wall": [f"{column_label}on_left_wall"],
            f"{filter_label}On Right Wall": [f"{column_label}on_right_wall"],
            f"{filter_label}On Back Wall": [f"{column_label}on_back_wall"],
            f"{filter_label}On Front Wall": [f"{column_label}on_front_wall"],
            # Goal zones
            f"{filter_label}In Own Goal": [f"{column_label}in_own_goal"],
            f"{filter_label}In Opponents Goal": [f"{column_label}in_opponents_goal"],
        }
        for label, cols in positioning_columns.items():
            perc, avg_stint = self.get_position_stats_for(game, cols, main_player)
            features[f"{filter_label}Percent Time {label}"] = perc
            features[f"{filter_label}Average Stint {label}"] = avg_stint

        speed_columns = {
            f"{filter_label}Stationary": [f"{column_label}is_still"],
            f"{filter_label}Slow": [f"{column_label}is_slow"],
            f"{filter_label}Semi-Slow": [f"{column_label}is_semi_slow"],
            f"{filter_label}Medium Speed": [f"{column_label}is_slow"],
            f"{filter_label}Semi-Fast": [f"{column_label}is_semi_fast"],
            f"{filter_label}Drive Speed": [f"{column_label}is_drive_speed"],
            f"{filter_label}Boost Speed": [f"{column_label}is_supersonic"],
            f"{filter_label}Supersonic": [f"{column_label}is_supersonic"],
        }
        for label, cols in speed_columns.items():
            perc, avg_stint = self.get_position_stats_for(game, cols, main_player)
            features[f"{filter_label}Percent Time while {label}"] = perc
            features[f"{filter_label}Average Stint while {label}"] = avg_stint

        return features

    def extract_round_features(
        self,
        game: pd.DataFrame,
        main_player: str,
        id: str,
        round: int,
        player_names: List[str],
        teams: Dict[str, List[str]],
        main_player_team: str,
        opponents_team: str,
    ):
        features: Dict[str, Any] = {
            "id": id,
            "round": round,
        }

        self.add_features_to_result(
            column_label="",
            filter_label="",
            features=features,
            main_player=main_player,
            player_names=player_names,
            game=game,
            teams=teams,
            main_player_team=main_player_team,
            opponents_team=opponents_team,
        )

        return features

    def extract_game_features(self, game: pd.DataFrame, main_player: str):
        features: List[Dict[str, Any]] = []
        player_names, teams = self.get_player_names_and_teams(game)
        main_player_team = [
            team for team, players in teams.items() if main_player in players
        ][0]
        opponents_team = [
            team for team, players in teams.items() if main_player not in players
        ][0]

        flip_x = True if not main_player or main_player in teams["Blue"] else False
        flip_y = True if main_player and main_player in teams["Orange"] else False

        game = game[game["active"]]

        if flip_x:
            x_cols = [
                c for c in game.columns if c.endswith("_x") and "rotation" not in c
            ]
            game.loc[:, x_cols] = -game[x_cols]
            # TODO: add flip x for rotation
        if flip_y:
            y_cols = [
                c for c in game.columns if c.endswith("_y") and "rotation" not in c
            ]
            game.loc[:, y_cols] = -game[y_cols]
            # TODO: add flip y for rotation

        id = game["id"].iloc[0]
        rounds: NDArray[np.int_] = game["round"].unique()  # type: ignore
        for round in rounds:
            round = cast(int, round)
            round_features = self.extract_round_features(
                game[game["round"] == round].copy(),
                main_player,
                id,
                round,
                player_names,
                teams,
                main_player_team,
                opponents_team,
            )
            features.append(round_features)
        return features

    def save_features(
        self, features: List[Dict[str, Any]], features_path: str, labels_path: str
    ):
        write_header = not os.path.exists(features_path)

        # --- Handle feature_labels.json ---
        if write_header:
            # Create JSON file with all current feature keys
            # feature_labels = {
            #     feature: FEATURE_LABELS[feature] for feature in list(features[0].keys())
            # }
            feature_labels = [feature for feature in list(features[0].keys())]
            with open(labels_path, "w", encoding="utf-8") as jf:
                json.dump(feature_labels, jf, indent=4)
        else:
            # Load JSON file to filter columns
            if os.path.exists(labels_path):
                with open(labels_path, "r", encoding="utf-8") as jf:
                    feature_labels = json.load(jf)
            else:
                # If JSON missing but CSV exists, fall back to all keys
                feature_labels = list(features[0].keys())

        # --- Write filtered rows to CSV ---
        features = [
            {k: feat[k] for k in feature_labels if k in feat} for feat in features
        ]

        with open(features_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=feature_labels)
            if write_header:
                writer.writeheader()
            writer.writerows(features)
        return None

    def load_features(self):
        features_path = os.path.join(FEATURES, "features.csv")
        labels_path = os.path.join(FEATURES, "feature_labels.json")
        if not os.path.exists(features_path):
            raise FileNotFoundError("Features file does not exist.")
        if not os.path.exists(features_path) or not os.path.exists(labels_path):
            raise FileNotFoundError("Feature Labels file does not exist.")

        with open(labels_path, "r", encoding="utf-8") as f:
            feature_labels = json.load(f)
        if isinstance(feature_labels, list):
            features = pd.read_csv(features_path, low_memory=False)  # type: ignore
        else:
            feature_dtypes: Dict[str, Any] = {
                feature: DTYPES[meta["dtype"]]
                for feature, meta in feature_labels.items()
            }
            features = pd.read_csv(features_path, dtype=feature_dtypes)  # type: ignore
        return features

    def extract_features(
        self,
        main_player: str | None = None,
        games: List[pd.DataFrame] | None = None,
        save_output: bool = True,
        overwrite: bool = False,
    ):
        self.logger.print(f"Extracting features from processed game files...")
        self.succeeded = 0
        self.skipped = 0
        self.failed = 0
        features: List[Dict[str, Any]] = []

        Path(FEATURES).mkdir(parents=True, exist_ok=True)
        features_path = os.path.join(FEATURES, "features.csv")
        labels_path = os.path.join(FEATURES, "feature_labels.json")
        if overwrite and os.path.exists(features_path):
            os.remove(features_path)
        if overwrite and os.path.exists(labels_path):
            os.remove(labels_path)

        if (
            not overwrite
            and os.path.exists(features_path)
            and os.path.exists(labels_path)
        ):
            return self.load_features()

        if games:
            for game in games:
                try:
                    if main_player:
                        players = [main_player]
                    else:
                        players, _ = self.get_player_names_and_teams(game)
                    for player in players:
                        game_features = self.extract_game_features(game.copy(), player)
                        if save_output:
                            self.save_features(
                                game_features, features_path, labels_path
                            )
                        else:
                            features += game_features

                    self.logger.print("OK.")
                    self.succeeded += 1
                except Exception as e:
                    self.logger.print("FAILED.")
                    self.logger.print(f"  {type(e).__name__}: {e}")
                    self.failed += 1
            if save_output:
                return None
            return features

        game_files = sorted([f for f in os.listdir(PROCESSED) if f.endswith(".csv")])
        if not game_files:
            self.logger.print(f"No files found in {PROCESSED}.")
            return None

        for file_name in game_files:
            # TODO: Check existing record in features.csv
            # processed_json = self.load_processed_file(file_name)
            # if processed_json and not overwrite:
            #     self.logger.print(f"Using existing JSON for {file_name}.")
            #     self.skipped += 1
            #     if not save_output:
            #         all_game_frames.append(processed_json)
            #     else:
            #         self.save_processed_file(file_name, processed_json)
            # else:

            self.logger.print(
                f"Parsing {file_name} -> features.csv ...", end=" ", flush=True
            )
            try:
                processed_csv_file_path = os.path.join(PROCESSED, file_name)
                game = pd.read_csv(processed_csv_file_path, low_memory=False)  # type: ignore
                if main_player:
                    players = [main_player]
                else:
                    players, _ = self.get_player_names_and_teams(game)
                for player in players:
                    game_features = self.extract_game_features(game.copy(), player)
                    if save_output:
                        self.save_features(game_features, features_path, labels_path)
                    else:
                        features += game_features
                self.logger.print("OK.")
                self.succeeded += 1
            except Exception as e:
                self.logger.print("FAILED.")
                self.logger.print(f"  {type(e).__name__}: {e}")
                self.failed += 1

        self.logger.print(
            f"Done. succeeded={self.succeeded}, skipped={self.skipped}, failed={self.failed}."
        )
        if save_output and self.succeeded:
            self.logger.print(f"Saved features to: {FEATURES}/features.csv")
        self.logger.print()

        if save_output:
            return None
        return pd.DataFrame(features)
