import os
import csv
import json
from pathlib import Path
from typing import cast, Any, Dict, List, Union
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from rocketleague_ml.core.frame_by_frame_processor import Frame_By_Frame_Processor
from rocketleague_ml.utils.logging import Logger
from rocketleague_ml.config import (
    PROCESSED,
    FEATURES,
    FIELD_Y,
    Z_GROUND,
    Z_CEILING,
    X_WALL,
    Y_WALL,
    X_GOAL,
    GOAL_DEPTH,
    TOL,
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
        third_of_z = Z_CEILING / 3

        field_positioning_cols[f"{main_player}_highest_half"] = (
            game[f"{main_player}_positioning_y"] > 0
        )
        field_positioning_cols[f"{main_player}_lowest_half"] = (
            game[f"{main_player}_positioning_y"] <= 0
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
    ):
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

        return perc_in_position, avg_stint

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

        distance_cols = self.extract_distances(game, player_names, main_player)
        field_positioning_cols = self.extract_field_positioning(
            game, player_names, main_player
        )
        speed_cols = self.extract_speeds(game, player_names, main_player)

        game = pd.concat(
            [
                game,
                pd.DataFrame(
                    distance_cols | field_positioning_cols | speed_cols,
                    index=game.index,
                ),
            ],
            axis=1,
        )

        ball_distance_cols = [
            c for c in distance_cols.keys() if c.endswith("_distance_to_ball")
        ]
        dependent_cols: Dict[str, pd.DataFrame | pd.Series] = {}
        dependent_cols[f"{main_player}_closest_to_ball"] = game[
            f"{main_player}_distance_to_ball"
        ] == game[ball_distance_cols].min(axis=1)
        dependent_cols[f"{main_player}_farthest_from_ball"] = game[
            f"{main_player}_distance_to_ball"
        ] == game[ball_distance_cols].max(axis=1)
        # Wall checks, excluding goal occupancy
        dependent_cols[f"{main_player}_on_back_wall"] = (
            game[f"{main_player}_positioning_y"] <= -(Y_WALL - TOL)
        ) & (~game[f"{main_player}_in_own_goal"])
        dependent_cols[f"{main_player}_on_front_wall"] = (
            game[f"{main_player}_positioning_y"] >= (Y_WALL - TOL)
        ) & (~game[f"{main_player}_in_opponents_goal"])

        dependent_cols[f"{main_player}_is_still"] = (
            game[f"{main_player}_positioning_linear_velocity"] <= 10
        )
        dependent_cols[f"{main_player}_is_slow"] = (
            game[f"{main_player}_positioning_linear_velocity"] <= 500
        )
        dependent_cols[f"{main_player}_is_semi_slow"] = (
            game[f"{main_player}_positioning_linear_velocity"] > 500
        ) & (game[f"{main_player}_positioning_linear_velocity"] <= 1000)
        dependent_cols[f"{main_player}_is_medium_speed"] = (
            game[f"{main_player}_positioning_linear_velocity"] > 1000
        ) & (game[f"{main_player}_positioning_linear_velocity"] <= 1500)
        dependent_cols[f"{main_player}_is_semi_fast"] = (
            game[f"{main_player}_positioning_linear_velocity"] > 1500
        ) & (game[f"{main_player}_positioning_linear_velocity"] <= 2000)
        dependent_cols[f"{main_player}_is_fast"] = (
            game[f"{main_player}_positioning_linear_velocity"] > 2000
        )

        dependent_cols[f"{main_player}_is_drive_speed"] = (
            game[f"{main_player}_positioning_linear_velocity"] <= 1410
        )
        dependent_cols[f"{main_player}_is_boost_speed"] = (
            game[f"{main_player}_positioning_linear_velocity"] > 1410
        ) & (game[f"{main_player}_positioning_linear_velocity"] < 2200)
        dependent_cols[f"{main_player}_is_supersonic"] = (
            game[f"{main_player}_positioning_linear_velocity"] >= 2200
        )

        game = pd.concat([game, pd.DataFrame(dependent_cols, index=game.index)], axis=1)
        game.loc[:, f"{main_player}_airborne"] = ~(
            game[f"{main_player}_grounded"]
            | game[f"{main_player}_on_ceiling"]
            | game[f"{main_player}_on_left_wall"]
            | game[f"{main_player}_on_right_wall"]
            | game[f"{main_player}_on_back_wall"]
            | game[f"{main_player}_on_front_wall"]
        )

        distance_columns: Dict[str, List[str]] = {
            "Closest to Ball": ["closest_to_ball"],
            "Farthest from Ball": ["farthest_from_ball"],
        }
        for label, cols in distance_columns.items():
            perc, avg_stint = self.get_position_stats_for(game, cols, main_player)
            features[f"Percent Time while {label}"] = perc
            features[f"Average Stint while {label}"] = avg_stint
        avg_distance_to_ball = game[f"{main_player}_distance_to_ball"].mean()
        features["Average Distance to Ball"] = avg_distance_to_ball
        teammate_distances = [
            f"{main_player}_distance_to_{teammate}"
            for teammate in teams[main_player_team]
            if teammate != main_player
        ]
        avg_distance_to_teammates = game[teammate_distances].mean(axis=1).mean()
        features["Average Distance to Teammates"] = avg_distance_to_teammates
        opponent_distances = [
            f"{main_player}_distance_to_{opponent}"
            for opponent in teams[opponents_team]
        ]
        avg_distance_to_opponents = game[opponent_distances].mean(axis=1).mean()
        features["Average Distance to Opponents"] = avg_distance_to_opponents

        positioning_columns = {
            # Halves
            "In Offensive Half": ["offensive_half"],
            "In Defensive Half": ["defensive_half"],
            "In Left Half": ["left_half"],
            "In Right Half": ["right_half"],
            "In Highest Half": ["highest_half"],
            "In Lowest Half": ["lowest_half"],
            # Thirds (longitudinal)
            "In Offensive Third": ["offensive_third"],
            "In Neutral Third": ["neutral_third"],
            "In Defensive Third": ["defensive_third"],
            # Thirds (lateral)
            "In Left Third": ["left_third"],
            "In Middle Third": ["middle_third"],
            "In Right Third": ["right_third"],
            # Third (Vertical)
            "In Highest Third": ["highest_third"],
            "In Middle Aerial Third": ["middle_aerial_third"],
            "In Lowest Third": ["lowest_third"],
            # Half intersections
            "In Offensive Left Half": ["offensive_half", "left_half"],
            "In Offensive Right Half": ["offensive_half", "right_half"],
            "In Defensive Left Half": ["defensive_half", "left_half"],
            "In Defensive Right Half": ["defensive_half", "right_half"],
            # Half intersections with verticality
            "In Offensive Left Highest Half": [
                "offensive_half",
                "left_half",
                "highest_half",
            ],
            "In Offensive Left Lowest Half": [
                "offensive_half",
                "left_half",
                "lowest_half",
            ],
            "In Offensive Right Highest Half": [
                "offensive_half",
                "right_half",
                "highest_half",
            ],
            "In Offensive Right Lowest Half": [
                "offensive_half",
                "right_half",
                "lowest_half",
            ],
            "In Defensive Left Highest Half": [
                "defensive_half",
                "left_half",
                "highest_half",
            ],
            "In Defensive Left Lowest Half": [
                "defensive_half",
                "left_half",
                "lowest_half",
            ],
            "In Defensive Right Highest Half": [
                "defensive_half",
                "right_half",
                "highest_half",
            ],
            "In Defensive Right Lowest Half": [
                "defensive_half",
                "right_half",
                "lowest_half",
            ],
            # Third intersections
            "In Offensive Left Third": ["offensive_third", "left_third"],
            "In Offensive Middle Third": ["offensive_third", "middle_third"],
            "In Offensive Right Third": ["offensive_third", "right_third"],
            "In Neutral Left Third": ["neutral_third", "left_third"],
            "In Neutral Middle Third": ["neutral_third", "middle_third"],
            "In Neutral Right Third": ["neutral_third", "right_third"],
            "In Defensive Left Third": ["defensive_third", "left_third"],
            "In Defensive Middle Third": ["defensive_third", "middle_third"],
            "In Defensive Right Third": ["defensive_third", "right_third"],
            # Third intersections with verticality
            "In Offensive Left Highest Third": [
                "offensive_third",
                "left_third",
                "highest_third",
            ],
            "In Offensive Middle Highest Third": [
                "offensive_third",
                "middle_third",
                "highest_third",
            ],
            "In Offensive Right Highest Third": [
                "offensive_third",
                "right_third",
                "highest_third",
            ],
            "In Neutral Left Highest Third": [
                "neutral_third",
                "left_third",
                "highest_third",
            ],
            "In Neutral Middle Highest Third": [
                "neutral_third",
                "middle_third",
                "highest_third",
            ],
            "In Neutral Right Highest Third": [
                "neutral_third",
                "right_third",
                "highest_third",
            ],
            "In Defensive Left Highest Third": [
                "defensive_third",
                "left_third",
                "highest_third",
            ],
            "In Defensive Middle Highest Third": [
                "defensive_third",
                "middle_third",
                "highest_third",
            ],
            "In Defensive Right Highest Third": [
                "defensive_third",
                "right_third",
                "highest_third",
            ],
            "In Offensive Left Middle-Aerial Third": [
                "offensive_third",
                "left_third",
                "middle_aerial_third",
            ],
            "In Offensive Middle Middle-Aerial Third": [
                "offensive_third",
                "middle_third",
                "middle_aerial_third",
            ],
            "In Offensive Right Middle-Aerial Third": [
                "offensive_third",
                "right_third",
                "middle_aerial_third",
            ],
            "In Neutral Left Middle-Aerial Third": [
                "neutral_third",
                "left_third",
                "middle_aerial_third",
            ],
            "In Neutral Middle Middle-Aerial Third": [
                "neutral_third",
                "middle_third",
                "middle_aerial_third",
            ],
            "In Neutral Right Middle-Aerial Third": [
                "neutral_third",
                "right_third",
                "middle_aerial_third",
            ],
            "In Defensive Left Middle-Aerial Third": [
                "defensive_third",
                "left_third",
                "middle_aerial_third",
            ],
            "In Defensive Middle Middle-Aerial Third": [
                "defensive_third",
                "middle_third",
                "middle_aerial_third",
            ],
            "In Defensive Right Middle-Aerial Third": [
                "defensive_third",
                "right_third",
                "middle_aerial_third",
            ],
            "In Offensive Left Lowest Third": [
                "offensive_third",
                "left_third",
                "lowest_third",
            ],
            "In Offensive Middle Lowest Third": [
                "offensive_third",
                "middle_third",
                "lowest_third",
            ],
            "In Offensive Right Lowest Third": [
                "offensive_third",
                "right_third",
                "lowest_third",
            ],
            "In Neutral Left Lowest Third": [
                "neutral_third",
                "left_third",
                "lowest_third",
            ],
            "In Neutral Middle Lowest Third": [
                "neutral_third",
                "middle_third",
                "lowest_third",
            ],
            "In Neutral Right Lowest Third": [
                "neutral_third",
                "right_third",
                "lowest_third",
            ],
            "In Defensive Left Lowest Third": [
                "defensive_third",
                "left_third",
                "lowest_third",
            ],
            "In Defensive Middle Lowest Third": [
                "defensive_third",
                "middle_third",
                "lowest_third",
            ],
            "In Defensive Right Lowest Third": [
                "defensive_third",
                "right_third",
                "lowest_third",
            ],
            # Ball-relative positioning
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
        for label, cols in positioning_columns.items():
            perc, avg_stint = self.get_position_stats_for(game, cols, main_player)
            features[f"Percent Time {label}"] = perc
            features[f"Average Stint {label}"] = avg_stint

        speed_columns = {
            "Stationary": ["is_still"],
            "Slow": ["is_slow"],
            "Semi-Slow": ["is_semi_slow"],
            "Medium Speed": ["is_slow"],
            "Semi-Fast": ["is_semi_fast"],
            "Drive Speed": ["is_drive_speed"],
            "Boost Speed": ["is_supersonic"],
            "Supersonic": ["is_supersonic"],
        }
        for label, cols in speed_columns.items():
            perc, avg_stint = self.get_position_stats_for(game, cols, main_player)
            features[f"Percent Time while {label}"] = perc
            features[f"Average Stint while {label}"] = avg_stint

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
        rounds = game["round"].unique()
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
            feature_labels = list(features[0].keys())
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
        filtered_features = [
            {k: feat[k] for k in feature_labels if k in feat} for feat in features
        ]

        with open(features_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=feature_labels)
            if write_header:
                writer.writeheader()
            writer.writerows(filtered_features)
        return None

    def extract_features(
        self,
        main_player: str,
        games: List[pd.DataFrame] | None = None,
        save_output: bool = True,
        overwrite: bool = False,
    ):
        self.logger.print(f"Processing game data files...")
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

        if games:
            for game in games:
                try:
                    game_features = self.extract_game_features(game, main_player)
                    if save_output:
                        self.save_features(features, features_path, labels_path)
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
                game_features = self.extract_game_features(game, main_player)
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

        self.logger.print()
        self.logger.print(
            f"Done. succeeded={self.succeeded}, skipped={self.skipped}, failed={self.failed}."
        )
        self.logger.print(f"Saved features to: {FEATURES}/features.csv")
        self.logger.print()

        if save_output:
            return features
        return None
