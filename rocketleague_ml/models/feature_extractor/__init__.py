import os
import csv
import json
from pathlib import Path
from typing import cast, Any, Dict, List, TypedDict
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from rocketleague_ml.models.frame_by_frame_processor import Frame_By_Frame_Processor
from rocketleague_ml.models.feature_extractor.extractors import (
    get_player_rotation_cols,
    get_simple_player_rotation_cols,
    get_boost_usage_cols,
    get_speed_cols,
    get_dependent_speed_cols,
    get_field_positioning_cols,
    get_touch_type_cols,
    get_distancing_cols,
    get_dependent_distancing_cols,
    get_dependent_field_positioning_cols,
    assign_jumps,
    assign_possession,
)
from rocketleague_ml.models.feature_extractor.aggregators import (
    aggregate_boost_usage,
    aggregate_player_rotations,
    aggregate_field_positioning,
    aggregate_speed,
    aggregate_mechanics,
    aggregate_distancing,
    aggregate_possession,
)
from rocketleague_ml.utils.logging import Logger
from rocketleague_ml.config import (
    DTYPES,
    # FEATURE_LABELS,
    FEATURES,
    PROCESSED,
)


class Config(TypedDict):
    include_touch_types: bool
    include_field_positioning: bool
    include_distancing: bool
    include_speed: bool
    include_possession: bool
    include_pressure: bool
    include_player_rotations: bool
    include_boost_usage: bool
    include_mechanics: bool


class Rocket_League_Feature_Extractor:
    def __init__(self, processor: Frame_By_Frame_Processor, logger: Logger = Logger()):
        self.processor = processor
        self.logger = logger
        self.config: Config = {
            "include_touch_types": True,
            "include_field_positioning": True,
            "include_distancing": True,
            "include_speed": True,
            "include_possession": True,
            "include_pressure": True,
            "include_player_rotations": True,
            "include_boost_usage": True,
            "include_mechanics": True,
        }

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

    def create_game_features(
        self,
        player_names: List[str],
        game: pd.DataFrame,
        teams: Dict[str, List[str]],
        column_label: str = "",
    ):
        game = game.copy()

        initial_feature_cols: Dict[
            str, pd.Series | NDArray[np.float64] | NDArray[np.int_]
        ] = {}

        if self.config["include_distancing"]:
            distancing_cols = get_distancing_cols(game=game, teams=teams)
            initial_feature_cols = initial_feature_cols | distancing_cols

        if self.config["include_field_positioning"]:
            field_positioning_cols = get_field_positioning_cols(game=game, teams=teams)
            initial_feature_cols = initial_feature_cols | field_positioning_cols

        if self.config["include_touch_types"]:
            touch_types_cols = get_touch_type_cols(game=game, teams=teams)
            initial_feature_cols = initial_feature_cols | touch_types_cols

        if len(initial_feature_cols):
            game = pd.concat(
                [
                    game,
                    pd.DataFrame(
                        initial_feature_cols,
                        index=game.index,
                    ),
                ],
                axis=1,
            )

        first_pass_dependent_feature_cols: Dict[
            str, pd.Series | NDArray[np.float64] | NDArray[np.int_]
        ] = {}

        if self.config["include_distancing"]:
            dependent_distancing_cols = get_dependent_distancing_cols(
                game=game, teams=teams
            )
            first_pass_dependent_feature_cols |= dependent_distancing_cols

        if self.config["include_field_positioning"]:
            dependent_field_positioning_cols = get_dependent_field_positioning_cols(
                game=game, teams=teams
            )
            first_pass_dependent_feature_cols |= dependent_field_positioning_cols

        if self.config["include_player_rotations"]:
            simple_player_rotation_cols = get_simple_player_rotation_cols(
                game=game, teams=teams
            )
            first_pass_dependent_feature_cols |= simple_player_rotation_cols

        if self.config["include_speed"]:
            speed_cols = get_speed_cols(game=game, teams=teams)
            first_pass_dependent_feature_cols |= speed_cols

        if len(first_pass_dependent_feature_cols):
            game = pd.concat(
                [
                    game,
                    pd.DataFrame(
                        first_pass_dependent_feature_cols,
                        index=game.index,
                    ),
                ],
                axis=1,
            )

        second_pass_dependent_feature_cols: Dict[
            str, pd.Series | NDArray[np.float64] | NDArray[np.int_]
        ] = {}
        if self.config["include_speed"]:
            dependent_speed_cols = get_dependent_speed_cols(game=game, teams=teams)
            second_pass_dependent_feature_cols |= dependent_speed_cols

        if self.config["include_possession"]:
            game = assign_possession(game, teams)

        if self.config["include_mechanics"]:
            game = assign_jumps(game=game, teams=teams)

        if self.config["include_boost_usage"]:
            boost_usage_cols = get_boost_usage_cols(game=game, teams=teams)
            second_pass_dependent_feature_cols |= boost_usage_cols

        if self.config["include_player_rotations"]:
            player_rotation_cols = get_player_rotation_cols(game=game, teams=teams)
            second_pass_dependent_feature_cols |= player_rotation_cols

        if len(second_pass_dependent_feature_cols):
            game = pd.concat(
                [
                    game,
                    pd.DataFrame(
                        second_pass_dependent_feature_cols,
                        index=game.index,
                    ),
                ],
                axis=1,
            )

        game.to_csv(os.path.join(FEATURES, "debug.csv"), index=False)
        return game

    def aggregate_features(
        self,
        game: pd.DataFrame,
        main_player: str,
        teams: Dict[str, List[str]],
        column_label: str = "",
        filter_label: str = "",
        config: Config | None = None,
    ) -> Dict[str, float]:
        game = game.copy()
        config = config or self.config

        features: Dict[str, float] = {}

        if config["include_distancing"]:
            distancing_features = aggregate_distancing(
                game=game, main_player=main_player, teams=teams
            )
            features = features | distancing_features

        if config["include_mechanics"]:
            mechanics_features = aggregate_mechanics(game=game, main_player=main_player)
            features = features | mechanics_features

        if config["include_field_positioning"]:
            field_positioning_features = aggregate_field_positioning(
                game=game, main_player=main_player
            )
            features = features | field_positioning_features

        if config["include_speed"]:
            speed_features = aggregate_speed(game=game, main_player=main_player)
            features = features | speed_features

        if config["include_boost_usage"]:
            boost_usage_features = aggregate_boost_usage(
                game=game, main_player=main_player
            )
            features = features | boost_usage_features

        if config["include_possession"]:
            possession_features = aggregate_possession(
                game=game, main_player=main_player, teams=teams
            )
            features = features | possession_features

        if config["include_player_rotations"]:
            player_rotation_features = aggregate_player_rotations(
                game=game, main_player=main_player
            )
            features = features | player_rotation_features

        return features

    def extract_round_features(
        self,
        game: pd.DataFrame,
        main_player: str,
        id: str,
        round: int,
        player_names: List[str],
        teams: Dict[str, List[str]],
    ):
        features: Dict[str, Any] = {
            "id": id,
            "round": round,
        }

        game = self.create_game_features(
            game=game,
            player_names=player_names,
            teams=teams,
        )

        features = self.aggregate_features(
            game=game,
            main_player=main_player,
            teams=teams,
        )

        team = next((t for t, players in teams.items() if main_player in players), None)
        if not team:
            raise ValueError("No team found")
        # TODO: these are not right, but have the right idea
        features["Scored Goal"] = (
            game[main_player + "_goal"].sum()
            if main_player + "_goal" in game.columns
            else 0
        )
        features["Team Scored Goal"] = max(
            (
                game[teams[team][0] + "_goal"].sum()
                if teams[team][0] + "_goal" in game.columns
                and teams[team][0] != main_player
                else 0
            ),
            (
                game[teams[team][1] + "_goal"].sum()
                if teams[team][1] + "_goal" in game.columns
                and teams[team][1] != main_player
                else 0
            ),
            (
                game[teams[team][2] + "_goal"].sum()
                if teams[team][2] + "_goal" in game.columns
                and teams[team][2] != main_player
                else 0
            ),
            0,
        )

        return features

    def extract_game_features(self, game: pd.DataFrame, main_player: str):
        features: List[Dict[str, Any]] = []
        player_names, teams = self.get_player_names_and_teams(game)

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
            round_frames = game[game["round"] == round].copy().reset_index(drop=True)
            round_features = self.extract_round_features(
                game=round_frames,
                main_player=main_player,
                id=id,
                round=round,
                player_names=player_names,
                teams=teams,
            )
            features.append(round_features)
        return features

    def save_features(
        self,
        features: List[Dict[str, Any]],
        features_path: str,
        labels_path: str,
        transposed: bool = False,
    ):
        write_header = not os.path.exists(features_path)

        # --- Handle feature_labels.json ---
        if write_header:
            feature_labels = [feature for feature in list(features[0].keys())]
            with open(labels_path, "w", encoding="utf-8") as jf:
                json.dump(feature_labels, jf, indent=4)
        else:
            if os.path.exists(labels_path):
                with open(labels_path, "r", encoding="utf-8") as jf:
                    feature_labels = json.load(jf)
            else:
                feature_labels = list(features[0].keys())

        # --- Filter features based on JSON ---
        features_filtered = [
            {k: feat[k] for k in feature_labels if k in feat} for feat in features
        ]

        # --- Write to CSV ---
        if not transposed:
            # Standard: features as rows, keys as columns
            with open(features_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=feature_labels)
                if write_header:
                    writer.writeheader()
                writer.writerows(features_filtered)
        else:
            # Transposed: feature names as rows, each column is one instance
            # First, collect values per feature
            transposed_rows: List[Dict[str, Any]] = []
            for feature in feature_labels:
                row = {"feature": feature}
                for idx, feat in enumerate(features_filtered):
                    row[f"instance_{idx}"] = feat.get(feature, "")
                transposed_rows.append(row)

            # Determine fieldnames: feature + all instances
            fieldnames = ["feature"] + [
                f"instance_{i}" for i in range(len(features_filtered))
            ]

            with open(features_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerows(transposed_rows)

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
        if save_output and overwrite and os.path.exists(features_path):
            os.remove(features_path)
        if save_output and overwrite and os.path.exists(labels_path):
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
                                game_features,
                                features_path,
                                labels_path,
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
            return pd.DataFrame(features)

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
