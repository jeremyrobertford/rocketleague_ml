import os
import csv
import json
from pathlib import Path
from typing import cast, Any, Dict, List, Union, TypedDict
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from rocketleague_ml.config import ROUND_LENGTH
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
)
from rocketleague_ml.models.feature_extractor.aggregators import (
    aggregate_boost_usage,
    aggregate_player_rotations,
    aggregate_field_positioning,
    aggregate_speed,
    aggregate_mechanics,
)
from rocketleague_ml.utils.logging import Logger
from rocketleague_ml.config import (
    DTYPES,
    # FEATURE_LABELS,
    FEATURES,
    PROCESSED,
    TOTAL_FIELD_DISTANCE,
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

    def get_position_stats_for(
        self, game: pd.DataFrame, cols: Union[str, List[str]], main_player: str
    ):
        total_time = ROUND_LENGTH

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

    def compute_first_man_possession_sequences(
        self,
        features: Dict[str, float],
        game: pd.DataFrame,
        main_player: str,
    ):
        """
        Computes statistics when main_player is first man relative to the ball:
        - Percentage of instances where first man takes possession
        - Percentage where first man contests possession
        - Percentage where first man either takes or contests possession
        - Percentage where first man rotates (becomes second or third man instead)
        Also computes for non-rotating instances:
        - % time opponent had possession before main_player intervened
        - average stint length of opponent possession before main_player intervened
        """

        # --- Masks for main_player ---
        first_man_mask = game[f"{main_player}_player_rotation_first_man"].astype(bool)
        player_poss = game[f"{main_player}_in_possession"].astype(bool)
        player_contested = game[f"{main_player}_in_contested_possession"].astype(bool)
        player_take_or_contest = player_poss | player_contested
        player_rotate_mask = ~first_man_mask & (
            (game[f"{main_player}_player_rotation_second_man"])
            | (game[f"{main_player}_player_rotation_third_man"])
        )

        # --- Opponent team ---
        # main_team = next(t for t, players in teams.items() if main_player in players)
        # opponent_team = [t for t in teams.keys() if t != main_team][0]
        # opponent_players = teams[opponent_team]
        # opponent_poss = game[[f"{p}_in_possession" for p in opponent_players]].any(
        #     axis=1
        # )
        # opponent_contested = game[[f"{p}_in_possession" for p in opponent_players]].any(
        #     axis=1
        # ) | game[[f"{p}_in_contested_possession" for p in opponent_players]].any(axis=1)
        # opponent_take_or_contest = opponent_poss | opponent_contested

        # --- Identify instances where main_player is first man ---
        first_man_frames = first_man_mask

        # Total instances (frames) where main_player is first man
        total_first_man = first_man_frames.sum()

        if total_first_man == 0:
            features["Percentage Possession Taken as First Man"] = 0
            features["Percentage Possession Contested as First Man"] = 0
            features["Percentage Possession Challenged as First Man"] = 0
            features["Percentage Rotated as First Man"] = 0
            features["Percentage Opponent Has Possession Before Taken as First Man"] = 0
            features[
                "Average Stint Opponent Has Possession Before Taken as First Man"
            ] = 0
            features[
                "Percentage Opponent Has Possession Before Contested as First Man"
            ] = 0
            features[
                "Average Stint Opponent Has Possession Before Contested as First Man"
            ] = 0
            features[
                "Percentage Opponent Has Possession Before Challenged as First Man"
            ] = 0
            features[
                "Average Stint Opponent Has Possession Before Challenged as First Man"
            ] = 0
            return features

        # --- Percentages for first man ---
        take_mask = first_man_frames & player_poss
        contest_mask = first_man_frames & player_contested
        take_or_contest_mask = first_man_frames & player_take_or_contest
        rotate_mask = first_man_frames & player_rotate_mask

        features["Percentage Possession Taken as First Man"] = (
            take_mask.sum() / total_first_man
        )
        features["Percentage Possession Contested as First Man"] = (
            contest_mask.sum() / total_first_man
        )
        features["Percentage Possession Challenged as First Man"] = (
            take_or_contest_mask.sum() / total_first_man
        )
        features["Percentage Rotated as First Man"] = (
            rotate_mask.sum() / total_first_man
        )

        # TODO: Finalize speed of challenge metrics for first man
        # --- Opponent possession before main_player intervenes ---
        # Only consider instances where main_player does NOT rotate
        # no_rotate_mask = first_man_frames & ~player_rotate_mask
        # opponent_before_mask = opponent_take_or_contest & no_rotate_mask

        # Compute % time opponent had possession in those frames
        # total_time_no_rotate = game.loc[no_rotate_mask, "delta"].sum()
        # time_opponent_before = game.loc[opponent_before_mask, "delta"].sum()
        # features["perc_opponent_poss_before_take"] = (
        #     time_opponent_before / total_time_no_rotate
        #     if total_time_no_rotate > 0
        #     else 0
        # )

        # # --- Average stint length of opponent possession before main_player intervenes ---
        # if opponent_before_mask.any():
        #     run_ids = opponent_before_mask.ne(opponent_before_mask.shift()).cumsum()
        #     stint_groups = game.loc[opponent_before_mask].groupby(run_ids)
        #     stint_durations = stint_groups["delta"].sum()
        #     features["avg_stint_opponent_before_take"] = (
        #         stint_durations.mean() if not stint_durations.empty else 0
        #     )
        # else:
        #     features["avg_stint_opponent_before_take"] = 0

        return features

    def get_possession_and_pressure_stats(
        self,
        features: Dict[str, float],
        game: pd.DataFrame,
        main_player: str,
        teams: Dict[str, List[str]],
        column_label: str = "",
        filter_label: str = "",
    ) -> Dict[str, float]:
        """
        Returns a dictionary of aggregated metrics:
        - Player possession / contested possession time (% of total and % of team possession)
        - Offensive pressure time
        - Defensive pressure time
        - Teammate possession time
        - Average stint durations for possession and contested possession
        """

        total_time = game["delta"].sum()

        team_name = next(t for t, players in teams.items() if main_player in players)
        team_players = teams[team_name]
        opponent_team = [t for t in teams.keys() if t != team_name][0]
        opponent_players = teams[opponent_team]

        # Helper to compute percentage and stints
        def compute_stats(mask: pd.Series):
            time_mask = (mask * game["delta"]).sum()
            perc_total = time_mask / total_time if total_time > 0 else 0

            # Runs / stints
            run_ids = mask.ne(mask.shift()).cumsum()
            stint_groups = game[mask].groupby(run_ids)  # type: ignore
            stint_durations = stint_groups["delta"].sum()
            avg_stint = stint_durations.mean() if not stint_durations.empty else 0
            perc_avg_stint = avg_stint / total_time if total_time > 0 else 0

            return perc_total, perc_avg_stint

        # ---- Player possession ----
        player_poss = game[f"{main_player}_in_possession"].astype(bool)
        player_contested = game[f"{main_player}_in_possession"].astype(bool) | game[
            f"{main_player}_in_contested_possession"
        ].astype(bool)

        # Player possession metrics
        perc_poss, perc_stint_poss = compute_stats(player_poss)
        perc_contested, perc_stint_contested = compute_stats(player_contested)

        features["Percentage With Possession"] = perc_poss
        features["Percentage With Contested Possession"] = perc_contested
        features["Average Stint With Possession"] = perc_stint_poss
        features["Average Stint With Contested Possession"] = perc_stint_contested

        # ---- Team possession ----
        team_poss = game[[f"{p}_in_possession" for p in team_players]].any(axis=1)
        team_contested = game[[f"{p}_in_possession" for p in team_players]].any(
            axis=1
        ) | game[[f"{p}_in_contested_possession" for p in team_players]].any(axis=1)

        perc_team_poss, perc_stint_team_poss = compute_stats(team_poss)
        perc_team_contested, perc_stint_team_contested = compute_stats(team_contested)

        features["Team in Possession"] = perc_team_poss
        features["Team in Contested Possession"] = perc_team_contested
        features["Average Stint Team in Possession"] = perc_stint_team_poss
        features["Average Stint Team in Contested Possession"] = (
            perc_stint_team_contested
        )

        # ---- Offensive pressure ----
        # Pressure = possession in offensive half
        offensive_mask = game[f"{main_player}_offensive_half"].astype(bool)
        player_offense = player_poss & offensive_mask
        player_offense_contested = player_contested & offensive_mask
        perc_offense, perc_stint_offense = compute_stats(player_offense)
        perc_offense_contested, perc_stint_offense_contested = compute_stats(
            player_offense_contested
        )
        features["Percentage With Offensive Pressure"] = perc_offense
        features["Percentage With Offensive Contested Pressure"] = (
            perc_offense_contested
        )
        features["Average Stint With Offensive Pressure"] = perc_stint_offense
        features["Average Stint With Offensive Contested Pressure"] = (
            perc_stint_offense_contested
        )

        team_offense = team_poss & offensive_mask
        team_offense_contested = team_contested & offensive_mask
        perc_team_offense, perc_stint_team_offense = compute_stats(team_offense)
        perc_team_offense_contested, perc_stint_team_offense_contested = compute_stats(
            team_offense_contested
        )
        features["Team With Offensive Pressure"] = perc_team_offense
        features["Team with Offensive Contested Pressure"] = perc_team_offense_contested
        features["Average Stint Team With Offensive Pressure"] = perc_stint_team_offense
        features["Average Stint Team with Offensive Contested Pressure"] = (
            perc_stint_team_offense_contested
        )

        # ---- Defensive pressure (opponent in possession in defensive half) ----
        defensive_mask = game[f"{main_player}_defensive_half"].astype(bool)
        opponent_poss = game[[f"{p}_in_possession" for p in opponent_players]].any(
            axis=1
        )
        opponent_contested = game[[f"{p}_in_possession" for p in opponent_players]].any(
            axis=1
        ) | game[[f"{p}_in_contested_possession" for p in opponent_players]].any(axis=1)
        perc_opp_poss, perc_stint_opp_poss = compute_stats(opponent_poss)
        perc_opp_contested, perc_stint_opp_contested = compute_stats(opponent_contested)

        features["Opponent Team in Possession"] = perc_opp_poss
        features["Opponent Team in Contested Possession"] = perc_opp_contested
        features["Average Stint Team in Opponent Possession"] = perc_stint_opp_poss
        features["Average Stint Team in Opponent Contested Possession"] = (
            perc_stint_opp_contested
        )

        perc_defense, _ = compute_stats(opponent_poss & defensive_mask)
        perc_defense_contested, _ = compute_stats(opponent_contested & defensive_mask)
        features["Percentage Under Defensive Pressure"] = perc_defense
        features["Percentage Under Contested Defensive Pressure"] = (
            perc_defense_contested
        )

        features = self.compute_first_man_possession_sequences(
            features=features,
            game=game,
            main_player=main_player,
        )

        return features

    def refine_possession_from_simple(
        self,
        game: pd.DataFrame,
        player_names: List[str],
        teams: Dict[str, List[str]],
        column_label: str,
        possession_threshold: float = 2.0,  # seconds of stability before "real" possession
    ) -> pd.DataFrame:
        """
        Convert simple possession into sustained possession and contested possession.

        Logic:
            - Each player's simple possession sequence is examined.
            - If it lasts >= possession_threshold before another touch occurs → real possession.
            - If another touch occurs sooner → contested possession.

        Adds columns:
            - {player}_in_possession
            - {player}_in_contested_possession
            - blue_in_possession
            - orange_in_possession
            - blue_in_contested_possession
            - orange_in_contested_possession
        """

        time = game["time"].to_numpy()
        # delta = game["delta"].to_numpy()
        n_frames = len(game)

        # Initialize arrays
        possession = np.zeros((n_frames, len(player_names)), dtype=int)
        contested = np.zeros((n_frames, len(player_names)), dtype=int)

        # For every player, process their simple possession spans
        for p_idx, player in enumerate(player_names):
            simple_mask = game[f"{player}_simple_possession"].to_numpy().astype(bool)

            # Find contiguous segments of simple possession
            in_segment = False
            start_idx = None
            for i in range(n_frames):
                if simple_mask[i] and not in_segment:
                    in_segment = True
                    start_idx = i
                elif not simple_mask[i] and in_segment:
                    end_idx = i - 1
                    in_segment = False

                    # Analyze segment
                    start_time = time[start_idx]
                    next_touch_time = time[i]

                    # Determine label
                    if (
                        next_touch_time - start_time <= possession_threshold
                        or game.loc[end_idx, "contested_simple_possession"]
                    ):
                        contested[start_idx : end_idx + 1, p_idx] = 1
                    else:
                        possession[start_idx : end_idx + 1, p_idx] = 1

            # Handle open segment if player still in possession at end
            if in_segment:
                end_idx = n_frames - 1
                possession[start_idx : end_idx + 1, p_idx] = 1  # default to possession

        # Write columns
        for idx, player in enumerate(player_names):
            game[f"{player}_in_possession"] = possession[:, idx]
            game[f"{player}_in_contested_possession"] = contested[:, idx]

        # Team-level aggregation
        for team_name in ["Blue", "Orange"]:
            players = teams[team_name]
            game[f"{team_name.lower()}_in_possession"] = (
                game[[f"{p}_in_possession" for p in players]].any(axis=1).astype(int)
            )
            game[f"{team_name.lower()}_in_contested_possession"] = (
                game[[f"{p}_in_contested_possession" for p in players]]
                .any(axis=1)
                .astype(int)
            )
            game[f"{team_name.lower()}_in_challenging_possession"] = (
                game[f"{team_name.lower()}_in_possession"].astype(bool)
                | game[f"{team_name.lower()}_in_contested_possession"].astype(bool)
            ).astype(int)
            # --- Applying pressure ---
            # Assumption: positive x = Blue’s attacking side; negative x = Orange’s attacking side
            if team_name == "Blue":
                on_opponent_side = game["ball_positioning_x"] > 0
            else:
                on_opponent_side = game["ball_positioning_x"] < 0

            game[f"{team_name.lower()}_applying_pressure"] = (
                game[f"{team_name.lower()}_in_possession"].astype(bool)
                & on_opponent_side
            ).astype(int)
            game[f"{team_name.lower()}_applying_contested_pressure"] = (
                game[f"{team_name.lower()}_in_contested_possession"].astype(bool)
                & on_opponent_side
            ).astype(int)
            game[f"{team_name.lower()}_applying_challenging_pressure"] = (
                game[f"{team_name.lower()}_in_challenging_possession"].astype(bool)
                & on_opponent_side
            ).astype(int)

        return game

    def extract_simple_possession(
        self,
        game: pd.DataFrame,
        player_names: List[str],
        teams: Dict[str, List[str]],
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
            game[[f"{p}_simple_possession" for p in blue_players]]
            .any(axis=1)
            .astype(int)
        )
        game["orange_simple_possession"] = (
            game[[f"{p}_simple_possession" for p in orange_players]]
            .any(axis=1)
            .astype(int)
        )

        # Contested = multiple players in possession this frame
        game["contested_simple_possession"] = (
            simple_possession.sum(axis=1) > 1
        ).astype(int)

        return game

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

        game = self.extract_simple_possession(
            game=game.copy(),
            player_names=player_names,
            teams=teams,
        )

        game = self.refine_possession_from_simple(
            game=game.copy(),
            player_names=player_names,
            teams=teams,
            column_label=column_label,
        )

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
        main_player_team = next(
            (t for t, players in teams.items() if main_player in players), None
        )
        if not main_player_team:
            raise ValueError("No team found")
        opponent_team = "Orange" if main_player_team == "Blue" else "Blue"

        features: Dict[str, float] = {}

        total_touches = game[f"{main_player}_touch_total"].sum()
        features["Fifty-Fifty Touch Percentage"] = (
            game[f"{main_player}_touch_fifty_fifty"].sum() / total_touches
            if total_touches > 0
            else 0
        )
        features["Towards Goal Touch Percentage"] = (
            game[f"{main_player}_touch_towards_goal"].sum() / total_touches
            if total_touches > 0
            else 0
        )
        features["Towards Teammate Touch Percentage"] = (
            game[f"{main_player}_touch_towards_teammate"].sum() / total_touches
            if total_touches > 0
            else 0
        )
        features["Towards Opponent Touch Percentage"] = (
            game[f"{main_player}_touch_towards_opponent"].sum() / total_touches
            if total_touches > 0
            else 0
        )
        features["Towards Open Space Touch Percentage"] = (
            game[f"{main_player}_touch_towards_open_space"].sum() / total_touches
            if total_touches > 0
            else 0
        )

        movement_columns: Dict[str, List[str]] = {
            # "Drifting": ["drift_active"]
        }
        for label, cols in movement_columns.items():
            perc, avg_stint = self.get_position_stats_for(game, cols, main_player)
            features[f"Percent Time while {label}"] = perc
            features[f"Average Stint while {label}"] = avg_stint

        distance_columns: Dict[str, List[str]] = {
            "Closest to Ball": ["closest_to_ball"],
            "Farthest from Ball": ["farthest_from_ball"],
        }
        for label, cols in distance_columns.items():
            perc, avg_stint = self.get_position_stats_for(game, cols, main_player)
            features[f"Percent Time while {label}"] = perc
            features[f"Average Stint while {label}"] = avg_stint
        avg_distance_to_ball = game[f"{main_player}_distance_to_ball"].mean()
        features["Average Distance to Ball"] = (
            avg_distance_to_ball / TOTAL_FIELD_DISTANCE
        )
        teammate_distances = [
            f"{main_player}_distance_to_{teammate}"
            for teammate in teams[main_player_team]
            if teammate != main_player
        ]
        avg_distance_to_teammates = game[teammate_distances].mean(axis=1).mean()
        features["Average Distance to Teammates"] = (
            avg_distance_to_teammates / TOTAL_FIELD_DISTANCE
        )
        opponent_distances = [
            f"{main_player}_distance_to_{opponent}" for opponent in teams[opponent_team]
        ]
        avg_distance_to_opponents = game[opponent_distances].mean(axis=1).mean()
        features["Average Distance to Opponents"] = (
            avg_distance_to_opponents / TOTAL_FIELD_DISTANCE
        )

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

        features = self.get_possession_and_pressure_stats(
            features=features,
            game=game,
            main_player=main_player,
            teams=teams,
            column_label=column_label,
            filter_label=filter_label,
        )

        if config["include_player_rotations"]:
            player_rotation_features = aggregate_player_rotations(
                game=game, main_player=main_player
            )
            features = features | player_rotation_features

        perc_blue_simple_possession, _ = self.get_position_stats_for(
            game, ["simple_possession"], "blue"
        )
        features["Percentage Blue With Possession"] = perc_blue_simple_possession
        perc_contested_simple_possession, _ = self.get_position_stats_for(
            game, ["simple_possession"], "contested"
        )
        features["Percentage Contested With Possession"] = (
            perc_contested_simple_possession
        )
        perc_orange_simple_possession, _ = self.get_position_stats_for(
            game, ["simple_possession"], "orange"
        )
        features["Percentage Orange With Possession"] = perc_orange_simple_possession

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
