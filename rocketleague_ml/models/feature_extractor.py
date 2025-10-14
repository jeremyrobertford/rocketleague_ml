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

    def compute_first_man_possession_sequences(
        self,
        features: Dict[str, float],
        game: pd.DataFrame,
        main_player: str,
        teams: Dict[str, List[str]],
        column_label: str = "",
        filter_label: str = "",
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
        first_man_mask = game[f"{main_player}_{column_label}simple_first_man"].astype(
            bool
        )
        player_poss = game[f"{main_player}_{column_label}in_possession"].astype(bool)
        player_contested = game[
            f"{main_player}_{column_label}in_contested_possession"
        ].astype(bool)
        player_take_or_contest = player_poss | player_contested
        player_rotate_mask = ~first_man_mask & (
            (game[f"{main_player}_{column_label}simple_second_man"])
            | (game[f"{main_player}_{column_label}simple_third_man"])
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
            features[f"{filter_label}Percentage Possession Taken as First Man"] = 0
            features[f"{filter_label}Percentage Possession Contested as First Man"] = 0
            features[f"{filter_label}Percentage Possession Challenged as First Man"] = 0
            features[f"{filter_label}Percentage Rotated as First Man"] = 0
            features[
                f"{filter_label}Percentage Opponent Has Possession Before Taken as First Man"
            ] = 0
            features[
                f"{filter_label}Average Stint Opponent Has Possession Before Taken as First Man"
            ] = 0
            features[
                f"{filter_label}Percentage Opponent Has Possession Before Contested as First Man"
            ] = 0
            features[
                f"{filter_label}Average Stint Opponent Has Possession Before Contested as First Man"
            ] = 0
            features[
                f"{filter_label}Percentage Opponent Has Possession Before Challenged as First Man"
            ] = 0
            features[
                f"{filter_label}Average Stint Opponent Has Possession Before Challenged as First Man"
            ] = 0
            return features

        # --- Percentages for first man ---
        take_mask = first_man_frames & player_poss
        contest_mask = first_man_frames & player_contested
        take_or_contest_mask = first_man_frames & player_take_or_contest
        rotate_mask = first_man_frames & player_rotate_mask

        features[f"{filter_label}Percentage Possession Taken as First Man"] = (
            take_mask.sum() / total_first_man
        )
        features[f"{filter_label}Percentage Possession Contested as First Man"] = (
            contest_mask.sum() / total_first_man
        )
        features[f"{filter_label}Percentage Possession Challenged as First Man"] = (
            take_or_contest_mask.sum() / total_first_man
        )
        features[f"{filter_label}Percentage Rotated as First Man"] = (
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
        player_poss = game[f"{main_player}_{column_label}in_possession"].astype(bool)
        player_contested = game[f"{main_player}_{column_label}in_possession"].astype(
            bool
        ) | game[f"{main_player}_{column_label}in_contested_possession"].astype(bool)

        # Player possession metrics
        perc_poss, perc_stint_poss = compute_stats(player_poss)
        perc_contested, perc_stint_contested = compute_stats(player_contested)

        features[f"{filter_label}With Possession"] = perc_poss
        features[f"{filter_label}With Contested Possession"] = perc_contested
        features[f"{filter_label}Average Stint With Possession"] = perc_stint_poss
        features[f"{filter_label}Average Stint With Contested Possession"] = (
            perc_stint_contested
        )

        # ---- Team possession ----
        team_poss = game[[f"{p}_in_possession" for p in team_players]].any(axis=1)
        team_contested = game[[f"{p}_in_possession" for p in team_players]].any(
            axis=1
        ) | game[[f"{p}_in_contested_possession" for p in team_players]].any(axis=1)

        perc_team_poss, perc_stint_team_poss = compute_stats(team_poss)
        perc_team_contested, perc_stint_team_contested = compute_stats(team_contested)

        features[f"{filter_label}Team in Possession"] = perc_team_poss
        features[f"{filter_label}Team in Contested Possession"] = perc_team_contested
        features[f"{filter_label}Average Stint Team in Possession"] = (
            perc_stint_team_poss
        )
        features[f"{filter_label}Average Stint Team in Contested Possession"] = (
            perc_stint_team_contested
        )

        # ---- Offensive pressure ----
        # Pressure = possession in offensive half
        offensive_mask = game[f"{main_player}_{column_label}offensive_half"].astype(
            bool
        )
        player_offense = player_poss & offensive_mask
        player_offense_contested = player_contested & offensive_mask
        perc_offense, perc_stint_offense = compute_stats(player_offense)
        perc_offense_contested, perc_stint_offense_contested = compute_stats(
            player_offense_contested
        )
        features[f"{filter_label}Percentage With Offensive Pressure"] = perc_offense
        features[f"{filter_label}Percentage With Offensive Contested Pressure"] = (
            perc_offense_contested
        )
        features[f"{filter_label}Average Stint With Offensive Pressure"] = (
            perc_stint_offense
        )
        features[f"{filter_label}Average Stint With Offensive Contested Pressure"] = (
            perc_stint_offense_contested
        )

        team_offense = team_poss & offensive_mask
        team_offense_contested = team_contested & offensive_mask
        perc_team_offense, perc_stint_team_offense = compute_stats(team_offense)
        perc_team_offense_contested, perc_stint_team_offense_contested = compute_stats(
            team_offense_contested
        )
        features[f"{filter_label}Team With Offensive Pressure"] = perc_team_offense
        features[f"{filter_label}Team with Offensive Contested Pressure"] = (
            perc_team_offense_contested
        )
        features[f"{filter_label}Average Stint Team With Offensive Pressure"] = (
            perc_stint_team_offense
        )
        features[
            f"{filter_label}Average Stint Team with Offensive Contested Pressure"
        ] = perc_stint_team_offense_contested

        # ---- Defensive pressure (opponent in possession in defensive half) ----
        defensive_mask = game[f"{main_player}_{column_label}defensive_half"].astype(
            bool
        )
        opponent_poss = game[[f"{p}_in_possession" for p in opponent_players]].any(
            axis=1
        )
        opponent_contested = game[[f"{p}_in_possession" for p in opponent_players]].any(
            axis=1
        ) | game[[f"{p}_in_contested_possession" for p in opponent_players]].any(axis=1)
        perc_opp_poss, perc_stint_opp_poss = compute_stats(opponent_poss)
        perc_opp_contested, perc_stint_opp_contested = compute_stats(opponent_contested)

        features[f"{filter_label}Opponent Team in Possession"] = perc_opp_poss
        features[f"{filter_label}Opponent Team in Contested Possession"] = (
            perc_opp_contested
        )
        features[f"{filter_label}Average Stint Team in Opponent Possession"] = (
            perc_stint_opp_poss
        )
        features[
            f"{filter_label}Average Stint Team in Opponent Contested Possession"
        ] = perc_stint_opp_contested

        perc_defense, _ = compute_stats(opponent_poss & defensive_mask)
        perc_defense_contested, _ = compute_stats(opponent_contested & defensive_mask)
        features[f"{filter_label}{main_player}_perc_defensive_pressure"] = perc_defense
        features[f"{filter_label}{main_player}_perc_defensive_contested_pressure"] = (
            perc_defense_contested
        )

        features = self.compute_first_man_possession_sequences(
            features=features,
            game=game,
            main_player=main_player,
            teams=teams,
            column_label=column_label,
            filter_label=filter_label,
        )

        return features

    def extract_touch_locations(
        self,
        game: pd.DataFrame,
        player_names: List[str],
        teams: Dict[str, List[str]],
    ) -> Dict[str, pd.Series]:
        toward_goal_tol = 0.7
        toward_teammate_tol = 0.6
        toward_opponent_tol = 0.6
        toward_open_space_tol: float | None = None
        touch_location_col: Dict[str, pd.Series] = {}

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

        for player in player_names:
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
            opponent_positions = [
                positions[p] for p in teams[opponent_team] if p != player
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
            touch_location_col[f"{player}_touch_total"] = pd.Series(
                touch_mask.astype(int), index=game.index
            )

            # Add direction dot products for analysis
            touch_location_col[f"{player}_touch_to_goal_dot"] = pd.Series(
                to_goal, index=game.index
            )
            touch_location_col[f"{player}_touch_to_teammate_dot"] = pd.Series(
                to_teammate, index=game.index
            )
            touch_location_col[f"{player}_touch_to_opponent_dot"] = pd.Series(
                to_opponent, index=game.index
            )
            touch_location_col[f"{player}_touch_open_space_score"] = pd.Series(
                open_space_score, index=game.index
            )

            # Add independent category flags
            for cat, mask in categories.items():
                touch_location_col[f"{player}_touch_{cat}"] = pd.Series(
                    (mask & touch_mask).astype(int), index=game.index
                )

        return touch_location_col

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
                        or game.loc[
                            end_idx, f"{column_label}contested_simple_possession"
                        ]
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

    def aggregate_features(
        self,
        game: pd.DataFrame,
        main_player: str,
        teams: Dict[str, List[str]],
        column_label: str = "",
        filter_label: str = "",
    ) -> Dict[str, Any]:
        main_player_team = next(
            (t for t, players in teams.items() if main_player in players), None
        )
        if not main_player_team:
            raise ValueError("No team found")
        opponent_team = "Orange" if main_player_team == "Blue" else "Blue"
        features: Dict[str, Any] = {}
        total_touches = game[f"{main_player}_{column_label}touch_total"].sum()
        features[f"{filter_label}Fifty-Fifty Touch Percentage"] = (
            game[f"{main_player}_{column_label}touch_fifty_fifty"].sum() / total_touches
        )
        features[f"{filter_label}Towards Goal Touch Percentage"] = (
            game[f"{main_player}_{column_label}touch_towards_goal"].sum()
            / total_touches
        )
        features[f"{filter_label}Towards Teammate Touch Percentage"] = (
            game[f"{main_player}_{column_label}touch_towards_teammate"].sum()
            / total_touches
        )
        features[f"{filter_label}Towards Opponent Touch Percentage"] = (
            game[f"{main_player}_{column_label}touch_towards_opponent"].sum()
            / total_touches
        )
        features[f"{filter_label}Towards Open Space Touch Percentage"] = (
            game[f"{main_player}_{column_label}touch_towards_open_space"].sum()
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
            f"{main_player}_{column_label}distance_to_ball"
        ].mean()
        features[f"{filter_label}Average Distance to Ball"] = (
            avg_distance_to_ball / TOTAL_FIELD_DISTANCE
        )
        teammate_distances = [
            f"{main_player}_{column_label}distance_to_{teammate}"
            for teammate in teams[main_player_team]
            if teammate != main_player
        ]
        avg_distance_to_teammates = game[teammate_distances].mean(axis=1).mean()
        features[f"{filter_label}Average Distance to Teammates"] = (
            avg_distance_to_teammates / TOTAL_FIELD_DISTANCE
        )
        opponent_distances = [
            f"{main_player}_{column_label}distance_to_{opponent}"
            for opponent in teams[opponent_team]
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

        features = self.get_possession_and_pressure_stats(
            features=features,
            game=game,
            main_player=main_player,
            teams=teams,
            column_label=column_label,
            filter_label=filter_label,
        )

        rotation_columns = {
            f"{filter_label}First Man": [f"{column_label}simple_first_man"],
            f"{filter_label}Second Man": [f"{column_label}simple_second_man"],
            f"{filter_label}Third Man": [f"{column_label}simple_third_man"],
        }
        for label, cols in rotation_columns.items():
            perc, avg_stint = self.get_position_stats_for(game, cols, main_player)
            features[f"{filter_label}Percent Time while {label}"] = perc
            features[f"{filter_label}Average Stint while {label}"] = avg_stint

        perc_blue_simple_possession, _ = self.get_position_stats_for(
            game, [f"{column_label}simple_possession"], "blue"
        )
        features[f"{filter_label}Percentage Blue With Possession"] = (
            perc_blue_simple_possession
        )
        perc_contested_simple_possession, _ = self.get_position_stats_for(
            game, [f"{column_label}simple_possession"], "contested"
        )
        features[f"{filter_label}Percentage Contested With Possession"] = (
            perc_contested_simple_possession
        )
        perc_orange_simple_possession, _ = self.get_position_stats_for(
            game, [f"{column_label}simple_possession"], "orange"
        )
        features[f"{filter_label}Percentage Orange With Possession"] = (
            perc_orange_simple_possession
        )

        return features

    def create_game_features(
        self,
        main_player: str,
        player_names: List[str],
        game: pd.DataFrame,
        teams: Dict[str, List[str]],
        column_label: str = "",
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
        dependent_cols[f"{main_player}_{column_label}closest_to_ball"] = game[
            f"{main_player}_{column_label}distance_to_ball"
        ] == game[ball_distance_cols].min(axis=1)
        dependent_cols[f"{main_player}_{column_label}farthest_from_ball"] = game[
            f"{main_player}_{column_label}distance_to_ball"
        ] == game[ball_distance_cols].max(axis=1)

        # Get all teammates excluding the main player
        team = next(t for t, players in teams.items() if main_player in players)
        teammates = [p for p in teams[team] if p != main_player]

        # Collect distance columns for teammates + main player
        teammate_cols = [f"{column_label}{p}_distance_to_ball" for p in teammates] + [
            f"{main_player}_{column_label}distance_to_ball"
        ]

        # Rank distances among teammates (1 = closest, max = farthest)
        ranks = game[teammate_cols].rank(
            axis=1, method="min"
        )  # ascending rank: 1 = closest

        # Extract main player's rank
        main_rank = ranks[f"{main_player}_{column_label}distance_to_ball"]

        dependent_cols[f"{main_player}_{column_label}simple_first_man"] = main_rank == 1
        dependent_cols[f"{main_player}_{column_label}simple_third_man"] = (
            main_rank == len(teammates) + 1
        )
        dependent_cols[f"{main_player}_{column_label}simple_second_man"] = ~(
            dependent_cols[f"{main_player}_{column_label}simple_first_man"]
            | dependent_cols[f"{main_player}_{column_label}simple_third_man"]
        )

        # Wall checks, excluding goal occupancy
        dependent_cols[f"{main_player}_{column_label}on_back_wall"] = (
            game[f"{main_player}_{column_label}positioning_y"] <= -(Y_WALL - TOL)
        ) & (~game[f"{main_player}_{column_label}in_own_goal"])
        dependent_cols[f"{main_player}_{column_label}on_front_wall"] = (
            game[f"{main_player}_{column_label}positioning_y"] >= (Y_WALL - TOL)
        ) & (~game[f"{main_player}_{column_label}in_opponents_goal"])

        dependent_cols[f"{main_player}_{column_label}is_still"] = (
            game[f"{main_player}_{column_label}positioning_linear_velocity"] <= 10
        )
        dependent_cols[f"{main_player}_{column_label}is_slow"] = (
            game[f"{main_player}_{column_label}positioning_linear_velocity"] <= 500
        )
        dependent_cols[f"{main_player}_{column_label}is_semi_slow"] = (
            game[f"{main_player}_{column_label}positioning_linear_velocity"] > 500
        ) & (game[f"{main_player}_{column_label}positioning_linear_velocity"] <= 1000)
        dependent_cols[f"{main_player}_{column_label}is_medium_speed"] = (
            game[f"{main_player}_{column_label}positioning_linear_velocity"] > 1000
        ) & (game[f"{main_player}_{column_label}positioning_linear_velocity"] <= 1500)
        dependent_cols[f"{main_player}_{column_label}is_semi_fast"] = (
            game[f"{main_player}_{column_label}positioning_linear_velocity"] > 1500
        ) & (game[f"{main_player}_{column_label}positioning_linear_velocity"] <= 2000)
        dependent_cols[f"{main_player}_{column_label}is_fast"] = (
            game[f"{main_player}_{column_label}positioning_linear_velocity"] > 2000
        )

        dependent_cols[f"{main_player}_{column_label}is_drive_speed"] = (
            game[f"{main_player}_{column_label}positioning_linear_velocity"] <= 1410
        )
        dependent_cols[f"{main_player}_{column_label}is_boost_speed"] = (
            game[f"{main_player}_{column_label}positioning_linear_velocity"] > 1410
        ) & (game[f"{main_player}_{column_label}positioning_linear_velocity"] < 2200)
        dependent_cols[f"{main_player}_{column_label}is_supersonic"] = (
            game[f"{main_player}_{column_label}positioning_linear_velocity"] >= 2200
        )

        game = pd.concat(
            [game, pd.DataFrame(dependent_cols, index=game.index)],
            axis=1,
        )
        game.loc[:, f"{main_player}_{column_label}airborne"] = ~(
            game[f"{main_player}_{column_label}grounded"]
            | game[f"{main_player}_{column_label}on_ceiling"]
            | game[f"{main_player}_{column_label}on_left_wall"]
            | game[f"{main_player}_{column_label}on_right_wall"]
            | game[f"{main_player}_{column_label}on_back_wall"]
            | game[f"{main_player}_{column_label}on_front_wall"]
        )

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

        game.to_csv(os.path.join(FEATURES, "debug.csv"), index=False)
        return game

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
            main_player=main_player,
        )

        features = self.aggregate_features(
            game=game,
            main_player=main_player,
            teams=teams,
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
                                game_features,
                                features_path,
                                labels_path,
                                transposed=True,
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
