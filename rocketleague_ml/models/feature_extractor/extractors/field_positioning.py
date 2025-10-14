import pandas as pd
from typing import Dict, List
from rocketleague_ml.config import (
    Y_WALL,
    GOAL_DEPTH,
    Z_CEILING,
    X_WALL,
    X_GOAL,
    TOL,
    Z_GROUND,
)


def get_field_positioning_cols(game: pd.DataFrame, teams: Dict[str, List[str]]):
    field_positioning_cols: Dict[str, pd.Series] = {}
    for players in teams.values():
        for p in players:
            max_y = Y_WALL * 2
            third_of_y = max_y / 3
            half_of_z = Z_CEILING / 2
            third_of_z = Z_CEILING / 3

            field_positioning_cols[f"{p}_highest_half"] = (
                game[f"{p}_positioning_z"] >= half_of_z
            )
            field_positioning_cols[f"{p}_lowest_half"] = (
                game[f"{p}_positioning_z"] < half_of_z
            )
            field_positioning_cols[f"{p}_highest_third"] = (
                game[f"{p}_positioning_z"] >= third_of_z * 2
            )
            field_positioning_cols[f"{p}_middle_aerial_third"] = (
                game[f"{p}_positioning_z"] > third_of_z
            ) & (game[f"{p}_positioning_z"] < third_of_z * 2)
            field_positioning_cols[f"{p}_lowest_third"] = (
                game[f"{p}_positioning_z"] <= third_of_z
            )

            field_positioning_cols[f"{p}_offensive_half"] = (
                game[f"{p}_positioning_y"] > 0
            )
            field_positioning_cols[f"{p}_defensive_half"] = (
                game[f"{p}_positioning_y"] <= 0
            )
            field_positioning_cols[f"{p}_offensive_third"] = (
                game[f"{p}_positioning_y"] >= third_of_y
            )
            field_positioning_cols[f"{p}_neutral_third"] = (
                game[f"{p}_positioning_y"] < third_of_y
            ) & (game[f"{p}_positioning_y"] > -third_of_y)
            field_positioning_cols[f"{p}_defensive_third"] = (
                game[f"{p}_positioning_y"] <= -third_of_y
            )

            field_positioning_cols[f"{p}_left_half"] = game[f"{p}_positioning_x"] > 0
            field_positioning_cols[f"{p}_right_half"] = game[f"{p}_positioning_x"] <= 0
            field_positioning_cols[f"{p}_left_third"] = (
                game[f"{p}_positioning_x"] >= third_of_y
            )
            field_positioning_cols[f"{p}_middle_third"] = (
                game[f"{p}_positioning_x"] < third_of_y
            ) & (game[f"{p}_positioning_x"] > -third_of_y)
            field_positioning_cols[f"{p}_right_third"] = (
                game[f"{p}_positioning_x"] <= -third_of_y
            )

            field_positioning_cols[f"{p}_grounded"] = game[f"{p}_positioning_z"].between(0, Z_GROUND)  # type: ignore
            field_positioning_cols[f"{p}_on_ceiling"] = (
                game[f"{p}_positioning_z"] >= Z_CEILING
            )

            field_positioning_cols[f"{p}_on_left_wall"] = (
                game[f"{p}_positioning_x"] <= -X_WALL + TOL
            )
            field_positioning_cols[f"{p}_on_right_wall"] = (
                game[f"{p}_positioning_x"] >= X_WALL - TOL
            )

            # Ball
            field_positioning_cols[f"{p}_in_front_of_ball"] = (
                game[f"{p}_positioning_y"] > game["ball_positioning_y"]
            )
            field_positioning_cols[f"{p}_behind_ball"] = (
                game[f"{p}_positioning_y"] < game["ball_positioning_y"]
            )

            # Goal regions
            in_own_goal = (
                (game[f"{p}_positioning_x"].between(-X_GOAL, X_GOAL))  # type: ignore
                & (game[f"{p}_positioning_y"] <= -(Y_WALL - TOL))
                & (game[f"{p}_positioning_y"] >= -(Y_WALL + GOAL_DEPTH + TOL))
            )
            field_positioning_cols[f"{p}_in_own_goal"] = in_own_goal
            in_opponents_goal = (
                (game[f"{p}_positioning_x"].between(-X_GOAL, X_GOAL))  # type: ignore
                & (game[f"{p}_positioning_y"] >= (Y_WALL - TOL))
                & (game[f"{p}_positioning_y"] <= (Y_WALL + GOAL_DEPTH + TOL))
            )
            field_positioning_cols[f"{p}_in_opponents_goal"] = in_opponents_goal

            field_positioning_cols[f"{p}_on_back_wall"] = (
                game[f"{p}_positioning_y"] <= -(Y_WALL - TOL)
            ) & (~in_own_goal)
            field_positioning_cols[f"{p}_on_front_wall"] = (
                game[f"{p}_positioning_y"] >= (Y_WALL - TOL)
            ) & (~in_opponents_goal)

    return field_positioning_cols


def get_dependent_field_positioning_cols(
    game: pd.DataFrame, teams: Dict[str, List[str]]
):
    dependent_field_positioning_cols: Dict[str, pd.Series] = {}
    for players in teams.values():
        for p in players:
            dependent_field_positioning_cols[f"{p}_airborne"] = ~(
                game[f"{p}_grounded"]
                | game[f"{p}_on_ceiling"]
                | game[f"{p}_on_left_wall"]
                | game[f"{p}_on_right_wall"]
                | game[f"{p}_on_back_wall"]
                | game[f"{p}_on_front_wall"]
            )

    return dependent_field_positioning_cols
