import numpy as np
import pandas as pd
from typing import Dict
from rocketleague_ml.models.feature_extractor.aggregators.helpers import (
    get_time_percentage_and_stint_for_cols,
)


def aggregate_boost_usage(game: pd.DataFrame, main_player: str) -> Dict[str, float]:
    features: Dict[str, float] = {}

    boost_usage_columns = {
        "While Boosting": ["boost_active"],
        "With No Boost": ["no_boost"],
        "With <=25 Boost": ["1st_quad_boost"],
        "With >25 and <= 50 Boost": ["2nd_quad_boost"],
        "With >50 and <= 75 BoostBoosting": ["3rd_quad_boost"],
        "With >75 and <= 100 Boost": ["4th_quad_boost"],
        "With Full Boost": ["full_boost"],
    }

    for label, cols in boost_usage_columns.items():
        percent_time, average_percent_stint = get_time_percentage_and_stint_for_cols(
            game, cols, main_player
        )
        features[f"Percent Time while {label}"] = percent_time
        features[f"Average Stint while {label}"] = average_percent_stint

    features[f"Average Boost Amount"] = game[f"{main_player}_boost_amount"].mean() / 100

    # --- Pickup masks ---
    boost_amount_col = f"{main_player}_boost_amount"
    boost_pickup_col = f"{main_player}_boost_pickup_amount"
    boost_y_col = f"{main_player}_boost_pickup_y"

    # pick-ups that actually gave usable boost (not overfill)
    usable_pickup = game[boost_pickup_col].copy()
    usable_pickup[game[boost_amount_col] == 100] = 0  # ignore overfill

    # big vs small
    big_mask = usable_pickup >= 100
    small_mask = (usable_pickup > 0) & (usable_pickup < 100)

    features["Average Boost Grabbed"] = usable_pickup[usable_pickup > 0].mean() / 100
    features["Average Big Boost Grabbed"] = usable_pickup[big_mask].mean() / 100
    features["Average Small Boost Grabbed"] = usable_pickup[small_mask].mean() / 100

    # overfill
    overfill_mask = (game[boost_amount_col] == 100) & (game[boost_pickup_col] > 0)
    features["Average Overfill"] = (
        game.loc[overfill_mask, boost_pickup_col].mean() / 100
    )

    # --- Ensure team column is filled ---
    team_col = f"{main_player}_team"
    if game[team_col].isna().any():
        game[team_col] = game[team_col].ffill().bfill()  # fill missing team values
    player_team = game[team_col].iloc[0]

    # --- Simple stolen: pickups in opponent half ---
    if player_team == "Blue":
        opponent_half_mask = game[boost_y_col] > 0
    else:
        opponent_half_mask = game[boost_y_col] < 0

    simple_stolen = usable_pickup[opponent_half_mask]
    features["Average Boost Simple Stolen"] = simple_stolen.mean() / 100
    features["Average Big Boost Simple Stolen"] = (
        simple_stolen[simple_stolen >= 100].mean() / 100
    )
    features["Average Small Boost Simple Stolen"] = (
        simple_stolen[(simple_stolen > 0) & (simple_stolen < 100)].mean() / 100
    )
    features["Average Boost Simple Stolen Overfill"] = (
        game.loc[overfill_mask & opponent_half_mask, boost_pickup_col].mean() / 100
    )

    # --- Simple efficiency (ignore overfill) ---
    boost_used: pd.Series[float] = game[boost_amount_col].diff().clip(lower=0)  # type: ignore
    total_used = boost_used.sum()
    total_gained = usable_pickup.sum()

    simple_efficiency = total_gained / (total_used + 1e-6)  # avoid div by zero
    features["Simple Boost Efficiency"] = simple_efficiency

    # --- Penalties ---
    not_airborne = ~game[f"{main_player}_is_airborne"]
    supersonic_mask = (game[f"{main_player}_is_supersonic"]) & (not_airborne)
    supersonic_penalty = 0.9
    supersonic_efficiency = simple_efficiency * np.prod(
        np.where(supersonic_mask, supersonic_penalty, 1)
    )
    features["Supersonic Speed Boost Efficiency"] = supersonic_efficiency

    drive_to_boost_mask = (
        game[f"{main_player}_is_drive_speed"].shift(1)
        & game[f"{main_player}_is_boost_speed"]
        & not_airborne
    ) & (boost_used > 20)
    drive_to_boost_penalty = 0.9
    drive_to_boost_speed_efficiency = simple_efficiency * np.prod(
        np.where(drive_to_boost_mask, drive_to_boost_penalty, 1)
    )
    features["Drive to Boost Speed Boost Efficiency"] = drive_to_boost_speed_efficiency

    # if you have ball distance
    ball_distance = game[f"{main_player}_ball_distance"]
    far_mask = ball_distance > 3000
    far_distance_penalty = 0.95
    far_distance_efficiency = simple_efficiency * np.prod(
        np.where(far_mask, far_distance_penalty, 1)
    )
    features["Far From Ball Boost Efficiency"] = far_distance_efficiency

    features["Boost Efficiency"] = (
        simple_efficiency
        * np.prod(np.where(supersonic_mask, supersonic_penalty, 1))
        * np.prod(np.where(drive_to_boost_mask, drive_to_boost_penalty, 1))
        * np.prod(np.where(far_mask, far_distance_penalty, 1))
    )

    return features
