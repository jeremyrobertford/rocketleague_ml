import numpy as np
import pandas as pd
from typing import Dict
from rocketleague_ml.models.feature_extractor.aggregators.helpers import (
    get_time_percentage_and_stint_for_cols,
)


def aggregate_boost_usage(game: pd.DataFrame, main_player: str) -> Dict[str, float]:
    features: Dict[str, float] = {}

    boost_amount_col = f"{main_player}_boost_amount"
    boost_pickup_col = f"{main_player}_boost_pickup_amount"
    boost_y_col = f"{main_player}_boost_pickup_y"

    if boost_pickup_col not in game.columns:
        game[boost_pickup_col] = np.nan

    # --- Boost usage percentages ---
    boost_usage_columns = {
        "While Boosting": ["boost_active"],
        "With No Boost": ["no_boost"],
        "With >0 and <=25 Boost": ["1st_quad_boost"],
        "With >25 and <= 50 Boost": ["2nd_quad_boost"],
        "With >50 and <= 75 Boost": ["3rd_quad_boost"],
        "With >75 and < 100 Boost": ["4th_quad_boost"],
        "With Full Boost": ["full_boost"],
    }

    for label, cols in boost_usage_columns.items():
        percent_time, average_percent_stint = get_time_percentage_and_stint_for_cols(
            game, cols, main_player
        )
        features[f"Percent Time {label}"] = percent_time
        features[f"Average Stint {label}"] = average_percent_stint

    avg_boost_amount = game[boost_amount_col].mean()
    features[f"Average Boost Amount"] = avg_boost_amount / 100

    # --- Ensure team column is filled ---
    team_col = f"{main_player}_team"
    player_team = game[team_col].iloc[-1]

    # --- Overfill-aware usable pickup ---
    boost_amount = game[boost_amount_col].to_numpy()
    boost_pickup = game[boost_pickup_col].to_numpy()
    prev_boost = np.roll(boost_amount, 1)
    prev_boost[0] = boost_amount[0]
    missing_boost = 100 - prev_boost
    usable_pickup = np.minimum(boost_pickup, missing_boost)

    # Big vs small pickups
    big_mask = boost_pickup == 100
    small_mask = boost_pickup == 12

    big_boost_grabbed = boost_pickup[big_mask]
    small_boost_grabbed = boost_pickup[small_mask]
    features["Average Boost Grabbed"] = (
        boost_pickup.mean() / 100 if len(boost_pickup) else 0
    )
    features["Percentage Big Boost Grabbed"] = len(big_boost_grabbed) / len(
        boost_pickup[boost_pickup > 0]
    )
    features["Percentage Small Boost Grabbed"] = len(small_boost_grabbed) / len(
        boost_pickup[boost_pickup > 0]
    )

    # Overfill
    overfill_mask = (boost_amount == 100) & (boost_pickup > 0)
    overfill = boost_pickup[overfill_mask]
    features["Average Overfill"] = overfill.mean() / 100 if len(overfill) else 0

    # --- Simple stolen: pickups in opponent half ---
    if player_team == "Blue":
        opponent_half_mask = game[boost_y_col] > 0
    else:
        opponent_half_mask = game[boost_y_col] < 0

    simple_stolen = boost_pickup[opponent_half_mask]
    simple_big_stolen = simple_stolen[simple_stolen == 100]
    simple_small_stolen = simple_stolen[simple_stolen == 12]
    features["Average Boost Simple Stolen"] = (
        simple_stolen.mean() / 100 if len(simple_stolen) else 0
    )
    features["Percentage Big Boost Simple Stolen"] = len(simple_big_stolen) / len(
        simple_stolen[simple_stolen > 0]
    )
    features["Percentage Small Boost Simple Stolen"] = len(simple_small_stolen) / len(
        simple_stolen[simple_stolen > 0]
    )
    simple_stolen_overfill = boost_pickup[overfill_mask & opponent_half_mask]
    features["Average Boost Simple Stolen Overfill"] = (
        simple_stolen_overfill.mean() / 100 if len(simple_stolen_overfill) else 0
    )

    # --- Simple Boost Efficiency (overfill-aware) ---
    boost_diff = np.diff(boost_amount, prepend=boost_amount[0])
    boost_used = -boost_diff
    total_used = boost_used.sum()
    total_gained = usable_pickup.sum()

    if total_gained == 0:
        features["Simple Boost Efficiency"] = np.nan
        features["Supersonic Speed Boost Efficiency"] = np.nan
        features["Drive to Boost Speed Boost Efficiency"] = np.nan
        features["Far From Ball Boost Efficiency"] = np.nan
        features["Boost Efficiency"] = np.nan
        return features

    simple_efficiency = total_used / total_gained
    if simple_efficiency < 0 or simple_efficiency > 1:
        pass
    features["Simple Boost Efficiency"] = simple_efficiency

    # --- Penalties ---
    not_airborne = ~game[f"{main_player}_airborne"]
    supersonic_mask = (game[f"{main_player}_is_supersonic"]) & not_airborne
    supersonic_penalty = 0.9
    supersonic_efficiency = simple_efficiency * np.prod(
        np.where(supersonic_mask, supersonic_penalty, 1)
    )
    features["Supersonic Speed Boost Efficiency"] = supersonic_efficiency

    drive_to_boost_mask = (
        game[f"{main_player}_is_drive_speed"].shift(1, fill_value=False)
        & game[f"{main_player}_is_boost_speed"]
        & not_airborne
    ) & (boost_used > 20)
    drive_to_boost_penalty = 0.9
    drive_to_boost_speed_efficiency = simple_efficiency * np.prod(
        np.where(drive_to_boost_mask, drive_to_boost_penalty, 1)
    )
    features["Drive to Boost Speed Boost Efficiency"] = drive_to_boost_speed_efficiency

    ball_distance = game[f"{main_player}_distance_to_ball"]
    far_mask = ball_distance > 3000
    far_distance_penalty = 0.95
    far_distance_efficiency = simple_efficiency * np.prod(
        np.where(far_mask, far_distance_penalty, 1)
    )
    features["Far From Ball Boost Efficiency"] = far_distance_efficiency

    # Combined Boost Efficiency
    features["Boost Efficiency"] = (
        simple_efficiency
        * np.prod(np.where(supersonic_mask, supersonic_penalty, 1))
        * np.prod(np.where(drive_to_boost_mask, drive_to_boost_penalty, 1))
        * np.prod(np.where(far_mask, far_distance_penalty, 1))
    )

    return features
