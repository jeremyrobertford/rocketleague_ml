import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union
from rocketleague_ml.models.feature_extractor.aggregators.helpers import (
    get_time_and_stint_for_cols,
)

mask_type = Union[pd.Series, np.ndarray[Tuple[int], np.dtype[np.bool_]]]
masks_type = Dict[str, mask_type]
active_array_type = Union[pd.Series, np.ndarray[Tuple[int], np.dtype[np.int_]]]


def calculate_efficiency_over_time(
    masks: masks_type,
    boost_active: active_array_type,
    fps: float = 30.0,
) -> Dict[str, np.float64]:
    """
    Calculate boost efficiency penalties based on how long boost is used
    in specific states (supersonic, drive_to_boost, far).

    Only penalizes frames where boost is actually being used.
    """

    # --- Helper to count seconds ---
    def count_seconds(
        mask: mask_type,
    ) -> float:
        active_mask = pd.Series(mask & boost_active)  # only count when using boost
        return active_mask.sum() / fps

    def feature_boost_efficiency_over_time(
        mask: mask_type,
        penalty: float,
    ) -> np.float64:
        """
        Returns fraction of simple efficiency after applying penalty over time of boost usage.
        If the mask never applies while boost_active, return 1.0 (no change).
        """
        seconds_in_state = count_seconds(mask)
        if seconds_in_state == 0:
            return np.float64(1.0)
        return np.float64(penalty**seconds_in_state)

    # Convert to pandas Series for consistent operations
    boost_active = pd.Series(boost_active)

    # --- Penalties ---
    supersonic_penalty = 0.9
    drive_to_boost_penalty = 0.9
    far_penalty = 0.95

    # --- Combine ---
    supersonic_efficiency = feature_boost_efficiency_over_time(
        mask=masks["supersonic"], penalty=supersonic_penalty
    )
    drive_to_boost_efficiency = feature_boost_efficiency_over_time(
        mask=masks["drive_to_boost"],
        penalty=drive_to_boost_penalty,
    )
    far_efficiency = feature_boost_efficiency_over_time(
        mask=masks["far"], penalty=far_penalty
    )
    boost_efficiency = supersonic_efficiency * drive_to_boost_penalty * far_efficiency
    return {
        "Supersonic Boost": supersonic_efficiency,
        "Drive to Boost Speed Boost": drive_to_boost_efficiency,
        "Far from Ball Boost": far_efficiency,
        "Boost": boost_efficiency,
    }


def calculate_efficiency_over_events(
    masks: masks_type,
    boost_active: active_array_type,
) -> Dict[str, np.float64]:
    """
    Calculate boost efficiency penalties based on event counts,
    but only during frames where boost is actually being used.

    Parameters
    ----------
    simple_efficiency : np.float64
        Base boost efficiency (used/collected ratio).
    masks : dict[str, array-like]
        Dictionary with boolean masks for conditions:
            - "supersonic"
            - "drive_to_boost"
            - "far"
    boost_active : array-like of bool
        Boolean array indicating when boost is actually being consumed.
    """

    # --- Helper to count event starts (rising edges) ---
    def count_events(
        mask: Union[pd.Series, np.ndarray[Tuple[int], np.dtype[np.bool_]]],
    ) -> int:
        mask = pd.Series(mask & boost_active)  # only count when using boost
        return (mask.astype(int).diff() == 1).sum()

    def feature_boost_efficiency_over_events(
        mask: mask_type,
        penalty: float,
    ) -> np.float64:
        """
        Returns fraction of simple efficiency after applying penalty over time of boost usage.
        If the mask never applies while boost_active, return 1.0 (no change).
        """
        events = count_events(mask)
        if events == 0:
            return np.float64(1.0)
        return np.float64(penalty**events)

    # --- Penalties ---
    supersonic_penalty = 0.9
    drive_to_boost_penalty = 0.9
    far_penalty = 0.95

    # --- Combine ---
    supersonic_efficiency = feature_boost_efficiency_over_events(
        mask=masks["supersonic"], penalty=supersonic_penalty
    )
    drive_to_boost_efficiency = feature_boost_efficiency_over_events(
        mask=masks["drive_to_boost"],
        penalty=drive_to_boost_penalty,
    )
    far_efficiency = feature_boost_efficiency_over_events(
        mask=masks["far"], penalty=far_penalty
    )
    boost_efficiency = supersonic_efficiency * drive_to_boost_penalty * far_efficiency
    return {
        "Supersonic Boost": supersonic_efficiency,
        "Drive to Boost Speed Boost": drive_to_boost_efficiency,
        "Far from Ball Boost": far_efficiency,
        "Boost": boost_efficiency,
    }


def aggregate_boost_usage(game: pd.DataFrame, main_player: str) -> Dict[str, float]:
    features: Dict[str, float] = {}

    boost_amount_col = f"{main_player}_boost_amount"
    boost_active_col = f"{main_player}_boost_active"
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
        percent_time, average_percent_stint = get_time_and_stint_for_cols(
            game, cols, main_player
        )
        features[f"Time {label}"] = percent_time
        features[f"Average Stint {label}"] = average_percent_stint

    avg_boost_amount = game[boost_amount_col].mean()
    features[f"Average Boost Amount"] = avg_boost_amount / 100

    # --- Ensure team column is filled ---
    team_col = f"{main_player}_team"
    player_team = game[team_col].iloc[-1]

    # --- Overfill-aware usable pickup ---
    boost_amount = game[boost_amount_col].to_numpy()

    # Find index of the first non-nan
    first_valid = (
        np.argmax(~np.isnan(boost_amount))
        if np.any(~np.isnan(boost_amount))
        else len(boost_amount)
    )

    # Fill leading nans with 33
    boost_amount[:first_valid] = 33

    # Optionally fill *other* nans with 0
    boost_amount = np.nan_to_num(boost_amount, nan=0)
    actual_boost_amount = np.copy(boost_amount).astype(float)
    boost_active = game[boost_active_col].fillna(0).to_numpy().astype(int)  # type: ignore
    for i in range(1, len(actual_boost_amount)):
        if boost_active[i]:
            actual_boost_amount[i] = max(0.0, actual_boost_amount[i - 1] - 1.15)
        else:
            # if current boost value in data is higher (pickup etc.), keep it
            actual_boost_amount[i] = max(
                actual_boost_amount[i], actual_boost_amount[i - 1]
            )

    # floor and convert to int
    actual_boost_amount = np.floor(actual_boost_amount).astype(int)

    boost_pickup = game[boost_pickup_col].to_numpy()
    prev_boost = np.roll(boost_amount, 1)
    prev_boost[0] = boost_amount[0]
    missing_boost = 100 - prev_boost
    usable_pickup = np.minimum(boost_pickup, missing_boost)
    usable_pickup = usable_pickup[~np.isnan(usable_pickup)]

    # Big vs small pickups
    boost_grabbed_mask = boost_pickup > 0
    big_mask = boost_pickup == 100
    small_mask = boost_pickup == 12

    boost_grabbed = boost_pickup[boost_grabbed_mask]
    big_boost_grabbed = boost_pickup[big_mask]
    small_boost_grabbed = boost_pickup[small_mask]
    features["Average Boost Grabbed"] = boost_pickup.mean() if len(boost_pickup) else 0
    features["Total Boost Grabbed"] = len(boost_grabbed)
    features["Big Boost Grabbed"] = len(big_boost_grabbed)
    features["Small Boost Grabbed"] = len(small_boost_grabbed)

    # Overfill
    overfill_mask = (boost_amount == 100) & (boost_pickup > 0)
    overfill = boost_pickup[overfill_mask]
    features["Average Overfill"] = overfill.mean() / 100 if len(overfill) else 0

    # --- Simple stolen: pickups in opponent half ---
    if player_team == "Blue":
        opponent_half_mask = game[boost_y_col] > 0
    else:
        opponent_half_mask = game[boost_y_col] < 0

    simple_stolen = boost_pickup[opponent_half_mask & boost_grabbed_mask]
    simple_big_stolen = simple_stolen[simple_stolen == 100]
    simple_small_stolen = simple_stolen[simple_stolen == 12]
    features["Average Boost Simple Stolen"] = (
        simple_stolen.mean() if len(simple_stolen) else 0
    )
    features["Boost Simple Stolen"] = len(simple_stolen)
    features["Big Boost Simple Stolen"] = len(simple_big_stolen)
    features["Small Boost Simple Stolen"] = len(simple_small_stolen)
    simple_stolen_overfill = boost_pickup[overfill_mask & opponent_half_mask]
    features["Average Boost Simple Stolen Overfill"] = (
        simple_stolen_overfill.mean() if len(simple_stolen_overfill) else 0
    )

    # --- Simple Boost Efficiency (overfill-aware) ---
    boost_diff = np.diff(actual_boost_amount, prepend=boost_amount[33])
    boost_used = np.array([-bu if bu < 0 else 0 for bu in boost_diff])

    # --- Masks ---
    not_airborne = ~game[f"{main_player}_airborne"]
    supersonic_mask = (game[f"{main_player}_is_supersonic"]) & not_airborne
    drive_to_boost_mask = (
        game[f"{main_player}_is_drive_speed"].shift(1, fill_value=False)
        & game[f"{main_player}_is_boost_speed"]
        & not_airborne
    ) & (boost_used > 20)
    far_mask = (game[f"{main_player}_is_semi_far_from_ball"]) & (
        game[f"ball_positioning_linear_velocity"] < 1500
    )
    masks: Dict[str, pd.Series | np.ndarray[Tuple[int], np.dtype[np.bool_]]] = {
        "supersonic": supersonic_mask,
        "drive_to_boost": drive_to_boost_mask,
        "far": far_mask,
    }

    # efficiencies = calculate_efficiency_over_events(
    #     masks=masks,
    #     boost_active=boost_active,
    # )
    efficiencies = calculate_efficiency_over_time(
        masks=masks,
        boost_active=boost_active,
    )
    for efficiency_type, efficiency in efficiencies.items():
        features[efficiency_type + " Efficiency"] = efficiency

    return features
