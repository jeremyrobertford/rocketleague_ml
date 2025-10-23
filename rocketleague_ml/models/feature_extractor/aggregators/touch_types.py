# type: ignore
import pandas as pd
from typing import Dict


def aggregate_touch_types(game: pd.DataFrame, main_player: str) -> Dict[str, float]:
    features: Dict[str, float] = {}

    # Masks
    total_mask = game[f"{main_player}_touch_total"].astype(bool)
    fifty_fifty_mask = game[f"{main_player}_touch_fifty_fifty"].astype(bool)
    non_5050_mask = total_mask & ~fifty_fifty_mask

    # ---- Non-50/50 touches ----
    features["Touches"] = non_5050_mask.sum()
    features["Touches Towards Goal"] = (
        game[f"{main_player}_touch_towards_goal"] & non_5050_mask
    ).sum()
    features["Touches Towards Teammate"] = (
        game[f"{main_player}_touch_towards_teammate"] & non_5050_mask
    ).sum()
    features["Touches Towards Opponent"] = (
        game[f"{main_player}_touch_towards_opponent"] & non_5050_mask
    ).sum()
    features["Touches Towards Open Space"] = (
        game[f"{main_player}_touch_open_space"] & non_5050_mask
    ).sum()

    # Hit strength stats (non-50/50)
    hit_strength = game[f"{main_player}_touch_strength"]
    features["Average Hit Strength"] = (
        hit_strength[non_5050_mask].mean() if non_5050_mask.any() else 0
    )
    features["Variance Hit Strength"] = (
        hit_strength[non_5050_mask].var() if non_5050_mask.any() else 0.0
    )

    # Alignment stats
    closest = game[f"{main_player}_hit_strength_alignment_closest_opp"]
    avg_opp = game[f"{main_player}_hit_strength_alignment_avg_opp"]
    features["Average Hit Strength Alignment Closest Opp"] = (
        closest[non_5050_mask].mean() if non_5050_mask.any() else 0
    )
    features["Variance Hit Strength Alignment Closest Opp"] = (
        closest[non_5050_mask].var() if non_5050_mask.any() else 0
    )
    features["Average Hit Strength Alignment Avg Opp"] = (
        avg_opp[non_5050_mask].mean() if non_5050_mask.any() else 0
    )
    features["Variance Hit Strength Alignment Avg Opp"] = (
        avg_opp[non_5050_mask].var() if non_5050_mask.any() else 0
    )

    # ---- 50/50 touches ----
    features["50-50 Touches"] = fifty_fifty_mask.sum()
    features["50-50 Touches Towards Goal"] = (
        game[f"{main_player}_touch_towards_goal"] & fifty_fifty_mask
    ).sum()
    features["50-50 Touches Towards Teammate"] = (
        game[f"{main_player}_touch_towards_teammate"] & fifty_fifty_mask
    ).sum()
    features["50-50 Touches Towards Opponent"] = (
        game[f"{main_player}_touch_towards_opponent"] & fifty_fifty_mask
    ).sum()
    features["50-50 Touches Towards Open Space"] = (
        game[f"{main_player}_touch_open_space"] & fifty_fifty_mask
    ).sum()

    # Hit strength stats (50/50)
    features["50-50 Average Hit Strength"] = (
        hit_strength[fifty_fifty_mask].mean() if fifty_fifty_mask.any() else 0
    )
    features["50-50 Variance Hit Strength"] = (
        hit_strength[fifty_fifty_mask].var() if fifty_fifty_mask.any() else 0
    )

    # Alignment stats (50/50)
    features["50-50 Average Hit Strength Alignment Closest Opp"] = (
        closest[fifty_fifty_mask].mean() if fifty_fifty_mask.any() else 0
    )
    features["50-50 Variance Hit Strength Alignment Closest Opp"] = (
        closest[fifty_fifty_mask].var() if fifty_fifty_mask.any() else 0
    )
    features["50-50 Average Hit Strength Alignment Avg Opp"] = (
        avg_opp[fifty_fifty_mask].mean() if fifty_fifty_mask.any() else 0
    )
    features["50-50 Variance Hit Strength Alignment Avg Opp"] = (
        avg_opp[fifty_fifty_mask].var() if fifty_fifty_mask.any() else 0
    )

    return features
