import numpy as np
import pandas as pd
from typing import Dict, List
from rocketleague_ml.models.feature_extractor.aggregators.helpers import (
    get_time_and_stint_for_cols,
)


def aggregate_player_rotations(
    game: pd.DataFrame, main_player: str
) -> Dict[str, float]:
    features: Dict[str, float] = {}

    player_rotation_columns = {
        "Simple First Man": ["simple_player_rotation_first_man"],
        "Simple Second Man": ["simple_player_rotation_second_man"],
        "Simple Third Man": ["simple_player_rotation_third_man"],
        "First Man": ["player_rotation_first_man"],
        "Second Man": ["player_rotation_second_man"],
        "Third Man": ["player_rotation_third_man"],
    }

    for label, cols in player_rotation_columns.items():
        percent_time, average_percent_stint = get_time_and_stint_for_cols(
            game, cols, main_player
        )
        features[f"Time while {label}"] = percent_time
        features[f"Average Stint while {label}"] = average_percent_stint

    player_rotation_columns = {
        "Simple": f"{main_player}_simple_player_rotation_position",
        "Full": f"{main_player}_player_rotation_position",
    }
    for label, col in player_rotation_columns.items():
        rotation_col = game[col].astype("Int64")
        delta_t = game["delta"].to_numpy()

        # find frame-to-frame transitions (ignore NaN)
        valid = rotation_col.notna()
        roles = rotation_col[valid].to_numpy()
        deltas = delta_t[valid]

        # mark transitions where role changes
        changes = roles[1:] != roles[:-1]
        change_indices = np.where(changes)[0]

        if len(change_indices) > 0:
            # average time between changes
            time_between_changes: List[float] = []
            last_change_idx = 0
            for idx in change_indices:
                stint_time = deltas[last_change_idx : idx + 1].sum()
                time_between_changes.append(stint_time)
                last_change_idx = idx + 1

            avg_rotation_speed: float = np.mean(time_between_changes)  # type: ignore
        else:
            avg_rotation_speed = np.nan

        features[f"{label} Rotation Speed"] = avg_rotation_speed

        # Compute directional rotation counts
        from_to_counts = {(i, j): 0 for i in (1, 2, 3) for j in (1, 2, 3) if i != j}
        total_from_counts = {i: 0 for i in (1, 2, 3)}

        for i in range(len(roles) - 1):
            if roles[i] != roles[i + 1]:
                from_to_counts[(roles[i], roles[i + 1])] += 1
                total_from_counts[roles[i]] += 1

        for (from_role, to_role), count in from_to_counts.items():
            features[f"Count Rotating From {label} {from_role} to {to_role}"] = count

    return features
