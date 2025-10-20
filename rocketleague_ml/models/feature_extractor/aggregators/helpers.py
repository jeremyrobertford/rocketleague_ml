import pandas as pd
from typing import List


def get_time_percentage_and_stint_for_cols(
    game: pd.DataFrame, cols: List[str], main_player: str
):
    total_time = game["delta"].sum()

    # build mask
    cols = [main_player + "_" + col for col in cols]
    mask = game[cols].all(axis=1)

    # percentage time
    time = game.loc[mask, "delta"].sum()
    percent_time = time / total_time if total_time > 0 else 0

    # find runs/stints
    run_ids = mask.ne(mask.shift()).cumsum()
    stint_groups = game[mask].groupby(run_ids)  # type: ignore

    # duration per stint
    stint_durations = stint_groups["delta"].sum()

    # average stint duration
    avg_stint = stint_durations.mean() if not stint_durations.empty else 0
    perc_avg_stint = avg_stint / total_time if total_time > 0 else 0

    return percent_time, perc_avg_stint


def get_time_and_stint_for_cols(game: pd.DataFrame, cols: List[str], main_player: str):
    if not all(col in game.columns for col in cols):
        return 0, 0

    # build mask
    cols = [main_player + "_" + col for col in cols]
    mask = game[cols].all(axis=1)

    # percentage time
    time = game.loc[mask, "delta"].sum()

    # find runs/stints
    run_ids = mask.ne(mask.shift()).cumsum()
    stint_groups = game[mask].groupby(run_ids)  # type: ignore

    # duration per stint
    stint_durations = stint_groups["delta"].sum()

    # average stint duration
    avg_stint = stint_durations.mean() if not stint_durations.empty else 0

    return time, avg_stint
