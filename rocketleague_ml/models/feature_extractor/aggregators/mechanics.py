import pandas as pd
from typing import Dict, cast, List
from .helpers import get_time_and_stint_for_cols


def aggregate_demos(game: pd.DataFrame, main_player: str) -> Dict[str, float]:
    features: Dict[str, float] = {}

    demo_col = f"{main_player}_demo"
    features[f"Average Demos"] = (
        (game[demo_col].count()) if demo_col in game.columns else 0
    )
    demoed_col = f"{main_player}_demoed"
    features[f"Average Demoed"] = (
        (game[demoed_col].count()) if demoed_col in game.columns else 0
    )
    return features


def aggregate_dodges(game: pd.DataFrame, main_player: str) -> Dict[str, float]:
    features: Dict[str, float] = {}

    dodge_col = f"{main_player}_dodge_torque_x"
    features[f"Average Dodges"] = (
        (game[dodge_col].count()) if dodge_col in game.columns else 0
    )
    return features


def aggregate_double_jumps(game: pd.DataFrame, main_player: str) -> Dict[str, float]:
    features: Dict[str, float] = {}

    double_jump_col = f"{main_player}_double_jump_x"
    features[f"Average Double Jumps"] = (
        (game[double_jump_col].count()) if double_jump_col in game.columns else 0
    )
    return features


def aggregate_flip_resets(game: pd.DataFrame, main_player: str) -> Dict[str, float]:
    features: Dict[str, float] = {}

    get_flip_back_col = f"{main_player}_component_usage_in_air_double_jump"
    flip_reset_mask = (game[get_flip_back_col] == 0) & (game[f"{main_player}_airborne"])
    features[f"Average Flip Resets"] = (
        cast(float, game[flip_reset_mask].count())
        if get_flip_back_col in game.columns
        else 0
    )
    return features


def aggregate_single_jumps(game: pd.DataFrame, main_player: str) -> Dict[str, float]:
    features: Dict[str, float] = {}

    single_jump_col = f"{main_player}_single_jump_x"
    features[f"Average Single Jumps"] = (
        game[single_jump_col].count() if single_jump_col in game.columns else 0
    )
    return features


def aggregate_mechanics(game: pd.DataFrame, main_player: str) -> Dict[str, float]:
    dodge_features = aggregate_dodges(game, main_player)
    demo_features = aggregate_demos(game, main_player)
    double_jump_features = aggregate_double_jumps(game, main_player)
    single_jump_features = aggregate_single_jumps(game, main_player)
    flip_reset_features = aggregate_flip_resets(game, main_player)

    features: Dict[str, float] = {}
    movement_columns: Dict[str, List[str]] = {"Drifting": ["drift_active"]}
    for label, cols in movement_columns.items():
        perc, avg_stint = get_time_and_stint_for_cols(
            game=game, cols=cols, main_player=main_player
        )
        features[f"Time while {label}"] = perc
        features[f"Average Stint while {label}"] = avg_stint

    return (
        features
        | dodge_features
        | demo_features
        | double_jump_features
        | single_jump_features
        | flip_reset_features
    )
