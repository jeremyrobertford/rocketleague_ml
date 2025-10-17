import pandas as pd
from typing import Dict, cast


def aggregate_demos(game: pd.DataFrame, main_player: str) -> Dict[str, float]:
    features: Dict[str, float] = {}

    demo_col = f"{main_player}_demo"
    features[f"Average Demos"] = (
        (game[demo_col].count() / 30) if demo_col in game.columns else 0
    )
    demoed_col = f"{main_player}_demoed"
    features[f"Average Demoed"] = (
        (game[demoed_col].count() / 30) if demoed_col in game.columns else 0
    )
    return features


def aggregate_dodges(game: pd.DataFrame, main_player: str) -> Dict[str, float]:
    features: Dict[str, float] = {}

    dodge_col = f"{main_player}_dodge_torque_x"
    features[f"Average Dodges"] = (
        (game[dodge_col].count() / 300) if dodge_col in game.columns else 0
    )
    return features


def aggregate_double_jumps(game: pd.DataFrame, main_player: str) -> Dict[str, float]:
    features: Dict[str, float] = {}

    double_jump_col = f"{main_player}_double_jump_x"
    features[f"Average Double Jumps"] = (
        (game[double_jump_col].count() / 300) if double_jump_col in game.columns else 0
    )
    return features


def aggregate_flip_resets(game: pd.DataFrame, main_player: str) -> Dict[str, float]:
    features: Dict[str, float] = {}

    get_flip_back_col = f"{main_player}_component_usage_in_air_double_jump"
    flip_reset_mask = (game[get_flip_back_col] == 0) & (game[f"{main_player}_airborne"])
    features[f"Average Flip Resets"] = (
        cast(float, game[flip_reset_mask].count() / 300)
        if get_flip_back_col in game.columns
        else 0
    )
    return features


def aggregate_single_jumps(game: pd.DataFrame, main_player: str) -> Dict[str, float]:
    features: Dict[str, float] = {}

    single_jump_col = f"{main_player}_single_jump_x"
    features[f"Average Single Jumps"] = (
        (game[single_jump_col].count() / 300) if single_jump_col in game.columns else 0
    )
    return features


def aggregate_mechanics(game: pd.DataFrame, main_player: str) -> Dict[str, float]:
    dodge_featues = aggregate_dodges(game, main_player)
    demo_featues = aggregate_dodges(game, main_player)
    return dodge_featues | demo_featues
