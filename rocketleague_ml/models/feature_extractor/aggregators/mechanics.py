import pandas as pd
from typing import Dict


def aggregate_mechanics(game: pd.DataFrame, main_player: str) -> Dict[str, float]:
    features: Dict[str, float] = {}

    dodge_col = f"{main_player}_dodge_torque_x"
    features[f"Average Dodges"] = (
        (game[dodge_col].count() / 300) if dodge_col in game.columns else 0
    )
    return features
