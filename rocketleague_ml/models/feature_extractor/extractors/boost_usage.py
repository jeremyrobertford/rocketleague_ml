import pandas as pd
from typing import Dict, List


def get_boost_usage_cols(game: pd.DataFrame, teams: Dict[str, List[str]]):
    boost_usage_cols: Dict[str, pd.Series] = {}
    for players in teams.values():
        for p in players:
            boost_amount = game[f"{p}_boost_amount"]
            boost_usage_cols[f"{p}_no_boost"] = boost_amount == 0
            boost_usage_cols[f"{p}_1st_quad_boost"] = (boost_amount > 0) & (
                boost_amount <= 25
            )
            boost_usage_cols[f"{p}_2nd_quad_boost"] = (boost_amount > 25) & (
                boost_amount <= 50
            )
            boost_usage_cols[f"{p}_3rd_quad_boost"] = (boost_amount > 50) & (
                boost_amount <= 75
            )
            boost_usage_cols[f"{p}_4th_quad_boost"] = (boost_amount > 75) & (
                boost_amount < 100
            )
            boost_usage_cols[f"{p}_full_boost"] = boost_amount == 100

    return boost_usage_cols
