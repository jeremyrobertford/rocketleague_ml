import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from rocketleague_ml.config import BOOST_PAD_MAP


def get_boost_usage_cols(game: pd.DataFrame, teams: Dict[str, List[str]]):
    tolerance = 300
    pad_positions = np.array([[v[0], v[1], v[2]] for v in BOOST_PAD_MAP.values()])
    pad_amounts = np.array([v[3] for v in BOOST_PAD_MAP.values()])
    boost_usage_cols: Dict[
        str, pd.Series | np.ndarray[Tuple[int, int], np.dtype[np.float64]]
    ] = {}
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

            # get pickup coords
            if f"{p}_boost_pickup_x" not in game.columns:
                s = pd.Series(0)
                boost_usage_cols[f"{p}_boost_pickup_amount"] = s
                continue

            px = game[f"{p}_boost_pickup_x"]
            py = game[f"{p}_boost_pickup_y"]
            pz = game[f"{p}_boost_pickup_z"]

            # distance matrix (frames x pads)
            coords = np.stack([px, py, pz], axis=1).astype(float)
            dists = np.linalg.norm(
                coords[:, None, :] - pad_positions[None, :, :], axis=2  # type: ignore
            )

            # nearest pad and amount
            nearest_idx = np.argmin(dists, axis=1)
            nearest_dist = dists[np.arange(len(dists)), nearest_idx]
            pickup_amount = np.where(
                nearest_dist < tolerance, pad_amounts[nearest_idx], np.nan
            )

            boost_usage_cols[f"{p}_boost_pickup_amount"] = pickup_amount  # type: ignore

    return boost_usage_cols
