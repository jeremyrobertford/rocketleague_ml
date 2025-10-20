# type: ignore
import numpy as np
import pandas as pd
from typing import Dict, List


def assign_jumps(
    game: pd.DataFrame, teams: Dict[str, List[str]], lookahead_frames: int = 5
) -> pd.DataFrame:
    """
    Identifies single jump events per player and estimates jump impulse vectors.
    Also marks dodges/double-jumps and flips without a prior jump.

    Updated for binary 'active' columns (1 = active, 0 = inactive).
    """

    game = game.copy()

    for players in teams.values():
        for player in players:
            jump_col = f"{player}_jump_active"
            dodge_col = f"{player}_dodge_active"
            double_col = f"{player}_double_jump_active"

            # Skip players missing required columns
            if jump_col not in game.columns:
                continue

            # Fill and convert to int
            jump_activity = game[jump_col].fillna(0).astype(int)
            dodge_activity = game[dodge_col].fillna(0).astype(int)
            double_activity = game[double_col].fillna(0).astype(int)

            # Initialize output columns
            for axis in ["x", "y", "z"]:
                game[f"{player}_single_jump_{axis}"] = np.nan
            game[f"{player}_flip_without_jump"] = 0

            # --- Detect jump starts (rising edge: 0 → 1)
            jump_starts = game.index[(jump_activity.diff() == 1)]

            for idx in jump_starts:
                lookahead_end = min(idx + lookahead_frames, game.index[-1])

                # Check for dodge/double activation in the lookahead window
                dodge_change = (dodge_activity.loc[idx:lookahead_end] > 0).any()
                double_change = (double_activity.loc[idx:lookahead_end] > 0).any()

                # Only count as single jump if it didn’t transition to dodge/double
                if not dodge_change and not double_change:
                    for axis in ["x", "y", "z"]:
                        vel_col = f"{player}_positioning_linear_velocity_{axis}"
                        if vel_col in game.columns:
                            v0 = game.at[idx, vel_col]
                            v1 = game.loc[lookahead_end, vel_col]
                            game.at[idx, f"{player}_single_jump_{axis}"] = v1 - v0
                        else:
                            game.at[idx, f"{player}_single_jump_{axis}"] = np.nan

            # --- Detect flips/dodges without recent jump
            flip_mask = (dodge_activity.diff() == 1) | (double_activity.diff() == 1)
            for idx in game.index[flip_mask]:
                # Has a jump in last N frames?
                recent_jump = jump_starts[(idx - jump_starts) <= lookahead_frames]
                if (
                    len(recent_jump) == 0
                    or (idx - recent_jump.max()) > lookahead_frames
                ):
                    game.at[idx, f"{player}_flip_without_jump"] = 1

    return game
