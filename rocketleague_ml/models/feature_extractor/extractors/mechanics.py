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

    Parameters
    ----------
    game : pd.DataFrame
        Frame-by-frame dataset (must include per-player jump, dodge, and double-jump bytes).
    teams : Dict[str, List[str]]
        Dict mapping team names -> list of player names.
    lookahead_frames : int
        Number of frames to look ahead for detecting jump continuation or dodge transitions.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with added columns:
            {player}_single_jump_x/y/z
            {player}_flip_without_jump
    """
    game = game.copy()

    for players in teams.values():
        for player in players:
            jump_col = f"{player}_jump_byte"
            dodge_col = f"{player}_dodge_byte"
            double_col = f"{player}_double_jump_byte"

            # Initialize new columns
            for axis in ["x", "y", "z"]:
                game[f"{player}_single_jump_{axis}"] = np.nan
            game[f"{player}_flip_without_jump"] = 0

            # Skip if jump column missing
            if jump_col not in game.columns:
                continue

            jump_bytes = game[jump_col].fillna(method="ffill").fillna(0).astype(int)
            dodge_bytes = (
                game[dodge_col].fillna(method="ffill").fillna(0).astype(int)
                if dodge_col in game.columns
                else np.zeros(len(game), dtype=int)
            )
            double_bytes = (
                game[double_col].fillna(method="ffill").fillna(0).astype(int)
                if double_col in game.columns
                else np.zeros(len(game), dtype=int)
            )

            # --- Detect jump starts (odd -> even pattern)
            # Each pair (1→2, 3→4, 5→6, etc.) = one jump press/release cycle
            jump_starts = game.index[(jump_bytes.diff() == 1) & (jump_bytes % 2 == 1)]

            for idx in jump_starts:
                # Look ahead a few frames to check if it becomes a dodge or double jump
                lookahead_end = min(idx + lookahead_frames, len(game) - 1)
                dodge_change = (dodge_bytes.iloc[idx:lookahead_end].diff() != 0).any()
                double_change = (double_bytes.iloc[idx:lookahead_end].diff() != 0).any()

                # Only count as single jump if it didn't transition
                if not dodge_change and not double_change:
                    # Placeholder: compute approximate impulse vector
                    # You can replace this with actual velocity delta if those columns exist
                    for axis in ["x", "y", "z"]:
                        vel_col = f"{player}_positoning_linear_velocity_{axis}"
                        if vel_col in game.columns:
                            v0 = game.at[idx, vel_col]
                            v1 = game.iloc[lookahead_end][vel_col]
                            game.at[idx, f"{player}_single_jump_{axis}"] = v1 - v0
                        else:
                            game.at[idx, f"{player}_single_jump_{axis}"] = np.nan

            # --- Mark flips/dodges without a jump
            # Flip occurred (dodge or double jump) but no recent jump in prior few frames
            flip_mask = (dodge_bytes.diff() != 0) | (double_bytes.diff() != 0)
            for idx in game.index[flip_mask]:
                # Check if a jump occurred in the last few frames
                recent_jump = ((idx - jump_starts) <= lookahead_frames) & (
                    (idx - jump_starts) >= 0
                )
                if not recent_jump.any():
                    game.at[idx, f"{player}_flip_without_jump"] = 1

    return game
