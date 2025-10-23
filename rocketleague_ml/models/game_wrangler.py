import numpy as np
import pandas as pd
from typing import List
from rocketleague_ml.utils.logging import Logger


class Game_Wrangler:
    def __init__(self, logger: Logger = Logger()):
        self.logger = logger

    def keep_min_substep_per_stint(self, game: pd.DataFrame):
        collision_cols = [
            c
            for c in game.columns
            if "_hit_ball_substep" in c or "_hit_player_substep_" in c
        ]

        for col in collision_cols:
            vals = game[col].to_numpy()
            active_mask = ~np.isnan(vals) & (vals > 0)
            active_indices = np.where(active_mask)[0]

            if not len(active_indices):
                continue

            stints: List[List[int]] = []
            current_group: List[int] = [active_indices[0]]

            for idx in active_indices[1:]:
                if idx - current_group[-1] > 1:  # gap -> new stint
                    stints.append(current_group)
                    current_group = [idx]
                else:
                    current_group.append(idx)
            stints.append(current_group)

            # keep only the min substep for each stint
            for stint in stints:
                stint_vals = vals[stint]
                min_idx = stint[np.argmin(stint_vals)]
                # zero out everything else
                for i in stint:
                    if i != min_idx:
                        vals[i] = np.nan  # or 0 depending on your preference

            game[col] = vals

        return game

    def collapse_hit_substeps_to_bool(self, game: pd.DataFrame):
        collision_cols = [
            c
            for c in game.columns
            if "_hit_ball_substep" in c or "_hit_player_substep_" in c
        ]

        for col in collision_cols:
            # Determine the new boolean column name
            if "_hit_ball_substep" in col:
                new_col = col.replace("_hit_ball_substep", "_hit_ball")
            elif "_hit_player_fsubstep_" in col:
                parts = col.split("_hit_player_substep_")
                new_col = f"{parts[0]}_hit_{parts[1]}"
            else:
                continue

            # Any non-NaN or positive value becomes True
            game[new_col] = (~game[col].isna()) & (game[col] > 0)

        # Drop the old substep columns
        game.drop(columns=collision_cols, inplace=True)

        return game

    def wrangle_game(self, game: pd.DataFrame):
        game = self.keep_min_substep_per_stint(game)
        game = self.collapse_hit_substeps_to_bool(game)
        return game
