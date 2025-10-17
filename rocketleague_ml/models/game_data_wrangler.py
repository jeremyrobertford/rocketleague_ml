import os
import json
from pathlib import Path
from typing import cast, List
from rocketleague_ml.config import PREPROCESSED, WRANGLED
from rocketleague_ml.types.attributes import Raw_Game_Data, Raw_Actor
from rocketleague_ml.models.rrrocket_json_preprocessor import RRRocket_JSON_Preprocessor
from rocketleague_ml.utils.logging import Logger


class Game_Data_Wrangler:
    def __init__(
        self, preprocessor: RRRocket_JSON_Preprocessor, logger: Logger = Logger()
    ):
        self.preprocessor = preprocessor
        self.logger = logger

    def sort_updated_actors(self, actor: Raw_Actor):
        attribute = actor.get("attribute")
        has_active_actor = attribute is not None and "ActiveActor" in attribute
        return (not has_active_actor, actor.get("actor_id", float("inf")))

    def load_wrangled_file(self, file_name: str):
        """
        Loads a wrangled CSV file from config.WRANGLED

        Parameters
        ----------
        file_name : str
            Name of the wrangled file.

        Returns
        -------
        pandas.DataFrame
            DataFrame constructed from wrangled file
        """
        wrangled_file_path = os.path.join(WRANGLED, file_name)
        if os.path.exists(wrangled_file_path):
            with open(wrangled_file_path, "r", encoding="utf-8") as f:
                game_data_json = json.load(f)
            return cast(Raw_Game_Data, game_data_json)
        raise FileExistsError(
            f"Preprocessed file does not exist at {wrangled_file_path}"
        )

    def try_load_wrangled_file(self, file_name: str):
        try:
            return self.load_wrangled_file(file_name)
        except Exception:
            return None

    def save_wrangled_file(self, file_name: str, game_data: Raw_Game_Data):
        """
        Saves a wrangled JSON file to config.WRANGLED

        Parameters
        ----------
        file_name : str
            Name of the wrangled file.
        game_data : str
            Game data to be saved
        """
        wrangled_file_path = os.path.join(WRANGLED, file_name)
        with open(wrangled_file_path, "w+") as f:
            json.dump(game_data, f, indent=4)
        return None

    def wrangle_game(self, game_data: Raw_Game_Data):
        for frame in game_data["network_frames"]["frames"]:
            frame["updated_actors"] = sorted(
                frame["updated_actors"], key=self.sort_updated_actors
            )
        return game_data

    def wrangle_games(
        self,
        save_output: bool = True,
        overwrite: bool = False,
        game_datas: List[Raw_Game_Data] | None = None,
    ):
        self.logger.print(f"Wrangling game data files...")
        self.succeeded = 0
        self.skipped = 0
        self.failed = 0
        wrangled_games: List[Raw_Game_Data] = []

        if game_datas:
            for game_data in game_datas:
                try:
                    wrangle_game_data = self.wrangle_game(game_data)
                    id = game_data["properties"]["Id"]
                    if save_output:
                        self.save_wrangled_file(f"{id}.json", wrangle_game_data)
                    else:
                        wrangled_games.append(wrangle_game_data)

                    self.logger.print("OK.")
                    self.succeeded += 1

                except Exception as e:
                    self.logger.print("FAILED.")
                    self.logger.print(f"  {type(e).__name__}: {e}")
                    self.failed += 1

            if save_output:
                return None
            return wrangled_games

        Path(WRANGLED).mkdir(parents=True, exist_ok=True)

        game_files = sorted([f for f in os.listdir(PREPROCESSED)])
        if not game_files:
            self.logger.print(f"No files found in {PREPROCESSED}.")
            return None

        for file_name in game_files:
            game = self.try_load_wrangled_file(file_name)
            if game is not None and not overwrite:
                self.logger.print(f"Using existing wrangled data for {file_name}.")
                self.skipped += 1
                if not save_output:
                    wrangled_games.append(game)
            else:
                self.logger.print(
                    f"Wrangling {file_name} -> {WRANGLED} ...", end=" ", flush=True
                )
                try:
                    game_data = self.preprocessor.load_preprocessed_file(file_name)
                    wrangle_game_data = self.wrangle_game(game_data)
                    if save_output:
                        self.save_wrangled_file(file_name, wrangle_game_data)
                    else:
                        wrangled_games.append(wrangle_game_data)
                    self.logger.print("OK.")
                    self.succeeded += 1
                except Exception as e:
                    self.logger.print("FAILED.")
                    self.logger.print(f"  {type(e).__name__}: {e}")
                    self.failed += 1

        self.logger.print(
            f"Done. succeeded={self.succeeded}, skipped={self.skipped}, failed={self.failed}."
        )
        if save_output and self.succeeded:
            self.logger.print(f"Saved wrangled games to: {WRANGLED}")
        self.logger.print()

        if save_output:
            return None
        return wrangled_games
