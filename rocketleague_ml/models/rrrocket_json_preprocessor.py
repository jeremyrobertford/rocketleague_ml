"""
Loader for Rocket League replays using rrrocket.exe.
Shells out to the rrrocket binary and returns parsed JSON.
"""

import subprocess
import json
import os
from pathlib import Path
from typing import cast, Any, List

from rocketleague_ml.config import DEFAULT_BIN_DIR, RAW_REPLAYS, PREPROCESSED
from rocketleague_ml.types.attributes import Raw_Game_Data
from rocketleague_ml.utils.logging import Logger


def game_3v3_filter(game_data: Raw_Game_Data):
    return game_data["properties"]["TeamSize"] == 3


class RRRocket_JSON_Preprocessor:
    """
    Loader for Rocket League replays using rrrocket.exe.
    Shells out to the rrrocket binary and returns parsed JSON.
    """

    def __init__(
        self, logger: Logger, raise_error: bool = True, replay_filter: Any = None
    ):
        self.raise_error = raise_error
        self.logger = logger
        self.replay_filter = replay_filter or game_3v3_filter
        self.find_rrrocket()
        return None

    def find_rrrocket(self):
        """
        Resolve rrrocket executable path.
        Priority:
        1. rrrocket_path argument
        2. bin/rrrocket.exe next to repo
        3. rrrocket on PATH (shutil.which)
        4.a. Returns True if the path was found or False if not found and raise_error=False
        4.b. Raises FileNotFoundError if not found and raise_error=True

        Returns
        -------
        bool
            Whether the path was found or not.

        Raises
        -------
        FileNotFoundError
            rrrocket.exe not found in config.DEFAULT_BIN_DIR.
        """
        rrrocket_path = os.path.join(DEFAULT_BIN_DIR, "rrrocket.exe")
        if os.path.exists(rrrocket_path):
            self.rrrocket_path = rrrocket_path
            return True
        if self.raise_error:
            raise FileNotFoundError(
                f"rrrocket executable not found. Place it in {DEFAULT_BIN_DIR}."
            )
        return False

    def convert_replay_name_to_preprocessed_path(self, replay_file_name: str):
        base_file_name = Path(replay_file_name).stem
        preprocessed_file_path = os.path.join(PREPROCESSED, base_file_name + ".json")
        return preprocessed_file_path

    def load_preprocessed_file(self, replay_file_name: str):
        """
        Loads a preprocessed JSON file from config.PREPROCESSED

        Parameters
        ----------
        file_name : str
            Name of the .replay file.

        Returns
        -------
        dict
            Parsed JSON output from preprocess JSON file.
        """
        preprocessed_file_path = self.convert_replay_name_to_preprocessed_path(
            replay_file_name
        )
        if os.path.exists(preprocessed_file_path):
            with open(preprocessed_file_path, "r", encoding="utf-8") as f:
                game_data_json = json.load(f)
            return cast(Raw_Game_Data, game_data_json)
        raise FileExistsError(
            f"Preprocessed file does not exist at {preprocessed_file_path}"
        )

    def try_load_preprocessed_file(self, replay_file_name: str):
        try:
            return self.load_preprocessed_file(replay_file_name)
        except Exception:
            return None

    def save_preprocessed_file(
        self, replay_file_name: str, game_data_json: Raw_Game_Data
    ):
        """
        Saves a preprocessed JSON file to config.PREPROCESSED

        Parameters
        ----------
        file_name : str
            Name of the .replay file.
        """
        preprocessed_file_path = self.convert_replay_name_to_preprocessed_path(
            replay_file_name
        )
        with open(preprocessed_file_path, "w+") as f:
            json.dump(game_data_json, f, indent=4)
        return None

    def convert_replay(
        self, file_name: str, save_output: bool = True, overwrite: bool = False
    ):
        """
        Run rrrocket on a replay and return parsed JSON as a Python dict. If
        save_output=True then save the parsed JSON to config.PREPROCESSED path
        before returning dict.

        Parameters
        ----------
        file_name : str
            Name of the .replay file.

        Returns
        -------
        dict
            Parsed JSON output from rrrocket.
        """
        preprocessed_json = self.try_load_preprocessed_file(file_name)
        if preprocessed_json and not overwrite:
            self.logger.print(f"Using existing JSON for {file_name}.")
            return preprocessed_json

        if not file_name.endswith(".replay"):
            raise ValueError(f"{file_name} is not a .replay file.")

        file_path = os.path.join(RAW_REPLAYS, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"{file_name} not found. Place it in {RAW_REPLAYS}."
            )

        self.logger.print(
            f"Parsing {file_name} -> {PREPROCESSED} ...", end=" ", flush=True
        )
        try:
            proc = subprocess.run(
                [self.rrrocket_path, file_path, "--network-parse", "--pretty"],
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"rrrocket failed on {file_path}. returncode={e.returncode}\nstderr:\n{e.stderr}"
            ) from e

        try:
            replay_json = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse JSON from rrrocket for {file_name}: {e}"
            ) from e

        use_game = self.replay_filter(replay_json) if self.replay_filter else False
        if use_game and save_output:
            self.save_preprocessed_file(file_name, replay_json)
        elif not use_game:
            self.logger.print("SKIPPED.")
            self.skipped += 1

        self.logger.print("OK.")
        self.succeeded += 1
        return cast(Raw_Game_Data, replay_json)

    def convert_replays(self, save_output: bool = True, overwrite: bool = False):
        """
        Run rrrocket on all raw replays and return a list of each parsed JSON
        as a Python dict. If save_output=True then save each parsed JSON to
        config.PREPROCESSED path and return None.

        Returns
        -------
        List[dict]
            List of parsed JSON output from rrrocket if save_output=False.
        None
            If save_output=True.
        """
        self.logger.print(f"Preprocessing replay files...")
        Path(PREPROCESSED).mkdir(parents=True, exist_ok=True)

        replay_files = sorted([f for f in os.listdir(RAW_REPLAYS)])
        if not replay_files:
            print(f"No .replay files found in {RAW_REPLAYS}.")
            return

        self.succeeded = 0
        self.skipped = 0
        self.failed = 0
        games: List[Raw_Game_Data] = []
        for file_name in replay_files:
            try:
                game = self.convert_replay(file_name, save_output, overwrite)
                if not save_output:
                    games.append(game)
            except Exception as e:
                self.logger.print("FAILED.")
                self.logger.print(f"  {type(e).__name__}: {e}")
                self.failed += 1

        self.logger.print(
            f"Done. succeeded={self.succeeded}, skipped={self.skipped}, failed={self.failed}."
        )
        if save_output and self.succeeded:
            self.logger.print(f"Saved JSON to: {PREPROCESSED}")
        self.logger.print()
        return games if not save_output else None
