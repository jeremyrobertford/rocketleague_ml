"""
Loader for Rocket League replays using rrrocket.exe.
Shells out to the rrrocket binary and returns parsed JSON.
"""

import subprocess
import json
import os
from pathlib import Path
from typing import cast

from rocketleague_ml.config import DEFAULT_BIN_DIR, RAW_REPLAYS, PREPROCESSED, LOG
from rocketleague_ml.types.attributes import Raw_Game_Data
from rocketleague_ml.utils.logging import Logger


class JSON_Preprocessor:
    """
    Loader for Rocket League replays using rrrocket.exe.
    Shells out to the rrrocket binary and returns parsed JSON.
    """

    def __init__(self, raise_error: bool = True):
        self.raise_error = raise_error
        self.logger = Logger(LOG)
        self.find_rrrocket()

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
        base_file_name = Path(file_name).stem
        preprocessed_file_path = os.path.join(PREPROCESSED, base_file_name + ".json")
        if os.path.exists(preprocessed_file_path) and not overwrite:
            self.logger.print(f"Using existing JSON for {file_name}.")
            with open(preprocessed_file_path, "r", encoding="utf-8") as f:
                replay_json = json.load(f)
            return cast(Raw_Game_Data, replay_json)

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

        if replay_json["properties"]["TeamSize"] == 3 and save_output:
            with open(preprocessed_file_path, "w+") as f:
                json.dump(replay_json, f, indent=4)
        else:
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
        Path(PREPROCESSED).mkdir(parents=True, exist_ok=True)

        replay_files = sorted([f for f in os.listdir(RAW_REPLAYS)])
        if not replay_files:
            print(f"No .replay files found in {RAW_REPLAYS}.")
            return

        self.succeeded = 0
        self.skipped = 0
        self.failed = 0
        for file_name in replay_files:
            try:
                self.convert_replay(file_name, save_output, overwrite)
            except Exception as e:
                self.logger.print("FAILED.")
                self.logger.print(f"  {type(e).__name__}: {e}")
                self.failed += 1

        print()
        print(
            f"Done. succeeded={self.succeeded}, skipped={self.skipped}, failed={self.failed}."
        )
        print(f"Saved JSON to: {PREPROCESSED}")
