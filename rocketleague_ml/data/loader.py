"""
Loader for Rocket League replays using rrrocket.exe.
Shells out to the rrrocket binary and returns parsed JSON.
"""

import subprocess
import json
import pickle
import os
import shutil
from typing import cast

from rocketleague_ml.config import BASE_DIR
from rocketleague_ml.types.attributes import Raw_Game_Data
from rocketleague_ml.core.game import Game

# Default location in the repo
DEFAULT_BIN_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "bin"))
DEFAULT_RRROCKET = os.path.join(DEFAULT_BIN_DIR, "rrrocket.exe")  # Windows binary name


def find_rrrocket(
    rrrocket_path: str | None = None, raise_error: bool = True
) -> str | None:
    """
    Resolve rrrocket executable path.
    Priority:
      1. rrrocket_path argument
      2. bin/rrrocket.exe next to repo
      3. rrrocket on PATH (shutil.which)
      4.a. Returns the path if found or None if not found and raise_error=False
      4.b. Raises FileNotFoundError if not found and raise_error=True.
    """
    if rrrocket_path and os.path.exists(rrrocket_path):
        return rrrocket_path
    default_path = os.path.join(DEFAULT_BIN_DIR, "rrrocket.exe")
    if os.path.exists(default_path):
        return default_path
    which_path = shutil.which("rrrocket")
    if which_path:
        return which_path
    if raise_error:
        raise FileNotFoundError(
            f"rrrocket executable not found. Place it in {DEFAULT_BIN_DIR} or supply --bin-path."
        )
    return None


def load_replay(file_path: str, rrrocket_path: str | None = None):
    """
    Run rrrocket on a replay and return parsed JSON as a Python dict.

    Parameters
    ----------
    file_path : str
        Path to the .replay file.
    rrrocket_path : str | None
        Optional path to rrrocket.exe. If None, uses default locations.

    Returns
    -------
    dict
        Parsed JSON output from rrrocket.
    """
    rrexe: str = find_rrrocket(rrrocket_path)  # pyright: ignore[reportAssignmentType]

    # call rrrocket. This assumes rrrocket supports a '--json' flag (common).
    # If your rrrocket build uses a different flag, change below.
    try:
        proc = subprocess.run(
            [rrexe, file_path, "--network-parse", "--pretty"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        # include stderr to help debugging
        raise RuntimeError(
            f"rrrocket failed on {file_path}. returncode={e.returncode}\nstderr:\n{e.stderr}"
        ) from e

    # parse stdout as JSON
    try:
        replay_json = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse JSON from rrrocket for {file_path}: {e}"
        ) from e

    return cast(Raw_Game_Data, replay_json)


def load_preprocessed_replay(file_path: str):
    """
    Load preprocessed replay JSON as a Python dict.

    Parameters
    ----------
    file_path : str
        Path to the preprocessed JSON file.

    Returns
    -------
    dict
        Parsed JSON output from rrrocket.
    """

    try:
        with open(file_path, "r") as f:
            replay_json = json.load(f)
            return cast(Raw_Game_Data, replay_json)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse JSON from rrrocket for {file_path}: {e}"
        ) from e


def load_processed_game(file_path: str):
    """
    Load processed .pkl game as a Game class.

    Parameters
    ----------
    file_path : str
        Path to the .pkl game file.

    Returns
    -------
    Game
        Parsed game output from .pkl file.
    """

    try:
        with open(file_path, "rb") as f:
            game = pickle.load(f)
            return cast(Game, game)
    except pickle.PickleError as e:
        raise ValueError(f"Failed to parse .pkl for {file_path}: {e}") from e
