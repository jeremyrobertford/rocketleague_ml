"""
Loader for Rocket League replays using rrrocket.exe.
Shells out to the rrrocket binary and returns parsed JSON.
"""

import subprocess
import json
import os
import shutil
from typing import Any

from rocketleague_ml.config import BASE_DIR

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


def load_replay(file_path: str, rrrocket_path: str | None = None) -> dict[str, Any]:
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

    return replay_json
