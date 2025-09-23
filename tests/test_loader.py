import os
import json
import tempfile
import shutil
import pytest

from rocketleague_ml.data.loader import find_rrrocket, load_replay
from rocketleague_ml.config import BASE_DIR

# Path to your single test replay
TEST_REPLAY = os.path.join(BASE_DIR, "..", "data", "raw", "replay.replay")
rrrocket_path = find_rrrocket(raise_error=False)


@pytest.mark.skipif(rrrocket_path is None, reason="rrrocket.exe not found")  # type: ignore
def test_load_replay_and_save_json():
    """End-to-end test: parse a replay and save JSON to a temp folder."""
    # Create temporary folder to save JSON
    temp_dir = tempfile.mkdtemp()
    try:
        # Load replay via rrrocket.exe
        replay_data = load_replay(TEST_REPLAY)
        assert isinstance(replay_data, dict), "Parsed replay should be a dictionary"

        # Save JSON
        out_file = os.path.join(temp_dir, "replay.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(replay_data, f, indent=2, ensure_ascii=False)

        # Check file exists
        assert os.path.exists(out_file), "JSON output file was not created"

        # Check JSON can be loaded back and has expected keys
        with open(out_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert isinstance(loaded, dict)
        for key in ["properties", "network_frames", "names", "objects"]:
            assert key in loaded, f"Key '{key}' not found in JSON"

    finally:
        # Clean up temp folder
        shutil.rmtree(temp_dir)
