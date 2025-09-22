"""
Extract numerical features from parsed replays.
Features will be fed into the ML model.
"""

def extract_features(replay_data):
    # Example features:
    # - player position (x, y, z)
    # - velocity
    # - boost level
    # - ball distance
    # - action taken (jump, boost, hit)
    return {
        "X": None,  # feature matrix
        "y": None   # labels (e.g. action chosen)
    }
