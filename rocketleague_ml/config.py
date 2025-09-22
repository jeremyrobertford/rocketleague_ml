import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
RAW_REPLAYS = os.path.join(DATA_DIR, "raw")
PROCESSED = os.path.join(DATA_DIR, "processed")

# Training configs
RANDOM_SEED = 42
TEST_SIZE = 0.2

# Feature configs
# Simplify entire map into a 3 x 3 x 3 grid
map_simplification_areas = [3, 3, 3]
