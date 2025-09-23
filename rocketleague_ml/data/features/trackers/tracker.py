from typing import Dict, Any
from rocketleague_ml.types import TaggedStatsWithMetrics

TAGS = [
    "as_first_man",
    "as_second_man",
    "as_third_man",
    "as_front_man",
    "as_middle_man",
    "as_back_man",
    "in_offensive_third",
    "in_neutral_third",
    "in_defensive_third",
    "in_offensive_half",
    "in_defensive_half",
    "with_possession",
    "first_half_of_game",
    "second_half_of_game",
]


class Tracker:
    """
    Base tracker class that manages tag-based augmentation.
    Specialized trackers should extend this and implement
    update_frame(frame, player_state, tags) to increment metrics.
    """

    def __init__(self):
        # base metrics (to be defined by child)
        self.metrics: Dict[str, Any] = {}

        # tag-augmented stats
        self.tagged_stats: Dict[str, TaggedStatsWithMetrics] = {}
        for tag in TAGS:
            tagged_stats: TaggedStatsWithMetrics = {
                "instances": 0,
                "total_seconds": 0.0,
                "deactivating_frame": None,
                "active": False,
                "metrics": {},  # tracker-specific metrics for this tag
            }
            self.tagged_stats[tag] = tagged_stats
