from typing import Dict, Any
from rocketleague_ml.types import TaggedStats
from rocketleague_ml.config import TAGS


class Tag_Manager:
    def __init__(self):
        self.tags: Dict[str, TaggedStats] = {}
        for tag in TAGS:
            tagged_stats: TaggedStats = {
                "instances": 0,
                "total_seconds": 0.0,
                "deactivating_frame": None,
                "active": False,
            }
            self.tags[tag] = tagged_stats

    def check_tag_activation(self, frame: Dict[str, Any]):
        self.check_man_type(frame)
        pass

    def check_man_type(self, frame: Dict[str, Any]):
        pass
