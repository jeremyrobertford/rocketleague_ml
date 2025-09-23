# rocketleague_ml/features/trackers/boost_tracker.py
from typing import Dict, Optional, Any
from collections import defaultdict


class BoostTracker:
    def __init__(self):
        # raw counters
        self.boost_collected = 0.0
        self.boost_used = 0.0
        self.time_on_0_boost = 0.0
        self.time_on_full_boost = 0.0

        # rolling boost level for computing averages
        self.boost_sum = 0.0
        self.frame_count = 0

        # optional augmented counters per tag
        self.tagged_stats: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

        # last frame boost level
        self._last_boost: Optional[float] = None

    def update(
        self,
        frame: Dict[str, Any],
        player_state: Dict[str, Any],
        tags: Optional[Dict[str, str]] = None,
    ):
        """
        frame: one network frame from rrrocket
        player_state: contains 'boost' field
        tags: optional dict like {'possession': 'with_possession', 'role': 'first_man'}
        """
        boost = player_state.get("Boost", 0.0)
        if self._last_boost is not None:
            delta = boost - self._last_boost
            if delta > 0:
                self.boost_collected += delta
            elif delta < 0:
                self.boost_used += -delta

        # time counters
        if boost == 0:
            self.time_on_0_boost += 1
        if boost >= 100:
            self.time_on_full_boost += 1

        # rolling average
        self.boost_sum += boost
        self.frame_count += 1

        # store tagged versions
        if tags:
            for tag_name, tag_value in tags.items():
                tstats = self.tagged_stats[tag_value]
                if self._last_boost is not None:
                    delta = boost - self._last_boost
                    if delta > 0:
                        tstats["boost_collected"] += delta
                    elif delta < 0:
                        tstats["boost_used"] += -delta
                if boost == 0:
                    tstats["time_on_0_boost"] += 1
                if boost >= 100:
                    tstats["time_on_full_boost"] += 1
                tstats["boost_sum"] += boost
                tstats["frame_count"] += 1

        self._last_boost = boost

    def average_boost(self) -> float:
        if self.frame_count == 0:
            return 0.0
        return self.boost_sum / self.frame_count

    def percentage_with_no_boost(self) -> float:
        if self.frame_count == 0:
            return 0.0
        return self.time_on_0_boost / self.frame_count

    def percentage_with_full_boost(self) -> float:
        if self.frame_count == 0:
            return 0.0
        return self.time_on_full_boost / self.frame_count

    def to_map(self) -> Dict[str, Any]:
        """
        Returns all boost-related stats, including tagged variants.
        """
        base_stats = {
            "boost_collected": self.boost_collected,
            "boost_used": self.boost_used,
            "average_boost": self.average_boost(),
            "percentage_with_no_boost": self.percentage_with_no_boost(),
            "percentage_with_full_boost": self.percentage_with_full_boost(),
        }

        # include tagged stats
        tagged = {}
        for tag, tstats in self.tagged_stats.items():
            frame_count = tstats.get("frame_count", 0)
            tagged[tag] = {
                "boost_collected": tstats.get("boost_collected", 0.0),
                "boost_used": tstats.get("boost_used", 0.0),
                "average_boost": (
                    (tstats.get("boost_sum", 0.0) / frame_count)
                    if frame_count > 0
                    else 0.0
                ),
                "percentage_with_no_boost": (
                    (tstats.get("time_on_0_boost", 0.0) / frame_count)
                    if frame_count > 0
                    else 0.0
                ),
                "percentage_with_full_boost": (
                    (tstats.get("time_on_full_boost", 0.0) / frame_count)
                    if frame_count > 0
                    else 0.0
                ),
            }

        return {"base": base_stats, "tags": tagged}
