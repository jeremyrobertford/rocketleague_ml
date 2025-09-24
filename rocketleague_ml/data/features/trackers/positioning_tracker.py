from typing import Dict, List, Any
from rocketleague_ml.types import RigidBodyPositioning, Positioning_Dict
from rocketleague_ml.data.features.trackers.tracker import Tracker


class Positioning_Tracker(Tracker):
    def __init__(
        self,
        positioning: RigidBodyPositioning,
    ):
        self.time: float | None = None
        self.round: int = 1
        self.round_updates: Dict[int, List[RigidBodyPositioning]] = {1: []}
        self.update(positioning, game_is_active=False)

    def update_round(self):
        self.round += 1
        self.round_updates[self.round] = []

    def update(
        self,
        new_positioning: RigidBodyPositioning,
        game_is_active: bool,
        frame: Any | None = None,
    ):
        if frame:
            self.time = frame["time"]
        if game_is_active:
            self.round_updates[self.round].append(new_positioning)
        self.sleeping = new_positioning["sleeping"]
        self.location = new_positioning["location"]
        self.rotation = new_positioning["rotation"]
        self.linear_velocity = new_positioning["linear_velocity"]
        self.angular_velocity = new_positioning["angular_velocity"]

    def to_dict(self) -> Positioning_Dict:
        return {
            "time": self.time,
            "sleeping": self.sleeping,
            "location": self.location,
            "rotation": self.rotation,
            "linear_velocity": self.linear_velocity,
            "angular_velocity": self.angular_velocity,
        }
