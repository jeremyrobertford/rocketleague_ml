from typing import List, Dict, Any
from rocketleague_ml.data.features.trackers.collision_tracker import Collision_Tracker
from rocketleague_ml.data.features.trackers.positioning_tracker import (
    Positioning_Tracker,
)
from rocketleague_ml.data.features.trackers.tag_manager import Tag_Manager
from rocketleague_ml.data.features.trackers.touch_tracker import Touch_Tracker
from rocketleague_ml.data.features.trackers.vision_tracker import Vision_Tracker


class Player_Tracker:
    def __init__(self, player_data: Any):
        self.name = player_data.name
        self.collision = Collision_Tracker()
        self.positioning = Positioning_Tracker(player_data.initial_positioning)
        self.touches = Touch_Tracker()
        self.vision = Vision_Tracker()
        self.tags = Tag_Manager()


class Playstyle_Tracker:
    def __init__(self, players: List[Any]):
        self.players: Dict[int, Player_Tracker] = {}
        for player in players:
            player_tracker = Player_Tracker(player)
            self.players[player.actor_id] = player_tracker
            self.players[player.car.actor_id] = player_tracker

    def activate_game(self):
        self.active = True

    def deactivate_game(self):
        self.active = False
