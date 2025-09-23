from __future__ import annotations
from typing import Dict
from rocketleague_ml.config import (
    LABELS,
    CAMERA_SETTINGS_LABELS,
    CAR_COMPONENT_LABELS,
    PLAYER_COMPONENT_LABELS,
)
from rocketleague_ml.types.attributes import Raw_Actor, Active_Actor_Attribute


class Actor:
    def __init__(self, actor: Raw_Actor, objects: Dict[int, str]):
        self.raw = actor
        self.actor_id = actor["actor_id"]
        self.active_actor_id = None
        self.objects = objects
        self.label_self()
        self.categorize_self()

        attribute = actor.get("attribute")
        if attribute is None:
            return
        active_actor: Active_Actor_Attribute | None = attribute.get("ActiveActor")
        if active_actor:
            self.active_actor_id = active_actor["actor"]

    def label_self(self):
        if self.raw["object_id"] in self.objects:
            self.object = self.objects[self.raw["object_id"]]
            return
        raise ValueError(f"Object id {self.raw["object_id"]} not found in object table")

    def categorize_self(self):
        matched_label = LABELS[self.object]
        if matched_label:
            self.category = matched_label
            self.secondary_category = None
            return

        is_camera_settings = CAMERA_SETTINGS_LABELS[self.object]
        if is_camera_settings:
            self.category = "camera_settings"
            self.secondary_category = is_camera_settings
            return

        is_car_component = CAR_COMPONENT_LABELS[self.object]
        if is_car_component:
            self.category = "car_component"
            self.secondary_category = is_car_component
            return

        is_player_component = PLAYER_COMPONENT_LABELS[self.object]
        if is_player_component:
            self.category = "player_component"
            self.secondary_category = is_player_component
            return

        self.category = "Misc"
        self.secondary_category = None
        return

    def is_self(self, possible_self: Actor):
        return self.actor_id == possible_self.actor_id

    def belongs_to(self, possible_parent: Actor):
        return self.active_actor_id == possible_parent.actor_id

    def owns(self, possible_child: Actor):
        return self.actor_id == possible_child.active_actor_id
