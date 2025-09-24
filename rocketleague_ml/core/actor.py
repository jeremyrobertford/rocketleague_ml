from __future__ import annotations
from typing import Dict, List
from rocketleague_ml.config import (
    LABELS,
    CAMERA_SETTINGS_LABELS,
    CAR_COMPONENT_LABELS,
    PLAYER_COMPONENT_LABELS,
)
from rocketleague_ml.types.attributes import (
    Raw_Actor,
    Rigid_Body_Positioning,
    Active_Actor_Attribute,
)
from rocketleague_ml.core.positioning import Positioning


class Actor:
    def __init__(self, actor: Raw_Actor, objects: Dict[int, str]):
        self.raw = actor
        self.actor_id = actor["actor_id"]
        self.active_actor_id = None
        self.positioning = None
        self.objects = objects
        self.label_self()
        self.categorize_self()
        self.updates: List[Rigid_Body_Positioning] = []

        initial_trajectory = actor.get("initial_trajectory")
        if initial_trajectory:
            positioning = Positioning(initial_trajectory)
            self.positioning = positioning
            self.updates.append(positioning.to_dict())

        attribute = actor.get("attribute")
        if attribute is None:
            return
        active_actor: Active_Actor_Attribute | None = attribute.get("ActiveActor")
        if active_actor:
            self.active_actor_id = active_actor["actor"]
            return

        rigid_body: Rigid_Body_Positioning | None = attribute.get("RigidBody")
        if rigid_body:
            self.positioning = Positioning(rigid_body)

    def label_self(self):
        if self.raw["object_id"] in self.objects:
            self.object = self.objects[self.raw["object_id"]]
            return
        raise ValueError(f"Object id {self.raw["object_id"]} not found in object table")

    def categorize_self(self):
        matched_label = LABELS.get(self.object)
        if matched_label:
            self.category = matched_label
            self.secondary_category = None
            return

        is_camera_settings = CAMERA_SETTINGS_LABELS.get(self.object)
        if is_camera_settings:
            self.category = "camera_settings"
            self.secondary_category = is_camera_settings
            return

        is_car_component = CAR_COMPONENT_LABELS.get(self.object)
        if is_car_component:
            self.category = "car_component"
            self.secondary_category = is_car_component
            return

        is_player_component = PLAYER_COMPONENT_LABELS.get(self.object)
        if is_player_component:
            self.category = "player_component"
            self.secondary_category = is_player_component
            return

        self.category = "Misc"
        self.secondary_category = None

    def is_self(self, possible_self: Actor):
        return self.actor_id == possible_self.actor_id

    def belongs_to(self, possible_parent: Actor):
        return self.active_actor_id == possible_parent.actor_id

    def owns(self, possible_child: Actor):
        return self.actor_id == possible_child.active_actor_id

    def update_position(self, updated_actor: Actor):
        if not updated_actor.positioning:
            raise ValueError(
                "Positoning not found when updating position for {self.actor_id}: {updated_actor.raw}"
            )
        self.position = updated_actor.positioning.copy()
        self.updates.append(updated_actor.positioning.to_dict())
