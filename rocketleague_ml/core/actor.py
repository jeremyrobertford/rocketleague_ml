from __future__ import annotations
from typing import Dict, List, cast
from rocketleague_ml.config import (
    LABELS,
    CAMERA_SETTINGS_LABELS,
    CAR_COMPONENT_LABELS,
    PLAYER_COMPONENT_LABELS,
)
from rocketleague_ml.types.attributes import (
    Actor_Export,
    Labeled_Attribute,
    Labeled_Stat_Event,
    Labeled_Stat_Event_Attribute,
    Labeled_Raw_Actor,
    Raw_Actor,
    Raw_Frame,
    Rigid_Body_Positioning,
    Time_Labeled_Rigid_Body_Positioning,
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
        self.updates_by_round: Dict[int, List[Time_Labeled_Rigid_Body_Positioning]] = {}

        initial_trajectory = actor.get("initial_trajectory")
        if initial_trajectory:
            positioning = Positioning(initial_trajectory)
            self.positioning = positioning

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

    @staticmethod
    def label(raw_actor: Raw_Actor, objects: Dict[int, str]):
        labeled_raw_actor = cast(Labeled_Raw_Actor, raw_actor)

        if not labeled_raw_actor["object_id"] in objects:
            raise ValueError(
                f"Object id {labeled_raw_actor["object_id"]} not found in object table"
            )

        labeled_raw_actor["object"] = objects[labeled_raw_actor["object_id"]]

        attribute: Labeled_Attribute | None = labeled_raw_actor.get("attribute")
        if not attribute:
            return labeled_raw_actor

        stat_event: Labeled_Stat_Event | None = attribute.get("StatEvent")
        if not stat_event:
            return labeled_raw_actor

        if stat_event["object_id"] != -1 and not stat_event["object_id"] in objects:
            raise ValueError(
                f"Object id {raw_actor["object_id"]} not found in object table"
            )

        stat_event["object"] = (
            objects[stat_event["object_id"]]
            if stat_event["object_id"] != -1
            else "unknown"
        )

        labeled_raw_actor["attribute"] = cast(
            Labeled_Stat_Event_Attribute, {"StatEvent": stat_event}
        )
        return labeled_raw_actor

    def label_self(self):
        labled_actor = Actor.label(self.raw, self.objects)
        self.object = labled_actor["object"]

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

    def update_position(
        self, updated_actor: Actor, active_game: bool, frame: Raw_Frame, round: int
    ):
        if not updated_actor.positioning:
            raise ValueError(
                "Positoning not found when updating position for {self.actor_id}: {updated_actor.raw}"
            )
        self.positioning = updated_actor.positioning.copy()
        if active_game:
            if round not in self.updates_by_round:
                self.updates_by_round[round] = []
            time_labeled_positioning = cast(
                Time_Labeled_Rigid_Body_Positioning, updated_actor.positioning.to_dict()
            )
            time_labeled_positioning["round"] = round
            time_labeled_positioning["time"] = frame["time"]
            time_labeled_positioning["delta"] = frame["delta"]
            self.updates_by_round[round].append(time_labeled_positioning)

    def to_dict(self) -> Actor_Export:
        return {
            "actor_id": self.actor_id,
            "updates_by_round": self.updates_by_round,
        }
