from __future__ import annotations
from rocketleague_ml.core.actor import Actor
from rocketleague_ml.core.positioning import Positioning
from rocketleague_ml.types.attributes import (
    Rigid_Body_Positioning,
)


class Rigid_Body(Actor):
    def __init__(self, rigid_body: Actor, label: str):
        super().__init__(rigid_body.raw, rigid_body.objects)
        self.body_label = label

        initial_trajectory = rigid_body.raw.get("initial_trajectory")
        if initial_trajectory:
            positioning = Positioning(initial_trajectory)
            self.positioning = positioning
            return None

        if not rigid_body.attribute:
            raise ValueError(
                f"Positoning not found when creating rigid body {rigid_body.raw}"
            )

        rigid_body_attr: Rigid_Body_Positioning | None = rigid_body.attribute.get(
            "RigidBody"
        )
        if rigid_body_attr:
            self.positioning = Positioning(rigid_body_attr)
            return None

        raise ValueError(
            f"Positoning not found when creating rigid body {rigid_body.raw}"
        )

    def __repr__(self):
        return f"Rigid_Body(actor_id={self.actor_id}, label={self.body_label}, position={self.positioning.location})"

    def update_position(self, updated_actor: Actor):
        updated_actor = Rigid_Body(updated_actor, "")
        updated_actor.positioning.previous_linear_velocity = (
            self.positioning.linear_velocity if self.positioning else None
        )
        self.positioning = updated_actor.positioning.copy()
        return None
