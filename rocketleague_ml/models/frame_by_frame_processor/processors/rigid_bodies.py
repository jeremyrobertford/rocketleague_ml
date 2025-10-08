from __future__ import annotations
from typing import TYPE_CHECKING
from rocketleague_ml.core.actor import Actor
from rocketleague_ml.core.frame import Frame
from rocketleague_ml.core.ball import Ball
from rocketleague_ml.core.car import Car
from rocketleague_ml.core.car_component import Car_Component
from rocketleague_ml.core.rigid_body import Rigid_Body


if TYPE_CHECKING:
    from rocketleague_ml.models.frame_by_frame_processor import Frame_By_Frame_Processor


def process_ball_position(
    processor: Frame_By_Frame_Processor, updated_actor: Actor, frame: Frame
):
    if not frame.game.ball.is_self(updated_actor):
        raise ValueError(
            f"Failed to update ball position because ball was not updated {updated_actor.raw}"
        )
    process_rigid_body_position(
        updated_actor, frame, processor.include_ball_positioning, frame.game.ball
    )
    return None


def process_car_position(
    processor: Frame_By_Frame_Processor, updated_actor: Actor, frame: Frame
):
    car = frame.game.cars.get(updated_actor.actor_id)
    if not car:
        raise ValueError(
            f"Failed to update car position because car was not found {updated_actor.raw}"
        )
    process_rigid_body_position(
        updated_actor, frame, processor.include_car_positioning, car
    )
    return None


def process_rigid_body_position(
    updated_actor: Actor,
    frame: Frame,
    include: bool,
    rigid_body: Ball | Car_Component | Car,
):
    updated_actor = Rigid_Body(updated_actor, "")
    if include:
        field_label = "ball_positioning"
        frame.processed_fields[field_label + "_sleeping"] = (
            1 if updated_actor.positioning.sleeping else 0
        )
        frame.processed_fields[field_label + "_x"] = (
            updated_actor.positioning.location.x
        )
        frame.processed_fields[field_label + "_y"] = (
            updated_actor.positioning.location.y
        )
        frame.processed_fields[field_label + "_z"] = (
            updated_actor.positioning.location.z
        )
        frame.processed_fields[field_label + "_rotation_x"] = (
            updated_actor.positioning.rotation.x
        )
        frame.processed_fields[field_label + "_rotation_y"] = (
            updated_actor.positioning.rotation.y
        )
        frame.processed_fields[field_label + "_rotation_z"] = (
            updated_actor.positioning.rotation.z
        )
        frame.processed_fields[field_label + "_rotation_w"] = (
            updated_actor.positioning.rotation.w
        )
        frame.processed_fields[field_label + "_linear_velocity_x"] = (
            updated_actor.positioning.linear_velocity.x
        )
        frame.processed_fields[field_label + "_linear_velocity_y"] = (
            updated_actor.positioning.linear_velocity.y
        )
        frame.processed_fields[field_label + "_linear_velocity_z"] = (
            updated_actor.positioning.linear_velocity.z
        )
        frame.processed_fields[field_label + "_angular_velocity_x"] = (
            updated_actor.positioning.angular_velocity.x
        )
        frame.processed_fields[field_label + "_angular_velocity_y"] = (
            updated_actor.positioning.angular_velocity.y
        )
        frame.processed_fields[field_label + "_angular_velocity_z"] = (
            updated_actor.positioning.angular_velocity.z
        )
        rigid_body.update_position(updated_actor)
        return None
