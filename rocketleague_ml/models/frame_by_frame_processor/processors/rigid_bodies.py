from __future__ import annotations
from typing import TYPE_CHECKING
from rocketleague_ml.core.actor import Actor
from rocketleague_ml.core.frame import Frame
from rocketleague_ml.core.rigid_body import Rigid_Body


if TYPE_CHECKING:
    from rocketleague_ml.models.frame_by_frame_processor import Frame_By_Frame_Processor


def _process_rigid_body_position(
    updated_actor: Actor,
    frame: Frame,
    include: bool,
    rigid_body: Rigid_Body,
):
    rigid_body.update_position(updated_actor)
    if include:
        field_label = f"{rigid_body.body_label}_positioning"
        frame.processed_fields[field_label + "_sleeping"] = (
            1 if rigid_body.positioning.sleeping else 0
        )
        frame.processed_fields[field_label + "_x"] = rigid_body.positioning.location.x
        frame.processed_fields[field_label + "_y"] = rigid_body.positioning.location.y
        frame.processed_fields[field_label + "_z"] = rigid_body.positioning.location.z
        frame.processed_fields[field_label + "_rotation_x"] = (
            rigid_body.positioning.rotation.x
        )
        frame.processed_fields[field_label + "_rotation_y"] = (
            rigid_body.positioning.rotation.y
        )
        frame.processed_fields[field_label + "_rotation_z"] = (
            rigid_body.positioning.rotation.z
        )
        frame.processed_fields[field_label + "_rotation_w"] = (
            rigid_body.positioning.rotation.w
        )
        frame.processed_fields[field_label + "_linear_velocity_x"] = (
            rigid_body.positioning.linear_velocity.x
        )
        frame.processed_fields[field_label + "_linear_velocity_y"] = (
            rigid_body.positioning.linear_velocity.y
        )
        frame.processed_fields[field_label + "_linear_velocity_z"] = (
            rigid_body.positioning.linear_velocity.z
        )
        frame.processed_fields[field_label + "_angular_velocity_x"] = (
            rigid_body.positioning.angular_velocity.x
        )
        frame.processed_fields[field_label + "_angular_velocity_y"] = (
            rigid_body.positioning.angular_velocity.y
        )
        frame.processed_fields[field_label + "_angular_velocity_z"] = (
            rigid_body.positioning.angular_velocity.z
        )
        rigid_body.update_position(rigid_body)
        return None


def process_rigid_body_position(
    processor: Frame_By_Frame_Processor, updated_actor: Actor, frame: Frame
):
    if frame.game.ball.is_self(updated_actor):
        _process_rigid_body_position(
            updated_actor, frame, processor.include_ball_positioning, frame.game.ball
        )
        return None

    car = frame.game.cars.get(updated_actor.actor_id)
    if not frame.resync and car and car.player.actor_id in frame.game.do_not_track:
        del frame.game.do_not_track[car.player.actor_id]
    if car:
        if car.player.actor_id not in frame.game.do_not_track:
            _process_rigid_body_position(
                updated_actor, frame, processor.include_car_positioning, car
            )
            car.update_position(updated_actor)
        return None

    car_component = frame.game.car_components.get(updated_actor.actor_id)
    if car_component:
        return None

    disconnected_car = frame.game.disconnected_cars.get(updated_actor.actor_id)
    if disconnected_car:
        frame.add_updated_actor_to_disconnected_car_component_updates(updated_actor)
        return None

    disconnected_car_component = frame.game.disconnected_car_components.get(
        updated_actor.actor_id
    )
    if disconnected_car_component:
        frame.add_updated_actor_to_disconnected_car_component_updates(updated_actor)
        return None

    raise ValueError(f"Failed to update rigid body position {updated_actor.raw}")
