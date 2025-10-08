from __future__ import annotations
from typing import TYPE_CHECKING
from rocketleague_ml.core.actor import Actor
from rocketleague_ml.core.frame import Frame
from rocketleague_ml.core.camera_settings import Camera_Settings
from rocketleague_ml.core.player_component import Player_Component

if TYPE_CHECKING:
    from rocketleague_ml.models.frame_by_frame_processor import Frame_By_Frame_Processor


def process_camera_settings(
    processor: Frame_By_Frame_Processor, updated_actor: Actor, frame: Frame
):
    player = frame.game.camera_settings[updated_actor.actor_id].player
    player.assign_camera_settings(updated_actor)
    if processor.include_advanced_vision:
        new_camera_settings = Camera_Settings(updated_actor, player)
        new_camera_settings.update_settings(updated_actor)
        field_label = f"{player.name}_vision"
        frame.processed_fields[field_label + "_fov"] = new_camera_settings.fov
        frame.processed_fields[field_label + "_height"] = new_camera_settings.height
        frame.processed_fields[field_label + "_angle"] = new_camera_settings.angle
        frame.processed_fields[field_label + "_distance"] = new_camera_settings.distance
        frame.processed_fields[field_label + "_stiffness"] = (
            new_camera_settings.stiffness
        )
        frame.processed_fields[field_label + "_transition"] = (
            new_camera_settings.transition
        )

    return None


def process_camera_settings_swivel(
    processor: Frame_By_Frame_Processor, updated_actor: Actor, frame: Frame
):
    player = frame.game.camera_settings[updated_actor.actor_id].player
    player.camera_settings.update_settings(updated_actor)
    if processor.include_simple_vision:
        field_label = f"{player.name}_vision_{updated_actor.secondary_category}"
        frame.processed_fields[field_label] = Player_Component(updated_actor).amount
    return None


def process_camera_settings_activate(
    processor: Frame_By_Frame_Processor, updated_actor: Actor, frame: Frame
):
    player = frame.game.camera_settings[updated_actor.actor_id].player
    player.camera_settings.update_settings(updated_actor)
    if processor.include_simple_vision:
        field_label = f"{player.name}_vision_{updated_actor.secondary_category}"
        frame.processed_fields[field_label] = Player_Component(updated_actor).active
    return None
