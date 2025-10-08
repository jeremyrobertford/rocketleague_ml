from __future__ import annotations
from typing import TYPE_CHECKING
from rocketleague_ml.core.actor import Actor
from rocketleague_ml.utils.helpers import convert_byte_to_float

if TYPE_CHECKING:
    from rocketleague_ml.core.player import Player


class Camera_Settings(Actor):
    def __init__(self, camera_settings: Actor, player: Player):
        super().__init__(camera_settings.raw, camera_settings.objects)
        self.yaw: float | None = None
        self.pitch: float | None = None
        self.car_cam: bool = False
        self.rear_cam: bool = False
        self.fov: float | None = None
        self.height: float | None = None
        self.angle: float | None = None
        self.distance: float | None = None
        self.stiffness: float | None = None
        self.swivel: float | None = None
        self.transition: float | None = None
        self.player: Player = player

    def update_settings(self, updated_settings: Actor):
        attribute = updated_settings.raw.get("attribute")
        if not attribute:
            raise ValueError("Attribute not found in {updated_settings.raw}")

        if updated_settings.secondary_category == "yaw":
            if "Byte" not in attribute:
                raise ValueError("Yaw not found with bytes {updated_settings.raw}")
            yaw_byte = attribute["Byte"]
            self.yaw = convert_byte_to_float(yaw_byte)
            return

        if updated_settings.secondary_category == "pitch":
            if "Byte" not in attribute:
                raise ValueError("Pitch not found with bytes {updated_settings.raw}")
            pitch_byte = attribute["Byte"]
            self.pitch = convert_byte_to_float(pitch_byte)
            return

        if updated_settings.secondary_category == "car_cam":
            if "Boolean" not in attribute:
                raise ValueError(
                    "Car cam not found with boolean {updated_settings.raw}"
                )
            self.car_cam = attribute["Boolean"]
            return

        if updated_settings.secondary_category == "rear_cam":
            if "Boolean" not in attribute:
                raise ValueError(
                    "Rear cam not found with boolean {updated_settings.raw}"
                )
            self.rear_cam = attribute["Boolean"]
            return

        if updated_settings.secondary_category == "camera_settings":
            if "CamSettings" not in attribute:
                raise ValueError(
                    "Camera settings not found with settings {updated_settings.raw}"
                )
            cam_settings = attribute["CamSettings"]
            self.fov = cam_settings["fov"]
            self.height = cam_settings["height"]
            self.angle = cam_settings["angle"]
            self.distance = cam_settings["distance"]
            self.stiffness = cam_settings["stiffness"]
            self.swivel = cam_settings["swivel"]
            self.transition = cam_settings["transition"]
