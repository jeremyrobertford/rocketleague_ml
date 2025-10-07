from typing import cast
from rocketleague_ml.types.attributes import String_Attribute, Actor_Export
from rocketleague_ml.core.actor import Actor
from rocketleague_ml.core.car import Car
from rocketleague_ml.core.camera_settings import Camera_Settings


class Player(Actor):
    def __init__(self, player: Actor):
        super().__init__(player.raw, player.objects)
        attribute = cast(String_Attribute, player.raw["attribute"])
        self.name = attribute["String"]
        self._car: Car | None = None
        self.team: str | None = None
        self.steering_sensitivity: float | None = None
        self.score: int = 0
        self.camera_settings: Camera_Settings | None = None
        self.components = []

    @property
    def car(self) -> Car:
        c = self._car
        if c is None:
            raise ValueError("Car not assigned")
        return c

    def assign_car(self, car: Car):
        if not self._car:
            self._car = car
            car.player = self
            return None

        self._car.actor_id = car.actor_id
        self._car.update_component_connections()

    def assign_camera_settings(self, camera_settings: Actor):
        if not self.camera_settings:
            self.camera_settings = Camera_Settings(camera_settings, self)
        else:
            self.camera_settings.update_settings(camera_settings)
        return None

    def update_camera_settings(self, camera_settings: Actor):
        if not self.camera_settings:
            raise ValueError(
                f"Player does not have camera settings assigned {camera_settings.raw}"
            )
        self.camera_settings.update_settings(camera_settings)

    def to_dict(self):
        base_player = super().to_dict()
        player: Actor_Export = {
            "name": self.name,
            "car": self.car.to_dict() if self.car else None,
            "steering_sensitivity": self.steering_sensitivity,
            "score": self.score,
            "camera_settings": (
                self.camera_settings.to_dict() if self.camera_settings else None
            ),
        }
        return cast(Actor_Export, base_player | player)
