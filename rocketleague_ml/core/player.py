from typing import cast, Literal
from rocketleague_ml.types.attributes import String_Attribute
from rocketleague_ml.core.actor import Actor
from rocketleague_ml.core.car import Car
from rocketleague_ml.core.camera_settings import Camera_Settings


class Player(Actor):
    def __init__(self, player: Actor):
        super().__init__(player.raw, player.objects)
        attribute = cast(String_Attribute, player.raw["attribute"])
        self.name = attribute["String"]
        self._car: Car | None = None
        self._camera_settings: Camera_Settings | None = None
        self._team: Literal["Blue"] | Literal["Orange"] | None = None
        self.steering_sensitivity: float | None = None
        self.score: int = 0
        self.components = []

    @property
    def team(self) -> Literal["Blue"] | Literal["Orange"]:
        t = self._team
        if t is None:
            raise ValueError("Team not assigned")
        return t

    @property
    def car(self) -> Car:
        c = self._car
        if c is None:
            raise ValueError("Car not assigned")
        return c

    @property
    def camera_settings(self) -> Camera_Settings:
        cs = self._camera_settings
        if cs is None:
            raise ValueError("Camera Settings not assigned")
        return cs

    def assign_team(self, team: str | None):
        if team == "Orange":
            self._team = team
            return None
        if team == "Blue":
            self._team = team
            return None
        raise ValueError(f"Unknown team assignment for player: team {team}")

    def assign_car(self, car: Car):
        if not self._car:
            self._car = car
            car.player = self
            return None

        self._car.actor_id = car.actor_id
        self._car.update_component_connections()
        return None

    def assign_camera_settings(self, camera_settings: Actor):
        if not self._camera_settings:
            self._camera_settings = Camera_Settings(camera_settings, self)
            return None

        self.camera_settings.update_settings(camera_settings)
        return None
