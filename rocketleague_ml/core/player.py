from typing import List, cast
from rocketleague_ml.utils.helpers import convert_byte_to_float
from rocketleague_ml.types.attributes import String_Attribute, Actor_Export
from rocketleague_ml.core.actor import Actor
from rocketleague_ml.core.car import Car
from rocketleague_ml.core.camera_settings import CameraSettings


class Player_Component(Actor):
    def __init__(self, car_component: Actor):
        super().__init__(car_component.raw, car_component.objects)
        self.amount = 0.0

    def update_amount(self, updated_car_component: Actor):
        attribute = updated_car_component.raw["attribute"]
        if not attribute or ("Float" not in attribute and "Byte" not in attribute):
            raise ValueError("Car component updated without value {car_component.raw}")

        if "Float" in attribute:
            self.amount = attribute["Float"]
            return
        if "Byte" in attribute:
            self.amount = convert_byte_to_float(attribute["Byte"])
            return

        raise ValueError("Car component updated with unknown value {car_component.raw}")


class Player(Actor):
    def __init__(self, player: Actor):
        super().__init__(player.raw, player.objects)
        attribute = cast(String_Attribute, player.raw["attribute"])
        self.name = attribute["String"]
        self.car: Car | None = None
        self.team: str | None = None
        self.steering_sensitivity: float | None = None
        self.score: int = 0
        self.camera_settings: CameraSettings | None = None
        self.components = []

    def assign_car(self, cars: List[Car], player_car_connections: List[Actor]):
        for connection in player_car_connections:
            if not self.owns(connection):
                continue

            for car in cars:
                if car.is_self(connection):
                    self.car = car
                    break

    def assign_team(
        self,
        teams: List[Player_Component],
        player_team_connections: List[Player_Component],
    ):
        for connection in player_team_connections:
            if not self.is_self(connection):
                continue

            for team in teams:
                if team.owns(connection):
                    self.team = team.secondary_category
                    break

    def assign_camera_settings(
        self, camera_settings: List[Actor], camera_settings_connections: List[Actor]
    ):
        for camera_setting in camera_settings_connections:
            if self.owns(camera_setting):
                self.camera_settings = CameraSettings(camera_setting)
                break

        if not self.camera_settings:
            raise ValueError("Failed to assign camera settings to player {self.name}")

        for camera_setting in camera_settings:
            if self.camera_settings.is_self(camera_setting):
                self.camera_settings.update_settings(camera_setting)

    def update_camera_settings(self, camera_settings: List[Actor]):
        if not self.camera_settings:
            return
        for camera_setting in camera_settings:
            if self.camera_settings.is_self(camera_setting):
                self.camera_settings.update_settings(camera_setting)

    def update_components(self, player_components: List[Player_Component]):
        skip_objects = [
            "TAGame.PRI_TA:ViralItemActor",
            "TAGame.PRI_TA:SpectatorShortcut",
            "TAGame.PRI_TA:Title",
            "TAGame.PRI_TA:PartyLeader",
            "TAGame.PRI_TA:ClientLoadoutsOnline",
            "TAGame.PRI_TA:ClientLoadouts",
            "TAGame.PRI_TA:PersistentCamera",
        ]
        for pc in player_components:
            player_component = Player_Component(pc)
            if player_component.object in skip_objects or not self.owns(
                player_component
            ):
                continue

            if player_component.secondary_category == "steering_sensitivity":
                self.steering_sensitivity = player_component.amount
            elif player_component.object == "TAGame.PRI_TA:MatchScore":
                self.score = int(player_component.amount)

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
