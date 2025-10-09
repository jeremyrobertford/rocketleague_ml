from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Any
from rocketleague_ml.config import ROUND_LENGTH
from rocketleague_ml.types.attributes import (
    Raw_Game_Data,
    Raw_Frame,
)
from rocketleague_ml.core.actor import Actor
from rocketleague_ml.core.rigid_body import Rigid_Body
from rocketleague_ml.core.ball import Ball
from rocketleague_ml.core.car import Car, Boost_Car_Component, Car_Component
from rocketleague_ml.core.camera_settings import Camera_Settings
from rocketleague_ml.core.player import Player


if TYPE_CHECKING:
    from rocketleague_ml.core.frame import Frame


class Game:

    def __init__(self, game_data: Raw_Game_Data):
        self.id = game_data["properties"]["Id"]
        self.names = {i: obj for i, obj in enumerate(game_data["names"])}
        self.objects = {i: obj for i, obj in enumerate(game_data["objects"])}
        self._previous_frame: Frame | None = None
        self.initial_frame: Raw_Frame = game_data["network_frames"]["frames"][0]
        self.frames: List[Raw_Frame] = game_data["network_frames"]["frames"][1:]
        self.active_clock: bool = False
        self.active: bool = False
        self.round: int = 0
        self.rounds: Dict[int, Any] = {}
        self.players: Dict[int, Player] = {}
        self.boost_pads: Dict[int, Rigid_Body] = {}
        self.match_time_remaining: float = ROUND_LENGTH + 3
        self.in_overtime: bool = False
        self.overtime_elapsed: float = 0.0
        self.processed_frames: List[Frame] = []
        return None

    @property
    def previous_frame(self) -> Frame:
        pf = self._previous_frame
        if pf is None:
            raise ValueError("Previous frame is not assigned")
        return pf

    @previous_frame.setter
    def previous_frame(self, new_frame: Frame):
        self._previous_frame = new_frame

    @property
    def ball(self) -> Ball:
        b = self._ball
        if b is None:
            raise ValueError("Ball not assigned")
        return b

    def connect_disconnected_car_component(self, connection: Actor):
        if connection.active_actor_id is None:
            raise ValueError(
                f"Connecting disconnected car components requires active actor id {connection.raw}"
            )
        car = self.cars[connection.active_actor_id]
        disconnected_car_component = self.disconnected_car_components[
            connection.actor_id
        ]
        car_component = Car_Component(disconnected_car_component, car)
        match car_component.secondary_category:
            case "boost":
                car_component = Boost_Car_Component(car_component)
                car.boost = car_component
            case "dodge":
                car.dodge = car_component
            case "double_jump":
                car.double_jump = car_component
            case "flip":
                car.flip = car_component
            case "jump":
                car.jump = car_component
            case _:
                raise ValueError(f"Unknown car component {car_component.raw}")
        del self.disconnected_car_components[connection.actor_id]
        self.car_components[car_component.actor_id] = car_component
        return None

    def connect_disconnected_player_car(self, connection: Actor):
        if connection.active_actor_id is None:
            raise ValueError(
                f"Connecting disconnected cars requires active actor id {connection.raw}"
            )
        player = self.players[connection.active_actor_id]
        disconnected_car = self.disconnected_cars[connection.actor_id]
        car = Car(disconnected_car, player)
        player.assign_car(car)

        del self.disconnected_cars[connection.actor_id]
        self.cars[car.actor_id] = car
        return None

    def process_setting_new_actor(
        self,
        new_actor: Actor,
        players: Dict[int, Player],
        player_cars: Dict[int, int],
        player_teams: Dict[int, Dict[int, int]],
        car_components: Dict[int, int],
        player_camera_settings: Dict[int, int],
        camera_settings: Dict[int, Actor],
        unhandled_new_actors: Dict[int, Actor],
        delayed_new_actors: Dict[int, Actor],
    ):
        match new_actor.category:
            case "car":
                if new_actor.actor_id not in player_cars:
                    self.disconnected_cars[new_actor.actor_id] = new_actor
                    return None
                player_actor_id = player_cars[new_actor.actor_id]
                player = players[player_actor_id]
                car = Car(new_actor, player)
                player.assign_car(car)
                self.players[player.actor_id] = player
                self.cars[car.actor_id] = car
            case "ball":
                self._ball = Ball(new_actor)
            case "car_component":
                if new_actor.actor_id not in car_components:
                    self.disconnected_car_components[new_actor.actor_id] = new_actor
                    return None
                car_actor_id = car_components[new_actor.actor_id]
                if car_actor_id not in self.cars:
                    delayed_new_actors[new_actor.actor_id] = new_actor
                    return None
                car = self.cars[car_actor_id]
                car_component = Car_Component(new_actor, car)
                car_component.car = car
                match new_actor.secondary_category:
                    case "boost":
                        car_component = Boost_Car_Component(car_component)
                        car.boost = car_component
                    case "dodge":
                        car.dodge = car_component
                    case "double_jump":
                        car.double_jump = car_component
                    case "flip":
                        car.flip = car_component
                    case "jump":
                        car.jump = car_component
                    case _:
                        raise ValueError(f"Unknown car component {new_actor.raw}")
                self.car_components[new_actor.actor_id] = car_component
            case "player_component":
                self.teams[new_actor.actor_id] = new_actor
                for player_actor_id in player_teams[new_actor.actor_id].values():
                    player = (
                        self.players.get(player_actor_id) or players[player_actor_id]
                    )
                    player.assign_team(new_actor.secondary_category)
            case "camera_settings":
                raw_camera_settings = camera_settings[new_actor.actor_id]
                player_actor_id = player_camera_settings[new_actor.actor_id]
                player = players[player_actor_id]
                players_camera_settings = Camera_Settings(raw_camera_settings, player)
                player.assign_camera_settings(players_camera_settings)
                if not player.camera_settings:
                    raise ValueError(f"Camera settings not assigned {new_actor.raw}")
                self.camera_settings[new_actor.actor_id] = player.camera_settings
            case _:
                unhandled_new_actors[new_actor.actor_id] = new_actor

    def set_actors(self, frame: Frame):
        self._ball: Ball | None = None
        self.camera_settings: Dict[int, Camera_Settings] = {}
        self.cars: Dict[int, Car] = {}
        self.car_components: Dict[int, Car_Component | Boost_Car_Component] = {}
        self.disconnected_cars: Dict[int, Actor] = {}
        self.disconnected_car_components: Dict[int, Actor] = {}
        self.disconnected_car_component_updates: Dict[int, List[Actor]] = {}
        self.teams: Dict[int, Actor] = {}

        players: Dict[int, Player] = {}
        player_cars: Dict[int, int] = {}
        player_teams: Dict[int, Dict[int, int]] = {}
        car_components: Dict[int, int] = {}
        player_camera_settings: Dict[int, int] = {}
        camera_settings: Dict[int, Actor] = {}
        unhandled_updated_actors: Dict[int, Actor] = {}
        for ua in frame.updated_actors:
            updated_actor = Actor(ua, self.objects)
            match updated_actor.category:
                case "player":
                    players[updated_actor.actor_id] = Player(updated_actor)
                case "car_to_player":
                    if updated_actor.active_actor_id is None:
                        raise ValueError(
                            f"Car to player does not have active actor {updated_actor.raw}"
                        )
                    player_cars[updated_actor.actor_id] = updated_actor.active_actor_id
                case "vehicle":
                    if updated_actor.active_actor_id is None:
                        raise ValueError(
                            f"Component to car does not have active actor {updated_actor.raw}"
                        )
                    car_components[updated_actor.actor_id] = (
                        updated_actor.active_actor_id
                    )
                case "settings_to_player":
                    if updated_actor.active_actor_id is None:
                        raise ValueError(
                            f"Camera settings to player does not have active actor {updated_actor.raw}"
                        )
                    player_camera_settings[updated_actor.actor_id] = (
                        updated_actor.active_actor_id
                    )
                case "player_component":
                    if not updated_actor.secondary_category == "team":
                        unhandled_updated_actors[updated_actor.actor_id] = updated_actor
                        continue

                    if updated_actor.active_actor_id is None:
                        raise ValueError(
                            f"Player component to player does not have active actor {updated_actor.raw}"
                        )
                    if updated_actor.active_actor_id not in player_teams:
                        player_teams[updated_actor.active_actor_id] = {}
                    player_teams[updated_actor.active_actor_id][
                        updated_actor.actor_id
                    ] = updated_actor.actor_id
                case "camera_settings":
                    camera_settings[updated_actor.actor_id] = updated_actor
                case _:
                    unhandled_updated_actors[updated_actor.actor_id] = updated_actor

        delayed_new_actors: Dict[int, Actor] = {}
        unhandled_new_actors: Dict[int, Actor] = {}
        for na in frame.new_actors:
            new_actor = Actor(na, self.objects)
            self.process_setting_new_actor(
                new_actor,
                players,
                player_cars,
                player_teams,
                car_components,
                player_camera_settings,
                camera_settings,
                unhandled_new_actors,
                delayed_new_actors,
            )

        for new_actor in delayed_new_actors.values():
            self.process_setting_new_actor(
                new_actor,
                players,
                player_cars,
                player_teams,
                car_components,
                player_camera_settings,
                camera_settings,
                unhandled_new_actors,
                delayed_new_actors,
            )

        return None

    def activate_game(self):
        self.round += 1
        self.active = True
        return None

    def deactivate_game(self):
        self.active = False
        self.active_clock = False

        # Unsure why this is necessary, but it makes the match time
        # line up better
        self.match_time_remaining += 3

        return None

    def update_player_car(self, player: Player, new_car: Car):
        old_car_actor_id = player.car.actor_id
        player.assign_car(new_car)
        del self.cars[old_car_actor_id]
        self.cars[new_car.actor_id] = new_car
        return None
