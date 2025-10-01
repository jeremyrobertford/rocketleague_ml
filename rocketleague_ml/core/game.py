from typing import Dict, List, Any
from rocketleague_ml.config import ROUND_LENGTH
from rocketleague_ml.types.attributes import (
    Raw_Game_Data,
    Raw_Frame,
    Actor_Export,
    Demolition_Attribute,
)
from rocketleague_ml.core.actor import Actor
from rocketleague_ml.core.ball import Ball
from rocketleague_ml.core.car import Car, Car_Component
from rocketleague_ml.core.player import Player, Player_Component


class Game:

    def __init__(self, game_data: Raw_Game_Data):
        self.id = game_data["properties"]["Id"]
        self.names = {i: obj for i, obj in enumerate(game_data["names"])}
        self.objects = {i: obj for i, obj in enumerate(game_data["objects"])}
        self.initial_frame = game_data["network_frames"]["frames"][0]
        self.frames: List[Raw_Frame] = game_data["network_frames"]["frames"][1:]
        self.last_frame: Raw_Frame | None = self.initial_frame
        self.active_clock = False
        self.active = False
        self.round = 0
        self.rounds: Dict[int, Any] = {}
        self.ball: Ball | None = None
        self.players: Dict[int, Player] = {}
        self.get_players()

        self.match_time_remaining: float = ROUND_LENGTH
        self.in_overtime: bool = False
        self.overtime_elapsed: float = 0.0

    def get_players(self):
        new_actors = self.initial_frame["new_actors"]
        updated_actors = self.initial_frame["updated_actors"]

        cars: List[Car] = []
        car_components: List[Car_Component] = []
        teams: List[Player_Component] = []

        for na in new_actors:
            new_actor = Actor(na, self.objects)
            if new_actor.category == "car":
                cars.append(Car(new_actor))
                continue
            if new_actor.category == "ball":
                self.ball = Ball(new_actor)
                continue
            if new_actor.category == "car_component":
                car_components.append(Car_Component(new_actor))
                continue
            if new_actor.category == "player_component":
                teams.append(Player_Component(new_actor))

        players: List[Actor] = []
        cars_to_players: List[Actor] = []
        settings_to_players: List[Actor] = []
        camera_settings: List[Actor] = []
        player_components: List[Player_Component] = []
        secondary_car_components: List[Car_Component] = []

        for ua in updated_actors:
            updated_actor = Actor(ua, self.objects)
            if updated_actor.category == "vehicle":
                for car_component in car_components:
                    if car_component.is_self(updated_actor):
                        car_component.active_actor_id = updated_actor.active_actor_id
                        break
                continue

            if updated_actor.category == "car_to_player":
                cars_to_players.append(updated_actor)
                continue

            if updated_actor.category == "settings_to_player":
                settings_to_players.append(updated_actor)
                continue

            if updated_actor.category == "car_component":
                secondary_car_components.append(Car_Component(updated_actor))
                continue

            if updated_actor.category == "player":
                players.append(updated_actor)
                continue

            if updated_actor.category == "camera_settings":
                camera_settings.append(updated_actor)
                continue

            if updated_actor.category == "player_component":
                player_components.append(Player_Component(updated_actor))
                continue

        for car in cars:
            car.assign_components(car_components)
            car.update_components(
                secondary_car_components, self.active, self.initial_frame, self.round
            )

        for p in players:
            player = Player(p)
            player.assign_car(cars, cars_to_players)
            player.assign_team(teams, player_components)
            player.assign_camera_settings(camera_settings, settings_to_players)
            player.update_components(player_components)
            self.players[player.actor_id] = player

    def activate_game(self, frame: Raw_Frame):
        self.round += 1
        self.active = True
        self.rounds[self.round] = {"start_time": frame["time"]}
        if not self.last_frame:
            raise ValueError("No last inactive frame in inactive game")
        if self.ball:
            self.ball.update_position(
                self.ball, self.active, self.last_frame, self.round
            )
        for player in self.players.values():
            if player.car:
                # initialize positions
                player.car.update_position(
                    player.car, self.active, self.last_frame, self.round
                )

    def deactivate_game(self, frame: Raw_Frame):
        self.active = False
        self.active_clock = False
        self.match_time_remaining += 3
        self.rounds[self.round]["end_time"] = frame["time"]

    def update_actor_position(self, updated_actor: Actor, frame: Raw_Frame):
        if self.ball and self.ball.actor_id == updated_actor.actor_id:
            self.ball.update_position(updated_actor, self.active, frame, self.round)
            return True
        for player in self.players.values():
            car = player.car
            if not car:
                return True
            if car.actor_id == updated_actor.actor_id:
                car.update_position(updated_actor, self.active, frame, self.round)
                return True
            if car.boost and car.boost.actor_id == updated_actor.actor_id:
                car.boost.update_position(updated_actor, self.active, frame, self.round)
                return True
            if car.jump and car.jump.actor_id == updated_actor.actor_id:
                car.jump.update_position(updated_actor, self.active, frame, self.round)
                return True
            if car.dodge and car.dodge.actor_id == updated_actor.actor_id:
                car.dodge.update_position(updated_actor, self.active, frame, self.round)
                return True
            if car.flip and car.flip.actor_id == updated_actor.actor_id:
                car.flip.update_position(updated_actor, self.active, frame, self.round)
                return True
            if car.double_jump and car.double_jump.actor_id == updated_actor.actor_id:
                car.double_jump.update_position(
                    updated_actor, self.active, frame, self.round
                )
                return True
        return False

    def update_actor_activity(self, updated_actor: Actor, frame: Raw_Frame):
        if updated_actor.category == "car_compnonent":
            for player in self.players.values():
                car = player.car
                if car and (car.is_self(updated_actor) or car.owns(updated_actor)):
                    car.update_components(
                        [Car_Component(updated_actor)],
                        self.active,
                        frame,
                        self.round,
                    )
                    return True
        return False

    def calculate_match_time(self, frame: Raw_Frame, frame_index: int):
        if self.active and not self.in_overtime:
            self.match_time_remaining = max(
                0.0, self.match_time_remaining - frame["delta"]
            )

            EPS = 1e-9
            if self.match_time_remaining <= EPS:
                self.match_time_remaining = 0.0
                self.in_overtime = True
                self.overtime_elapsed = 0.0
        elif self.active and self.in_overtime:
            self.overtime_elapsed += frame["delta"]

        if not self.in_overtime:
            frame["match_time"] = float(self.match_time_remaining)
            frame["in_overtime"] = False
            mins, secs = divmod(int(self.match_time_remaining), 60)
            frame["match_time_label"] = f"{mins}:{secs:02d}"
        else:
            frame["match_time"] = -float(self.overtime_elapsed)
            frame["in_overtime"] = True
            mins, secs = divmod(int(self.overtime_elapsed), 60)
            frame["match_time_label"] = f"+{mins}:{secs:02d}"

        self.frames[frame_index] = frame
        return frame

    def analyze_frame(self, frame: Raw_Frame, frame_index: int):
        resync_frame = len(frame["new_actors"]) + len(frame["updated_actors"]) > 80
        labeled_new_actors = [  # pyright: ignore[reportUnusedVariable]
            Actor.label(a, self.objects) for a in frame["new_actors"]
        ]
        labeled_updated_actors = [  # pyright: ignore[reportUnusedVariable]
            Actor.label(a, self.objects) for a in frame["updated_actors"]
        ]

        events: List[Actor] = []

        new_cars: Dict[int, Car] = {}
        new_car_components: List[Car_Component] = []
        for na in frame["new_actors"]:
            new_actor = Actor(na, self.objects)
            if new_actor.category == "car":
                new_cars[new_actor.actor_id] = Car(new_actor)
                continue
            if new_actor.category == "car_component":
                new_car_components.append(Car_Component(new_actor))
                continue

        updated_actors: List[Actor] = []
        new_cars_for_players: Dict[int, int] = {}
        new_components_for_cars: Dict[int, int] = {}
        for ua in frame["updated_actors"]:
            updated_actor = Actor(ua, self.objects)
            if (
                updated_actor.category == "car_to_player"
                and updated_actor.active_actor_id
            ):
                new_cars_for_players[updated_actor.active_actor_id] = (
                    updated_actor.actor_id
                )
                continue

            if updated_actor.category == "vehicle" and updated_actor.active_actor_id:
                new_components_for_cars[updated_actor.actor_id] = (
                    updated_actor.active_actor_id
                )
                continue

            if updated_actor.category == "round_countdown_event":
                self.active_clock = True
                continue

            if updated_actor.category == "game_start_event" and not resync_frame:
                attribute = updated_actor.raw.get("attribute")
                if not attribute or "Int" not in attribute:
                    raise ValueError("Game active state not valid {updated_actor.raw}")
                name = self.names[attribute["Int"]]
                if name == "Active":
                    self.activate_game(frame)
                elif name == "PostGoalScored":
                    self.deactivate_game(frame)
                continue

            updated_actors.append(updated_actor)

        if frame_index == 2045:
            pass

        frame = self.calculate_match_time(frame, frame_index)

        for p in new_cars_for_players:
            if p == -1:
                continue
            player = self.players[p]
            car_actor_id = new_cars_for_players[p]
            if not car_actor_id or len(new_cars) == 0:
                continue
            car = new_cars[car_actor_id]
            if not car or not player.car:
                continue
            player.car.actor_id = car.actor_id

            if len(new_components_for_cars) == 0:
                continue
            for car_component in new_car_components:
                if car_component.actor_id in new_components_for_cars:
                    active_actor_id = new_components_for_cars[car_component.actor_id]
                    car_component.active_actor_id = active_actor_id
            player.car.assign_components(new_car_components)

        for updated_actor in updated_actors:
            if updated_actor.actor_id == 8:
                events.append(updated_actor)
                continue

            attribute = updated_actor.raw.get("attribute")
            if not attribute:
                continue

            if frame_index >= 245:
                pass

            if frame_index >= 889:
                pass

            if updated_actor.actor_id == 52 or updated_actor.active_actor_id == 52:
                pass

            activated_actor = attribute.get("Boolean")
            if activated_actor is not None:
                changed = self.update_actor_activity(updated_actor, frame)
                if changed:
                    continue

            rigid_body = attribute.get("RigidBody")
            if rigid_body:
                changed = self.update_actor_position(updated_actor, frame)
                if changed:
                    continue

            if updated_actor.category == "car_component":
                for player in self.players.values():
                    if not player.car:
                        continue
                    if player.car.owns(updated_actor) or player.car.is_self(
                        updated_actor
                    ):
                        player.car.update_components(
                            [Car_Component(updated_actor)],
                            self.active,
                            frame,
                            self.round,
                        )

            demolition: Demolition_Attribute | None = attribute.get("DemolishExtended")
            if demolition:
                attacker_car = None
                victim_car = None
                for player in self.players.values():
                    if not player.car:
                        continue
                    if demolition["attacker"]["actor"] == player.car.actor_id:
                        attacker_car = player.car
                    if demolition["victim"]["actor"] == player.car.actor_id:
                        victim_car = player.car
                if attacker_car and victim_car:
                    attacker_car.demo(victim_car, updated_actor, frame, self.round)
                    victim_car.demod(attacker_car, updated_actor, frame, self.round)
                    continue

        if self.last_frame:
            self.last_frame["active"] = self.active
            self.frames[frame_index - 1] = self.last_frame
        self.last_frame = frame
        return

    def to_dict(self) -> Actor_Export:
        return {
            "rounds": self.round,
            "players": {p: player.to_dict() for p, player in self.players.items()},
        }
