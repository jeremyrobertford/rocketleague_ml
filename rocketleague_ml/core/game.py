from typing import Dict, List, Any, cast
from rocketleague_ml.config import ROUND_LENGTH
from rocketleague_ml.utils.helpers import parse_boost_actor_name
from rocketleague_ml.types.attributes import (
    Categorized_Updated_Actors,
    Categorized_New_Actors,
    Raw_Game_Data,
    Raw_Frame,
    Actor_Export,
    Demolition_Attribute,
    Boost_Grab_Attribute,
)
from rocketleague_ml.core.actor import Actor
from rocketleague_ml.core.ball import Ball
from rocketleague_ml.core.car import Car, Boost_Car_Component, Car_Component
from rocketleague_ml.core.player import PlayerWithCar, Player, Player_Component


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
        self._ball: Ball | None = None
        self.players: Dict[int, PlayerWithCar] = {}
        self.boost_pads: Dict[int, Actor] = {}
        self.get_players()

        self.match_time_remaining: float = ROUND_LENGTH
        self.in_overtime: bool = False
        self.overtime_elapsed: float = 0.0

    @property
    def ball(self) -> Ball:
        b = self._ball
        if b is None:
            raise ValueError("Ball not assigned")
        return b

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
                self._ball = Ball(new_actor)
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

            if updated_actor.secondary_category == "rep_boost":
                secondary_car_components.append(Boost_Car_Component(updated_actor))
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
            self.players[player.actor_id] = PlayerWithCar(player)

    def activate_game(self, frame: Raw_Frame):
        self.round += 1
        self.active = True
        self.rounds[self.round] = {"start_time": frame["time"]}
        if not self.last_frame:
            raise ValueError("No last inactive frame in inactive game")
        self.ball.update_position(self.ball, self.active, self.last_frame, self.round)
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
        if self.ball.is_self(updated_actor):
            self.ball.update_position(updated_actor, self.active, frame, self.round)
            return

        for player in self.players.values():
            if player.car.is_self(updated_actor):
                player.car.update_position(
                    updated_actor, self.active, frame, self.round
                )
                return True
            if player.car.boost and player.car.boost.actor_id == updated_actor.actor_id:
                player.car.boost.update_position(
                    updated_actor, self.active, frame, self.round
                )
                return True
            if player.car.jump and player.car.jump.actor_id == updated_actor.actor_id:
                player.car.jump.update_position(
                    updated_actor, self.active, frame, self.round
                )
                return True
            if player.car.dodge and player.car.dodge.actor_id == updated_actor.actor_id:
                player.car.dodge.update_position(
                    updated_actor, self.active, frame, self.round
                )
                return True
            if player.car.flip and player.car.flip.actor_id == updated_actor.actor_id:
                player.car.flip.update_position(
                    updated_actor, self.active, frame, self.round
                )
                return True
            if (
                player.car.double_jump
                and player.car.double_jump.actor_id == updated_actor.actor_id
            ):
                player.car.double_jump.update_position(
                    updated_actor, self.active, frame, self.round
                )
                return True
        return False

    def update_actor_activity(self, updated_actor: Actor, frame: Raw_Frame):
        if updated_actor.category == "car_component":
            for player in self.players.values():
                car = player.car
                if car.is_self(updated_actor) or car.owns(updated_actor):
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

    def categorize_new_actor(
        self,
        new_actor: Actor,
        frame: Raw_Frame,
        categorized_actors: Categorized_New_Actors,
    ):
        if ".TheWorld:PersistentLevel.VehiclePickup_Boost_TA_" in new_actor.object:
            self.boost_pads[new_actor.actor_id] = new_actor
            return

        if new_actor.category == "car":
            categorized_actors["cars"].append(Car(new_actor))
            return

        if new_actor.category == "car_component":
            categorized_actors["car_components"].append(Car_Component(new_actor))
            return

        return

    def categorize_new_actors(self, frame: Raw_Frame):
        categorized_actors: Categorized_New_Actors = {"cars": [], "car_components": []}
        for na in frame["new_actors"]:
            new_actor = Actor(na, self.objects)
            self.categorize_new_actor(new_actor, frame, categorized_actors)
        return categorized_actors

    def categorize_updated_actor(
        self,
        updated_actor: Actor,
        frame: Raw_Frame,
        categorized_actors: Categorized_Updated_Actors,
        new_actors: Categorized_New_Actors,
    ):
        if "event" in updated_actor.category:
            categorized_actors["events"].append(updated_actor)
            return

        if updated_actor.category == "car_to_player" and updated_actor.active_actor_id:
            categorized_actors["cars_for_players"].append(updated_actor)
            return

        if updated_actor.category == "vehicle" and updated_actor.active_actor_id:
            for car_component in new_actors["car_components"]:
                if car_component.is_self(updated_actor):
                    car_component.active_actor_id = updated_actor.active_actor_id
                    return

        if ".TheWorld:PersistentLevel.VehiclePickup_Boost_TA_" in updated_actor.object:
            self.boost_pads[updated_actor.actor_id] = updated_actor
            return

        if updated_actor.category == "game_start_event" and not frame["resync"]:
            attribute = updated_actor.raw.get("attribute")
            if not attribute or "Int" not in attribute:
                raise ValueError("Game active state not valid {updated_actor.raw}")
            name = self.names[attribute["Int"]]
            if name == "Active":
                self.activate_game(frame)
            elif name == "PostGoalScored":
                self.deactivate_game(frame)
            return

        categorized_actors["others"].append(updated_actor)
        return

    def categorize_updated_actors(
        self, frame: Raw_Frame, new_actors: Categorized_New_Actors
    ):
        categorized_actors: Categorized_Updated_Actors = {
            "events": [],
            "others": [],
            "cars_for_players": [],
        }
        for ua in frame["updated_actors"]:
            updated_actor = Actor(ua, self.objects)
            self.categorize_updated_actor(
                updated_actor, frame, categorized_actors, new_actors
            )
        return categorized_actors

    def process_updated_actor(self, updated_actor: Actor, frame: Raw_Frame):
        attribute = updated_actor.raw.get("attribute")
        if not attribute:
            return

        attribute_key = [*attribute.keys()][0]

        match attribute_key:
            case "Boolean":
                self.update_actor_activity(updated_actor, frame)
                return
            case "RigidBody":
                self.update_actor_position(updated_actor, frame)
                return
            case "DemolishExtended":
                demolition = cast(Demolition_Attribute, attribute)
                attacker_car = None
                victim_car = None
                for player in self.players.values():
                    if demolition["attacker"]["actor"] == player.car.actor_id:
                        attacker_car = player.car
                    if demolition["victim"]["actor"] == player.car.actor_id:
                        victim_car = player.car
                if not attacker_car:
                    raise ValueError(
                        f"No matching attacker car in demo actor {updated_actor.raw}"
                    )
                if not victim_car:
                    raise ValueError(
                        f"No matching victim car in demo actor {updated_actor.raw}"
                    )
                attacker_car.demo(victim_car, updated_actor, frame, self.round)
                victim_car.demod(attacker_car, updated_actor, frame, self.round)
            case "PickupNew":
                boost_grab = cast(Boost_Grab_Attribute, attribute)
                boost_actor = self.boost_pads[updated_actor.actor_id]
                if not boost_actor:
                    raise ValueError(
                        f"Cannot find boost actor {updated_actor.actor_id}"
                    )
                boost_data = parse_boost_actor_name(boost_actor.object)
                if not boost_data:
                    raise ValueError(
                        f"Boost data not found for boost actor {boost_actor.object}"
                    )
                for player in self.players.values():
                    if not player.car.boost:
                        raise ValueError(
                            f"No matching player car boost for boost pickup {updated_actor.raw}"
                        )
                    if player.car.actor_id == boost_grab["instigator"]:
                        player.car.boost.pickup(
                            boost_data, updated_actor, frame, self.round
                        )
                        return
                raise ValueError(
                    f"Player car boost pickup never updated {updated_actor.raw}"
                )
            case _:
                pass

        match updated_actor.category:
            case "car_component":
                for player in self.players.values():
                    if player.car.owns(updated_actor) or player.car.is_self(
                        updated_actor
                    ):
                        car_component = (
                            Boost_Car_Component(updated_actor)
                            if updated_actor.secondary_category == "rep_boost"
                            else Car_Component(updated_actor)
                        )
                        player.car.update_components(
                            [car_component],
                            self.active,
                            frame,
                            self.round,
                        )
                        return
                raise ValueError(f"Never updated car component {updated_actor.raw}")
            case "camera_settings":
                for player in self.players.values():
                    player.update_camera_settings([updated_actor])
                return
            case "player_component":
                return
            case "team_ball_hit":
                self.ball.team_hit(updated_actor, frame, self.round)
                return
            case _:
                pass

        return

    def process_updated_actors(self, updated_actors: List[Actor], frame: Raw_Frame):
        for updated_actor in updated_actors:
            self.process_updated_actor(updated_actor, frame)

    def analyze_frame(self, frame: Raw_Frame, frame_index: int):
        resync_frame = len(frame["new_actors"]) + len(frame["updated_actors"]) > 80
        if resync_frame:
            frame["resync"] = True

        categorized_new_actors = self.categorize_new_actors(frame)
        categorized_updated_actors = self.categorize_updated_actors(
            frame, categorized_new_actors
        )

        if frame_index == 2045:
            pass

        frame = self.calculate_match_time(frame, frame_index)

        if len(categorized_updated_actors["cars_for_players"]):
            for player in self.players.values():
                player.assign_car(
                    categorized_new_actors["cars"],
                    categorized_updated_actors["cars_for_players"],
                )
                player.car.assign_components(categorized_new_actors["car_components"])

        self.process_updated_actors(categorized_updated_actors["others"], frame)

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
