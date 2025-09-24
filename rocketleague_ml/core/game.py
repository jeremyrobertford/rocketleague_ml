from typing import Dict, List
from rocketleague_ml.types.attributes import Raw_Game_Data, Raw_Frame
from rocketleague_ml.core.actor import Actor
from rocketleague_ml.core.car import Car, Car_Component
from rocketleague_ml.core.player import Player, Player_Component


class Game:

    def __init__(self, game_data: Raw_Game_Data):
        self.names = {i: obj for i, obj in enumerate(game_data["names"])}
        self.objects = {i: obj for i, obj in enumerate(game_data["objects"])}
        self.initial_frame = game_data["network_frames"]["frames"][0]
        self.frames = game_data["network_frames"]["frames"][1:]
        self.last_frame: Raw_Frame | None = self.initial_frame
        self.active = False
        self.round = 0
        self.cars: Dict[int, Car] = {}
        self.players: Dict[int, Player] = {}
        self.get_players()

    def get_players(self):
        new_actors = self.initial_frame["new_actors"]
        updated_actors = self.initial_frame["updated_actors"]

        cars: List[Car] = []
        car_components: List[Car_Component] = []

        for na in new_actors:
            new_actor = Actor(na, self.objects)
            if new_actor.category == "car":
                cars.append(Car(new_actor))
                continue
            if new_actor.category == "car_component":
                car_components.append(Car_Component(new_actor))

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
            car.update_components(secondary_car_components)

        for p in players:
            player = Player(p)
            player.assign_car(cars, cars_to_players)
            player.assign_camera_settings(camera_settings, settings_to_players)
            player.update_components(player_components)
            self.players[player.actor_id] = player
            if player.car:
                self.cars[player.car.actor_id] = player.car

    def activate_game(self):
        self.round += 1
        self.active = True
        if not self.last_frame:
            raise ValueError("No last inactive frame in inactive game")
        for car in self.cars.values():
            # initialize positions
            car.update_position(car, self.active, self.last_frame, self.round)

    def deactivate_game(self):
        self.active = False

    def resync(self, frame: Raw_Frame):
        self.last_frame = frame
        return

    def analyze_frame(self, frame: Raw_Frame, frame_index: int | None = None):
        resync_frame = len(frame["new_actors"]) + len(frame["updated_actors"]) > 80

        if frame_index and frame_index > 1754 and frame_index < 1780:
            print(2)

        events: List[Actor] = []

        updated_actors: List[Actor] = []
        for ua in frame["updated_actors"]:
            updated_actor = Actor(ua, self.objects)
            updated_actors.append(updated_actor)

            if updated_actor.category == "game_start_event" and not resync_frame:
                attribute = updated_actor.raw.get("attribute")
                if not attribute or "Int" not in attribute:
                    raise ValueError("Game active state not valid {updated_actor.raw}")
                name = self.names[attribute["Int"]]
                if name == "Active":
                    self.activate_game()
                elif name == "PostGoalScored":
                    self.deactivate_game()
                else:
                    print(name)

        for updated_actor in updated_actors:
            if updated_actor.actor_id == 8:
                events.append(updated_actor)
            attribute = updated_actor.raw.get("attribute")
            if not attribute:
                continue
            rigid_body = attribute.get("RigidBody")
            if not rigid_body:
                continue

            car = self.cars.get(updated_actor.actor_id)
            if car:
                car.update_position(updated_actor, self.active, frame, self.round)

        if len(events) > 0:
            print(1)

        self.last_frame = frame
        return
