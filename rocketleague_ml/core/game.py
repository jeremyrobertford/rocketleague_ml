from typing import Dict, Any, List
from rocketleague_ml.types.attributes import Raw_Game_Data
from rocketleague_ml.core.actor import Actor
from rocketleague_ml.core.car import Car, Car_Component
from rocketleague_ml.core.player import Player, Player_Component


class Game:

    def __init__(self, game_data: Raw_Game_Data):
        self.objects = {i: obj for i, obj in enumerate(game_data["objects"])}
        self.frames = game_data["network_frames"]["frames"]
        self.active = False
        self.players: Dict[int, Player] = {}
        self.get_players()

    def get_players(self):
        new_actors = self.frames[0]["new_actors"]
        updated_actors = self.frames[0]["updated_actors"]

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
        camera_settings: List[Actor] = []
        player_components: List[Player_Component] = []
        unmatched_car_components: List[Car_Component] = []
        unmatched_vehicles: Dict[int, int | None] = {}

        for ua in updated_actors:
            updated_actor = Actor(ua, self.objects)
            if updated_actor.category == "vehicle":
                car_component_matched_to_vehicle = False
                for car_component in car_components:
                    if car_component.is_self(updated_actor):
                        car_component.active_actor_id = updated_actor.active_actor_id
                        break

                if not car_component_matched_to_vehicle:
                    unmatched_vehicles[updated_actor.actor_id] = (
                        updated_actor.active_actor_id
                    )
                continue

            if updated_actor.category == "car_to_player":
                cars_to_players.append(updated_actor)
                continue

            if updated_actor.category == "car_component":
                unmatched_car_components.append(Car_Component(updated_actor))
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

        for unmatched_car_component in unmatched_car_components:
            unmatched_car_component.active_actor_id = unmatched_vehicles[
                unmatched_car_component.actor_id
            ]
            car_components.append(unmatched_car_component)

        for car in cars:
            car.assign_components(car_components)

        for p in players:
            player = Player(p)
            player.assign_car(cars, cars_to_players)
            player.assign_camera_settings(camera_settings)
            player.update_components(player_components)
            self.players[player.actor_id] = player

    def analyze_frame(self, frame: Dict[str, Any]):

        if frame["time"] >= 22 and frame["time"] <= 25:
            print(1)

        return
