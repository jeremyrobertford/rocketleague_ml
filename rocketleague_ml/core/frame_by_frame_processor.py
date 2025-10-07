import os
from pathlib import Path
from typing import cast, Any, Dict, List
import pandas as pd
from rocketleague_ml.config import PREPROCESSED, PROCESSED
from rocketleague_ml.core.actor import Actor
from rocketleague_ml.core.game import Game
from rocketleague_ml.core.frame import Frame
from rocketleague_ml.core.car import Car_Component, Boost_Car_Component
from rocketleague_ml.core.rrrocket_json_preprocessor import RRRocket_JSON_Preprocessor
from rocketleague_ml.utils.helpers import parse_boost_actor_name
from rocketleague_ml.utils.logging import Logger
from rocketleague_ml.types.attributes import (
    Raw_Game_Data,
    Raw_Frame,
    Boolean_Attribute,
    Demolish_Extended_Attribute,
    Pickup_New_Attribute,
    Location_Attribute,
)


class Frame_By_Frame_Processor:
    def __init__(
        self,
        preprocessor: RRRocket_JSON_Preprocessor,
        logger: Logger = Logger(),
        include_ball_positioning: bool = True,
        include_simple_positioning: bool = True,
        include_advanced_positioning: bool = True,
        include_speed: bool = True,
        include_simple_vision: bool = True,
        include_advanced_vision: bool = True,
        include_advanced_movement: bool = True,
        include_steering: bool = True,
        include_player_demos: bool = True,
        include_scoreboard_metrics: bool = True,
        include_boost_management: bool = True,
        # requires rl_utilities
        include_ball_collisions: bool = False,
        include_player_collisions: bool = False,
        include_custom_scoreboard_metrics: bool = False,
        include_pressure: bool = False,
        include_possession: bool = False,
        include_mechanics: bool = False,
    ):
        if include_advanced_positioning and not include_simple_positioning:
            raise ValueError(
                "Cannot enable advanced positioning without simple positioning"
            )
        if include_advanced_vision and not include_simple_vision:
            raise ValueError("Cannot enable advanced vision without simple vision")

        self.config(
            preprocessor,
            logger,
            include_ball_positioning,
            include_simple_positioning,
            include_advanced_positioning,
            include_speed,
            include_simple_vision,
            include_advanced_vision,
            include_advanced_movement,
            include_steering,
            include_player_demos,
            include_scoreboard_metrics,
            include_boost_management,
            include_ball_collisions,
            include_player_collisions,
            include_custom_scoreboard_metrics,
            include_pressure,
            include_possession,
            include_mechanics,
        )
        return None

    def config(
        self,
        preprocessor: RRRocket_JSON_Preprocessor | None = None,
        logger: Logger = Logger(),
        include_ball_positioning: bool | None = None,
        include_simple_positioning: bool | None = None,
        include_advanced_positioning: bool | None = None,
        include_speed: bool | None = None,
        include_simple_vision: bool | None = None,
        include_advanced_vision: bool | None = None,
        include_advanced_movement: bool | None = None,
        include_steering: bool | None = None,
        include_player_demos: bool | None = None,
        include_scoreboard_metrics: bool | None = None,
        include_boost_management: bool | None = None,
        include_ball_collisions: bool | None = None,
        include_player_collisions: bool | None = None,
        include_custom_scoreboard_metrics: bool | None = None,
        include_pressure: bool | None = None,
        include_possession: bool | None = None,
        include_mechanics: bool | None = None,
    ):
        include_simple_positioning = (
            True
            if include_advanced_positioning
            or (
                include_advanced_positioning is None
                and self.include_advanced_positioning
            )
            else self.include_simple_positioning
        )
        self.include_simple_positioning = include_simple_positioning

        include_simple_vision = (
            True
            if include_advanced_vision
            or (include_advanced_vision is None and self.include_advanced_vision)
            else self.include_simple_vision
        )
        self.include_simple_vision = include_simple_vision

        self.preprocessor = preprocessor if preprocessor else self.preprocessor
        self.logger = logger if logger else self.logger
        self.include_ball_positioning = (
            include_ball_positioning
            if include_ball_positioning is not None
            else self.include_ball_positioning
        )
        self.include_advanced_positioning = (
            include_advanced_positioning
            if include_advanced_positioning is not None
            else self.include_advanced_positioning
        )
        self.include_speed = (
            include_speed if include_speed is not None else self.include_speed
        )
        self.include_advanced_vision = (
            include_advanced_vision
            if include_advanced_vision is not None
            else self.include_advanced_vision
        )
        self.include_advanced_movement = (
            include_advanced_movement
            if include_advanced_movement is not None
            else self.include_advanced_movement
        )
        self.include_steering = (
            include_steering if include_steering is not None else self.include_steering
        )
        self.include_player_demos = (
            include_player_demos
            if include_player_demos is not None
            else self.include_player_demos
        )
        self.include_scoreboard_metrics = (
            include_scoreboard_metrics
            if include_scoreboard_metrics is not None
            else self.include_scoreboard_metrics
        )
        self.include_boost_management = (
            include_boost_management
            if include_boost_management is not None
            else self.include_boost_management
        )
        self.include_ball_collisions = (
            include_ball_collisions
            if include_ball_collisions is not None
            else self.include_ball_collisions
        )
        self.include_player_collisions = (
            include_player_collisions
            if include_player_collisions is not None
            else self.include_player_collisions
        )
        self.include_custom_scoreboard_metrics = (
            include_custom_scoreboard_metrics
            if include_custom_scoreboard_metrics is not None
            else self.include_custom_scoreboard_metrics
        )
        self.include_pressure = (
            include_pressure if include_pressure is not None else self.include_pressure
        )
        self.include_possession = (
            include_possession
            if include_possession is not None
            else self.include_possession
        )
        self.include_mechanics = (
            include_mechanics
            if include_mechanics is not None
            else self.include_mechanics
        )
        return None

    def add_updated_actor_to_disconnected_car_component_updates(
        self, updated_actor: Actor, frame: Frame
    ):
        if updated_actor.actor_id not in frame.game.disconnected_car_component_updates:
            frame.game.disconnected_car_component_updates[updated_actor.actor_id] = []
        frame.game.disconnected_car_component_updates[updated_actor.actor_id].append(
            updated_actor
        )
        return None

    def process_updated_actor(self, updated_actor: Actor, frame: Frame):
        if ".TheWorld:PersistentLevel.VehiclePickup_Boost_TA_" in updated_actor.object:
            frame.game.boost_pads[updated_actor.actor_id] = updated_actor
            return None

        if (
            updated_actor.actor_id in frame.game.disconnected_cars
            and updated_actor.active_actor_id in frame.game.players
        ):
            frame.game.connect_disconnected_player_car(updated_actor)
            return None

        if updated_actor.actor_id in frame.game.disconnected_car_components:
            if (
                updated_actor.active_actor_id
                and updated_actor.active_actor_id in frame.game.cars
            ):
                frame.game.connect_disconnected_car_component(updated_actor)
                if (
                    updated_actor.actor_id
                    in frame.game.disconnected_car_component_updates
                ):
                    for update in frame.game.disconnected_car_component_updates[
                        updated_actor.actor_id
                    ]:
                        self.process_updated_actor(update, frame)
                    del frame.game.disconnected_car_component_updates[
                        updated_actor.actor_id
                    ]
                return None
            self.add_updated_actor_to_disconnected_car_component_updates(
                updated_actor, frame
            )
            return None

        match updated_actor.category:
            case "game_start_event":
                if frame.resync:
                    return None
                if not updated_actor.attribute or "Int" not in updated_actor.attribute:
                    raise ValueError(f"Game active state not valid {updated_actor.raw}")
                name = frame.game.names[updated_actor.attribute["Int"]]
                if name == "Active":
                    frame.game.activate_game()
                    frame.calculate_match_time()
                elif name == "PostGoalScored":
                    frame.game.deactivate_game()
                    frame.calculate_match_time()
                return None
            case "car_component":
                car_component = frame.game.car_components.get(updated_actor.actor_id)
                if car_component:
                    match updated_actor.secondary_category:
                        case "component_usage_in_air":
                            return None
                        case "double_jump":
                            attribute = cast(
                                Location_Attribute, updated_actor.attribute
                            )
                            if self.include_boost_management:
                                field_label = (
                                    car_component.car.player.name + "_double_jump"
                                )
                                frame.processed_fields[field_label + "_x"] = attribute[
                                    "Location"
                                ]["x"]
                                frame.processed_fields[field_label + "_y"] = attribute[
                                    "Location"
                                ]["y"]
                                frame.processed_fields[field_label + "_z"] = attribute[
                                    "Location"
                                ]["z"]
                            return None
                        case "active":
                            boost_car_component = Boost_Car_Component(car_component)
                            car_component.car.boost.update_activity(updated_actor)
                            if self.include_boost_management:
                                frame.processed_fields[
                                    car_component.car.player.name + "_boost_active"
                                ] = boost_car_component.active
                            return None
                        case "dodge":
                            attribute = cast(
                                Location_Attribute, updated_actor.attribute
                            )
                            if self.include_boost_management:
                                field_label = car_component.car.player.name + "_dodge"
                                frame.processed_fields[field_label + "_x"] = attribute[
                                    "Location"
                                ]["x"]
                                frame.processed_fields[field_label + "_y"] = attribute[
                                    "Location"
                                ]["y"]
                                frame.processed_fields[field_label + "_z"] = attribute[
                                    "Location"
                                ]["z"]
                            return None
                        case "rep_boost":
                            if not car_component.car.boost:
                                raise ValueError(
                                    f"Car component car does not have boost to update {updated_actor.raw}"
                                )
                            car_component.car.boost.update_amount(updated_actor)
                            if self.include_boost_management:
                                frame.processed_fields[
                                    car_component.car.player.name + "_boost_amount"
                                ] = Boost_Car_Component(
                                    Car_Component(updated_actor, car_component.car)
                                ).amount
                            return None
                        case _:
                            pass

                car = frame.game.cars.get(updated_actor.actor_id)
                if car:
                    match updated_actor.secondary_category:
                        case "steer":
                            return None
                        case "throttle":
                            return None
                        case "handbrake":
                            return None
                        case "handbrake_active":
                            attribute = cast(Boolean_Attribute, updated_actor.attribute)
                            if not car.handbrake:
                                car.handbrake = Car_Component(updated_actor, car)
                            car.handbrake.update_activity(updated_actor)
                            if self.include_boost_management:
                                frame.processed_fields[
                                    car.player.name + "_drift_active"
                                ] = (1 if attribute["Boolean"] else 0)
                            return None
                        case "driving":
                            return None
                        case _:
                            pass
                if (
                    not car_component
                    and not car
                    and frame.resync
                    and frame.game.disconnected_car_components.get(
                        updated_actor.actor_id
                    )
                ):
                    return None

                if (
                    not car_component
                    and not car
                    and frame.game.boost_pads.get(updated_actor.actor_id)
                ):
                    # I have no idea why this happens
                    return None

                if not car_component and not car:
                    self.add_updated_actor_to_disconnected_car_component_updates(
                        updated_actor, frame
                    )
                    return None

                raise ValueError(
                    f"Unknown car component not updated {updated_actor.raw}"
                )
            case "camera_settings":
                player = frame.game.camera_settings[updated_actor.actor_id].player
                player.update_camera_settings(updated_actor)
                return None
            # case "team_ball_hit":
            #     team = updated_actor.attribute["Team"]
            #     self.ball.update_team_hit(team)
            #     if self.include_possession or self.include_pressure:
            #         frame.processed_fields[team + "_ball_hit"] = 1
            #     return None
            case _:
                pass

        match updated_actor.secondary_category:
            case "team":
                if not updated_actor.active_actor_id:
                    raise ValueError(
                        f"Team assignment must come with active actor {updated_actor.raw}"
                    )
                player = frame.game.players[updated_actor.actor_id]
                team = frame.game.teams[updated_actor.active_actor_id]
                field_label = f"{player.name}_team"
                frame.processed_fields[field_label] = (
                    player.team or team.secondary_category
                )
            case _:
                pass

        if not updated_actor.attribute:
            return None
        attribute_key = [*updated_actor.attribute.keys()][0]

        match attribute_key:
            case "Boolean":
                attribute = cast(Boolean_Attribute, updated_actor.attribute)
                car_component = frame.game.car_components.get(updated_actor.actor_id)
                if car_component and car_component.car:
                    if (
                        self.include_boost_management
                        and updated_actor.secondary_category == "rep_boost"
                    ):
                        field_label = f"{car_component.car.player.name}_{updated_actor.secondary_category}_active"
                        frame.processed_fields[field_label] = 1 if attribute else 0
                    car_component.update_activity(updated_actor)
                    return None
                camera_settings = frame.game.camera_settings.get(updated_actor.actor_id)
                if camera_settings:
                    if self.include_simple_vision:
                        field_label = f"{camera_settings.player.name}_{updated_actor.secondary_category}_active"
                        frame.processed_fields[field_label] = 1 if attribute else 0
                    camera_settings.update_settings(updated_actor)
                    return None

                ignore_objects = {
                    "ProjectX.GRI_X:bGameStarted",
                    "Engine.Actor:bBlockActors",
                    "Engine.Actor:bCollideActors",
                    "TAGame.PRI_TA:bIsDistracted",
                    "TAGame.PRI_TA:bReady",
                    "Engine.Actor:bHidden",
                }
                if (
                    updated_actor.object in ignore_objects
                    or "GameEvent" in updated_actor.object
                ):
                    return None
                raise ValueError(f"Actor activity not updated {updated_actor.raw}")
            case "RigidBody":
                if not updated_actor.positioning:
                    raise ValueError(
                        f"Actor contains RigidBody but no positioning {updated_actor.raw}"
                    )
                ball = (
                    frame.game.ball if frame.game.ball.is_self(updated_actor) else None
                )
                if ball:
                    if self.include_ball_positioning:
                        field_label = "ball_positioning"
                        frame.processed_fields[field_label + "_sleeping"] = (
                            updated_actor.positioning.sleeping
                        )
                        frame.processed_fields[field_label + "_x"] = (
                            updated_actor.positioning.location.x
                        )
                        frame.processed_fields[field_label + "_y"] = (
                            updated_actor.positioning.location.y
                        )
                        frame.processed_fields[field_label + "_z"] = (
                            updated_actor.positioning.location.z
                        )
                        frame.processed_fields[field_label + "_rotation_x"] = (
                            updated_actor.positioning.rotation.x
                        )
                        frame.processed_fields[field_label + "_rotation_y"] = (
                            updated_actor.positioning.rotation.y
                        )
                        frame.processed_fields[field_label + "_rotation_z"] = (
                            updated_actor.positioning.rotation.z
                        )
                        frame.processed_fields[field_label + "_rotation_w"] = (
                            updated_actor.positioning.rotation.w
                        )
                        frame.processed_fields[field_label + "_linear_velocity_x"] = (
                            updated_actor.positioning.linear_velocity.x
                        )
                        frame.processed_fields[field_label + "_linear_velocity_y"] = (
                            updated_actor.positioning.linear_velocity.y
                        )
                        frame.processed_fields[field_label + "_linear_velocity_z"] = (
                            updated_actor.positioning.linear_velocity.z
                        )
                        frame.processed_fields[field_label + "_angular_velocity_x"] = (
                            updated_actor.positioning.angular_velocity.x
                        )
                        frame.processed_fields[field_label + "_angular_velocity_y"] = (
                            updated_actor.positioning.angular_velocity.y
                        )
                        frame.processed_fields[field_label + "_angular_velocity_z"] = (
                            updated_actor.positioning.angular_velocity.z
                        )
                    ball.update_position(updated_actor)
                    return None

                car = frame.game.cars.get(updated_actor.actor_id)
                if car:
                    if self.include_simple_positioning:
                        field_label = f"{car.player.name}_positioning"
                        frame.processed_fields[field_label + "_sleeping"] = (
                            updated_actor.positioning.sleeping
                        )
                        frame.processed_fields[field_label + "_x"] = (
                            updated_actor.positioning.location.x
                        )
                        frame.processed_fields[field_label + "_y"] = (
                            updated_actor.positioning.location.y
                        )
                        frame.processed_fields[field_label + "_z"] = (
                            updated_actor.positioning.location.z
                        )
                        frame.processed_fields[field_label + "_rotation_x"] = (
                            updated_actor.positioning.rotation.x
                        )
                        frame.processed_fields[field_label + "_rotation_y"] = (
                            updated_actor.positioning.rotation.y
                        )
                        frame.processed_fields[field_label + "_rotation_z"] = (
                            updated_actor.positioning.rotation.z
                        )
                        frame.processed_fields[field_label + "_rotation_w"] = (
                            updated_actor.positioning.rotation.w
                        )
                        frame.processed_fields[field_label + "_linear_velocity_x"] = (
                            updated_actor.positioning.linear_velocity.x
                        )
                        frame.processed_fields[field_label + "_linear_velocity_y"] = (
                            updated_actor.positioning.linear_velocity.y
                        )
                        frame.processed_fields[field_label + "_linear_velocity_z"] = (
                            updated_actor.positioning.linear_velocity.z
                        )
                        frame.processed_fields[field_label + "_angular_velocity_x"] = (
                            updated_actor.positioning.angular_velocity.x
                        )
                        frame.processed_fields[field_label + "_angular_velocity_y"] = (
                            updated_actor.positioning.angular_velocity.y
                        )
                        frame.processed_fields[field_label + "_angular_velocity_z"] = (
                            updated_actor.positioning.angular_velocity.z
                        )
                    car.update_position(updated_actor)
                    return None

                car_component = frame.game.car_components.get(updated_actor.actor_id)
                if car_component:
                    if self.include_advanced_positioning:
                        field_label = f"{car_component.car.player.name}_{updated_actor.secondary_category}_positioning"
                        frame.processed_fields[field_label + "_sleeping"] = (
                            updated_actor.positioning.sleeping
                        )
                        frame.processed_fields[field_label + "_x"] = (
                            updated_actor.positioning.location.x
                        )
                        frame.processed_fields[field_label + "_y"] = (
                            updated_actor.positioning.location.y
                        )
                        frame.processed_fields[field_label + "_z"] = (
                            updated_actor.positioning.location.z
                        )
                        frame.processed_fields[field_label + "_rotation_x"] = (
                            updated_actor.positioning.rotation.x
                        )
                        frame.processed_fields[field_label + "_rotation_y"] = (
                            updated_actor.positioning.rotation.y
                        )
                        frame.processed_fields[field_label + "_rotation_z"] = (
                            updated_actor.positioning.rotation.z
                        )
                        frame.processed_fields[field_label + "_rotation_w"] = (
                            updated_actor.positioning.rotation.w
                        )
                        frame.processed_fields[field_label + "_linear_velocity_x"] = (
                            updated_actor.positioning.linear_velocity.x
                        )
                        frame.processed_fields[field_label + "_linear_velocity_y"] = (
                            updated_actor.positioning.linear_velocity.y
                        )
                        frame.processed_fields[field_label + "_linear_velocity_z"] = (
                            updated_actor.positioning.linear_velocity.z
                        )
                        frame.processed_fields[field_label + "_angular_velocity_x"] = (
                            updated_actor.positioning.angular_velocity.x
                        )
                        frame.processed_fields[field_label + "_angular_velocity_y"] = (
                            updated_actor.positioning.angular_velocity.y
                        )
                        frame.processed_fields[field_label + "_angular_velocity_z"] = (
                            updated_actor.positioning.angular_velocity.z
                        )
                    car_component.update_position(updated_actor)
                    return None

                boost_actor = frame.game.boost_pads.get(updated_actor.actor_id)
                if boost_actor:
                    boost_actor.update_position(updated_actor)
                    return None

                self.add_updated_actor_to_disconnected_car_component_updates(
                    updated_actor, frame
                )
            case "DemolishExtended":
                attribute = cast(Demolish_Extended_Attribute, updated_actor.attribute)
                demolition = attribute["DemolishExtended"]
                attacker_car_actor_id = demolition["attacker"]["actor"]
                victim_car_actor_id = demolition["victim"]["actor"]
                if (
                    attacker_car_actor_id not in frame.game.cars
                    or victim_car_actor_id not in frame.game.cars
                ):
                    self.add_updated_actor_to_disconnected_car_component_updates(
                        updated_actor, frame
                    )
                    return None
                attacker_car = frame.game.cars[attacker_car_actor_id]
                victim_car = frame.game.cars[victim_car_actor_id]
                if not attacker_car:
                    raise ValueError(
                        f"No matching attacker car in demo actor {updated_actor.raw}"
                    )
                if not victim_car:
                    raise ValueError(
                        f"No matching victim car in demo actor {updated_actor.raw}"
                    )
                if self.include_player_demos:
                    field_label = f"{attacker_car.player.name}_demo"
                    frame.processed_fields[field_label + "_x"] = demolition[
                        "attacker_velocity"
                    ]["x"]
                    frame.processed_fields[field_label + "_y"] = demolition[
                        "attacker_velocity"
                    ]["y"]
                    frame.processed_fields[field_label + "_z"] = demolition[
                        "attacker_velocity"
                    ]["z"]
                    field_label = f"{victim_car.player.name}_demod"
                    frame.processed_fields[field_label + "_x"] = demolition[
                        "victim_velocity"
                    ]["x"]
                    frame.processed_fields[field_label + "_y"] = demolition[
                        "victim_velocity"
                    ]["y"]
                    frame.processed_fields[field_label + "_z"] = demolition[
                        "victim_velocity"
                    ]["z"]
                return None
            case "PickupNew":
                boost_grab = cast(Pickup_New_Attribute, updated_actor.attribute)
                boost_actor = frame.game.boost_pads[updated_actor.actor_id]
                if not boost_actor:
                    raise ValueError(
                        f"Cannot find boost actor {updated_actor.actor_id}"
                    )
                boost_data = parse_boost_actor_name(boost_actor.object)
                if not boost_data:
                    # print(
                    #     f"Cannot find boost {updated_actor.actor_id}, {boost_actor.object}"
                    # )
                    # raise ValueError(
                    #     f"Boost data not found for boost actor {boost_actor.object}"
                    # )
                    return None
                if (
                    not boost_grab["PickupNew"]["instigator"]
                    or boost_grab["PickupNew"]["instigator"] == -1
                ):
                    # print(f"Null boost pickup {updated_actor.raw}")
                    return None
                car = frame.game.cars[boost_grab["PickupNew"]["instigator"]]
                if self.include_boost_management:
                    field_label = f"{car.player.name}_boost_pickup"
                    frame.processed_fields[field_label + "_x"] = boost_data[0]
                    frame.processed_fields[field_label + "_y"] = boost_data[1]
                    frame.processed_fields[field_label + "_z"] = boost_data[2]
                    frame.processed_fields[field_label + "_amount"] = boost_data[3]
                return None
            case _:
                pass

        return None

    def process_frame(self, raw_frame: Raw_Frame, game: Game, f: int = 0):
        frame = Frame(raw_frame, game)
        frame.calculate_match_time()
        frame.set_values_from_previous()

        new_actors: Dict[int, Actor] = {}

        labeled_new_actors = [  # type: ignore
            Actor.label(a, frame.game.objects) for a in frame.new_actors
        ]
        labeled_updated_actors = [  # type: ignore
            Actor.label(a, game.objects) for a in frame.updated_actors
        ]

        if frame.resync:
            frame.game.set_actors(frame)

        for na in frame.new_actors:
            new_actor = Actor(na, frame.game.objects)
            if new_actor.secondary_category == "team":
                pass

            if ".TheWorld:PersistentLevel.VehiclePickup_Boost_TA_" in new_actor.object:
                game.boost_pads[new_actor.actor_id] = new_actor
            elif new_actor.category == "car" and not frame.resync:
                frame.game.disconnected_cars[new_actor.actor_id] = new_actor
            elif new_actor.category == "car_component":
                frame.game.disconnected_car_components[new_actor.actor_id] = new_actor
            else:
                new_actors[new_actor.actor_id] = new_actor

        frame.processed_new_actors = new_actors

        delayed_updated_actors: List[Actor] = []
        for ua in frame.updated_actors:
            updated_actor = Actor(ua, frame.game.objects)
            if updated_actor.secondary_category == "team":
                pass
            self.process_updated_actor(updated_actor, frame)

        for updated_actor in delayed_updated_actors:
            self.process_updated_actor(updated_actor, frame)

        frame.game.previous_frame = frame
        return frame

    def process_game(self, game_data: Raw_Game_Data):
        game = Game(game_data)

        initial_frame = Frame(game.initial_frame, game)
        game.set_actors(initial_frame)
        game.previous_frame = initial_frame

        frames: List[Dict[str, Any]] = []
        for f, raw_frame in enumerate(game.frames):
            processed_frame = self.process_frame(raw_frame, game, f)
            frames.append(processed_frame.to_dict())

        return frames

    def load_processed_file(self, file_name: str):
        """
        Loads a processed CSV file from config.PROCESSED

        Parameters
        ----------
        file_name : str
            Name of the processed file.

        Returns
        -------
        pandas.DataFrame
            DataFrame constructed from processed file
        """
        base_file_name = Path(file_name).stem
        processed_file_path = os.path.join(PROCESSED, base_file_name + ".csv")
        if os.path.exists(processed_file_path):
            game = pd.read_csv(processed_file_path, low_memory=False)  # type: ignore
            return game
        raise FileExistsError(f"Processed file does not exist at {processed_file_path}")

    def try_load_processed_file(self, file_name: str):
        try:
            return self.load_processed_file(file_name)
        except Exception:
            return None

    def save_processed_file(self, file_name: str, game_frames: List[Dict[str, Any]]):
        """
        Saves a processed CSV file to config.PROCESSED

        Parameters
        ----------
        file_name : str
            Name of the processed file.
        """
        base_file_name = Path(file_name).stem
        processed_csv_file_path = os.path.join(PROCESSED, base_file_name + ".csv")
        pd.DataFrame(game_frames).to_csv(processed_csv_file_path, index=False)
        return None

    def process_games(
        self,
        save_output: bool = True,
        overwrite: bool = False,
        game_datas: List[Raw_Game_Data] | None = None,
    ):
        self.logger.print(f"Processing game data files...")
        self.succeeded = 0
        self.skipped = 0
        self.failed = 0
        all_game_frames: List[List[Dict[str, Any]] | pd.DataFrame] = []

        if game_datas:
            for game_data in game_datas:
                try:
                    game_frames = self.process_game(game_data)
                    all_game_frames.append(game_frames)
                    if save_output:
                        self.save_processed_file("custom", game_frames)

                    self.logger.print("OK.")
                    self.succeeded += 1

                except Exception as e:
                    self.logger.print("FAILED.")
                    self.logger.print(f"  {type(e).__name__}: {e}")
                    self.failed += 1

            if save_output:
                return None
            return all_game_frames

        Path(PROCESSED).mkdir(parents=True, exist_ok=True)

        game_files = sorted([f for f in os.listdir(PREPROCESSED)])
        if not game_files:
            self.logger.print(f"No files found in {PREPROCESSED}.")
            return None

        for file_name in game_files:
            game = self.try_load_processed_file(file_name)
            if game is not None and not game.empty and not overwrite:
                self.logger.print(f"Using existing processed data for {file_name}.")
                self.skipped += 1
                if not save_output:
                    all_game_frames.append(game)
            else:
                self.logger.print(
                    f"Parsing {file_name} -> {PROCESSED} ...", end=" ", flush=True
                )
                try:
                    game_data = self.preprocessor.load_preprocessed_file(file_name)
                    game_frames = self.process_game(game_data)
                    if not save_output:
                        all_game_frames.append(game_frames)
                    else:
                        self.save_processed_file(file_name, game_frames)
                    self.logger.print("OK.")
                    self.succeeded += 1
                except Exception as e:
                    self.logger.print("FAILED.")
                    self.logger.print(f"  {type(e).__name__}: {e}")
                    self.failed += 1

        self.logger.print()
        self.logger.print(
            f"Done. succeeded={self.succeeded}, skipped={self.skipped}, failed={self.failed}."
        )
        self.logger.print(f"Saved processed games to: {PROCESSED}")
        self.logger.print()
        return None
