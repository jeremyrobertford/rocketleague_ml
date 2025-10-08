import os
from pathlib import Path
from typing import cast, Any, Dict, List
import pandas as pd
from rocketleague_ml.config import PREPROCESSED, PROCESSED, WRANGLED
from rocketleague_ml.core.actor import Actor
from rocketleague_ml.core.rigid_body import Rigid_Body
from rocketleague_ml.core.game import Game
from rocketleague_ml.core.frame import Frame
from rocketleague_ml.models.rrrocket_json_preprocessor import RRRocket_JSON_Preprocessor
from rocketleague_ml.utils.helpers import parse_boost_actor_name
from rocketleague_ml.utils.logging import Logger
from rocketleague_ml.types.attributes import (
    Raw_Game_Data,
    Raw_Frame,
    Boolean_Attribute,
    Demolish_Extended_Attribute,
    Pickup_New_Attribute,
)
from rocketleague_ml.models.frame_by_frame_processor.processors import processors
from rocketleague_ml.models.game_data_wrangler import Game_Data_Wrangler


class Frame_By_Frame_Processor:
    def __init__(
        self,
        preprocessor: RRRocket_JSON_Preprocessor,
        wrangler: Game_Data_Wrangler | None = None,
        logger: Logger = Logger(),
        include_ball_positioning: bool = True,
        include_car_positioning: bool = True,
        include_simple_vision: bool = True,
        include_advanced_vision: bool = True,
        include_movement: bool = True,
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
        self.include_advanced_vision: bool = include_advanced_vision
        self.include_movement: bool = include_movement
        self.config(
            preprocessor=preprocessor,
            wrangler=wrangler,
            logger=logger,
            include_ball_positioning=include_ball_positioning,
            include_car_positioning=include_car_positioning,
            include_simple_vision=include_simple_vision,
            include_advanced_vision=include_advanced_vision,
            include_movement=include_movement,
            include_player_demos=include_player_demos,
            include_scoreboard_metrics=include_scoreboard_metrics,
            include_boost_management=include_boost_management,
            include_ball_collisions=include_ball_collisions,
            include_player_collisions=include_player_collisions,
            include_custom_scoreboard_metrics=include_custom_scoreboard_metrics,
            include_pressure=include_pressure,
            include_possession=include_possession,
            include_mechanics=include_mechanics,
        )
        return None

    def config(
        self,
        preprocessor: RRRocket_JSON_Preprocessor | None = None,
        wrangler: Game_Data_Wrangler | None = None,
        logger: Logger | None = None,
        include_ball_positioning: bool | None = None,
        include_car_positioning: bool | None = None,
        include_simple_vision: bool | None = None,
        include_advanced_vision: bool | None = None,
        include_movement: bool | None = None,
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
        include_simple_vision = (
            include_simple_vision
            if include_simple_vision is not None
            else self.include_simple_vision
        )
        include_advanced_vision = (
            include_advanced_vision
            if include_advanced_vision is not None
            else self.include_advanced_vision
        )

        if include_advanced_vision and include_simple_vision == False:
            raise ValueError("Cannot enable advanced vision without simple vision")

        self.include_simple_vision = include_simple_vision
        self.include_advanced_vision = include_advanced_vision

        self.preprocessor = preprocessor if preprocessor else self.preprocessor
        self.logger = logger if logger else self.logger
        self.wrangler = wrangler if wrangler else self.wrangler
        self.include_car_positioning = (
            include_car_positioning
            if include_car_positioning is not None
            else self.include_car_positioning
        )
        self.include_ball_positioning = (
            include_ball_positioning
            if include_ball_positioning is not None
            else self.include_ball_positioning
        )
        self.include_movement = (
            include_movement if include_movement is not None else self.include_movement
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

    def process_updated_actor(self, updated_actor: Actor, frame: Frame):
        if ".TheWorld:PersistentLevel.VehiclePickup_Boost_TA_" in updated_actor.object:
            frame.game.boost_pads[updated_actor.actor_id] = Rigid_Body(
                updated_actor, "boost_pad"
            )
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
            frame.add_updated_actor_to_disconnected_car_component_updates(
                updated_actor,
            )
            return None

        specific_processor = (
            processors.get(
                updated_actor.category + "." + updated_actor.secondary_category
            )
            if updated_actor.secondary_category
            else None
        )
        if specific_processor:
            specific_processor(self, updated_actor, frame)
            return None

        general_processor = processors.get(updated_actor.category)
        if general_processor:
            general_processor(self, updated_actor, frame)
            return None

        if not updated_actor.attribute:
            return None
        attribute_key = [*updated_actor.attribute.keys()][0]

        match attribute_key:
            case "Boolean":
                attribute = cast(Boolean_Attribute, updated_actor.attribute)
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
                updated_actor = Rigid_Body(updated_actor, "")
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
                    if self.include_car_positioning:
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

                boost_actor = frame.game.boost_pads.get(updated_actor.actor_id)
                if boost_actor:
                    boost_actor.update_position(updated_actor)
                    return None

                frame.add_updated_actor_to_disconnected_car_component_updates(
                    updated_actor
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
                    frame.add_updated_actor_to_disconnected_car_component_updates(
                        updated_actor
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
                boost_pickup = cast(Pickup_New_Attribute, updated_actor.attribute)
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
                    not boost_pickup["PickupNew"]["instigator"]
                    or boost_pickup["PickupNew"]["instigator"] == -1
                ):
                    # print(f"Null boost pickup {updated_actor.raw}")
                    return None
                car = frame.game.cars[boost_pickup["PickupNew"]["instigator"]]
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

    def process_frame(self, raw_frame: Raw_Frame, game: Game, f: int):
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

        if f > 515:
            pass

        if frame.resync:
            frame.game.set_actors(frame)

        for na in frame.new_actors:
            new_actor = Actor(na, frame.game.objects)
            if new_actor.secondary_category == "team":
                pass

            if ".TheWorld:PersistentLevel.VehiclePickup_Boost_TA_" in new_actor.object:
                game.boost_pads[new_actor.actor_id] = Rigid_Body(new_actor, "boost_pad")
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
        all_game_frames: List[List[Dict[str, Any]]] = []

        input_files = WRANGLED if self.wrangler else PREPROCESSED

        if game_datas:
            for game_data in game_datas:
                self.logger.print(
                    f"Parsing {game_data["properties"]["Id"]} -> {PROCESSED} ...",
                    end=" ",
                    flush=True,
                )
                try:
                    game_frames = self.process_game(game_data)
                    if save_output:
                        self.save_processed_file("custom", game_frames)
                    else:
                        all_game_frames.append(game_frames)

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

        game_files = sorted([f for f in os.listdir(input_files)])
        if not game_files:
            self.logger.print(f"No files found in {input_files}.")
            return None

        for file_name in game_files:
            game = self.try_load_processed_file(file_name)
            if game is not None and not game.empty and not overwrite:
                self.logger.print(f"Using existing processed data for {file_name}.")
                self.skipped += 1
                if not save_output:
                    all_game_frames.append(
                        game.to_dict(  # pyright: ignore[reportArgumentType, reportUnknownMemberType]
                            orient="records"
                        )
                    )
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

        self.logger.print(
            f"Done. succeeded={self.succeeded}, skipped={self.skipped}, failed={self.failed}."
        )
        if save_output and self.succeeded:
            self.logger.print(f"Saved processed games to: {PROCESSED}")
        self.logger.print()

        if save_output:
            return None
        return all_game_frames
