import os
import math
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, TypedDict
from rocketleague_ml.config import PREPROCESSED, PROCESSED, WRANGLED, ROUND_LENGTH
from rocketleague_ml.core.actor import Actor
from rocketleague_ml.core.rigid_body import Rigid_Body
from rocketleague_ml.core.game import Game
from rocketleague_ml.core.frame import Frame
from rocketleague_ml.models.rrrocket_json_preprocessor import RRRocket_JSON_Preprocessor
from rocketleague_ml.models.cd1 import RocketLeagueCollisionDetector
from rocketleague_ml.utils.logging import Logger
from rocketleague_ml.types.attributes import (
    Raw_Game_Data,
    Raw_Frame,
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
        include_ball_collisions: bool = True,
        # requires rl_utilities
        include_player_collisions: bool = False,
        include_custom_scoreboard_metrics: bool = False,
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

        include_custom_scoreboard_metrics = (
            include_custom_scoreboard_metrics
            if include_custom_scoreboard_metrics is not None
            else self.include_custom_scoreboard_metrics
        )
        include_player_collisions = (
            include_player_collisions
            if include_player_collisions is not None
            else self.include_player_collisions
        )
        include_ball_collisions = (
            include_ball_collisions
            if include_ball_collisions is not None
            else self.include_ball_collisions
        )

        if not self.include_ball_positioning and include_ball_collisions:
            raise ValueError(
                "include_ball_collisions requires include_ball_positioning."
            )
        if not self.include_ball_positioning and include_custom_scoreboard_metrics:
            raise ValueError(
                "include_custom_scoreboard_metrics requires include_ball_positioning."
            )

        if not self.include_car_positioning and include_ball_collisions:
            raise ValueError(
                "include_ball_collisions requires include_car_positioning."
            )
        if not self.include_car_positioning and include_custom_scoreboard_metrics:
            raise ValueError(
                "include_custom_scoreboard_metrics requires include_car_positioning."
            )

        self.include_ball_collisions = include_ball_collisions
        self.include_player_collisions = include_player_collisions
        self.include_custom_scoreboard_metrics = include_custom_scoreboard_metrics

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
            case _:
                pass

        if not updated_actor.secondary_category:
            pass

        return None

    def determine_kickoff_players(self, frame: Frame, eps: float = 1e-6):
        """
        Return kickoff taker for each team as {'blue': (name, distance), 'orange': (name, distance)}
        or None if no player found for a team.
        """
        # extract ball position
        bx = frame.processed_fields.get("ball_positioning_x")
        by = frame.processed_fields.get("ball_positioning_y")
        if bx is None or by is None:
            raise ValueError("Frame missing ball_positioning_x / y")

        # collect players: find all prefixes that have *_positioning_x keys except 'ball'
        class Player_Pos(TypedDict):
            x: float
            y: float
            team: str
            name: str
            dist: float

        players: Dict[str, Player_Pos] = {}
        for player in frame.game.players.values():
            # read x,y (and optionally z)
            x = frame.processed_fields.get(f"{player.name}_positioning_x")
            y = frame.processed_fields.get(f"{player.name}_positioning_y")
            if x is None or y is None:
                continue
            # assign team by x sign (x < 0 => blue, x > 0 => orange)
            dist = math.hypot(x - bx, y - by)
            players[player.name] = {
                "x": x,
                "y": y,
                "team": player.team,
                "dist": dist,
                "name": player.name,
            }

        # find min per team with tie-breaker
        result: Dict[str, Player_Pos | None] = {"Blue": None, "Orange": None}

        def better(a: Player_Pos | None, b: Player_Pos | None):
            """Return True if player a is better (should win) than b given tie rules."""
            if a is None:
                return False
            if b is None:
                return True
            da = a["dist"]
            db = b["dist"]
            if abs(da - db) > eps:
                return da < db
            # distances equal -> tie-break by smaller x (left)
            if a["y"] != b["y"]:
                return a["y"] < b["y"]
            return a["x"] > b["x"] if a["team"] == "Blue" else a["x"] < b["x"]

        for info in players.values():
            team = info["team"]
            current = result[team]
            is_better = better(info, current)
            if is_better:
                result[team] = info.copy()

        # convert to (name, distance) or None
        kickoff_players: List[str] = []
        for team in ("Blue", "Orange"):
            team_result = result[team]
            if team_result is not None:
                kickoff_players.append(team_result["name"])

        return kickoff_players

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

        if f >= 10327:
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

        if not frame.game.previous_frame.active and frame.active:
            kickoff_players = self.determine_kickoff_players(game.previous_frame)
            for player_name in kickoff_players:
                frame.processed_fields[f"{player_name}_kickoff"] = 1

        return frame

    def process_game(self, game_data: Raw_Game_Data) -> List[Dict[str, Any]]:
        game = Game(game_data)
        collision_detector = RocketLeagueCollisionDetector()

        initial_frame = Frame(game.initial_frame, game)
        game.set_actors(initial_frame)
        game.previous_frame = initial_frame

        frames: List[Dict[str, Any]] = []
        for f, raw_frame in enumerate(game.frames):
            try:
                processed_frame = self.process_frame(raw_frame, game, f)
            except Exception:
                if len(frames) == 0:
                    return []
                previous_frame = frames[-1]
                previous_round = previous_frame["round"] - 1
                if previous_round <= 0:
                    return []
                frames = [f for f in frames if f["round"] <= previous_round]
                return frames
            if f == 0:
                processors["rigid_body"](self, game.ball, processed_frame)
                for car in game.cars.values():
                    processors["rigid_body"](self, car, processed_frame)
            include_collision = (
                self.include_ball_collisions
                or self.include_custom_scoreboard_metrics
                or self.include_player_collisions
            )
            if processed_frame.active and include_collision:
                collisions = collision_detector.detect_all_collisions(
                    prev_frame=processed_frame.game.previous_frame,
                    curr_frame=processed_frame,
                    frame_number=f,
                )
                if self.include_ball_collisions:
                    field_label = "ball_hit"
                    for collision in collisions.ball_environment_collisions:
                        processed_frame.processed_fields[
                            field_label + "_ball_impact_x"
                        ] = collision.ball_impact_point.x
                        processed_frame.processed_fields[
                            field_label + "_ball_impact_y"
                        ] = collision.ball_impact_point.y
                        processed_frame.processed_fields[
                            field_label + "_ball_impact_z"
                        ] = collision.ball_impact_point.z
                        processed_frame.processed_fields[
                            field_label + "_ball_collision_confidence"
                        ] = collision.confidence
                        processed_frame.processed_fields[
                            field_label + "_ball_impulse_x"
                        ] = collision.impulse_vector.x
                        processed_frame.processed_fields[
                            field_label + "_ball_impulse_y"
                        ] = collision.impulse_vector.y
                        processed_frame.processed_fields[
                            field_label + "_ball_impulse_z"
                        ] = collision.impulse_vector.z
                if self.include_player_collisions:
                    for collision in collisions.player_environment_collisions:
                        pass
                if self.include_ball_collisions and self.include_car_positioning:
                    for collision in collisions.ball_player_collisions:
                        field_label = f"{collision.player_name}_hit_ball"
                        processed_frame.processed_fields[
                            field_label + "_ball_impact_x"
                        ] = collision.ball_impact_point.x
                        processed_frame.processed_fields[
                            field_label + "_ball_impact_y"
                        ] = collision.ball_impact_point.y
                        processed_frame.processed_fields[
                            field_label + "_ball_impact_z"
                        ] = collision.ball_impact_point.z
                        processed_frame.processed_fields[
                            field_label + "_car_impact_x"
                        ] = collision.car_impact_point.x
                        processed_frame.processed_fields[
                            field_label + "_car_impact_y"
                        ] = collision.car_impact_point.y
                        processed_frame.processed_fields[
                            field_label + "_car_impact_z"
                        ] = collision.car_impact_point.z
                        processed_frame.processed_fields[
                            field_label + "_collision_normal_x"
                        ] = collision.collision_normal.x
                        processed_frame.processed_fields[
                            field_label + "_collision_normal_y"
                        ] = collision.collision_normal.y
                        processed_frame.processed_fields[
                            field_label + "_collision_normal_z"
                        ] = collision.collision_normal.z
                        processed_frame.processed_fields[
                            field_label + "_collision_confidence"
                        ] = collision.confidence
                        processed_frame.processed_fields[field_label + "_impulse_x"] = (
                            collision.impulse_vector.x
                        )
                        processed_frame.processed_fields[field_label + "_impulse_y"] = (
                            collision.impulse_vector.y
                        )
                        processed_frame.processed_fields[field_label + "_impulse_z"] = (
                            collision.impulse_vector.z
                        )
                    pass
                if self.include_player_collisions:
                    for collision in collisions.player_player_collisions:
                        pass
            frames.append(processed_frame.to_dict())
            processed_frame.game.previous_frame = processed_frame

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
                game_id = game_data["properties"]["Id"]
                self.logger.print(
                    f"Parsing {game_id} -> {PROCESSED} ...",
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
                    game_total_time_played = game_data["properties"][
                        "TotalSecondsPlayed"
                    ]
                    if game_total_time_played > ROUND_LENGTH * 1.75:
                        self.logger.print("SKIPPED.")
                        self.skipped += 1
                    else:
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
