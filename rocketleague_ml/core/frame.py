from __future__ import annotations
from math import ceil
from typing import TYPE_CHECKING, cast, Any, Dict, List
from rocketleague_ml.core.actor import Actor
from rocketleague_ml.types.attributes import Raw_Frame

if TYPE_CHECKING:
    from rocketleague_ml.core.game import Game


class Frame:
    def __init__(self, raw_frame: Raw_Frame, game: Game):
        self.raw = raw_frame
        self.time = raw_frame["time"]
        self.delta = raw_frame["delta"]
        self.new_actors = raw_frame["new_actors"]
        self.updated_actors = raw_frame["updated_actors"]
        self.deleted_actors = raw_frame["deleted_actors"]
        self.delayed_updated_actors: List[Actor] = []

        self.processed_fields: Dict[str, Any] = {}
        self.processed_new_actors: Dict[int, Actor] = {}
        self.processed_updated_actors: Dict[int, Actor] = {}

        self.resync = len(self.new_actors) + len(self.updated_actors) > 80
        self.game = game

        self.calculate_match_time()
        return None

    def __repr__(self):
        return f"Frame(time={self.match_time_label}, resync={self.resync}, new={len(self.new_actors)}, updated={len(self.updated_actors)})"

    def add_updated_actor_to_disconnected_car_component_updates(
        self, updated_actor: Actor
    ):
        if updated_actor.actor_id not in self.game.disconnected_car_component_updates:
            self.game.disconnected_car_component_updates[updated_actor.actor_id] = []
        self.game.disconnected_car_component_updates[updated_actor.actor_id].append(
            updated_actor
        )
        return None

    def set_values_from_previous(self):
        one_time_event_keys = [
            "kickoff",
            "demo",
            "demoed",
            "boost_pickup",
            "score",
            "shot",
            "save",
            "assist",
            "goal",
            "team_ball_hit",
            "hit_ball",
            "_double_jump_x",
            "_double_jump_y",
            "_double_jump_z",
            "_dodge_x",
            "_dodge_y",
            "_dodge_z",
            "_activation",
            "_component_usage_in_air",
        ]
        clear_positions_for_players = [
            car.player.name for car in self.game.do_not_track.values()
        ]
        for key, value in self.game.previous_frame.processed_fields.items():
            ignore = False
            for one_time_event_key in one_time_event_keys:
                if one_time_event_key in key:
                    ignore = True
                    break
            for player in clear_positions_for_players:
                if key.startswith(f"{player}_positioning"):
                    ignore = True
                    break
            if not ignore:
                self.processed_fields[key] = value

    def calculate_match_time(self):
        self.round = self.game.round
        self.active = self.game.active
        self.match_time_remaining = self.game.match_time_remaining
        self.in_overtime = self.game.in_overtime
        self.overtime_elapsed = self.game.overtime_elapsed

        if self.active and not self.in_overtime:
            new_match_time_remaining = max(0.0, self.match_time_remaining - self.delta)
            self.match_time_remaining = new_match_time_remaining
            self.game.match_time_remaining = new_match_time_remaining

        elif self.active and self.in_overtime:
            new_overtime_elapsed = self.overtime_elapsed + self.delta
            self.overtime_elapsed = new_overtime_elapsed
            self.game.overtime_elapsed = new_overtime_elapsed

        if not self.in_overtime:
            display_time = ceil(self.match_time_remaining)
            mins, secs = divmod(display_time, 60)
            self.match_time_label = f"{mins}:{secs:02d}"
        else:
            mins, secs = divmod(int(self.overtime_elapsed), 60)
            self.match_time_label = f"+{mins}:{secs:02d}"

        return None

    def to_dict(self) -> Dict[str, Any]:
        return cast(
            Dict[str, Any],
            {
                "id": self.game.id,
                "time": self.time,
                "delta": self.delta,
                "round": self.round,
                "active": self.active,
                "match_time_label": "5:00",
            }
            | self.processed_fields,
        )
