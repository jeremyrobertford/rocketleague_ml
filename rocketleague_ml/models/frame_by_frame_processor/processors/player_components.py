from __future__ import annotations
from typing import TYPE_CHECKING
from rocketleague_ml.core.actor import Actor
from rocketleague_ml.core.frame import Frame
from rocketleague_ml.core.player_component import Player_Component

if TYPE_CHECKING:
    from rocketleague_ml.models.frame_by_frame_processor import Frame_By_Frame_Processor


def process_player_team(
    processor: Frame_By_Frame_Processor, updated_actor: Actor, frame: Frame
):
    if updated_actor.active_actor_id is None:
        raise ValueError(
            f"Team assignment must come with active actor {updated_actor.raw}"
        )
    player = frame.game.players[updated_actor.actor_id]
    team = frame.game.teams[updated_actor.active_actor_id]
    field_label = f"{player.name}_team"
    frame.processed_fields[field_label] = player.team or team.secondary_category
    return None


def process_player_gained_points(
    processor: Frame_By_Frame_Processor, updated_actor: Actor, frame: Frame
):
    updated_actor = Player_Component(updated_actor)
    player = frame.game.players[updated_actor.actor_id]
    if processor.include_scoreboard_metrics:
        field_label = f"{player.name}_score"
        frame.processed_fields[field_label] = updated_actor.amount
    return None


def process_player_scored(
    processor: Frame_By_Frame_Processor, updated_actor: Actor, frame: Frame
):
    updated_actor = Player_Component(updated_actor)
    player = frame.game.players[updated_actor.actor_id]
    if processor.include_scoreboard_metrics:
        field_label = f"{player.name}_goal"
        frame.processed_fields[field_label] = 1
    return None


def process_player_shot(
    processor: Frame_By_Frame_Processor, updated_actor: Actor, frame: Frame
):
    updated_actor = Player_Component(updated_actor)
    player = frame.game.players[updated_actor.actor_id]
    if processor.include_scoreboard_metrics:
        field_label = f"{player.name}_shot"
        frame.processed_fields[field_label] = 1
    return None


def process_player_assisted(
    processor: Frame_By_Frame_Processor, updated_actor: Actor, frame: Frame
):
    updated_actor = Player_Component(updated_actor)
    player = frame.game.players[updated_actor.actor_id]
    if processor.include_scoreboard_metrics:
        field_label = f"{player.name}_assist"
        frame.processed_fields[field_label] = 1
    return None


def process_player_saved(
    processor: Frame_By_Frame_Processor, updated_actor: Actor, frame: Frame
):
    updated_actor = Player_Component(updated_actor)
    player = frame.game.players[updated_actor.actor_id]
    if processor.include_scoreboard_metrics:
        field_label = f"{player.name}_save"
        frame.processed_fields[field_label] = 1
    return None
