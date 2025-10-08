from __future__ import annotations
from typing import TYPE_CHECKING
from rocketleague_ml.core.actor import Actor
from rocketleague_ml.core.frame import Frame

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
