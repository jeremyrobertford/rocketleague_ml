from __future__ import annotations
from typing import TYPE_CHECKING
from rocketleague_ml.core.actor import Actor
from rocketleague_ml.core.frame import Frame

if TYPE_CHECKING:
    from rocketleague_ml.models.frame_by_frame_processor import Frame_By_Frame_Processor


def process_game_start(
    processor: Frame_By_Frame_Processor, updated_actor: Actor, frame: Frame
):
    if frame.resync:
        return None
    if not updated_actor.attribute or "Int" not in updated_actor.attribute:
        raise ValueError(f"Game active state not valid {updated_actor.raw}")
    name = frame.game.names[updated_actor.attribute["Int"]]
    if name == "Active":
        frame.game.activate_game(frame)
        frame.calculate_match_time()
        for key in frame.processed_fields:
            if "_kickoff" not in key:
                continue
            frame.processed_fields[key] = 0
    elif name == "PostGoalScored":
        frame.game.deactivate_game()
        frame.calculate_match_time()
    return None
