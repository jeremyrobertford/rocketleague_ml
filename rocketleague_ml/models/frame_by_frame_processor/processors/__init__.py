from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Callable
from rocketleague_ml.models.frame_by_frame_processor.processors.car_components import (
    process_dodge,
    process_double_jump,
    process_boost,
    process_activate_boost,
    process_steer,
    process_car_component,
)
from rocketleague_ml.models.frame_by_frame_processor.processors.camera_settings import (
    process_camera_settings,
    process_camera_settings_swivel,
    process_camera_settings_activate,
)
from rocketleague_ml.models.frame_by_frame_processor.processors.player_components import (
    process_player_team,
    process_player_assisted,
    process_player_scored,
    process_player_saved,
    process_player_gained_points,
    process_player_shot,
)
from rocketleague_ml.models.frame_by_frame_processor.processors.events import (
    process_game_start,
)
from rocketleague_ml.models.frame_by_frame_processor.processors.rigid_bodies import (
    process_ball_position,
    process_car_position,
)
from rocketleague_ml.models.frame_by_frame_processor.processors.misc import (
    process_demo,
    process_boost_pickup,
    process_team_ball_hit,
)
from rocketleague_ml.core.actor import Actor
from rocketleague_ml.core.frame import Frame

if TYPE_CHECKING:
    from rocketleague_ml.models.frame_by_frame_processor import Frame_By_Frame_Processor

processors: Dict[str, Callable[[Frame_By_Frame_Processor, Actor, Frame], None]] = {
    "camera_settings.camera_settings": process_camera_settings,
    "camera_settings.yaw": process_camera_settings_swivel,
    "camera_settings.pitch": process_camera_settings_swivel,
    "camera_settings.car_cam": process_camera_settings_activate,
    "camera_settings.rear_cam": process_camera_settings_activate,
    "car_component.dodge": process_dodge,
    "car_component.double_jump": process_double_jump,
    "car_component.boost": process_boost,
    "car_component.rep_boost": process_boost,
    "car_component.activate_boost": process_activate_boost,
    "car_component.steer": process_steer,
    "car_component.activate_handbrake": process_activate_boost,
    "car_component": process_car_component,
    "player_component.team": process_player_team,
    "game_start_event": process_game_start,
    "ball": process_ball_position,
    "car": process_car_position,
    "player_demo": process_demo,
    "boost_pickup": process_boost_pickup,
    "player_component.assists": process_player_assisted,
    "player_component.saves": process_player_saved,
    "player_component.score": process_player_gained_points,
    "player_component.shots": process_player_shot,
    "player_component.goals": process_player_scored,
    "team_ball_hit": process_team_ball_hit,
}
