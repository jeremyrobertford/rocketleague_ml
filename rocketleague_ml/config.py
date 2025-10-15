import os
import numpy as np
from typing import Any, Dict

LOG = True

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
RAW_REPLAYS = (
    "C:/Users/jerem/OneDrive/Documents/My Games/Rocket League/TAGame/DemosEpic"
)
# RAW_REPLAYS = os.path.join(DATA_DIR, "raw")
PREPROCESSED = os.path.join(DATA_DIR, "preprocessed")
WRANGLED = os.path.join(DATA_DIR, "wrangled")
PROCESSED = os.path.join(DATA_DIR, "processed")
FEATURES = os.path.join(DATA_DIR, "features")
DEFAULT_BIN_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "bin"))

# Training configs
RANDOM_SEED = 42
TEST_SIZE = 0.2

# Feature configs
# Simplify entire map into a 3 x 3 x 3 grid
MAP_SIMPLIFICATION_AREAS = [3, 3, 3]
TAGS = [
    # "as_first_man",
    # "as_second_man",
    # "as_third_man",
    "as_front_man",
    "as_middle_man",
    "as_back_man",
    "in_offensive_third",
    "in_neutral_third",
    "in_defensive_third",
    "in_offensive_half",
    "in_defensive_half",
    "with_possession",
    "first_half_of_game",
    "second_half_of_game",
]
CAR_COMPONENT_LABELS = {
    "Archetypes.CarComponents.CarComponent_Boost": "boost",
    "Archetypes.CarComponents.CarComponent_Jump": "jump",
    "Archetypes.CarComponents.CarComponent_Dodge": "dodge",
    "Archetypes.CarComponents.CarComponent_FlipCar": "flip",
    "Archetypes.CarComponents.CarComponent_DoubleJump": "double_jump",
    "TAGame.Vehicle_TA:ReplicatedSteer": "steer",
    "TAGame.Vehicle_TA:ReplicatedThrottle": "throttle",
    "TAGame.Vehicle_TA:ReplicatedHandbrake": "handbrake",
    "TAGame.Vehicle_TA:bReplicatedHandbrake": "activate_handbrake",
    "TAGame.Vehicle_TA:ReplicatedJump": "rep_jump",
    "TAGame.Vehicle_TA:ReplicatedDodge": "rep_dodge",
    "TAGame.Vehicle_TA:ReplicatedBoost": "rep_boost",
    "TAGame.CarComponent_Boost_TA:ReplicatedBoost": "rep_boost",
    "TAGame.CarComponent_Boost_TA:ReplicatedBoostAmount": "rep_boost_amt",
    "TAGame.Vehicle_TA:bDriving": "driving",
    "TAGame.CarComponent_TA:ReplicatedActive": "activate_boost",
    "TAGame.CarComponent_Dodge_TA:DodgeTorque": "dodge",
    "TAGame.CarComponent_AirActivate_TA:AirActivateCount": "component_usage_in_air",
    "TAGame.CarComponent_DoubleJump_TA:DoubleJumpImpulse": "double_jump",
}
PLAYER_COMPONENT_LABELS = {
    "TAGame.Default__PRI_TA": "",
    "Engine.PlayerReplicationInfo:Team": "team",
    "Archetypes.Teams.Team0": "Blue",
    "Archetypes.Teams.Team1": "Orange",
    "TAGame.PRI_TA:SteeringSensitivity": "steering_sensitivity",
    "TAGame.PRI_TA:MatchAssists": "assists",
    "TAGame.PRI_TA:MatchSaves": "saves",
    "TAGame.PRI_TA:MatchScore": "score",
    "TAGame.PRI_TA:MatchShots": "shots",
    "TAGame.PRI_TA:MatchGoals": "goals",
    "TAGame.PRI_TA:TotalGameTimePlayed": "active_time",
    "Engine.PlayerReplicationInfo:Ping": "ping",
}
CAMERA_SETTINGS_LABELS = {
    "TAGame.Default__CameraSettingsActor_TA": "settings",
    "TAGame.CameraSettingsActor_TA:CameraYaw": "yaw",
    "TAGame.CameraSettingsActor_TA:CameraPitch": "pitch",
    "TAGame.CameraSettingsActor_TA:bUsingSecondaryCamera": "car_cam",
    "TAGame.CameraSettingsActor_TA:bUsingBehindView": "rear_cam",
    "TAGame.CameraSettingsActor_TA:ProfileSettings": "camera_settings",
}
LABELS = {
    "TAGame.CarComponent_TA:Vehicle": "vehicle",
    "TAGame.Ball_TA:GameEvent": "event",
    "TAGame.PRI_TA:ReplicatedGameEvent": "event",
    "TAGame.GameEvent_TA:ReplicatedGameStateTimeRemaining": "event",
    "TAGame.GameEvent_TA:ReplicatedStateName": "game_start_event",
    "TAGame.GameEvent_TA:ReplicatedRoundCountDownNumber": "round_countdown_event",
    "TAGame.GameEvent_Soccar_TA:bBallHasBeenHit": "team_hit_ball_change_event",
    "TAGame.GameEvent_Soccar_TA:RoundNum": "round_number_event",
    "StatEvents.Events.Goal": "goal_event",
    "Archetypes.Ball.Ball_Default": "ball",
    "Archetypes.Car.Car_Default": "car",
    "Engine.PlayerReplicationInfo:PlayerName": "player",
    "Engine.Pawn:PlayerReplicationInfo": "car_to_player",
    "TAGame.RBActor_TA:ReplicatedRBState": "rigid_body",
    "TAGame.CameraSettingsActor_TA:PRI": "settings_to_player",
    "TAGame.Car_TA:ReplicatedDemolishExtended": "player_demod",
    "TAGame.VehiclePickup_TA:NewReplicatedPickupData": "boost_pickup",
    "TAGame.Ball_TA:HitTeamNum": "team_ball_hit",
}

BALL_RADIUS = 91.25
SMALL_BOOST_RADIUS = 144.0
BIG_BOOST_RADIUS = 208.0
Z_GROUND = 17
Z_CEILING = 2044
X_WALL = 4096
Y_WALL = 5120
X_GOAL = 892
GOAL_DEPTH = 880
TOL = 50
FIELD_X = [-4096, 4096, 4096, -4096, -4096]
FIELD_Y = [-5120, -5120, 5120, 5120, -5120]
ROUND_LENGTH = 300
TOTAL_FIELD_DISTANCE = np.sqrt(
    (X_WALL * 2) ** 2 + (Y_WALL * 2) ** 2 + (Z_CEILING * 2) ** 2
)
BOOST_NEAR_MISS_THRESHOLD = 200


# Boost pad mapping for DFH Stadium (stadium_day_p, standard soccar layout).
# Data source: RLBot "useful game values" + community docs.

BOOST_PAD_MAP = {
    0: (0.0, -4240.0, 70.0, 12),
    1: (-1792.0, -4184.0, 70.0, 12),
    2: (1792.0, -4184.0, 70.0, 12),
    3: (-3072.0, -4096.0, 73.0, 100),  # big
    4: (3072.0, -4096.0, 73.0, 100),  # big
    5: (-940.0, -3308.0, 70.0, 12),
    6: (940.0, -3308.0, 70.0, 12),
    7: (0.0, -2816.0, 70.0, 12),
    8: (-3584.0, -2484.0, 70.0, 12),
    9: (3584.0, -2484.0, 70.0, 12),
    10: (-1788.0, -2300.0, 70.0, 12),
    11: (1788.0, -2300.0, 70.0, 12),
    12: (-2048.0, -1036.0, 70.0, 12),
    13: (0.0, -1024.0, 70.0, 12),
    14: (2048.0, -1036.0, 70.0, 12),
    15: (-3584.0, 0.0, 73.0, 100),  # big
    16: (-1024.0, 0.0, 70.0, 12),
    17: (1024.0, 0.0, 70.0, 12),
    18: (3584.0, 0.0, 73.0, 100),  # big
    19: (-2048.0, 1036.0, 70.0, 12),
    20: (0.0, 1024.0, 70.0, 12),
    21: (2048.0, 1036.0, 70.0, 12),
    22: (-1788.0, 2300.0, 70.0, 12),
    23: (1788.0, 2300.0, 70.0, 12),
    24: (-3584.0, 2484.0, 70.0, 12),
    25: (3584.0, 2484.0, 70.0, 12),
    26: (0.0, 2816.0, 70.0, 12),
    27: (-940.0, 3308.0, 70.0, 12),
    28: (940.0, 3308.0, 70.0, 12),
    29: (-3072.0, 4096.0, 73.0, 100),  # big
    30: (3072.0, 4096.0, 73.0, 100),  # big
}


DTYPES: Dict[str, Any] = {
    "float": float,
    "int": int,
    "str": str,
}

FEATURE_LABELS = {
    # Metadata
    "id": {
        "dtype": "str",
        "description": "Unique identifier for the game or round",
        "concept": "metadata",
    },
    "round": {
        "dtype": "int",
        "description": "Round number within the game",
        "concept": "metadata",
    },
    # Ball Engagement
    "Percent Time while Closest to Ball": {
        "dtype": "float",
        "description": "Percentage of time the player was the closest to the ball",
        "concept": "ball_engagement",
    },
    "Average Stint while Closest to Ball": {
        "dtype": "float",
        "description": "Average continuous time spent closest to the ball",
        "concept": "ball_engagement",
    },
    "Percent Time while Farthest from Ball": {
        "dtype": "float",
        "description": "Percentage of time the player was the farthest from the ball",
        "concept": "ball_engagement",
    },
    "Average Stint while Farthest from Ball": {
        "dtype": "float",
        "description": "Average continuous time spent farthest from the ball",
        "concept": "ball_engagement",
    },
    "Average Distance to Ball": {
        "dtype": "float",
        "description": "Mean distance from the ball during gameplay",
        "concept": "ball_engagement",
    },
    # Team Positioning
    "Average Distance to Teammates": {
        "dtype": "float",
        "description": "Average distance to teammates during gameplay",
        "concept": "team_positioning",
    },
    "Average Distance to Opponents": {
        "dtype": "float",
        "description": "Average distance to opponents during gameplay",
        "concept": "team_positioning",
    },
    # Field Control - Halves
    "Percent Time In Offensive Half": {
        "dtype": "float",
        "description": "Percent of time spent in the offensive half of the field",
        "concept": "field_control",
    },
    "Average Stint In Offensive Half": {
        "dtype": "float",
        "description": "Average continuous time spent in the offensive half",
        "concept": "field_control",
    },
    "Percent Time In Defensive Half": {
        "dtype": "float",
        "description": "Percent of time spent in the defensive half of the field",
        "concept": "field_control",
    },
    "Average Stint In Defensive Half": {
        "dtype": "float",
        "description": "Average continuous time spent in the defensive half",
        "concept": "field_control",
    },
    "Percent Time In Left Half": {
        "dtype": "float",
        "description": "Percent of time spent in the left half of the field",
        "concept": "field_control",
    },
    "Average Stint In Left Half": {
        "dtype": "float",
        "description": "Average continuous time spent in the left half",
        "concept": "field_control",
    },
    "Percent Time In Right Half": {
        "dtype": "float",
        "description": "Percent of time spent in the right half of the field",
        "concept": "field_control",
    },
    "Average Stint In Right Half": {
        "dtype": "float",
        "description": "Average continuous time spent in the right half",
        "concept": "field_control",
    },
    "Percent Time In Highest Half": {
        "dtype": "float",
        "description": "Percent of time spent in the upper half of the field (Z-axis)",
        "concept": "field_control",
    },
    "Average Stint In Highest Half": {
        "dtype": "float",
        "description": "Average continuous time spent in the upper half",
        "concept": "field_control",
    },
    "Percent Time In Lowest Half": {
        "dtype": "float",
        "description": "Percent of time spent in the lower half of the field (Z-axis)",
        "concept": "field_control",
    },
    "Average Stint In Lowest Half": {
        "dtype": "float",
        "description": "Average continuous time spent in the lower half",
        "concept": "field_control",
    },
    # Field Control - Thirds (Longitudinal)
    "Percent Time In Offensive Third": {
        "dtype": "float",
        "description": "Percent of time spent in offensive third of the field",
        "concept": "field_control",
    },
    "Average Stint In Offensive Third": {
        "dtype": "float",
        "description": "Average continuous time spent in offensive third",
        "concept": "field_control",
    },
    "Percent Time In Neutral Third": {
        "dtype": "float",
        "description": "Percent of time spent in neutral (middle) third",
        "concept": "field_control",
    },
    "Average Stint In Neutral Third": {
        "dtype": "float",
        "description": "Average continuous time spent in neutral third",
        "concept": "field_control",
    },
    "Percent Time In Defensive Third": {
        "dtype": "float",
        "description": "Percent of time spent in defensive third",
        "concept": "field_control",
    },
    "Average Stint In Defensive Third": {
        "dtype": "float",
        "description": "Average continuous time spent in defensive third",
        "concept": "field_control",
    },
    # Field Control - Thirds (Lateral)
    "Percent Time In Left Third": {
        "dtype": "float",
        "description": "Percent of time spent in left third of the field",
        "concept": "field_control",
    },
    "Average Stint In Left Third": {
        "dtype": "float",
        "description": "Average continuous time spent in left third",
        "concept": "field_control",
    },
    "Percent Time In Middle Third": {
        "dtype": "float",
        "description": "Percent of time spent in middle third of the field",
        "concept": "field_control",
    },
    "Average Stint In Middle Third": {
        "dtype": "float",
        "description": "Average continuous time spent in middle third",
        "concept": "field_control",
    },
    "Percent Time In Right Third": {
        "dtype": "float",
        "description": "Percent of time spent in right third of the field",
        "concept": "field_control",
    },
    "Average Stint In Right Third": {
        "dtype": "float",
        "description": "Average continuous time spent in right third",
        "concept": "field_control",
    },
    # Field Control - Thirds (Vertical)
    "Percent Time In Highest Third": {
        "dtype": "float",
        "description": "Percent of time spent in highest vertical third",
        "concept": "field_control",
    },
    "Average Stint In Highest Third": {
        "dtype": "float",
        "description": "Average continuous time spent in highest vertical third",
        "concept": "field_control",
    },
    "Percent Time In Middle Aerial Third": {
        "dtype": "float",
        "description": "Percent of time in middle vertical third",
        "concept": "field_control",
    },
    "Average Stint In Middle Aerial Third": {
        "dtype": "float",
        "description": "Average continuous time in middle vertical third",
        "concept": "field_control",
    },
    "Percent Time In Lowest Third": {
        "dtype": "float",
        "description": "Percent of time in lowest vertical third",
        "concept": "field_control",
    },
    "Average Stint In Lowest Third": {
        "dtype": "float",
        "description": "Average continuous time in lowest vertical third",
        "concept": "field_control",
    },
    # Offensive Left Half
    "Percent Time In Offensive Left Half": {
        "dtype": "float",
        "description": "Percent of time in the offensive left quadrant",
        "concept": "field_control",
    },
    "Average Stint In Offensive Left Half": {
        "dtype": "float",
        "description": "Average continuous time in offensive left quadrant",
        "concept": "field_control",
    },
    # Offensive Right Half
    "Percent Time In Offensive Right Half": {
        "dtype": "float",
        "description": "Percent of time in the offensive right quadrant",
        "concept": "field_control",
    },
    "Average Stint In Offensive Right Half": {
        "dtype": "float",
        "description": "Average continuous time in offensive right quadrant",
        "concept": "field_control",
    },
    # Defensive Left Half
    "Percent Time In Defensive Left Half": {
        "dtype": "float",
        "description": "Percent of time in the defensive left quadrant",
        "concept": "field_control",
    },
    "Average Stint In Defensive Left Half": {
        "dtype": "float",
        "description": "Average continuous time in defensive left quadrant",
        "concept": "field_control",
    },
    # Defensive Right Half
    "Percent Time In Defensive Right Half": {
        "dtype": "float",
        "description": "Percent of time in the defensive right quadrant",
        "concept": "field_control",
    },
    "Average Stint In Defensive Right Half": {
        "dtype": "float",
        "description": "Average continuous time in defensive right quadrant",
        "concept": "field_control",
    },
    # Offensive Left Vertical
    "Percent Time In Offensive Left Highest Half": {
        "dtype": "float",
        "description": "Percent of time in offensive left upper region",
        "concept": "field_control",
    },
    "Average Stint In Offensive Left Highest Half": {
        "dtype": "float",
        "description": "Average continuous time in offensive left upper region",
        "concept": "field_control",
    },
    "Percent Time In Offensive Left Lowest Half": {
        "dtype": "float",
        "description": "Percent of time in offensive left lower region",
        "concept": "field_control",
    },
    "Average Stint In Offensive Left Lowest Half": {
        "dtype": "float",
        "description": "Average continuous time in offensive left lower region",
        "concept": "field_control",
    },
    # Offensive Right Vertical
    "Percent Time In Offensive Right Highest Half": {
        "dtype": "float",
        "description": "Percent of time in offensive right upper region",
        "concept": "field_control",
    },
    "Average Stint In Offensive Right Highest Half": {
        "dtype": "float",
        "description": "Average continuous time in offensive right upper region",
        "concept": "field_control",
    },
    "Percent Time In Offensive Right Lowest Half": {
        "dtype": "float",
        "description": "Percent of time in offensive right lower region",
        "concept": "field_control",
    },
    "Average Stint In Offensive Right Lowest Half": {
        "dtype": "float",
        "description": "Average continuous time in offensive right lower region",
        "concept": "field_control",
    },
    # Defensive Left Vertical
    "Percent Time In Defensive Left Highest Half": {
        "dtype": "float",
        "description": "Percent of time in defensive left upper region",
        "concept": "field_control",
    },
    "Average Stint In Defensive Left Highest Half": {
        "dtype": "float",
        "description": "Average continuous time in defensive left upper region",
        "concept": "field_control",
    },
    "Percent Time In Defensive Left Lowest Half": {
        "dtype": "float",
        "description": "Percent of time in defensive left lower region",
        "concept": "field_control",
    },
    "Average Stint In Defensive Left Lowest Half": {
        "dtype": "float",
        "description": "Average continuous time in defensive left lower region",
        "concept": "field_control",
    },
    # Defensive Right Vertical
    "Percent Time In Defensive Right Highest Half": {
        "dtype": "float",
        "description": "Percent of time in defensive right upper region",
        "concept": "field_control",
    },
    "Average Stint In Defensive Right Highest Half": {
        "dtype": "float",
        "description": "Average continuous time in defensive right upper region",
        "concept": "field_control",
    },
    "Percent Time In Defensive Right Lowest Half": {
        "dtype": "float",
        "description": "Percent of time in defensive right lower region",
        "concept": "field_control",
    },
    "Average Stint In Defensive Right Lowest Half": {
        "dtype": "float",
        "description": "Average continuous time in defensive right lower region",
        "concept": "field_control",
    },
    # Offensive Thirds (detailed)
    "Percent Time In Offensive Left Third": {
        "dtype": "float",
        "description": "Percent of time in offensive left third",
        "concept": "field_control",
    },
    "Average Stint In Offensive Left Third": {
        "dtype": "float",
        "description": "Average continuous time in offensive left third",
        "concept": "field_control",
    },
    "Percent Time In Offensive Middle Third": {
        "dtype": "float",
        "description": "Percent of time in offensive middle third",
        "concept": "field_control",
    },
    "Average Stint In Offensive Middle Third": {
        "dtype": "float",
        "description": "Average continuous time in offensive middle third",
        "concept": "field_control",
    },
    "Percent Time In Offensive Right Third": {
        "dtype": "float",
        "description": "Percent of time in offensive right third",
        "concept": "field_control",
    },
    "Average Stint In Offensive Right Third": {
        "dtype": "float",
        "description": "Average continuous time in offensive right third",
        "concept": "field_control",
    },
    # Neutral Thirds (detailed)
    "Percent Time In Neutral Left Third": {
        "dtype": "float",
        "description": "Percent of time in neutral left third",
        "concept": "field_control",
    },
    "Average Stint In Neutral Left Third": {
        "dtype": "float",
        "description": "Average continuous time in neutral left third",
        "concept": "field_control",
    },
    "Percent Time In Neutral Middle Third": {
        "dtype": "float",
        "description": "Percent of time in neutral middle third",
        "concept": "field_control",
    },
    "Average Stint In Neutral Middle Third": {
        "dtype": "float",
        "description": "Average continuous time in neutral middle third",
        "concept": "field_control",
    },
    "Percent Time In Neutral Right Third": {
        "dtype": "float",
        "description": "Percent of time in neutral right third",
        "concept": "field_control",
    },
    "Average Stint In Neutral Right Third": {
        "dtype": "float",
        "description": "Average continuous time in neutral right third",
        "concept": "field_control",
    },
    # Defensive Thirds (detailed)
    "Percent Time In Defensive Left Third": {
        "dtype": "float",
        "description": "Percent of time in defensive left third",
        "concept": "field_control",
    },
    "Average Stint In Defensive Left Third": {
        "dtype": "float",
        "description": "Average continuous time in defensive left third",
        "concept": "field_control",
    },
    "Percent Time In Defensive Middle Third": {
        "dtype": "float",
        "description": "Percent of time in defensive middle third",
        "concept": "field_control",
    },
    "Average Stint In Defensive Middle Third": {
        "dtype": "float",
        "description": "Average continuous time in defensive middle third",
        "concept": "field_control",
    },
    "Percent Time In Defensive Right Third": {
        "dtype": "float",
        "description": "Percent of time in defensive right third",
        "concept": "field_control",
    },
    "Average Stint In Defensive Right Third": {
        "dtype": "float",
        "description": "Average continuous time in defensive right third",
        "concept": "field_control",
    },
    # Offensive Highest Thirds
    "Percent Time In Offensive Left Highest Third": {
        "dtype": "float",
        "description": "Percent of time in offensive left upper area",
        "concept": "field_control",
    },
    "Average Stint In Offensive Left Highest Third": {
        "dtype": "float",
        "description": "Average continuous time in offensive left upper area",
        "concept": "field_control",
    },
    "Percent Time In Offensive Middle Highest Third": {
        "dtype": "float",
        "description": "Percent of time in offensive center upper area",
        "concept": "field_control",
    },
    "Average Stint In Offensive Middle Highest Third": {
        "dtype": "float",
        "description": "Average continuous time in offensive center upper area",
        "concept": "field_control",
    },
    "Percent Time In Offensive Right Highest Third": {
        "dtype": "float",
        "description": "Percent of time in offensive right upper area",
        "concept": "field_control",
    },
    "Average Stint In Offensive Right Highest Third": {
        "dtype": "float",
        "description": "Average continuous time in offensive right upper area",
        "concept": "field_control",
    },
    # Neutral Highest Thirds
    "Percent Time In Neutral Left Highest Third": {
        "dtype": "float",
        "description": "Percent of time in neutral left upper area",
        "concept": "field_control",
    },
    "Average Stint In Neutral Left Highest Third": {
        "dtype": "float",
        "description": "Average continuous time in neutral left upper area",
        "concept": "field_control",
    },
    "Percent Time In Neutral Middle Highest Third": {
        "dtype": "float",
        "description": "Percent of time in neutral center upper area",
        "concept": "field_control",
    },
    "Average Stint In Neutral Middle Highest Third": {
        "dtype": "float",
        "description": "Average continuous time in neutral center upper area",
        "concept": "field_control",
    },
    "Percent Time In Neutral Right Highest Third": {
        "dtype": "float",
        "description": "Percent of time in neutral right upper area",
        "concept": "field_control",
    },
    "Average Stint In Neutral Right Highest Third": {
        "dtype": "float",
        "description": "Average continuous time in neutral right upper area",
        "concept": "field_control",
    },
    # Defensive Highest Thirds
    "Percent Time In Defensive Left Highest Third": {
        "dtype": "float",
        "description": "Percent of time in defensive left upper area",
        "concept": "field_control",
    },
    "Average Stint In Defensive Left Highest Third": {
        "dtype": "float",
        "description": "Average continuous time in defensive left upper area",
        "concept": "field_control",
    },
    "Percent Time In Defensive Middle Highest Third": {
        "dtype": "float",
        "description": "Percent of time in defensive center upper area",
        "concept": "field_control",
    },
    "Average Stint In Defensive Middle Highest Third": {
        "dtype": "float",
        "description": "Average continuous time in defensive center upper area",
        "concept": "field_control",
    },
    "Percent Time In Defensive Right Highest Third": {
        "dtype": "float",
        "description": "Percent of time in defensive right upper area",
        "concept": "field_control",
    },
    "Average Stint In Defensive Right Highest Third": {
        "dtype": "float",
        "description": "Average continuous time in defensive right upper area",
        "concept": "field_control",
    },
    # Offensive Middle-Aerial Thirds
    "Percent Time In Offensive Left Middle-Aerial Third": {
        "dtype": "float",
        "description": "Percent of time in offensive left mid-air area",
        "concept": "field_control",
    },
    "Average Stint In Offensive Left Middle-Aerial Third": {
        "dtype": "float",
        "description": "Average continuous time in offensive left mid-air area",
        "concept": "field_control",
    },
    "Percent Time In Offensive Middle Middle-Aerial Third": {
        "dtype": "float",
        "description": "Percent of time in offensive center mid-air area",
        "concept": "field_control",
    },
    "Average Stint In Offensive Middle Middle-Aerial Third": {
        "dtype": "float",
        "description": "Average continuous time in offensive center mid-air area",
        "concept": "field_control",
    },
    "Percent Time In Offensive Right Middle-Aerial Third": {
        "dtype": "float",
        "description": "Percent of time in offensive right mid-air area",
        "concept": "field_control",
    },
    "Average Stint In Offensive Right Middle-Aerial Third": {
        "dtype": "float",
        "description": "Average continuous time in offensive right mid-air area",
        "concept": "field_control",
    },
    # Neutral Middle-Aerial Thirds
    "Percent Time In Neutral Left Middle-Aerial Third": {
        "dtype": "float",
        "description": "Percent of time in neutral left mid-air area",
        "concept": "field_control",
    },
    "Average Stint In Neutral Left Middle-Aerial Third": {
        "dtype": "float",
        "description": "Average continuous time in neutral left mid-air area",
        "concept": "field_control",
    },
    "Percent Time In Neutral Middle Middle-Aerial Third": {
        "dtype": "float",
        "description": "Percent of time in neutral center mid-air area",
        "concept": "field_control",
    },
    "Average Stint In Neutral Middle Middle-Aerial Third": {
        "dtype": "float",
        "description": "Average continuous time in neutral center mid-air area",
        "concept": "field_control",
    },
    "Percent Time In Neutral Right Middle-Aerial Third": {
        "dtype": "float",
        "description": "Percent of time in neutral right mid-air area",
        "concept": "field_control",
    },
    "Average Stint In Neutral Right Middle-Aerial Third": {
        "dtype": "float",
        "description": "Average continuous time in neutral right mid-air area",
        "concept": "field_control",
    },
    # Defensive Middle-Aerial Thirds
    "Percent Time In Defensive Left Middle-Aerial Third": {
        "dtype": "float",
        "description": "Percent of time in defensive left mid-air area",
        "concept": "field_control",
    },
    "Average Stint In Defensive Left Middle-Aerial Third": {
        "dtype": "float",
        "description": "Average continuous time in defensive left mid-air area",
        "concept": "field_control",
    },
    "Percent Time In Defensive Middle Middle-Aerial Third": {
        "dtype": "float",
        "description": "Percent of time in defensive center mid-air area",
        "concept": "field_control",
    },
    "Average Stint In Defensive Middle Middle-Aerial Third": {
        "dtype": "float",
        "description": "Average continuous time in defensive center mid-air area",
        "concept": "field_control",
    },
    "Percent Time In Defensive Right Middle-Aerial Third": {
        "dtype": "float",
        "description": "Percent of time in defensive right mid-air area",
        "concept": "field_control",
    },
    "Average Stint In Defensive Right Middle-Aerial Third": {
        "dtype": "float",
        "description": "Average continuous time in defensive right mid-air area",
        "concept": "field_control",
    },
    # Offensive Lowest Thirds
    "Percent Time In Offensive Left Lowest Third": {
        "dtype": "float",
        "description": "Percent of time in offensive left lower area",
        "concept": "field_control",
    },
    "Average Stint In Offensive Left Lowest Third": {
        "dtype": "float",
        "description": "Average continuous time in offensive left lower area",
        "concept": "field_control",
    },
    "Percent Time In Offensive Middle Lowest Third": {
        "dtype": "float",
        "description": "Percent of time in offensive center lower area",
        "concept": "field_control",
    },
    "Average Stint In Offensive Middle Lowest Third": {
        "dtype": "float",
        "description": "Average continuous time in offensive center lower area",
        "concept": "field_control",
    },
    "Percent Time In Offensive Right Lowest Third": {
        "dtype": "float",
        "description": "Percent of time in offensive right lower area",
        "concept": "field_control",
    },
    "Average Stint In Offensive Right Lowest Third": {
        "dtype": "float",
        "description": "Average continuous time in offensive right lower area",
        "concept": "field_control",
    },
    # Neutral Lowest Thirds
    "Percent Time In Neutral Left Lowest Third": {
        "dtype": "float",
        "description": "Percent of time in neutral left lower area",
        "concept": "field_control",
    },
    "Average Stint In Neutral Left Lowest Third": {
        "dtype": "float",
        "description": "Average continuous time in neutral left lower area",
        "concept": "field_control",
    },
    "Percent Time In Neutral Middle Lowest Third": {
        "dtype": "float",
        "description": "Percent of time in neutral center lower area",
        "concept": "field_control",
    },
    "Average Stint In Neutral Middle Lowest Third": {
        "dtype": "float",
        "description": "Average continuous time in neutral center lower area",
        "concept": "field_control",
    },
    "Percent Time In Neutral Right Lowest Third": {
        "dtype": "float",
        "description": "Percent of time in neutral right lower area",
        "concept": "field_control",
    },
    "Average Stint In Neutral Right Lowest Third": {
        "dtype": "float",
        "description": "Average continuous time in neutral right lower area",
        "concept": "field_control",
    },
    # Defensive Lowest Thirds
    "Percent Time In Defensive Left Lowest Third": {
        "dtype": "float",
        "description": "Percent of time in defensive left lower area",
        "concept": "field_control",
    },
    "Average Stint In Defensive Left Lowest Third": {
        "dtype": "float",
        "description": "Average continuous time in defensive left lower area",
        "concept": "field_control",
    },
    "Percent Time In Defensive Middle Lowest Third": {
        "dtype": "float",
        "description": "Percent of time in defensive center lower area",
        "concept": "field_control",
    },
    "Average Stint In Defensive Middle Lowest Third": {
        "dtype": "float",
        "description": "Average continuous time in defensive center lower area",
        "concept": "field_control",
    },
    "Percent Time In Defensive Right Lowest Third": {
        "dtype": "float",
        "description": "Percent of time in defensive right lower area",
        "concept": "field_control",
    },
    "Average Stint In Defensive Right Lowest Third": {
        "dtype": "float",
        "description": "Average continuous time in defensive right lower area",
        "concept": "field_control",
    },
    # Ball Orientation
    "Percent Time In Front of Ball": {
        "dtype": "float",
        "description": "Percent of time positioned in front of the ball relative to the goal",
        "concept": "ball_orientation",
    },
    "Average Stint In Front of Ball": {
        "dtype": "float",
        "description": "Average continuous time positioned in front of the ball",
        "concept": "ball_orientation",
    },
    "Percent Time Behind Ball": {
        "dtype": "float",
        "description": "Percent of time positioned behind the ball relative to the goal",
        "concept": "ball_orientation",
    },
    "Average Stint Behind Ball": {
        "dtype": "float",
        "description": "Average continuous time positioned behind the ball",
        "concept": "ball_orientation",
    },
    # Movement State - Grounded/Airborne
    "Percent Time Grounded": {
        "dtype": "float",
        "description": "Percent of time with all wheels on the ground",
        "concept": "movement_state",
    },
    "Average Stint Grounded": {
        "dtype": "float",
        "description": "Average continuous time spent grounded",
        "concept": "movement_state",
    },
    "Percent Time Airborne": {
        "dtype": "float",
        "description": "Percent of time with wheels off the ground (jumping or flying)",
        "concept": "movement_state",
    },
    "Average Stint Airborne": {
        "dtype": "float",
        "description": "Average continuous time spent airborne",
        "concept": "movement_state",
    },
    # Wall/Ceiling Positioning
    "Percent Time On Ceiling": {
        "dtype": "float",
        "description": "Percent of time driving on the ceiling",
        "concept": "movement_state",
    },
    "Average Stint On Ceiling": {
        "dtype": "float",
        "description": "Average continuous time spent on the ceiling",
        "concept": "movement_state",
    },
    "Percent Time On Left Wall": {
        "dtype": "float",
        "description": "Percent of time driving on the left wall",
        "concept": "movement_state",
    },
    "Average Stint On Left Wall": {
        "dtype": "float",
        "description": "Average continuous time spent on the left wall",
        "concept": "movement_state",
    },
    "Percent Time On Right Wall": {
        "dtype": "float",
        "description": "Percent of time driving on the right wall",
        "concept": "movement_state",
    },
    "Average Stint On Right Wall": {
        "dtype": "float",
        "description": "Average continuous time spent on the right wall",
        "concept": "movement_state",
    },
    "Percent Time On Back Wall": {
        "dtype": "float",
        "description": "Percent of time driving on the back wall",
        "concept": "movement_state",
    },
    "Average Stint On Back Wall": {
        "dtype": "float",
        "description": "Average continuous time spent on the back wall",
        "concept": "movement_state",
    },
    "Percent Time On Front Wall": {
        "dtype": "float",
        "description": "Percent of time driving on the front wall",
        "concept": "movement_state",
    },
    "Average Stint On Front Wall": {
        "dtype": "float",
        "description": "Average continuous time spent on the front wall",
        "concept": "movement_state",
    },
    # Goal Areas
    "Percent Time In Own Goal": {
        "dtype": "float",
        "description": "Percent of time spent in own goal area",
        "concept": "field_control",
    },
    "Average Stint In Own Goal": {
        "dtype": "float",
        "description": "Average continuous time spent in own goal area",
        "concept": "field_control",
    },
    "Percent Time In Opponents Goal": {
        "dtype": "float",
        "description": "Percent of time spent in opponent's goal area",
        "concept": "field_control",
    },
    "Average Stint In Opponents Goal": {
        "dtype": "float",
        "description": "Average continuous time spent in opponent's goal area",
        "concept": "field_control",
    },
    # Speed States
    "Percent Time while Stationary": {
        "dtype": "float",
        "description": "Percent of time the player was stationary (no movement)",
        "concept": "speed_engagement",
    },
    "Average Stint while Stationary": {
        "dtype": "float",
        "description": "Average continuous time spent stationary",
        "concept": "speed_engagement",
    },
    "Percent Time while Slow": {
        "dtype": "float",
        "description": "Percent of time moving at slow speed (low throttle)",
        "concept": "speed_engagement",
    },
    "Average Stint while Slow": {
        "dtype": "float",
        "description": "Average continuous time at slow speed",
        "concept": "speed_engagement",
    },
    "Percent Time while Semi-Slow": {
        "dtype": "float",
        "description": "Percent of time moving at semi-slow speed",
        "concept": "speed_engagement",
    },
    "Average Stint while Semi-Slow": {
        "dtype": "float",
        "description": "Average continuous time at semi-slow speed",
        "concept": "speed_engagement",
    },
    "Percent Time while Medium Speed": {
        "dtype": "float",
        "description": "Percent of time moving at medium speed",
        "concept": "speed_engagement",
    },
    "Average Stint while Medium Speed": {
        "dtype": "float",
        "description": "Average continuous time at medium speed",
        "concept": "speed_engagement",
    },
    "Percent Time while Semi-Fast": {
        "dtype": "float",
        "description": "Percent of time moving at semi-fast speed",
        "concept": "speed_engagement",
    },
    "Average Stint while Semi-Fast": {
        "dtype": "float",
        "description": "Average continuous time at semi-fast speed",
        "concept": "speed_engagement",
    },
    "Percent Time while Drive Speed": {
        "dtype": "float",
        "description": "Percent of time moving at standard drive speed",
        "concept": "speed_engagement",
    },
    "Average Stint while Drive Speed": {
        "dtype": "float",
        "description": "Average continuous time at drive speed",
        "concept": "speed_engagement",
    },
    "Percent Time while Boost Speed": {
        "dtype": "float",
        "description": "Percent of time moving at boosted speed",
        "concept": "speed_engagement",
    },
    "Average Stint while Boost Speed": {
        "dtype": "float",
        "description": "Average continuous time at boosted speed",
        "concept": "speed_engagement",
    },
    "Percent Time while Supersonic": {
        "dtype": "float",
        "description": "Percent of time moving at supersonic speed",
        "concept": "speed_engagement",
    },
    "Average Stint while Supersonic": {
        "dtype": "float",
        "description": "Average continuous time at supersonic speed",
        "concept": "speed_engagement",
    },
}
