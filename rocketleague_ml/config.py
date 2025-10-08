import os
from typing import Any, Dict

LOG = True

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
RAW_REPLAYS = os.path.join(DATA_DIR, "raw")
PREPROCESSED = os.path.join(DATA_DIR, "preprocessed")
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
    "TAGame.Vehicle_TA:bReplicatedHandbrake": "handbrake_active",
    "TAGame.Vehicle_TA:ReplicatedJump": "rep_jump",
    "TAGame.Vehicle_TA:ReplicatedDodge": "rep_dodge",
    "TAGame.Vehicle_TA:ReplicatedBoost": "rep_boost",
    "TAGame.CarComponent_Boost_TA:ReplicatedBoost": "rep_boost",
    "TAGame.CarComponent_Boost_TA:ReplicatedBoostAmount": "rep_boost_amt",
    "TAGame.Vehicle_TA:bDriving": "driving",
    "TAGame.CarComponent_TA:ReplicatedActive": "active",
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
    "TAGame.VehiclePickup_TA:NewReplicatedPickupData": "boost_grab",
    "TAGame.Ball_TA:HitTeamNum": "team_ball_hit",
}

Z_GROUND = 30
Z_CEILING = 2000
X_WALL = 4096
Y_WALL = 5120
X_GOAL = 892
GOAL_DEPTH = 880
TOL = 50
FIELD_X = [-4096, 4096, 4096, -4096, -4096]
FIELD_Y = [-5120, -5120, 5120, 5120, -5120]
ROUND_LENGTH = 300

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
    # Ball proximity
    "Percent Time while Closest to Ball": {
        "dtype": "float",
        "description": "Percentage of time the player was the closest to the ball",
        "concept": "ball_proximity",
    },
    "Average Stint while Closest to Ball": {
        "dtype": "float",
        "description": "Average continuous time spent closest to the ball",
        "concept": "ball_proximity",
    },
    "Percent Time while Farthest from Ball": {
        "dtype": "float",
        "description": "Percentage of time the player was the farthest from the ball",
        "concept": "ball_proximity",
    },
    "Average Stint while Farthest from Ball": {
        "dtype": "float",
        "description": "Average continuous time spent farthest from the ball",
        "concept": "ball_proximity",
    },
    "Average Distance to Ball": {
        "dtype": "float",
        "description": "Mean distance from the ball during gameplay",
        "concept": "ball_proximity",
    },
    # Player spacing
    "Average Distance to Teammates": {
        "dtype": "float",
        "description": "Average distance to teammates during gameplay",
        "concept": "positioning",
    },
    "Average Distance to Opponents": {
        "dtype": "float",
        "description": "Average distance to opponents during gameplay",
        "concept": "positioning",
    },
    # Field halves
    "Percent Time In Offensive Half": {
        "dtype": "float",
        "description": "Percent of time spent in the offensive half of the field",
        "concept": "positioning",
    },
    "Average Stint In Offensive Half": {
        "dtype": "float",
        "description": "Average continuous time spent in the offensive half",
        "concept": "positioning",
    },
    "Percent Time In Defensive Half": {
        "dtype": "float",
        "description": "Percent of time spent in the defensive half of the field",
        "concept": "positioning",
    },
    "Average Stint In Defensive Half": {
        "dtype": "float",
        "description": "Average continuous time spent in the defensive half",
        "concept": "positioning",
    },
    "Percent Time In Left Half": {
        "dtype": "float",
        "description": "Percent of time spent in the left half of the field",
        "concept": "positioning",
    },
    "Average Stint In Left Half": {
        "dtype": "float",
        "description": "Average continuous time spent in the left half",
        "concept": "positioning",
    },
    "Percent Time In Right Half": {
        "dtype": "float",
        "description": "Percent of time spent in the right half of the field",
        "concept": "positioning",
    },
    "Average Stint In Right Half": {
        "dtype": "float",
        "description": "Average continuous time spent in the right half",
        "concept": "positioning",
    },
    "Percent Time In Highest Half": {
        "dtype": "float",
        "description": "Percent of time spent in the upper half of the field (Z-axis)",
        "concept": "positioning",
    },
    "Average Stint In Highest Half": {
        "dtype": "float",
        "description": "Average continuous time spent in the upper half",
        "concept": "positioning",
    },
    "Percent Time In Lowest Half": {
        "dtype": "float",
        "description": "Percent of time spent in the lower half of the field (Z-axis)",
        "concept": "positioning",
    },
    "Average Stint In Lowest Half": {
        "dtype": "float",
        "description": "Average continuous time spent in the lower half",
        "concept": "positioning",
    },
    # Field thirds
    "Percent Time In Offensive Third": {
        "dtype": "float",
        "description": "Percent of time spent in offensive third of the field",
        "concept": "positioning",
    },
    "Average Stint In Offensive Third": {
        "dtype": "float",
        "description": "Average continuous time spent in offensive third",
        "concept": "positioning",
    },
    "Percent Time In Neutral Third": {
        "dtype": "float",
        "description": "Percent of time spent in neutral (middle) third",
        "concept": "positioning",
    },
    "Average Stint In Neutral Third": {
        "dtype": "float",
        "description": "Average continuous time spent in neutral third",
        "concept": "positioning",
    },
    "Percent Time In Defensive Third": {
        "dtype": "float",
        "description": "Percent of time spent in defensive third",
        "concept": "positioning",
    },
    "Average Stint In Defensive Third": {
        "dtype": "float",
        "description": "Average continuous time spent in defensive third",
        "concept": "positioning",
    },
    # Left/Right thirds
    "Percent Time In Left Third": {
        "dtype": "float",
        "description": "Percent of time spent in left third of the field",
        "concept": "positioning",
    },
    "Average Stint In Left Third": {
        "dtype": "float",
        "description": "Average continuous time spent in left third",
        "concept": "positioning",
    },
    "Percent Time In Middle Third": {
        "dtype": "float",
        "description": "Percent of time spent in middle third of the field",
        "concept": "positioning",
    },
    "Average Stint In Middle Third": {
        "dtype": "float",
        "description": "Average continuous time spent in middle third",
        "concept": "positioning",
    },
    "Percent Time In Right Third": {
        "dtype": "float",
        "description": "Percent of time spent in right third of the field",
        "concept": "positioning",
    },
    "Average Stint In Right Third": {
        "dtype": "float",
        "description": "Average continuous time spent in right third",
        "concept": "positioning",
    },
    # Aerial thirds
    "Percent Time In Highest Third": {
        "dtype": "float",
        "description": "Percent of time spent in highest vertical third",
        "concept": "positioning",
    },
    "Average Stint In Highest Third": {
        "dtype": "float",
        "description": "Average continuous time spent in highest vertical third",
        "concept": "positioning",
    },
    "Percent Time In Middle Aerial Third": {
        "dtype": "float",
        "description": "Percent of time in middle vertical third",
        "concept": "positioning",
    },
    "Average Stint In Middle Aerial Third": {
        "dtype": "float",
        "description": "Average continuous time in middle vertical third",
        "concept": "positioning",
    },
    "Percent Time In Lowest Third": {
        "dtype": "float",
        "description": "Percent of time in lowest vertical third",
        "concept": "positioning",
    },
    "Average Stint In Lowest Third": {
        "dtype": "float",
        "description": "Average continuous time in lowest vertical third",
        "concept": "positioning",
    },
    # Ball orientation
    "Percent Time In Front of Ball": {
        "dtype": "float",
        "description": "Percent of time in front of ball relative to goal",
        "concept": "orientation",
    },
    "Average Stint In Front of Ball": {
        "dtype": "float",
        "description": "Average continuous time spent in front of the ball",
        "concept": "orientation",
    },
    "Percent Time Behind Ball": {
        "dtype": "float",
        "description": "Percent of time behind the ball relative to goal",
        "concept": "orientation",
    },
    "Average Stint Behind Ball": {
        "dtype": "float",
        "description": "Average continuous time spent behind the ball",
        "concept": "orientation",
    },
    # Speed states
    "Percent Time while Stationary": {
        "dtype": "float",
        "description": "Percent of time the player was stationary",
        "concept": "speed",
    },
    "Average Stint while Stationary": {
        "dtype": "float",
        "description": "Average continuous time spent stationary",
        "concept": "speed",
    },
    "Percent Time while Slow": {
        "dtype": "float",
        "description": "Percent of time moving at slow speed",
        "concept": "speed",
    },
    "Average Stint while Slow": {
        "dtype": "float",
        "description": "Average continuous time at slow speed",
        "concept": "speed",
    },
    "Percent Time while Semi-Slow": {
        "dtype": "float",
        "description": "Percent of time moving at semi-slow speed",
        "concept": "speed",
    },
    "Average Stint while Semi-Slow": {
        "dtype": "float",
        "description": "Average continuous time at semi-slow speed",
        "concept": "speed",
    },
    "Percent Time while Medium Speed": {
        "dtype": "float",
        "description": "Percent of time moving at medium speed",
        "concept": "speed",
    },
    "Average Stint while Medium Speed": {
        "dtype": "float",
        "description": "Average continuous time at medium speed",
        "concept": "speed",
    },
    "Percent Time while Semi-Fast": {
        "dtype": "float",
        "description": "Percent of time moving at semi-fast speed",
        "concept": "speed",
    },
    "Average Stint while Semi-Fast": {
        "dtype": "float",
        "description": "Average continuous time at semi-fast speed",
        "concept": "speed",
    },
    "Percent Time while Drive Speed": {
        "dtype": "float",
        "description": "Percent of time moving at standard drive speed",
        "concept": "speed",
    },
    "Average Stint while Drive Speed": {
        "dtype": "float",
        "description": "Average continuous time at drive speed",
        "concept": "speed",
    },
    "Percent Time while Boost Speed": {
        "dtype": "float",
        "description": "Percent of time moving at boosted speed",
        "concept": "speed",
    },
    "Average Stint while Boost Speed": {
        "dtype": "float",
        "description": "Average continuous time at boosted speed",
        "concept": "speed",
    },
    "Percent Time while Supersonic": {
        "dtype": "float",
        "description": "Percent of time moving at supersonic speed",
        "concept": "speed",
    },
    "Average Stint while Supersonic": {
        "dtype": "float",
        "description": "Average continuous time at supersonic speed",
        "concept": "speed",
    },
    # Movement state
    "Percent Time Grounded": {
        "dtype": "float",
        "description": "Percent of time on the ground",
        "concept": "movement_state",
    },
    "Average Stint Grounded": {
        "dtype": "float",
        "description": "Average continuous time on the ground",
        "concept": "movement_state",
    },
    "Percent Time Airborne": {
        "dtype": "float",
        "description": "Percent of time in the air",
        "concept": "movement_state",
    },
    "Average Stint Airborne": {
        "dtype": "float",
        "description": "Average continuous time airborne",
        "concept": "movement_state",
    },
    "Percent Time On Ceiling": {
        "dtype": "float",
        "description": "Percent of time on the ceiling",
        "concept": "movement_state",
    },
    "Average Stint On Ceiling": {
        "dtype": "float",
        "description": "Average continuous time on ceiling",
        "concept": "movement_state",
    },
    "Percent Time On Left Wall": {
        "dtype": "float",
        "description": "Percent of time on left wall",
        "concept": "movement_state",
    },
    "Average Stint On Left Wall": {
        "dtype": "float",
        "description": "Average continuous time on left wall",
        "concept": "movement_state",
    },
    "Percent Time On Right Wall": {
        "dtype": "float",
        "description": "Percent of time on right wall",
        "concept": "movement_state",
    },
    "Average Stint On Right Wall": {
        "dtype": "float",
        "description": "Average continuous time on right wall",
        "concept": "movement_state",
    },
    "Percent Time On Back Wall": {
        "dtype": "float",
        "description": "Percent of time on back wall",
        "concept": "movement_state",
    },
    "Average Stint On Back Wall": {
        "dtype": "float",
        "description": "Average continuous time on back wall",
        "concept": "movement_state",
    },
    "Percent Time On Front Wall": {
        "dtype": "float",
        "description": "Percent of time on front wall",
        "concept": "movement_state",
    },
    "Average Stint On Front Wall": {
        "dtype": "float",
        "description": "Average continuous time on front wall",
        "concept": "movement_state",
    },
    # Goal areas
    "Percent Time In Own Goal": {
        "dtype": "float",
        "description": "Percent of time in own goal area",
        "concept": "positioning",
    },
    "Average Stint In Own Goal": {
        "dtype": "float",
        "description": "Average continuous time in own goal area",
        "concept": "positioning",
    },
    "Percent Time In Opponents Goal": {
        "dtype": "float",
        "description": "Percent of time in opponent's goal area",
        "concept": "positioning",
    },
    "Average Stint In Opponents Goal": {
        "dtype": "float",
        "description": "Average continuous time in opponent's goal area",
        "concept": "positioning",
    },
}
