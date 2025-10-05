import os

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
}
PLAYER_COMPONENT_LABELS = {
    "TAGame.Default__PRI_TA": "",
    "Engine.PlayerReplicationInfo:Team": "team",
    "Archetypes.Teams.Team0": "Blue",
    "Archetypes.Teams.Team1": "Orange",
    "TAGame.PRI_TA:SteeringSensitivity": "steering_sensitivity",
    "TAGame.PRI_TA:MatchScore": "score",
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
