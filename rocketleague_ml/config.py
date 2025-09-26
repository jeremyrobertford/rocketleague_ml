import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
RAW_REPLAYS = os.path.join(DATA_DIR, "raw")
PREPROCESSED = os.path.join(DATA_DIR, "preprocessed")
PROCESSED = os.path.join(DATA_DIR, "processed")

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
    "TAGame.Vehicle_TA:ReplicatedJump": "rep_jump",
    "TAGame.Vehicle_TA:ReplicatedDodge": "rep_dodge",
    "TAGame.Vehicle_TA:ReplicatedBoost": "rep_boost",
    "TAGame.CarComponent_Boost_TA:ReplicatedBoostAmount": "rep_boost_amt",
}
PLAYER_COMPONENT_LABELS = {
    "TAGame.Default__PRI_TA": "",
    "Archetypes.Teams.Team0": "team",
    "Archetypes.Teams.Team1": "team",
    "TAGame.PRI_TA:SteeringSensitivity": "steering_sensitivity",
    "TAGame.PRI_TA:MatchScore": "score",
}
CAMERA_SETTINGS_LABELS = {
    "TAGame.Default__CameraSettingsActor_TA": "settings",
    "TAGame.CameraSettingsActor_TA:CameraYaw": "yaw",
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
}
