from __future__ import annotations
from typing import TypedDict, List, Union, Dict
from rocketleague_ml.core.car import Car, Car_Component
from rocketleague_ml.core.actor import Actor


class Position_Dict(TypedDict):
    x: float
    y: float
    z: float


class Rotation_Dict(TypedDict):
    x: float
    y: float
    z: float
    w: float


class Euler_Rotation_Dict(TypedDict):
    yaw: float | None  # radians
    pitch: float | None  # degrees
    roll: float | None


class Trajectory(TypedDict):
    location: Position_Dict
    rotation: Euler_Rotation_Dict | None


class Rigid_Body_Positioning(TypedDict):
    sleeping: bool
    location: Position_Dict
    rotation: Rotation_Dict | Euler_Rotation_Dict
    previous_linear_velocity: Position_Dict | None
    linear_velocity: Position_Dict | None
    angular_velocity: Position_Dict | None


class Time_Labeled_Rigid_Body_Positioning(TypedDict):
    round: int
    time: float
    match_time: float
    delta: float
    sleeping: bool
    location: Position_Dict
    rotation: Rotation_Dict
    linear_velocity: Position_Dict | None
    angular_velocity: Position_Dict | None


class Time_Labeled_Boost_Pickup(TypedDict):
    round: int
    time: float
    match_time: float
    delta: float
    location: Position_Dict
    amount: int


class Time_Labeled_Activity(TypedDict):
    round: int
    time: float
    match_time: float
    delta: float
    active: bool


class Time_Labeled_Amount(TypedDict):
    round: int
    time: float
    match_time: float
    delta: float
    amount: float


class Time_Labeled_Attacker_Demo(TypedDict):
    round: int
    time: float
    match_time: float
    delta: float
    linear_velocity: Position_Dict
    collision: Position_Dict
    victim_actor_id: int
    victim_linear_velocity: Position_Dict | None
    victim_collision: Position_Dict


class Time_Labeled_Victim_Demo(TypedDict):
    round: int
    time: float
    match_time: float
    delta: float
    linear_velocity: Position_Dict | None
    collision: Position_Dict
    attacker_actor_id: int
    attacker_linear_velocity: Position_Dict
    attacker_collision: Position_Dict


class Positioning_Dict(Rigid_Body_Positioning):
    time: float | None


class Rigid_Body_Attribute(TypedDict):
    RigidBody: Rigid_Body_Positioning


class Boolean_Attribute(TypedDict):
    Boolean: bool


class Float_Attribute(TypedDict):
    Float: float


class Byte_Attribute(TypedDict):
    Byte: int


class Int_Attribute(TypedDict):
    Int: int


class String_Attribute(TypedDict):
    String: str


class Active_Actor_Attribute(TypedDict):
    actor: int
    active: bool


class Active_Attribute(TypedDict):
    ActiveActor: Active_Actor_Attribute


class Location_Attribute(TypedDict):
    Location: Position_Dict


class Camera_Settings_Attribute(TypedDict):
    fov: float
    height: float
    angle: float
    distance: float
    stiffness: float
    swivel: float
    transition: float


class Cam_Settings_Attribute(TypedDict):
    CamSettings: Camera_Settings_Attribute


class Stat_Event(TypedDict):
    unknown1: bool
    object_id: int


class Stat_Event_Attribute(TypedDict):
    StatEvent: Stat_Event


class Labeled_Stat_Event(TypedDict):
    unknown1: bool
    object_id: int
    object: str


class Labeled_Stat_Event_Attribute(TypedDict):
    StatEvent: Labeled_Stat_Event


class Demolition_Attribute(TypedDict):
    attacker_pri: Active_Actor_Attribute
    self_demo: Active_Actor_Attribute
    self_demolish: bool
    goal_explosion_owner: Active_Actor_Attribute
    attacker: Active_Actor_Attribute
    victim: Active_Actor_Attribute
    attacker_velocity: Position_Dict
    victim_velocity: Position_Dict


class Demolish_Extended_Attribute(TypedDict):
    DemolishExtended: Demolition_Attribute


class Replicated_Boost(TypedDict):
    grant_count: int
    boost_amount: int


class Replicated_Boost_Attribute(TypedDict):
    ReplicatedBoost: Replicated_Boost


class Boost_Grab_Attribute(TypedDict):
    instigator: int
    picked_up: int


class Pickup_New_Attribute(TypedDict):
    PickupNew: Boost_Grab_Attribute


Attribute = (
    String_Attribute
    | Float_Attribute
    | Byte_Attribute
    | Int_Attribute
    | Rigid_Body_Attribute
    | Active_Attribute
    | Boolean_Attribute
    | Cam_Settings_Attribute
    | Stat_Event_Attribute
    | Demolish_Extended_Attribute
    | Replicated_Boost_Attribute
    | Location_Attribute
    | Pickup_New_Attribute
)

Labeled_Attribute = (
    String_Attribute
    | Float_Attribute
    | Byte_Attribute
    | Int_Attribute
    | Rigid_Body_Attribute
    | Active_Attribute
    | Boolean_Attribute
    | Cam_Settings_Attribute
    | Labeled_Stat_Event_Attribute
    | Demolish_Extended_Attribute
    | Replicated_Boost_Attribute
    | Location_Attribute
    | Pickup_New_Attribute
)


class Categorized_New_Actors(TypedDict):
    cars: List[Car]
    car_components: List[Car_Component]


class Categorized_Updated_Actors(TypedDict):
    events: List[Actor]
    others: List[Actor]
    cars_for_players: List[Actor]


class Raw_Actor(TypedDict):
    actor_id: int
    active_actor_id: int | None
    name_id: int | None
    steam_id: int | None
    object_id: int
    attribute: Attribute | None
    initial_trajectory: Trajectory | None


class Labeled_Raw_Actor(TypedDict):
    name: str | None
    object: str
    actor_id: int
    active_actor_id: int | None
    name_id: int | None
    steam_id: int | None
    object_id: int
    attribute: Labeled_Attribute | None
    initial_trajectory: Trajectory | None


class Raw_Frame(TypedDict):
    resync: bool | None
    time: float
    match_time: float
    match_time_label: str
    in_overtime: bool
    overtime_elapsed: float
    delta: float
    active: bool | None
    new_actors: List[Raw_Actor]
    deleted_actors: List[int]
    updated_actors: List[Raw_Actor]
    initial_trajectory: Trajectory | None


class Network_Frames(TypedDict):
    frames: List[Raw_Frame]


class Raw_Game_Properties(TypedDict):
    Id: str
    TeamSize: int


class Raw_Game_Data(TypedDict):
    names: List[str]
    objects: List[str]
    network_frames: Network_Frames
    properties: Raw_Game_Properties


JsonPrimitive = str | float | int | bool | None
JsonLike = Union[
    JsonPrimitive,
    "Actor_Export",
    List["Actor_Export"],
    Time_Labeled_Rigid_Body_Positioning,
    List[Time_Labeled_Rigid_Body_Positioning],
]

Actor_Export = Union[
    Dict[str, JsonLike],
    Dict[int, JsonLike],
    Dict[int, List[Time_Labeled_Rigid_Body_Positioning]],
]
