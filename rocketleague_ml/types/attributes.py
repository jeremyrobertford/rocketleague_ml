from typing import TypedDict, List


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
    linear_velocity: Position_Dict | None
    angular_velocity: Position_Dict | None


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


Attribute = (
    String_Attribute
    | Float_Attribute
    | Byte_Attribute
    | Int_Attribute
    | Rigid_Body_Attribute
    | Active_Attribute
    | Boolean_Attribute
    | Cam_Settings_Attribute
)


class Raw_Actor(TypedDict):
    actor_id: int
    active_actor_id: int | None
    name_id: int | None
    steam_id: int | None
    object_id: int
    attribute: Attribute | None
    initial_trajectory: Trajectory | None


class Labeled_Raw_Actor(Raw_Actor):
    name: str | None
    object: str


class Raw_Frame(TypedDict):
    time: float
    delta: float
    new_actors: List[Raw_Actor]
    deleted_actors: List[int]
    updated_actors: List[Raw_Actor]
    initial_trajectory: Trajectory | None


class Network_Frames(TypedDict):
    frames: List[Raw_Frame]


class Raw_Game_Data(TypedDict):
    objects: List[str]
    network_frames: Network_Frames
