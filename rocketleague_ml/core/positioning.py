from typing import Tuple
from math import sqrt, atan2, acos
from rocketleague_ml.utils.helpers import convert_euler_to_quat
from rocketleague_ml.types.attributes import (
    Position_Dict,
    Rotation_Dict,
    Rigid_Body_Positioning,
    Trajectory,
)


class Base_Position:
    def __init__(self, position: Position_Dict):
        self.raw = position
        self.x = position["x"] or 0
        self.y = position["y"] or 0
        self.z = position["z"] or 0

    def copy(self):
        return Base_Position(self.to_dict())

    def to_tuple(self) -> Tuple[float | int, float | int, float | int]:
        return (
            self.x,
            self.y,
            self.z,
        )

    def to_dict(self) -> Position_Dict:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
        }


class Rotation:
    def __init__(self, rotation: Rotation_Dict):
        self.raw = rotation
        self.x = rotation["x"]
        self.y = rotation["y"]
        self.z = rotation["z"]
        self.w = rotation["w"]

    def copy(self):
        return Rotation(self.to_dict())

    def to_tuple(self) -> Tuple[float | int, float | int, float | int, float | int]:
        return (
            self.x,
            self.y,
            self.z,
            self.w,
        )

    def to_dict(self) -> Rotation_Dict:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "w": self.w,
        }


class Vector:
    def __init__(self, start_position: Base_Position, end_position: Base_Position):
        self.start = start_position
        self.end = end_position
        self.displacement = self.get_displacement()
        self.magnitude = self.get_magnitude()
        self.polar = self.convert_to_polar()

        self.xy_start = Base_Position({"x": self.start.x, "y": self.start.y, "z": 0})
        self.xy_end = Base_Position({"x": self.end.x, "y": self.end.y, "z": 0})
        self.xy_displacement = self.get_displacement(self.xy_start, self.xy_end)
        self.xy_magnitude = self.get_magnitude(self.xy_displacement)
        self.xy_polar = self.convert_to_polar(self.xy_displacement)

    def get_displacement(
        self,
        start_position: Base_Position | None = None,
        end_position: Base_Position | None = None,
    ):
        start = start_position or self.start
        end = end_position or self.end
        return Base_Position(
            {
                "x": end.x - start.x,
                "y": end.y - start.y,
                "z": end.z - start.z,
            }
        )

    def get_magnitude(self, displacement: Base_Position | None = None):
        displ = displacement or self.displacement
        return sqrt(displ.x**2 + displ.y**2 + displ.z**2)

    def convert_to_polar(self, displacement: Base_Position | None = None):
        displ = displacement or self.displacement
        magnitude = self.get_magnitude(displ)
        xy_rotation = atan2(displ.y, displ.x)
        z_rotation = (
            acos(displ.z / magnitude) if displ.z > 0 and magnitude != 0 else 0.0
        )
        return [magnitude, xy_rotation, z_rotation]


class Position(Base_Position):
    def __init__(self, position: Position_Dict):
        super().__init__(position)

    def get_vector_to(self, end_position: Base_Position):
        return Vector(self, end_position)

    def get_vector_from(self, start_position: Base_Position):
        return Vector(start_position, self)


class Velocity(Position):
    def __init__(self, position: Position_Dict):
        super().__init__(position)


class Positioning:
    def __init__(self, positioning: Rigid_Body_Positioning | Trajectory):
        default_position: Position_Dict = {"x": 0, "y": 0, "z": 0}
        location = positioning["location"] or default_position
        self.location = Position(location)
        rotation = positioning["rotation"] or {"x": 0, "y": 0, "z": 0, "w": 0}
        if "yaw" in rotation:
            self.rotation = Rotation(
                convert_euler_to_quat(
                    rotation["yaw"], rotation["pitch"], rotation["roll"]
                )
            )
        else:
            self.rotation = Rotation(rotation)

        self.sleeping = positioning["sleeping"] if "sleeping" in positioning else True
        self.previous_linear_velocity = (
            Position(positioning["previous_linear_velocity"])
            if "previous_linear_velocity" in positioning
            and positioning["previous_linear_velocity"]
            else None
        )
        self.linear_velocity = (
            Position(positioning["linear_velocity"])
            if "linear_velocity" in positioning and positioning["linear_velocity"]
            else Position(default_position)
        )
        self.angular_velocity = (
            Position(positioning["angular_velocity"])
            if "angular_velocity" in positioning and positioning["angular_velocity"]
            else Position(default_position)
        )

    def copy(self):
        return Positioning(self.to_dict())

    def to_dict(self) -> Rigid_Body_Positioning:
        location = self.location.to_dict()
        rotation = self.rotation.to_dict()
        linear_velocity = (
            self.linear_velocity.to_dict() if self.linear_velocity else None
        )
        previous_linear_velocity = (
            self.previous_linear_velocity.to_dict()
            if self.previous_linear_velocity
            else None
        )
        angular_velocity = (
            self.angular_velocity.to_dict() if self.angular_velocity else None
        )
        return {
            "sleeping": self.sleeping,
            "location": location,
            "rotation": rotation,
            "linear_velocity": linear_velocity,
            "previous_linear_velocity": previous_linear_velocity,
            "angular_velocity": angular_velocity,
        }
