from math import sqrt, atan2, acos
from rocketleague_ml.types.attributes import (
    Position_Dict,
    Rotation_Dict,
    Rigid_Body_Positioning,
)


class Base_Position:
    def __init__(self, position: Position_Dict):
        self.raw = position
        self.x = position["x"]
        self.y = position["y"]
        self.z = position["z"]


class Rotation:
    def __init__(self, rotation: Rotation_Dict):
        self.raw = rotation
        self.x = rotation["x"]
        self.y = rotation["y"]
        self.z = rotation["z"]
        self.w = rotation["w"]


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
    def __init__(self, positioning: Rigid_Body_Positioning):
        self.sleeping = positioning["sleeping"]
        self.location = positioning["location"]
        self.rotation = positioning["rotation"]
        self.linear_velocty = positioning["linear_velocty"]
        self.angular_velocty = positioning["angular_velocty"]
