import math
from rocketleague_ml.core.frame import Frame
from typing import Dict, List, Optional, Tuple, TypedDict
from dataclasses import dataclass
from enum import Enum


class CollisionType(Enum):
    PLAYER = "player"
    GROUND = "ground"
    CEILING = "ceiling"
    WALL = "wall"
    CORNER_WALL = "corner_wall"
    GOAL_WALL = "goal_wall"
    UNKNOWN = "unknown"


@dataclass
class Vector3:
    x: float
    y: float
    z: float

    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalized(self) -> "Vector3":
        mag = self.magnitude()
        if mag == 0:
            return Vector3(0, 0, 0)
        return Vector3(self.x / mag, self.y / mag, self.z / mag)

    def __sub__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __add__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __mul__(self, scalar: float) -> "Vector3":
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def dot(self, other: "Vector3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vector3") -> "Vector3":
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def distance_to(self, other: "Vector3") -> float:
        return (self - other).magnitude()

    def __repr__(self) -> str:
        return f"({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"


@dataclass
class Quaternion:
    x: float
    y: float
    z: float
    w: float

    def to_rotation_matrix(self) -> List[List[float]]:
        """Convert quaternion to 3x3 rotation matrix."""
        xx = self.x * self.x
        yy = self.y * self.y
        zz = self.z * self.z
        xy = self.x * self.y
        xz = self.x * self.z
        yz = self.y * self.z
        wx = self.w * self.x
        wy = self.w * self.y
        wz = self.w * self.z

        return [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ]

    def rotate_vector(self, v: Vector3) -> Vector3:
        """Rotate a vector by this quaternion."""
        matrix = self.to_rotation_matrix()
        return Vector3(
            matrix[0][0] * v.x + matrix[0][1] * v.y + matrix[0][2] * v.z,
            matrix[1][0] * v.x + matrix[1][1] * v.y + matrix[1][2] * v.z,
            matrix[2][0] * v.x + matrix[2][1] * v.y + matrix[2][2] * v.z,
        )

    def get_forward_vector(self) -> Vector3:
        """Get the forward direction vector from the quaternion."""
        return self.rotate_vector(Vector3(1, 0, 0))

    def get_up_vector(self) -> Vector3:
        """Get the up direction vector from the quaternion."""
        return self.rotate_vector(Vector3(0, 0, 1))

    def get_right_vector(self) -> Vector3:
        """Get the right direction vector from the quaternion."""
        return self.rotate_vector(Vector3(0, 1, 0))


@dataclass
class CarHitbox:
    length: float
    width: float
    height: float
    offset: Vector3 | None = None  # Offset from car origin to hitbox center

    def __post_init__(self):
        if self.offset is None:
            self.offset = Vector3(0, 0, 0)


@dataclass
class CollisionInfo:
    collision_type: CollisionType
    frame_number: Optional[int] = None

    # Ball info
    ball_position: Optional[Vector3] = None
    ball_velocity_before: Optional[Vector3] = None
    ball_velocity_after: Optional[Vector3] = None
    ball_velocity_change: Optional[Vector3] = None
    ball_impact_point: Optional[Vector3] = None  # Point on ball surface

    # Collision physics
    impulse_magnitude: float = 0
    impulse_vector: Optional[Vector3] = None
    collision_normal: Optional[Vector3] = None

    # Player collision specific
    player_name: Optional[str] = None
    player_position: Optional[Vector3] = None
    player_rotation: Optional[Quaternion] = None
    car_impact_point: Optional[Vector3] = None  # Point on car hitbox
    car_impact_surface: Optional[str] = None  # "front", "roof", "side", etc
    distance_to_ball: Optional[float] = None

    # Environment collision specific
    environment_surface: Optional[str] = None  # "ground", "ceiling", "wall", etc
    wall_normal: Optional[Vector3] = None

    confidence: str = "unknown"  # "high", "medium", "low"

    def is_player_collision(self) -> bool:
        return self.collision_type == CollisionType.PLAYER

    def is_environment_collision(self) -> bool:
        return self.collision_type in [
            CollisionType.GROUND,
            CollisionType.CEILING,
            CollisionType.WALL,
            CollisionType.CORNER_WALL,
            CollisionType.GOAL_WALL,
        ]


class RocketLeagueArena:
    """Standard Rocket League arena dimensions (Soccar)."""

    # Field dimensions
    FIELD_LENGTH = 10240  # x-axis (-5120 to 5120)
    FIELD_WIDTH = 8192  # y-axis (-4096 to 4096)
    FIELD_HEIGHT = 2044  # z-axis (0 to 2044)

    # Goal dimensions
    GOAL_WIDTH = 1786
    GOAL_HEIGHT = 642
    GOAL_DEPTH = 880

    # Corner rounding
    CORNER_RADIUS = 1152

    @classmethod
    def get_boundaries(cls) -> Dict[str, Tuple[float, float]]:
        """Get arena boundaries as (min, max) tuples."""
        return {
            "x": (-cls.FIELD_LENGTH / 2, cls.FIELD_LENGTH / 2),
            "y": (-cls.FIELD_WIDTH / 2, cls.FIELD_WIDTH / 2),
            "z": (0, cls.FIELD_HEIGHT),
        }


class RocketLeagueCollisionDetector:
    # Rocket League constants
    BALL_MASS = 30.0  # mass units
    BALL_RADIUS = 91.25  # unreal units

    # Car hitbox presets (from RocketLeague documentation)
    HITBOX_PRESETS = {
        "octane": CarHitbox(118.007, 84.1995, 36.1591, Vector3(13.88, 0, 20.75)),
        "dominus": CarHitbox(127.927, 83.2800, 31.3000, Vector3(9.00, 0, 15.75)),
        "plank": CarHitbox(128.82, 84.67, 29.39, Vector3(3.50, 0, 18.26)),
        "breakout": CarHitbox(131.49, 80.52, 30.30, Vector3(1.42, 0, 16.50)),
        "hybrid": CarHitbox(127.02, 82.19, 34.16, Vector3(0.97, 0, 18.29)),
        "merc": CarHitbox(121.41, 83.91, 41.66, Vector3(-2.30, 0, 28.30)),
    }

    # Detection thresholds
    MAX_COLLISION_DISTANCE = 250.0  # uu - max distance for likely collision
    MIN_BALL_VELOCITY_CHANGE = 50.0  # uu/s - minimum change to consider collision
    ENVIRONMENT_COLLISION_THRESHOLD = (
        25.0  # Distance to surface for environment collision
    )

    def __init__(self, ball_mass: float = BALL_MASS, default_hitbox: str = "octane"):
        self.ball_mass = ball_mass
        self.default_hitbox = self.HITBOX_PRESETS.get(
            default_hitbox, self.HITBOX_PRESETS["octane"]
        )
        self.arena = RocketLeagueArena()

    def parse_frame(
        self, frame: Frame
    ) -> Tuple[Vector3, Vector3, Dict[str, Tuple[Vector3, Vector3, Quaternion]]]:
        """
        Parse frame data into ball position/velocity and player positions/velocities/rotations.

        Returns:
            ball_pos, ball_vel, players_dict
            where players_dict = {name: (position, velocity, rotation)}
        """
        frame_data = frame.processed_fields
        ball_pos = Vector3(
            frame_data.get("ball_positioning_x", 0),
            frame_data.get("ball_positioning_y", 0),
            frame_data.get("ball_positioning_z", 0),
        )
        ball_vel = Vector3(
            frame_data.get("ball_positioning_linear_velocity_x", 0),
            frame_data.get("ball_positioning_linear_velocity_y", 0),
            frame_data.get("ball_positioning_linear_velocity_z", 0),
        )

        players: Dict[str, Tuple[Vector3, Vector3, Quaternion]] = {}
        # Find all unique player names
        for player in frame.game.players.values():
            name = player.name
            pos = Vector3(
                frame_data.get(f"{name}_positioning_x", 0),
                frame_data.get(f"{name}_positioning_y", 0),
                frame_data.get(f"{name}_positioning_z", 0),
            )
            vel = Vector3(
                frame_data.get(f"{name}_positioning_linear_velocity_x", 0),
                frame_data.get(f"{name}_positioning_linear_velocity_y", 0),
                frame_data.get(f"{name}_positioning_linear_velocity_z", 0),
            )
            rot = Quaternion(
                frame_data.get(f"{name}_positioning_rotation_x", 0),
                frame_data.get(f"{name}_positioning_rotation_y", 0),
                frame_data.get(f"{name}_positioning_rotation_z", 0),
                frame_data.get(f"{name}_positioning_rotation_w", 1),
            )
            players[name] = (pos, vel, rot)

        return ball_pos, ball_vel, players

    def check_environment_collision(
        self, prev_ball_pos: Vector3, curr_ball_pos: Vector3, ball_vel_change: Vector3
    ) -> Optional[CollisionInfo]:
        """
        Check if the ball collided with ground, ceiling, or walls.
        """
        bounds = self.arena.get_boundaries()
        collision_info = None

        # Check ground collision
        if curr_ball_pos.z <= (self.BALL_RADIUS + self.ENVIRONMENT_COLLISION_THRESHOLD):
            if ball_vel_change.z > 0:  # Ball bouncing up
                collision_info = CollisionInfo(
                    collision_type=CollisionType.GROUND,
                    environment_surface="ground",
                    wall_normal=Vector3(0, 0, 1),
                    confidence="high",
                )

        # Check ceiling collision
        elif curr_ball_pos.z >= (
            bounds["z"][1] - self.BALL_RADIUS - self.ENVIRONMENT_COLLISION_THRESHOLD
        ):
            if ball_vel_change.z < 0:  # Ball bouncing down
                collision_info = CollisionInfo(
                    collision_type=CollisionType.CEILING,
                    environment_surface="ceiling",
                    wall_normal=Vector3(0, 0, -1),
                    confidence="high",
                )

        # Check side walls (y-axis)
        if abs(curr_ball_pos.y) >= (
            bounds["y"][1] - self.BALL_RADIUS - self.ENVIRONMENT_COLLISION_THRESHOLD
        ):
            side = 1 if curr_ball_pos.y > 0 else -1
            # Check if in goal area
            in_goal_x = abs(curr_ball_pos.x) > bounds["x"][1] - self.arena.GOAL_DEPTH
            in_goal_z = curr_ball_pos.z < self.arena.GOAL_HEIGHT

            if in_goal_x and in_goal_z:
                collision_info = CollisionInfo(
                    collision_type=CollisionType.GOAL_WALL,
                    environment_surface="goal_wall",
                    wall_normal=Vector3(0, -side, 0),
                    confidence="high",
                )
            else:
                collision_info = CollisionInfo(
                    collision_type=CollisionType.WALL,
                    environment_surface="side_wall",
                    wall_normal=Vector3(0, -side, 0),
                    confidence="medium",
                )

        # Check back walls (x-axis)
        if abs(curr_ball_pos.x) >= (
            bounds["x"][1] - self.BALL_RADIUS - self.ENVIRONMENT_COLLISION_THRESHOLD
        ):
            side = 1 if curr_ball_pos.x > 0 else -1
            # Check if in goal
            in_goal_y = abs(curr_ball_pos.y) < self.arena.GOAL_WIDTH / 2
            in_goal_z = curr_ball_pos.z < self.arena.GOAL_HEIGHT

            if in_goal_y and in_goal_z:
                collision_info = CollisionInfo(
                    collision_type=CollisionType.GOAL_WALL,
                    environment_surface="back_wall_goal",
                    wall_normal=Vector3(-side, 0, 0),
                    confidence="high",
                )
            else:
                collision_info = CollisionInfo(
                    collision_type=CollisionType.WALL,
                    environment_surface="back_wall",
                    wall_normal=Vector3(-side, 0, 0),
                    confidence="medium",
                )

        # Check corners (simplified - actual corners are rounded)
        corner_threshold = 100  # Distance to corner
        if (
            abs(curr_ball_pos.x) > bounds["x"][1] - corner_threshold
            and abs(curr_ball_pos.y) > bounds["y"][1] - corner_threshold
        ):
            x_side = 1 if curr_ball_pos.x > 0 else -1
            y_side = 1 if curr_ball_pos.y > 0 else -1
            collision_info = CollisionInfo(
                collision_type=CollisionType.CORNER_WALL,
                environment_surface="corner",
                wall_normal=Vector3(-x_side, -y_side, 0).normalized(),
                confidence="low",
            )

        return collision_info

    def calculate_ball_impact_point(
        self, ball_pos: Vector3, collision_normal: Vector3
    ) -> Vector3:
        """
        Calculate the point on the ball surface where the collision occurred.
        The impact point is on the ball surface in the direction opposite to the collision normal.
        """
        return ball_pos - (collision_normal * self.BALL_RADIUS)

    def calculate_car_impact_point(
        self,
        car_pos: Vector3,
        car_rot: Quaternion,
        ball_pos: Vector3,
        hitbox: CarHitbox,
    ) -> Tuple[Vector3, str]:
        """
        Calculate the point on the car hitbox where the collision occurred and which surface.

        Returns:
            (impact_point, surface_name)
        """
        # Get car orientation vectors
        forward = car_rot.get_forward_vector()
        right = car_rot.get_right_vector()
        up = car_rot.get_up_vector()

        # Get hitbox center in world space
        hitbox_center = car_pos + car_rot.rotate_vector(
            hitbox.offset or Vector3(0, 0, 0)
        )

        # Vector from hitbox center to ball
        to_ball = ball_pos - hitbox_center

        # Project onto car's local axes
        local_x = to_ball.dot(forward)  # forward/back
        local_y = to_ball.dot(right)  # left/right
        local_z = to_ball.dot(up)  # up/down

        # Clamp to hitbox dimensions
        half_length = hitbox.length / 2
        half_width = hitbox.width / 2
        half_height = hitbox.height / 2

        clamped_x = max(-half_length, min(half_length, local_x))
        clamped_y = max(-half_width, min(half_width, local_y))
        clamped_z = max(-half_height, min(half_height, local_z))

        # Determine which surface (which coordinate hit the boundary)
        surfaces: List[str] = []
        if abs(clamped_x - local_x) < 0.01:
            surfaces.append("front" if local_x > 0 else "back")
        if abs(clamped_y - local_y) < 0.01:
            surfaces.append("right" if local_y > 0 else "left")
        if abs(clamped_z - local_z) < 0.01:
            surfaces.append("roof" if local_z > 0 else "bottom")

        surface_name = "_".join(surfaces) if surfaces else "center"

        # Convert back to world space
        local_impact = (forward * clamped_x) + (right * clamped_y) + (up * clamped_z)
        impact_point = hitbox_center + local_impact

        return impact_point, surface_name

    def detect_collision(
        self,
        prev_frame: Frame,
        curr_frame: Frame,
        frame_number: Optional[int] = None,
        player_hitboxes: Optional[Dict[str, str]] = None,
    ) -> Optional[CollisionInfo]:
        """
        Detect if a collision occurred between two frames and identify the type and details.

        Args:
            prev_frame: Previous frame data
            curr_frame: Current frame data
            frame_number: Optional frame number for tracking
            player_hitboxes: Optional dict mapping player names to hitbox types

        Returns:
            CollisionInfo if collision detected, None otherwise
        """
        # Parse frames
        prev_ball_pos, prev_ball_vel, prev_players = self.parse_frame(prev_frame)
        curr_ball_pos, curr_ball_vel, curr_players = self.parse_frame(curr_frame)

        # Calculate ball velocity change
        ball_vel_change = curr_ball_vel - prev_ball_vel
        vel_change_magnitude = ball_vel_change.magnitude()

        # Check if ball velocity changed significantly
        if vel_change_magnitude < self.MIN_BALL_VELOCITY_CHANGE:
            return None

        # First check for environment collisions
        env_collision = self.check_environment_collision(
            prev_ball_pos, curr_ball_pos, ball_vel_change
        )

        # Calculate common collision properties
        impulse_vec = ball_vel_change * self.ball_mass
        impulse_mag = impulse_vec.magnitude()
        collision_normal = ball_vel_change.normalized()
        ball_impact_point = self.calculate_ball_impact_point(
            curr_ball_pos, collision_normal
        )

        # If strong environment collision detected, return it (unless a player is very close)
        if env_collision and env_collision.confidence == "high":
            env_collision.ball_position = curr_ball_pos
            env_collision.ball_velocity_before = prev_ball_vel
            env_collision.ball_velocity_after = curr_ball_vel
            env_collision.ball_velocity_change = ball_vel_change
            env_collision.impulse_magnitude = impulse_mag
            env_collision.impulse_vector = impulse_vec
            env_collision.collision_normal = collision_normal
            env_collision.ball_impact_point = ball_impact_point
            env_collision.frame_number = frame_number

            # Check if any player is suspiciously close (might be player collision misidentified)
            min_player_dist = float("inf")
            for name, (curr_pos, _, _) in curr_players.items():
                dist = curr_pos.distance_to(curr_ball_pos)
                min_player_dist = min(min_player_dist, dist)

            if min_player_dist > 150:  # No player close enough
                return env_collision

        # Look for player collisions
        class Candidate(TypedDict):
            name: str
            score: float | int
            curr_pos: Vector3
            curr_rot: Quaternion
            dist_to_ball: float
            moved: bool
            car_impact_point: Vector3
            car_surface: str
            hitbox: CarHitbox

        candidates: List[Candidate] = []

        for name, (curr_pos, _, curr_rot) in curr_players.items():
            # Get previous position
            if name not in prev_players:
                continue
            prev_pos, prev_vel, _ = prev_players[name]

            # Get hitbox for this player
            hitbox_type = (
                player_hitboxes.get(name, "octane") if player_hitboxes else "octane"
            )
            hitbox = self.HITBOX_PRESETS.get(hitbox_type, self.default_hitbox)

            # Calculate distances
            dist_to_ball_curr = curr_pos.distance_to(curr_ball_pos)
            dist_to_ball_prev = prev_pos.distance_to(prev_ball_pos)

            # Check if player moved
            player_moved = (curr_pos - prev_pos).magnitude() > 1.0

            # Calculate how much closer they got to the ball
            approach_distance = dist_to_ball_prev - dist_to_ball_curr

            # Calculate car impact point and surface
            car_impact_point, car_surface = self.calculate_car_impact_point(
                curr_pos, curr_rot, curr_ball_pos, hitbox
            )
            actual_impact_distance = car_impact_point.distance_to(curr_ball_pos)

            # Score this candidate
            score = 0
            if actual_impact_distance < self.BALL_RADIUS + 50:  # Very close to touching
                score += 200
            elif dist_to_ball_curr < self.MAX_COLLISION_DISTANCE:
                score += 100

            if player_moved:
                score += 50

            if approach_distance > 0:
                score += approach_distance

            # Bonus: check if player velocity aligns with ball velocity change
            if prev_vel.magnitude() > 0:
                vel_alignment = prev_vel.normalized().dot(ball_vel_change.normalized())
                if vel_alignment > 0:
                    score += vel_alignment * 50

            # Penalty if environment collision is likely
            if env_collision and env_collision.confidence == "high":
                score *= 0.3  # Reduce score significantly

            candidate: Candidate = {
                "name": name,
                "score": score,
                "curr_pos": curr_pos,
                "curr_rot": curr_rot,
                "dist_to_ball": dist_to_ball_curr,
                "moved": player_moved,
                "car_impact_point": car_impact_point,
                "car_surface": car_surface,
                "hitbox": hitbox,
            }
            candidates.append(candidate)

        if not candidates:
            return env_collision if env_collision else None

        # Sort by score
        candidates.sort(key=lambda x: x["score"], reverse=True)
        best = candidates[0]

        # If best player score is too low, prefer environment collision
        if best["score"] < 50 and env_collision:
            return env_collision

        # Determine confidence
        if best["dist_to_ball"] < 150 and best["moved"]:
            confidence = "high"
        elif best["dist_to_ball"] < self.MAX_COLLISION_DISTANCE:
            confidence = "medium"
        else:
            confidence = "low"

        return CollisionInfo(
            collision_type=CollisionType.PLAYER,
            frame_number=frame_number,
            player_name=best["name"],
            player_position=best["curr_pos"],
            player_rotation=best["curr_rot"],
            car_impact_point=best["car_impact_point"],
            car_impact_surface=best["car_surface"],
            distance_to_ball=best["dist_to_ball"],
            ball_position=curr_ball_pos,
            ball_velocity_before=prev_ball_vel,
            ball_velocity_after=curr_ball_vel,
            ball_velocity_change=ball_vel_change,
            ball_impact_point=ball_impact_point,
            impulse_magnitude=impulse_mag,
            impulse_vector=impulse_vec,
            collision_normal=collision_normal,
            confidence=confidence,
        )

    def get_collision_report(self, collision: CollisionInfo) -> str:
        """Generate a human-readable report of the collision."""
        lines = [
            f"Collision Detected - Type: {collision.collision_type.value.upper()}",
            "=" * 60,
        ]

        if collision.frame_number is not None:
            lines.append(f"Frame: {collision.frame_number}")

        lines.append(f"Confidence: {collision.confidence.upper()}")
        lines.append("")

        if collision.is_player_collision():
            lines.extend(
                [
                    f"Player: {collision.player_name}",
                    f"Player Position: {collision.player_position}",
                    f"Distance to ball: {collision.distance_to_ball:.1f} uu",
                    f"Impact Surface: {collision.car_impact_surface}",
                    f"Car Impact Point: {collision.car_impact_point}",
                    "",
                ]
            )
        elif collision.is_environment_collision():
            lines.extend(
                [
                    f"Surface: {collision.environment_surface}",
                    f"Wall Normal: {collision.wall_normal}",
                    "",
                ]
            )

        lines.extend(
            [
                "Ball Impact:",
                f"  Position: {collision.ball_position}",
                f"  Impact Point on Ball: {collision.ball_impact_point}",
                f"  Velocity Before: {collision.ball_velocity_before}",
                f"  Velocity After: {collision.ball_velocity_after}",
                f"  Velocity Change: {collision.ball_velocity_change}",
                f"  Speed Change: {(collision.ball_velocity_change.magnitude() if collision.ball_velocity_change else 0):.1f} uu/s",
                "",
                f"Collision Normal: {collision.collision_normal}",
                "",
                "Impulse:",
                f"  Magnitude: {collision.impulse_magnitude:.1f} mass·uu/s",
                f"  Vector: {collision.impulse_vector}",
            ]
        )

        return "\n".join(lines)
