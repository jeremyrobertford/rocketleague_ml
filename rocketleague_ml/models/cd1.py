# pyright: reportUnusedVariable=false
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from rocketleague_ml.core.frame import Frame


class CollisionType(Enum):
    BALL_PLAYER = "ball_player"
    BALL_ENVIRONMENT = "ball_environment"
    PLAYER_PLAYER = "player_player"
    PLAYER_ENVIRONMENT = "player_environment"


class EnvironmentSurface(Enum):
    GROUND = "ground"
    CEILING = "ceiling"
    SIDE_WALL = "side_wall"
    BACK_WALL = "back_wall"
    CORNER = "corner"
    GOAL_WALL = "goal_wall"


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
        return self.rotate_vector(Vector3(1, 0, 0))

    def get_up_vector(self) -> Vector3:
        return self.rotate_vector(Vector3(0, 0, 1))

    def get_right_vector(self) -> Vector3:
        return self.rotate_vector(Vector3(0, 1, 0))


@dataclass
class CarHitbox:
    length: float
    width: float
    height: float
    offset: Vector3 | None = None

    def __post_init__(self):
        if self.offset is None:
            self.offset = Vector3(0, 0, 0)


@dataclass
class BallPlayerCollision:
    """Collision between ball and a player."""

    collision_type: CollisionType = CollisionType.BALL_PLAYER
    player_name: str = ""

    # Ball data
    ball_position: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    ball_velocity_before: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    ball_velocity_after: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    ball_velocity_change: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    ball_impact_point: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))

    # Player data
    player_position: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    player_rotation: Optional[Quaternion] = None
    player_velocity: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    car_impact_point: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    car_impact_surface: str = ""

    # Physics
    impulse_magnitude: float = 0.0
    impulse_vector: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    collision_normal: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))

    confidence: float = 0.0  # 0.0 to 1.0


@dataclass
class BallEnvironmentCollision:
    """Collision between ball and environment (walls, ground, ceiling)."""

    collision_type: CollisionType = CollisionType.BALL_ENVIRONMENT
    surface: EnvironmentSurface = EnvironmentSurface.GROUND

    # Ball data
    ball_position: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    ball_velocity_before: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    ball_velocity_after: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    ball_velocity_change: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    ball_impact_point: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))

    # Environment data
    wall_normal: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    impact_location: Vector3 = field(
        default_factory=lambda: Vector3(0, 0, 0)
    )  # Where on the wall

    # Physics
    impulse_magnitude: float = 0.0
    impulse_vector: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))

    confidence: float = 0.0


@dataclass
class PlayerPlayerCollision:
    """Collision between two players."""

    collision_type: CollisionType = CollisionType.PLAYER_PLAYER
    player1_name: str = ""
    player2_name: str = ""

    # Player 1 data
    player1_position: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    player1_velocity_before: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    player1_velocity_after: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    player1_impact_point: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))

    # Player 2 data
    player2_position: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    player2_velocity_before: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    player2_velocity_after: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    player2_impact_point: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))

    # Physics
    collision_normal: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    relative_velocity: float = 0.0

    confidence: float = 0.0


@dataclass
class PlayerEnvironmentCollision:
    """Collision between player and environment."""

    collision_type: CollisionType = CollisionType.PLAYER_ENVIRONMENT
    player_name: str = ""
    surface: EnvironmentSurface = EnvironmentSurface.GROUND

    # Player data
    player_position: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    player_velocity_before: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    player_velocity_after: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    player_impact_surface: str = ""  # Which part of car hit

    # Environment data
    wall_normal: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    impact_location: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))

    confidence: float = 0.0


CollisionEvent = (
    BallPlayerCollision
    | BallEnvironmentCollision
    | PlayerPlayerCollision
    | PlayerEnvironmentCollision
)


@dataclass
class FrameCollisions:
    """All collisions detected in a single frame."""

    frame_number: int
    ball_player_collisions: List[BallPlayerCollision] = field(default_factory=list)  # type: ignore
    ball_environment_collisions: List[BallEnvironmentCollision] = field(  # type: ignore
        default_factory=list
    )
    player_player_collisions: List[PlayerPlayerCollision] = field(default_factory=list)  # type: ignore
    player_environment_collisions: List[PlayerEnvironmentCollision] = field(  # type: ignore
        default_factory=list
    )

    def get_all_collisions(self) -> List[CollisionEvent]:
        """Get all collisions as a flat list."""
        return (
            self.ball_player_collisions
            + self.ball_environment_collisions
            + self.player_player_collisions
            + self.player_environment_collisions
        )

    def has_collisions(self) -> bool:
        """Check if any collisions occurred."""
        return len(self.get_all_collisions()) > 0

    def get_ball_collisions(
        self,
    ) -> List[BallPlayerCollision | BallEnvironmentCollision]:
        """Get all collisions involving the ball."""
        return self.ball_player_collisions + self.ball_environment_collisions


class RocketLeagueArena:
    """Standard Rocket League arena dimensions (Soccar)."""

    FIELD_LENGTH = 10240
    FIELD_WIDTH = 8192
    FIELD_HEIGHT = 2044
    GOAL_WIDTH = 1786
    GOAL_HEIGHT = 642
    GOAL_DEPTH = 880
    CORNER_RADIUS = 1152

    @classmethod
    def get_boundaries(cls) -> Dict[str, Tuple[float, float]]:
        return {
            "x": (-cls.FIELD_LENGTH / 2, cls.FIELD_LENGTH / 2),
            "y": (-cls.FIELD_WIDTH / 2, cls.FIELD_WIDTH / 2),
            "z": (0, cls.FIELD_HEIGHT),
        }


class RocketLeagueCollisionDetector:
    BALL_MASS = 30.0
    BALL_RADIUS = 91.25

    HITBOX_PRESETS = {
        "octane": CarHitbox(118.007, 84.1995, 36.1591, Vector3(13.88, 0, 20.75)),
        "dominus": CarHitbox(127.927, 83.2800, 31.3000, Vector3(9.00, 0, 15.75)),
        "plank": CarHitbox(128.82, 84.67, 29.39, Vector3(3.50, 0, 18.26)),
        "breakout": CarHitbox(131.49, 80.52, 30.30, Vector3(1.42, 0, 16.50)),
        "hybrid": CarHitbox(127.02, 82.19, 34.16, Vector3(0.97, 0, 18.29)),
        "merc": CarHitbox(121.41, 83.91, 41.66, Vector3(-2.30, 0, 28.30)),
    }

    # Detection thresholds
    BALL_PLAYER_MAX_DISTANCE = 200.0
    BALL_VELOCITY_CHANGE_THRESHOLD = 50.0
    PLAYER_VELOCITY_CHANGE_THRESHOLD = 100.0
    ENVIRONMENT_THRESHOLD = 25.0
    PLAYER_PLAYER_MAX_DISTANCE = 150.0

    def __init__(self, ball_mass: float = BALL_MASS, default_hitbox: str = "octane"):
        self.ball_mass = ball_mass
        self.default_hitbox = self.HITBOX_PRESETS.get(
            default_hitbox, self.HITBOX_PRESETS["octane"]
        )
        self.arena = RocketLeagueArena()

    def parse_frame(self, frame: Frame) -> Tuple[
        Tuple[Vector3, Vector3],  # Ball (pos, vel)
        Dict[str, Tuple[Vector3, Vector3, Quaternion]],  # Players
    ]:
        """Parse frame into structured data."""
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

        return (ball_pos, ball_vel), players

    def calculate_car_impact_point(
        self,
        car_pos: Vector3,
        car_rot: Quaternion,
        ball_pos: Vector3,
        hitbox: CarHitbox,
    ) -> Tuple[Vector3, str]:
        """Calculate impact point and surface on car."""
        forward = car_rot.get_forward_vector()
        right = car_rot.get_right_vector()
        up = car_rot.get_up_vector()

        hitbox_center = car_pos + car_rot.rotate_vector(
            hitbox.offset or Vector3(0, 0, 0)
        )
        to_ball = ball_pos - hitbox_center

        local_x = to_ball.dot(forward)
        local_y = to_ball.dot(right)
        local_z = to_ball.dot(up)

        half_length = hitbox.length / 2
        half_width = hitbox.width / 2
        half_height = hitbox.height / 2

        clamped_x = max(-half_length, min(half_length, local_x))
        clamped_y = max(-half_width, min(half_width, local_y))
        clamped_z = max(-half_height, min(half_height, local_z))

        surfaces: List[str] = []
        if abs(clamped_x - local_x) < 0.01:
            surfaces.append("front" if local_x > 0 else "back")
        if abs(clamped_y - local_y) < 0.01:
            surfaces.append("right" if local_y > 0 else "left")
        if abs(clamped_z - local_z) < 0.01:
            surfaces.append("roof" if local_z > 0 else "bottom")

        surface_name = "_".join(surfaces) if surfaces else "center"
        local_impact = (forward * clamped_x) + (right * clamped_y) + (up * clamped_z)
        impact_point = hitbox_center + local_impact

        return impact_point, surface_name

    def detect_ball_player_collisions(
        self,
        prev_ball: Tuple[Vector3, Vector3],
        curr_ball: Tuple[Vector3, Vector3],
        prev_players: Dict[str, Tuple[Vector3, Vector3, Quaternion]],
        curr_players: Dict[str, Tuple[Vector3, Vector3, Quaternion]],
        player_hitboxes: Optional[Dict[str, str]] = None,
    ) -> List[BallPlayerCollision]:
        """Detect all ball-player collisions in this frame transition."""
        prev_ball_pos, prev_ball_vel = prev_ball
        curr_ball_pos, curr_ball_vel = curr_ball

        ball_vel_change = curr_ball_vel - prev_ball_vel

        if ball_vel_change.magnitude() < self.BALL_VELOCITY_CHANGE_THRESHOLD:
            return []

        collisions: List[BallPlayerCollision] = []
        collision_candidates: List[Tuple[str, float, BallPlayerCollision]] = []

        for name, (curr_pos, curr_vel, curr_rot) in curr_players.items():
            if name not in prev_players:
                continue

            prev_pos, prev_vel, prev_rot = prev_players[name]

            hitbox_type = (
                player_hitboxes.get(name, "octane") if player_hitboxes else "octane"
            )
            hitbox = self.HITBOX_PRESETS.get(hitbox_type, self.default_hitbox)

            car_impact_point, car_surface = self.calculate_car_impact_point(
                curr_pos, curr_rot, curr_ball_pos, hitbox
            )

            impact_distance = car_impact_point.distance_to(curr_ball_pos)

            # Score based on proximity and physics
            if impact_distance > self.BALL_PLAYER_MAX_DISTANCE:
                continue

            confidence = 0.0
            if impact_distance < self.BALL_RADIUS + 20:
                confidence = 1.0
            elif impact_distance < self.BALL_RADIUS + 50:
                confidence = 0.8
            elif impact_distance < 150:
                confidence = 0.5
            else:
                confidence = 0.2

            # Velocity alignment bonus
            if prev_vel.magnitude() > 0:
                vel_align = prev_vel.normalized().dot(ball_vel_change.normalized())
                if vel_align > 0:
                    confidence = min(1.0, confidence + vel_align * 0.2)

            collision_normal = ball_vel_change.normalized()
            ball_impact_point = curr_ball_pos - (collision_normal * self.BALL_RADIUS)
            impulse_vec = ball_vel_change * self.ball_mass

            collision = BallPlayerCollision(
                player_name=name,
                ball_position=curr_ball_pos,
                ball_velocity_before=prev_ball_vel,
                ball_velocity_after=curr_ball_vel,
                ball_velocity_change=ball_vel_change,
                ball_impact_point=ball_impact_point,
                player_position=curr_pos,
                player_rotation=curr_rot,
                player_velocity=curr_vel,
                car_impact_point=car_impact_point,
                car_impact_surface=car_surface,
                impulse_magnitude=impulse_vec.magnitude(),
                impulse_vector=impulse_vec,
                collision_normal=collision_normal,
                confidence=confidence,
            )

            collision_candidates.append((name, confidence, collision))

        # Sort by confidence and take top candidates
        collision_candidates.sort(key=lambda x: x[1], reverse=True)

        # In a pinch, multiple players can hit simultaneously
        # Take all with high confidence or top 2 with medium+ confidence
        for name, conf, collision in collision_candidates:
            if conf > 0.7:  # High confidence
                collisions.append(collision)
            elif conf > 0.4 and len(collisions) < 2:  # Possible pinch
                collisions.append(collision)

        return collisions

    def is_ball_rolling_on_surface(
        self,
        ball_pos: Vector3,
        ball_vel: Vector3,
        surface_normal: Vector3,
        surface_height: float,
    ) -> bool:
        """
        Check if ball is rolling on a surface vs bouncing off it.

        Rolling = ball is on surface and velocity is mostly parallel to it.
        Bouncing = ball velocity has significant perpendicular component.
        """
        # Check if ball is resting on surface
        distance_to_surface = abs(ball_pos.dot(surface_normal) - surface_height)
        if distance_to_surface > self.BALL_RADIUS + 10:
            return False  # Not on surface

        # Check velocity direction relative to surface
        normal_velocity = abs(ball_vel.dot(surface_normal))
        total_velocity = ball_vel.magnitude()

        if total_velocity < 10:
            return True  # Nearly stationary on surface = rolling

        # If most velocity is perpendicular to surface = bouncing
        # If most velocity is parallel to surface = rolling
        perpendicular_ratio = normal_velocity / total_velocity

        return perpendicular_ratio < 0.3  # Less than 30% perpendicular = rolling

    def detect_ball_environment_collisions(
        self,
        prev_ball: Tuple[Vector3, Vector3],
        curr_ball: Tuple[Vector3, Vector3],
    ) -> List[BallEnvironmentCollision]:
        """Detect ball collisions with walls, ground, ceiling (bounces only, not rolling)."""
        prev_ball_pos, prev_ball_vel = prev_ball
        curr_ball_pos, curr_ball_vel = curr_ball

        ball_vel_change = curr_ball_vel - prev_ball_vel

        if ball_vel_change.magnitude() < self.BALL_VELOCITY_CHANGE_THRESHOLD:
            return []

        collisions: List[BallEnvironmentCollision] = []
        bounds = self.arena.get_boundaries()

        # Ground
        if curr_ball_pos.z <= (self.BALL_RADIUS + self.ENVIRONMENT_THRESHOLD):
            ground_normal = Vector3(0, 0, 1)

            # Skip if ball is just rolling on ground
            if self.is_ball_rolling_on_surface(
                curr_ball_pos, curr_ball_vel, ground_normal, 0
            ):
                return collisions  # Ball rolling, not bouncing

            # Check for actual bounce (velocity reversed in z direction)
            if ball_vel_change.z > 0 and prev_ball_vel.z < 0:  # Was falling, now rising
                impulse_vec = ball_vel_change * self.ball_mass
                collisions.append(
                    BallEnvironmentCollision(
                        surface=EnvironmentSurface.GROUND,
                        ball_position=curr_ball_pos,
                        ball_velocity_before=prev_ball_vel,
                        ball_velocity_after=curr_ball_vel,
                        ball_velocity_change=ball_vel_change,
                        ball_impact_point=Vector3(curr_ball_pos.x, curr_ball_pos.y, 0),
                        wall_normal=ground_normal,
                        impact_location=Vector3(curr_ball_pos.x, curr_ball_pos.y, 0),
                        impulse_magnitude=impulse_vec.magnitude(),
                        impulse_vector=impulse_vec,
                        confidence=1.0,
                    )
                )

        # Ceiling
        if curr_ball_pos.z >= (
            bounds["z"][1] - self.BALL_RADIUS - self.ENVIRONMENT_THRESHOLD
        ):
            ceiling_normal = Vector3(0, 0, -1)
            ceiling_height = bounds["z"][1]

            # Skip if ball is rolling on ceiling
            if self.is_ball_rolling_on_surface(
                curr_ball_pos, curr_ball_vel, ceiling_normal, ceiling_height
            ):
                return collisions

            if ball_vel_change.z < 0 and prev_ball_vel.z > 0:  # Was rising, now falling
                impulse_vec = ball_vel_change * self.ball_mass
                collisions.append(
                    BallEnvironmentCollision(
                        surface=EnvironmentSurface.CEILING,
                        ball_position=curr_ball_pos,
                        ball_velocity_before=prev_ball_vel,
                        ball_velocity_after=curr_ball_vel,
                        ball_velocity_change=ball_vel_change,
                        ball_impact_point=Vector3(
                            curr_ball_pos.x, curr_ball_pos.y, ceiling_height
                        ),
                        wall_normal=ceiling_normal,
                        impact_location=Vector3(
                            curr_ball_pos.x, curr_ball_pos.y, ceiling_height
                        ),
                        impulse_magnitude=impulse_vec.magnitude(),
                        impulse_vector=impulse_vec,
                        confidence=1.0,
                    )
                )

        # Side walls
        if abs(curr_ball_pos.y) >= (
            bounds["y"][1] - self.BALL_RADIUS - self.ENVIRONMENT_THRESHOLD
        ):
            side = 1 if curr_ball_pos.y > 0 else -1
            wall_normal = Vector3(0, -side, 0)
            wall_height = side * bounds["y"][1]

            # Skip if ball is rolling along wall
            if self.is_ball_rolling_on_surface(
                Vector3(curr_ball_pos.x, curr_ball_pos.y, curr_ball_pos.z),
                curr_ball_vel,
                wall_normal,
                wall_height,
            ):
                return collisions

            # Check for bounce (velocity reversed in y direction)
            if (ball_vel_change.y * -side) > 0 and (prev_ball_vel.y * side) > 0:
                in_goal_x = (
                    abs(curr_ball_pos.x) > bounds["x"][1] - self.arena.GOAL_DEPTH
                )
                in_goal_z = curr_ball_pos.z < self.arena.GOAL_HEIGHT

                surface = (
                    EnvironmentSurface.GOAL_WALL
                    if (in_goal_x and in_goal_z)
                    else EnvironmentSurface.SIDE_WALL
                )
                impulse_vec = ball_vel_change * self.ball_mass

                collisions.append(
                    BallEnvironmentCollision(
                        surface=surface,
                        ball_position=curr_ball_pos,
                        ball_velocity_before=prev_ball_vel,
                        ball_velocity_after=curr_ball_vel,
                        ball_velocity_change=ball_vel_change,
                        ball_impact_point=Vector3(
                            curr_ball_pos.x, wall_height, curr_ball_pos.z
                        ),
                        wall_normal=wall_normal,
                        impact_location=Vector3(
                            curr_ball_pos.x, wall_height, curr_ball_pos.z
                        ),
                        impulse_magnitude=impulse_vec.magnitude(),
                        impulse_vector=impulse_vec,
                        confidence=0.9,
                    )
                )

        # Back walls
        if abs(curr_ball_pos.x) >= (
            bounds["x"][1] - self.BALL_RADIUS - self.ENVIRONMENT_THRESHOLD
        ):
            side = 1 if curr_ball_pos.x > 0 else -1
            wall_normal = Vector3(-side, 0, 0)
            wall_height = side * bounds["x"][1]

            # Skip if ball is rolling along wall
            if self.is_ball_rolling_on_surface(
                Vector3(curr_ball_pos.x, curr_ball_pos.y, curr_ball_pos.z),
                curr_ball_vel,
                wall_normal,
                wall_height,
            ):
                return collisions

            # Check for bounce (velocity reversed in x direction)
            if (ball_vel_change.x * -side) > 0 and (prev_ball_vel.x * side) > 0:
                in_goal_y = abs(curr_ball_pos.y) < self.arena.GOAL_WIDTH / 2
                in_goal_z = curr_ball_pos.z < self.arena.GOAL_HEIGHT

                surface = (
                    EnvironmentSurface.GOAL_WALL
                    if (in_goal_y and in_goal_z)
                    else EnvironmentSurface.BACK_WALL
                )
                impulse_vec = ball_vel_change * self.ball_mass

                collisions.append(
                    BallEnvironmentCollision(
                        surface=surface,
                        ball_position=curr_ball_pos,
                        ball_velocity_before=prev_ball_vel,
                        ball_velocity_after=curr_ball_vel,
                        ball_velocity_change=ball_vel_change,
                        ball_impact_point=Vector3(
                            wall_height, curr_ball_pos.y, curr_ball_pos.z
                        ),
                        wall_normal=wall_normal,
                        impact_location=Vector3(
                            wall_height, curr_ball_pos.y, curr_ball_pos.z
                        ),
                        impulse_magnitude=impulse_vec.magnitude(),
                        impulse_vector=impulse_vec,
                        confidence=0.9,
                    )
                )

        return collisions

    def detect_player_player_collisions(
        self,
        prev_players: Dict[str, Tuple[Vector3, Vector3, Quaternion]],
        curr_players: Dict[str, Tuple[Vector3, Vector3, Quaternion]],
    ) -> List[PlayerPlayerCollision]:
        """Detect collisions between players."""
        collisions: List[PlayerPlayerCollision] = []
        player_names = list(curr_players.keys())

        for i, name1 in enumerate(player_names):
            for name2 in player_names[i + 1 :]:
                if name1 not in prev_players or name2 not in prev_players:
                    continue

                curr_pos1, curr_vel1, _ = curr_players[name1]
                curr_pos2, curr_vel2, _ = curr_players[name2]
                prev_pos1, prev_vel1, _ = prev_players[name1]
                prev_pos2, prev_vel2, _ = prev_players[name2]

                distance = curr_pos1.distance_to(curr_pos2)

                if distance > self.PLAYER_PLAYER_MAX_DISTANCE:
                    continue

                # Check for velocity changes indicating collision
                vel_change1 = curr_vel1 - prev_vel1
                vel_change2 = curr_vel2 - prev_vel2

                if (
                    vel_change1.magnitude() < self.PLAYER_VELOCITY_CHANGE_THRESHOLD
                    and vel_change2.magnitude() < self.PLAYER_VELOCITY_CHANGE_THRESHOLD
                ):
                    continue

                collision_normal = (curr_pos2 - curr_pos1).normalized()
                relative_vel = (prev_vel1 - prev_vel2).magnitude()

                confidence = 0.5
                if distance < 100:
                    confidence = 0.8
                if distance < 75:
                    confidence = 1.0

                collisions.append(
                    PlayerPlayerCollision(
                        player1_name=name1,
                        player2_name=name2,
                        player1_position=curr_pos1,
                        player1_velocity_before=prev_vel1,
                        player1_velocity_after=curr_vel1,
                        player1_impact_point=curr_pos1 + (collision_normal * 50),
                        player2_position=curr_pos2,
                        player2_velocity_before=prev_vel2,
                        player2_velocity_after=curr_vel2,
                        player2_impact_point=curr_pos2 - (collision_normal * 50),
                        collision_normal=collision_normal,
                        relative_velocity=relative_vel,
                        confidence=confidence,
                    )
                )

        return collisions

    def is_player_driving_on_surface(
        self,
        player_rot: Quaternion,
        surface_normal: Vector3,
        player_vel: Vector3,
    ) -> bool:
        """
        Check if a player is driving on a surface (controlled) vs impacting it.

        A player is "driving" if their car's bottom/wheels are aligned with the surface.
        """
        car_up = player_rot.get_up_vector()

        # Check alignment between car's up vector and surface normal
        # If car is oriented with the surface, they're driving on it
        alignment = car_up.dot(surface_normal)

        # Alignment > 0.7 means car is roughly aligned with surface (within ~45 degrees)
        # This indicates controlled driving, not a crash
        return alignment > 0.7

    def is_smooth_transition(
        self,
        prev_vel: Vector3,
        curr_vel: Vector3,
        surface_normal: Vector3,
    ) -> bool:
        """
        Check if velocity change is a smooth transition (driving) vs impact (crash).

        Smooth transitions have gradual velocity changes parallel to the surface.
        Impacts have sudden velocity changes perpendicular to the surface.
        """
        vel_change = curr_vel - prev_vel

        # Project velocity change onto surface normal
        normal_component = abs(vel_change.dot(surface_normal))

        # Project onto surface tangent (perpendicular to normal)
        tangent_component = vel_change.magnitude()

        # If most of the change is perpendicular to surface = impact
        # If most of the change is parallel to surface = smooth driving
        if tangent_component < 0.1:
            return False

        ratio = normal_component / tangent_component

        # High ratio = mostly perpendicular change = impact
        # Low ratio = mostly parallel change = smooth transition
        return ratio < 0.5  # Less than 50% of change is perpendicular

    def detect_player_environment_collisions(
        self,
        prev_players: Dict[str, Tuple[Vector3, Vector3, Quaternion]],
        curr_players: Dict[str, Tuple[Vector3, Vector3, Quaternion]],
    ) -> List[PlayerEnvironmentCollision]:
        """Detect player collisions with environment (actual impacts, not driving transitions)."""
        collisions: List[PlayerEnvironmentCollision] = []
        bounds = self.arena.get_boundaries()

        for name, (curr_pos, curr_vel, curr_rot) in curr_players.items():
            if name not in prev_players:
                continue

            prev_pos, prev_vel, prev_rot = prev_players[name]
            vel_change = curr_vel - prev_vel

            if vel_change.magnitude() < self.PLAYER_VELOCITY_CHANGE_THRESHOLD:
                continue

            # Ground collision - must be falling (negative z velocity) and near ground
            if curr_pos.z <= (20 + self.ENVIRONMENT_THRESHOLD):
                ground_normal = Vector3(0, 0, 1)

                # Check if player is already driving on ground (not an impact)
                if self.is_player_driving_on_surface(curr_rot, ground_normal, curr_vel):
                    continue  # Already driving on ground, not a landing

                # Check if this is a smooth transition (driving from wall to ground)
                if self.is_smooth_transition(prev_vel, curr_vel, ground_normal):
                    continue  # Smooth transition, not an impact

                # Check if player was falling before impact
                if prev_vel.z < -100:  # Was moving downward with significant speed
                    # Check if velocity changed (bounced/landed)
                    if vel_change.z > 50:  # Velocity changed upward (landed/bounced)
                        collisions.append(
                            PlayerEnvironmentCollision(
                                player_name=name,
                                surface=EnvironmentSurface.GROUND,
                                player_position=curr_pos,
                                player_velocity_before=prev_vel,
                                player_velocity_after=curr_vel,
                                player_impact_surface="bottom",
                                wall_normal=ground_normal,
                                impact_location=Vector3(curr_pos.x, curr_pos.y, 0),
                                confidence=0.9,
                            )
                        )

            # Ceiling collision - must be rising (positive z velocity) and near ceiling
            if curr_pos.z >= (bounds["z"][1] - 100):
                ceiling_normal = Vector3(0, 0, -1)

                # Check if driving on ceiling (not an impact)
                if self.is_player_driving_on_surface(
                    curr_rot, ceiling_normal, curr_vel
                ):
                    continue

                if self.is_smooth_transition(prev_vel, curr_vel, ceiling_normal):
                    continue

                if prev_vel.z > 100:  # Was moving upward
                    if vel_change.z < -50:  # Velocity changed downward
                        collisions.append(
                            PlayerEnvironmentCollision(
                                player_name=name,
                                surface=EnvironmentSurface.CEILING,
                                player_position=curr_pos,
                                player_velocity_before=prev_vel,
                                player_velocity_after=curr_vel,
                                player_impact_surface="roof",
                                wall_normal=ceiling_normal,
                                impact_location=Vector3(
                                    curr_pos.x, curr_pos.y, bounds["z"][1]
                                ),
                                confidence=0.9,
                            )
                        )

            # Side walls - must be moving toward the wall
            if abs(curr_pos.y) >= (bounds["y"][1] - 100):
                side = 1 if curr_pos.y > 0 else -1
                wall_normal = Vector3(0, -side, 0)

                # Check if driving on wall (not an impact)
                if self.is_player_driving_on_surface(curr_rot, wall_normal, curr_vel):
                    continue

                if self.is_smooth_transition(prev_vel, curr_vel, wall_normal):
                    continue

                # Check if moving toward the wall (positive y-vel for positive side, negative for negative)
                moving_toward_wall = (
                    prev_vel.y * side
                ) > 200  # Moving toward wall with speed
                velocity_reversed = (
                    vel_change.y * side
                ) < -100  # Velocity reversed/reduced

                if moving_toward_wall and velocity_reversed:
                    collisions.append(
                        PlayerEnvironmentCollision(
                            player_name=name,
                            surface=EnvironmentSurface.SIDE_WALL,
                            player_position=curr_pos,
                            player_velocity_before=prev_vel,
                            player_velocity_after=curr_vel,
                            player_impact_surface="side",
                            wall_normal=wall_normal,
                            impact_location=Vector3(
                                curr_pos.x, side * bounds["y"][1], curr_pos.z
                            ),
                            confidence=0.8,
                        )
                    )

            # Back walls - must be moving toward the wall
            if abs(curr_pos.x) >= (bounds["x"][1] - 100):
                side = 1 if curr_pos.x > 0 else -1
                wall_normal = Vector3(-side, 0, 0)

                # Check if driving on wall (not an impact)
                if self.is_player_driving_on_surface(curr_rot, wall_normal, curr_vel):
                    continue

                if self.is_smooth_transition(prev_vel, curr_vel, wall_normal):
                    continue

                moving_toward_wall = (prev_vel.x * side) > 200
                velocity_reversed = (vel_change.x * side) < -100

                if moving_toward_wall and velocity_reversed:
                    collisions.append(
                        PlayerEnvironmentCollision(
                            player_name=name,
                            surface=EnvironmentSurface.BACK_WALL,
                            player_position=curr_pos,
                            player_velocity_before=prev_vel,
                            player_velocity_after=curr_vel,
                            player_impact_surface="front_or_back",
                            wall_normal=wall_normal,
                            impact_location=Vector3(
                                side * bounds["x"][1], curr_pos.y, curr_pos.z
                            ),
                            confidence=0.8,
                        )
                    )

        return collisions

    def detect_all_collisions(
        self,
        prev_frame: Frame,
        curr_frame: Frame,
        frame_number: int,
        player_hitboxes: Optional[Dict[str, str]] = None,
    ) -> FrameCollisions:
        """
        Detect ALL collisions that occurred between two frames.

        Returns a FrameCollisions object containing all detected collisions.
        """
        prev_ball, prev_players = self.parse_frame(prev_frame)
        curr_ball, curr_players = self.parse_frame(curr_frame)

        frame_collisions = FrameCollisions(frame_number=frame_number)

        # Detect ball-player collisions
        frame_collisions.ball_player_collisions = self.detect_ball_player_collisions(
            prev_ball, curr_ball, prev_players, curr_players, player_hitboxes
        )

        # Detect ball-environment collisions
        frame_collisions.ball_environment_collisions = (
            self.detect_ball_environment_collisions(prev_ball, curr_ball)
        )

        # Detect player-player collisions
        frame_collisions.player_player_collisions = (
            self.detect_player_player_collisions(prev_players, curr_players)
        )

        # Detect player-environment collisions
        frame_collisions.player_environment_collisions = (
            self.detect_player_environment_collisions(prev_players, curr_players)
        )

        return frame_collisions

    def get_collision_summary(self, frame_collisions: FrameCollisions) -> str:
        """Generate a summary report of all collisions in a frame."""
        lines = [
            f"=" * 70,
            f"FRAME {frame_collisions.frame_number} COLLISION SUMMARY",
            f"=" * 70,
            f"Total collisions detected: {len(frame_collisions.get_all_collisions())}",
            "",
        ]

        if frame_collisions.ball_player_collisions:
            lines.append(
                f"Ball-Player Collisions: {len(frame_collisions.ball_player_collisions)}"
            )
            for i, col in enumerate(frame_collisions.ball_player_collisions, 1):
                lines.extend(
                    [
                        f"  [{i}] {col.player_name} (confidence: {col.confidence:.2f})",
                        f"      Surface: {col.car_impact_surface}",
                        f"      Ball Δv: {col.ball_velocity_change.magnitude():.1f} uu/s",
                        f"      Impulse: {col.impulse_magnitude:.1f} mass·uu/s",
                    ]
                )
            lines.append("")

        if frame_collisions.ball_environment_collisions:
            lines.append(
                f"Ball-Environment Collisions: {len(frame_collisions.ball_environment_collisions)}"
            )
            for i, col in enumerate(frame_collisions.ball_environment_collisions, 1):
                lines.extend(
                    [
                        f"  [{i}] {col.surface.value.upper()} (confidence: {col.confidence:.2f})",
                        f"      Ball Δv: {col.ball_velocity_change.magnitude():.1f} uu/s",
                        f"      Normal: {col.wall_normal}",
                    ]
                )
            lines.append("")

        if frame_collisions.player_player_collisions:
            lines.append(
                f"Player-Player Collisions: {len(frame_collisions.player_player_collisions)}"
            )
            for i, col in enumerate(frame_collisions.player_player_collisions, 1):
                lines.extend(
                    [
                        f"  [{i}] {col.player1_name} vs {col.player2_name} (confidence: {col.confidence:.2f})",
                        f"      Relative velocity: {col.relative_velocity:.1f} uu/s",
                    ]
                )
            lines.append("")

        if frame_collisions.player_environment_collisions:
            lines.append(
                f"Player-Environment Collisions: {len(frame_collisions.player_environment_collisions)}"
            )
            for i, col in enumerate(frame_collisions.player_environment_collisions, 1):
                lines.extend(
                    [
                        f"  [{i}] {col.player_name} -> {col.surface.value.upper()} (confidence: {col.confidence:.2f})",
                    ]
                )
            lines.append("")

        return "\n".join(lines)

    def get_detailed_collision_report(self, collision: CollisionEvent) -> str:
        """Generate a detailed report for a single collision."""
        lines = ["=" * 70]

        if isinstance(collision, BallPlayerCollision):
            lines.extend(
                [
                    "BALL-PLAYER COLLISION",
                    "=" * 70,
                    f"Player: {collision.player_name}",
                    f"Confidence: {collision.confidence:.2f}",
                    "",
                    "Ball Impact:",
                    f"  Position: {collision.ball_position}",
                    f"  Impact Point: {collision.ball_impact_point}",
                    f"  Velocity Before: {collision.ball_velocity_before}",
                    f"  Velocity After: {collision.ball_velocity_after}",
                    f"  Velocity Change: {collision.ball_velocity_change}",
                    f"  Speed Change: {collision.ball_velocity_change.magnitude():.1f} uu/s",
                    "",
                    "Car Impact:",
                    f"  Position: {collision.player_position}",
                    f"  Impact Point: {collision.car_impact_point}",
                    f"  Impact Surface: {collision.car_impact_surface}",
                    "",
                    "Physics:",
                    f"  Collision Normal: {collision.collision_normal}",
                    f"  Impulse Magnitude: {collision.impulse_magnitude:.1f} mass·uu/s",
                    f"  Impulse Vector: {collision.impulse_vector}",
                ]
            )

        elif isinstance(collision, BallEnvironmentCollision):
            lines.extend(
                [
                    "BALL-ENVIRONMENT COLLISION",
                    "=" * 70,
                    f"Surface: {collision.surface.value.upper()}",
                    f"Confidence: {collision.confidence:.2f}",
                    "",
                    "Ball Impact:",
                    f"  Position: {collision.ball_position}",
                    f"  Impact Point: {collision.ball_impact_point}",
                    f"  Velocity Before: {collision.ball_velocity_before}",
                    f"  Velocity After: {collision.ball_velocity_after}",
                    f"  Velocity Change: {collision.ball_velocity_change}",
                    f"  Speed Change: {collision.ball_velocity_change.magnitude():.1f} uu/s",
                    "",
                    "Environment:",
                    f"  Wall Normal: {collision.wall_normal}",
                    f"  Impact Location: {collision.impact_location}",
                    f"  Impulse: {collision.impulse_magnitude:.1f} mass·uu/s",
                ]
            )

        elif isinstance(collision, PlayerPlayerCollision):
            lines.extend(
                [
                    "PLAYER-PLAYER COLLISION",
                    "=" * 70,
                    f"Players: {collision.player1_name} vs {collision.player2_name}",
                    f"Confidence: {collision.confidence:.2f}",
                    "",
                    f"Player 1 ({collision.player1_name}):",
                    f"  Position: {collision.player1_position}",
                    f"  Velocity Before: {collision.player1_velocity_before}",
                    f"  Velocity After: {collision.player1_velocity_after}",
                    "",
                    f"Player 2 ({collision.player2_name}):",
                    f"  Position: {collision.player2_position}",
                    f"  Velocity Before: {collision.player2_velocity_before}",
                    f"  Velocity After: {collision.player2_velocity_after}",
                    "",
                    "Physics:",
                    f"  Collision Normal: {collision.collision_normal}",
                    f"  Relative Velocity: {collision.relative_velocity:.1f} uu/s",
                ]
            )

        else:
            lines.extend(
                [
                    "PLAYER-ENVIRONMENT COLLISION",
                    "=" * 70,
                    f"Player: {collision.player_name}",
                    f"Surface: {collision.surface.value.upper()}",
                    f"Confidence: {collision.confidence:.2f}",
                    "",
                    "Player Impact:",
                    f"  Position: {collision.player_position}",
                    f"  Impact Surface: {collision.player_impact_surface}",
                    f"  Velocity Before: {collision.player_velocity_before}",
                    f"  Velocity After: {collision.player_velocity_after}",
                    "",
                    "Environment:",
                    f"  Wall Normal: {collision.wall_normal}",
                    f"  Impact Location: {collision.impact_location}",
                ]
            )

        return "\n".join(lines)
