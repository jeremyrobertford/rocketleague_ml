# type: ignore
# pyright: reportUnusedVariable=false
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# RLUtilities imports
from rlutilities.simulation import Game, Ball, Car, Input
from rlutilities.linear_algebra import vec3, mat3, transpose, norm, dot
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

    @staticmethod
    def from_vec3(v: vec3) -> "Vector3":
        """Convert RLUtilities vec3 to Vector3."""
        return Vector3(v[0], v[1], v[2])

    def to_vec3(self) -> vec3:
        """Convert to RLUtilities vec3."""
        return vec3(self.x, self.y, self.z)

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

    def distance_to(self, other: "Vector3") -> float:
        return (self - other).magnitude()

    def __repr__(self) -> str:
        return f"({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"


@dataclass
class BallPlayerCollision:
    """Collision between ball and a player."""

    collision_type: CollisionType = CollisionType.BALL_PLAYER
    player_name: str = ""
    player_collision_region: str = ""
    frame_number: int = 0
    substep: int = 0  # Which substep in simulation

    # Ball data
    ball_velocity: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    player_velocity: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))

    # Physics
    impulse: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    collision_normal: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    ball_contact_point: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    player_contact_point: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))


@dataclass
class BallEnvironmentCollision:
    """Collision between ball and environment."""

    collision_type: CollisionType = CollisionType.BALL_ENVIRONMENT
    surface: EnvironmentSurface = EnvironmentSurface.GROUND
    frame_number: int = 0
    substep: int = 0

    ball_position: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    ball_velocity_before: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    ball_velocity_after: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    ball_velocity_change: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))

    wall_normal: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    impulse_magnitude: float = 0.0


@dataclass
class PlayerPlayerCollision:
    """Collision between two players."""

    collision_type: CollisionType = CollisionType.PLAYER_PLAYER
    player1_name: str = ""
    player2_name: str = ""
    frame_number: int = 0
    substep: int = 0

    player1_position: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    player1_velocity_before: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    player1_velocity_after: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))

    player2_position: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    player2_velocity_before: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    player2_velocity_after: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))

    collision_normal: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    relative_velocity: float = 0.0


@dataclass
class PlayerEnvironmentCollision:
    """Collision between player and environment."""

    collision_type: CollisionType = CollisionType.PLAYER_ENVIRONMENT
    player_name: str = ""
    surface: EnvironmentSurface = EnvironmentSurface.GROUND
    frame_number: int = 0
    substep: int = 0

    player_position: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    player_velocity_before: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    player_velocity_after: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))

    wall_normal: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))


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
    ball_player_collisions: List[BallPlayerCollision] = field(default_factory=list)
    ball_environment_collisions: List[BallEnvironmentCollision] = field(
        default_factory=list
    )
    player_player_collisions: List[PlayerPlayerCollision] = field(default_factory=list)
    player_environment_collisions: List[PlayerEnvironmentCollision] = field(
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


def mat3_vec3_mul(M: mat3, v: vec3) -> vec3:
    # Use M[(row, col)] indexing
    return vec3(
        M[(0, 0)] * v[0] + M[(0, 1)] * v[1] + M[(0, 2)] * v[2],
        M[(1, 0)] * v[0] + M[(1, 1)] * v[1] + M[(1, 2)] * v[2],
        M[(2, 0)] * v[0] + M[(2, 1)] * v[1] + M[(2, 2)] * v[2],
    )


def get_car_collision_region(car, contact_point: vec3) -> str:
    """
    Map a world-space contact point on the car to a descriptive region string.
    'tire', 'roof', and 'windshield' are used only when contact is clearly
    near those specific surfaces.
    """
    rel = contact_point - car.position
    local = mat3_vec3_mul(transpose(car.orientation), rel - car.hitbox_offset)
    local[0] = -local[0]

    hx, hy, hz = car.hitbox_widths[0], car.hitbox_widths[1], car.hitbox_widths[2]

    # Thresholds (tweakable)
    front_t = 0.35 * hy
    rear_t = -0.35 * hy
    side_t = 0.35 * hx

    # Height zones (relative to center)
    roof_zone = 0.7 * hz  # top 30%
    windshield_zone = 0.3 * hz  # upper-mid band
    floor_zone = -0.7 * hz  # bottom 30%

    # X region
    if local[1] > front_t:
        x_region = "front"
    elif local[1] < rear_t:
        x_region = "rear"
    else:
        x_region = "center"

    # Y region
    if local[0] > side_t:
        y_region = "right"
    elif local[0] < -side_t:
        y_region = "left"
    else:
        y_region = "center"

    # Z-based refinement
    z = local[2]

    # Distinguish by clear height bands
    if z > roof_zone:
        z_region = "roof"
    elif z < floor_zone:
        z_region = "floor"
    else:
        z_region = "mid"

    # --- Combined logic ---
    if z_region == "floor" and x_region in ("front", "rear"):
        region = f"{x_region}-{y_region}-tire"
    elif z_region == "roof" and x_region in ("front", "rear"):
        region = f"{x_region}-{y_region}-roof"
    elif (
        x_region == "front" and y_region == "center" and windshield_zone < z < roof_zone
    ):
        region = "windshield"
    else:
        region = f"{x_region}-{y_region}-core"

    return region


def compute_collision(
    ball: Ball,
    player: Car,
):
    """
    Estimate collision normal and impulse from two consecutive frames of Rocket League data.
    """

    # Approximate contact normal (ball moved toward car)
    collision_normal = ball.position - player.position
    collision_normal /= norm(collision_normal)

    # Relative velocity before collision
    relative_velocity = ball.velocity - player.velocity
    velocity_along_normal = dot(relative_velocity, collision_normal)

    # Impulse magnitude
    j = -(1 + ball.restitution) * velocity_along_normal / (1 / ball.mass + 1 / 180)
    impulse = j * collision_normal

    # Apply impulse to velocities
    ball_velocity = ball.velocity + impulse / ball.mass
    player_velocity = player.velocity + impulse / 180

    contact_point = ball.position - collision_normal * ball.collision_radius
    car_contact_point = closest_point_on_hitbox(player, contact_point)

    return {
        "collision_normal": collision_normal,
        "impulse": impulse,
        "ball_velocity": ball_velocity,
        "player_velocity": player_velocity,
        "player_collision_region": get_car_collision_region(
            player, ball.position - collision_normal * ball.collision_radius
        ),
        "ball_contact_point": contact_point,
        "player_contact_point": car_contact_point,
    }


def closest_point_on_hitbox(car: Car, point: vec3) -> vec3:

    rel = vec3(
        point[0] - car.position[0],
        point[1] - car.position[1],
        point[2] - car.position[2],
    )
    local_point = mat3_vec3_mul(transpose(car.orientation), rel)
    local_point = vec3(
        local_point[0] - car.hitbox_offset[0],
        local_point[1] - car.hitbox_offset[1],
        local_point[2] - car.hitbox_offset[2],
    )

    # clamp to half-widths
    lx = max(-car.hitbox_widths[0], min(local_point[0], car.hitbox_widths[0]))
    ly = max(-car.hitbox_widths[1], min(local_point[1], car.hitbox_widths[1]))
    lz = max(-car.hitbox_widths[2], min(local_point[2], car.hitbox_widths[2]))

    local_clamped = vec3(lx, ly, lz)
    # back to world: car.position + R * (hitbox_offset + local_clamped)
    world_offset = vec3(
        car.hitbox_offset[0] + local_clamped[0],
        car.hitbox_offset[1] + local_clamped[1],
        car.hitbox_offset[2] + local_clamped[2],
    )
    world_point = vec3(
        car.position[0] + mat3_vec3_mul(car.orientation, world_offset)[0],
        car.position[1] + mat3_vec3_mul(car.orientation, world_offset)[1],
        car.position[2] + mat3_vec3_mul(car.orientation, world_offset)[2],
    )
    return world_point


class RocketLeagueCollisionDetector:
    """
    High-granularity collision detector using RLUtilities simulation.

    This detector:
    1. Takes a Frame and sets up an RLUtilities Game state
    2. Steps the simulation forward with small timesteps
    3. Detects collisions by monitoring state changes between substeps
    4. Returns detailed collision information
    """

    # Physics constants
    BALL_MASS = 30.0
    BALL_RADIUS = 91.25
    CAR_MASS = 180.0  # Approximate

    # Detection thresholds (much tighter for substep detection)
    BALL_VELOCITY_THRESHOLD = 10.0  # Smaller threshold for substeps
    PLAYER_VELOCITY_THRESHOLD = 50.0
    COLLISION_DISTANCE_THRESHOLD = 150.0

    # Simulation parameters
    SUBSTEP_DT = 1.0 / 120.0  # Step at 144Hz (default RL physics rate)
    FRAME_TIME = 1.0 / 30.0  # Standard replay frame time

    def __init__(self, substep_dt: float = SUBSTEP_DT):
        """
        Initialize the collision detector.

        Args:
            substep_dt: Time delta for each simulation substep (default: 1/120s)
        """
        self.substep_dt = substep_dt
        self.substeps_per_frame = (
            int(self.FRAME_TIME / substep_dt) + 1
        )  # add extra for frame time rounding

    def get_player_input(self, frame: Frame, player_name: str) -> Input:
        """Extract player inputs from frame data."""
        frame_data = frame.processed_fields

        controls = Input()
        controls.throttle = frame_data.get(f"{player_name}_throttle", 0)
        controls.steer = frame_data.get(f"{player_name}_steering", 0)
        controls.jump = frame_data.get(f"{player_name}_jump_active", 0) == 1
        controls.boost = frame_data.get(f"{player_name}_boost_active", 0) == 1
        controls.handbrake = frame_data.get(f"{player_name}_drift_active", 0) == 1

        return controls

    def frame_to_game_state(self, frame: Frame) -> Tuple[Game, Dict[str, int]]:
        """
        Convert a Frame to RLUtilities Game state.

        Returns:
            Tuple of (Game object, dict mapping player names to car indices)
        """
        game = Game()
        frame_data = frame.processed_fields
        players = [
            c.removesuffix("_positioning_x")
            for c in frame_data
            if c.endswith("_positioning_x") and not c.startswith("ball_")
        ]
        game.cars = [Car() for c in range(len(players))]

        # Set ball state
        game.ball.position = vec3(
            frame_data["ball_positioning_x"],
            frame_data["ball_positioning_y"],
            frame_data["ball_positioning_z"],
        )
        game.ball.velocity = vec3(
            frame_data["ball_positioning_linear_velocity_x"],
            frame_data["ball_positioning_linear_velocity_y"],
            frame_data["ball_positioning_linear_velocity_z"],
        )
        game.ball.angular_velocity = vec3(
            frame_data["ball_positioning_angular_velocity_x"],
            frame_data["ball_positioning_angular_velocity_y"],
            frame_data["ball_positioning_angular_velocity_z"],
        )

        # Set up cars
        player_name_to_index: Dict[str, int] = {}
        for idx, name in enumerate(players):
            player_name_to_index[name] = idx

            car = game.cars[idx]

            # Set car state
            car.position = vec3(
                frame_data[f"{name}_positioning_x"],
                frame_data[f"{name}_positioning_y"],
                frame_data[f"{name}_positioning_z"],
            )
            car.velocity = vec3(
                frame_data[f"{name}_positioning_linear_velocity_x"],
                frame_data[f"{name}_positioning_linear_velocity_y"],
                frame_data[f"{name}_positioning_linear_velocity_z"],
            )
            car.angular_velocity = vec3(
                frame_data[f"{name}_positioning_angular_velocity_x"],
                frame_data[f"{name}_positioning_angular_velocity_y"],
                frame_data[f"{name}_positioning_angular_velocity_z"],
            )

            # Set rotation (quaternion -> orientation matrix)
            qx = frame_data[f"{name}_positioning_rotation_x"]
            qy = frame_data[f"{name}_positioning_rotation_y"]
            qz = frame_data[f"{name}_positioning_rotation_z"]
            qw = frame_data[f"{name}_positioning_rotation_w"]

            # Convert quaternion to rotation matrix
            car.orientation = self._quaternion_to_matrix(qx, qy, qz, qw)

            # Set default input (no control)
            car.controls = self.get_player_input(frame, name)

        return game, player_name_to_index

    def _quaternion_to_matrix(self, x: float, y: float, z: float, w: float) -> mat3:
        """Convert quaternion to rotation matrix."""
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        return mat3(
            1 - 2 * (yy + zz),
            2 * (xy - wz),
            2 * (xz + wy),
            2 * (xy + wz),
            1 - 2 * (xx + zz),
            2 * (yz - wx),
            2 * (xz - wy),
            2 * (yz + wx),
            1 - 2 * (xx + yy),
        )

    def get_hitbox_corners(self, car):
        offsets = [
            vec3(x, y, z)
            for x in (-car.hitbox_widths[0], car.hitbox_widths[0])
            for y in (-car.hitbox_widths[1], car.hitbox_widths[1])
            for z in (-car.hitbox_widths[2], car.hitbox_widths[2])
        ]
        return [
            car.position + car.orientation @ (car.hitbox_offset + o) for o in offsets
        ]

    def detect_collisions_between_states(
        self,
        prev_game: Game,
        curr_game: Game,
        player_name_map: Dict[str, int],
        frame_number: int,
        substep: int,
    ) -> FrameCollisions:
        """
        Detect collisions by comparing two game states.

        Args:
            prev_game: Game state before step
            curr_game: Game state after step
            player_name_map: Map of player names to car indices
            frame_number: Current frame number
            substep: Current substep within frame
        """
        collisions = FrameCollisions(frame_number=frame_number)

        # Check ball-player collisions
        for name, idx in player_name_map.items():
            car = curr_game.cars[idx]
            prev_car = prev_game.cars[idx]
            ball_center = curr_game.ball.position
            closest_point = closest_point_on_hitbox(car, ball_center)
            distance = Vector3.from_vec3(closest_point - ball_center).magnitude()

            will_collide = distance <= curr_game.ball.collision_radius

            if will_collide:
                collision_values = compute_collision(
                    ball=prev_game.ball,
                    player=prev_car,
                )
                collision = BallPlayerCollision(
                    player_name=name,
                    frame_number=frame_number,
                    substep=substep,
                    ball_velocity=Vector3.from_vec3(collision_values["ball_velocity"]),
                    player_velocity=Vector3.from_vec3(
                        collision_values["player_velocity"]
                    ),
                    ball_contact_point=Vector3.from_vec3(
                        collision_values["ball_contact_point"]
                    ),
                    player_contact_point=Vector3.from_vec3(
                        collision_values["player_contact_point"]
                    ),
                    impulse=Vector3.from_vec3(collision_values["impulse"]),
                    collision_normal=Vector3.from_vec3(
                        collision_values["collision_normal"]
                    ),
                    player_collision_region=collision_values["player_collision_region"],
                )
                collisions.ball_player_collisions.append(collision)

        # Check ball-environment collisions
        env_collision = self._detect_ball_environment_collision(
            prev_game.ball, curr_game.ball, frame_number, substep
        )
        if env_collision:
            collisions.ball_environment_collisions.append(env_collision)

        # Detect player collisions
        for name, idx in player_name_map.items():
            if idx >= len(curr_game.cars):
                continue

            car = curr_game.cars[idx]
            prev_car = prev_game.cars[idx]

            car_vel_change = Vector3.from_vec3(car.velocity) - Vector3.from_vec3(
                prev_car.velocity
            )

            if car_vel_change.magnitude() > self.PLAYER_VELOCITY_THRESHOLD:
                # Check player-player collisions
                for other_name, other_idx in player_name_map.items():
                    if other_name == name or other_idx >= len(curr_game.cars):
                        continue

                    other_car = curr_game.cars[other_idx]
                    distance = Vector3.from_vec3(car.position).distance_to(
                        Vector3.from_vec3(other_car.position)
                    )

                    if distance < self.COLLISION_DISTANCE_THRESHOLD:
                        collision = PlayerPlayerCollision(
                            player1_name=name,
                            player2_name=other_name,
                            frame_number=frame_number,
                            substep=substep,
                            player1_position=Vector3.from_vec3(car.position),
                            player1_velocity_before=Vector3.from_vec3(
                                prev_car.velocity
                            ),
                            player1_velocity_after=Vector3.from_vec3(car.velocity),
                            player2_position=Vector3.from_vec3(other_car.position),
                            player2_velocity_before=Vector3.from_vec3(
                                prev_game.cars[other_idx].velocity
                            ),
                            player2_velocity_after=Vector3.from_vec3(
                                other_car.velocity
                            ),
                            collision_normal=car_vel_change.normalized(),
                            relative_velocity=car_vel_change.magnitude(),
                        )
                        collisions.player_player_collisions.append(collision)

                # Check player-environment collisions
                env_collision = self._detect_player_environment_collision(
                    prev_car, car, name, frame_number, substep
                )
                if env_collision:
                    collisions.player_environment_collisions.append(env_collision)

        return collisions

    def _detect_ball_environment_collision(
        self,
        prev_ball: Ball,
        curr_ball: Ball,
        frame_number: int,
        substep: int,
    ) -> Optional[BallEnvironmentCollision]:
        """Detect if ball collided with environment."""
        pos = Vector3.from_vec3(curr_ball.position)
        prev_vel = Vector3.from_vec3(prev_ball.velocity)
        curr_vel = Vector3.from_vec3(curr_ball.velocity)
        vel_change = curr_vel - prev_vel

        # Ground
        if pos.z <= self.BALL_RADIUS + 10 and vel_change.z > 0:
            return BallEnvironmentCollision(
                surface=EnvironmentSurface.GROUND,
                frame_number=frame_number,
                substep=substep,
                ball_position=pos,
                ball_velocity_before=prev_vel,
                ball_velocity_after=curr_vel,
                ball_velocity_change=vel_change,
                wall_normal=Vector3(0, 0, 1),
                impulse_magnitude=vel_change.magnitude() * self.BALL_MASS,
            )

        # Ceiling (z = 2044)
        if pos.z >= 2044 - self.BALL_RADIUS - 10 and vel_change.z < 0:
            return BallEnvironmentCollision(
                surface=EnvironmentSurface.CEILING,
                frame_number=frame_number,
                substep=substep,
                ball_position=pos,
                ball_velocity_before=prev_vel,
                ball_velocity_after=curr_vel,
                ball_velocity_change=vel_change,
                wall_normal=Vector3(0, 0, -1),
                impulse_magnitude=vel_change.magnitude() * self.BALL_MASS,
            )

        # Side walls (y = ±4096)
        if abs(pos.y) >= 4096 - self.BALL_RADIUS - 10:
            side = 1 if pos.y > 0 else -1
            if vel_change.y * -side > 0:
                return BallEnvironmentCollision(
                    surface=EnvironmentSurface.SIDE_WALL,
                    frame_number=frame_number,
                    substep=substep,
                    ball_position=pos,
                    ball_velocity_before=prev_vel,
                    ball_velocity_after=curr_vel,
                    ball_velocity_change=vel_change,
                    wall_normal=Vector3(0, -side, 0),
                    impulse_magnitude=vel_change.magnitude() * self.BALL_MASS,
                )

        # Back walls (x = ±5120)
        if abs(pos.x) >= 5120 - self.BALL_RADIUS - 10:
            side = 1 if pos.x > 0 else -1
            if vel_change.x * -side > 0:
                return BallEnvironmentCollision(
                    surface=EnvironmentSurface.BACK_WALL,
                    frame_number=frame_number,
                    substep=substep,
                    ball_position=pos,
                    ball_velocity_before=prev_vel,
                    ball_velocity_after=curr_vel,
                    ball_velocity_change=vel_change,
                    wall_normal=Vector3(-side, 0, 0),
                    impulse_magnitude=vel_change.magnitude() * self.BALL_MASS,
                )

        return None

    def _detect_player_environment_collision(
        self,
        prev_car: Car,
        curr_car: Car,
        player_name: str,
        frame_number: int,
        substep: int,
    ) -> Optional[PlayerEnvironmentCollision]:
        """Detect if player collided with environment."""
        pos = Vector3.from_vec3(curr_car.position)
        prev_vel = Vector3.from_vec3(prev_car.velocity)
        curr_vel = Vector3.from_vec3(curr_car.velocity)
        vel_change = curr_vel - prev_vel

        # Only detect significant impacts
        if vel_change.magnitude() < self.PLAYER_VELOCITY_THRESHOLD:
            return None

        # Check if on_ground changed (landed or took off)
        if not prev_car.on_ground and curr_car.on_ground:
            return PlayerEnvironmentCollision(
                player_name=player_name,
                surface=EnvironmentSurface.GROUND,
                frame_number=frame_number,
                substep=substep,
                player_position=pos,
                player_velocity_before=prev_vel,
                player_velocity_after=curr_vel,
                wall_normal=Vector3(0, 0, 1),
            )

        # Check walls based on position and velocity change
        if abs(pos.y) >= 4096 - 100:
            side = 1 if pos.y > 0 else -1
            if vel_change.y * -side > 0:
                return PlayerEnvironmentCollision(
                    player_name=player_name,
                    surface=EnvironmentSurface.SIDE_WALL,
                    frame_number=frame_number,
                    substep=substep,
                    player_position=pos,
                    player_velocity_before=prev_vel,
                    player_velocity_after=curr_vel,
                    wall_normal=Vector3(0, -side, 0),
                )

        if abs(pos.x) >= 5120 - 100:
            side = 1 if pos.x > 0 else -1
            if vel_change.x * -side > 0:
                return PlayerEnvironmentCollision(
                    player_name=player_name,
                    surface=EnvironmentSurface.BACK_WALL,
                    frame_number=frame_number,
                    substep=substep,
                    player_position=pos,
                    player_velocity_before=prev_vel,
                    player_velocity_after=curr_vel,
                    wall_normal=Vector3(-side, 0, 0),
                )

        return None

    def detect_collisions_in_frame(
        self,
        frame: Frame,
        previous_frame: Frame,
        frame_number: int,
    ) -> List[FrameCollisions]:
        """
        Detect all collisions that occur during a frame by simulating it with substeps.

        Args:
            frame: Starting frame state
            frame_number: Frame number for tracking

        Returns:
            List of FrameCollisions, one for each substep that had collisions
        """
        game, player_map = self.frame_to_game_state(frame)
        prev_game, _ = self.frame_to_game_state(previous_frame)

        all_collisions: List[FrameCollisions] = []

        max_substeps = 30  # arbitrary upper bound
        substep_count = 0

        prev_game = self._copy_game_state(game)

        if frame_number == 171:
            pass

        while substep_count < max_substeps:
            dt = self.substep_dt

            # Step simulation
            game.ball.step(dt)
            for car in game.cars:
                car.step(controls=car.controls, dt=dt)

            # Detect collisions between previous and current step
            collisions = self.detect_collisions_between_states(
                prev_game, game, player_map, frame_number, substep_count
            )
            if collisions.has_collisions():
                all_collisions.append(collisions)
                for collision in collisions.ball_player_collisions:
                    ball = game.ball
                    car = game.cars[player_map[collision.player_name]]

                    # Apply the new velocities
                    ball.velocity = collision.ball_velocity.to_vec3()
                    car.velocity = collision.player_velocity.to_vec3()

                    # Optionally: update positions to the moment of impact (if your collision object provides contact_point)
                    ball.position = (
                        collision.ball_contact_point.to_vec3()
                        + collision.collision_normal.to_vec3() * ball.collision_radius
                    )

            # Update for next substep
            prev_game = self._copy_game_state(game)
            substep_count += 1

            # Vector from car to ball
            to_ball = game.ball.position - car.position  # vec3

            # Unit vector in the direction of the car's velocity
            if norm(car.velocity) > 0:
                car_dir = car.velocity / norm(car.velocity)  # normalize velocity
            else:
                car_dir = vec3(0, 0, 0)  # car is stationary

            # Distance along the velocity direction
            distance_along_velocity = dot(to_ball, car_dir)

            # Compute car hitbox half-diagonal (approximate radius of OBB)
            car_half_diagonal = norm(car.hitbox_widths)

            # Relative speed along that direction
            speed_along_direction = dot(car.velocity, car_dir)  # scalar

            # Avoid division by zero
            if speed_along_direction > 0:
                # Time until potential contact
                time_to_contact = (
                    distance_along_velocity
                    - game.ball.collision_radius
                    - car_half_diagonal
                ) / speed_along_direction
            else:
                time_to_contact = float("inf")  # car not moving toward ball
            if time_to_contact > 1:
                break

        return all_collisions

    def _copy_game_state(self, game: Game) -> Game:
        """Create a copy of game state for comparison."""
        new_game = Game()

        # Copy ball
        new_game.ball.position = vec3(game.ball.position)
        new_game.ball.velocity = vec3(game.ball.velocity)
        new_game.ball.angular_velocity = vec3(game.ball.angular_velocity)

        # Copy cars
        cars = []
        for car in game.cars:
            new_car = Car()
            new_car.position = vec3(car.position)
            new_car.velocity = vec3(car.velocity)
            new_car.angular_velocity = vec3(car.angular_velocity)
            new_car.orientation = car.orientation
            new_car.on_ground = car.on_ground
            cars.append(new_car)
        new_game.cars = cars

        return new_game

    def get_collision_summary(self, frame_collisions: FrameCollisions) -> str:
        """Generate a summary report of collisions."""
        lines = [
            f"=" * 70,
            f"FRAME {frame_collisions.frame_number} COLLISION SUMMARY",
            f"=" * 70,
            f"Total collisions: {len(frame_collisions.get_all_collisions())}",
            "",
        ]

        if frame_collisions.ball_player_collisions:
            lines.append(f"Ball-Player: {len(frame_collisions.ball_player_collisions)}")
            for col in frame_collisions.ball_player_collisions:
                lines.append(
                    f"  [{col.substep:3d}] {col.player_name}: "
                    f"Δv={col.ball_velocity_change.magnitude():.1f} uu/s"
                )

        if frame_collisions.ball_environment_collisions:
            lines.append(
                f"Ball-Environment: {len(frame_collisions.ball_environment_collisions)}"
            )
            for col in frame_collisions.ball_environment_collisions:
                lines.append(f"  [{col.substep:3d}] {col.surface.value}")

        if frame_collisions.player_player_collisions:
            lines.append(
                f"Player-Player: {len(frame_collisions.player_player_collisions)}"
            )
            for col in frame_collisions.player_player_collisions:
                lines.append(
                    f"  [{col.substep:3d}] {col.player1_name} vs {col.player2_name}"
                )

        if frame_collisions.player_environment_collisions:
            lines.append(
                f"Player-Environment: {len(frame_collisions.player_environment_collisions)}"
            )
            for col in frame_collisions.player_environment_collisions:
                lines.append(
                    f"  [{col.substep:3d}] {col.player_name} -> {col.surface.value}"
                )

        return "\n".join(lines)
