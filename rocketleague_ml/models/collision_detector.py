import math
from typing import Dict, List, Optional, Tuple, TypedDict
from dataclasses import dataclass
from rocketleague_ml.core.frame import Frame


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

    def distance_to(self, other: "Vector3") -> float:
        return (self - other).magnitude()


@dataclass
class CollisionInfo:
    player_name: str
    impulse_magnitude: float
    impulse_vector: Vector3
    collision_normal: Vector3
    ball_velocity_change: Vector3
    player_position: Vector3
    ball_position: Vector3
    distance_to_ball: float
    confidence: str  # "high", "medium", "low"


class RocketLeagueCollisionDetector:
    # Rocket League constants
    BALL_MASS = 30.0  # mass units
    BALL_RADIUS = 91.25  # unreal units

    # Typical car hitbox dimensions (Octane as default, can be customized)
    CAR_HITBOX = {"length": 118.0, "width": 84.0, "height": 36.0}

    # Detection thresholds
    MAX_COLLISION_DISTANCE = 200.0  # uu - max distance for likely collision
    MIN_BALL_VELOCITY_CHANGE = 100.0  # uu/s - minimum change to consider collision

    def __init__(self, ball_mass: float = BALL_MASS):
        self.ball_mass = ball_mass

    def parse_frame(
        self, frame: Frame
    ) -> Tuple[Vector3, Vector3, Dict[str, Tuple[Vector3, Vector3]]]:
        """
        Parse frame data into ball position/velocity and player positions/velocities.

        Returns:
            ball_pos, ball_vel, players_dict
            where players_dict = {name: (position, velocity)}
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

        players: Dict[str, Tuple[Vector3, Vector3]] = {}
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
            players[name] = (pos, vel)

        return ball_pos, ball_vel, players

    def detect_collision(
        self, prev_frame: Frame, curr_frame: Frame
    ) -> Optional[CollisionInfo]:
        """
        Detect if a collision occurred between two frames and identify the player.

        Args:
            prev_frame: Previous frame data
            curr_frame: Current frame data

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

        # Find the player closest to the ball in current frame
        class Candidate(TypedDict):
            name: str
            score: float
            curr_pos: Vector3
            dist_to_ball: float
            moved: bool

        candidates: List[Candidate] = []

        for name, (curr_pos, _) in curr_players.items():
            # Get previous position
            if name not in prev_players:
                continue
            prev_pos, prev_vel = prev_players[name]

            # Calculate distances
            dist_to_ball_curr = curr_pos.distance_to(curr_ball_pos)
            dist_to_ball_prev = prev_pos.distance_to(prev_ball_pos)

            # Check if player moved (ignore stationary players)
            player_moved = (curr_pos - prev_pos).magnitude() > 1.0

            # Calculate how much closer they got to the ball
            approach_distance = dist_to_ball_prev - dist_to_ball_curr

            # Score this candidate
            score = 0
            if dist_to_ball_curr < self.MAX_COLLISION_DISTANCE:
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

            candidates.append(
                {
                    "name": name,
                    "score": score,
                    "curr_pos": curr_pos,
                    "dist_to_ball": dist_to_ball_curr,
                    "moved": player_moved,
                }
            )

        if not candidates:
            return None

        # Sort by score
        candidates.sort(key=lambda x: x["score"], reverse=True)
        best = candidates[0]

        # Determine confidence
        if best["dist_to_ball"] < 150 and best["moved"]:
            confidence = "high"
        elif best["dist_to_ball"] < self.MAX_COLLISION_DISTANCE:
            confidence = "medium"
        else:
            confidence = "low"

        # Calculate impulse
        impulse_vec = ball_vel_change * self.ball_mass
        impulse_mag = impulse_vec.magnitude()

        # Collision normal (direction of ball velocity change)
        collision_normal = ball_vel_change.normalized()

        return CollisionInfo(
            player_name=best["name"],
            impulse_magnitude=impulse_mag,
            impulse_vector=impulse_vec,
            collision_normal=collision_normal,
            ball_velocity_change=ball_vel_change,
            player_position=best["curr_pos"],
            ball_position=curr_ball_pos,
            distance_to_ball=best["dist_to_ball"],
            confidence=confidence,
        )

    def get_collision_report(self, collision: CollisionInfo) -> str:
        """Generate a human-readable report of the collision."""
        report = f"""
Collision Detected - Confidence: {collision.confidence.upper()}
{'='*50}
Player: {collision.player_name}
Position: ({collision.player_position.x:.1f}, {collision.player_position.y:.1f}, {collision.player_position.z:.1f})
Distance to ball: {collision.distance_to_ball:.1f} uu

Ball Impact:
  Position: ({collision.ball_position.x:.1f}, {collision.ball_position.y:.1f}, {collision.ball_position.z:.1f})
  Velocity change: ({collision.ball_velocity_change.x:.1f}, {collision.ball_velocity_change.y:.1f}, {collision.ball_velocity_change.z:.1f})
  Speed change: {collision.ball_velocity_change.magnitude():.1f} uu/s

Collision Normal: ({collision.collision_normal.x:.3f}, {collision.collision_normal.y:.3f}, {collision.collision_normal.z:.3f})

Impulse:
  Magnitude: {collision.impulse_magnitude:.1f} mass·uu/s
  Vector: ({collision.impulse_vector.x:.1f}, {collision.impulse_vector.y:.1f}, {collision.impulse_vector.z:.1f})
"""
        return report
