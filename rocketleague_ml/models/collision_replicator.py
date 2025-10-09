# type: ignore
import numpy as np
from rlutilities.linear_algebra import vec3
from rlutilities.simulation import Car, Ball, Field, obb, sphere
from rocketleague_ml.core.frame import Frame
from rocketleague_ml.config import (
    X_WALL,
    Y_WALL,
    Z_GROUND,
    Z_CEILING,
    BALL_RADIUS,
    SMALL_BOOST_RADIUS,
    BIG_BOOST_RADIUS,
    BOOST_PAD_MAP,
    BOOST_NEAR_MISS_THRESHOLD,
)

import numpy as np


def quaternion_to_matrix(q):
    """Convert quaternion [x, y, z, w] to 3x3 rotation matrix."""
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ]
    )


def collides_with(a, b):
    """
    Check if two OBBs collide using the Separating Axis Theorem.

    Each OBB is a dict with:
      center: np.array([x, y, z])
      half_size: np.array([hx, hy, hz])
      rotation: np.array([x, y, z, w]) (quaternion)
    """
    # unpack
    C1, E1 = np.array(a["center"]), np.array(a["half_size"])
    C2, E2 = np.array(b["center"]), np.array(b["half_size"])
    R1 = quaternion_to_matrix(a["rotation"])
    R2 = quaternion_to_matrix(b["rotation"])

    # rotation matrix expressing b in a's frame
    R = R1.T @ R2
    T = R1.T @ (C2 - C1)  # translation in a’s frame

    # numerical tolerance
    EPS = 1e-6
    absR = np.abs(R) + EPS

    # test 15 possible separating axes
    for i in range(3):  # A’s axes
        ra = E1[i]
        rb = np.dot(E2, absR[i])
        if abs(T[i]) > ra + rb:
            return False

    for i in range(3):  # B’s axes
        ra = np.dot(E1, absR[:, i])
        rb = E2[i]
        if abs(np.dot(T, R[:, i])) > ra + rb:
            return False

    # cross products of axes
    for i in range(3):
        for j in range(3):
            ra = (
                E1[(i + 1) % 3] * absR[(i + 2) % 3, j]
                + E1[(i + 2) % 3] * absR[(i + 1) % 3, j]
            )
            rb = (
                E2[(j + 1) % 3] * absR[i, (j + 2) % 3]
                + E2[(j + 2) % 3] * absR[i, (j + 1) % 3]
            )
            t = T[(i + 2) % 3] * R[(i + 1) % 3, j] - T[(i + 1) % 3] * R[(i + 2) % 3, j]
            if abs(t) > ra + rb:
                return False

    return True


class Collision_Replicator:
    # Collision radii and dimensions (in Unreal Units)
    # Near miss thresholds

    def __init__(self):
        """Initialize with boost pad locations."""
        self.boost_pads = BOOST_PAD_MAP

    @staticmethod
    def _aabb_overlap(pos_a, half_a, pos_b, half_b):
        """Cheap AABB overlap check before real collision test."""
        return np.all(np.abs(pos_a - pos_b) <= (half_a + half_b))

    def _get_position(self, frame: Frame, prefix: str) -> vec3:
        """Extract position from frame.processed_fields as RLU vec3."""
        return vec3(
            frame.processed_fields[f"{prefix}_positioning_x"],
            frame.processed_fields[f"{prefix}_positioning_y"],
            frame.processed_fields[f"{prefix}_positioning_z"],
        )

    def _get_velocity(self, frame: Frame, prefix: str) -> vec3:
        """Extract velocity from frame.processed_fields as RLU vec3."""
        return vec3(
            frame.processed_fields[f"{prefix}_positioning_linear_velocity_x"],
            frame.processed_fields[f"{prefix}_positioning_linear_velocity_y"],
            frame.processed_fields[f"{prefix}_positioning_linear_velocity_z"],
        )

    def _create_field(self):
        return Field()

    def _create_car_hitbox(self, position: vec3) -> Car:
        """Create an OBB (oriented bounding box) for a car at given position."""
        car = Car()
        car.position = position
        return car.hitbox()

    def _create_ball_sphere(self, position: vec3, velocity: vec3) -> Ball:
        """Create a sphere for the ball at given position."""
        ball = Ball()
        ball.position = position
        ball.velocity = velocity
        return ball

    def _check_wall_collision(self, pos: vec3, radius: float) -> bool:
        """Check if object collides with arena walls."""
        x, y, z = pos[0], pos[1], pos[2]

        # Side walls
        if abs(x) + radius >= X_WALL:
            return True

        # End walls
        if abs(y) + radius >= Y_WALL:
            return True

        # Ceiling
        if z + radius >= Z_CEILING:
            return True

        return False

    def _check_ground_collision(self, pos: vec3, hitbox: obb) -> bool:
        """Check if object collides with ground."""
        half_height = (
            hitbox.radius if getattr(hitbox, "radius") else hitbox.half_width[3]
        )
        z_bottom = hitbox.center[2] - half_height
        return z_bottom <= 2.0

    def _check_player_ball_collision(
        self, player_hitbox: obb, ball_sphere: Ball
    ) -> bool:
        """Check collision between player hitbox and ball using RLU."""
        return collides_with(ball_sphere, player_hitbox)

    def _check_player_player_collision(self, hitbox_a: obb, hitbox_b: obb) -> bool:
        """Check collision between two player hitboxes using RLU."""
        return collides_with(hitbox_a, hitbox_b)

    def _check_boost_collision(
        self, player_pos: vec3, boost_pos: vec3, boost_size: int
    ) -> bool:
        """Check if player collects a boost pad."""
        radius = BIG_BOOST_RADIUS if boost_size == 100 else SMALL_BOOST_RADIUS

        # Create spheres for collision check
        player_sphere = sphere(player_pos, 120.0)  # Approximate car as sphere
        boost_sphere = sphere(boost_pos, radius)

        return collides_with(player_sphere, boost_sphere)

    def _check_boost_near_miss(self, player_pos: vec3, boost_pos: vec3) -> bool:
        """Check if player passed near a boost pad without collecting it."""
        diff = player_pos - boost_pos
        distance = diff.magnitude()
        return distance <= BOOST_NEAR_MISS_THRESHOLD

    def process_frame(self, frame: Frame) -> Frame:
        """Process all collisions for the given frame."""

        # Get ball position and create ball sphere
        ball_pos = self._get_position(frame, "ball")
        ball_vel = self._get_velocity(frame, "ball")
        ball = self._create_ball_sphere(ball_pos, ball_vel)

        # Check ball collisions
        frame.processed_fields["ball_collision_with_wall"] = self._check_wall_collision(
            ball_pos, BALL_RADIUS
        )
        frame.processed_fields["ball_collision_with_ground"] = (
            self._check_ground_collision(ball_pos, ball.hitbox())
        )

        # Get all player positions and hitboxes
        player_names = [player.name for player in frame.game.players.values()]
        player_positions = {}
        player_velocities = {}
        player_hitboxes = {}

        for player_name in player_names:
            pos = self._get_position(frame, player_name)
            vel = self._get_velocity(frame, player_name)
            player_positions[player_name] = pos
            player_velocities[player_name] = vel
            player_hitboxes[player_name] = self._create_car_hitbox(pos)

        # Process each player
        for player_name in player_names:
            player_pos = player_positions[player_name]
            player_hitbox = player_hitboxes[player_name]

            # Player-ball collision using RLU
            collision_with_ball = self._check_player_ball_collision(player_hitbox, ball)
            # game.last_player_to_touch_ball
            # game.previous_teammate_to_touch_ball
            # game.possession_change
            # mechanics.fifty-fifty
            # mechanics.team-pinch
            # mechanics.ball_hit_speed
            # mechanics.shot
            # mechanics.save

            # Player-wall collision
            collision_with_wall = self._check_wall_collision(
                player_pos, 150.0  # Approximate car radius
            )
            # mechanics.pinch

            # Player-ground collision
            collision_with_ground = self._check_ground_collision(
                player_pos, player_hitbox
            )
            # mechanics.ground_pinch

            # Player-ground collision
            collision_with_ceiling = self._check_ceiling_collision(
                player_pos, player_hitbox
            )
            # mechanics.ceiling_pinch

            # Player-player collisions using RLU
            collision_with_player = False
            collision_player_names = []

            for other_name in player_names:
                if other_name != player_name:
                    if self._check_player_player_collision(
                        player_hitbox, player_hitboxes[other_name]
                    ):
                        collision_with_player = True
                        collision_player_names.append(other_name)
                        # player_collisions.bump
                        # player_collisions.bump_dodge
                        # player_collisions.bump_jump
                        # player_collisions.bump_double_jump
                        # player_collisions.bump_turn
                        # player_collisions.bump_brake
                        # player_collisions.bump_speed_up
                    # player_collisions.near_bump
                    # player_collisions.near_bump_dodge
                    # player_collisions.near_bump_jump
                    # player_collisions.near_bump_double_jump
                    # player_collisions.near_bump_turn
                    # player_collisions.near_bump_brake
                    # player_collisions.near_bump_speed_up

            # Player-boost collisions and near misses
            collision_with_boost = False
            boost_near_miss = False
            collected_boost_ids = []
            near_miss_boost_ids = []

            for boost_id, (bx, by, bz, size) in self.boost_pads.items():
                boost_pos = vec3(bx, by, bz)

                if self._check_boost_collision(player_pos, boost_pos, size):
                    collision_with_boost = True
                    collected_boost_ids.append(boost_id)
                    # boost_management.big_pads_collected
                    # boost_management.small_pads_collected
                    # boost_management.big_pads_stolen
                    # boost_management.small_pads_stolen
                    # boost_management.overfill
                    # boost_management.boost_stolen_by
                    # boost_management.stolen_overfill
                    # boost_management.stolen_boost_amount
                elif self._check_boost_near_miss(player_pos, boost_pos):
                    boost_near_miss = True
                    near_miss_boost_ids.append(boost_id)
                    # boost_managemnet.near_missed_stolen_boost

            # Store results
            frame.processed_fields[f"{player_name}_collision_with_ball"] = (
                collision_with_ball
            )
            frame.processed_fields[f"{player_name}_collision_with_wall"] = (
                collision_with_wall
            )
            frame.processed_fields[f"{player_name}_collision_with_ground"] = (
                collision_with_ground
            )
            frame.processed_fields[f"{player_name}_collision_with_player"] = (
                collision_with_player
            )
            frame.processed_fields[f"{player_name}_collision_with_boost"] = (
                collision_with_boost
            )
            frame.processed_fields[f"{player_name}_boost_near_miss"] = boost_near_miss

            # Store detailed collision info (optional)
            frame.processed_fields[f"{player_name}_collision_player_names"] = (
                collision_player_names
            )
            frame.processed_fields[f"{player_name}_collected_boost_ids"] = (
                collected_boost_ids
            )
            frame.processed_fields[f"{player_name}_near_miss_boost_ids"] = (
                near_miss_boost_ids
            )

        return frame
