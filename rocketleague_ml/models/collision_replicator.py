import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import cast, TypedDict, Dict, Any, Optional, List
from rlutilities.linear_algebra import vec3  # type: ignore
from rlutilities.simulation import Car, Ball, Field, obb, sphere  # type: ignore
from rocketleague_ml.core.frame import Frame
from rocketleague_ml.config import (
    X_WALL,
    Y_WALL,
    Z_CEILING,
    # SMALL_BOOST_RADIUS,
    # BIG_BOOST_RADIUS,
    # BOOST_PAD_MAP,
    # BOOST_NEAR_MISS_THRESHOLD,
    # CAR_NEAR_MISS_THRESHOLD,
    # BALL_NEAR_MISS_THRESHOLD,
)


class BallHitbox:
    center: NDArray[np.float64]
    radius: np.float64


class RLUtilities_Ball:
    def __init__(self):
        self.position: NDArray[np.float64] = np.array([])
        self.rotation: NDArray[np.float64] = np.array([])
        self.velocity: NDArray[np.float64] = np.array([])
        self.angular_velocity: NDArray[np.float64] = np.array([])

    def hitbox(self):
        return BallHitbox()


class CarHitbox:
    center: NDArray[np.float64]
    half_width: NDArray[np.float64]


class RLUtilities_Car:
    def __init__(self):
        self.position: NDArray[np.float64] = np.array([])
        self.rotation: NDArray[np.float64] = np.array([])
        self.velocity: NDArray[np.float64] = np.array([])
        self.angular_velocity: NDArray[np.float64] = np.array([])

    def hitbox(self):
        return CarHitbox()


class Collision(TypedDict):
    contact_point: NDArray[np.float64] | None
    normal: NDArray[np.float64] | None
    penetration: float | None


class SphereBoxContact(TypedDict):
    ball_hitbox: BallHitbox
    ball_point: NDArray[np.float64] | None
    car_hitbox: CarHitbox
    car_point: NDArray[np.float64] | None
    normal: NDArray[np.float64] | None


# ---------- Helper Math ----------


def quat_to_mat(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Convert quaternion [x, y, z, w] to 3x3 rotation matrix.
    """
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


def normalize(v: NDArray[np.float64]) -> NDArray[np.float64]:
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return np.zeros_like(v)
    return v / norm


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


# ---------- Collision Dataclass ----------


@dataclass
class CollisionContact:
    object_a: str
    object_b: str

    contact_point: NDArray[np.float64]
    contact_normal: NDArray[np.float64]  # Points from A → B

    penetration_vec: NDArray[np.float64]
    penetration_depth: float

    relative_velocity: NDArray[np.float64]
    relative_speed: float

    restitution: float
    friction: float

    impulse_magnitude: Optional[float] = None
    hardness: Optional[float] = None

    def predict_exit_velocity(
        self,
        v_a: NDArray[np.float64],
        v_b: NDArray[np.float64],
        m_a: float = 1.0,
        m_b: float = 1.0,
    ):
        """
        Predict post-collision linear velocities of object A and B
        using 1D collision along contact normal.
        """

        n = self.contact_normal / np.linalg.norm(self.contact_normal)

        # velocity along the normal
        v_rel = np.dot(v_b - v_a, n)

        # Effective restitution
        e = self.restitution

        # Impulse magnitude along normal
        j = -(1 + e) * v_rel / (1 / m_a + 1 / m_b)

        # Update velocities
        v_a_post = v_a - (j / m_a) * n
        v_b_post = v_b + (j / m_b) * n

        return v_a_post, v_b_post, j


# ---------- Collision Replicator ----------


class Collision_Replicator:
    def __init__(self, ball_radius: float = 91.25):
        self.ball_radius = ball_radius
        self.restitution_default = 0.65
        self.friction_default = 0.35
        self.hardness_scale = 200.0  # tuning constant for sigmoid scaling

    # ---------------- Public ----------------
    def process_frame(self, frame: Frame):
        """
        Given a frame with:
            frame.game.ball
            frame.game.players
            frame.processed_fields (dict)
        computes per-object collisions and adds metrics.
        """
        fields = frame.processed_fields

        ball = frame.game.ball.positioning
        ball_pos = np.array(ball.location.to_tuple())
        ball_vel = np.array(ball.linear_velocity.to_tuple())
        ball_angvel = np.array(ball.angular_velocity.to_tuple())

        for player in frame.game.players.values():
            car = player.car.positioning
            car_pos = np.array(car.location.to_tuple())
            car_rot = np.array(car.rotation.to_tuple())
            car_vel = np.array(car.linear_velocity.to_tuple())
            car_angvel = np.array(car.angular_velocity.to_tuple())
            rl_car = cast(RLUtilities_Car, Car())
            car_hitbox = rl_car.hitbox()

            # Ball ↔ Car
            contact = self._sphere_box_contact(
                ball_pos,
                ball_vel,
                ball_angvel,
                self.ball_radius,
                car_pos,
                car_rot,
                car_vel,
                car_angvel,
                car_hitbox,
                restitution=self.restitution_default,
                friction=self.friction_default,
                name_a="ball",
                name_b=player.name,
            )

            if contact:
                self._add_collision_metrics(fields, contact)

        # Car ↔ Car
        players = list(frame.game.players.values())
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                c1, c2 = players[i].car.positioning, players[j].car.positioning
                rl_car_1, rl_car_2 = cast(RLUtilities_Car, Car()), cast(
                    RLUtilities_Car, Car()
                )
                car_hitbox_1, car_hitbox_2 = rl_car_1.hitbox(), rl_car_2.hitbox()
                contact = self._box_box_contact(
                    np.array(c1.location.to_tuple()),
                    np.array(c1.rotation.to_tuple()),
                    np.array(c1.linear_velocity.to_tuple()),
                    np.array(c1.angular_velocity.to_tuple()),
                    car_hitbox_1,
                    np.array(c2.linear_velocity.to_tuple()),
                    np.array(c2.rotation.to_tuple()),
                    np.array(c2.linear_velocity.to_tuple()),
                    np.array(c2.angular_velocity.to_tuple()),
                    car_hitbox_2,
                    restitution=self.restitution_default,
                    friction=self.friction_default,
                    name_a=players[i].name,
                    name_b=players[j].name,
                )
                if contact:
                    self._add_collision_metrics(fields, contact)

    # ---------------- Collision: Sphere ↔ Box ----------------
    def _sphere_box_contact(
        self,
        sphere_pos: NDArray[np.float64],
        sphere_vel: NDArray[np.float64],
        sphere_angvel: NDArray[np.float64],
        radius: float,
        box_pos: NDArray[np.float64],
        box_rot: NDArray[np.float64],
        box_vel: NDArray[np.float64],
        box_angvel: NDArray[np.float64],
        hitbox: CarHitbox,
        restitution: float,
        friction: float,
        name_a: str,
        name_b: str,
    ) -> Optional[CollisionContact]:
        # Transform sphere center into box local space
        box_rot = quat_to_mat(box_rot)
        rel_center = np.dot(box_rot.T, (sphere_pos - box_pos))

        half_width = np.array(
            [hitbox.half_width[0], hitbox.half_width[1], hitbox.half_width[2]],
            dtype=float,
        )

        # Clamp to box extents
        local_clamped = np.maximum(
            -half_width,
            np.minimum(rel_center, half_width),
        )
        nearest_world = np.dot(box_rot, local_clamped) + box_pos

        # Vector from box to sphere center
        diff = sphere_pos - nearest_world
        dist = np.linalg.norm(diff)

        if dist > radius:
            return None  # No overlap

        normal = normalize(diff) if dist > 1e-8 else np.array([0, 0, 1])
        penetration_depth = cast(float, radius - dist)
        penetration_vec = normal * penetration_depth

        # Relative velocity at contact point
        r_sphere = nearest_world - sphere_pos
        r_box = nearest_world - box_pos
        sphere_point_vel = cast(
            NDArray[np.float64], sphere_vel + np.cross(sphere_angvel, r_sphere)
        )
        box_point_vel = cast(NDArray[np.float64], box_vel + np.cross(box_angvel, r_box))
        rel_vel = sphere_point_vel - box_point_vel
        rel_speed = np.dot(rel_vel, normal)

        impulse = (1 + restitution) * max(rel_speed, 0.0)
        hardness = sigmoid(
            (restitution * abs(rel_speed) * penetration_depth) / self.hardness_scale
        )

        return CollisionContact(
            object_a=name_a,
            object_b=name_b,
            contact_point=nearest_world,
            contact_normal=normal,
            penetration_vec=penetration_vec,
            penetration_depth=penetration_depth,
            relative_velocity=rel_vel,
            relative_speed=abs(rel_speed),
            restitution=restitution,
            friction=friction,
            impulse_magnitude=impulse,
            hardness=hardness,
        )

    # ---------------- Collision: Box ↔ Box ----------------
    def _box_box_contact(
        self,
        pos_a: NDArray[np.float64],
        rot_a: NDArray[np.float64],  # quaternion (4,) OR rotation matrix (3,3)
        vel_a: NDArray[np.float64],
        angvel_a: NDArray[np.float64],
        hb_a: CarHitbox,  # must have .half_width (3,)
        pos_b: NDArray[np.float64],
        rot_b: NDArray[np.float64],  # quaternion (4,) OR rotation matrix (3,3)
        vel_b: NDArray[np.float64],
        angvel_b: NDArray[np.float64],
        hb_b: CarHitbox,
        restitution: float,
        friction: float,
        name_a: str,
        name_b: str,
        hardness_scale: float = 200.0,
        EPS: float = 1e-8,
    ) -> Optional[CollisionContact]:
        """
        Full OBB-OBB SAT. Returns CollisionContact or None if separated.
        """

        # ensure rotation matrices
        if rot_a.shape == (4,):
            rot_a = quat_to_mat(rot_a)
        if rot_b.shape == (4,):
            rot_b = quat_to_mat(rot_b)

        # local axes (columns of rotation matrix)
        A = [rot_a[:, 0], rot_a[:, 1], rot_a[:, 2]]  # world-space axes of box A
        B = [rot_b[:, 0], rot_b[:, 1], rot_b[:, 2]]  # world-space axes of box B
        a = np.array(
            [hb_a.half_width[0], hb_a.half_width[1], hb_a.half_width[2]], dtype=float
        )
        b = np.array(
            [hb_b.half_width[0], hb_b.half_width[1], hb_b.half_width[2]], dtype=float
        )

        # Rotation matrix expressing B in A's coordinate frame: R[i,j] = Ai·Bj
        R = np.zeros((3, 3), dtype=float)
        for i in range(3):
            for j in range(3):
                R[i, j] = np.dot(A[i], B[j])

        # Translation vector from A to B expressed in A's frame: t = (pos_b - pos_a) in A coords
        t_world = pos_b - pos_a
        t = np.array([np.dot(t_world, A[i]) for i in range(3)])

        # Absolute value matrix with epsilon to account for arithmetic errors
        absR = np.abs(R) + 1e-9

        # Track smallest overlap
        smallest_overlap = np.inf
        smallest_axis = None  # will be a tuple ('A', i) or ('B', j) or ('CROSS', i, j)
        axis_sign = 1.0

        # 1) Test axes A0, A1, A2
        for i in range(3):
            ra = a[i]
            rb = b[0] * absR[i, 0] + b[1] * absR[i, 1] + b[2] * absR[i, 2]
            overlap = (ra + rb) - abs(t[i])
            if overlap <= 0:
                return None  # separating axis found
            if overlap < smallest_overlap:
                smallest_overlap = overlap
                smallest_axis = ("A", i)
                axis_sign = 1.0 if t[i] >= 0 else -1.0

        # 2) Test axes B0, B1, B2
        for j in range(3):
            ra = a[0] * absR[0, j] + a[1] * absR[1, j] + a[2] * absR[2, j]
            rb = b[j]
            # t projected onto Bj: t_dot = t · R[:,j] = sum_i t[i] * R[i,j]
            t_dot = t[0] * R[0, j] + t[1] * R[1, j] + t[2] * R[2, j]
            overlap = (ra + rb) - abs(t_dot)
            if overlap <= 0:
                return None
            if overlap < smallest_overlap:
                smallest_overlap = overlap
                smallest_axis = ("B", j)
                axis_sign = 1.0 if t_dot >= 0 else -1.0

        # 3) Test cross-product axes Ai x Bj
        # using indices and the standard formula from Gottschalk
        for i in range(3):
            for j in range(3):
                # ra = a[(i+1)%3] * absR[(i+2)%3, j] + a[(i+2)%3] * absR[(i+1)%3, j]
                ii1 = (i + 1) % 3
                ii2 = (i + 2) % 3
                jj1 = (j + 1) % 3
                jj2 = (j + 2) % 3

                ra = a[ii1] * absR[ii2, j] + a[ii2] * absR[ii1, j]
                rb = b[jj1] * absR[i, jj2] + b[jj2] * absR[i, jj1]

                # tval = | t[ii2] * R[ii1, j] - t[ii1] * R[ii2, j] |
                tval = abs(t[ii2] * R[ii1, j] - t[ii1] * R[ii2, j])
                overlap = (ra + rb) - tval
                if overlap <= 0:
                    return None
                if overlap < smallest_overlap:
                    smallest_overlap = overlap
                    smallest_axis = ("CROSS", i, j)
                    # sign determination for cross axis will be handled below
                    axis_sign = 1.0

        # If we reach here, boxes intersect. smallest_axis gives penetration axis and depth.
        penetration_depth = float(smallest_overlap)
        if not smallest_axis:
            raise Exception("Do not know how we got here.")

        # Determine contact normal in world coordinates
        if smallest_axis[0] == "A":
            i = smallest_axis[1]
            normal = A[i] * axis_sign
        elif smallest_axis[0] == "B":
            j = smallest_axis[1]
            normal = B[j] * axis_sign
        else:  # CROSS
            _, i, j = smallest_axis
            # axis = Ai x Bj
            axis = np.cross(A[i], B[j])
            n_norm = np.linalg.norm(axis)
            if n_norm < EPS:
                # fallback: choose vector between centers
                normal = normalize(t_world)
                if normal.size == 0:
                    normal = np.array([1.0, 0.0, 0.0])
            else:
                normal = axis / n_norm
                # determine sign so normal points from A -> B
                if np.dot(normal, t_world) < 0:
                    normal = -normal

        normal = normalize(normal)

        # Compute an approximate contact point.
        # Project each center onto the normal, compute extents along normal, and place contact in the middle.
        # extent along normal for A and B:
        projA = (
            abs(np.dot(A[0], normal)) * a[0]
            + abs(np.dot(A[1], normal)) * a[1]
            + abs(np.dot(A[2], normal)) * a[2]
        )
        projB = (
            abs(np.dot(B[0], normal)) * b[0]
            + abs(np.dot(B[1], normal)) * b[1]
            + abs(np.dot(B[2], normal)) * b[2]
        )

        # point on A surface along normal (world)
        pointA = pos_a + normal * projA
        pointB = pos_b - normal * projB
        contact_point = (pointA + pointB) * 0.5

        # Relative velocity at contact point
        r_a = contact_point - pos_a
        r_b = contact_point - pos_b
        v_a_point = cast(NDArray[np.float64], vel_a + np.cross(angvel_a, r_a))
        v_b_point = cast(NDArray[np.float64], vel_b + np.cross(angvel_b, r_b))
        rel_vel = v_b_point - v_a_point
        rel_speed = float(
            np.dot(rel_vel, normal)
        )  # signed: >0 means separating along normal from A->B

        # Impulse magnitude (simple proxy, consistent with earlier code)
        # Use compressive component (when rel_speed < 0 meaning closing)
        compressive = max(-rel_speed, 0.0)
        impulse = (1.0 + restitution) * compressive

        # Hardness proxy (sigmoid scale)
        hardness = 1.0 / (
            1.0
            + np.exp(-(restitution * compressive * penetration_depth) / hardness_scale)
        )

        return CollisionContact(
            object_a=name_a,
            object_b=name_b,
            contact_point=contact_point,
            contact_normal=normal,
            penetration_vec=normal * penetration_depth,
            penetration_depth=penetration_depth,
            relative_velocity=rel_vel,
            relative_speed=abs(rel_speed),
            restitution=restitution,
            friction=friction,
            impulse_magnitude=impulse,
            hardness=hardness,
        )

    # ---------------- Add to Frame ----------------
    def _add_collision_metrics(self, fields: Dict[str, Any], contact: CollisionContact):
        prefix = f"{contact.object_a}_collides_{contact.object_b}"
        fields[f"{prefix}_contact"] = True
        fields[f"{prefix}_penetration"] = contact.penetration_depth
        fields[f"{prefix}_rel_speed"] = contact.relative_speed
        fields[f"{prefix}_impulse"] = contact.impulse_magnitude
        fields[f"{prefix}_hardness"] = contact.hardness
        fields[f"{prefix}_friction"] = contact.friction
        fields[f"{prefix}_restitution"] = contact.restitution

        # ---------------- Collision: Environment ----------------

    def _environment_contact(
        self,
        obj_name: str,
        obj_pos: NDArray[np.float64],
        obj_vel: NDArray[np.float64],
        obj_angvel: NDArray[np.float64],
        obj_radius: Optional[float] = None,
        obj_rot: Optional[
            NDArray[np.float64]
        ] = None,  # quaternion or rotation matrix for boxes
        obj_hitbox: Optional[Any] = None,  # must have .half_width for boxes
        restitution: float = 0.65,
        friction: float = 0.35,
    ) -> List[CollisionContact]:
        """
        Detect collisions with walls, ground, ceiling.
        Handles both spheres (balls) and rotated boxes (cars).
        Returns list of CollisionContact objects.
        """
        bounds: Dict[str, int] = {
            "x_min": -X_WALL,
            "x_max": X_WALL,
            "y_min": -Y_WALL,
            "y_max": Y_WALL,
            "z_min": 0,
            "z_max": Z_CEILING,
        }

        contacts: List[CollisionContact] = []

        # ----- Determine object points to test -----
        # Spheres: single center + radius
        if obj_radius is not None:
            points = [obj_pos]
            radii = [obj_radius]
        # Boxes: generate 8 world-space corners
        elif obj_rot is not None and obj_hitbox is not None:
            # convert quaternion to rotation matrix
            if obj_rot.shape == (4,):
                rot_mat = quat_to_mat(obj_rot)
            else:
                rot_mat = obj_rot

            hx, hy, hz = obj_hitbox.half_width
            # 8 corners
            local_corners = np.array(
                [
                    [+hx, +hy, +hz],
                    [+hx, +hy, -hz],
                    [+hx, -hy, +hz],
                    [+hx, -hy, -hz],
                    [-hx, +hy, +hz],
                    [-hx, +hy, -hz],
                    [-hx, -hy, +hz],
                    [-hx, -hy, -hz],
                ]
            )
            points = (rot_mat @ local_corners.T).T + obj_pos
            radii = [0.0] * 8  # corners are points
        else:
            raise ValueError(
                "Must provide either obj_radius (sphere) or obj_rot + obj_hitbox (box)"
            )

        # ----- Check each point against each surface -----
        for p_idx, point in enumerate(points):
            for surf_name, (surf_normal, bound, axis, direction) in {
                "ground": (np.array([0, 0, 1]), bounds["z_min"], "z", 1),
                "ceiling": (np.array([0, 0, -1]), bounds["z_max"], "z", -1),
                "wall_neg_x": (np.array([1, 0, 0]), bounds["x_min"], "x", 1),
                "wall_pos_x": (np.array([-1, 0, 0]), bounds["x_max"], "x", -1),
                "wall_neg_y": (np.array([0, 1, 0]), bounds["y_min"], "y", 1),
                "wall_pos_y": (np.array([0, -1, 0]), bounds["y_max"], "y", -1),
            }.items():
                idx = {"x": 0, "y": 1, "z": 2}[axis]
                r = radii[p_idx]
                dist = (point[idx] - bound) * direction
                penetration = r - dist
                if penetration > 0:
                    contact_point = point - surf_normal * r
                    r = contact_point - obj_pos
                    vel_at_point = obj_vel + np.cross(obj_angvel, r)
                    rel_speed = np.dot(vel_at_point, surf_normal)
                    impulse = (1 + restitution) * max(rel_speed, 0.0)
                    hardness = 1.0 / (
                        1.0
                        + np.exp(
                            -(restitution * abs(rel_speed) * penetration)
                            / self.hardness_scale
                        )
                    )
                    contacts.append(
                        CollisionContact(
                            object_a=obj_name,
                            object_b=surf_name,
                            contact_point=contact_point,
                            contact_normal=surf_normal,
                            penetration_vec=surf_normal * penetration,
                            penetration_depth=penetration,
                            relative_velocity=obj_vel,
                            relative_speed=abs(rel_speed),
                            restitution=restitution,
                            friction=friction,
                            impulse_magnitude=impulse,
                            hardness=hardness,
                        )
                    )

        return contacts
