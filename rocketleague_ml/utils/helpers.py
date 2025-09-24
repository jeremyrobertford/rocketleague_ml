import math
from rocketleague_ml.types.attributes import Rotation_Dict


def convert_byte_to_float(bytes: int):
    byte_val = max(0, min(255, bytes))
    return (byte_val - 128) / 127.0


def convert_euler_to_quat(
    yaw: float | None, pitch: float | None, roll: float | None
) -> Rotation_Dict:
    y = yaw or 0
    p = pitch or 0
    r = roll or 0
    degrees = abs(y) > math.pi or abs(p) > math.pi or abs(r) > math.pi
    if degrees:
        yaw = math.radians(y)
        pitch = math.radians(p)
        roll = math.radians(r)

    cy = math.cos(y * 0.5)
    sy = math.sin(y * 0.5)
    cp = math.cos(p * 0.5)
    sp = math.sin(p * 0.5)
    cr = math.cos(r * 0.5)
    sr = math.sin(r * 0.5)

    # z-y-x order (yaw-pitch-roll)
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy

    return {"x": x, "y": y, "z": z, "w": w}
