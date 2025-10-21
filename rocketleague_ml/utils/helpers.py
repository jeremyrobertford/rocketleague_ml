import math
import numpy as np
import pandas as pd
from typing import Tuple
from rocketleague_ml.types.attributes import Rotation_Dict
from rocketleague_ml.config import BOOST_PAD_MAP


def convert_byte_to_float(bytes: int):
    byte_val = max(0, min(255, bytes))
    return np.clip((byte_val - 128) / 128.0, -1, 1)


def quat_to_euler(
    qx: float,
    qy: float,
    qz: float,
    qw: float,
) -> Tuple[float, float, float]:
    """
    Converts quaternion rotation (x, y, z, w) to yaw, pitch, and roll in radians.

    Returns:
        yaw   (float or np.ndarray): rotation about Z (facing direction, horizontal turn)
        pitch (float or np.ndarray): rotation about Y (nose up/down)
        roll  (float or np.ndarray): rotation about X (banking/tilting)
    """
    # Yaw (around Z axis)
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    # Pitch (around Y axis)
    pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0))
    # Roll (around X axis)
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))

    return yaw, pitch, roll  # type: ignore


def series_quat_to_euler(
    qx: pd.Series,
    qy: pd.Series,
    qz: pd.Series,
    qw: pd.Series,
) -> Tuple[
    np.ndarray[Tuple[int], np.dtype[np.float64]],
    np.ndarray[Tuple[int], np.dtype[np.float64]],
    np.ndarray[Tuple[int], np.dtype[np.float64]],
]:
    """
    Converts quaternion rotation (x, y, z, w) to yaw, pitch, and roll in radians.

    Returns:
        yaw   (float or np.ndarray): rotation about Z (facing direction, horizontal turn)
        pitch (float or np.ndarray): rotation about Y (nose up/down)
        roll  (float or np.ndarray): rotation about X (banking/tilting)
    """
    # Yaw (around Z axis)
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    # Pitch (around Y axis)
    pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0))
    # Roll (around X axis)
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))

    return yaw, pitch, roll  # type: ignore


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


def parse_boost_actor_name(object_str: str):
    try:
        idx = int(object_str.split("_")[-1])  # pickup suffix number
    except ValueError:
        return None

    return BOOST_PAD_MAP.get(idx, None)
