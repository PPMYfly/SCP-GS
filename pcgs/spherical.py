"""Pure-NumPy spherical reference frames for Phase8.7.

The original spherical context used directions from the world origin.  That
choice changes when an otherwise identical scene is translated.  This module
defines a portable reference profile instead: camera positions select a robust
geometric median, while scenes without camera metadata use their geometry
bounding-box centre.  The profile stores the reference in JSON-safe float32
form so an encoder and decoder can use exactly the same transform.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


SPHERICAL_PROFILE_FORMAT = "phase8.7_spherical_reference_v1"
SPHERICAL_PROFILE_DTYPE = "float32"
CAMERA_CENTER_METHOD = "camera_geometric_median_weiszfeld_v1"
BBOX_CENTER_METHOD = "geometry_bbox_center_v1"
RADIUS_SCALE_METHOD = "median_positive_geometry_radius_v1"
DIRECTION_TRANSFORM = "unit_offset_zero_at_center_v1"
LOG_RADIUS_TRANSFORM = "log1p_radius_over_scale_v1"

_PROFILE_KEYS = {
    "format",
    "dtype",
    "center",
    "center_method",
    "camera_count",
    "radius_scale",
    "radius_scale_method",
    "radius_epsilon",
    "direction_transform",
    "log_radius_transform",
    "weiszfeld_tolerance",
    "weiszfeld_max_iterations",
    "weiszfeld_iterations",
}

__all__ = [
    "BBOX_CENTER_METHOD",
    "CAMERA_CENTER_METHOD",
    "DIRECTION_TRANSFORM",
    "LOG_RADIUS_TRANSFORM",
    "RADIUS_SCALE_METHOD",
    "SPHERICAL_PROFILE_DTYPE",
    "SPHERICAL_PROFILE_FORMAT",
    "build_spherical_profile",
    "geometric_median",
    "load_camera_positions",
    "load_spherical_profile",
    "save_spherical_profile",
    "spherical_features",
    "validate_spherical_profile",
]


def _point_array(value: Any, name: str, *, nonempty: bool = True) -> np.ndarray:
    """Return a finite float64 ``(N, 3)`` array without accepting booleans."""

    raw = np.asarray(value)
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    if nonempty and raw.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one point")
    if np.issubdtype(raw.dtype, np.bool_) or not (
        np.issubdtype(raw.dtype, np.integer)
        or np.issubdtype(raw.dtype, np.floating)
    ):
        raise ValueError(f"{name} must contain real numeric coordinates")
    result = raw.astype(np.float64, copy=False)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite coordinates")
    return result


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _float32_scalar(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a real number")
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    encoded = np.float32(numeric)
    if not np.isfinite(encoded):
        raise ValueError(f"{name} is outside the float32 range")
    result = float(encoded)
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def load_camera_positions(path: str | Path) -> np.ndarray:
    """Load and strictly validate ``position`` entries from cameras.json.

    Supplying a camera file is an explicit request to use it.  Consequently an
    empty or malformed file is rejected instead of silently falling back to a
    different reference centre.
    """

    camera_path = Path(path)
    try:
        records = json.loads(camera_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read valid camera JSON from {camera_path}") from error
    if not isinstance(records, list) or not records:
        raise ValueError("cameras.json must contain a non-empty camera list")

    positions: list[list[float]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"camera {index} must be a JSON object")
        position = record.get("position")
        if not isinstance(position, list) or len(position) != 3:
            raise ValueError(f"camera {index} position must be a three-number list")
        coordinates: list[float] = []
        for axis, coordinate in enumerate(position):
            if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
                raise ValueError(
                    f"camera {index} position[{axis}] must be a real number"
                )
            numeric = float(coordinate)
            if not np.isfinite(numeric):
                raise ValueError(f"camera {index} position must be finite")
            coordinates.append(numeric)
        positions.append(coordinates)
    return np.asarray(positions, dtype=np.float64)


def _weiszfeld(
    points: np.ndarray,
    *,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, int]:
    """Modified Weiszfeld iteration, including coincident-sample handling."""

    # A coordinate median is deterministic and translation equivariant.
    estimate = np.median(points, axis=0)
    extent = np.ptp(points, axis=0)
    scale = float(np.hypot(np.hypot(extent[0], extent[1]), extent[2]))
    if scale == 0.0:
        return points[0].copy(), 0
    stop_distance = tolerance * scale
    coincidence_distance = np.finfo(np.float64).eps * max(scale, 1.0)

    for iteration in range(1, max_iterations + 1):
        offsets = points - estimate
        distances = np.hypot(np.hypot(offsets[:, 0], offsets[:, 1]), offsets[:, 2])
        coincident = distances <= coincidence_distance
        nonzero = ~coincident
        if not np.any(nonzero):
            return estimate, iteration - 1

        weights = 1.0 / distances[nonzero]
        candidate = np.sum(points[nonzero] * weights[:, None], axis=0) / np.sum(weights)

        # Modified Weiszfeld (Vardi-Zhang) avoids division by zero and decides
        # correctly whether a coincident sample itself is already optimal.
        coincident_count = int(coincident.sum())
        if coincident_count:
            residual = np.sum(offsets[nonzero] / distances[nonzero, None], axis=0)
            residual_norm = float(
                np.hypot(np.hypot(residual[0], residual[1]), residual[2])
            )
            if residual_norm <= coincident_count:
                return estimate, iteration - 1
            blend = coincident_count / residual_norm
            candidate = (1.0 - blend) * candidate + blend * estimate

        movement = float(
            np.hypot(
                np.hypot(candidate[0] - estimate[0], candidate[1] - estimate[1]),
                candidate[2] - estimate[2],
            )
        )
        estimate = candidate
        if movement <= stop_distance:
            return estimate, iteration
    return estimate, max_iterations


def geometric_median(
    points: Any,
    *,
    tolerance: float = 1e-6,
    max_iterations: int = 256,
) -> np.ndarray:
    """Compute a robust three-dimensional geometric median with Weiszfeld."""

    values = _point_array(points, "points")
    tolerance_value = float(tolerance)
    if not np.isfinite(tolerance_value) or tolerance_value <= 0.0:
        raise ValueError("tolerance must be a finite positive number")
    iteration_limit = _strict_int(max_iterations, "max_iterations", minimum=1)
    median, _ = _weiszfeld(
        values,
        tolerance=tolerance_value,
        max_iterations=iteration_limit,
    )
    if not np.isfinite(median).all():
        raise ValueError("geometric median is not finite")
    return median


def _bbox_center(xyz: np.ndarray) -> np.ndarray:
    minimum = np.min(xyz, axis=0)
    maximum = np.max(xyz, axis=0)
    # Half-sums avoid overflowing ``maximum - minimum`` for extreme inputs.
    return minimum * 0.5 + maximum * 0.5


def _radius_scale(xyz: np.ndarray, center: np.ndarray) -> float:
    offsets = xyz - center
    radii = np.hypot(np.hypot(offsets[:, 0], offsets[:, 1]), offsets[:, 2])
    if not np.isfinite(radii).all():
        raise ValueError("geometry radii are outside the supported numeric range")
    positive = radii[radii > 0.0]
    raw_scale = float(np.median(positive)) if positive.size else 1.0
    limits = np.finfo(np.float32)
    # Clamping only handles representational extremes; ordinary scenes retain
    # their exact robust median-positive radius.
    raw_scale = min(max(raw_scale, float(limits.tiny)), float(limits.max))
    return _float32_scalar(raw_scale, "radius_scale", positive=True)


def build_spherical_profile(
    xyz: Any,
    cameras_path: str | Path | None = None,
    *,
    tolerance: float = 1e-6,
    max_iterations: int = 256,
    radius_epsilon: float = 1e-6,
) -> dict[str, Any]:
    """Build a deterministic JSON-safe float32 spherical reference profile."""

    geometry = _point_array(xyz, "xyz")
    tolerance_value = _float32_scalar(
        tolerance, "weiszfeld_tolerance", positive=True
    )
    iteration_limit = _strict_int(max_iterations, "max_iterations", minimum=1)
    epsilon_value = _float32_scalar(radius_epsilon, "radius_epsilon", positive=True)

    if cameras_path is None:
        raw_center = _bbox_center(geometry)
        center_method = BBOX_CENTER_METHOD
        camera_count = 0
        iterations = 0
    else:
        camera_positions = load_camera_positions(cameras_path)
        raw_center, iterations = _weiszfeld(
            camera_positions,
            tolerance=float(tolerance_value),
            max_iterations=iteration_limit,
        )
        center_method = CAMERA_CENTER_METHOD
        camera_count = int(camera_positions.shape[0])

    center = np.asarray(
        [_float32_scalar(value, f"center[{axis}]") for axis, value in enumerate(raw_center)],
        dtype=np.float32,
    )
    scale = _radius_scale(geometry, center.astype(np.float64))
    profile: dict[str, Any] = {
        "format": SPHERICAL_PROFILE_FORMAT,
        "dtype": SPHERICAL_PROFILE_DTYPE,
        "center": [float(value) for value in center],
        "center_method": center_method,
        "camera_count": camera_count,
        "radius_scale": scale,
        "radius_scale_method": RADIUS_SCALE_METHOD,
        "radius_epsilon": epsilon_value,
        "direction_transform": DIRECTION_TRANSFORM,
        "log_radius_transform": LOG_RADIUS_TRANSFORM,
        "weiszfeld_tolerance": tolerance_value,
        "weiszfeld_max_iterations": iteration_limit,
        "weiszfeld_iterations": int(iterations),
    }
    normalized = validate_spherical_profile(profile)
    # This also guards accidental NumPy scalars and non-standard NaN literals.
    json.dumps(normalized, allow_nan=False)
    return normalized


def validate_spherical_profile(profile: Any) -> dict[str, Any]:
    """Validate the exact profile schema and return canonical JSON values."""

    if not isinstance(profile, dict) or set(profile) != _PROFILE_KEYS:
        raise ValueError("spherical profile has invalid top-level fields")
    if profile.get("format") != SPHERICAL_PROFILE_FORMAT:
        raise ValueError("missing or unsupported spherical profile format")
    if profile.get("dtype") != SPHERICAL_PROFILE_DTYPE:
        raise ValueError("spherical profile dtype must be float32")

    raw_center = profile.get("center")
    if not isinstance(raw_center, list) or len(raw_center) != 3:
        raise ValueError("spherical profile center must be a three-number list")
    center = [
        _float32_scalar(value, f"spherical profile center[{axis}]")
        for axis, value in enumerate(raw_center)
    ]

    method = profile.get("center_method")
    if method not in (CAMERA_CENTER_METHOD, BBOX_CENTER_METHOD):
        raise ValueError("spherical profile has an unsupported center_method")
    camera_count = _strict_int(
        profile.get("camera_count"), "spherical profile camera_count"
    )
    if method == CAMERA_CENTER_METHOD and camera_count == 0:
        raise ValueError("camera center_method requires at least one camera")
    if method == BBOX_CENTER_METHOD and camera_count != 0:
        raise ValueError("bbox center_method requires camera_count == 0")

    radius_scale = _float32_scalar(
        profile.get("radius_scale"), "spherical profile radius_scale", positive=True
    )
    radius_epsilon = _float32_scalar(
        profile.get("radius_epsilon"), "spherical profile radius_epsilon", positive=True
    )
    tolerance = _float32_scalar(
        profile.get("weiszfeld_tolerance"),
        "spherical profile weiszfeld_tolerance",
        positive=True,
    )
    maximum = _strict_int(
        profile.get("weiszfeld_max_iterations"),
        "spherical profile weiszfeld_max_iterations",
        minimum=1,
    )
    iterations = _strict_int(
        profile.get("weiszfeld_iterations"),
        "spherical profile weiszfeld_iterations",
    )
    if iterations > maximum:
        raise ValueError("spherical profile Weiszfeld iterations exceed its limit")
    if method == BBOX_CENTER_METHOD and iterations != 0:
        raise ValueError("bbox center_method must not report Weiszfeld iterations")

    if profile.get("radius_scale_method") != RADIUS_SCALE_METHOD:
        raise ValueError("spherical profile has an unsupported radius_scale_method")
    if profile.get("direction_transform") != DIRECTION_TRANSFORM:
        raise ValueError("spherical profile has an unsupported direction_transform")
    if profile.get("log_radius_transform") != LOG_RADIUS_TRANSFORM:
        raise ValueError("spherical profile has an unsupported log_radius_transform")

    normalized: dict[str, Any] = {
        "format": SPHERICAL_PROFILE_FORMAT,
        "dtype": SPHERICAL_PROFILE_DTYPE,
        "center": center,
        "center_method": method,
        "camera_count": camera_count,
        "radius_scale": radius_scale,
        "radius_scale_method": RADIUS_SCALE_METHOD,
        "radius_epsilon": radius_epsilon,
        "direction_transform": DIRECTION_TRANSFORM,
        "log_radius_transform": LOG_RADIUS_TRANSFORM,
        "weiszfeld_tolerance": tolerance,
        "weiszfeld_max_iterations": maximum,
        "weiszfeld_iterations": iterations,
    }
    try:
        json.dumps(normalized, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("spherical profile is not strict JSON") from error
    return normalized


def spherical_features(
    xyz: Any,
    profile: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return translation-invariant unit directions and log-radius channels.

    Directions are ``(xyz - center) / ||xyz - center||``.  A point at the
    centre receives the deterministic zero vector.  The second result has
    shape ``(N, 1)`` and stores ``log1p(radius / radius_scale)``, which is zero
    and finite at the centre.
    """

    geometry = _point_array(xyz, "xyz")
    metadata = validate_spherical_profile(profile)
    center = np.asarray(metadata["center"], dtype=np.float64)
    scale = float(metadata["radius_scale"])
    epsilon = float(metadata["radius_epsilon"])

    offsets = geometry - center
    radii = np.hypot(np.hypot(offsets[:, 0], offsets[:, 1]), offsets[:, 2])
    if not np.isfinite(radii).all():
        raise ValueError("geometry radii are outside the supported numeric range")
    directions = np.zeros_like(offsets, dtype=np.float64)
    noncentral = radii > epsilon * scale
    directions[noncentral] = offsets[noncentral] / radii[noncentral, None]
    log_radius = np.log1p(radii / scale)
    if not np.isfinite(directions).all() or not np.isfinite(log_radius).all():
        raise ValueError("spherical features are outside the supported numeric range")
    return (
        np.ascontiguousarray(directions, dtype=np.float32),
        np.ascontiguousarray(log_radius[:, None], dtype=np.float32),
    )


def save_spherical_profile(path: str | Path, profile: Any) -> None:
    """Write a validated profile as portable strict JSON."""

    normalized = validate_spherical_profile(profile)
    Path(path).write_text(
        json.dumps(normalized, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_spherical_profile(path: str | Path) -> dict[str, Any]:
    """Load a profile and reject malformed JSON or non-canonical metadata."""

    profile_path = Path(path)
    try:
        value = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read valid spherical profile from {profile_path}") from error
    return validate_spherical_profile(value)
