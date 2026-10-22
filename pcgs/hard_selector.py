"""Versioned per-Gaussian high-precision profiles for Phase8.8."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


FORMAT = "phase8.8_panoramic_hard_point_profile_v1"


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype=np.uint8).tobytes()).hexdigest()


def load_hard_profile(
    path: str | Path,
    *,
    expected_fingerprint: str,
    expected_points: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a source-bound binary precision mask without pickle execution."""

    profile_path = Path(path).expanduser().resolve(strict=True)
    if profile_path.stat().st_size > max(16 * 1024 * 1024, expected_points * 2 + 65536):
        raise ValueError("Hard-point profile exceeds the bounded canonical size")
    with np.load(profile_path, allow_pickle=False) as archive:
        if set(archive.files) != {"hard_flags", "metadata"}:
            raise ValueError("Hard-point profile must contain exactly hard_flags and metadata")
        flags = archive["hard_flags"]
        metadata_array = archive["metadata"]
        if metadata_array.shape != () or metadata_array.dtype.kind not in {"U", "S"}:
            raise ValueError("Hard-point profile metadata must be a scalar string")
        try:
            metadata = json.loads(str(metadata_array.item()))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("Hard-point profile metadata is invalid JSON") from error
    if not isinstance(metadata, dict) or metadata.get("format") != FORMAT:
        raise ValueError("Unsupported hard-point profile format")
    if metadata.get("scene_fingerprint") != expected_fingerprint:
        raise ValueError("Hard-point profile source fingerprint mismatch")
    if metadata.get("point_count") != int(expected_points):
        raise ValueError("Hard-point profile point-count metadata mismatch")
    if flags.dtype != np.uint8 or flags.shape != (expected_points,):
        raise ValueError("Hard-point flags must be an exact uint8 [N] array")
    if np.any((flags != 0) & (flags != 1)):
        raise ValueError("Hard-point flags must be binary")
    selected_count = int(flags.sum())
    if metadata.get("selected_count") != selected_count:
        raise ValueError("Hard-point profile selected_count mismatch")
    if metadata.get("hard_flags_sha256") != _sha256_array(flags):
        raise ValueError("Hard-point profile flag digest mismatch")
    objective = metadata.get("objective")
    if objective not in {"sphere", "planar", "phase6.4_original", "random_matched"}:
        raise ValueError("Hard-point profile objective label is invalid")
    return flags.copy(), metadata
