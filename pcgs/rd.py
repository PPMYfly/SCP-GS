"""Versioned Phase8.4 ERP-RD multiplier profiles.

The native ERP optimizer and the PanoCompactGS codec must describe the same quantizer.
Phase8.4 keeps the codec's 56 per-channel min/max centers and base steps, then
applies seven positive semantic-group multipliers.  Panorama hard points remain
a codec invariant with an additional fixed ``q * 0.5`` scale.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ATTRIBUTE_DIM = 56
HARD_Q_SCALE = 0.5
RD_PROFILE_FORMAT = "phase8.4_rd_multiplier_profile_v4"
BASE_QUANTIZATION_FORMAT = "phase8.4_per_channel_symmetric_minmax_v2"
CENTERED_CLAMP_CENTER = "zero"
CENTERED_SYMBOL_LIMIT = 15_000
MIN_GROUP_MULTIPLIER = 0.5
MAX_GROUP_MULTIPLIER = 4.0
MULTIPLIER_BOUNDS = [MIN_GROUP_MULTIPLIER, MAX_GROUP_MULTIPLIER]
ATTRIBUTE_GROUP_SLICES = {
    "sh_dc": slice(0, 3),
    "sh_d1": slice(3, 12),
    "sh_d2": slice(12, 27),
    "sh_d3": slice(27, 48),
    "opacity": slice(48, 49),
    "scale": slice(49, 52),
    "rotation": slice(52, 56),
}
ATTRIBUTE_GROUP_ORDER = tuple(ATTRIBUTE_GROUP_SLICES)


def _vector(value: Any, name: str, *, positive: bool = False) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (ATTRIBUTE_DIM,):
        raise ValueError(f"{name} must contain exactly {ATTRIBUTE_DIM} values")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    if positive and np.any(result <= 0):
        raise ValueError(f"{name} must contain only positive values")
    return result


def _same_base(reference: np.ndarray, recorded: np.ndarray) -> bool:
    # C++/LibTorch and NumPy both operate on float32 source attributes, but the
    # JSON decimal round-trip may differ at the final bit.  This tolerance is
    # tight enough to reject a profile produced for a different scene/q setting.
    return bool(np.allclose(reference, recorded, rtol=1e-6, atol=1e-8))


def load_rd_profile(
    path: str | Path,
    base_centers: np.ndarray,
    base_steps: np.ndarray,
    quant_bits: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Validate and apply a native ERP group-multiplier profile.

    Absolute group-step files from the initial Phase8.4 draft are deliberately
    rejected.  Accepting them would replace heterogeneous channel steps and no
    longer match the quantizer used by the arithmetic encoder/decoder.
    """

    profile_path = Path(path).expanduser().resolve()
    data = profile_path.read_bytes()
    values = json.loads(data)
    if not isinstance(values, dict):
        raise ValueError("RD profile root must be a JSON object")

    profile_format = values.get("format")
    if profile_format is None and any(name in values for name in ATTRIBUTE_GROUP_ORDER):
        raise ValueError(
            "Legacy absolute-step RD profiles are not supported; rerun the Phase8.4 "
            "native optimizer to produce a versioned multiplier profile"
        )
    if profile_format != RD_PROFILE_FORMAT:
        raise ValueError(
            f"Unsupported RD profile format {profile_format!r}; expected {RD_PROFILE_FORMAT!r}"
        )
    if values.get("attribute_dim") != ATTRIBUTE_DIM:
        raise ValueError(f"RD profile attribute_dim must equal {ATTRIBUTE_DIM}")
    if values.get("group_order") != list(ATTRIBUTE_GROUP_ORDER):
        raise ValueError("RD profile group_order does not match the Phase8.4 contract")

    try:
        recorded_hard_scale = float(values["hard_q_scale"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("RD profile hard_q_scale is missing or invalid") from error
    if not math.isclose(recorded_hard_scale, HARD_Q_SCALE, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"RD profile hard_q_scale must remain fixed at {HARD_Q_SCALE}, "
            f"got {recorded_hard_scale}"
        )
    if values.get("geometry_policy") != "fixed_by_hac_geometry_bits":
        raise ValueError("RD profile must leave geometry controlled by HAC geometry_bits")
    if values.get("centered_clamp_center") != CENTERED_CLAMP_CENTER:
        raise ValueError("RD profile must declare the zero-centered clamp contract")
    if values.get("centered_symbol_limit") != CENTERED_SYMBOL_LIMIT:
        raise ValueError(
            f"RD profile centered_symbol_limit must equal {CENTERED_SYMBOL_LIMIT}"
        )
    bounds = values.get("multiplier_bounds")
    if (
        not isinstance(bounds, list)
        or len(bounds) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in bounds
        )
        or not all(math.isfinite(float(value)) for value in bounds)
        or float(bounds[0]) != MIN_GROUP_MULTIPLIER
        or float(bounds[1]) != MAX_GROUP_MULTIPLIER
    ):
        raise ValueError(
            "RD profile multiplier_bounds must be exactly "
            f"[{MIN_GROUP_MULTIPLIER}, {MAX_GROUP_MULTIPLIER}]"
        )

    base = values.get("base_quantization")
    if not isinstance(base, dict):
        raise ValueError("RD profile base_quantization must be an object")
    if base.get("format") != BASE_QUANTIZATION_FORMAT:
        raise ValueError("RD profile uses an unsupported base quantization contract")
    if base.get("quant_bits") != int(quant_bits):
        raise ValueError(
            f"RD profile quant_bits mismatch: {base.get('quant_bits')!r} != {int(quant_bits)}"
        )

    local_centers = _vector(base_centers, "local base centers")
    local_steps = _vector(base_steps, "local base steps", positive=True)
    recorded_centers = _vector(base.get("centers"), "profile base centers")
    recorded_steps = _vector(base.get("steps"), "profile base steps", positive=True)
    if not _same_base(local_centers, recorded_centers):
        raise ValueError("RD profile base centers do not match this scene")
    if not _same_base(local_steps, recorded_steps):
        raise ValueError("RD profile base steps do not match this scene/quant_bits setting")

    multipliers = values.get("multipliers")
    if not isinstance(multipliers, dict):
        raise ValueError("RD profile multipliers must be an object")
    expected_groups = set(ATTRIBUTE_GROUP_ORDER)
    actual_groups = set(multipliers)
    missing = sorted(expected_groups - actual_groups)
    extra = sorted(actual_groups - expected_groups)
    if missing or extra:
        detail = []
        if missing:
            detail.append(f"missing: {', '.join(missing)}")
        if extra:
            detail.append(f"unexpected: {', '.join(extra)}")
        raise ValueError("RD multiplier groups do not match contract (" + "; ".join(detail) + ")")

    resolved_steps = local_steps.copy()
    resolved_multipliers: dict[str, float] = {}
    for name, selection in ATTRIBUTE_GROUP_SLICES.items():
        try:
            multiplier = float(multipliers[name])
        except (TypeError, ValueError) as error:
            raise ValueError(f"RD multiplier {name!r} is invalid") from error
        if (
            not math.isfinite(multiplier)
            or multiplier < MIN_GROUP_MULTIPLIER
            or multiplier > MAX_GROUP_MULTIPLIER
        ):
            raise ValueError(
                f"RD multiplier {name!r} must be finite and within "
                f"[{MIN_GROUP_MULTIPLIER}, {MAX_GROUP_MULTIPLIER}], got {multiplier}"
            )
        resolved_steps[selection] *= np.float32(multiplier)
        resolved_multipliers[name] = multiplier
    if not np.isfinite(resolved_steps).all() or np.any(resolved_steps <= 0):
        raise ValueError("RD multipliers produced invalid resolved channel steps")

    metadata = {
        "format": RD_PROFILE_FORMAT,
        "sha256": hashlib.sha256(data).hexdigest(),
        "base_quantization_format": BASE_QUANTIZATION_FORMAT,
        "quant_bits": int(quant_bits),
        "attribute_dim": ATTRIBUTE_DIM,
        "group_order": list(ATTRIBUTE_GROUP_ORDER),
        "multipliers": resolved_multipliers,
        "hard_q_scale": HARD_Q_SCALE,
        "centered_clamp_center": CENTERED_CLAMP_CENTER,
        "centered_symbol_limit": CENTERED_SYMBOL_LIMIT,
        "multiplier_bounds": list(MULTIPLIER_BOUNDS),
        "geometry_policy": "fixed_by_hac_geometry_bits",
        "quantizer": (
            "center_j + round(clamp(attribute_ij-center_j, -15000*q_ij, "
            "+15000*q_ij) / q_ij) * q_ij; "
            "q_ij=base_step_j*group_multiplier_j*hard_scale_i"
        ),
    }
    return resolved_steps, metadata
