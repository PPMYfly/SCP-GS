"""Pure-CPU planning and reconstruction for hierarchical SH-level streams."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


SH_HIERARCHY_FORMAT = "phase8.5_conditional_sh_levels_v1"
SH_STAGE_FILES = {degree: f"sh_ge_{degree}.b" for degree in range(1, 4)}
SH_STAGE_CONDITIONS = {
    1: "all_points",
    2: "sh_ge_1",
    3: "sh_ge_2",
}
_STAGE_KEYS = {
    "degree",
    "condition",
    "mode",
    "symbol_count",
    "ones_count",
    "file",
    "constant",
}


def _strict_nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _binary_vector(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional binary vector")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must contain only 0 and 1")
    return array.astype(np.uint8, copy=False)


def levels_to_nested_masks(levels: np.ndarray) -> dict[int, np.ndarray]:
    """Convert integer levels 0..3 into the three nested full-point masks."""

    values = np.asarray(levels)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("SH levels must be a one-dimensional integer vector")
    if values.size == 0:
        raise ValueError("SH levels must contain at least one point")
    if np.any((values < 0) | (values > 3)):
        raise ValueError("SH levels must be in [0, 3]")
    return {
        degree: (values >= degree).astype(np.uint8)
        for degree in range(1, 4)
    }


def nested_masks_to_levels(
    sh_ge_1: np.ndarray,
    sh_ge_2: np.ndarray,
    sh_ge_3: np.ndarray,
) -> np.ndarray:
    """Validate mask nesting and reconstruct integer SH levels."""

    masks = [
        _binary_vector(sh_ge_1, "sh_ge_1"),
        _binary_vector(sh_ge_2, "sh_ge_2"),
        _binary_vector(sh_ge_3, "sh_ge_3"),
    ]
    if len({mask.shape[0] for mask in masks}) != 1:
        raise ValueError("Hierarchical SH masks must have equal lengths")
    if np.any(masks[1] > masks[0]) or np.any(masks[2] > masks[1]):
        raise ValueError("Hierarchical SH masks must satisfy sh_ge_3 <= sh_ge_2 <= sh_ge_1")
    return (masks[0] + masks[1] + masks[2]).astype(np.int32)


def build_hierarchical_sh_stream(
    levels: np.ndarray,
) -> tuple[dict[str, Any], dict[int, np.ndarray]]:
    """Plan conditional symbols and omit arithmetic streams for constants.

    Degree one is selected over every point.  Degree two is selected only at
    points whose degree-one mask is active, and degree three only at points
    whose degree-two mask is active.  The returned payload mapping contains
    entries only for genuinely mixed binary stages that require arithmetic
    coding.
    """

    masks = levels_to_nested_masks(levels)
    point_count = int(np.asarray(levels).shape[0])
    parent = np.ones(point_count, dtype=bool)
    stages: list[dict[str, Any]] = []
    payloads: dict[int, np.ndarray] = {}
    for degree in range(1, 4):
        selected = np.ascontiguousarray(masks[degree][parent], dtype=np.uint8)
        symbol_count = int(selected.size)
        ones_count = int(selected.sum())
        if symbol_count == 0 or ones_count in (0, symbol_count):
            mode = "constant"
            constant = 1 if symbol_count and ones_count == symbol_count else 0
            file_name = None
        else:
            mode = "arithmetic"
            constant = None
            file_name = SH_STAGE_FILES[degree]
            payloads[degree] = selected
        stages.append(
            {
                "degree": degree,
                "condition": SH_STAGE_CONDITIONS[degree],
                "mode": mode,
                "symbol_count": symbol_count,
                "ones_count": ones_count,
                "file": file_name,
                "constant": constant,
            }
        )
        parent = masks[degree].astype(bool, copy=False)
    metadata = {
        "format": SH_HIERARCHY_FORMAT,
        "point_count": point_count,
        "stages": stages,
    }
    # Keep encoder-produced metadata subject to the same contract as external
    # manifests rather than relying on construction by convention.
    validate_hierarchical_sh_metadata(metadata, point_count=point_count)
    return metadata, payloads


def validate_hierarchical_sh_metadata(
    metadata: Any,
    *,
    point_count: int | None = None,
    payload_files: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Strictly validate hierarchy counts, modes, constants and filenames."""

    if not isinstance(metadata, dict) or metadata.get("format") != SH_HIERARCHY_FORMAT:
        raise ValueError("Missing or unsupported hierarchical SH stream metadata")
    encoded_point_count = _strict_nonnegative_int(
        metadata.get("point_count"), "SH hierarchy point_count"
    )
    if encoded_point_count == 0:
        raise ValueError("SH hierarchy point_count must be positive")
    if point_count is not None and encoded_point_count != int(point_count):
        raise ValueError("SH hierarchy point_count does not match the scene")
    stages = metadata.get("stages")
    if not isinstance(stages, list) or len(stages) != 3:
        raise ValueError("SH hierarchy must contain exactly three stages")

    expected_symbol_count = encoded_point_count
    expected_files: set[str] = set()
    normalized_stages: list[dict[str, Any]] = []
    for degree, stage in enumerate(stages, start=1):
        if not isinstance(stage, dict) or set(stage) != _STAGE_KEYS:
            raise ValueError(f"SH hierarchy stage {degree} has an invalid schema")
        if _strict_nonnegative_int(stage.get("degree"), "SH stage degree") != degree:
            raise ValueError("SH hierarchy stages must be ordered as degrees 1, 2, 3")
        if stage.get("condition") != SH_STAGE_CONDITIONS[degree]:
            raise ValueError(f"SH hierarchy stage {degree} has an invalid condition")
        symbol_count = _strict_nonnegative_int(
            stage.get("symbol_count"), f"SH stage {degree} symbol_count"
        )
        ones_count = _strict_nonnegative_int(
            stage.get("ones_count"), f"SH stage {degree} ones_count"
        )
        if symbol_count != expected_symbol_count:
            raise ValueError(
                f"SH stage {degree} symbol_count {symbol_count} does not match "
                f"its parent-active count {expected_symbol_count}"
            )
        if ones_count > symbol_count:
            raise ValueError(f"SH stage {degree} ones_count exceeds symbol_count")

        mode = stage.get("mode")
        file_name = stage.get("file")
        constant = stage.get("constant")
        if mode == "constant":
            if isinstance(constant, bool) or constant not in (0, 1) or file_name is not None:
                raise ValueError(f"SH constant stage {degree} has invalid fields")
            if ones_count != int(constant) * symbol_count:
                raise ValueError(f"SH constant stage {degree} has an inconsistent ones_count")
        elif mode == "arithmetic":
            if constant is not None or file_name != SH_STAGE_FILES[degree]:
                raise ValueError(f"SH arithmetic stage {degree} has invalid fields")
            if symbol_count == 0 or ones_count in (0, symbol_count):
                raise ValueError(
                    f"SH arithmetic stage {degree} is constant and must not use the coder"
                )
            expected_files.add(file_name)
        else:
            raise ValueError(f"SH hierarchy stage {degree} has an unsupported mode")
        normalized_stages.append(dict(stage))
        expected_symbol_count = ones_count

    if payload_files is not None:
        actual_files = {str(name) for name in payload_files if str(name).startswith("sh_ge_")}
        if actual_files != expected_files:
            raise ValueError(
                "Hierarchical SH payload inventory mismatch: "
                f"expected={sorted(expected_files)}, actual={sorted(actual_files)}"
            )
    return {
        "format": SH_HIERARCHY_FORMAT,
        "point_count": encoded_point_count,
        "stages": normalized_stages,
    }


def reconstruct_hierarchical_sh_levels(
    metadata: Any,
    coded_payloads: Mapping[int, np.ndarray],
) -> np.ndarray:
    """Expand conditional stage symbols and return validated levels 0..3."""

    normalized = validate_hierarchical_sh_metadata(metadata)
    payload_degrees: set[int] = set()
    for key in coded_payloads:
        if isinstance(key, bool) or not isinstance(key, (int, np.integer)):
            raise ValueError("Hierarchical SH payload keys must be integer degrees")
        payload_degrees.add(int(key))
    expected_payload_degrees = {
        int(stage["degree"])
        for stage in normalized["stages"]
        if stage["mode"] == "arithmetic"
    }
    if payload_degrees != expected_payload_degrees:
        raise ValueError(
            "Hierarchical SH coded payload set mismatch: "
            f"expected={sorted(expected_payload_degrees)}, actual={sorted(payload_degrees)}"
        )

    point_count = int(normalized["point_count"])
    parent = np.ones(point_count, dtype=bool)
    full_masks: dict[int, np.ndarray] = {}
    for stage in normalized["stages"]:
        degree = int(stage["degree"])
        symbol_count = int(stage["symbol_count"])
        if int(parent.sum()) != symbol_count:
            raise ValueError(f"SH stage {degree} parent selection does not match its metadata")
        if stage["mode"] == "constant":
            selected = np.full(symbol_count, int(stage["constant"]), dtype=np.uint8)
        else:
            selected = _binary_vector(coded_payloads[degree], f"sh_ge_{degree} payload")
            if selected.size != symbol_count:
                raise ValueError(f"SH stage {degree} payload length does not match symbol_count")
        if int(selected.sum()) != int(stage["ones_count"]):
            raise ValueError(f"SH stage {degree} payload ones_count mismatch")
        full = np.zeros(point_count, dtype=np.uint8)
        full[parent] = selected
        full_masks[degree] = full
        parent = full.astype(bool, copy=False)

    return nested_masks_to_levels(full_masks[1], full_masks[2], full_masks[3])


__all__ = [
    "SH_HIERARCHY_FORMAT",
    "SH_STAGE_FILES",
    "build_hierarchical_sh_stream",
    "levels_to_nested_masks",
    "nested_masks_to_levels",
    "reconstruct_hierarchical_sh_levels",
    "validate_hierarchical_sh_metadata",
]
