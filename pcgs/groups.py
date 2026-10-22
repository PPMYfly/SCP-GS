"""Pure-CPU semantic attribute grouping contract for Phase8.6.

The contract deliberately contains no codec or model dependency.  It defines
one canonical causal order over the 56 stored Gaussian attributes and packs
only rows whose SH level activates a group.  Point order is never changed, so
the decoder can reconstruct row selection from the same SH-level vector.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


ATTRIBUTE_DIM = 56
GROUP_SCHEMA_FORMAT = "phase8.6_semantic_groups_v1"


@dataclass(frozen=True)
class GroupSpec:
    """One contiguous source slice in the canonical semantic coding order."""

    name: str
    start: int
    stop: int
    min_sh_level: int | None

    @property
    def width(self) -> int:
        return self.stop - self.start

    @property
    def source_slice(self) -> slice:
        return slice(self.start, self.stop)


GROUP_LAYOUT = (
    GroupSpec("dc", 0, 3, None),
    GroupSpec("opacity_scale", 48, 52, None),
    GroupSpec("rotation", 52, 56, None),
    GroupSpec("sh_d1", 3, 12, 1),
    GroupSpec("sh_d2", 12, 27, 2),
    GroupSpec("sh_d3", 27, 48, 3),
)
SEMANTIC_GROUPS = GROUP_LAYOUT
GROUP_ORDER = tuple(spec.name for spec in GROUP_LAYOUT)

_LAYOUT_KEYS = {"name", "start", "stop", "min_sh_level"}
_SCHEMA_KEYS = {"format", "point_count", "attribute_dim", "dtype", "groups"}
_SCHEMA_GROUP_KEYS = {
    "name",
    "start",
    "stop",
    "width",
    "min_sh_level",
    "active_count",
}


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _normalize_layout_entry(value: Any, index: int) -> GroupSpec:
    if isinstance(value, GroupSpec):
        spec = value
    elif isinstance(value, Mapping):
        if set(value) != _LAYOUT_KEYS:
            raise ValueError(f"group layout entry {index} has an invalid schema")
        name = value.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"group layout entry {index} name must be a non-empty string")
        start = _strict_int(value.get("start"), f"group layout entry {index} start")
        stop = _strict_int(value.get("stop"), f"group layout entry {index} stop")
        raw_level = value.get("min_sh_level")
        min_sh_level = (
            None
            if raw_level is None
            else _strict_int(raw_level, f"group layout entry {index} min_sh_level")
        )
        spec = GroupSpec(name, start, stop, min_sh_level)
    else:
        raise ValueError(f"group layout entry {index} must be a GroupSpec or mapping")

    if not isinstance(spec.name, str) or not spec.name:
        raise ValueError(f"group layout entry {index} name must be a non-empty string")
    start = _strict_int(spec.start, f"group layout entry {index} start")
    stop = _strict_int(spec.stop, f"group layout entry {index} stop")
    if start >= stop or stop > ATTRIBUTE_DIM:
        raise ValueError(
            f"group layout entry {index} range [{start}, {stop}) is outside 0:{ATTRIBUTE_DIM}"
        )
    if spec.min_sh_level is not None:
        level = _strict_int(
            spec.min_sh_level, f"group layout entry {index} min_sh_level", minimum=1
        )
        if level > 3:
            raise ValueError("SH group min_sh_level must be in [1, 3]")
    return GroupSpec(spec.name, start, stop, spec.min_sh_level)


def validate_group_layout(
    layout: Sequence[GroupSpec | Mapping[str, Any]] = GROUP_LAYOUT,
) -> tuple[GroupSpec, ...]:
    """Validate exact order, ranges, SH thresholds, and 56-channel coverage."""

    if isinstance(layout, (str, bytes)) or not isinstance(layout, Sequence):
        raise ValueError("group layout must be a sequence")
    if len(layout) != len(GROUP_LAYOUT):
        raise ValueError(f"group layout must contain exactly {len(GROUP_LAYOUT)} groups")
    normalized = tuple(_normalize_layout_entry(value, index) for index, value in enumerate(layout))

    names = [spec.name for spec in normalized]
    if len(set(names)) != len(names):
        raise ValueError("group layout names must be unique")
    coverage = np.zeros(ATTRIBUTE_DIM, dtype=np.int8)
    for spec in normalized:
        coverage[spec.start : spec.stop] += 1
    missing = np.flatnonzero(coverage == 0)
    overlap = np.flatnonzero(coverage > 1)
    if missing.size or overlap.size:
        raise ValueError(
            "group layout must cover every one of the 56 channels exactly once; "
            f"missing={missing.tolist()}, overlap={overlap.tolist()}"
        )
    if normalized != GROUP_LAYOUT:
        expected = [
            (spec.name, spec.start, spec.stop, spec.min_sh_level) for spec in GROUP_LAYOUT
        ]
        actual = [
            (spec.name, spec.start, spec.stop, spec.min_sh_level) for spec in normalized
        ]
        raise ValueError(
            f"group layout must match the canonical fixed order; expected={expected}, actual={actual}"
        )
    return normalized


def _validate_levels(sh_levels: Any, *, point_count: int | None = None) -> np.ndarray:
    levels = np.asarray(sh_levels)
    if levels.ndim != 1 or not np.issubdtype(levels.dtype, np.integer):
        raise ValueError("SH levels must be a one-dimensional integer vector")
    if levels.size == 0:
        raise ValueError("SH levels must contain at least one point")
    if point_count is not None and levels.size != point_count:
        raise ValueError(
            f"SH level count {levels.size} does not match point_count {point_count}"
        )
    if np.any((levels < 0) | (levels > 3)):
        raise ValueError("SH levels must be in [0, 3]")
    return levels.astype(np.int8, copy=False)


def group_active_mask(
    sh_levels: int | np.integer | Sequence[int] | np.ndarray,
    *,
    layout: Sequence[GroupSpec | Mapping[str, Any]] = GROUP_LAYOUT,
) -> np.ndarray:
    """Return active semantic groups for SH levels 0..3.

    A scalar level returns shape ``(6,)``.  A one-dimensional level vector
    returns shape ``(N, 6)``.  The first three base groups are always active;
    ``sh_d1``, ``sh_d2``, and ``sh_d3`` require levels 1, 2, and 3.
    """

    specs = validate_group_layout(layout)
    raw = np.asarray(sh_levels)
    scalar = raw.ndim == 0
    if scalar:
        if isinstance(sh_levels, bool) or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("SH level must be an integer in [0, 3]")
        values = raw.reshape(1)
    else:
        values = raw
    levels = _validate_levels(values)
    result = np.empty((levels.size, len(specs)), dtype=bool)
    for column, spec in enumerate(specs):
        result[:, column] = (
            True if spec.min_sh_level is None else levels >= spec.min_sh_level
        )
    return result[0] if scalar else result


active_group_mask = group_active_mask


def _validate_attribute_array(attributes: Any, *, point_count: int | None = None) -> np.ndarray:
    if not isinstance(attributes, np.ndarray):
        raise ValueError("attributes must be a NumPy array")
    if attributes.ndim != 2 or attributes.shape[1] != ATTRIBUTE_DIM:
        raise ValueError(f"attributes must have shape (N, {ATTRIBUTE_DIM})")
    if attributes.shape[0] == 0:
        raise ValueError("attributes must contain at least one point")
    if point_count is not None and attributes.shape[0] != point_count:
        raise ValueError("attribute point count does not match the schema")
    dtype = attributes.dtype
    if not (
        np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.floating)
    ) or np.issubdtype(dtype, np.bool_):
        raise ValueError("attribute dtype must be a real integer or floating type")
    if not np.isfinite(attributes).all():
        raise ValueError("attributes must contain only finite values")
    return attributes


def _canonical_dtype(value: Any) -> np.dtype[Any]:
    if not isinstance(value, str):
        raise ValueError("group schema dtype must be a canonical NumPy dtype string")
    try:
        dtype = np.dtype(value)
    except TypeError as error:
        raise ValueError("group schema dtype is invalid") from error
    if dtype.str != value:
        raise ValueError(
            f"group schema dtype must use canonical spelling {dtype.str!r}, got {value!r}"
        )
    if not (
        np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.floating)
    ) or np.issubdtype(dtype, np.bool_):
        raise ValueError("group schema dtype must be a real integer or floating type")
    return dtype


def _expected_active_counts(levels: np.ndarray, specs: tuple[GroupSpec, ...]) -> list[int]:
    masks = group_active_mask(levels, layout=specs)
    return [int(masks[:, index].sum()) for index in range(len(specs))]


def validate_group_schema(
    schema: Any,
    *,
    sh_levels: Sequence[int] | np.ndarray | None = None,
    point_count: int | None = None,
    payloads: Mapping[str, np.ndarray] | None = None,
    layout: Sequence[GroupSpec | Mapping[str, Any]] = GROUP_LAYOUT,
) -> dict[str, Any]:
    """Strictly validate schema fields and optionally payload inventory/data."""

    specs = validate_group_layout(layout)
    if not isinstance(schema, dict) or set(schema) != _SCHEMA_KEYS:
        raise ValueError("group schema has invalid top-level fields")
    if schema.get("format") != GROUP_SCHEMA_FORMAT:
        raise ValueError("missing or unsupported semantic group schema format")
    encoded_points = _strict_int(schema.get("point_count"), "group schema point_count", minimum=1)
    if point_count is not None:
        expected_points = _strict_int(point_count, "point_count", minimum=1)
        if encoded_points != expected_points:
            raise ValueError("group schema point_count does not match the expected point count")
    if _strict_int(schema.get("attribute_dim"), "group schema attribute_dim") != ATTRIBUTE_DIM:
        raise ValueError(f"group schema attribute_dim must be {ATTRIBUTE_DIM}")
    dtype = _canonical_dtype(schema.get("dtype"))

    levels = None
    if sh_levels is not None:
        levels = _validate_levels(sh_levels, point_count=encoded_points)
        expected_counts = _expected_active_counts(levels, specs)
    else:
        expected_counts = []

    groups = schema.get("groups")
    if not isinstance(groups, list) or len(groups) != len(specs):
        raise ValueError(f"group schema must contain exactly {len(specs)} ordered groups")
    normalized_groups: list[dict[str, Any]] = []
    active_counts: list[int] = []
    for index, (entry, spec) in enumerate(zip(groups, specs, strict=True)):
        if not isinstance(entry, dict) or set(entry) != _SCHEMA_GROUP_KEYS:
            raise ValueError(f"group schema entry {index} has invalid fields")
        exact_values = {
            "name": spec.name,
            "start": spec.start,
            "stop": spec.stop,
            "width": spec.width,
            "min_sh_level": spec.min_sh_level,
        }
        for field, expected in exact_values.items():
            if entry.get(field) != expected or (
                field in {"start", "stop", "width"} and isinstance(entry.get(field), bool)
            ):
                raise ValueError(
                    f"group schema entry {index} field {field!r} must be {expected!r}"
                )
        active_count = _strict_int(
            entry.get("active_count"), f"group schema entry {index} active_count"
        )
        if active_count > encoded_points:
            raise ValueError(f"group schema entry {index} active_count exceeds point_count")
        if spec.min_sh_level is None and active_count != encoded_points:
            raise ValueError(f"base group {spec.name} must be active for every point")
        if levels is not None and active_count != expected_counts[index]:
            raise ValueError(
                f"group {spec.name} active_count {active_count} does not match SH levels "
                f"({expected_counts[index]})"
            )
        active_counts.append(active_count)
        normalized_groups.append(
            {
                **exact_values,
                "active_count": active_count,
            }
        )
    if not (
        active_counts[3] >= active_counts[4] >= active_counts[5]
    ):
        raise ValueError("SH group active counts must be nested: sh_d1 >= sh_d2 >= sh_d3")

    normalized = {
        "format": GROUP_SCHEMA_FORMAT,
        "point_count": encoded_points,
        "attribute_dim": ATTRIBUTE_DIM,
        "dtype": dtype.str,
        "groups": normalized_groups,
    }
    if payloads is not None:
        _validate_payloads(payloads, normalized, dtype)
    return normalized


def _validate_payloads(
    payloads: Any,
    schema: dict[str, Any],
    dtype: np.dtype[Any],
) -> None:
    if not isinstance(payloads, Mapping):
        raise ValueError("group payloads must be a mapping")
    actual_names = set(payloads)
    expected_names = set(GROUP_ORDER)
    if actual_names != expected_names or any(not isinstance(name, str) for name in payloads):
        raise ValueError(
            "group payload inventory mismatch: "
            f"expected={list(GROUP_ORDER)}, actual={sorted(str(name) for name in actual_names)}"
        )
    for entry in schema["groups"]:
        name = entry["name"]
        payload = payloads[name]
        if not isinstance(payload, np.ndarray):
            raise ValueError(f"group payload {name} must be a NumPy array")
        expected_shape = (entry["active_count"], entry["width"])
        if payload.shape != expected_shape:
            raise ValueError(
                f"group payload {name} must have shape {expected_shape}, got {payload.shape}"
            )
        if payload.dtype != dtype:
            raise ValueError(
                f"group payload {name} dtype {payload.dtype.str!r} does not match "
                f"schema dtype {dtype.str!r}"
            )
        if not np.isfinite(payload).all():
            raise ValueError(f"group payload {name} must contain only finite values")


def pack_groups(
    attributes: np.ndarray,
    sh_levels: Sequence[int] | np.ndarray,
    *,
    layout: Sequence[GroupSpec | Mapping[str, Any]] = GROUP_LAYOUT,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Pack attributes into canonical semantic groups in stable point order."""

    specs = validate_group_layout(layout)
    values = _validate_attribute_array(attributes)
    levels = _validate_levels(sh_levels, point_count=values.shape[0])
    masks = group_active_mask(levels, layout=specs)
    payloads: dict[str, np.ndarray] = {}
    groups: list[dict[str, Any]] = []
    for index, spec in enumerate(specs):
        selected = masks[:, index]
        payloads[spec.name] = np.ascontiguousarray(
            values[selected, spec.source_slice]
        )
        groups.append(
            {
                "name": spec.name,
                "start": spec.start,
                "stop": spec.stop,
                "width": spec.width,
                "min_sh_level": spec.min_sh_level,
                "active_count": int(selected.sum()),
            }
        )
    schema = {
        "format": GROUP_SCHEMA_FORMAT,
        "point_count": int(values.shape[0]),
        "attribute_dim": ATTRIBUTE_DIM,
        "dtype": values.dtype.str,
        "groups": groups,
    }
    normalized = validate_group_schema(
        schema,
        sh_levels=levels,
        payloads=payloads,
        layout=specs,
    )
    return normalized, payloads


def unpack_groups(
    schema: Any,
    payloads: Mapping[str, np.ndarray],
    sh_levels: Sequence[int] | np.ndarray,
    *,
    layout: Sequence[GroupSpec | Mapping[str, Any]] = GROUP_LAYOUT,
) -> np.ndarray:
    """Reconstruct the 56 attributes; inactive SH channels are always zero."""

    specs = validate_group_layout(layout)
    normalized = validate_group_schema(
        schema,
        sh_levels=sh_levels,
        payloads=payloads,
        layout=specs,
    )
    levels = _validate_levels(sh_levels, point_count=normalized["point_count"])
    masks = group_active_mask(levels, layout=specs)
    result = np.zeros(
        (normalized["point_count"], ATTRIBUTE_DIM),
        dtype=np.dtype(normalized["dtype"]),
    )
    for index, spec in enumerate(specs):
        result[masks[:, index], spec.source_slice] = payloads[spec.name]
    return result


pack_semantic_groups = pack_groups
unpack_semantic_groups = unpack_groups


__all__ = [
    "ATTRIBUTE_DIM",
    "GROUP_LAYOUT",
    "GROUP_ORDER",
    "GROUP_SCHEMA_FORMAT",
    "SEMANTIC_GROUPS",
    "GroupSpec",
    "active_group_mask",
    "group_active_mask",
    "pack_groups",
    "pack_semantic_groups",
    "unpack_groups",
    "unpack_semantic_groups",
    "validate_group_layout",
    "validate_group_schema",
]

