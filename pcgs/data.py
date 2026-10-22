"""Phase6.4 frozen-package I/O with no dependency on earlier project code."""

from __future__ import annotations

import json
import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ATTRIBUTE_LAYOUT = (
    ("features_dc", 3),
    ("features_rest_target", 45),
    ("opacity", 1),
    ("scaling", 3),
    ("rotation", 4),
)
ATTRIBUTE_DIM = sum(width for _, width in ATTRIBUTE_LAYOUT)
COLLISION_MAGIC = b"P83C"
COLLISION_VERSION = 1


def _morton_split_by_3(values: np.ndarray) -> np.ndarray:
    """Expand 21-bit integers so three coordinates can be interleaved."""

    values = np.asarray(values, dtype=np.uint64) & np.uint64(0x1FFFFF)
    values = (values | (values << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
    values = (values | (values << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
    values = (values | (values << np.uint64(8))) & np.uint64(0x100F00F00F00F00F)
    values = (values | (values << np.uint64(4))) & np.uint64(0x10C30C30C30C30C3)
    values = (values | (values << np.uint64(2))) & np.uint64(0x1249249249249249)
    return values


def canonical_morton_order(xyz_int: np.ndarray) -> np.ndarray:
    """Return a stable, overflow-safe 3D Morton order for <=21-bit coordinates."""

    xyz_int = np.asarray(xyz_int)
    if xyz_int.ndim != 2 or xyz_int.shape[1] != 3:
        raise ValueError(f"Expected Nx3 geometry, got {xyz_int.shape}")
    if xyz_int.size and (xyz_int.min() < 0 or xyz_int.max() >= (1 << 21)):
        raise ValueError("Morton ordering supports coordinates in [0, 2^21)")
    values = xyz_int.astype(np.uint64, copy=False)
    morton = (
        _morton_split_by_3(values[:, 0])
        | (_morton_split_by_3(values[:, 1]) << np.uint64(1))
        | (_morton_split_by_3(values[:, 2]) << np.uint64(2))
    )
    # Stable sorting makes the encoder deterministic for duplicate voxels.  At
    # decode time duplicate coordinate rows are interchangeable; the attribute
    # stream supplies their deterministic source order.
    return np.argsort(morton, kind="stable").astype(np.int64, copy=False)


def geometry_collision_stats(xyz_int: np.ndarray) -> dict[str, int]:
    """Summarize duplicate quantized positions for manifests and audits."""

    xyz_int = np.asarray(xyz_int)
    if xyz_int.shape[0] == 0:
        return {
            "groups": 0,
            "points": 0,
            "extra_points": 0,
            "max_multiplicity": 0,
        }
    _, counts = np.unique(xyz_int, axis=0, return_counts=True)
    duplicate = counts[counts > 1]
    return {
        "groups": int(duplicate.size),
        "points": int(duplicate.sum()) if duplicate.size else 0,
        "extra_points": int((duplicate - 1).sum()) if duplicate.size else 0,
        "max_multiplicity": int(duplicate.max()) if duplicate.size else 1,
    }


def _quantize_xyz(
    xyz: np.ndarray,
    xyz_min: np.ndarray,
    xyz_max: np.ndarray,
    geometry_bits: int,
) -> np.ndarray:
    extent = np.maximum(xyz_max - xyz_min, np.float32(1e-12))
    scale = float((1 << geometry_bits) - 1)
    return np.rint((xyz - xyz_min) / extent * scale).astype(np.int32)


def build_collision_sidecar(scene: "FrozenScene", geometry_bits: int) -> bytes:
    """Store exact source xyz only for positions that collide after q-geometry."""

    xyz_min = scene.xyz.min(axis=0).astype(np.float32)
    xyz_max = scene.xyz.max(axis=0).astype(np.float32)
    unsorted = _quantize_xyz(scene.xyz, xyz_min, xyz_max, geometry_bits)
    order = canonical_morton_order(unsorted)
    xyz_int = unsorted[order]
    xyz_raw = scene.xyz[order].astype(np.float32, copy=False)
    groups: list[tuple[np.ndarray, np.ndarray]] = []
    start = 0
    while start < xyz_int.shape[0]:
        end = start + 1
        while end < xyz_int.shape[0] and np.array_equal(xyz_int[end], xyz_int[start]):
            end += 1
        if end - start > 1:
            groups.append((xyz_int[start], xyz_raw[start:end]))
        start = end
    record_count = sum(values.shape[0] for _, values in groups)
    output = bytearray(struct.pack("<4sIII", COLLISION_MAGIC, COLLISION_VERSION, len(groups), record_count))
    for coordinate, values in groups:
        output.extend(
            struct.pack(
                "<iiiI",
                int(coordinate[0]),
                int(coordinate[1]),
                int(coordinate[2]),
                int(values.shape[0]),
            )
        )
        output.extend(np.ascontiguousarray(values, dtype="<f4").tobytes())
    return bytes(output)


def apply_collision_sidecar(
    xyz_int: np.ndarray,
    xyz: np.ndarray,
    payload: bytes,
    xyz_min: np.ndarray,
    xyz_max: np.ndarray,
    geometry_bits: int,
) -> np.ndarray:
    """Validate collision records and restore their exact source coordinates."""

    if len(payload) < 16:
        raise ValueError("Collision sidecar is truncated")
    magic, version, group_count, record_count = struct.unpack_from("<4sIII", payload, 0)
    if magic != COLLISION_MAGIC or version != COLLISION_VERSION:
        raise ValueError("Unsupported collision sidecar")
    duplicate_runs: list[tuple[int, int]] = []
    start = 0
    while start < xyz_int.shape[0]:
        end = start + 1
        while end < xyz_int.shape[0] and np.array_equal(xyz_int[end], xyz_int[start]):
            end += 1
        if end - start > 1:
            duplicate_runs.append((start, end))
        start = end
    if len(duplicate_runs) != group_count:
        raise ValueError("Collision sidecar group count does not match decoded geometry")
    result = np.asarray(xyz, dtype=np.float32).copy()
    cursor = 16
    consumed_records = 0
    for start, end in duplicate_runs:
        if cursor + 16 > len(payload):
            raise ValueError("Collision sidecar group header is truncated")
        qx, qy, qz, multiplicity = struct.unpack_from("<iiiI", payload, cursor)
        cursor += 16
        coordinate = np.asarray([qx, qy, qz], dtype=np.int32)
        if not np.array_equal(coordinate, xyz_int[start]) or multiplicity != end - start:
            raise ValueError("Collision sidecar coordinate or multiplicity mismatch")
        byte_count = int(multiplicity) * 3 * 4
        if cursor + byte_count > len(payload):
            raise ValueError("Collision sidecar xyz records are truncated")
        raw = np.frombuffer(payload, dtype="<f4", count=int(multiplicity) * 3, offset=cursor).reshape(-1, 3)
        cursor += byte_count
        requantized = _quantize_xyz(raw, xyz_min, xyz_max, geometry_bits)
        if not np.all(requantized == coordinate[None, :]):
            raise ValueError("Collision sidecar raw xyz does not map to its declared voxel")
        result[start:end] = raw
        consumed_records += int(multiplicity)
    if cursor != len(payload) or consumed_records != record_count:
        raise ValueError("Collision sidecar contains trailing or inconsistent records")
    return result


def canonicalize_decoded_attributes(attributes: np.ndarray) -> np.ndarray:
    """Return the exact finite float32 values representable by frozen float16 files."""

    values = np.asarray(attributes, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Decoded attributes contain NaN or infinity")
    limit = np.float32(np.finfo(np.float16).max)
    clipped = np.clip(values, -limit, limit)
    stored = np.ascontiguousarray(clipped, dtype="<f2")
    if not np.isfinite(stored).all():
        raise ValueError("Decoded attributes cannot be represented as finite float16")
    return stored.astype(np.float32)


def decoded_tensor_digest(
    xyz: np.ndarray,
    attributes: np.ndarray,
    sh_levels: np.ndarray,
    hard_flags: np.ndarray,
) -> str:
    """Hash the exact tensor dtypes written to a decoded frozen package."""

    digest = hashlib.sha256()
    digest.update(b"phase8.3-decoded-tensors-v1\0")
    tensors = (
        ("xyz", np.ascontiguousarray(xyz, dtype="<f4")),
        (
            "attributes",
            np.ascontiguousarray(canonicalize_decoded_attributes(attributes), dtype="<f2"),
        ),
        ("sh_levels", np.ascontiguousarray(sh_levels, dtype="<i4")),
        ("hard_flags", np.ascontiguousarray(hard_flags, dtype=np.uint8)),
    )
    for name, value in tensors:
        digest.update(name.encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def scene_fingerprint(scene: "FrozenScene") -> str:
    """Bind a checkpoint to the exact ordered tensors it was trained for."""

    digest = hashlib.sha256()
    digest.update(b"phase8.3-scene-fingerprint-v1\0")
    for name, value in (
        ("xyz", scene.xyz),
        ("attributes", scene.attributes),
        ("sh_levels", scene.sh_levels),
        ("hard_flags", scene.hard_flags),
    ):
        array = np.ascontiguousarray(value)
        digest.update(name.encode("ascii") + b"\0")
        digest.update(str(array.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _read(path: Path, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
    result = np.fromfile(path, dtype=dtype)
    expected = int(np.prod(shape))
    if result.size != expected:
        raise ValueError(f"{path} contains {result.size} values; expected {expected}")
    return result.reshape(shape)


def unpack_lsb_bits(path: Path, count: int) -> np.ndarray:
    """Decode the little-endian bit packing used by Phase6.4 bitpack_utils."""

    if not path.exists():
        return np.zeros(count, dtype=np.uint8)
    packed = np.fromfile(path, dtype=np.uint8)
    bits = ((packed[:, None] >> np.arange(8, dtype=np.uint8)) & 1).reshape(-1)
    if bits.size < count:
        raise ValueError(f"Packed flag file {path} is truncated")
    return bits[:count].astype(np.uint8, copy=False)


def pack_lsb_bits(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint8).reshape(-1) & 1
    packed = np.zeros((values.size + 7) // 8, dtype=np.uint8)
    for bit in range(8):
        part = values[bit::8]
        packed[: part.size] |= part << bit
    return packed


@dataclass
class FrozenScene:
    path: Path
    metadata: dict[str, Any]
    xyz: np.ndarray
    attributes: np.ndarray
    sh_levels: np.ndarray
    hard_flags: np.ndarray

    @property
    def num_points(self) -> int:
        return int(self.xyz.shape[0])


def load_frozen_scene(path: str | Path, max_points: int | None = None) -> FrozenScene:
    root = Path(path).expanduser().resolve()
    if root.is_file():
        root = root.parent
    metadata_path = root / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing Phase6.4 metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format") != "phase2_residual_field_frozen":
        raise ValueError(
            f"Expected format=phase2_residual_field_frozen, got {metadata.get('format')!r}"
        )
    count = int(metadata["num_points"])
    if count <= 0:
        raise ValueError("Phase6.4 input must contain at least one point")
    xyz = _read(root / "xyz.bin", np.float32, (count, 3)).astype(np.float32)
    arrays: list[np.ndarray] = []
    for name, width in ATTRIBUTE_LAYOUT:
        arrays.append(_read(root / f"{name}.bin", np.float16, (count, width)).astype(np.float32))
    attributes = np.concatenate(arrays, axis=1)
    sh_levels = _read(root / "sh_levels.bin", np.int32, (count,)).astype(np.int32)
    hard_flags = unpack_lsb_bits(root / "hard_point_flags.packed.bin", count)
    if not np.isfinite(xyz).all() or not np.isfinite(attributes).all():
        raise ValueError("Phase6.4 input contains NaN or infinite values")
    if np.any((sh_levels < 0) | (sh_levels > 3)):
        raise ValueError("Phase6.4 SH levels must be in [0, 3]")
    if np.any((hard_flags != 0) & (hard_flags != 1)):
        raise ValueError("Phase6.4 hard flags must be binary")
    if max_points is not None:
        if int(max_points) <= 0:
            raise ValueError("max_points must be positive")
        count = min(count, int(max_points))
        xyz = xyz[:count]
        attributes = attributes[:count]
        sh_levels = sh_levels[:count]
        hard_flags = hard_flags[:count]
    return FrozenScene(root, metadata, xyz, attributes, sh_levels, hard_flags)


def voxelize_and_sort(
    scene: FrozenScene,
    geometry_bits: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Use PanoCompactGS's integer-anchor ordering and return aligned scene tensors."""

    if not 12 <= geometry_bits <= 24:
        raise ValueError("geometry_bits must be in [12, 24]")
    if geometry_bits > 21:
        raise ValueError("Phase8.3 stable Morton ordering supports at most 21 geometry bits")
    xyz_min = scene.xyz.min(axis=0).astype(np.float32)
    xyz_max = scene.xyz.max(axis=0).astype(np.float32)
    xyz_int = _quantize_xyz(scene.xyz, xyz_min, xyz_max, geometry_bits)
    order = canonical_morton_order(xyz_int)
    return (
        xyz_int[order],
        scene.attributes[order],
        scene.sh_levels[order],
        scene.hard_flags[order],
        xyz_min,
        xyz_max,
    )


def dequantize_xyz(
    xyz_int: np.ndarray,
    xyz_min: np.ndarray,
    xyz_max: np.ndarray,
    geometry_bits: int,
) -> np.ndarray:
    scale = np.float32((1 << geometry_bits) - 1)
    return xyz_int.astype(np.float32) / scale * (xyz_max - xyz_min) + xyz_min


def split_attributes(attributes: np.ndarray) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    offset = 0
    for name, width in ATTRIBUTE_LAYOUT:
        result[name] = attributes[:, offset : offset + width]
        offset += width
    return result


def write_decoded_scene(
    output: str | Path,
    xyz: np.ndarray,
    attributes: np.ndarray,
    sh_levels: np.ndarray,
    source_metadata: dict[str, Any],
    codec_metadata: dict[str, Any] | None = None,
    hard_flags: np.ndarray | None = None,
) -> None:
    """Write the exact frozen-package files consumed by GaussianMapper::loadCompact."""

    root = Path(output).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    count = int(xyz.shape[0])
    xyz = np.asarray(xyz, dtype=np.float32)
    attributes = canonicalize_decoded_attributes(attributes)
    sh_levels = np.asarray(sh_levels, dtype=np.int32)
    if xyz.shape != (count, 3) or attributes.shape != (count, ATTRIBUTE_DIM):
        raise ValueError("Decoded xyz or attribute shape is invalid")
    if sh_levels.shape != (count,) or np.any((sh_levels < 0) | (sh_levels > 3)):
        raise ValueError("Decoded SH levels are invalid")
    if not np.isfinite(xyz).all():
        raise ValueError("Decoded xyz contains NaN or infinity")
    if hard_flags is not None:
        hard_flags = np.asarray(hard_flags, dtype=np.uint8)
        if hard_flags.shape != (count,) or np.any((hard_flags != 0) & (hard_flags != 1)):
            raise ValueError("Decoded hard flags are invalid")
    values = split_attributes(attributes)
    xyz_min = xyz.min(axis=0).astype(np.float32)
    xyz_max = xyz.max(axis=0).astype(np.float32)
    extent = np.maximum(xyz_max - xyz_min, np.float32(1e-12))
    xyz_normalized = ((xyz - xyz_min) / extent).astype(np.float16)

    xyz.tofile(root / "xyz.bin")
    xyz_normalized.tofile(root / "xyz_normalized.bin")
    for name, _ in ATTRIBUTE_LAYOUT:
        values[name].astype(np.float16).tofile(root / f"{name}.bin")
    np.asarray(sh_levels, dtype=np.int32).tofile(root / "sh_levels.bin")
    identity = np.arange(count, dtype=np.int64)
    identity.tofile(root / "morton_order.bin")
    identity.tofile(root / "inverse_morton_order.bin")
    if hard_flags is not None:
        pack_lsb_bits(hard_flags).tofile(root / "hard_point_flags.packed.bin")

    meta = {
        "format": "phase2_residual_field_frozen",
        "codec_source": "PanoCompactGS direct port",
        "topology_frozen": True,
        "sorted_by_morton": True,
        "normalized_xyz": True,
        "num_points": count,
        "num_blocks": 0,
        "max_sh_degree": int(source_metadata.get("max_sh_degree", 3)),
        "active_sh_degree": int(source_metadata.get("active_sh_degree", 3)),
        "scene_extent": float(source_metadata.get("scene_extent", 1.0)),
        "use_locality_base": False,
        "locality_high_sh_block_size": 64,
        "locality_low_sh_block_size": 128,
        "mask_features_rest_by_sh_level": True,
        "xyz_shape": [count, 3],
        "xyz_normalized_shape": [count, 3],
        "features_dc_shape": [count, 1, 3],
        "features_rest_base_shape": [count, 15, 3],
        "features_rest_target_shape": [count, 15, 3],
        "opacity_base_shape": [count, 1],
        "opacity_shape": [count, 1],
        "scaling_base_shape": [count, 3],
        "scaling_shape": [count, 3],
        "rotation_base_shape": [count, 4],
        "rotation_shape": [count, 4],
        "sh_levels_shape": [count],
        "morton_order_shape": [count],
        "inverse_morton_order_shape": [count],
        "xyz_bbox_min": xyz_min.tolist(),
        "xyz_bbox_max": xyz_max.tolist(),
        "hybrid_selector": {"enabled": False},
    }
    if codec_metadata is not None:
        meta["hacpp"] = codec_metadata
    (root / "metadata.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
