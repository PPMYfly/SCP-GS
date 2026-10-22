"""Safe compact binary deployment state for Phase8.8.

The format is intentionally independent from the scene codec and entropy
model.  Floating tensors with two or more dimensions use symmetric INT8
quantization per output channel in ``int8`` mode.  Other floating tensors are
stored as FP16.  Integer and boolean buffers are stored losslessly.

All records are sorted by tensor name.  A SHA-256 digest covers the immutable
header fields and the complete body, and the reader validates the entire
schema before constructing any Torch tensor.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import re
import struct
import tempfile
from typing import Any

import numpy as np
import torch


MAGIC = b"H64DPL88"
VERSION = 1
MODE_INT8 = "int8"
MODE_FP16 = "fp16"

_MODE_TO_CODE = {MODE_INT8: 1, MODE_FP16: 2}
_CODE_TO_MODE = {value: key for key, value in _MODE_TO_CODE.items()}

# magic, version, mode, flags, tensor_count, body_nbytes
_HEADER_PREFIX_STRUCT = struct.Struct("<8sHBBIQ")
_HEADER_STRUCT = struct.Struct("<8sHBBIQ32s")
# name_nbytes, ndim, source_dtype, encoding, flags, data_nbytes, scale_nbytes
_RECORD_STRUCT = struct.Struct("<HBBBBQQ")
_DIM_STRUCT = struct.Struct("<Q")

_ENCODING_INT8_PER_OUTPUT = 1
_ENCODING_FP16 = 2
_ENCODING_RAW = 3

_MAX_TENSORS = 100_000
_MAX_NAME_BYTES = 1_024
_MAX_NDIM = 16
_MAX_DIM = (1 << 31) - 1
_MAX_TENSOR_ELEMENTS = (1 << 31) - 1
_MAX_FILE_BYTES = 16 * 1024 * 1024 * 1024
_NAME_PATTERN = re.compile(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*\Z")


class DeploymentFormatError(ValueError):
    """Raised when a deployment file violates the binary contract."""


@dataclass(frozen=True)
class _DTypeSpec:
    code: int
    name: str
    torch_dtype: torch.dtype
    numpy_dtype: str
    floating: bool

    @property
    def itemsize(self) -> int:
        return np.dtype(self.numpy_dtype).itemsize


_DTYPES = (
    _DTypeSpec(1, "float16", torch.float16, "<f2", True),
    _DTypeSpec(2, "float32", torch.float32, "<f4", True),
    _DTypeSpec(3, "float64", torch.float64, "<f8", True),
    _DTypeSpec(10, "int8", torch.int8, "|i1", False),
    _DTypeSpec(11, "uint8", torch.uint8, "|u1", False),
    _DTypeSpec(12, "int16", torch.int16, "<i2", False),
    _DTypeSpec(13, "int32", torch.int32, "<i4", False),
    _DTypeSpec(14, "int64", torch.int64, "<i8", False),
    _DTypeSpec(15, "bool", torch.bool, "|b1", False),
)
_DTYPE_BY_CODE = {spec.code: spec for spec in _DTYPES}
_DTYPE_BY_NAME = {spec.name: spec for spec in _DTYPES}
_DTYPE_BY_TORCH = {spec.torch_dtype: spec for spec in _DTYPES}

_SCHEMA_ENTRY_KEYS = {"shape", "dtype"}


@dataclass(frozen=True)
class _PreparedTensor:
    name: str
    shape: tuple[int, ...]
    dtype: _DTypeSpec
    tensor: torch.Tensor


@dataclass(frozen=True)
class _EncodedRecord:
    name: str
    shape: tuple[int, ...]
    dtype: _DTypeSpec
    encoding: int
    scales: bytes
    data: bytes


@dataclass(frozen=True)
class _ParsedRecord:
    name: str
    shape: tuple[int, ...]
    dtype: _DTypeSpec
    encoding: int
    scales: memoryview
    data: memoryview


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise DeploymentFormatError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise DeploymentFormatError(f"{name} must be >= {minimum}")
    return result


def _validate_name(name: Any) -> tuple[str, bytes]:
    if not isinstance(name, str) or not _NAME_PATTERN.fullmatch(name):
        raise DeploymentFormatError(
            "tensor name must contain only non-empty dot-separated ASCII "
            "alphanumeric or underscore segments"
        )
    encoded = name.encode("ascii")
    if len(encoded) > _MAX_NAME_BYTES:
        raise DeploymentFormatError(
            f"tensor name exceeds the {_MAX_NAME_BYTES}-byte limit"
        )
    return name, encoded


def _validate_shape(shape: Any, *, name: str) -> tuple[int, ...]:
    if not isinstance(shape, (list, tuple)):
        raise DeploymentFormatError(f"tensor {name} shape must be a list or tuple")
    if len(shape) > _MAX_NDIM:
        raise DeploymentFormatError(f"tensor {name} ndim exceeds {_MAX_NDIM}")
    normalized: list[int] = []
    for index, raw in enumerate(shape):
        dimension = _strict_int(raw, f"tensor {name} shape[{index}]", minimum=1)
        if dimension > _MAX_DIM:
            raise DeploymentFormatError(f"tensor {name} dimension exceeds {_MAX_DIM}")
        normalized.append(dimension)
    elements = math.prod(normalized)
    if elements > _MAX_TENSOR_ELEMENTS:
        raise DeploymentFormatError(
            f"tensor {name} exceeds {_MAX_TENSOR_ELEMENTS} elements"
        )
    return tuple(normalized)


def _prepare_state(state: Any) -> list[_PreparedTensor]:
    if not isinstance(state, Mapping) or not state:
        raise DeploymentFormatError("deployment state must be a non-empty mapping")
    prepared: list[_PreparedTensor] = []
    seen: set[str] = set()
    for raw_name, raw_tensor in state.items():
        name, _ = _validate_name(raw_name)
        if name in seen:
            raise DeploymentFormatError(f"duplicate tensor name: {name}")
        seen.add(name)
        if not isinstance(raw_tensor, torch.Tensor):
            raise DeploymentFormatError(f"deployment value {name} must be a Torch tensor")
        if raw_tensor.layout != torch.strided:
            raise DeploymentFormatError(f"tensor {name} must use strided layout")
        dtype = _DTYPE_BY_TORCH.get(raw_tensor.dtype)
        if dtype is None:
            raise DeploymentFormatError(
                f"tensor {name} has unsupported dtype {raw_tensor.dtype}"
            )
        shape = _validate_shape(list(raw_tensor.shape), name=name)
        tensor = raw_tensor.detach().cpu().contiguous()
        if dtype.floating and not torch.isfinite(tensor).all():
            raise DeploymentFormatError(f"tensor {name} must contain only finite values")
        prepared.append(_PreparedTensor(name, shape, dtype, tensor))
    prepared.sort(key=lambda entry: entry.name)
    if len(prepared) > _MAX_TENSORS:
        raise DeploymentFormatError(f"deployment contains more than {_MAX_TENSORS} tensors")
    return prepared


def tensor_schema(state: Mapping[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    """Return the strict public name/shape/source-dtype schema for a state."""

    return {
        entry.name: {"shape": list(entry.shape), "dtype": entry.dtype.name}
        for entry in _prepare_state(state)
    }


def validate_tensor_schema(schema: Any) -> dict[str, dict[str, Any]]:
    """Normalize and strictly validate an expected deployment tensor schema."""

    if not isinstance(schema, Mapping) or not schema:
        raise DeploymentFormatError("tensor schema must be a non-empty mapping")
    normalized: dict[str, dict[str, Any]] = {}
    for raw_name, raw_entry in schema.items():
        name, _ = _validate_name(raw_name)
        if name in normalized:
            raise DeploymentFormatError(f"duplicate tensor name in schema: {name}")
        if not isinstance(raw_entry, dict) or set(raw_entry) != _SCHEMA_ENTRY_KEYS:
            raise DeploymentFormatError(f"tensor schema entry {name} has invalid fields")
        shape = _validate_shape(raw_entry.get("shape"), name=name)
        dtype_name = raw_entry.get("dtype")
        if not isinstance(dtype_name, str) or dtype_name not in _DTYPE_BY_NAME:
            raise DeploymentFormatError(f"tensor schema entry {name} has unsupported dtype")
        normalized[name] = {"shape": list(shape), "dtype": dtype_name}
    if len(normalized) > _MAX_TENSORS:
        raise DeploymentFormatError(f"tensor schema contains more than {_MAX_TENSORS} tensors")
    return {name: normalized[name] for name in sorted(normalized)}


def deployment_nbytes_for_schema(
    schema: Mapping[str, Mapping[str, Any]], mode: str
) -> int:
    """Return the only valid container size for a schema and precision mode."""

    if mode not in _MODE_TO_CODE:
        raise DeploymentFormatError("deployment mode is unsupported")
    normalized = validate_tensor_schema(schema)
    body_nbytes = 0
    for name, entry in normalized.items():
        shape = tuple(entry["shape"])
        dtype = _DTYPE_BY_NAME[entry["dtype"]]
        elements = math.prod(shape)
        metadata_nbytes = (
            _RECORD_STRUCT.size
            + len(shape) * _DIM_STRUCT.size
            + len(name.encode("ascii"))
        )
        if dtype.floating and mode == MODE_INT8 and len(shape) >= 2:
            data_nbytes = elements
            scale_nbytes = shape[0] * np.dtype("<f4").itemsize
        elif dtype.floating:
            data_nbytes = elements * np.dtype("<f2").itemsize
            scale_nbytes = 0
        else:
            data_nbytes = elements * dtype.itemsize
            scale_nbytes = 0
        body_nbytes += metadata_nbytes + scale_nbytes + data_nbytes
    total = _HEADER_STRUCT.size + body_nbytes
    if total > _MAX_FILE_BYTES:
        raise DeploymentFormatError(
            f"deployment schema requires more than {_MAX_FILE_BYTES} bytes"
        )
    return total


def _validate_expected_schema(
    actual: dict[str, dict[str, Any]],
    expected: Mapping[str, Mapping[str, Any]] | None,
) -> None:
    if expected is None:
        return
    normalized = validate_tensor_schema(expected)
    if set(actual) != set(normalized):
        raise DeploymentFormatError(
            "deployment tensor names do not match expected schema: "
            f"expected={sorted(normalized)}, actual={sorted(actual)}"
        )
    for name in sorted(actual):
        if actual[name] != normalized[name]:
            raise DeploymentFormatError(
                f"deployment tensor schema mismatch for {name}: "
                f"expected={normalized[name]}, actual={actual[name]}"
            )


def _encode_tensor(entry: _PreparedTensor, mode: str) -> _EncodedRecord:
    if entry.dtype.floating and mode == MODE_INT8 and len(entry.shape) >= 2:
        values = entry.tensor.to(torch.float32).numpy()
        if not np.isfinite(values).all():
            raise DeploymentFormatError(
                f"tensor {entry.name} overflows while converting to FP32 for INT8 quantization"
            )
        rows = values.reshape(entry.shape[0], -1)
        maximum = np.max(np.abs(rows), axis=1).astype(np.float32, copy=False)
        scales = maximum / np.float32(127.0)
        scales = np.where(maximum == 0, np.float32(1.0), scales).astype(
            np.float32, copy=False
        )
        if not np.isfinite(scales).all() or np.any(scales <= 0):
            raise DeploymentFormatError(
                f"tensor {entry.name} INT8 scale underflowed or is non-finite"
            )
        quantized = np.rint(rows / scales[:, None])
        quantized = np.clip(quantized, -127, 127).astype(np.int8)
        reconstructed_max = (
            np.max(np.abs(quantized), axis=1).astype(np.float64)
            * scales.astype(np.float64)
        )
        restored_limit = float(np.finfo(entry.dtype.numpy_dtype).max)
        if np.any(reconstructed_max > restored_limit):
            raise DeploymentFormatError(
                f"tensor {entry.name} overflows its source dtype during INT8 dequantization"
            )
        return _EncodedRecord(
            entry.name,
            entry.shape,
            entry.dtype,
            _ENCODING_INT8_PER_OUTPUT,
            scales.astype("<f4", copy=False).tobytes(order="C"),
            quantized.reshape(entry.shape).tobytes(order="C"),
        )

    if entry.dtype.floating:
        values = entry.tensor.to(torch.float16).numpy().astype("<f2", copy=False)
        if not np.isfinite(values).all():
            raise DeploymentFormatError(
                f"tensor {entry.name} overflows while converting to FP16"
            )
        return _EncodedRecord(
            entry.name,
            entry.shape,
            entry.dtype,
            _ENCODING_FP16,
            b"",
            values.tobytes(order="C"),
        )

    values = entry.tensor.numpy().astype(entry.dtype.numpy_dtype, copy=False)
    return _EncodedRecord(
        entry.name,
        entry.shape,
        entry.dtype,
        _ENCODING_RAW,
        b"",
        values.tobytes(order="C"),
    )


def _record_bytes(record: _EncodedRecord) -> bytes:
    _, name = _validate_name(record.name)
    header = _RECORD_STRUCT.pack(
        len(name),
        len(record.shape),
        record.dtype.code,
        record.encoding,
        0,
        len(record.data),
        len(record.scales),
    )
    dimensions = b"".join(_DIM_STRUCT.pack(value) for value in record.shape)
    return header + dimensions + name + record.scales + record.data


def write_deployment(
    path: str | Path,
    state: Mapping[str, torch.Tensor],
    *,
    mode: str = MODE_INT8,
    expected_schema: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Atomically write a deterministic compact deployment state."""

    if mode not in _MODE_TO_CODE:
        raise DeploymentFormatError(
            f"deployment mode must be one of {tuple(_MODE_TO_CODE)}, got {mode!r}"
        )
    prepared = _prepare_state(state)
    schema = {
        entry.name: {"shape": list(entry.shape), "dtype": entry.dtype.name}
        for entry in prepared
    }
    _validate_expected_schema(schema, expected_schema)
    records = [_encode_tensor(entry, mode) for entry in prepared]
    body = b"".join(_record_bytes(record) for record in records)
    prefix = _HEADER_PREFIX_STRUCT.pack(
        MAGIC,
        VERSION,
        _MODE_TO_CODE[mode],
        0,
        len(records),
        len(body),
    )
    digest = hashlib.sha256(prefix + body).digest()
    blob = _HEADER_STRUCT.pack(
        MAGIC,
        VERSION,
        _MODE_TO_CODE[mode],
        0,
        len(records),
        len(body),
        digest,
    ) + body

    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(blob)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return {
        "format": "phase8.8_compact_deployment",
        "version": VERSION,
        "mode": mode,
        "tensor_count": len(records),
        "bytes": len(blob),
        "sha256": digest.hex(),
        "schema": schema,
    }


def _take(body: memoryview, offset: int, size: int, label: str) -> tuple[memoryview, int]:
    if size < 0 or offset < 0 or offset > len(body) or size > len(body) - offset:
        raise DeploymentFormatError(f"truncated deployment while reading {label}")
    end = offset + size
    return body[offset:end], end


def _parse_records(body: memoryview, count: int, mode: str) -> list[_ParsedRecord]:
    records: list[_ParsedRecord] = []
    names: set[str] = set()
    previous_name: str | None = None
    offset = 0
    for record_index in range(count):
        raw_header, offset = _take(
            body, offset, _RECORD_STRUCT.size, f"tensor record {record_index} header"
        )
        (
            name_nbytes,
            ndim,
            dtype_code,
            encoding,
            flags,
            data_nbytes,
            scale_nbytes,
        ) = _RECORD_STRUCT.unpack(raw_header)
        if flags != 0:
            raise DeploymentFormatError(f"tensor record {record_index} has unsupported flags")
        if name_nbytes == 0 or name_nbytes > _MAX_NAME_BYTES:
            raise DeploymentFormatError(f"tensor record {record_index} has invalid name length")
        if ndim > _MAX_NDIM:
            raise DeploymentFormatError(f"tensor record {record_index} ndim exceeds {_MAX_NDIM}")
        dimensions: list[int] = []
        for dimension_index in range(ndim):
            raw_dimension, offset = _take(
                body,
                offset,
                _DIM_STRUCT.size,
                f"tensor record {record_index} shape[{dimension_index}]",
            )
            (dimension,) = _DIM_STRUCT.unpack(raw_dimension)
            dimensions.append(dimension)
        raw_name, offset = _take(
            body, offset, name_nbytes, f"tensor record {record_index} name"
        )
        try:
            name = bytes(raw_name).decode("ascii")
        except UnicodeDecodeError as error:
            raise DeploymentFormatError(
                f"tensor record {record_index} name is not ASCII"
            ) from error
        _validate_name(name)
        if name in names:
            raise DeploymentFormatError(f"duplicate tensor name: {name}")
        if previous_name is not None and name <= previous_name:
            raise DeploymentFormatError("tensor records are not in canonical name order")
        names.add(name)
        previous_name = name

        shape = _validate_shape(dimensions, name=name)
        dtype = _DTYPE_BY_CODE.get(dtype_code)
        if dtype is None:
            raise DeploymentFormatError(f"tensor {name} has unsupported source dtype code")
        elements = math.prod(shape)
        if encoding == _ENCODING_INT8_PER_OUTPUT:
            if mode != MODE_INT8 or not dtype.floating or len(shape) < 2:
                raise DeploymentFormatError(f"tensor {name} has invalid INT8 encoding schema")
            expected_data = elements
            expected_scales = shape[0] * np.dtype("<f4").itemsize
        elif encoding == _ENCODING_FP16:
            if not dtype.floating:
                raise DeploymentFormatError(f"non-floating tensor {name} cannot use FP16 encoding")
            if mode == MODE_INT8 and len(shape) >= 2:
                raise DeploymentFormatError(
                    f"floating tensor {name} with ndim >= 2 must use per-output INT8"
                )
            expected_data = elements * np.dtype("<f2").itemsize
            expected_scales = 0
        elif encoding == _ENCODING_RAW:
            if dtype.floating:
                raise DeploymentFormatError(f"floating tensor {name} cannot use raw encoding")
            expected_data = elements * dtype.itemsize
            expected_scales = 0
        else:
            raise DeploymentFormatError(f"tensor {name} has unsupported encoding code")
        if data_nbytes != expected_data or scale_nbytes != expected_scales:
            raise DeploymentFormatError(
                f"tensor {name} payload lengths do not match shape/dtype/encoding schema"
            )
        scales, offset = _take(body, offset, scale_nbytes, f"tensor {name} scales")
        data, offset = _take(body, offset, data_nbytes, f"tensor {name} data")
        if encoding == _ENCODING_INT8_PER_OUTPUT:
            scale_values = np.frombuffer(scales, dtype="<f4")
            if not np.isfinite(scale_values).all() or np.any(scale_values <= 0):
                raise DeploymentFormatError(f"tensor {name} has invalid INT8 scales")
            quantized = np.frombuffer(data, dtype=np.int8)
            if np.any(quantized == -128):
                raise DeploymentFormatError(
                    f"tensor {name} violates symmetric INT8 range [-127, 127]"
                )
            rows = quantized.reshape(shape[0], -1)
            reconstructed_max = (
                np.max(np.abs(rows), axis=1).astype(np.float64)
                * scale_values.astype(np.float64)
            )
            if np.any(
                reconstructed_max > float(np.finfo(dtype.numpy_dtype).max)
            ):
                raise DeploymentFormatError(
                    f"tensor {name} overflows its source dtype during INT8 dequantization"
                )
        elif encoding == _ENCODING_FP16:
            values = np.frombuffer(data, dtype="<f2")
            if not np.isfinite(values).all():
                raise DeploymentFormatError(f"tensor {name} contains non-finite FP16 values")
        records.append(_ParsedRecord(name, shape, dtype, encoding, scales, data))
    if offset != len(body):
        raise DeploymentFormatError(
            f"trailing data inside deployment body: {len(body) - offset} bytes"
        )
    return records


def _decode_record(record: _ParsedRecord) -> torch.Tensor:
    if record.encoding == _ENCODING_INT8_PER_OUTPUT:
        quantized = np.frombuffer(record.data, dtype=np.int8).reshape(record.shape)
        scales = np.frombuffer(record.scales, dtype="<f4")
        scale_shape = (record.shape[0],) + (1,) * (len(record.shape) - 1)
        values = quantized.astype(np.float32) * scales.reshape(scale_shape)
        if not np.isfinite(values).all():
            raise DeploymentFormatError(
                f"tensor {record.name} produced non-finite INT8 dequantization"
            )
        result = torch.from_numpy(np.ascontiguousarray(values)).to(
            record.dtype.torch_dtype
        )
    elif record.encoding == _ENCODING_FP16:
        values = np.frombuffer(record.data, dtype="<f2").reshape(record.shape).copy()
        result = torch.from_numpy(values).to(record.dtype.torch_dtype)
    else:
        values = np.frombuffer(record.data, dtype=record.dtype.numpy_dtype).reshape(
            record.shape
        ).copy()
        result = torch.from_numpy(values).to(record.dtype.torch_dtype)
    if result.is_floating_point() and not torch.isfinite(result).all():
        raise DeploymentFormatError(
            f"tensor {record.name} is non-finite after restoring its source dtype"
        )
    return result


def read_deployment(
    path: str | Path,
    *,
    expected_schema: Mapping[str, Mapping[str, Any]] | None = None,
    expected_mode: str | None = None,
    expected_sha256: str | None = None,
) -> dict[str, torch.Tensor]:
    """Validate and read a deployment file into dequantized CPU tensors.

    ``expected_mode`` and ``expected_sha256`` bind claims in the outer scene
    manifest to the authenticated deployment header rather than merely
    validating their syntax.
    """

    if expected_mode is not None and expected_mode not in _MODE_TO_CODE:
        raise DeploymentFormatError("expected deployment mode is unsupported")
    if expected_sha256 is not None and (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise DeploymentFormatError("expected deployment SHA-256 is malformed")

    source = Path(path).expanduser().resolve()
    try:
        handle = source.open("rb")
    except OSError as error:
        raise DeploymentFormatError(f"cannot open deployment file: {source}") from error
    with handle:
        try:
            file_size = os.fstat(handle.fileno()).st_size
        except OSError as error:
            raise DeploymentFormatError(f"cannot stat deployment file: {source}") from error
        if file_size > _MAX_FILE_BYTES:
            raise DeploymentFormatError(f"deployment file exceeds {_MAX_FILE_BYTES} bytes")
        header = handle.read(_HEADER_STRUCT.size)
        if len(header) < _HEADER_STRUCT.size:
            raise DeploymentFormatError("truncated deployment header")
        (
            magic,
            version,
            mode_code,
            flags,
            tensor_count,
            body_nbytes,
            expected_digest,
        ) = _HEADER_STRUCT.unpack(header)
        if magic != MAGIC:
            raise DeploymentFormatError("invalid deployment magic")
        if version != VERSION:
            raise DeploymentFormatError(
                f"unsupported deployment version {version}; expected {VERSION}"
            )
        mode = _CODE_TO_MODE.get(mode_code)
        if mode is None or flags != 0:
            raise DeploymentFormatError("unsupported deployment header mode or flags")
        if expected_mode is not None and mode != expected_mode:
            raise DeploymentFormatError(
                f"deployment mode mismatch: expected {expected_mode}, got {mode}"
            )
        if tensor_count == 0 or tensor_count > _MAX_TENSORS:
            raise DeploymentFormatError(
                "deployment tensor_count is outside the supported range"
            )
        expected_size = _HEADER_STRUCT.size + body_nbytes
        if file_size < expected_size:
            raise DeploymentFormatError(
                f"truncated deployment body: expected {expected_size} bytes, got {file_size}"
            )
        if file_size > expected_size:
            raise DeploymentFormatError(
                f"trailing deployment bytes: expected {expected_size}, got {file_size}"
            )
        if expected_schema is not None:
            normalized_schema = validate_tensor_schema(expected_schema)
            canonical_size = deployment_nbytes_for_schema(normalized_schema, mode)
            if tensor_count != len(normalized_schema):
                raise DeploymentFormatError(
                    "deployment tensor names/count do not match expected schema"
                )
            if file_size != canonical_size:
                raise DeploymentFormatError(
                    "deployment tensor schema mismatch: file size does not match the "
                    "canonical expected schema"
                )
        body_bytes = handle.read()
        if len(body_bytes) != body_nbytes:
            raise DeploymentFormatError("deployment body changed while it was being read")
    body = memoryview(body_bytes)
    prefix = _HEADER_PREFIX_STRUCT.pack(
        magic, version, mode_code, flags, tensor_count, body_nbytes
    )
    digest_builder = hashlib.sha256(prefix)
    digest_builder.update(body)
    actual_digest = digest_builder.digest()
    if not hmac.compare_digest(actual_digest, expected_digest):
        raise DeploymentFormatError("deployment SHA-256 mismatch")
    if expected_sha256 is not None and not hmac.compare_digest(
        expected_digest.hex(), expected_sha256
    ):
        raise DeploymentFormatError("deployment SHA-256 differs from manifest")

    records = _parse_records(body, tensor_count, mode)
    actual_schema = {
        record.name: {"shape": list(record.shape), "dtype": record.dtype.name}
        for record in records
    }
    _validate_expected_schema(actual_schema, expected_schema)
    return {record.name: _decode_record(record) for record in records}


write_deployment_state = write_deployment
read_deployment_state = read_deployment


__all__ = [
    "DeploymentFormatError",
    "MAGIC",
    "MODE_FP16",
    "MODE_INT8",
    "VERSION",
    "deployment_nbytes_for_schema",
    "read_deployment",
    "read_deployment_state",
    "tensor_schema",
    "validate_tensor_schema",
    "write_deployment",
    "write_deployment_state",
]
