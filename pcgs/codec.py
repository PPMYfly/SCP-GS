"""Train, encode and decode Phase6.4 with the official PanoCompactGS primitives."""

from __future__ import annotations

import json
import math
import mmap
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils.stateless import functional_call

from .data import (
    ATTRIBUTE_DIM,
    ATTRIBUTE_LAYOUT,
    apply_collision_sidecar,
    build_collision_sidecar,
    canonical_morton_order,
    canonicalize_decoded_attributes,
    decoded_tensor_digest,
    dequantize_xyz,
    geometry_collision_stats,
    load_frozen_scene,
    scene_fingerprint,
    voxelize_and_sort,
    write_decoded_scene,
)
from .deployment import (
    MODE_FP16,
    MODE_INT8,
    VERSION as DEPLOYMENT_VERSION,
    deployment_nbytes_for_schema,
    read_deployment,
    tensor_schema,
    validate_tensor_schema,
    write_deployment,
)
from .integrity import load_verified_manifest, logical_stream_bytes, write_verified_manifest
from .groups import (
    GROUP_LAYOUT,
    GROUP_ORDER,
    GROUP_SCHEMA_FORMAT,
    group_active_mask,
    validate_group_schema,
)
from .rd import load_rd_profile
from .sh_selector import load_sh_profile
from .hard_selector import load_hard_profile
from .sh_stream import (
    build_hierarchical_sh_stream,
    reconstruct_hierarchical_sh_levels,
    validate_hierarchical_sh_metadata,
)
from .spherical import (
    build_spherical_profile,
    spherical_features,
    validate_spherical_profile,
)
from .model import EntropyModel, ModelConfig
from .backend import (
    STE_multistep,
    _read_checked_arithmetic_stream,
    compress_gpcc,
    decoder,
    decoder_gaussian_chunk,
    decoder_gaussian_mixed_chunk,
    decompress_gpcc,
    encoder,
    encoder_gaussian_chunk,
    encoder_gaussian_mixed_chunk,
)


FORMAT = "phase8.8_hacpp_phase64_v10"
HARD_Q_SCALE = 0.5
CENTERED_CLAMP_CENTER = "zero"
CENTERED_SYMBOL_LIMIT = 15_000
MAX_CHECKPOINT_BYTES = 128 * 1024 * 1024
MAX_CDF_CELLS = 64 * 1024 * 1024
DEFAULT_ARITHMETIC_CHUNK_SIZE = 2_000
DEPLOYMENT_TRAINING_FORMAT = "phase8.8_fake_quant_training_v2"


def _deployment_contract(mode: str) -> dict[str, Any]:
    if mode not in {MODE_INT8, MODE_FP16}:
        raise ValueError("deployment mode must be 'int8' or 'fp16'")
    return {
        "format": DEPLOYMENT_TRAINING_FORMAT,
        "mode": mode,
        "fp32_master_weights": True,
        "ste_fake_quant_forward": True,
        "destructive_projection": False,
    }


def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("PanoCompactGS arithmetic and grid extensions require an NVIDIA CUDA GPU")
    return torch.device("cuda")


def _contexts(
    xyz_int: np.ndarray,
    geometry_bits: int,
    xyz_min: np.ndarray,
    xyz_max: np.ndarray,
    spherical_profile: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = float((1 << geometry_bits) - 1)
    xyz_normalized = torch.from_numpy(xyz_int.astype(np.float32)).to(device) / scale
    # Use the exact dequantized geometry shared by encoder and decoder.  The
    # serialized reference removes Phase8.6's accidental world-origin prior.
    world = dequantize_xyz(xyz_int, xyz_min, xyz_max, geometry_bits)
    direction, log_radius = spherical_features(world, spherical_profile)
    spherical_context = torch.from_numpy(
        np.concatenate((direction, log_radius), axis=1)
    ).to(device)
    return xyz_normalized, spherical_context


def _quantization(attributes: np.ndarray, quant_bits: int) -> tuple[np.ndarray, np.ndarray]:
    if not 6 <= quant_bits <= 14:
        raise ValueError("quant_bits must be in [6, 14]")
    low = attributes.min(axis=0)
    high = attributes.max(axis=0)
    centers = ((low + high) * 0.5).astype(np.float32)
    # Centered rounding needs symmetric signed endpoints.  Using 2^b-1
    # creates half-integer endpoints and may round one step outside the
    # source/frozen-float16 domain.
    signed_levels = float((1 << (quant_bits - 1)) - 1)
    steps = ((high - low) / (2.0 * signed_levels)).astype(np.float32)
    steps = np.maximum(steps, np.float32(1e-6))
    return centers, steps


def _quantized(
    attributes: torch.Tensor,
    centers: torch.Tensor,
    steps: torch.Tensor,
    hard_flags: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    centered = attributes - centers
    importance_scale = torch.where(
        hard_flags.view(-1, 1) > 0,
        torch.full_like(hard_flags.view(-1, 1), HARD_Q_SCALE),
        torch.ones_like(hard_flags.view(-1, 1)),
    )
    q = steps.expand_as(centered) * importance_scale
    # PanoCompactGS clamps to +/-15000 symbols around input_mean.  A batch-derived
    # mean makes training/evaluation/whole-scene encoding disagree when an
    # aggressive RD profile reaches that bound.  Our values are already
    # centered by the serialized per-channel midpoint, so zero is the only
    # decoder-reproducible clamp center.
    clamp_center = torch.zeros_like(centers)
    quantized = STE_multistep.apply(centered, q, clamp_center)
    return quantized, q


def _active_attribute_mask(sh_levels: torch.Tensor) -> torch.Tensor:
    """Mask inactive Phase6.4 SH bands while retaining all other attributes."""

    count = sh_levels.numel()
    mask = torch.ones((count, ATTRIBUTE_DIM), dtype=torch.bool, device=sh_levels.device)
    rest = mask[:, 3:48]
    for degree, (begin, end) in enumerate(((0, 3), (3, 8), (8, 15)), start=1):
        rest[:, begin * 3 : end * 3] = (sh_levels >= degree).view(-1, 1)
    return mask


def _minimal_source_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "max_sh_degree": int(metadata.get("max_sh_degree", 3)),
        "active_sh_degree": int(metadata.get("active_sh_degree", 3)),
        "scene_extent": float(metadata.get("scene_extent", 1.0)),
    }


def _install_nonhash_state(model: EntropyModel, state: dict[str, torch.Tensor]) -> None:
    incompatible = model.load_state_dict(state, strict=False)
    expected_missing = {
        name
        for name in model.state_dict()
        if name.endswith((".params", ".offsets_list", ".resolutions_list"))
    }
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise ValueError(
            "Entropy model state mismatch: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )


def _validate_cdf_chunk(symbol_count: int, span: int) -> None:
    if symbol_count < 0 or span < 0:
        raise ValueError("Arithmetic chunk dimensions must be non-negative")
    if span > 32767:
        raise ValueError(
            f"Arithmetic symbol span {span} exceeds the upstream int16 limit 32767"
        )
    cells = symbol_count * (span + 2)
    if cells > MAX_CDF_CELLS:
        raise ValueError(
            f"Arithmetic chunk requires {cells} CDF cells; limit is "
            f"{MAX_CDF_CELLS}. Use a smaller --chunk-size."
        )


def _assert_symbol_span(values: torch.Tensor, q: torch.Tensor, chunk_size: int) -> None:
    if chunk_size <= 0:
        raise ValueError("arithmetic_chunk_size must be positive")
    if not torch.isfinite(values).all() or not torch.isfinite(q).all() or not torch.all(q > 0):
        raise ValueError("Arithmetic symbols and quantization steps must be finite; q must be positive")
    symbols = torch.round(values.reshape(-1) / q.reshape(-1))
    for begin in range(0, symbols.numel(), chunk_size):
        chunk = symbols[begin : begin + chunk_size]
        span = int((chunk.max() - chunk.min()).item()) if chunk.numel() else 0
        _validate_cdf_chunk(int(chunk.numel()), span)


def _assert_binary_tensor(value: torch.Tensor, name: str) -> None:
    if not torch.all((value == 0) | (value == 1)):
        raise RuntimeError(f"Decoded {name} stream is not binary")


_GROUP_STREAM_KEYS = {
    "name",
    "file_base",
    "width",
    "active_count",
    "symbol_count",
    "chunk_count",
}


def _attribute_group_schema(sh_levels: np.ndarray) -> dict[str, Any]:
    levels = np.asarray(sh_levels, dtype=np.int32)
    active = group_active_mask(levels)
    schema = {
        "format": GROUP_SCHEMA_FORMAT,
        "point_count": int(levels.size),
        "attribute_dim": ATTRIBUTE_DIM,
        "dtype": np.dtype(np.float32).str,
        "groups": [
            {
                "name": spec.name,
                "start": spec.start,
                "stop": spec.stop,
                "width": spec.width,
                "min_sh_level": spec.min_sh_level,
                "active_count": int(active[:, index].sum()),
            }
            for index, spec in enumerate(GROUP_LAYOUT)
        ],
    }
    return validate_group_schema(schema, sh_levels=levels)


def _attribute_group_streams(
    schema: dict[str, Any], chunk_size: int
) -> list[dict[str, Any]]:
    if chunk_size <= 0:
        raise ValueError("arithmetic_chunk_size must be positive")
    streams: list[dict[str, Any]] = []
    for entry in schema["groups"]:
        symbol_count = int(entry["active_count"]) * int(entry["width"])
        streams.append(
            {
                "name": entry["name"],
                "file_base": f"attributes_{entry['name']}.b",
                "width": int(entry["width"]),
                "active_count": int(entry["active_count"]),
                "symbol_count": symbol_count,
                "chunk_count": math.ceil(symbol_count / chunk_size),
            }
        )
    return streams


def _validate_attribute_group_streams(
    manifest: dict[str, Any], files: set[str]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    count = int(manifest["num_points"])
    schema = validate_group_schema(
        manifest.get("attribute_group_schema"), point_count=count
    )
    chunk_size = int(manifest["arithmetic_chunk_size"])
    records = manifest.get("attribute_group_streams")
    if not isinstance(records, list) or len(records) != len(GROUP_LAYOUT):
        raise ValueError("Manifest must contain six ordered attribute group streams")
    expected_records = _attribute_group_streams(schema, chunk_size)
    for index, (actual, expected) in enumerate(zip(records, expected_records, strict=True)):
        if not isinstance(actual, dict) or set(actual) != _GROUP_STREAM_KEYS:
            raise ValueError(f"Attribute group stream {index} has an invalid schema")
        if actual != expected:
            raise ValueError(
                f"Attribute group stream {index} differs from the canonical plan"
            )
    actual_files = {
        name for name in files if name.startswith("attributes_") and name.endswith(".b")
    }
    expected_file_count = sum(int(record["chunk_count"]) for record in expected_records)
    if expected_file_count != len(actual_files):
        raise ValueError(
            "Grouped attribute chunk count differs from the payload inventory"
        )
    # Generate names only after the count is tied to the finite on-disk
    # inventory; an attacker cannot amplify a tiny manifest into billions of
    # Python set entries by declaring chunk_size=1 and an enormous point count.
    expected_files = {
        record["file_base"].replace(".b", f"_{chunk}.b")
        for record in expected_records
        for chunk in range(int(record["chunk_count"]))
    }
    if actual_files != expected_files:
        raise ValueError(
            "Grouped attribute chunk inventory mismatch: "
            f"expected={sorted(expected_files)}, actual={sorted(actual_files)}"
        )
    total_symbols = sum(int(record["symbol_count"]) for record in expected_records)
    if total_symbols != int(manifest.get("attribute_symbol_count", -1)):
        raise ValueError("Grouped attribute symbols do not match attribute_symbol_count")
    return schema, expected_records


_DEPLOYMENT_METADATA_KEYS = {
    "format",
    "version",
    "mode",
    "tensor_count",
    "bytes",
    "sha256",
    "schema",
}


def _expected_deployment_schema(model_config: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(model_config, dict):
        raise ValueError("Manifest model_config must be an object")
    if model_config != ModelConfig().to_dict():
        raise ValueError("Manifest model_config is not the canonical Phase8.8 architecture")
    try:
        model = EntropyModel(ModelConfig.from_dict(model_config))
        return tensor_schema(model.deployment_state())
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Manifest model_config is invalid") from error


def _validate_deployment_metadata(
    manifest: dict[str, Any], payload_records: list[dict[str, Any]]
) -> dict[str, Any]:
    metadata = manifest.get("deployment")
    if not isinstance(metadata, dict) or set(metadata) != _DEPLOYMENT_METADATA_KEYS:
        raise ValueError("Manifest deployment metadata has an invalid schema")
    if metadata.get("format") != "phase8.8_compact_deployment":
        raise ValueError("Manifest deployment format mismatch")
    if not isinstance(metadata.get("version"), int) or isinstance(
        metadata.get("version"), bool
    ) or metadata.get("version") != DEPLOYMENT_VERSION:
        raise ValueError("Manifest deployment version mismatch")
    if metadata.get("mode") not in {MODE_INT8, MODE_FP16}:
        raise ValueError("Manifest deployment mode mismatch")
    if isinstance(metadata.get("tensor_count"), bool) or not isinstance(
        metadata.get("tensor_count"), int
    ):
        raise ValueError("Manifest deployment tensor_count must be an integer")
    if isinstance(metadata.get("bytes"), bool) or not isinstance(metadata.get("bytes"), int):
        raise ValueError("Manifest deployment bytes must be an integer")
    digest = metadata.get("sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("Manifest deployment SHA256 is malformed")
    try:
        schema = validate_tensor_schema(metadata.get("schema"))
    except ValueError as error:
        raise ValueError("Manifest deployment tensor schema is invalid") from error
    expected_schema = _expected_deployment_schema(manifest.get("model_config"))
    if schema != expected_schema:
        raise ValueError("Manifest deployment schema does not match model_config")
    if metadata["tensor_count"] != len(schema):
        raise ValueError("Manifest deployment tensor_count mismatch")
    expected_model_bytes = deployment_nbytes_for_schema(schema, str(metadata["mode"]))
    if metadata["bytes"] != expected_model_bytes:
        raise ValueError("Manifest deployment bytes do not match canonical schema/mode")
    model_records = [record for record in payload_records if record.get("name") == "model.bin"]
    if len(model_records) != 1 or model_records[0].get("bytes") != metadata["bytes"]:
        raise ValueError("Manifest deployment byte count does not match model.bin")
    expected_training = _deployment_contract(str(metadata["mode"]))
    if manifest.get("deployment_training") != expected_training:
        raise ValueError("Manifest deployment training contract mismatch")
    model = EntropyModel(ModelConfig.from_dict(manifest["model_config"]))
    expected_hash_numel = sum(grid.params.numel() for grid in model.hash_grids())
    hash_numel = manifest.get("hash_numel")
    if (
        not isinstance(hash_numel, int)
        or isinstance(hash_numel, bool)
        or hash_numel != expected_hash_numel
    ):
        raise ValueError("Manifest hash_numel does not match model_config")
    return {
        **metadata,
        "schema": schema,
    }


def _validate_manifest_schema(
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    count = int(manifest.get("num_points", 0))
    if count <= 0:
        raise ValueError("Manifest num_points must be positive")
    if int(manifest.get("attribute_dim", -1)) != ATTRIBUTE_DIM:
        raise ValueError("Manifest attribute_dim mismatch")
    if manifest.get("attribute_layout") != [list(item) for item in ATTRIBUTE_LAYOUT]:
        raise ValueError("Manifest attribute layout mismatch")
    geometry_bits = int(manifest.get("geometry_bits", 0))
    quant_bits = int(manifest.get("quant_bits", 0))
    if not 12 <= geometry_bits <= 21 or not 6 <= quant_bits <= 14:
        raise ValueError("Manifest quantization settings are outside supported ranges")
    if float(manifest.get("hard_q_scale", -1)) != HARD_Q_SCALE:
        raise ValueError("Manifest hard_q_scale does not match the codec contract")
    centers = np.asarray(manifest.get("centers"), dtype=np.float32)
    base_steps = np.asarray(manifest.get("base_steps"), dtype=np.float32)
    steps = np.asarray(manifest.get("steps"), dtype=np.float32)
    if any(value.shape != (ATTRIBUTE_DIM,) for value in (centers, base_steps, steps)):
        raise ValueError("Manifest centers/base_steps/steps shape mismatch")
    if (
        not np.isfinite(centers).all()
        or not np.isfinite(base_steps).all()
        or not np.isfinite(steps).all()
        or np.any(base_steps <= 0)
        or np.any(steps <= 0)
    ):
        raise ValueError("Manifest centers/base_steps/steps are invalid")
    if manifest.get("quantization_contract") != {
        "format": "phase8.4_56_channel_group_multiplier_v3",
        "hard_q_scale": HARD_Q_SCALE,
        "centered_clamp_center": CENTERED_CLAMP_CENTER,
        "centered_symbol_limit": CENTERED_SYMBOL_LIMIT,
        "geometry_policy": "fixed_by_hac_geometry_bits",
    }:
        raise ValueError("Manifest quantization contract mismatch")
    xyz_min = np.asarray(manifest.get("xyz_min"), dtype=np.float32)
    xyz_max = np.asarray(manifest.get("xyz_max"), dtype=np.float32)
    if xyz_min.shape != (3,) or xyz_max.shape != (3,):
        raise ValueError("Manifest xyz bounds shape mismatch")
    if not np.isfinite(xyz_min).all() or not np.isfinite(xyz_max).all() or np.any(xyz_max < xyz_min):
        raise ValueError("Manifest xyz bounds are invalid")
    symbols = int(manifest.get("attribute_symbol_count", -1))
    chunk_size = int(manifest.get("arithmetic_chunk_size", 0))
    if symbols <= 0 or chunk_size <= 0:
        raise ValueError("Manifest attribute symbol count/chunk size is invalid")
    payload_records = manifest["integrity"]["files"]
    files = {item["name"] for item in payload_records}
    group_schema, group_streams = _validate_attribute_group_streams(manifest, files)
    required = {
        "xyz_gpcc.bin",
        "collision_records.bin",
        "hash.b",
        "hard_points.b",
        "model.bin",
    }
    if not required.issubset(files):
        raise ValueError(f"Stream is missing required payloads: {sorted(required - files)}")
    sh_stream = validate_hierarchical_sh_metadata(
        manifest.get("sh_level_stream"), point_count=count, payload_files=files
    )
    validate_spherical_profile(manifest.get("spherical_profile"))
    _validate_deployment_metadata(manifest, payload_records)
    return sh_stream, group_schema, group_streams


def preflight_arithmetic_payloads(
    root: str | Path,
    manifest: dict[str, Any],
    sh_stream: dict[str, Any],
    group_streams: list[dict[str, Any]],
) -> dict[str, int]:
    """Validate every arithmetic payload without allocating a CUDA tensor."""

    stream_root = Path(root).expanduser().resolve()
    checked_files = 0
    checked_symbols = 0

    def check(name: str, symbols: int, *, binary: bool) -> None:
        nonlocal checked_files, checked_symbols
        _read_checked_arithmetic_stream(
            stream_root / name, int(symbols), binary=binary
        )
        checked_files += 1
        checked_symbols += int(symbols)

    check("hash.b", int(manifest["hash_numel"]), binary=True)
    check("hard_points.b", int(manifest["num_points"]), binary=True)
    for stage in sh_stream["stages"]:
        if stage["mode"] == "arithmetic":
            check(str(stage["file"]), int(stage["symbol_count"]), binary=True)

    chunk_size = int(manifest["arithmetic_chunk_size"])
    for stream in group_streams:
        symbol_count = int(stream["symbol_count"])
        for chunk in range(int(stream["chunk_count"])):
            chunk_symbols = min(chunk_size, symbol_count - chunk * chunk_size)
            name = str(stream["file_base"]).replace(".b", f"_{chunk}.b")
            check(name, chunk_symbols, binary=False)
    return {"files": checked_files, "symbols": checked_symbols}


def _validate_checkpoint_schema(checkpoint: dict[str, Any]) -> None:
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != FORMAT:
        value = checkpoint.get("format") if isinstance(checkpoint, dict) else None
        raise ValueError(f"Unsupported checkpoint format: {value!r}")
    count = int(checkpoint.get("num_points", 0))
    geometry_bits = int(checkpoint.get("geometry_bits", 0))
    quant_bits = int(checkpoint.get("quant_bits", 0))
    if count <= 0:
        raise ValueError("Checkpoint num_points must be positive")
    if not 12 <= geometry_bits <= 21 or not 6 <= quant_bits <= 14:
        raise ValueError("Checkpoint quantization settings are outside supported ranges")
    fingerprint = checkpoint.get("scene_fingerprint")
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise ValueError("Checkpoint scene fingerprint is malformed")
    raw_quantizers = (
        checkpoint.get("centers"),
        checkpoint.get("base_steps"),
        checkpoint.get("steps"),
    )
    if not all(
        isinstance(value, torch.Tensor)
        and value.device.type == "cpu"
        and value.dtype == torch.float32
        for value in raw_quantizers
    ):
        raise ValueError("Checkpoint centers/base_steps/steps must be CPU float32 tensors")
    centers, base_steps, steps = raw_quantizers
    if any(value.shape != (ATTRIBUTE_DIM,) for value in (centers, base_steps, steps)):
        raise ValueError("Checkpoint centers/base_steps/steps shape mismatch")
    if (
        not torch.isfinite(centers).all()
        or not torch.isfinite(base_steps).all()
        or not torch.isfinite(steps).all()
        or not torch.all(base_steps > 0)
        or not torch.all(steps > 0)
    ):
        raise ValueError("Checkpoint centers/base_steps/steps are invalid")
    contract = checkpoint.get("quantization_contract")
    if not isinstance(contract, dict) or contract != {
        "format": "phase8.4_56_channel_group_multiplier_v3",
        "hard_q_scale": HARD_Q_SCALE,
        "centered_clamp_center": CENTERED_CLAMP_CENTER,
        "centered_symbol_limit": CENTERED_SYMBOL_LIMIT,
        "geometry_policy": "fixed_by_hac_geometry_bits",
    }:
        raise ValueError("Checkpoint quantization contract mismatch")
    config = ModelConfig.from_dict(checkpoint.get("model_config", {}))
    if checkpoint.get("model_config") != ModelConfig().to_dict():
        raise ValueError("Checkpoint model_config is not the canonical Phase8.8 architecture")
    if config.attribute_dim != ATTRIBUTE_DIM:
        raise ValueError("Checkpoint model attribute dimension mismatch")
    if tuple(config.group_order) != GROUP_ORDER:
        raise ValueError("Checkpoint semantic group order mismatch")
    validate_spherical_profile(checkpoint.get("spherical_profile"))
    deployment_training = checkpoint.get("deployment_training")
    if not isinstance(deployment_training, dict):
        raise ValueError("Checkpoint deployment training contract is missing")
    try:
        expected_deployment_training = _deployment_contract(
            deployment_training.get("mode")
        )
    except ValueError as error:
        raise ValueError("Checkpoint deployment mode is invalid") from error
    if deployment_training != expected_deployment_training:
        raise ValueError("Checkpoint deployment training contract mismatch")
    state = checkpoint.get("model_state")
    if not isinstance(state, dict) or not state or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in state.items()
    ):
        raise ValueError("Checkpoint model_state must contain named tensors")
    reference_state = EntropyModel(ModelConfig()).state_dict()
    if set(state) != set(reference_state):
        raise ValueError("Checkpoint model_state tensor names mismatch")
    structural_suffixes = (".offsets_list", ".resolutions_list")
    for name, reference in reference_state.items():
        value = state[name]
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise ValueError(f"Checkpoint tensor schema mismatch for {name}")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"Checkpoint tensor {name} contains non-finite values")
        if name.endswith(structural_suffixes) and not torch.equal(value, reference):
            raise ValueError(f"Checkpoint structural buffer mismatch for {name}")
    selected = checkpoint.get("selected_sh_levels")
    if selected is not None:
        if (
            not isinstance(selected, torch.Tensor)
            or selected.device.type != "cpu"
            or selected.dtype != torch.int32
            or selected.shape != (count,)
            or torch.any((selected < 0) | (selected > 3))
        ):
            raise ValueError("Checkpoint selected SH levels are invalid")
    selected_hard = checkpoint.get("selected_hard_flags")
    if selected_hard is not None:
        if (
            not isinstance(selected_hard, torch.Tensor)
            or selected_hard.device.type != "cpu"
            or selected_hard.dtype != torch.uint8
            or selected_hard.shape != (count,)
            or torch.any((selected_hard != 0) & (selected_hard != 1))
        ):
            raise ValueError("Checkpoint selected hard-point flags are invalid")
        selector = checkpoint.get("hard_selector")
        if not isinstance(selector, dict) or selector.get("selected_count") != int(selected_hard.sum()):
            raise ValueError("Checkpoint hard-point selector metadata mismatch")


def _group_metrics(reference: np.ndarray, decoded: np.ndarray) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    offset = 0
    for name, width in ATTRIBUTE_LAYOUT:
        ref = reference[:, offset : offset + width].astype(np.float64)
        dec = decoded[:, offset : offset + width].astype(np.float64)
        error = ref - dec
        mse = float(np.mean(error * error))
        peak = float(max(ref.max() - ref.min(), 1e-12))
        psnr = float("inf") if mse == 0 else float(10.0 * math.log10(peak * peak / mse))
        metrics[name] = {"mse": mse, "max_abs": float(np.max(np.abs(error))), "psnr": psnr}
        offset += width
    return metrics


def train(
    input_path: str | Path,
    checkpoint_path: str | Path,
    *,
    steps: int = 2000,
    batch_size: int = 4096,
    learning_rate: float = 1e-3,
    quant_bits: int = 10,
    geometry_bits: int = 20,
    max_points: int | None = None,
    seed: int = 0,
    rd_profile: str | Path | None = None,
    sh_profile: str | Path | None = None,
    hard_profile: str | Path | None = None,
    cameras_path: str | Path | None = None,
    deployment_mode: str = MODE_INT8,
) -> dict[str, Any]:
    end_to_end_started = time.perf_counter()
    device = _device()
    if steps <= 0 or batch_size <= 0:
        raise ValueError("steps and batch_size must be positive")
    deployment_training = _deployment_contract(deployment_mode)
    torch.manual_seed(seed)
    np.random.seed(seed)
    scene = load_frozen_scene(input_path, max_points=max_points)
    source_fingerprint = scene_fingerprint(scene)
    selected_sh_levels = None
    sh_selector_metadata = None
    if sh_profile is not None:
        selected_sh_levels, sh_selector_metadata = load_sh_profile(
            sh_profile,
            expected_fingerprint=source_fingerprint,
            expected_points=scene.num_points,
        )
        if np.any(selected_sh_levels > scene.sh_levels):
            raise ValueError("SH profile may not activate bands absent from the source")
        scene.sh_levels = selected_sh_levels
    selected_hard_flags = None
    hard_selector_metadata = None
    if hard_profile is not None:
        selected_hard_flags, hard_selector_metadata = load_hard_profile(
            hard_profile,
            expected_fingerprint=source_fingerprint,
            expected_points=scene.num_points,
        )
        scene.hard_flags = selected_hard_flags
    xyz_int, attrs_np, sh_levels, hard_flags, xyz_min, xyz_max = voxelize_and_sort(
        scene, geometry_bits
    )
    spherical_profile = build_spherical_profile(scene.xyz, cameras_path)
    centers_np, base_steps_np = _quantization(attrs_np, quant_bits)
    steps_np = base_steps_np.copy()
    rd_metadata = None
    if rd_profile is not None:
        steps_np, rd_metadata = load_rd_profile(
            rd_profile, centers_np, base_steps_np, quant_bits
        )
    xyz, spherical = _contexts(
        xyz_int, geometry_bits, xyz_min, xyz_max, spherical_profile, device
    )
    attrs = torch.from_numpy(attrs_np).to(device)
    hard = torch.from_numpy(hard_flags.astype(np.float32)).to(device)
    levels = torch.from_numpy(sh_levels.astype(np.int32)).to(device)
    active = _active_attribute_mask(levels)
    centers = torch.from_numpy(centers_np).to(device).view(1, -1)
    q_steps = torch.from_numpy(steps_np).to(device).view(1, -1)

    config = ModelConfig()
    model = EntropyModel(config).to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator(device=device).manual_seed(seed)

    def estimate_deployed_bits(
        quantized_value: torch.Tensor,
        xyz_value: torch.Tensor,
        spherical_value: torch.Tensor,
        q_value: torch.Tensor,
        levels_value: torch.Tensor,
        hard_value: torch.Tensor,
        active_value: torch.Tensor,
    ) -> torch.Tensor:
        overrides = model.fake_quantized_parameter_overrides(deployment_mode)
        return functional_call(
            model,
            overrides,
            (
                quantized_value,
                xyz_value,
                spherical_value,
                q_value,
                levels_value,
                hard_value,
                active_value,
            ),
            strict=False,
        )

    @torch.no_grad()
    def evaluate_full_scene() -> float:
        """Evaluate both endpoints on the same full scene through fake quantization."""
        model.eval()
        bit_sum = 0.0
        symbol_count = 0
        for begin in range(0, scene.num_points, batch_size):
            end = min(begin + batch_size, scene.num_points)
            selection = slice(begin, end)
            quantized_eval, q_eval = _quantized(
                attrs[selection], centers, q_steps, hard[selection]
            )
            bits_eval = estimate_deployed_bits(
                quantized_eval,
                xyz[selection],
                spherical[selection],
                q_eval,
                levels[selection],
                hard[selection],
                active[selection],
            )
            active_eval = active[selection]
            bit_sum += float(bits_eval[active_eval].sum().item())
            symbol_count += int(active_eval.sum().item())
        model.train()
        if symbol_count <= 0:
            raise RuntimeError("Full-scene evaluation found no active attribute symbols")
        return bit_sum / symbol_count

    initial_evaluation = evaluate_full_scene()
    history: list[dict[str, float]] = []
    optimization_started = time.perf_counter()
    for step in range(steps):
        index = torch.randint(0, scene.num_points, (min(batch_size, scene.num_points),), device=device, generator=generator)
        quantized, q = _quantized(attrs[index], centers, q_steps, hard[index])
        bits = estimate_deployed_bits(
            quantized,
            xyz[index],
            spherical[index],
            q,
            levels[index],
            hard[index],
            active[index],
        )
        loss = bits[active[index]].mean()
        if not torch.isfinite(loss).item():
            raise RuntimeError("Training produced a non-finite entropy loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % max(1, steps // 20) == 0 or step + 1 == steps:
            history.append({"step": step + 1, "bits_per_attribute": float(loss.detach())})

    optimization_seconds = time.perf_counter() - optimization_started
    final_evaluation = evaluate_full_scene()

    checkpoint = {
        "format": FORMAT,
        "upstream_commit": (Path(__file__).resolve().parents[1] / "third_party/entropy_backend" / "UPSTREAM_COMMIT").read_text().strip(),
        "model_config": config.to_dict(),
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "centers": torch.from_numpy(centers_np),
        "base_steps": torch.from_numpy(base_steps_np),
        "steps": torch.from_numpy(steps_np),
        "quant_bits": quant_bits,
        "geometry_bits": geometry_bits,
        "source_metadata": _minimal_source_metadata(scene.metadata),
        "scene_fingerprint": source_fingerprint,
        "num_points": scene.num_points,
        "rd_profile": rd_metadata,
        "quantization_contract": {
            "format": "phase8.4_56_channel_group_multiplier_v3",
            "hard_q_scale": HARD_Q_SCALE,
            "centered_clamp_center": CENTERED_CLAMP_CENTER,
            "centered_symbol_limit": CENTERED_SYMBOL_LIMIT,
            "geometry_policy": "fixed_by_hac_geometry_bits",
        },
        "selected_sh_levels": (
            torch.from_numpy(selected_sh_levels.astype(np.int32, copy=False))
            if selected_sh_levels is not None
            else None
        ),
        "sh_selector": sh_selector_metadata,
        "selected_hard_flags": (
            torch.from_numpy(selected_hard_flags.astype(np.uint8, copy=False))
            if selected_hard_flags is not None
            else None
        ),
        "hard_selector": hard_selector_metadata,
        "spherical_profile": spherical_profile,
        "deployment_training": deployment_training,
        "evaluation": {
            "format": "phase8.8_full_scene_fake_quant_bits_v2",
            "population": "all_active_attribute_symbols",
            "initial_bits_per_attribute": initial_evaluation,
            "final_bits_per_attribute": final_evaluation,
        },
        "history": history,
    }
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    finished = time.perf_counter()
    end_to_end_seconds = finished - end_to_end_started
    result = {
        "checkpoint": str(checkpoint_path),
        "num_points": scene.num_points,
        "steps": steps,
        "seconds": end_to_end_seconds,
        "end_to_end_seconds": end_to_end_seconds,
        "optimization_seconds": optimization_seconds,
        "initial_bits_per_attribute": initial_evaluation,
        "final_bits_per_attribute": final_evaluation,
        "deployment_mode": deployment_mode,
        "hard_profile_enabled": selected_hard_flags is not None,
        "selected_hard_point_count": (
            int(selected_hard_flags.sum()) if selected_hard_flags is not None else int(scene.hard_flags.sum())
        ),
        "hard_selector": hard_selector_metadata,
    }
    return result


def _load_checkpoint(path: str | Path, device: torch.device) -> tuple[dict, EntropyModel]:
    checkpoint_path = Path(path).expanduser().resolve()
    if checkpoint_path.stat().st_size > MAX_CHECKPOINT_BYTES:
        raise ValueError("Checkpoint exceeds the bounded canonical size")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    _validate_checkpoint_schema(checkpoint)
    model = EntropyModel(ModelConfig.from_dict(checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return checkpoint, model


@torch.no_grad()
def encode(
    input_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    tmc3_path: str | Path,
    max_points: int | None = None,
    arithmetic_chunk_size: int = DEFAULT_ARITHMETIC_CHUNK_SIZE,
) -> dict[str, Any]:
    end_to_end_started = time.perf_counter()
    device = _device()
    checkpoint, model = _load_checkpoint(checkpoint_path, device)
    geometry_bits = int(checkpoint["geometry_bits"])
    scene = load_frozen_scene(input_path, max_points=max_points)
    expected_fingerprint = checkpoint.get("scene_fingerprint")
    actual_fingerprint = scene_fingerprint(scene)
    if expected_fingerprint != actual_fingerprint:
        raise ValueError(
            "Checkpoint/input scene fingerprint mismatch; retrain for this exact frozen package"
        )
    if int(checkpoint.get("num_points", -1)) != scene.num_points:
        raise ValueError("Checkpoint/input point-count mismatch")
    selected_sh_levels = checkpoint.get("selected_sh_levels")
    if selected_sh_levels is not None:
        selected_sh_levels = np.asarray(selected_sh_levels, dtype=np.int32)
        if selected_sh_levels.shape != (scene.num_points,):
            raise ValueError("Checkpoint SH-level selection has the wrong shape")
        if np.any(selected_sh_levels > scene.sh_levels):
            raise ValueError("Checkpoint SH selection activates absent source bands")
        scene.sh_levels = selected_sh_levels
    selected_hard_flags = checkpoint.get("selected_hard_flags")
    if selected_hard_flags is not None:
        selected_hard_flags = np.asarray(selected_hard_flags, dtype=np.uint8)
        if selected_hard_flags.shape != (scene.num_points,):
            raise ValueError("Checkpoint hard-point selection has the wrong shape")
        if np.any((selected_hard_flags != 0) & (selected_hard_flags != 1)):
            raise ValueError("Checkpoint hard-point selection is not binary")
        scene.hard_flags = selected_hard_flags
    xyz_int, attrs_np, sh_levels, hard_flags, xyz_min, xyz_max = voxelize_and_sort(
        scene, geometry_bits
    )
    output = Path(output_path).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output stream directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    tmc3 = Path(tmc3_path).expanduser().resolve()
    if not tmc3.is_file():
        raise FileNotFoundError(f"Missing TMC13 executable: {tmc3}; run ./build.sh")
    core_codec_started = time.perf_counter()

    geometry_stream = compress_gpcc(
        torch.from_numpy(xyz_int.astype(np.float32)), gpcc_codec_path=str(tmc3)
    )
    (output / "xyz_gpcc.bin").write_bytes(geometry_stream)
    collision_payload = build_collision_sidecar(scene, geometry_bits)
    (output / "collision_records.bin").write_bytes(collision_payload)

    spherical_profile = validate_spherical_profile(checkpoint["spherical_profile"])
    xyz, spherical = _contexts(
        xyz_int, geometry_bits, xyz_min, xyz_max, spherical_profile, device
    )
    attrs = torch.from_numpy(attrs_np).to(device)
    hard = torch.from_numpy(hard_flags.astype(np.float32)).to(device)
    levels = torch.from_numpy(sh_levels.astype(np.int32)).to(device)
    active = _active_attribute_mask(levels)
    centers = torch.as_tensor(
        checkpoint["centers"], dtype=torch.float32, device=device
    ).view(1, ATTRIBUTE_DIM)
    steps = torch.as_tensor(
        checkpoint["steps"], dtype=torch.float32, device=device
    ).view(1, ATTRIBUTE_DIM)
    quantized, q = _quantized(attrs, centers, steps, hard)
    # Materialize the deployable model first, then reload its dequantized
    # tensors before constructing a single attribute CDF.  This ordering is
    # essential: coding with the training weights and merely saving a smaller
    # side model would create a compact but undecodable stream.
    deployment_mode = str(checkpoint["deployment_training"]["mode"])
    source_deployment_state = model.deployment_state()
    expected_deployment_schema = tensor_schema(source_deployment_state)
    deployment_metadata = write_deployment(
        output / "model.bin",
        source_deployment_state,
        mode=deployment_mode,
        expected_schema=expected_deployment_schema,
    )
    deployed_state = read_deployment(
        output / "model.bin",
        expected_schema=expected_deployment_schema,
        expected_mode=deployment_mode,
        expected_sha256=str(deployment_metadata["sha256"]),
    )
    _install_nonhash_state(model, deployed_state)
    spatial_context = model.spatial_context(xyz, spherical)
    prefix = model.initial_prefix(spatial_context)

    hash_values = model.binary_hash().reshape(-1)
    hash_symbols = (hash_values + 1.0) * 0.5
    _assert_binary_tensor(hash_symbols, "encoder hash")
    hash_bits = encoder(hash_symbols, file_name=str(output / "hash.b"))
    hard_bits = encoder(hard, file_name=str(output / "hard_points.b"))
    attribute_group_schema = _attribute_group_schema(sh_levels)
    attribute_group_streams = _attribute_group_streams(
        attribute_group_schema, arithmetic_chunk_size
    )
    attribute_group_bits: dict[str, int] = {}
    for group_index, (spec, stream) in enumerate(
        zip(GROUP_LAYOUT, attribute_group_streams, strict=True)
    ):
        selection = spec.source_slice
        group_active = active[:, selection]
        mean1, scale1, mean2, scale2, prob1, prob2 = model.group_distribution(
            spatial_context, prefix, levels, hard, group_index
        )
        values = quantized[:, selection][group_active]
        q_values = q[:, selection][group_active]
        _assert_symbol_span(values, q_values, arithmetic_chunk_size)
        if values.numel():
            bits = encoder_gaussian_mixed_chunk(
                values,
                [mean1[group_active], mean2[group_active]],
                [scale1[group_active], scale2[group_active]],
                [prob1[group_active], prob2[group_active]],
                q_values,
                file_name=str(output / stream["file_base"]),
                chunk_size=arithmetic_chunk_size,
            )
        else:
            bits = 0
        attribute_group_bits[spec.name] = int(bits)
        prefix = model.update_prefix(
            prefix,
            quantized[:, selection],
            q[:, selection],
            group_active,
            group_index,
        )
    attr_bits = sum(attribute_group_bits.values())
    sh_level_stream, sh_payloads = build_hierarchical_sh_stream(sh_levels)
    sh_bits = 0
    for stage in sh_level_stream["stages"]:
        if stage["mode"] != "arithmetic":
            continue
        degree = int(stage["degree"])
        mask = torch.from_numpy(sh_payloads[degree].astype(np.float32, copy=False)).to(device)
        sh_bits += encoder(mask, file_name=str(output / str(stage["file"])))

    expected_attributes = canonicalize_decoded_attributes(
        (quantized + centers).cpu().numpy()
    )
    expected_rest = expected_attributes[:, 3:48].reshape(scene.num_points, 15, 3)
    for degree, (begin, end) in enumerate(((0, 3), (3, 8), (8, 15)), start=1):
        expected_rest[sh_levels < degree, begin:end, :] = 0.0
    expected_xyz = dequantize_xyz(xyz_int, xyz_min, xyz_max, geometry_bits)
    expected_xyz = apply_collision_sidecar(
        xyz_int,
        expected_xyz,
        collision_payload,
        xyz_min,
        xyz_max,
        geometry_bits,
    )
    tensor_sha256 = decoded_tensor_digest(
        expected_xyz, expected_attributes, sh_levels, hard_flags
    )
    source_meta = _minimal_source_metadata(scene.metadata)
    manifest = {
        "format": FORMAT,
        "phase": "8.8",
        "upstream_commit": checkpoint["upstream_commit"],
        "num_points": scene.num_points,
        "attribute_dim": ATTRIBUTE_DIM,
        "attribute_layout": list(ATTRIBUTE_LAYOUT),
        "geometry_bits": geometry_bits,
        "quant_bits": int(checkpoint["quant_bits"]),
        "xyz_min": xyz_min.tolist(),
        "xyz_max": xyz_max.tolist(),
        "centers": np.asarray(checkpoint["centers"]).tolist(),
        "steps": np.asarray(checkpoint["steps"]).tolist(),
        "base_steps": np.asarray(checkpoint["base_steps"]).tolist(),
        "arithmetic_chunk_size": arithmetic_chunk_size,
        "hash_numel": int(hash_values.numel()),
        "scene_fingerprint": actual_fingerprint,
        "geometry_order": "stable_morton_3d_60bit_v1",
        "geometry_collisions": geometry_collision_stats(xyz_int),
        "collision_codec": "raw_f32_collision_records_v1",
        "decoded_tensor_sha256": tensor_sha256,
        "rd_profile": checkpoint.get("rd_profile"),
        "quantization_contract": checkpoint["quantization_contract"],
        "sh_selector": checkpoint.get("sh_selector"),
        "hard_selector": checkpoint.get("hard_selector"),
        "spherical_profile": spherical_profile,
        "model_config": checkpoint["model_config"],
        "deployment": deployment_metadata,
        "deployment_training": checkpoint["deployment_training"],
        "sh_level_stream": sh_level_stream,
        "attribute_group_schema": attribute_group_schema,
        "attribute_group_streams": attribute_group_streams,
        "hard_q_scale": HARD_Q_SCALE,
        "attribute_symbol_count": int(active.sum()),
        "improvements": [
            "seam_free_spherical_hash_context",
            "two_gaussian_mixture_entropy_model",
            "phase6.4_hard_point_adaptive_quantization",
            "safe_compact_int8_or_fp16_entropy_decoder",
            "sh_level_sparse_entropy_stream",
            "stable_true_morton_geometry_order",
            "checkpoint_scene_binding",
            "sha256_stream_integrity",
            "native_erp_rate_distortion_quantization_profile",
            "sensitivity_aware_learned_sh_level_selector",
            "panoramic_objective_high_precision_gaussian_selector",
            "conditional_hierarchical_sh_level_stream",
            "causal_semantic_group_entropy_context",
            "decoded_integer_prefix_conditioning",
            "camera_centered_direction_and_log_radius_context",
            "fp32_master_ste_fake_quant_training",
            "encode_after_exact_deployment_reload",
        ],
        "coded_stream_bits": {
            "geometry": len(geometry_stream) * 8,
            "geometry_collisions": len(collision_payload) * 8,
            "hash": int(hash_bits),
            "model": int(deployment_metadata["bytes"]) * 8,
            "attributes": int(attr_bits),
            "attribute_groups": attribute_group_bits,
            "hard_points": int(hard_bits),
            "sh_levels": int(sh_bits),
        },
        "source_metadata": source_meta,
    }
    manifest = write_verified_manifest(output, manifest)
    total_bytes = logical_stream_bytes(output)
    finished = time.perf_counter()
    end_to_end_seconds = finished - end_to_end_started
    return {
        "output": str(output),
        "num_points": scene.num_points,
        "bytes": total_bytes,
        "total_logical_bits": total_bytes * 8,
        "bits_per_gaussian": total_bytes * 8.0 / scene.num_points,
        "seconds": end_to_end_seconds,
        "end_to_end_seconds": end_to_end_seconds,
        "core_codec_seconds": finished - core_codec_started,
    }


@torch.no_grad()
def decode(
    input_path: str | Path,
    output_path: str | Path,
    *,
    tmc3_path: str | Path,
) -> dict[str, Any]:
    end_to_end_started = time.perf_counter()
    device = _device()
    root = Path(input_path).expanduser().resolve()
    decoded_output = Path(output_path).expanduser().resolve()
    if decoded_output.exists() and (
        not decoded_output.is_dir() or any(decoded_output.iterdir())
    ):
        raise FileExistsError(f"Decoded output directory must be empty: {decoded_output}")
    tmc3 = Path(tmc3_path).expanduser().resolve()
    if not tmc3.is_file():
        raise FileNotFoundError(f"Missing TMC13 executable: {tmc3}; run ./build.sh")
    manifest = load_verified_manifest(root, expected_format=FORMAT)
    count = int(manifest["num_points"])
    sh_level_stream, attribute_group_schema, attribute_group_streams = (
        _validate_manifest_schema(manifest)
    )
    preflight_arithmetic_payloads(
        root, manifest, sh_level_stream, attribute_group_streams
    )
    config = ModelConfig.from_dict(manifest["model_config"])
    model = EntropyModel(config).to(device)
    expected_deployment_schema = tensor_schema(model.deployment_state())
    if manifest["deployment"]["schema"] != expected_deployment_schema:
        raise ValueError("Deployment schema changed after manifest validation")
    deployed_state = read_deployment(
        root / "model.bin",
        expected_schema=expected_deployment_schema,
        expected_mode=str(manifest["deployment"]["mode"]),
        expected_sha256=str(manifest["deployment"]["sha256"]),
    )
    _install_nonhash_state(model, deployed_state)
    hash_binary = decoder(int(manifest["hash_numel"]), file_name=str(root / "hash.b"), device="cuda")
    _assert_binary_tensor(hash_binary, "hash")
    model.install_binary_hash(hash_binary.to(torch.float32) * 2.0 - 1.0)
    model.eval()

    core_codec_started = time.perf_counter()
    geometry_bits = int(manifest["geometry_bits"])
    xyz_int_t = decompress_gpcc(
        root / "xyz_gpcc.bin",
        gpcc_codec_path=str(tmc3),
        expected_points=count,
    )
    xyz_values = xyz_int_t.cpu().numpy()
    if (
        not np.isfinite(xyz_values).all()
        or np.any(xyz_values != np.rint(xyz_values))
        or np.any(xyz_values < 0)
        or np.any(xyz_values >= (1 << geometry_bits))
    ):
        raise RuntimeError("GPCC decoded coordinates violate the integer geometry domain")
    xyz_int_unsorted = xyz_values.astype(np.int32)
    order = canonical_morton_order(xyz_int_unsorted)
    xyz_int = xyz_int_unsorted[order]
    if xyz_int.shape[0] != count:
        raise RuntimeError(f"GPCC decoded {xyz_int.shape[0]} points; expected {count}")
    expected_collisions = manifest.get("geometry_collisions")
    actual_collisions = geometry_collision_stats(xyz_int)
    if expected_collisions != actual_collisions:
        raise RuntimeError(
            f"Decoded geometry collision profile differs: {actual_collisions} != {expected_collisions}"
        )
    xyz_min = np.asarray(manifest["xyz_min"], dtype=np.float32)
    xyz_max = np.asarray(manifest["xyz_max"], dtype=np.float32)
    spherical_profile = validate_spherical_profile(manifest["spherical_profile"])
    xyz_context, spherical = _contexts(
        xyz_int, geometry_bits, xyz_min, xyz_max, spherical_profile, device
    )
    spatial_context = model.spatial_context(xyz_context, spherical)
    centers = torch.tensor(manifest["centers"], dtype=torch.float32, device=device).view(1, -1)
    steps = torch.tensor(manifest["steps"], dtype=torch.float32, device=device).view(1, -1)
    hard = decoder(count, file_name=str(root / "hard_points.b"), device="cuda").to(torch.float32)
    _assert_binary_tensor(hard, "hard-point")
    sh_payloads: dict[int, np.ndarray] = {}
    for stage in sh_level_stream["stages"]:
        if stage["mode"] != "arithmetic":
            continue
        degree = int(stage["degree"])
        mask = decoder(
            int(stage["symbol_count"]),
            file_name=str(root / str(stage["file"])),
            device="cuda",
        )
        _assert_binary_tensor(mask, f"sh_ge_{degree}")
        sh_payloads[degree] = mask.detach().cpu().numpy()
    sh_levels = reconstruct_hierarchical_sh_levels(sh_level_stream, sh_payloads)
    attribute_group_schema = validate_group_schema(
        attribute_group_schema, sh_levels=sh_levels, point_count=count
    )
    levels = torch.from_numpy(sh_levels).to(device=device, dtype=torch.int32)
    active = _active_attribute_mask(levels)
    if int(active.sum()) != int(manifest["attribute_symbol_count"]):
        raise RuntimeError("Decoded SH mask does not match the encoded attribute symbol count")
    importance_scale = torch.where(
        hard.view(-1, 1) > 0,
        torch.full((count, 1), float(manifest["hard_q_scale"]), device=device),
        torch.ones((count, 1), device=device),
    )
    q = steps.expand(count, -1) * importance_scale
    quantized = torch.zeros((count, ATTRIBUTE_DIM), dtype=torch.float32, device=device)
    prefix = model.initial_prefix(spatial_context)
    for group_index, (spec, stream, schema_entry) in enumerate(
        zip(
            GROUP_LAYOUT,
            attribute_group_streams,
            attribute_group_schema["groups"],
            strict=True,
        )
    ):
        selection = spec.source_slice
        group_active = active[:, selection]
        if int(group_active.sum()) != int(stream["symbol_count"]):
            raise RuntimeError(f"Decoded active mask disagrees for group {spec.name}")
        if int(schema_entry["active_count"]) * spec.width != int(stream["symbol_count"]):
            raise RuntimeError(f"Decoded group schema disagrees for {spec.name}")
        mean1, scale1, mean2, scale2, prob1, prob2 = model.group_distribution(
            spatial_context, prefix, levels, hard, group_index
        )
        group_quantized = quantized[:, selection]
        if int(stream["symbol_count"]):
            decoded_group = decoder_gaussian_mixed_chunk(
                [mean1[group_active], mean2[group_active]],
                [scale1[group_active], scale2[group_active]],
                [prob1[group_active], prob2[group_active]],
                q[:, selection][group_active],
                file_name=str(root / stream["file_base"]),
                chunk_size=int(manifest["arithmetic_chunk_size"]),
            )
            group_quantized[group_active] = decoded_group
        prefix = model.update_prefix(
            prefix,
            group_quantized,
            q[:, selection],
            group_active,
            group_index,
        )
    attributes = canonicalize_decoded_attributes((quantized + centers).cpu().numpy())
    # Phase6.4's canonical loader requires every inactive SH band to be
    # exactly zero.  Centered quantization can otherwise turn a source zero
    # into a tiny non-zero reconstruction.
    rest = attributes[:, 3:48].reshape(count, 15, 3)
    for degree, (begin, end) in enumerate(((0, 3), (3, 8), (8, 15)), start=1):
        rest[sh_levels < degree, begin:end, :] = 0.0
    xyz = dequantize_xyz(
        xyz_int,
        xyz_min,
        xyz_max,
        geometry_bits,
    )
    collision_path = root / "collision_records.bin"
    maximum_collision_bytes = 16 + 20 * count
    collision_bytes = collision_path.stat().st_size
    if collision_bytes < 16 or collision_bytes > maximum_collision_bytes:
        raise RuntimeError(
            "Collision sidecar size is invalid for the point count"
        )
    with collision_path.open("rb") as collision_file:
        with mmap.mmap(collision_file.fileno(), 0, access=mmap.ACCESS_READ) as collision_payload:
            xyz = apply_collision_sidecar(
                xyz_int,
                xyz,
                collision_payload,
                xyz_min,
                xyz_max,
                geometry_bits,
            )
    hard_np = hard.cpu().numpy().astype(np.uint8)
    actual_tensor_sha256 = decoded_tensor_digest(xyz, attributes, sh_levels, hard_np)
    if actual_tensor_sha256 != manifest.get("decoded_tensor_sha256"):
        raise RuntimeError("Decoded tensor SHA256 mismatch before writing output")
    write_decoded_scene(
        decoded_output,
        xyz,
        attributes,
        sh_levels,
        manifest["source_metadata"],
        codec_metadata={
            "format": manifest["format"],
            "phase": manifest["phase"],
            "geometry_bits": geometry_bits,
            "upstream_commit": manifest["upstream_commit"],
        },
        hard_flags=hard_np,
    )
    reloaded = load_frozen_scene(decoded_output)
    reloaded_sha256 = decoded_tensor_digest(
        reloaded.xyz, reloaded.attributes, reloaded.sh_levels, reloaded.hard_flags
    )
    if reloaded_sha256 != manifest.get("decoded_tensor_sha256"):
        raise RuntimeError("Decoded tensor SHA256 mismatch after frozen-package reload")
    finished = time.perf_counter()
    end_to_end_seconds = finished - end_to_end_started
    return {
        "output": str(decoded_output),
        "num_points": count,
        "seconds": end_to_end_seconds,
        "end_to_end_seconds": end_to_end_seconds,
        "core_codec_seconds": finished - core_codec_started,
    }


def validate(
    source_path: str | Path,
    decoded_path: str | Path,
    *,
    max_points: int | None = None,
) -> dict[str, Any]:
    source = load_frozen_scene(source_path, max_points=max_points)
    decoded = load_frozen_scene(decoded_path)
    if source.num_points != decoded.num_points:
        raise ValueError(f"Point counts differ: {source.num_points} vs {decoded.num_points}")
    hacpp_meta = decoded.metadata.get("hacpp", {})
    if "geometry_bits" not in hacpp_meta:
        raise ValueError("Decoded metadata is missing hacpp.geometry_bits")
    geometry_bits = int(hacpp_meta["geometry_bits"])
    (
        source_xyz_int,
        source_attr,
        source_levels_sorted,
        source_hard,
        xyz_min,
        xyz_max,
    ) = voxelize_and_sort(source, geometry_bits)
    source_sidecar = build_collision_sidecar(source, geometry_bits)
    source_xyz = dequantize_xyz(source_xyz_int, xyz_min, xyz_max, geometry_bits)
    source_xyz = apply_collision_sidecar(
        source_xyz_int,
        source_xyz,
        source_sidecar,
        xyz_min,
        xyz_max,
        geometry_bits,
    )
    unsorted_xyz_int = np.rint(
        (source.xyz - xyz_min)
        / np.maximum(xyz_max - xyz_min, np.float32(1e-12))
        * float((1 << geometry_bits) - 1)
    ).astype(np.int32)
    raw_order = canonical_morton_order(unsorted_xyz_int)
    raw_source_xyz = source.xyz[raw_order]
    decoded_xyz, decoded_attr = decoded.xyz, decoded.attributes
    xyz_error = source_xyz.astype(np.float64) - decoded_xyz.astype(np.float64)
    raw_xyz_error = raw_source_xyz.astype(np.float64) - decoded_xyz.astype(np.float64)
    result = {
        "num_points": source.num_points,
        "xyz_mse": float(np.mean(xyz_error * xyz_error)),
        "xyz_max_abs": float(np.max(np.abs(xyz_error))),
        "codec_roundtrip_xyz_mse": float(np.mean(xyz_error * xyz_error)),
        "codec_roundtrip_xyz_max_abs": float(np.max(np.abs(xyz_error))),
        "raw_to_codec_xyz_mse": float(np.mean(raw_xyz_error * raw_xyz_error)),
        "raw_to_codec_xyz_max_abs": float(np.max(np.abs(raw_xyz_error))),
        "attribute_groups": _group_metrics(source_attr, decoded_attr),
        "source_hard_point_count": int(source_hard.sum()),
        "decoded_hard_point_count": int(decoded.hard_flags.sum()),
        "hard_point_count": int(decoded.hard_flags.sum()),
        "source_active_attribute_symbols": int(
            _active_attribute_mask(torch.from_numpy(source.sh_levels.astype(np.int32))).sum()
        ),
        "decoded_active_attribute_symbols": int(
            _active_attribute_mask(torch.from_numpy(decoded.sh_levels.astype(np.int32))).sum()
        ),
        "sh_level_downgraded_points": int(
            np.count_nonzero(decoded.sh_levels < source_levels_sorted)
        ),
    }
    hard_mask = decoded.hard_flags.astype(bool)
    if hard_mask.any():
        result["hard_attribute_groups"] = _group_metrics(
            source_attr[hard_mask], decoded_attr[hard_mask]
        )
    return result
