"""Sensitivity-aware learned SH-level selector for Phase8.5."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import FrozenScene, load_frozen_scene, scene_fingerprint


ACTIVE_DIMS = np.asarray([11, 20, 35, 56], dtype=np.int64)
SH_DIMS = np.asarray([0, 9, 24, 45], dtype=np.float32)
RENDERER_BAND_IMPORTANCE_KIND = "renderer_sh_counterfactual_band_first_order_v1"
RENDERER_BAND_IMPORTANCE_KINDS = frozenset(
    {
        RENDERER_BAND_IMPORTANCE_KIND,
        "renderer_sh_counterfactual_band_first_order_planar_v2",
        "renderer_sh_counterfactual_band_first_order_sphere_v2",
        "renderer_sh_counterfactual_band_signed_integrated_planar_v2",
        "renderer_sh_counterfactual_band_signed_integrated_sphere_v2",
    }
)
SELECTION_MODES = ("learned", "oracle")


class SHLevelSelector(nn.Module):
    def __init__(self, input_dim: int = 7, hidden_dim: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def apply_hard_sh_floor(
    learned_levels: np.ndarray,
    source_levels: np.ndarray,
    hard_flags: np.ndarray,
    hard_floor: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Apply a source-bounded minimum SH level to hard points only.

    The operation is deterministic, never activates a band absent from the
    source representation, and reports the exact added active-symbol budget.
    """

    if isinstance(hard_floor, bool) or not isinstance(hard_floor, (int, np.integer)):
        raise ValueError("hard_floor must be an integer in [0, 3]")
    floor = int(hard_floor)
    if floor < 0 or floor > 3:
        raise ValueError("hard_floor must be an integer in [0, 3]")
    learned = np.asarray(learned_levels, dtype=np.int64)
    source = np.asarray(source_levels, dtype=np.int64)
    hard = np.asarray(hard_flags).reshape(-1) > 0
    if learned.ndim != 1 or source.shape != learned.shape or hard.shape != learned.shape:
        raise ValueError("learned/source/hard arrays must be aligned one-dimensional vectors")
    if np.any((learned < 0) | (learned > 3)) or np.any((source < 0) | (source > 3)):
        raise ValueError("learned/source SH levels must be in [0, 3]")
    if np.any(learned > source):
        raise ValueError("learned SH levels may not exceed source levels")
    adjusted = learned.copy()
    if floor > 0:
        adjusted[hard] = np.minimum(source[hard], np.maximum(adjusted[hard], floor))
    changed = adjusted != learned
    diagnostics = {
        "hard_floor": floor,
        "hard_point_count": int(np.count_nonzero(hard)),
        "hard_floor_eligible_points": int(np.count_nonzero(hard & (source >= floor))),
        "hard_floor_adjusted_points": int(np.count_nonzero(changed)),
        "hard_floor_added_active_symbols": int(
            np.sum(ACTIVE_DIMS[adjusted] - ACTIVE_DIMS[learned])
        ),
    }
    return adjusted.astype(np.uint8), diagnostics


def _band_energy(scene: FrozenScene) -> np.ndarray:
    rest = scene.attributes[:, 3:48].reshape(scene.num_points, 15, 3).astype(np.float64)
    bands = (rest[:, 0:3], rest[:, 3:8], rest[:, 8:15])
    return np.stack([np.sum(band * band, axis=(1, 2)) for band in bands], axis=1)


def _selector_features(
    scene: FrozenScene,
    energies: np.ndarray,
    band_importance: np.ndarray | None = None,
) -> np.ndarray:
    low = scene.xyz.min(axis=0)
    extent = np.maximum(scene.xyz.max(axis=0) - low, np.float32(1e-12))
    xyz = (scene.xyz - low) / extent
    log_energy = np.log1p(energies)
    parts = [
        xyz.astype(np.float64),
        log_energy,
        scene.hard_flags[:, None].astype(np.float64),
    ]
    if band_importance is not None:
        band = np.asarray(band_importance, dtype=np.float64)
        if band.shape != (scene.num_points, 3):
            raise ValueError(
                f"Band importance shape {band.shape}; expected ({scene.num_points}, 3)"
            )
        band_mean = np.maximum(band.mean(axis=0, keepdims=True), 1e-12)
        parts.append(np.log1p(band / band_mean))
    values = np.concatenate(parts, axis=1)
    mean = values.mean(axis=0, keepdims=True)
    std = np.maximum(values.std(axis=0, keepdims=True), 1e-6)
    return ((values - mean) / std).astype(np.float32)


def _importance(
    scene: FrozenScene,
    path: str | Path | None,
    hard_weight: float,
    importance_kind: str | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    if path is None:
        value = np.ones(scene.num_points, dtype=np.float64)
        band_value = None
    else:
        raw = np.fromfile(Path(path).expanduser().resolve(), dtype=np.float32).astype(np.float64)
        if raw.size == scene.num_points:
            value = raw
            band_value = None
        elif (
            raw.size == scene.num_points * 3
            and importance_kind in RENDERER_BAND_IMPORTANCE_KINDS
        ):
            value = np.ones(scene.num_points, dtype=np.float64)
            band_value = raw.reshape(scene.num_points, 3)
        else:
            raise ValueError(
                "Importance contains "
                f"{raw.size} float32 values; expected {scene.num_points} point "
                f"values or {scene.num_points * 3} values with importance_kind="
                f"one of {sorted(RENDERER_BAND_IMPORTANCE_KINDS)}"
            )
        if not np.isfinite(raw).all() or np.any(raw < 0):
            raise ValueError("Importance must be finite and non-negative")
        if band_value is None:
            positive_mean = float(value.mean())
            value = value / max(positive_mean, 1e-12)
    hard_scale = np.where(scene.hard_flags > 0, float(hard_weight), 1.0)
    value *= hard_scale
    if band_value is not None:
        band_value = band_value * hard_scale[:, None]
    return value, band_value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _ordinal_level_rate_bits(source_levels: np.ndarray) -> np.ndarray:
    """Approximate a nested ge1/ge2/ge3 stream with conditional Bernoulli costs."""

    source = np.clip(np.asarray(source_levels, dtype=np.int64), 0, 3)
    count = max(int(source.size), 1)
    # Jeffreys smoothing avoids zero-cost or infinite-cost branches when a
    # source scene has no level-3 points (the office scene is such a case).
    p1 = (float(np.count_nonzero(source >= 1)) + 0.5) / (count + 1.0)
    parent1 = max(int(np.count_nonzero(source >= 1)), 1)
    p2 = (float(np.count_nonzero(source >= 2)) + 0.5) / (parent1 + 1.0)
    parent2 = max(int(np.count_nonzero(source >= 2)), 1)
    p3 = (float(np.count_nonzero(source >= 3)) + 0.5) / (parent2 + 1.0)
    probabilities = (p1, p2, p3)
    rate = np.zeros(4, dtype=np.float64)
    for level in range(4):
        total = 0.0
        for degree, probability in enumerate(probabilities, start=1):
            if level < degree - 1:
                break
            outcome = level >= degree
            selected = probability if outcome else 1.0 - probability
            total -= math.log2(max(selected, 1e-12))
            if not outcome:
                break
        rate[level] = total
    return rate.astype(np.float32)


def candidate_costs(
    scene: FrozenScene,
    *,
    lambda_rate: float,
    bits_per_symbol: float,
    importance_path: str | Path | None = None,
    importance_kind: str | None = None,
    hard_weight: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build four per-point RD costs, never increasing the source SH level."""

    if lambda_rate < 0 or bits_per_symbol <= 0 or hard_weight <= 0:
        raise ValueError("lambda_rate must be >=0; bits_per_symbol and hard_weight must be >0")
    energies = _band_energy(scene)
    importance, band_importance = _importance(
        scene, importance_path, hard_weight, importance_kind
    )
    if band_importance is None:
        dropped = np.stack(
            [
                energies[:, 0] + energies[:, 1] + energies[:, 2],
                energies[:, 1] + energies[:, 2],
                energies[:, 2],
                np.zeros(scene.num_points, dtype=np.float64),
            ],
            axis=1,
        )
        distortion = dropped * importance[:, None]
    else:
        distortion = np.stack(
            [
                band_importance[:, 0] + band_importance[:, 1] + band_importance[:, 2],
                band_importance[:, 1] + band_importance[:, 2],
                band_importance[:, 2],
                np.zeros(scene.num_points, dtype=np.float64),
            ],
            axis=1,
        )
    scale = max(float(np.mean(distortion[:, 0])), 1e-12)
    distortion /= scale
    per_point_symbol_bits = float(bits_per_symbol) + (scene.hard_flags > 0).astype(np.float32)
    coefficient_rate = per_point_symbol_bits[:, None] * SH_DIMS[None, :]
    level_rate = _ordinal_level_rate_bits(scene.sh_levels)[None, :]
    rate = coefficient_rate + level_rate
    costs = distortion + float(lambda_rate) * rate
    source_levels = np.clip(scene.sh_levels.astype(np.int64), 0, 3)
    candidates = np.arange(4, dtype=np.int64)[None, :]
    # A fixed 1e6 sentinel becomes smaller than a valid cost at large lambda,
    # allowing an argmin to illegally increase the source level.  Keep a
    # finite value (so 0 * cost stays well-defined in the soft expectation)
    # but place it far above every supported proxy configuration.
    costs[candidates > source_levels[:, None]] = 1e30
    return costs.astype(np.float32), energies


def direct_oracle_levels(costs: np.ndarray, source_levels: np.ndarray) -> np.ndarray:
    """Return the deterministic source-bounded minimum-cost SH level vector."""

    values = np.asarray(costs)
    source = np.asarray(source_levels, dtype=np.int64).reshape(-1)
    if values.shape != (source.size, 4) or not np.isfinite(values).all():
        raise ValueError("costs must be a finite [num_points,4] array")
    if np.any((source < 0) | (source > 3)):
        raise ValueError("source SH levels must be in [0, 3]")
    result = np.argmin(values, axis=1).astype(np.uint8)
    if np.any(result > source):
        raise ValueError("minimum-cost SH levels exceed the source-level bound")
    return result


def learn_sh_profile(
    input_path: str | Path,
    output_path: str | Path,
    *,
    steps: int = 500,
    batch_size: int = 4096,
    learning_rate: float = 2e-3,
    lambda_rate: float = 0.05,
    bits_per_symbol: float = 8.0,
    importance_path: str | Path | None = None,
    importance_kind: str | None = None,
    hard_weight: float = 4.0,
    hard_floor: int = 0,
    seed: int = 0,
    selection_mode: str = "learned",
) -> dict[str, Any]:
    """Select source-bounded SH levels with an MLP or deterministic cost oracle."""

    if steps <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("steps, batch_size and learning_rate must be positive")
    if selection_mode not in SELECTION_MODES:
        raise ValueError(f"selection_mode must be one of {SELECTION_MODES}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    scene = load_frozen_scene(input_path)
    importance_record: dict[str, Any]
    if importance_path is None:
        if importance_kind is not None:
            raise ValueError("importance_kind requires importance_path")
        importance_record = {
            "importance": "uniform_plus_hard",
            "importance_sha256": None,
        }
    else:
        resolved_importance = Path(importance_path).expanduser().resolve(strict=True)
        resolved_kind = importance_kind or "external_point_importance"
        if not resolved_kind or any(character.isspace() for character in resolved_kind):
            raise ValueError("importance_kind must be a non-empty token without whitespace")
        importance_record = {
            "importance": resolved_kind,
            "importance_sha256": _sha256_file(resolved_importance),
        }
        importance_path = resolved_importance
    costs_np, energies = candidate_costs(
        scene,
        lambda_rate=lambda_rate,
        bits_per_symbol=bits_per_symbol,
        importance_path=importance_path,
        importance_kind=importance_kind,
        hard_weight=hard_weight,
    )
    _, band_importance = _importance(
        scene, importance_path, hard_weight, importance_kind
    )
    features_np = _selector_features(scene, energies, band_importance)
    features = torch.from_numpy(features_np)
    costs = torch.from_numpy(costs_np)
    oracle = direct_oracle_levels(costs_np, scene.sh_levels)
    target = torch.from_numpy(oracle.astype(np.int64))
    source_levels_tensor = torch.from_numpy(
        np.clip(scene.sh_levels.astype(np.int64), 0, 3)
    )
    level_ids = torch.arange(4).view(1, 4)
    count = scene.num_points
    if selection_mode == "learned":
        model = SHLevelSelector(input_dim=features.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        generator = torch.Generator().manual_seed(seed)
        for _ in range(steps):
            index = torch.randint(0, count, (min(batch_size, count),), generator=generator)
            logits = model(features[index])
            valid = level_ids <= source_levels_tensor[index, None]
            masked_logits = logits.masked_fill(~valid, -1e9)
            probabilities = torch.softmax(masked_logits, dim=1)
            expected_cost = torch.sum(probabilities * costs[index], dim=1).mean()
            classification = F.cross_entropy(masked_logits, target[index])
            loss = expected_cost + 0.1 * classification
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            logits = model(features)
            valid = level_ids <= source_levels_tensor[:, None]
            learned = torch.argmax(
                logits.masked_fill(~valid, -1e9), dim=1
            ).cpu().numpy().astype(np.uint8)
        optimizer_steps_executed = int(steps)
    else:
        learned = oracle.copy()
        optimizer_steps_executed = 0
    source_levels = np.clip(scene.sh_levels, 0, 3).astype(np.uint8)
    # A learned level may never activate a band absent from the source package.
    learned = np.minimum(learned, source_levels)
    rows = np.arange(count)
    learned_point_cost = costs_np[rows, learned]
    source_point_cost = costs_np[rows, source_levels]
    fallback = learned_point_cost > source_point_cost
    learned[fallback] = source_levels[fallback]
    learned, hard_floor_diagnostics = apply_hard_sh_floor(
        learned, source_levels, scene.hard_flags, hard_floor
    )
    learned_cost = float(np.mean(costs_np[np.arange(count), learned]))
    source_cost = float(np.mean(costs_np[np.arange(count), source_levels]))
    oracle_cost = float(np.mean(costs_np[np.arange(count), oracle]))
    level_rate = _ordinal_level_rate_bits(source_levels)
    symbol_rate = (
        (float(bits_per_symbol) + (scene.hard_flags > 0).astype(np.float32))[:, None]
        * SH_DIMS[None, :]
        + level_rate[None, :]
    )
    metadata = {
        "format": "phase8.5_learned_sh_levels_v1",
        "scene_fingerprint": scene_fingerprint(scene),
        "num_points": count,
        "steps": int(steps),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "selection_mode": selection_mode,
        "optimizer_steps_executed": optimizer_steps_executed,
        "lambda_rate": float(lambda_rate),
        "bits_per_symbol_proxy": float(bits_per_symbol),
        "hard_weight": float(hard_weight),
        **hard_floor_diagnostics,
        "hard_floor_application": "after_source_fallback;source_level_bounded;hard_points_only",
        **importance_record,
        "importance_layout": (
            "point_x_sh_band_direct_counterfactual_cost"
            if band_importance is not None
            else "point_scalar_times_band_energy"
        ),
        "selector_input_dim": int(features.shape[1]),
        "source_active_symbols": int(ACTIVE_DIMS[source_levels].sum()),
        "learned_active_symbols": int(ACTIVE_DIMS[learned].sum()),
        "downgraded_points": int(np.count_nonzero(learned < source_levels)),
        "source_fallback_points": int(np.count_nonzero(fallback)),
        "oracle_agreement": float(np.mean(learned == oracle)),
        "level_vector_sha256": hashlib.sha256(learned.tobytes()).hexdigest(),
        "source_proxy_cost": source_cost,
        "learned_proxy_cost": learned_cost,
        "oracle_proxy_cost": oracle_cost,
        "source_proxy_rate_bits": float(np.sum(symbol_rate[rows, source_levels])),
        "learned_proxy_rate_bits": float(np.sum(symbol_rate[rows, learned])),
        "ordinal_level_rate_bits": level_rate.tolist(),
    }
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        levels=learned,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return {"output": str(output), **metadata}


def load_sh_profile(
    path: str | Path,
    *,
    expected_fingerprint: str,
    expected_points: int,
) -> tuple[np.ndarray, dict]:
    with np.load(Path(path).expanduser().resolve(), allow_pickle=False) as archive:
        levels = np.asarray(archive["levels"], dtype=np.uint8)
        metadata = json.loads(str(archive["metadata"].item()))
    if metadata.get("format") != "phase8.5_learned_sh_levels_v1":
        raise ValueError("Unsupported SH profile format")
    if metadata.get("scene_fingerprint") != expected_fingerprint:
        raise ValueError("SH profile/input scene fingerprint mismatch")
    if levels.shape != (expected_points,):
        raise ValueError(f"SH profile shape {levels.shape} != ({expected_points},)")
    if np.any(levels > 3):
        raise ValueError("SH profile levels must be in [0, 3]")
    return levels.astype(np.int32), metadata
