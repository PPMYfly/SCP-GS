"""Load the vendored, upstream PanoCompactGS CUDA/Python implementation."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "third_party" / "entropy_backend"
GRID_ROOT = BACKEND_ROOT / "extensions" / "gridencoder"
ARITHMETIC_ROOT = BACKEND_ROOT / "extensions" / "arithmetic"


def configure_imports() -> None:
    """Make PanoCompactGS's original top-level packages importable without installation."""

    for path in (ARITHMETIC_ROOT, GRID_ROOT, BACKEND_ROOT):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


configure_imports()

try:
    from backend_utils.encodings import GridEncoder, STE_binary, STE_multistep
    from backend_utils.encodings_cuda import (
        _read_checked_arithmetic_stream,
        decoder,
        decoder_gaussian_chunk,
        decoder_gaussian_mixed_chunk,
        encoder,
        encoder_gaussian_chunk,
        encoder_gaussian_mixed_chunk,
    )
    from backend_utils.entropy_models import Entropy_gaussian, Entropy_gaussian_mix_prob_2
    from backend_utils.gpcc_utils import calculate_morton_order, compress_gpcc, decompress_gpcc
except ImportError as exc:  # pragma: no cover - error text is exercised by CLI
    raise ImportError(
        "PanoCompactGS CUDA extensions are unavailable. Run ./build.sh first. "
        f"Original error: {exc}"
    ) from exc


__all__ = [
    "ARITHMETIC_ROOT",
    "GRID_ROOT",
    "BACKEND_ROOT",
    "Entropy_gaussian",
    "Entropy_gaussian_mix_prob_2",
    "GridEncoder",
    "STE_binary",
    "STE_multistep",
    "_read_checked_arithmetic_stream",
    "calculate_morton_order",
    "compress_gpcc",
    "decoder",
    "decoder_gaussian_chunk",
    "decoder_gaussian_mixed_chunk",
    "decompress_gpcc",
    "encoder",
    "encoder_gaussian_chunk",
    "encoder_gaussian_mixed_chunk",
]
