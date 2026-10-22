#!/usr/bin/env python3
"""Verify locally rebuilt PanoCompactGS extensions and record their ABI fingerprint."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    modules = {}
    for name, directory in (
        ("arithmetic", ROOT / "third_party/entropy_backend/extensions/arithmetic"),
        ("_gridencoder", ROOT / "third_party/entropy_backend/extensions/gridencoder"),
    ):
        sys.path.insert(0, str(directory))
        module = importlib.import_module(name)
        binary = Path(module.__file__).resolve()
        if not binary.is_relative_to(ROOT) or binary.suffix != ".so":
            raise RuntimeError(f"{name} resolved outside this snapshot: {binary}")
        modules[name] = {
            "path": str(binary.relative_to(ROOT)),
            "bytes": binary.stat().st_size,
            "sha256": _sha256(binary),
        }

    payload = {
        "format": "phase8_hacpp_build_environment_v1",
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
        "cuda_available": torch.cuda.is_available(),
        "extensions": modules,
    }
    destination = ROOT / "results/build_environment.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
