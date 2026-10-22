#!/usr/bin/env python3
"""Reproduce Phase8.8 compact-model component sizes without a CUDA device."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import tempfile

import torch

from pcgs.deployment import (
    MODE_FP16,
    MODE_INT8,
    deployment_nbytes_for_schema,
    tensor_schema,
    write_deployment,
)
from pcgs.model import EntropyModel, ModelConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()

    torch.manual_seed(88)
    model = EntropyModel(ModelConfig()).eval()
    state = model.deployment_state()
    schema = tensor_schema(state)
    result = {
        "format": "phase8.8_deployment_size_probe_v2",
        "seed": 88,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "model_config": model.config.to_dict(),
        "trainable_parameters_including_separate_hash": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "deployment_tensor_count": len(state),
        "deployment_element_count": sum(value.numel() for value in state.values()),
        "containers": {},
    }
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for mode in (MODE_INT8, MODE_FP16):
            path = root / f"model-{mode}.bin"
            metadata = write_deployment(
                path, state, mode=mode, expected_schema=schema
            )
            expected_bytes = deployment_nbytes_for_schema(schema, mode)
            if path.stat().st_size != expected_bytes:
                raise RuntimeError("serialized size differs from schema-derived size")
            result["containers"][mode] = {
                "bytes": expected_bytes,
                "decimal_mb": round(expected_bytes / 1_000_000, 6),
                "mib": round(expected_bytes / (1024 * 1024), 6),
                "sha256": metadata["sha256"],
            }

    fp16_bytes = int(result["containers"][MODE_FP16]["bytes"])
    int8_bytes = int(result["containers"][MODE_INT8]["bytes"])
    saved_bytes = fp16_bytes - int8_bytes
    result["int8_savings_vs_fp16"] = {
        "bytes": saved_bytes,
        "decimal_mb": round(saved_bytes / 1_000_000, 6),
        "mib": round(saved_bytes / (1024 * 1024), 6),
        "percent": round(100.0 * saved_bytes / fp16_bytes, 4),
    }

    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
