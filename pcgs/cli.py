"""Command line interface for the direct PanoCompactGS → Phase6.4 port."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TMC3 = ROOT / "third_party" / "mpeg-pcc-tmc13" / "build" / "tmc3" / "tmc3"


def _json(value) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=True))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Direct PanoCompactGS port for Phase6.4 frozen Gaussians")
    sub = result.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train PanoCompactGS hash-grid entropy context")
    train_p.add_argument("--input", required=True)
    train_p.add_argument("--checkpoint", required=True)
    train_p.add_argument("--steps", type=int, default=2000)
    train_p.add_argument("--batch-size", type=int, default=4096)
    train_p.add_argument("--learning-rate", type=float, default=1e-3)
    train_p.add_argument("--quant-bits", type=int, default=10)
    train_p.add_argument("--geometry-bits", type=int, default=20)
    train_p.add_argument("--max-points", type=int)
    train_p.add_argument("--seed", type=int, default=0)
    train_p.add_argument(
        "--rd-profile",
        help=(
            "Versioned native ERP multiplier profile; preserves the 56 per-channel "
            "base steps and rejects legacy absolute steps.json files"
        ),
    )
    train_p.add_argument("--sh-profile", help="Phase8.5 learned SH-level .npz profile")
    train_p.add_argument(
        "--hard-profile",
        help="Source-bound per-Gaussian high-precision .npz profile",
    )
    train_p.add_argument(
        "--cameras",
        help=(
            "Optional Phase6.4 cameras.json. Its camera-position geometric median "
            "defines the spherical origin; omission explicitly uses the geometry "
            "bounding-box centre"
        ),
    )
    train_p.add_argument(
        "--model-precision",
        choices=("int8", "fp16"),
        default="int8",
        help="Fake-quantized training and self-contained decoder-model precision",
    )

    encode_p = sub.add_parser("encode", help="Create the PanoCompactGS bitstream directory")
    encode_p.add_argument("--input", required=True)
    encode_p.add_argument("--checkpoint", required=True)
    encode_p.add_argument("--output", required=True)
    encode_p.add_argument("--tmc3", default=str(DEFAULT_TMC3))
    encode_p.add_argument("--max-points", type=int)
    encode_p.add_argument("--chunk-size", type=int, default=2_000)

    decode_p = sub.add_parser("decode", help="Decode to a Phase6.4 frozen package")
    decode_p.add_argument("--input", required=True)
    decode_p.add_argument("--output", required=True)
    decode_p.add_argument("--tmc3", default=str(DEFAULT_TMC3))

    validate_p = sub.add_parser("validate", help="Compare source and decoded frozen packages")
    validate_p.add_argument("--source", required=True)
    validate_p.add_argument("--decoded", required=True)
    validate_p.add_argument("--max-points", type=int)

    verify_p = sub.add_parser("verify", help="Verify stream manifest and payload checksums without CUDA")
    verify_p.add_argument("--input", required=True)

    sh_p = sub.add_parser("learn-sh", help="Learn sensitivity-aware per-point SH levels")
    sh_p.add_argument("--input", required=True)
    sh_p.add_argument("--output", required=True)
    sh_p.add_argument("--steps", type=int, default=500)
    sh_p.add_argument("--batch-size", type=int, default=4096)
    sh_p.add_argument("--learning-rate", type=float, default=2e-3)
    sh_p.add_argument("--lambda-rate", type=float, default=0.05)
    sh_p.add_argument("--bits-per-symbol", type=float, default=8.0)
    sh_p.add_argument("--importance", help="Optional native point_importance_f32.bin")
    sh_p.add_argument(
        "--importance-kind",
        help="Auditable semantic label for --importance",
    )
    sh_p.add_argument("--hard-weight", type=float, default=4.0)
    sh_p.add_argument(
        "--selection-mode",
        choices=("learned", "oracle"),
        default="learned",
        help="Encoder-only SH decision rule; oracle writes the deterministic proxy argmin",
    )
    sh_p.add_argument(
        "--hard-floor",
        type=int,
        choices=(0, 1, 2, 3),
        default=0,
        help="Minimum source-bounded SH level for hard points (0 disables)",
    )
    sh_p.add_argument("--seed", type=int, default=0)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "train":
        from .codec import train
        result = train(
            args.input,
            args.checkpoint,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            quant_bits=args.quant_bits,
            geometry_bits=args.geometry_bits,
            max_points=args.max_points,
            seed=args.seed,
            rd_profile=args.rd_profile,
            sh_profile=args.sh_profile,
            hard_profile=args.hard_profile,
            cameras_path=args.cameras,
            deployment_mode=args.model_precision,
        )
    elif args.command == "encode":
        from .codec import encode
        result = encode(
            args.input,
            args.checkpoint,
            args.output,
            tmc3_path=args.tmc3,
            max_points=args.max_points,
            arithmetic_chunk_size=args.chunk_size,
        )
    elif args.command == "decode":
        from .codec import decode
        result = decode(args.input, args.output, tmc3_path=args.tmc3)
    elif args.command == "validate":
        from .codec import validate
        result = validate(args.source, args.decoded, max_points=args.max_points)
    elif args.command == "learn-sh":
        from .sh_selector import learn_sh_profile

        result = learn_sh_profile(
            args.input,
            args.output,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            lambda_rate=args.lambda_rate,
            bits_per_symbol=args.bits_per_symbol,
            importance_path=args.importance,
            importance_kind=args.importance_kind,
            hard_weight=args.hard_weight,
            hard_floor=args.hard_floor,
            seed=args.seed,
            selection_mode=args.selection_mode,
        )
    else:
        from .codec import (
            FORMAT,
            _validate_manifest_schema,
            preflight_arithmetic_payloads,
        )
        from .deployment import read_deployment, tensor_schema
        from .integrity import load_verified_manifest, logical_stream_bytes
        from .model import EntropyModel, ModelConfig

        manifest = load_verified_manifest(args.input, expected_format=FORMAT)
        sh_level_stream, attribute_group_schema, attribute_group_streams = (
            _validate_manifest_schema(manifest)
        )
        arithmetic_preflight = preflight_arithmetic_payloads(
            args.input, manifest, sh_level_stream, attribute_group_streams
        )
        model = EntropyModel(ModelConfig.from_dict(manifest["model_config"]))
        expected_schema = tensor_schema(model.deployment_state())
        read_deployment(
            Path(args.input).expanduser().resolve() / "model.bin",
            expected_schema=expected_schema,
            expected_mode=str(manifest["deployment"]["mode"]),
            expected_sha256=str(manifest["deployment"]["sha256"]),
        )
        result = {
            "input": str(Path(args.input).expanduser().resolve()),
            "format": manifest["format"],
            "verified": True,
            "logical_bytes": logical_stream_bytes(args.input),
            "payload_files": manifest["integrity"]["payload_file_count"],
            "sh_level_stream": sh_level_stream,
            "attribute_group_schema": attribute_group_schema,
            "attribute_group_streams": attribute_group_streams,
            "deployment": manifest["deployment"],
            "arithmetic_preflight": arithmetic_preflight,
        }
    _json(result)


if __name__ == "__main__":
    main()
