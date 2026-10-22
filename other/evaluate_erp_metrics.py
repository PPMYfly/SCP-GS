#!/usr/bin/env python3
"""Standalone ERP image metrics for the 43-scene release."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _images(root: Path, *, ground_truth: bool) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {root}")
    result: dict[str, Path] = {}
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        key = path.stem
        if ground_truth and key.endswith("_gt"):
            key = key[:-3]
        if key in result:
            raise ValueError(f"duplicate normalized image id {key!r} in {root}")
        result[key] = path
    if not result:
        raise ValueError(f"no supported images in {root}")
    return result


def _rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / np.float32(255.0)


def _psnr(error: np.ndarray, weights: np.ndarray | None = None) -> float:
    squared = np.square(error.astype(np.float64, copy=False))
    if weights is None:
        mse = float(np.mean(squared))
    else:
        mse = float(
            np.sum(squared * weights[:, None, None])
            / (np.sum(weights) * squared.shape[1] * squared.shape[2])
        )
    return 99.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)


def _ssim(prediction: np.ndarray, reference: np.ndarray) -> tuple[float, np.ndarray]:
    _, score_map = structural_similarity(
        reference,
        prediction,
        channel_axis=2,
        data_range=1.0,
        full=True,
    )
    if score_map.ndim == 3:
        score_map = score_map.mean(axis=2)
    # Match the established Phase6.4 protocol: both ordinary and weighted
    # SSIM reduce the same full-resolution map (unweighted vs latitude-weighted).
    value = float(np.mean(score_map))
    return value, np.asarray(score_map, dtype=np.float64)


def _latitude_weights(height: int) -> np.ndarray:
    row = np.arange(height, dtype=np.float64)
    latitude = math.pi * (row + 0.5) / height - math.pi / 2.0
    return np.maximum(np.cos(latitude), 1e-12)


def _lpips(prediction: np.ndarray, reference: np.ndarray, model: Any, device: Any) -> float:
    import torch

    pred = torch.from_numpy(prediction).permute(2, 0, 1).unsqueeze(0).to(device)
    ref = torch.from_numpy(reference).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        return float(model(pred * 2.0 - 1.0, ref * 2.0 - 1.0).item())


def _render_time(path: Path | None, warmup: int) -> tuple[float | None, float | None, int]:
    if warmup < 0:
        raise ValueError("warmup frames must be non-negative")
    if path is None or not path.is_file():
        return None, None, 0
    values: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            values.append(float(line.split()[-1]))
        except (IndexError, ValueError):
            continue
    values = values[warmup:]
    if not values:
        return None, None, 0
    mean_ms = float(np.mean(values))
    return mean_ms, (1000.0 / mean_ms if mean_ms > 0.0 else None), len(values)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    pred_root = Path(args.pred_dir).expanduser().resolve()
    gt_root = Path(args.gt_dir).expanduser().resolve()
    predictions = _images(pred_root, ground_truth=False)
    references = _images(gt_root, ground_truth=True)
    missing = sorted(set(references) - set(predictions))
    extra = sorted(set(predictions) - set(references))
    if missing or extra:
        raise ValueError(
            f"image ids differ: missing_predictions={missing[:8]}, extra_predictions={extra[:8]}"
        )

    lpips_model = None
    device = None
    if args.compute_lpips:
        import lpips
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("LPIPS was requested but CUDA is unavailable")
        device = torch.device("cuda")
        # LPIPS defaults to verbose model-loading messages on stdout.  Stdout is
        # the machine-readable result channel for the batch runner, so keep the
        # dependency quiet and reserve diagnostics for exceptions/stderr.
        lpips_model = lpips.LPIPS(net=args.lpips_backbone, verbose=False).to(device).eval()

    rows: list[dict[str, Any]] = []
    for key in sorted(references):
        prediction = _rgb(predictions[key])
        reference = _rgb(references[key])
        if prediction.shape != reference.shape:
            raise ValueError(
                f"resolution mismatch for {key}: {prediction.shape} != {reference.shape}; resizing is forbidden"
            )
        weights = _latitude_weights(prediction.shape[0])
        error = prediction - reference
        ssim, score_map = _ssim(prediction, reference)
        ws_ssim = float(
            np.sum(score_map * weights[:, None]) / (np.sum(weights) * score_map.shape[1])
        )
        rows.append(
            {
                "image_id": key,
                "width": int(prediction.shape[1]),
                "height": int(prediction.shape[0]),
                "psnr": _psnr(error),
                "ssim": ssim,
                "ws_psnr": _psnr(error, weights),
                "ws_ssim": ws_ssim,
                "lpips": (
                    _lpips(prediction, reference, lpips_model, device)
                    if lpips_model is not None
                    else None
                ),
            }
        )

    def mean(name: str) -> float | None:
        values = [float(row[name]) for row in rows if row[name] is not None]
        return float(np.mean(values)) if values else None

    render_path = Path(args.render_time_file).resolve() if args.render_time_file else None
    render_ms, fps, timing_frames = _render_time(render_path, args.warmup_frames)
    summary = {
        "format": "phase8.8_erp_metrics_v1",
        "dataset": args.dataset,
        "scene": args.scene,
        "num_images": len(rows),
        "resolution": {"width": rows[0]["width"], "height": rows[0]["height"]},
        "psnr": mean("psnr"),
        "ssim": mean("ssim"),
        "ws_psnr": mean("ws_psnr"),
        "ws_ssim": mean("ws_ssim"),
        "lpips": mean("lpips"),
        "lpips_backbone": args.lpips_backbone if args.compute_lpips else None,
        "render_ms": render_ms,
        "fps": fps,
        "timing_frames": timing_frames,
        "pred_dir": str(pred_root),
        "gt_dir": str(gt_root),
        "metric_contract": {
            "rgb_range": "decoded_8bit_rgb_[0,1]",
            "resize": "forbidden",
            "ws_weight": "cosine_of_erp_pixel_center_latitude",
            "ssim_reduction": "mean_of_full_skimage_score_map_phase6.4_compatible",
            "scene_reduction": "arithmetic_mean_over_matched_test_views",
        },
    }

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    per_image = output.parent / "per_image_metrics.csv"
    with per_image.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary["per_image_csv"] = str(per_image)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute standalone formal ERP metrics")
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--render-time-file", default="")
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--compute-lpips", action="store_true")
    parser.add_argument("--lpips-backbone", choices=("alex", "vgg", "squeeze"), default="alex")
    result = evaluate(parser.parse_args())
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
