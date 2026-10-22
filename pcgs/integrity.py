"""Self-contained Phase8.3 stream integrity and logical-size helpers."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any


MANIFEST_NAME = "manifest.json"
MANIFEST_DIGEST_NAME = "manifest.sha256"
INTEGRITY_FORMAT = "sha256-tree-v1"
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_MANIFEST_DIGEST_BYTES = 256


def _payload_files(root: Path) -> list[Path]:
    excluded = {MANIFEST_NAME, MANIFEST_DIGEST_NAME}
    result: list[Path] = []
    for path in root.iterdir():
        if path.name in excluded:
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Stream control member is not a regular file: {path.name}")
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Stream member is not a regular file: {path.name}")
        result.append(path)
    result.sort(key=lambda path: path.name)
    for path in result:
        if path.name in {".", ".."} or "/" in path.name or "\\" in path.name:
            raise ValueError(f"Unsafe stream member name: {path.name!r}")
    return result


def _digest_payload(root: Path) -> tuple[str, list[dict[str, int | str]]]:
    digest = hashlib.sha256()
    digest.update(b"phase8.3-payload-tree-v1\0")
    records: list[dict[str, int | str]] = []
    for path in _payload_files(root):
        name = path.name.encode("utf-8")
        size = path.stat().st_size
        digest.update(struct.pack("<I", len(name)))
        digest.update(name)
        digest.update(struct.pack("<Q", size))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        records.append({"name": path.name, "bytes": int(size)})
    return digest.hexdigest(), records


def logical_stream_bytes(root: str | Path) -> int:
    root = Path(root)
    return sum(path.stat().st_size for path in root.iterdir() if path.is_file())


def write_verified_manifest(root: str | Path, manifest: dict[str, Any]) -> dict[str, Any]:
    root = Path(root)
    payload_sha256, files = _digest_payload(root)
    result = dict(manifest)
    result["integrity"] = {
        "format": INTEGRITY_FORMAT,
        "payload_sha256": payload_sha256,
        "payload_file_count": len(files),
        "payload_bytes": int(sum(int(item["bytes"]) for item in files)),
        "files": files,
    }
    data = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    (root / MANIFEST_NAME).write_bytes(data)
    manifest_sha256 = hashlib.sha256(data).hexdigest()
    (root / MANIFEST_DIGEST_NAME).write_text(
        f"{manifest_sha256}  {MANIFEST_NAME}\n", encoding="ascii"
    )
    return result


def load_verified_manifest(
    root: str | Path,
    *,
    expected_format: str | None = None,
) -> dict[str, Any]:
    root = Path(root)
    manifest_path = root / MANIFEST_NAME
    digest_path = root / MANIFEST_DIGEST_NAME
    if (
        manifest_path.is_symlink()
        or digest_path.is_symlink()
        or not manifest_path.is_file()
        or not digest_path.is_file()
    ):
        raise FileNotFoundError("Stream requires manifest.json and manifest.sha256")
    if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError("manifest.json exceeds the bounded control-file size")
    if digest_path.stat().st_size > MAX_MANIFEST_DIGEST_BYTES:
        raise ValueError("manifest.sha256 exceeds the bounded control-file size")
    data = manifest_path.read_bytes()
    fields = digest_path.read_text(encoding="ascii").strip().split()
    if len(fields) != 2 or fields[1] != MANIFEST_NAME:
        raise ValueError("Malformed manifest.sha256")
    actual_manifest_sha256 = hashlib.sha256(data).hexdigest()
    if fields[0] != actual_manifest_sha256:
        raise ValueError("Manifest SHA256 mismatch")
    manifest = json.loads(data)
    if expected_format is not None and manifest.get("format") != expected_format:
        raise ValueError(f"Unsupported stream format: {manifest.get('format')!r}")
    integrity = manifest.get("integrity")
    if not isinstance(integrity, dict) or integrity.get("format") != INTEGRITY_FORMAT:
        raise ValueError("Missing or unsupported stream integrity record")
    actual_sha256, actual_files = _digest_payload(root)
    if actual_files != integrity.get("files"):
        raise ValueError("Stream payload file list or size mismatch")
    if actual_sha256 != integrity.get("payload_sha256"):
        raise ValueError("Stream payload SHA256 mismatch")
    if len(actual_files) != int(integrity.get("payload_file_count", -1)):
        raise ValueError("Stream payload file-count mismatch")
    actual_bytes = sum(int(item["bytes"]) for item in actual_files)
    if actual_bytes != int(integrity.get("payload_bytes", -1)):
        raise ValueError("Stream payload byte-count mismatch")
    return manifest
