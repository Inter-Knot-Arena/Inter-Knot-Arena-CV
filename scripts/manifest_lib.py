from __future__ import annotations

import hashlib
import json
import os
import tempfile
from json import JSONDecodeError
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_manifest(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except JSONDecodeError as exc:
        decoder = json.JSONDecoder()
        payload, end = decoder.raw_decode(text)
        if not isinstance(payload, dict):
            raise ValueError(f"Manifest must be an object: {path}") from exc
        # Recover from trailing corruption by keeping first valid JSON object.
        save_manifest(path, payload)
        return payload
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must be an object: {path}")
    return payload


def save_manifest(path: Path, payload: Dict[str, Any]) -> None:
    payload["updatedAt"] = utc_now()
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, ensure_ascii=True, indent=2) + "\n"
    fd, tmp_raw = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_raw)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def ensure_manifest_defaults(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload.setdefault("schemaVersion", "dataset-manifest-v1")
    payload.setdefault("splitSeed", 42)
    payload.setdefault("splitRatio", {"train": 0.8, "val": 0.1, "test": 0.1})
    payload.setdefault("splits", {"train": [], "val": [], "test": []})
    payload.setdefault(
        "qaStatus",
        {
            "prelabel": "pending",
            "humanReview": "pending",
            "qaPass2": "pending",
            "doubleReviewRate": 0.1,
            "interAnnotatorAgreement": None,
        },
    )
    payload.setdefault("hardSamplesRef", "")
    payload.setdefault("sources", [])
    payload.setdefault("records", [])
    payload.setdefault("privateStorageRoot", "")
    payload.setdefault("directoryLayout", {})
    return payload


def source_exists(sources: Iterable[Dict[str, Any]], source_id: str) -> bool:
    for source in sources:
        if str(source.get("sourceId")) == source_id:
            return True
    return False


def hash_file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
