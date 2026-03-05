from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
from PIL import ImageGrab

from manifest_lib import ensure_manifest_defaults, load_manifest, save_manifest, source_exists, utc_now

try:
    import dxcam  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    dxcam = None


def _parse_region(raw: str) -> Tuple[int, int, int, int] | None:
    if not raw.strip():
        return None
    parts = [item.strip() for item in raw.split(",")]
    if len(parts) != 4:
        raise ValueError("region must be x,y,width,height")
    x, y, w, h = [int(value) for value in parts]
    if w <= 0 or h <= 0:
        raise ValueError("region width/height must be positive")
    return x, y, w, h


def _capture_pil(region: Tuple[int, int, int, int] | None) -> np.ndarray | None:
    try:
        bbox = None
        if region:
            x, y, w, h = region
            bbox = (x, y, x + w, y + h)
        image = ImageGrab.grab(bbox=bbox, all_screens=True)
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def _build_dxcam_camera():
    if dxcam is None:
        return None
    output_idx_raw = os.getenv("IKA_CAPTURE_OUTPUT_IDX", "0")
    try:
        output_idx = max(0, int(output_idx_raw))
    except Exception:
        output_idx = 0
    try:
        return dxcam.create(output_idx=output_idx, output_color="BGR")
    except Exception:
        return None


def _capture_dxcam(camera, region: Tuple[int, int, int, int] | None) -> np.ndarray | None:
    if camera is None:
        return None
    capture_region = None
    if region:
        x, y, w, h = region
        capture_region = (x, y, x + w, y + h)
    try:
        frame = camera.grab(region=capture_region)
    except Exception:
        return None
    if frame is None:
        return None
    if frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def _append_record(
    manifest: Dict[str, Any],
    record_id: str,
    source_id: str,
    output_path: Path,
    state: str,
    locale: str,
    resolution: str,
    session_id: str,
) -> None:
    records = manifest.setdefault("records", [])
    for record in records:
        if str(record.get("id")) == record_id:
            return
    records.append(
        {
            "id": record_id,
            "sourceId": source_id,
            "sessionId": session_id,
            "matchId": "",
            "kind": "frame_capture",
            "state": state,
            "locale": locale,
            "resolution": resolution,
            "path": str(output_path.resolve()),
            "labelsPath": "",
            "qaStatus": "unlabeled",
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture live session frames and append them to CV manifest.")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--mode", default="cv", choices=["cv", "ocr"])
    parser.add_argument("--duration-sec", type=int, default=120)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--region", default="", help="x,y,width,height")
    parser.add_argument("--state", default="inrun", choices=["precheck", "inrun", "other"])
    parser.add_argument("--locale", default="unknown")
    parser.add_argument("--resolution", default="unknown")
    parser.add_argument("--collector", default=os.getenv("USERNAME", "unknown"))
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = ensure_manifest_defaults(load_manifest(manifest_path))
    layout = manifest.get("directoryLayout", {})
    if not isinstance(layout, dict):
        raise ValueError("manifest.directoryLayout must be an object")
    raw_folder = str(layout.get("raw", ""))
    if not raw_folder:
        raise ValueError("raw directory is not configured; run bootstrap_dataset.py first")
    raw_dir = Path(raw_folder).expanduser().resolve()
    session_id = f"session_{int(time.time())}"
    target_dir = raw_dir / "live" / session_id
    target_dir.mkdir(parents=True, exist_ok=True)

    source_id = f"live_{session_id}"
    if not source_exists(manifest.get("sources", []), source_id):
        manifest.setdefault("sources", []).append(
            {
                "sourceId": source_id,
                "url": "",
                "captureDate": utc_now(),
                "licenseNote": "self-captured gameplay",
                "locale": args.locale,
                "resolution": args.resolution,
                "gamePatch": "unknown",
                "collector": args.collector,
                "sourceType": "live-capture",
            }
        )

    interval_sec = 1.0 / max(0.1, args.fps)
    region = _parse_region(args.region)
    camera = _build_dxcam_camera()
    captured = 0
    started = time.perf_counter()
    next_tick = started
    while True:
        now = time.perf_counter()
        if now - started >= max(1, args.duration_sec):
            break
        if now < next_tick:
            time.sleep(min(0.02, next_tick - now))
            continue

        frame = _capture_dxcam(camera, region=region)
        if frame is None:
            frame = _capture_pil(region=region)
        if frame is not None:
            file_name = f"{session_id}_{captured:06d}.jpg"
            output_path = target_dir / file_name
            if cv2.imwrite(str(output_path), frame):
                record_id = f"live-{session_id}-{captured:06d}"
                _append_record(
                    manifest=manifest,
                    record_id=record_id,
                    source_id=source_id,
                    output_path=output_path,
                    state=args.state,
                    locale=args.locale,
                    resolution=args.resolution,
                    session_id=session_id,
                )
                captured += 1

        next_tick += interval_sec

    save_manifest(manifest_path, manifest)
    print(
        json.dumps(
            {
                "sessionId": session_id,
                "capturedFrames": captured,
                "outputDir": str(target_dir),
                "captureBackend": "dxcam" if camera is not None else "PIL",
            },
            ensure_ascii=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

