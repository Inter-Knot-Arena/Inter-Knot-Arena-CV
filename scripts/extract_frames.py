from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Set, Tuple

import cv2
import numpy as np

from manifest_lib import ensure_manifest_defaults, load_manifest, save_manifest

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}


def _load_source_ids(source_ids_file: str) -> Set[str]:
    if not source_ids_file:
        return set()
    file_path = Path(source_ids_file).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"source ids file not found: {file_path}")
    source_ids: Set[str] = set()
    for line in file_path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value:
            source_ids.add(value)
    return source_ids


def _matches_source_filter(source_id: str, explicit_ids: Set[str], prefixes: Sequence[str]) -> bool:
    if explicit_ids and source_id in explicit_ids:
        return True
    if prefixes and any(source_id.startswith(prefix) for prefix in prefixes):
        return True
    return not explicit_ids and not prefixes


def _iter_videos(
    raw_dir: Path,
    records: List[Dict[str, Any]],
    explicit_ids: Set[str],
    prefixes: Sequence[str],
) -> Iterator[Tuple[str, Path]]:
    yielded: set[Path] = set()
    for record in records:
        if str(record.get("kind")) != "raw_clip":
            continue
        source_id = str(record.get("sourceId") or "src_unknown")
        if not _matches_source_filter(source_id=source_id, explicit_ids=explicit_ids, prefixes=prefixes):
            continue
        path = Path(str(record.get("path") or "")).expanduser()
        if path.exists() and path.suffix.lower() in VIDEO_EXTENSIONS:
            resolved = path.resolve()
            yielded.add(resolved)
            yield source_id, resolved

    if explicit_ids or prefixes:
        return

    for path in raw_dir.rglob("*"):
        if path.suffix.lower() not in VIDEO_EXTENSIONS or not path.is_file():
            continue
        resolved = path.resolve()
        if resolved in yielded:
            continue
        yield "src_unknown", resolved


def _frame_hist_delta(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    hsv_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2HSV)
    hsv_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2HSV)
    hist_a = cv2.calcHist([hsv_a], [0, 1], None, [32, 32], [0, 180, 0, 256])
    hist_b = cv2.calcHist([hsv_b], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist_a, hist_a, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_b, hist_b, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    similarity = float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))
    return 1.0 - max(-1.0, min(1.0, similarity))


def _should_save(scene_aware: bool, current_frame: np.ndarray, previous_saved: np.ndarray | None, threshold: float) -> bool:
    if not scene_aware or previous_saved is None:
        return True
    delta = _frame_hist_delta(current_frame, previous_saved)
    return delta >= threshold


def _resolve_output_dir(manifest: Dict[str, Any], state: str, override: str) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    layout = manifest.get("directoryLayout", {})
    if not isinstance(layout, dict):
        layout = {}
    if state == "precheck":
        key = "frames_precheck"
    elif state == "inrun":
        key = "frames_inrun"
    else:
        key = "frames_precheck"
    folder = str(layout.get(key, ""))
    if not folder:
        raise ValueError("output directory is not configured in manifest; run bootstrap_dataset.py or pass --output-dir")
    return Path(folder).expanduser().resolve()


def _append_record(
    manifest: Dict[str, Any],
    record_id: str,
    source_id: str,
    output_path: Path,
    state: str,
    locale: str,
    resolution: str,
    frame_ts_ms: float,
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
            "kind": "frame_crop",
            "state": state,
            "locale": locale,
            "resolution": resolution,
            "path": str(output_path),
            "frameTsMs": round(frame_ts_ms, 2),
            "labelsPath": "",
            "qaStatus": "unlabeled",
        }
    )


def _source_index(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    sources = manifest.get("sources", [])
    if not isinstance(sources, list):
        return {}
    index: Dict[str, Dict[str, Any]] = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        source_id = str(source.get("sourceId") or "").strip()
        if source_id:
            index[source_id] = source
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract training frames from raw clips and append records to CV manifest.")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--raw-dir", default="", help="Override raw media directory.")
    parser.add_argument("--output-dir", default="", help="Override extracted frames directory.")
    parser.add_argument("--state", default="precheck", choices=["precheck", "inrun", "other"])
    parser.add_argument("--locale", default="auto", help="Record locale or 'auto' to inherit from source metadata.")
    parser.add_argument("--resolution", default="auto", help="Record resolution or 'auto' to inherit from source metadata.")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-frames-per-clip", type=int, default=400)
    parser.add_argument("--scene-aware", action="store_true", default=False)
    parser.add_argument("--scene-threshold", type=float, default=0.16)
    parser.add_argument("--session-id", default="")
    parser.add_argument(
        "--source-id-prefix",
        action="append",
        default=[],
        help="Only process sourceIds starting with this prefix. Repeatable.",
    )
    parser.add_argument(
        "--source-ids-file",
        default="",
        help="Optional text file with one sourceId per line to process.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = ensure_manifest_defaults(load_manifest(manifest_path))
    layout = manifest.get("directoryLayout", {})
    if not isinstance(layout, dict):
        raise ValueError("manifest.directoryLayout must be an object")

    raw_folder = args.raw_dir or str(layout.get("raw", ""))
    if not raw_folder:
        raise ValueError("raw directory is not configured; run bootstrap_dataset.py or pass --raw-dir")
    raw_dir = Path(raw_folder).expanduser().resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)

    output_dir = _resolve_output_dir(manifest=manifest, state=args.state, override=args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = manifest.get("records", [])
    if not isinstance(records, list):
        raise ValueError("manifest.records must be an array")
    source_index = _source_index(manifest)
    source_id_prefixes = [str(prefix).strip() for prefix in args.source_id_prefix if str(prefix).strip()]
    explicit_source_ids = _load_source_ids(args.source_ids_file)

    frame_total = 0
    processed_clips = 0
    for source_id, video_path in _iter_videos(
        raw_dir=raw_dir,
        records=records,
        explicit_ids=explicit_source_ids,
        prefixes=source_id_prefixes,
    ):
        source_meta = source_index.get(source_id, {})
        locale = str(args.locale).strip()
        resolution = str(args.resolution).strip()
        if not locale or locale.lower() == "auto":
            locale = str(source_meta.get("locale") or "unknown")
        if not resolution or resolution.lower() == "auto":
            resolution = str(source_meta.get("resolution") or "unknown")

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            continue
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        source_step = max(1, int(round((fps if fps > 0 else 30.0) / max(0.05, args.fps))))
        frame_index = 0
        saved_count = 0
        previous_saved: np.ndarray | None = None
        processed_clips += 1

        while capture.isOpened():
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if frame_index % source_step != 0:
                frame_index += 1
                continue

            if not _should_save(
                scene_aware=args.scene_aware,
                current_frame=frame,
                previous_saved=previous_saved,
                threshold=max(0.01, args.scene_threshold),
            ):
                frame_index += 1
                continue

            frame_ts_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            file_name = f"{source_id}_{video_path.stem}_{args.state}_f{frame_index:06d}.jpg"
            output_path = output_dir / file_name
            if cv2.imwrite(str(output_path), frame):
                record_id = f"frame-{source_id}-{video_path.stem}-{args.state}-{frame_index:06d}"
                _append_record(
                    manifest=manifest,
                    record_id=record_id,
                    source_id=source_id,
                    output_path=output_path.resolve(),
                    state=args.state,
                    locale=locale,
                    resolution=resolution,
                    frame_ts_ms=frame_ts_ms,
                    session_id=args.session_id,
                )
                frame_total += 1
                saved_count += 1
                previous_saved = frame

            frame_index += 1
            if saved_count >= max(1, args.max_frames_per_clip):
                break
        capture.release()

    save_manifest(manifest_path, manifest)
    print(
        json.dumps(
            {
                "clipsProcessed": processed_clips,
                "framesExtracted": frame_total,
                "outputDir": str(output_dir),
                "sourceIdPrefixes": source_id_prefixes,
                "sourceIdsFile": args.source_ids_file,
            },
            ensure_ascii=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
