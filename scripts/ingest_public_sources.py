from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from manifest_lib import ensure_manifest_defaults, hash_file_sha256, load_manifest, save_manifest, source_exists, utc_now

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}


def _read_sources_file(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("sources file must contain an array of source objects")
    output: List[Dict[str, Any]] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        output.append(entry)
    return output


def _run_command(command: list[str]) -> tuple[int, str]:
    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return process.returncode, process.stdout


def _resolve_ytdlp_command() -> list[str]:
    ytdlp = shutil.which("yt-dlp")
    if ytdlp:
        return [ytdlp]
    if importlib.util.find_spec("yt_dlp") is not None:
        return [sys.executable, "-m", "yt_dlp"]
    raise RuntimeError("yt-dlp not found. Install via PATH or 'pip install yt-dlp'.")


def _resolve_ffmpeg_command() -> str | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _copy_local_media(url_or_path: str, target_dir: Path, source_id: str) -> List[Path]:
    source_path = Path(url_or_path).expanduser()
    if not source_path.exists():
        return []
    target_dir.mkdir(parents=True, exist_ok=True)
    copied: List[Path] = []
    if source_path.is_file():
        destination = target_dir / f"{source_id}{source_path.suffix.lower()}"
        shutil.copy2(source_path, destination)
        copied.append(destination)
        return copied

    for path in source_path.rglob("*"):
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        relative = path.relative_to(source_path)
        destination = target_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        copied.append(destination)
    return copied


def _download_with_ytdlp(url: str, target_dir: Path, source_id: str, max_items: int) -> List[Path]:
    ytdlp_cmd = _resolve_ytdlp_command()
    target_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(target_dir / f"{source_id}_%(autonumber)03d.%(ext)s")
    existing = {path.resolve() for path in target_dir.rglob("*") if path.is_file()}
    command = ytdlp_cmd + [
        "--no-progress",
        "--no-warnings",
        "--ignore-errors",
        "--no-playlist",
        "--output",
        pattern,
        url,
    ]
    if max_items > 0:
        command.extend(["--max-downloads", str(max_items)])
    code, output = _run_command(command)
    if code != 0:
        raise RuntimeError(f"yt-dlp failed ({code}): {output.strip()[:3000]}")
    downloaded = [
        path
        for path in target_dir.rglob("*")
        if path.is_file() and path.resolve() not in existing and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return downloaded


def _normalize_with_ffmpeg(input_path: Path, overwrite: bool) -> Path:
    ffmpeg = _resolve_ffmpeg_command()
    if not ffmpeg:
        return input_path
    normalized_path = input_path.with_name(f"{input_path.stem}_norm.mp4")
    command = [
        ffmpeg,
        "-y" if overwrite else "-n",
        "-i",
        str(input_path),
        "-vf",
        "fps=60,scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(normalized_path),
    ]
    code, output = _run_command(command)
    if code != 0:
        return input_path
    return normalized_path


def _append_record(manifest: Dict[str, Any], source_id: str, path: Path, resolution: str, locale: str) -> None:
    records = manifest.setdefault("records", [])
    record_id = f"raw-{source_id}-{path.stem}"
    for record in records:
        if str(record.get("id")) == record_id:
            return
    records.append(
        {
            "id": record_id,
            "sourceId": source_id,
            "sessionId": "",
            "matchId": "",
            "kind": "raw_clip",
            "state": "other",
            "locale": locale,
            "resolution": resolution,
            "path": str(path),
            "sha256": hash_file_sha256(path),
            "captureDate": utc_now(),
            "labelsPath": "",
            "qaStatus": "unlabeled",
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest public media sources and persist provenance in CV manifest.")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--sources-file", required=True, help="JSON array with source descriptors.")
    parser.add_argument("--raw-dir", default="", help="Override raw storage directory.")
    parser.add_argument("--max-downloads", type=int, default=1)
    parser.add_argument("--skip-download", action="store_true", default=False)
    parser.add_argument("--normalize", action="store_true", default=False)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = ensure_manifest_defaults(load_manifest(manifest_path))
    raw_dir = (
        Path(args.raw_dir).resolve()
        if args.raw_dir
        else Path(str(manifest.get("directoryLayout", {}).get("raw", ""))).expanduser().resolve()
    )
    if not str(raw_dir):
        raise ValueError("raw directory is not set; run bootstrap_dataset.py first or pass --raw-dir")
    raw_dir.mkdir(parents=True, exist_ok=True)

    sources = _read_sources_file(Path(args.sources_file).resolve())
    summary: Dict[str, Any] = {"sources": 0, "clips": 0, "errors": []}
    for index, source in enumerate(sources):
        source_id = str(source.get("sourceId") or f"source_{index + 1}").strip()
        url = str(source.get("url") or source.get("path") or "").strip()
        if not url:
            summary["errors"].append({"sourceId": source_id, "error": "missing url/path"})
            continue

        source_payload = {
            "sourceId": source_id,
            "url": url,
            "captureDate": str(source.get("captureDate") or utc_now()),
            "licenseNote": str(source.get("licenseNote") or "unspecified"),
            "locale": str(source.get("locale") or "unknown"),
            "resolution": str(source.get("resolution") or "unknown"),
            "gamePatch": str(source.get("gamePatch") or "unknown"),
            "collector": str(source.get("collector") or "unknown"),
            "sourceType": str(source.get("sourceType") or "public"),
        }

        if not source_exists(manifest.get("sources", []), source_id):
            manifest.setdefault("sources", []).append(source_payload)
        summary["sources"] += 1

        if args.skip_download:
            continue

        target_dir = raw_dir / "public" / source_id
        downloaded_files: List[Path] = []
        try:
            if url.startswith(("http://", "https://")):
                downloaded_files = _download_with_ytdlp(url, target_dir, source_id, args.max_downloads)
            else:
                downloaded_files = _copy_local_media(url, target_dir, source_id)
        except Exception as exc:
            summary["errors"].append({"sourceId": source_id, "error": str(exc)})
            continue

        if args.normalize:
            normalized: List[Path] = []
            for file_path in downloaded_files:
                normalized.append(_normalize_with_ffmpeg(file_path, overwrite=True))
            downloaded_files = normalized

        for file_path in downloaded_files:
            _append_record(
                manifest,
                source_id=source_id,
                path=file_path.resolve(),
                resolution=str(source_payload["resolution"]),
                locale=str(source_payload["locale"]),
            )
        summary["clips"] += len(downloaded_files)

    save_manifest(manifest_path, manifest)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
