from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from manifest_lib import ensure_manifest_defaults, load_manifest, save_manifest

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _phash(path: Path) -> int | None:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        return None
    image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(image.astype(np.float32))
    dct_low = dct[:8, :8]
    median = float(np.median(dct_low[1:, 1:]))
    bits = dct_low > median
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bool(bit))
    return value


def _hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _ssim_score(path_a: Path, path_b: Path) -> float:
    image_a = cv2.imread(str(path_a), cv2.IMREAD_GRAYSCALE)
    image_b = cv2.imread(str(path_b), cv2.IMREAD_GRAYSCALE)
    if image_a is None or image_b is None:
        return 0.0
    if image_a.shape != image_b.shape:
        image_b = cv2.resize(image_b, (image_a.shape[1], image_a.shape[0]), interpolation=cv2.INTER_AREA)

    image_a = image_a.astype(np.float64)
    image_b = image_b.astype(np.float64)
    c1 = 6.5025
    c2 = 58.5225

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = kernel @ kernel.T

    mu1 = cv2.filter2D(image_a, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(image_b, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(image_a**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(image_b**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(image_a * image_b, -1, window)[5:-5, 5:-5] - mu1_mu2

    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = numerator / np.maximum(denominator, 1e-9)
    return float(np.mean(ssim_map))


def _iter_images(input_dir: Path) -> List[Path]:
    return [path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]


def main() -> int:
    parser = argparse.ArgumentParser(description="Remove duplicated frames using pHash + SSIM.")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--input-dir", default="", help="Override frame directory.")
    parser.add_argument("--phash-threshold", type=int, default=6)
    parser.add_argument("--ssim-threshold", type=float, default=0.97)
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = ensure_manifest_defaults(load_manifest(manifest_path))
    layout = manifest.get("directoryLayout", {})
    if not isinstance(layout, dict):
        raise ValueError("manifest.directoryLayout must be an object")

    input_folder = args.input_dir or str(layout.get("frames_precheck", ""))
    if not input_folder:
        raise ValueError("input directory is not configured; pass --input-dir or run bootstrap_dataset.py")
    input_dir = Path(input_folder).expanduser().resolve()
    input_dir.mkdir(parents=True, exist_ok=True)

    images = _iter_images(input_dir)
    accepted: List[Dict[str, Any]] = []
    buckets: dict[int, List[int]] = {}
    duplicates: List[Path] = []
    duplicate_map: dict[str, str] = {}

    for path in sorted(images):
        image_hash = _phash(path)
        if image_hash is None:
            continue
        bucket_key = image_hash >> 48
        candidate_indices = buckets.get(bucket_key, [])
        is_duplicate = False
        duplicate_of = ""
        for idx in candidate_indices:
            candidate = accepted[idx]
            distance = _hamming_distance(image_hash, int(candidate["hash"]))
            if distance > max(0, args.phash_threshold):
                continue
            score = _ssim_score(path, Path(str(candidate["path"])))
            if score >= min(1.0, max(0.0, args.ssim_threshold)):
                is_duplicate = True
                duplicate_of = str(candidate["path"])
                break

        if is_duplicate:
            duplicates.append(path)
            duplicate_map[str(path.resolve())] = duplicate_of
            continue

        accepted.append({"path": str(path.resolve()), "hash": image_hash})
        buckets.setdefault(bucket_key, []).append(len(accepted) - 1)

    if not args.dry_run:
        for path in duplicates:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass

    records = manifest.get("records", [])
    if not isinstance(records, list):
        records = []
    before_count = len(records)
    duplicate_paths = set(duplicate_map.keys())
    filtered_records = [
        record for record in records if str(Path(str(record.get("path") or "")).resolve()) not in duplicate_paths
    ]
    manifest["records"] = filtered_records
    save_manifest(manifest_path, manifest)

    print(
        json.dumps(
            {
                "inputCount": len(images),
                "duplicateCount": len(duplicates),
                "recordsBefore": before_count,
                "recordsAfter": len(filtered_records),
                "dryRun": bool(args.dry_run),
            },
            ensure_ascii=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

