from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


def parse_ratio(raw: str) -> tuple[float, float, float]:
    parts = [float(value.strip()) for value in raw.split(",")]
    if len(parts) != 3:
        raise ValueError("ratio must contain exactly three comma-separated numbers.")
    total = sum(parts)
    if total <= 0:
        raise ValueError("ratio sum must be positive.")
    return tuple(value / total for value in parts)  # type: ignore[return-value]


def assign_splits(records: List[Dict[str, Any]], seed: int, ratio: tuple[float, float, float]) -> Dict[str, List[str]]:
    rng = random.Random(seed)
    ids = [str(record.get("id", f"record_{idx}")) for idx, record in enumerate(records)]
    rng.shuffle(ids)

    train_cut = int(len(ids) * ratio[0])
    val_cut = train_cut + int(len(ids) * ratio[1])
    return {
        "train": ids[:train_cut],
        "val": ids[train_cut:val_cut],
        "test": ids[val_cut:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate train/val/test split in CV dataset manifest.")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratio", default="0.8,0.1,0.1")
    args = parser.parse_args()

    ratio = parse_ratio(args.ratio)
    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError("manifest.records must be an array.")

    payload["splits"] = assign_splits(records, args.seed, ratio)
    payload["splitSeed"] = args.seed
    payload["splitRatio"] = {
        "train": ratio[0],
        "val": ratio[1],
        "test": ratio[2],
    }

    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=True, indent=2)
        fh.write("\n")

    print(f"Updated splits for {len(records)} records in {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
