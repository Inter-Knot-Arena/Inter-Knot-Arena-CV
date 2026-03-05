from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_ratio(raw: str) -> tuple[float, float, float]:
    parts = [float(value.strip()) for value in raw.split(",")]
    if len(parts) != 3:
        raise ValueError("ratio must contain exactly three comma-separated numbers.")
    total = sum(parts)
    if total <= 0:
        raise ValueError("ratio sum must be positive.")
    normalized = tuple(value / total for value in parts)
    return normalized  # type: ignore[return-value]


def _group_key(record: Dict[str, Any]) -> str:
    source_id = str(record.get("sourceId") or "src_unknown")
    session_id = str(record.get("sessionId") or "")
    match_id = str(record.get("matchId") or "")
    if session_id:
        return f"{source_id}::session::{session_id}"
    if match_id:
        return f"{source_id}::match::{match_id}"
    record_id = str(record.get("id") or "")
    return f"{source_id}::record::{record_id}"


def assign_splits(records: List[Dict[str, Any]], seed: int, ratio: tuple[float, float, float]) -> Dict[str, List[str]]:
    rng = random.Random(seed)
    grouped: dict[str, list[str]] = defaultdict(list)
    for idx, record in enumerate(records):
        record_id = str(record.get("id", f"record_{idx}"))
        grouped[_group_key(record)].append(record_id)

    group_ids = list(grouped.keys())
    rng.shuffle(group_ids)
    train_cut = int(len(group_ids) * ratio[0])
    val_cut = train_cut + int(len(group_ids) * ratio[1])
    train_groups = set(group_ids[:train_cut])
    val_groups = set(group_ids[train_cut:val_cut])

    split_ids = {"train": [], "val": [], "test": []}
    for group_id, record_ids in grouped.items():
        if group_id in train_groups:
            split_ids["train"].extend(record_ids)
        elif group_id in val_groups:
            split_ids["val"].extend(record_ids)
        else:
            split_ids["test"].extend(record_ids)
    return {
        "train": split_ids["train"],
        "val": split_ids["val"],
        "test": split_ids["test"],
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
    payload["updatedAt"] = _utc_now()

    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=True, indent=2)
        fh.write("\n")

    print(f"Updated splits for {len(records)} records in {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
