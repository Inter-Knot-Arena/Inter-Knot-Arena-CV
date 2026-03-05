from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

from manifest_lib import ensure_manifest_defaults, load_manifest, save_manifest


def _record_max_confidence(record: Dict[str, Any]) -> float | None:
    labels = record.get("labels")
    if not isinstance(labels, dict):
        return None
    confidence = labels.get("confidence")
    if isinstance(confidence, (int, float)):
        return float(confidence)
    if isinstance(confidence, dict):
        values = [float(v) for v in confidence.values() if isinstance(v, (int, float))]
        if values:
            return max(values)
    return None


def _parse_set(values: Iterable[str] | None) -> Set[str]:
    if not values:
        return set()
    return {value.strip() for value in values if value and value.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prune CV manifest records by source/state/confidence.")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--drop-source", action="append", default=[])
    parser.add_argument("--drop-state", action="append", default=[])
    parser.add_argument("--max-confidence-below", type=float, default=-1.0)
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = ensure_manifest_defaults(load_manifest(manifest_path))
    records = manifest.get("records", [])
    if not isinstance(records, list):
        raise ValueError("manifest.records must be an array")

    drop_sources = _parse_set(args.drop_source)
    drop_states = _parse_set(args.drop_state)
    conf_threshold = float(args.max_confidence_below)

    kept: List[Dict[str, Any]] = []
    removed = 0
    removed_ids: Set[str] = set()
    remove_reasons: Dict[str, int] = {}

    for record in records:
        if not isinstance(record, dict):
            removed += 1
            remove_reasons["invalid_record"] = remove_reasons.get("invalid_record", 0) + 1
            continue
        rid = str(record.get("id") or "")
        source_id = str(record.get("sourceId") or "")
        state = str(record.get("state") or "")

        reason = ""
        if drop_sources and source_id in drop_sources:
            reason = "drop_source"
        elif drop_states and state in drop_states:
            reason = "drop_state"
        elif conf_threshold >= 0.0:
            max_conf = _record_max_confidence(record)
            if max_conf is not None and max_conf < conf_threshold:
                reason = "drop_low_conf"

        if reason:
            removed += 1
            removed_ids.add(rid)
            remove_reasons[reason] = remove_reasons.get(reason, 0) + 1
            continue
        kept.append(record)

    if not args.dry_run:
        manifest["records"] = kept
        # Remove unreferenced sources.
        used_sources = {str(record.get("sourceId") or "") for record in kept}
        sources = manifest.get("sources", [])
        if isinstance(sources, list):
            manifest["sources"] = [
                source
                for source in sources
                if isinstance(source, dict) and str(source.get("sourceId") or "") in used_sources
            ]
        # Remove dropped ids from split buckets.
        splits = manifest.get("splits")
        if isinstance(splits, dict):
            for bucket in ("train", "val", "test"):
                values = splits.get(bucket)
                if isinstance(values, list):
                    splits[bucket] = [item for item in values if str(item) not in removed_ids]
        save_manifest(manifest_path, manifest)

    print(
        json.dumps(
            {
                "before": len(records),
                "after": len(kept),
                "removed": removed,
                "reasons": remove_reasons,
                "dryRun": bool(args.dry_run),
            },
            ensure_ascii=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

