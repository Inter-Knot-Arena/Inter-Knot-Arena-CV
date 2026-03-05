from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from manifest_lib import ensure_manifest_defaults, load_manifest, save_manifest, utc_now


PROTECTED_KEYS = {"reviewFinal", "reviewerA", "reviewerB"}


def _has_human_review(labels: Dict[str, Any]) -> bool:
    return any(isinstance(labels.get(key), dict) for key in PROTECTED_KEYS)


def _merge_suggestions(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(existing)
    for key, value in incoming.items():
        if key in PROTECTED_KEYS:
            continue
        merged[key] = value
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description="Demote auto-prelabels to suggestedLabels so only reviewed labels count as ground truth.")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--status", default="prelabeled,needs_review", help="Comma-separated qaStatus values to demote.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = ensure_manifest_defaults(load_manifest(manifest_path))
    records = manifest.get("records", [])
    if not isinstance(records, list):
        raise ValueError("manifest.records must be an array")

    target_statuses = {item.strip().lower() for item in str(args.status).split(",") if item.strip()}
    demoted = 0
    skipped = 0

    for record in records:
        if not isinstance(record, dict):
            continue
        labels = record.get("labels")
        if not isinstance(labels, dict) or not labels:
            skipped += 1
            continue
        if _has_human_review(labels):
            skipped += 1
            continue
        status = str(record.get("qaStatus") or "").strip().lower()
        has_prelabel_marker = any(key in labels for key in ("prelabelVersion", "prelabelAt"))
        if status not in target_statuses and not has_prelabel_marker:
            skipped += 1
            continue

        suggested = record.get("suggestedLabels")
        if not isinstance(suggested, dict):
            suggested = {}
        record["suggestedLabels"] = _merge_suggestions(suggested, labels)
        record["labels"] = {}
        record["qaStatus"] = "needs_review" if record["suggestedLabels"] else "unlabeled"
        demoted += 1

    qa_status = manifest.setdefault("qaStatus", {})
    if isinstance(qa_status, dict):
        qa_status["prelabel"] = "completed" if demoted > 0 else qa_status.get("prelabel", "pending")
        qa_status["humanReview"] = "pending"
        qa_status["qaPass2"] = "pending"
        qa_status["qaUpdatedAt"] = utc_now()

    save_manifest(manifest_path, manifest)
    print(json.dumps({"demoted": demoted, "skipped": skipped}, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
