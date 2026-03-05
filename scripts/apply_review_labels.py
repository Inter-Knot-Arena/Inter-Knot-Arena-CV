from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict

from manifest_lib import ensure_manifest_defaults, load_manifest, save_manifest, utc_now

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roster_taxonomy import canonicalize_agent_label


def _to_bool(raw: str) -> bool | None:
    value = raw.strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    return None


def _normalize_slot(raw: str, fallback: str) -> str:
    candidate = str(raw or "").strip() or str(fallback or "").strip()
    if not candidate:
        return ""
    canonical = canonicalize_agent_label(candidate)
    if canonical:
        return canonical
    if candidate == "unknown":
        return "unknown"
    raise ValueError(f"Invalid CV agent label: {candidate}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply reviewed CV labels from CSV to dataset manifest.")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--review-round", choices=["A", "B", "final"], default="final")
    parser.add_argument("--reviewer-id", default="human")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = ensure_manifest_defaults(load_manifest(manifest_path))
    records = manifest.get("records", [])
    if not isinstance(records, list):
        raise ValueError("manifest.records must be an array")
    record_map = {str(record.get("id") or ""): record for record in records if isinstance(record, dict)}

    applied = 0
    missing = 0
    invalid = 0
    with Path(args.input_csv).resolve().open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            record_id = str(row.get("record_id") or "").strip()
            if not record_id:
                continue
            record = record_map.get(record_id)
            if record is None:
                missing += 1
                continue

            labels = record.get("labels")
            if not isinstance(labels, dict):
                labels = {}
                record["labels"] = labels
            suggested = record.get("suggestedLabels")
            if not isinstance(suggested, dict):
                suggested = {}

            review_payload: Dict[str, Any] = {
                "slot_1_agent": str(row.get("slot_1_agent") or "").strip(),
                "slot_2_agent": str(row.get("slot_2_agent") or "").strip(),
                "slot_3_agent": str(row.get("slot_3_agent") or "").strip(),
                "reviewer": str(row.get("reviewer") or args.reviewer_id).strip() or args.reviewer_id,
                "reviewedAt": utc_now(),
                "notes": str(row.get("notes") or "").strip(),
            }
            unknown_flag = _to_bool(str(row.get("unknown_flag") or ""))
            if unknown_flag is not None:
                review_payload["unknown_flag"] = unknown_flag

            if args.review_round == "A":
                labels["reviewerA"] = review_payload
                record["qaStatus"] = "needs_review"
            elif args.review_round == "B":
                labels["reviewerB"] = review_payload
                record["qaStatus"] = "needs_review"
            else:
                try:
                    resolved_slots = {
                        key: _normalize_slot(
                            str(review_payload.get(key) or ""),
                            str(labels.get(key) or suggested.get(key) or ""),
                        )
                        for key in ("slot_1_agent", "slot_2_agent", "slot_3_agent")
                    }
                except ValueError:
                    invalid += 1
                    record["qaStatus"] = "needs_review"
                    continue
                for key, value in resolved_slots.items():
                    labels[key] = value or "unknown"

                explicit_unknown = review_payload.get("unknown_flag")
                if "unknown_flag" in review_payload:
                    labels["unknown_flag"] = bool(explicit_unknown)
                else:
                    labels["unknown_flag"] = any(value == "unknown" for value in resolved_slots.values())
                labels["reviewFinal"] = {
                    "reviewer": review_payload["reviewer"],
                    "reviewedAt": review_payload["reviewedAt"],
                    "notes": review_payload["notes"],
                }
                record["qaStatus"] = "reviewed"
            applied += 1

    qa_status = manifest.setdefault("qaStatus", {})
    if isinstance(qa_status, dict):
        qa_status["humanReview"] = "in_progress" if args.review_round in {"A", "B"} else "completed"
        qa_status["qaPass2"] = "completed" if args.review_round == "final" else "pending"
        qa_status["qaUpdatedAt"] = utc_now()

    save_manifest(manifest_path, manifest)
    print(f"Applied labels: {applied}; missing records: {missing}; invalid rows: {invalid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
