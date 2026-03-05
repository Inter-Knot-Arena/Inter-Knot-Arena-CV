from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from manifest_lib import ensure_manifest_defaults, load_manifest, save_manifest, utc_now


def _label_payload(record: Dict[str, Any], suggested: bool) -> Dict[str, Any]:
    key = "suggestedLabels" if suggested else "labels"
    labels = record.get(key)
    if not isinstance(labels, dict):
        return {}
    return labels


def _record_label(record: Dict[str, Any], slot: str, *, suggested: bool) -> str:
    labels = _label_payload(record, suggested=suggested)
    if not labels:
        return ""
    value = labels.get(slot)
    return str(value).strip() if isinstance(value, str) else ""


def _compute_slot_agreement(record: Dict[str, Any]) -> tuple[int, int]:
    labels = record.get("labels")
    if not isinstance(labels, dict):
        return 0, 0
    reviewer_a = labels.get("reviewerA")
    reviewer_b = labels.get("reviewerB")
    if not isinstance(reviewer_a, dict) or not isinstance(reviewer_b, dict):
        return 0, 0
    matches = 0
    total = 0
    for key in ("slot_1_agent", "slot_2_agent", "slot_3_agent"):
        a = str(reviewer_a.get(key) or "").strip()
        b = str(reviewer_b.get(key) or "").strip()
        if not a or not b:
            continue
        total += 1
        if a == b:
            matches += 1
    return matches, total


def _sample_for_double_review(records: List[Dict[str, Any]], ratio: float, seed: int) -> List[str]:
    candidates = [str(record.get("id") or "") for record in records if isinstance(record, dict)]
    candidates = [item for item in candidates if item]
    if not candidates:
        return []
    rng = random.Random(seed)
    rng.shuffle(candidates)
    take = max(1, int(len(candidates) * max(0.0, min(1.0, ratio))))
    return candidates[:take]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate QA audit report for CV manifest labels.")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--output-file", default="docs/qa_report.json")
    parser.add_argument("--double-review-file", default="docs/double_review_samples.json")
    parser.add_argument("--double-review-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = ensure_manifest_defaults(load_manifest(manifest_path))
    records = manifest.get("records", [])
    if not isinstance(records, list):
        raise ValueError("manifest.records must be an array")

    reviewed = 0
    suggested = 0
    needs_review = 0
    reviewed_unknown = 0
    suggested_unknown = 0
    agreement_match = 0
    agreement_total = 0
    state_counts: Dict[str, int] = {}
    reviewed_label_counts: Dict[str, int] = {}
    suggested_label_counts: Dict[str, int] = {}

    valid_records: List[Dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        valid_records.append(record)
        state = str(record.get("state") or "other")
        state_counts[state] = state_counts.get(state, 0) + 1

        if str(record.get("qaStatus") or "").lower() == "needs_review":
            needs_review += 1

        reviewed_labels = _label_payload(record, suggested=False)
        if reviewed_labels:
            reviewed += 1
            if bool(reviewed_labels.get("unknown_flag", False)):
                reviewed_unknown += 1
            for slot in ("slot_1_agent", "slot_2_agent", "slot_3_agent"):
                value = _record_label(record, slot, suggested=False)
                if value:
                    reviewed_label_counts[value] = reviewed_label_counts.get(value, 0) + 1

        suggested_labels = _label_payload(record, suggested=True)
        if suggested_labels:
            suggested += 1
            if bool(suggested_labels.get("unknown_flag", False)):
                suggested_unknown += 1
            for slot in ("slot_1_agent", "slot_2_agent", "slot_3_agent"):
                value = _record_label(record, slot, suggested=True)
                if value:
                    suggested_label_counts[value] = suggested_label_counts.get(value, 0) + 1
        matches, total = _compute_slot_agreement(record)
        agreement_match += matches
        agreement_total += total

    double_review_ids = _sample_for_double_review(
        records=valid_records,
        ratio=max(0.0, min(1.0, args.double_review_ratio)),
        seed=args.seed,
    )
    agreement = round(agreement_match / agreement_total, 4) if agreement_total > 0 else None

    report = {
        "recordCount": len(valid_records),
        "labeledCount": reviewed,
        "reviewedCount": reviewed,
        "suggestedCount": suggested,
        "needsReviewCount": needs_review,
        "unknownFlagCount": reviewed_unknown,
        "reviewedUnknownFlagCount": reviewed_unknown,
        "suggestedUnknownFlagCount": suggested_unknown,
        "stateCounts": state_counts,
        "labelCounts": reviewed_label_counts,
        "reviewedLabelCounts": reviewed_label_counts,
        "suggestedLabelCounts": suggested_label_counts,
        "interAnnotatorAgreement": agreement,
        "agreementComparedSlots": agreement_total,
        "generatedAt": utc_now(),
    }

    output_path = Path(args.output_file).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=True, indent=2)
        fh.write("\n")

    review_path = Path(args.double_review_file).resolve()
    review_path.parent.mkdir(parents=True, exist_ok=True)
    with review_path.open("w", encoding="utf-8") as fh:
        json.dump({"recordIds": double_review_ids, "generatedAt": utc_now()}, fh, ensure_ascii=True, indent=2)
        fh.write("\n")

    qa_status = manifest.setdefault("qaStatus", {})
    if isinstance(qa_status, dict):
        qa_status["doubleReviewRate"] = max(0.0, min(1.0, args.double_review_ratio))
        qa_status["interAnnotatorAgreement"] = agreement
        qa_status["humanReview"] = "in_progress" if needs_review > 0 else "completed"
        qa_status["qaPass2"] = "pending"
        qa_status["qaUpdatedAt"] = utc_now()

    save_manifest(manifest_path, manifest)
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
