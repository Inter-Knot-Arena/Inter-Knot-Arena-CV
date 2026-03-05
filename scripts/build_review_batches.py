from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from manifest_lib import ensure_manifest_defaults, load_manifest, utc_now

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roster_taxonomy import current_agent_ids, source_focus_agent_ids


FIELDS = [
    "batch_id",
    "priority_score",
    "priority_reasons",
    "record_id",
    "qa_status",
    "source_id",
    "focus_agent_id",
    "state",
    "locale",
    "resolution",
    "path",
    "reviewed_slot_1_agent",
    "reviewed_slot_2_agent",
    "reviewed_slot_3_agent",
    "suggested_slot_1_agent",
    "suggested_slot_2_agent",
    "suggested_slot_3_agent",
    "reviewed_unknown_flag",
    "suggested_unknown_flag",
    "suggested_confidence",
]


def _labels(record: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = record.get(key)
    if not isinstance(value, dict):
        return {}
    return value


def _str_value(payload: Dict[str, Any], key: str) -> str:
    value = payload.get(key)
    return str(value).strip() if isinstance(value, str) else ""


def _bool_value(payload: Dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value.strip().lower()
    return ""


def _confidence_text(payload: Dict[str, Any]) -> str:
    confidence = payload.get("confidence")
    if isinstance(confidence, dict):
        return "; ".join(f"{key}:{value}" for key, value in confidence.items())
    return ""


def _suggested_agent_ids(record: Dict[str, Any]) -> List[str]:
    suggested = _labels(record, "suggestedLabels")
    output: List[str] = []
    for key in ("slot_1_agent", "slot_2_agent", "slot_3_agent"):
        value = _str_value(suggested, key)
        if value and value not in output and value != "unknown":
            output.append(value)
    return output


def _reviewed_agent_counts(records: List[Dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        if not isinstance(record, dict):
            continue
        labels = _labels(record, "labels")
        for key in ("slot_1_agent", "slot_2_agent", "slot_3_agent"):
            value = _str_value(labels, key)
            if value and value != "unknown":
                counts[value] += 1
    return counts


def _focus_agent(source_index: Dict[str, Dict[str, Any]], source_id: str) -> str:
    source = source_index.get(source_id, {})
    focus_ids = source_focus_agent_ids(source)
    return focus_ids[0] if focus_ids else ""


def _score_record(record: Dict[str, Any], source_index: Dict[str, Dict[str, Any]], missing_reviewed: set[str]) -> tuple[int, List[str], str]:
    score = 0
    reasons: List[str] = []
    source_id = str(record.get("sourceId") or "src_unknown")
    focus_agent_id = _focus_agent(source_index, source_id)
    suggested = _labels(record, "suggestedLabels")
    suggested_agents = _suggested_agent_ids(record)
    status = str(record.get("qaStatus") or "").strip().lower()

    if status == "needs_review":
        score += 20
        reasons.append("needs_review")
    elif status == "unlabeled":
        score += 12
        reasons.append("unlabeled")

    if focus_agent_id and focus_agent_id in missing_reviewed:
        score += 80
        reasons.append("missing_reviewed_focus")
    if any(agent in missing_reviewed for agent in suggested_agents):
        score += 60
        reasons.append("missing_reviewed_suggestion")
    if not suggested_agents:
        score += 25
        reasons.append("no_agent_suggestion")
    if _bool_value(suggested, "unknown_flag") in {"true", "1", "yes", "y"}:
        score += 18
        reasons.append("unknown_flag")

    confidence = suggested.get("confidence")
    if isinstance(confidence, dict):
        numeric = [float(value) for value in confidence.values() if isinstance(value, (int, float))]
        if numeric:
            min_conf = min(numeric)
            if min_conf < 0.45:
                score += 20
                reasons.append("very_low_conf")
            elif min_conf < 0.60:
                score += 10
                reasons.append("low_conf")

    if str(record.get("state") or "").strip().lower() == "precheck":
        score += 4
        reasons.append("precheck")

    group_key = focus_agent_id or (suggested_agents[0] if suggested_agents else source_id)
    return score, reasons, group_key


def _round_robin_order(scored: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in scored:
        buckets[str(item["groupKey"])].append(item)
    for values in buckets.values():
        values.sort(key=lambda row: (-int(row["priorityScore"]), str(row["record"]["id"])))

    ordered: List[Dict[str, Any]] = []
    while buckets:
        keys = sorted(
            buckets.keys(),
            key=lambda key: (-int(buckets[key][0]["priorityScore"]), key),
        )
        for key in keys:
            ordered.append(buckets[key].pop(0))
            if not buckets[key]:
                del buckets[key]
    return ordered


def _row(item: Dict[str, Any], source_index: Dict[str, Dict[str, Any]], batch_id: str) -> Dict[str, str]:
    record = item["record"]
    reviewed = _labels(record, "labels")
    suggested = _labels(record, "suggestedLabels")
    source_id = str(record.get("sourceId") or "src_unknown")
    return {
        "batch_id": batch_id,
        "priority_score": str(item["priorityScore"]),
        "priority_reasons": ";".join(item["priorityReasons"]),
        "record_id": str(record.get("id") or ""),
        "qa_status": str(record.get("qaStatus") or ""),
        "source_id": source_id,
        "focus_agent_id": _focus_agent(source_index, source_id),
        "state": str(record.get("state") or ""),
        "locale": str(record.get("locale") or ""),
        "resolution": str(record.get("resolution") or ""),
        "path": str(record.get("path") or ""),
        "reviewed_slot_1_agent": _str_value(reviewed, "slot_1_agent"),
        "reviewed_slot_2_agent": _str_value(reviewed, "slot_2_agent"),
        "reviewed_slot_3_agent": _str_value(reviewed, "slot_3_agent"),
        "suggested_slot_1_agent": _str_value(suggested, "slot_1_agent"),
        "suggested_slot_2_agent": _str_value(suggested, "slot_2_agent"),
        "suggested_slot_3_agent": _str_value(suggested, "slot_3_agent"),
        "reviewed_unknown_flag": _bool_value(reviewed, "unknown_flag"),
        "suggested_unknown_flag": _bool_value(suggested, "unknown_flag"),
        "suggested_confidence": _confidence_text(suggested),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build prioritized CV human-review batches.")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--output-csv", default="docs/review_queue.priority.csv")
    parser.add_argument("--output-json", default="docs/review_batches.json")
    parser.add_argument("--status", default="needs_review")
    parser.add_argument("--batch-size", type=int, default=120)
    parser.add_argument("--max-batches", type=int, default=8)
    args = parser.parse_args()

    manifest = ensure_manifest_defaults(load_manifest(Path(args.manifest).resolve()))
    records = manifest.get("records", [])
    sources = manifest.get("sources", [])
    if not isinstance(records, list):
        raise ValueError("manifest.records must be an array")
    source_index = {
        str(source.get("sourceId") or ""): source
        for source in sources
        if isinstance(source, dict) and str(source.get("sourceId") or "")
    }

    reviewed_counts = _reviewed_agent_counts(records)
    missing_reviewed = {agent for agent in current_agent_ids() if reviewed_counts.get(agent, 0) <= 0}
    status_filter = {item.strip().lower() for item in str(args.status).split(",") if item.strip()}

    scored: List[Dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        status = str(record.get("qaStatus") or "").strip().lower()
        if status_filter and status not in status_filter:
            continue
        priority_score, priority_reasons, group_key = _score_record(record, source_index, missing_reviewed)
        if priority_score <= 0:
            continue
        scored.append(
            {
                "record": record,
                "priorityScore": priority_score,
                "priorityReasons": priority_reasons,
                "groupKey": group_key,
            }
        )

    ordered = _round_robin_order(scored)
    max_rows = max(1, int(args.batch_size)) * max(1, int(args.max_batches))
    ordered = ordered[:max_rows]

    rows: List[Dict[str, str]] = []
    batches: List[Dict[str, Any]] = []
    for index in range(0, len(ordered), max(1, int(args.batch_size))):
        chunk = ordered[index : index + max(1, int(args.batch_size))]
        batch_id = f"cv-review-{(index // max(1, int(args.batch_size))) + 1:03d}"
        focus_counts = Counter()
        for item in chunk:
            source_id = str(item["record"].get("sourceId") or "src_unknown")
            focus_agent_id = _focus_agent(source_index, source_id)
            if focus_agent_id:
                focus_counts[focus_agent_id] += 1
            rows.append(_row(item, source_index, batch_id))
        batches.append(
            {
                "batchId": batch_id,
                "size": len(chunk),
                "maxPriority": max(int(item["priorityScore"]) for item in chunk),
                "focusAgentCounts": dict(focus_counts),
            }
        )

    output_csv = Path(args.output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "generatedAt": utc_now(),
        "statusFilter": sorted(status_filter),
        "batchSize": max(1, int(args.batch_size)),
        "batchCount": len(batches),
        "queuedRecords": len(rows),
        "missingReviewedAgents": sorted(missing_reviewed),
        "batches": batches,
    }
    output_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
