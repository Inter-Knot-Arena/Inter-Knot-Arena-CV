from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from manifest_lib import ensure_manifest_defaults, load_manifest, save_manifest, utc_now

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roster_taxonomy import canonicalize_agent_label
from runtime.layout import extract_team_slot_crops
from runtime.model_runtime import CvAgentClassifier, template_scores


def _normalize_agent_label(label: str) -> str:
    return canonicalize_agent_label(label) or "unknown"


def _predict_slot(classifier: CvAgentClassifier, crop: np.ndarray) -> Tuple[str, float]:
    prediction = classifier.predict(crop)
    templates = template_scores(crop)

    best_template_label = ""
    best_template_score = 0.0
    if templates:
        best_template_label, best_template_score = max(templates.items(), key=lambda item: item[1])

    chosen_label = _normalize_agent_label(prediction.label)
    if best_template_label and best_template_score > max(0.58, prediction.confidence + 0.05):
        chosen_label = _normalize_agent_label(best_template_label)

    template_score_for_chosen = float(templates.get(chosen_label, 0.0))
    confidence = max(
        0.0,
        min(
            0.999,
            prediction.confidence * 0.55 + template_score_for_chosen * 0.45,
        ),
    )
    return chosen_label, confidence


def _has_labels(record: Dict[str, Any]) -> bool:
    labels = record.get("labels")
    if isinstance(labels, dict):
        if isinstance(labels.get("reviewFinal"), dict):
            return True
        if any(isinstance(value, str) and value for value in labels.values()):
            return True
    suggested = record.get("suggestedLabels")
    return isinstance(suggested, dict) and any(isinstance(value, str) and value for value in suggested.values())


def _prelabel_record(
    record: Dict[str, Any],
    classifier: CvAgentClassifier,
    confidence_threshold: float,
) -> Tuple[bool, str]:
    path_value = str(record.get("path") or "")
    if not path_value:
        return False, "missing_path"
    image_path = Path(path_value)
    if not image_path.exists():
        return False, "path_not_found"

    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame is None:
        return False, "decode_failed"

    state = str(record.get("state") or "other").lower()

    labels: Dict[str, Any] = {
        "state": state,
        "occlusion": "unknown",
        "ui_variant": str(record.get("uiVariant") or "default"),
        "unknown_flag": False,
        "prelabelVersion": "cv-prelabel-v2",
        "prelabelAt": utc_now(),
    }
    conf_map: Dict[str, float] = {}
    unknown = False

    slot_crops = extract_team_slot_crops(frame, slots=3)
    if len(slot_crops) != 3:
        return False, "team_strip_missing"

    for index, crop in enumerate(slot_crops, start=1):
        label, confidence = _predict_slot(classifier=classifier, crop=crop)
        if confidence < confidence_threshold:
            label = "unknown"
            unknown = True
        labels[f"slot_{index}_agent"] = label
        conf_map[f"slot_{index}_agent"] = round(confidence, 4)

    labels["unknown_flag"] = unknown
    labels["confidence"] = conf_map
    record["suggestedLabels"] = labels
    record["qaStatus"] = "needs_review"
    return True, "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description="Semi-automatic prelabel for CV dataset records.")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = ensure_manifest_defaults(load_manifest(manifest_path))
    records = manifest.get("records", [])
    if not isinstance(records, list):
        raise ValueError("manifest.records must be an array")

    if not CvAgentClassifier.exists():
        raise FileNotFoundError("CV model is missing. Expected models/cv_agent_icon.onnx and labels.")
    classifier = CvAgentClassifier.instance()

    processed = 0
    updated = 0
    skipped_existing = 0
    errors: Dict[str, int] = {}
    limit = max(0, int(args.max_records))

    for record in records:
        if not isinstance(record, dict):
            continue
        if not args.overwrite and _has_labels(record):
            skipped_existing += 1
            continue
        if limit and processed >= limit:
            break
        processed += 1
        ok, reason = _prelabel_record(
            record=record,
            classifier=classifier,
            confidence_threshold=max(0.0, min(1.0, args.confidence_threshold)),
        )
        if ok:
            updated += 1
        else:
            errors[reason] = errors.get(reason, 0) + 1

    qa_status = manifest.setdefault("qaStatus", {})
    if isinstance(qa_status, dict):
        qa_status["prelabel"] = "completed" if updated > 0 else "pending"
        qa_status["prelabelUpdatedAt"] = utc_now()
        qa_status["prelabelProcessed"] = processed
        qa_status["prelabelUpdated"] = updated

    save_manifest(manifest_path, manifest)
    print(
        json.dumps(
            {
                "processed": processed,
                "updated": updated,
                "skippedExisting": skipped_existing,
                "errors": errors,
            },
            ensure_ascii=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
