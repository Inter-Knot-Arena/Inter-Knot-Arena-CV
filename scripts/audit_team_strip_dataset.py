from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict

import cv2

from manifest_lib import ensure_manifest_defaults, load_manifest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime.layout import detect_team_strip


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit how many CV manifest frames contain a valid team-strip ROI.")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--output-json", default="docs/team_strip_audit.json")
    args = parser.parse_args()

    manifest = ensure_manifest_defaults(load_manifest(Path(args.manifest).resolve()))
    records = manifest.get("records", [])
    if not isinstance(records, list):
        raise ValueError("manifest.records must be an array")

    totals = Counter()
    valid = Counter()
    invalid_sources: Counter[str] = Counter()

    for record in records:
        if not isinstance(record, dict):
            continue
        state = str(record.get("state") or "").strip().lower()
        if state not in {"precheck", "inrun"}:
            continue

        path = Path(str(record.get("path") or ""))
        if not path.exists():
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        totals[state] += 1
        totals["all"] += 1
        detection = detect_team_strip(image, slots=3)
        if detection is not None:
            valid[state] += 1
            valid["all"] += 1
            valid[f"layout:{detection.layout_name}"] += 1
            continue
        invalid_sources[str(record.get("sourceId") or "src_unknown")] += 1

    output = {
        "totals": {
            "all": int(totals["all"]),
            "precheck": int(totals["precheck"]),
            "inrun": int(totals["inrun"]),
        },
        "valid": {
            "all": int(valid["all"]),
            "precheck": int(valid["precheck"]),
            "inrun": int(valid["inrun"]),
            "gameplay_top_bar": int(valid["layout:gameplay_top_bar"]),
            "menu_top_bar": int(valid["layout:menu_top_bar"]),
        },
        "invalid": {
            "all": max(0, int(totals["all"]) - int(valid["all"])),
            "precheck": max(0, int(totals["precheck"]) - int(valid["precheck"])),
            "inrun": max(0, int(totals["inrun"]) - int(valid["inrun"])),
        },
        "invalidSourcesTop20": [{"sourceId": source_id, "count": int(count)} for source_id, count in invalid_sources.most_common(20)],
    }

    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=True, indent=2)
        fh.write("\n")

    print(json.dumps(output, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
