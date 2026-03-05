from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict

from manifest_lib import ensure_manifest_defaults, load_manifest


def _extract_agent_label(record: Dict[str, Any]) -> str:
    labels = record.get("labels")
    if isinstance(labels, dict):
        for key in ("slot_1_agent", "slot_2_agent", "slot_3_agent", "agentId", "label"):
            value = labels.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for key in ("agentId", "label"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def _is_unknown(record: Dict[str, Any]) -> bool:
    labels = record.get("labels")
    if isinstance(labels, dict):
        unknown = labels.get("unknown_flag")
        if isinstance(unknown, bool):
            return unknown
        if isinstance(unknown, str):
            return unknown.strip().lower() in {"1", "true", "yes", "y"}
    value = record.get("unknownFlag")
    if isinstance(value, bool):
        return value
    return _extract_agent_label(record) == "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build class-balance sampling plan from CV dataset manifest.")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--target-per-agent", type=int, default=6000)
    parser.add_argument("--target-unknown", type=int, default=8000)
    parser.add_argument("--output-file", default="")
    args = parser.parse_args()

    manifest = ensure_manifest_defaults(load_manifest(Path(args.manifest).resolve()))
    records = manifest.get("records", [])
    if not isinstance(records, list):
        raise ValueError("manifest.records must be an array")

    agent_counts: Counter[str] = Counter()
    locale_counts: Counter[str] = Counter()
    resolution_counts: Counter[str] = Counter()
    state_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    unknown_count = 0

    for record in records:
        if not isinstance(record, dict):
            continue
        agent = _extract_agent_label(record)
        agent_counts[agent] += 1
        locale_counts[str(record.get("locale") or "unknown")] += 1
        resolution_counts[str(record.get("resolution") or "unknown")] += 1
        state_counts[str(record.get("state") or "other")] += 1
        source_counts[str(record.get("sourceId") or "src_unknown")] += 1
        if _is_unknown(record):
            unknown_count += 1

    deficits: Dict[str, int] = {}
    for agent, count in sorted(agent_counts.items()):
        target = max(1, args.target_per_agent)
        deficits[agent] = max(0, target - count)
    deficits["unknown"] = max(0, max(1, args.target_unknown) - unknown_count)

    per_locale_resolution = defaultdict(int)
    for record in records:
        if not isinstance(record, dict):
            continue
        key = f"{str(record.get('locale') or 'unknown')}:{str(record.get('resolution') or 'unknown')}"
        per_locale_resolution[key] += 1

    plan = {
        "recordCount": len(records),
        "agentCounts": dict(agent_counts),
        "unknownCount": unknown_count,
        "localeCounts": dict(locale_counts),
        "resolutionCounts": dict(resolution_counts),
        "stateCounts": dict(state_counts),
        "sourceCounts": dict(source_counts),
        "localeResolutionCounts": dict(per_locale_resolution),
        "deficits": deficits,
    }

    if args.output_file:
        output_path = Path(args.output_file).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(plan, fh, ensure_ascii=True, indent=2)
            fh.write("\n")

    print(json.dumps(plan, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

