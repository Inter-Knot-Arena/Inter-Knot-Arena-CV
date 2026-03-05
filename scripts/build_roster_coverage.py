from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict

from manifest_lib import ensure_manifest_defaults, load_manifest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roster_taxonomy import canonicalize_agent_label, current_agent_ids, focus_agents_from_sources


def _label_payload(record: Dict[str, Any], suggested: bool) -> Dict[str, Any]:
    key = "suggestedLabels" if suggested else "labels"
    labels = record.get(key)
    if not isinstance(labels, dict):
        return {}
    return labels


def _iter_slot_labels(record: Dict[str, Any], *, suggested: bool) -> list[str]:
    labels = _label_payload(record, suggested=suggested)
    if not labels:
        return []
    output: list[str] = []
    for key in ("slot_1_agent", "slot_2_agent", "slot_3_agent"):
        value = labels.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        canonical = canonicalize_agent_label(value.strip())
        if canonical:
            output.append(canonical)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Build roster coverage report for CV manifest.")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--output-file", default="docs/roster_coverage.json")
    args = parser.parse_args()

    manifest = ensure_manifest_defaults(load_manifest(Path(args.manifest).resolve()))
    records = manifest.get("records", [])
    sources = manifest.get("sources", [])
    if not isinstance(records, list):
        raise ValueError("manifest.records must be an array")
    if not isinstance(sources, list):
        raise ValueError("manifest.sources must be an array")

    roster_agents = current_agent_ids()
    focused_counts = focus_agents_from_sources(sources)
    reviewed_counts: Counter[str] = Counter()
    suggested_counts: Counter[str] = Counter()

    for record in records:
        if not isinstance(record, dict):
            continue
        for agent_id in _iter_slot_labels(record, suggested=False):
            reviewed_counts[agent_id] += 1
        for agent_id in _iter_slot_labels(record, suggested=True):
            suggested_counts[agent_id] += 1

    report = {
        "rosterAgentCount": len(roster_agents),
        "sourceFocusedAgentCount": sum(1 for agent in roster_agents if focused_counts.get(agent, 0) > 0),
        "labeledAgentCount": sum(1 for agent in roster_agents if reviewed_counts.get(agent, 0) > 0),
        "reviewedAgentCount": sum(1 for agent in roster_agents if reviewed_counts.get(agent, 0) > 0),
        "suggestedAgentCount": sum(1 for agent in roster_agents if suggested_counts.get(agent, 0) > 0),
        "sourceFocusedCounts": {agent: int(focused_counts.get(agent, 0)) for agent in roster_agents},
        "labeledCounts": {agent: int(reviewed_counts.get(agent, 0)) for agent in roster_agents},
        "reviewedCounts": {agent: int(reviewed_counts.get(agent, 0)) for agent in roster_agents},
        "suggestedCounts": {agent: int(suggested_counts.get(agent, 0)) for agent in roster_agents},
        "missingSourceFocusAgents": [agent for agent in roster_agents if focused_counts.get(agent, 0) <= 0],
        "missingLabeledAgents": [agent for agent in roster_agents if reviewed_counts.get(agent, 0) <= 0],
        "missingReviewedAgents": [agent for agent in roster_agents if reviewed_counts.get(agent, 0) <= 0],
        "missingSuggestedAgents": [agent for agent in roster_agents if suggested_counts.get(agent, 0) <= 0],
    }

    output_path = Path(args.output_file).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=True, indent=2)
        fh.write("\n")

    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
