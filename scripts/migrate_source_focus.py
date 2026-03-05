from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roster_taxonomy import source_focus_agent_ids


def _normalize_source(source: dict) -> dict:
    focus_ids = source_focus_agent_ids(source)
    if focus_ids:
        source["focusAgentId"] = focus_ids[0]
        if len(focus_ids) > 1:
            source["focusAgentIds"] = focus_ids
        else:
            source.pop("focusAgentIds", None)
    return source


def main() -> int:
    parser = argparse.ArgumentParser(description="Populate explicit focusAgentId/focusAgentIds from legacy source metadata.")
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    path = Path(args.input).resolve()
    payload: Any = json.loads(path.read_text(encoding="utf-8"))

    updated = 0
    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            before = json.dumps(item, ensure_ascii=True, sort_keys=True)
            _normalize_source(item)
            after = json.dumps(item, ensure_ascii=True, sort_keys=True)
            if before != after:
                updated += 1
    elif isinstance(payload, dict):
        sources = payload.get("sources")
        if not isinstance(sources, list):
            raise ValueError("Expected manifest with sources[]")
        for item in sources:
            if not isinstance(item, dict):
                continue
            before = json.dumps(item, ensure_ascii=True, sort_keys=True)
            _normalize_source(item)
            after = json.dumps(item, ensure_ascii=True, sort_keys=True)
            if before != after:
                updated += 1
    else:
        raise ValueError("Unsupported JSON shape")

    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"updated": updated, "path": str(path)}, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
