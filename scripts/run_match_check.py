from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.matcher import evaluate_detection


def parse_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CV precheck/in-run matcher")
    parser.add_argument("--mode", choices=["PRECHECK", "INRUN"], default="PRECHECK")
    parser.add_argument("--expected", required=True, help="Comma-separated expected agents")
    parser.add_argument("--detected", required=True, help="Comma-separated detected agents")
    parser.add_argument("--history", default="", help="Comma-separated previous in-run detections")
    parser.add_argument("--locale", default="EN", choices=["EN", "RU"])
    parser.add_argument("--resolution", default="1080p", choices=["1080p", "1440p"])
    parser.add_argument("--frame-path", default="", help="Optional screenshot path for real detection")
    parser.add_argument("--orientation", default="vertical", choices=["vertical", "horizontal"])
    args = parser.parse_args()

    result = evaluate_detection(
        expected_agents=parse_list(args.expected),
        detected_agents=parse_list(args.detected),
        mode=args.mode,
        locale=args.locale,
        resolution=args.resolution,
        history_agents=parse_list(args.history),
        frame_path=args.frame_path or None,
        orientation=args.orientation,
    )
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
