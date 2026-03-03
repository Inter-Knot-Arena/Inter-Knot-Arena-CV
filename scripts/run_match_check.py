from __future__ import annotations

import argparse
import json

from runtime.matcher import evaluate_detection


def parse_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CV precheck/in-run matcher")
    parser.add_argument("--mode", choices=["PRECHECK", "INRUN"], default="PRECHECK")
    parser.add_argument("--expected", required=True, help="Comma-separated expected agents")
    parser.add_argument("--detected", required=True, help="Comma-separated detected agents")
    args = parser.parse_args()

    result = evaluate_detection(
        expected_agents=parse_list(args.expected),
        detected_agents=parse_list(args.detected),
        mode=args.mode,
    )
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
