from __future__ import annotations

import hashlib
from typing import Dict, Iterable, List


def evaluate_detection(
    expected_agents: Iterable[str],
    detected_agents: Iterable[str],
    mode: str,
) -> Dict[str, object]:
    expected = list(dict.fromkeys(expected_agents))
    detected = list(dict.fromkeys(detected_agents))

    unexpected = [agent for agent in detected if agent not in expected]
    missing = [agent for agent in expected if agent not in detected]

    if unexpected:
        result = "VIOLATION"
    elif missing:
        result = "LOW_CONF"
    else:
        result = "PASS"

    frame_seed = f"{mode}:{','.join(expected)}:{','.join(detected)}"
    frame_hash = hashlib.sha1(frame_seed.encode("utf-8")).hexdigest()

    confidence = {agent: 0.97 for agent in detected}
    confidence_by_field = {
        "detection": 0.95 if result == "PASS" else 0.82,
        "matching": 0.96 if not unexpected else 0.7,
    }

    return {
        "type": mode,
        "detectedAgents": detected,
        "unexpectedAgents": unexpected,
        "confidence": confidence,
        "confidenceByField": confidence_by_field,
        "result": result,
        "frameHash": frame_hash,
        "modelVersion": "cv-hybrid-v1",
    }
