from __future__ import annotations

import hashlib
import time
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple

SUPPORTED_LOCALES = {"RU", "EN"}
SUPPORTED_RESOLUTIONS = {"1080p", "1440p"}
SUPPORTED_MODES = {"PRECHECK", "INRUN"}
DEFAULT_MODEL_VERSION = "cv-hybrid-v1.1"


def _normalize(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def _normalize_mode(mode: str) -> str:
    normalized = mode.strip().upper()
    if normalized not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported mode: {mode}")
    return normalized


def _validate_layout(locale: str, resolution: str) -> list[str]:
    reasons: list[str] = []
    if locale.upper() not in SUPPORTED_LOCALES:
        reasons.append(f"unsupported_locale:{locale}")
    if resolution.lower() not in SUPPORTED_RESOLUTIONS:
        reasons.append(f"unsupported_resolution:{resolution}")
    return reasons


def _temporal_support(agent: str, history: Iterable[str]) -> float:
    counter = Counter(_normalize(item) for item in history if item)
    occurrences = counter.get(_normalize(agent), 0)
    if occurrences <= 0:
        return 0.0
    return min(0.06, occurrences * 0.015)


def _build_frame_hash(
    mode: str,
    expected: List[str],
    detected: List[str],
    locale: str,
    resolution: str,
) -> str:
    payload = f"{mode}:{locale}:{resolution}:{','.join(expected)}:{','.join(detected)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def evaluate_detection(
    expected_agents: Iterable[str],
    detected_agents: Iterable[str],
    mode: str,
    locale: str = "EN",
    resolution: str = "1080p",
    history_agents: Iterable[str] | None = None,
    frame_hash_hint: str | None = None,
) -> Dict[str, Any]:
    started = time.perf_counter()

    normalized_mode = _normalize_mode(mode)
    layout_reasons = _validate_layout(locale=locale, resolution=resolution)

    expected = list(dict.fromkeys(_normalize(agent) for agent in expected_agents if _normalize(agent)))
    detected = list(dict.fromkeys(_normalize(agent) for agent in detected_agents if _normalize(agent)))
    history = list(history_agents or [])

    unexpected = [agent for agent in detected if agent not in expected]
    missing = [agent for agent in expected if agent not in detected]

    low_conf_reasons: list[str] = []
    low_conf_reasons.extend(layout_reasons)
    if missing:
        low_conf_reasons.append("missing_expected_agents")
    if not detected:
        low_conf_reasons.append("no_detected_agents")

    confidence: Dict[str, float] = {}
    for agent in detected:
        base = 0.91 if agent in expected else 0.68
        if normalized_mode == "INRUN":
            base += _temporal_support(agent, history)
        confidence[agent] = round(min(0.995, max(0.15, base)), 4)

    if unexpected:
        result = "VIOLATION"
    elif low_conf_reasons:
        result = "LOW_CONF"
    else:
        result = "PASS"

    confidence_by_field = {
        "detection": round(sum(confidence.values()) / max(len(confidence), 1), 4),
        "matching": 0.99 if result == "PASS" else (0.42 if result == "VIOLATION" else 0.79),
        "temporal": round(
            0.9 if normalized_mode == "INRUN" and history else 0.7 if normalized_mode == "INRUN" else 1.0,
            4,
        ),
    }

    output = {
        "type": normalized_mode,
        "detectedAgents": detected,
        "unexpectedAgents": unexpected,
        "confidence": confidence,
        "confidenceByField": confidence_by_field,
        "result": result,
        "frameHash": frame_hash_hint or _build_frame_hash(normalized_mode, expected, detected, locale, resolution),
        "modelVersion": DEFAULT_MODEL_VERSION,
        "lowConfReasons": sorted(set(low_conf_reasons)),
        "timingMs": round((time.perf_counter() - started) * 1000.0, 2),
        "resolution": resolution.lower(),
        "locale": locale.upper(),
    }
    return output
