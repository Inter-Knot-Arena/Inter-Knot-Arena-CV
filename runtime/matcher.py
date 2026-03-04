from __future__ import annotations

import hashlib
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np
from PIL import ImageGrab

from .model_runtime import CvAgentClassifier, template_scores

SUPPORTED_LOCALES = {"RU", "EN"}
SUPPORTED_RESOLUTIONS = {"1080p", "1440p"}
SUPPORTED_MODES = {"PRECHECK", "INRUN"}
DEFAULT_MODEL_VERSION = "cv-onnx-template-v1.2"


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
    return min(0.09, occurrences * 0.02)


def _build_frame_hash(
    mode: str,
    expected: List[str],
    detected: List[str],
    locale: str,
    resolution: str,
) -> str:
    payload = f"{mode}:{locale}:{resolution}:{','.join(expected)}:{','.join(detected)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _capture_frame(region: Tuple[int, int, int, int] | None) -> np.ndarray | None:
    try:
        bbox = None
        if region:
            x, y, w, h = region
            bbox = (x, y, x + w, y + h)
        image = ImageGrab.grab(bbox=bbox, all_screens=True)
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def _read_frame(frame_path: str | None, region: Tuple[int, int, int, int] | None) -> np.ndarray | None:
    if frame_path:
        path = Path(frame_path)
        if path.exists():
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is not None:
                return image
    return _capture_frame(region)


def _slot_crops(frame: np.ndarray, orientation: str = "vertical", slots: int = 3) -> list[np.ndarray]:
    h, w = frame.shape[:2]
    crops: list[np.ndarray] = []
    if orientation == "horizontal":
        step = max(1, w // slots)
        for idx in range(slots):
            x0 = idx * step
            x1 = w if idx == slots - 1 else min(w, (idx + 1) * step)
            crops.append(frame[:, x0:x1])
    else:
        step = max(1, h // slots)
        for idx in range(slots):
            y0 = idx * step
            y1 = h if idx == slots - 1 else min(h, (idx + 1) * step)
            crops.append(frame[y0:y1, :])
    return crops


def _classify_from_frame(
    frame: np.ndarray,
    orientation: str,
    history_agents: Iterable[str],
) -> Tuple[list[str], Dict[str, float], list[str]]:
    reasons: list[str] = []
    detected: list[str] = []
    confidence: Dict[str, float] = {}

    if not CvAgentClassifier.exists():
        reasons.append("cv_model_missing")
        return detected, confidence, reasons

    classifier = CvAgentClassifier.instance()
    for crop in _slot_crops(frame, orientation=orientation, slots=3):
        prediction = classifier.predict(crop)
        template = template_scores(crop)
        template_score = float(template.get(prediction.label, 0.0))
        temporal = _temporal_support(prediction.label, history_agents)
        merged_conf = max(0.0, min(0.995, prediction.confidence * 0.7 + template_score * 0.2 + temporal * 0.1))
        if merged_conf < 0.55:
            reasons.append(f"low_conf_slot:{prediction.label}")
            continue
        if prediction.label not in detected:
            detected.append(prediction.label)
            confidence[prediction.label] = round(merged_conf, 4)

    if not detected:
        reasons.append("no_detected_agents")
    return detected, confidence, reasons


def evaluate_detection(
    expected_agents: Iterable[str],
    detected_agents: Iterable[str],
    mode: str,
    locale: str = "EN",
    resolution: str = "1080p",
    history_agents: Iterable[str] | None = None,
    frame_hash_hint: str | None = None,
    frame_path: str | None = None,
    capture_region: Tuple[int, int, int, int] | None = None,
    orientation: str = "vertical",
    capture_screen: bool = False,
) -> Dict[str, Any]:
    started = time.perf_counter()

    normalized_mode = _normalize_mode(mode)
    layout_reasons = _validate_layout(locale=locale, resolution=resolution)

    expected = list(dict.fromkeys(_normalize(agent) for agent in expected_agents if _normalize(agent)))
    detected_payload = list(dict.fromkeys(_normalize(agent) for agent in detected_agents if _normalize(agent)))
    history = list(history_agents or [])

    confidence: Dict[str, float] = {}
    low_conf_reasons: list[str] = []
    low_conf_reasons.extend(layout_reasons)

    should_capture = bool(frame_path) or capture_region is not None or capture_screen
    frame = _read_frame(frame_path=frame_path, region=capture_region) if should_capture else None
    if frame is not None:
        detected_from_frame, frame_confidence, frame_reasons = _classify_from_frame(
            frame=frame, orientation=orientation, history_agents=history
        )
        if detected_from_frame:
            detected = detected_from_frame
            confidence = frame_confidence
        else:
            detected = detected_payload
        low_conf_reasons.extend(frame_reasons)
    else:
        detected = detected_payload
        if not detected:
            low_conf_reasons.append("frame_capture_failed")

    unexpected = [agent for agent in detected if agent not in expected]
    missing = [agent for agent in expected if agent not in detected]
    if missing:
        low_conf_reasons.append("missing_expected_agents")

    for agent in detected:
        if agent not in confidence:
            base = 0.9 if agent in expected else 0.68
            if normalized_mode == "INRUN":
                base += _temporal_support(agent, history)
            confidence[agent] = round(min(0.995, max(0.1, base)), 4)

    if unexpected:
        result = "VIOLATION"
    elif low_conf_reasons:
        result = "LOW_CONF"
    else:
        result = "PASS"

    confidence_by_field = {
        "detection": round(sum(confidence.values()) / max(len(confidence), 1), 4),
        "matching": 0.99 if result == "PASS" else (0.45 if result == "VIOLATION" else 0.78),
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
