from __future__ import annotations

import json
import threading
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Dict, Sequence

import cv2
import numpy as np
import onnxruntime as ort

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "assets" / "templates"
MODEL_MANIFEST_PATH = MODEL_DIR / "model_manifest.json"


def _provider_priority() -> list[str]:
    available = set(ort.get_available_providers())
    preferred = ["DmlExecutionProvider", "CPUExecutionProvider"]
    return [provider for provider in preferred if provider in available] or ["CPUExecutionProvider"]


def _read_labels(path: Path) -> tuple[list[str], Dict[int, str]]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    labels = payload.get("labels")
    if not isinstance(labels, list) or not labels:
        raise ValueError(f"Invalid labels file: {path}")
    normalized_labels = [str(item) for item in labels]
    class_ids_raw = payload.get("classIds")
    class_id_map: Dict[int, str] = {}
    if isinstance(class_ids_raw, list) and len(class_ids_raw) == len(normalized_labels):
        for class_id_value, label in zip(class_ids_raw, normalized_labels):
            if isinstance(class_id_value, (int, np.integer)):
                class_id_map[int(class_id_value)] = label
    return normalized_labels, class_id_map


def get_model_metadata(default_version: str) -> Dict[str, str]:
    if not MODEL_MANIFEST_PATH.exists():
        return {"modelVersion": default_version, "dataVersion": "unknown"}
    try:
        with MODEL_MANIFEST_PATH.open("r", encoding="utf-8") as fh:
            payload: Any = json.load(fh)
        if not isinstance(payload, dict):
            return {"modelVersion": default_version, "dataVersion": "unknown"}
        model_version = str(payload.get("version") or default_version)
        data_version = str(payload.get("dataVersion") or "unknown")
        return {"modelVersion": model_version, "dataVersion": data_version}
    except Exception:
        return {"modelVersion": default_version, "dataVersion": "unknown"}


def _extract_probabilities(raw_output: object, labels: Sequence[str], class_id_map: Dict[int, str]) -> Dict[str, float]:
    if isinstance(raw_output, list) and raw_output and isinstance(raw_output[0], dict):
        probabilities: Dict[str, float] = {}
        for key, value in raw_output[0].items():
            label = _map_label_key(key, labels, class_id_map)
            probabilities[label] = float(value)
        return probabilities

    if isinstance(raw_output, np.ndarray):
        array = raw_output
        if array.ndim == 1:
            probs = array
        elif array.ndim >= 2:
            probs = array[0]
        else:
            probs = np.array([], dtype=np.float32)
        probs = np.asarray(probs).reshape(-1)
        if probs.size == 1 and len(labels) > 1:
            first_value = float(probs[0])
            if np.issubdtype(probs.dtype, np.integer) or first_value < 0.0 or first_value > 1.0:
                return {}
        return {
            label: float(probs[idx])
            for idx, label in enumerate(labels)
            if idx < probs.shape[0]
        }
    return {}


def _probability_quality(probabilities: Dict[str, float]) -> tuple[int, int, float, float]:
    if not probabilities:
        return (0, 0, 0.0, 0.0)
    values = [float(value) for value in probabilities.values() if math.isfinite(float(value))]
    if not values:
        return (0, 0, 0.0, 0.0)
    in_unit_range = sum(1 for value in values if 0.0 <= value <= 1.0)
    value_sum = float(sum(values))
    peak = float(max(values))
    return (len(values), in_unit_range, value_sum, peak)


def _map_label_key(raw: object, labels: Sequence[str], class_id_map: Dict[int, str]) -> str:
    if isinstance(raw, (np.integer, int)):
        idx = int(raw)
        mapped = class_id_map.get(idx)
        if mapped:
            return mapped
        if 0 <= idx < len(labels):
            return labels[idx]
        return str(idx)

    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="ignore")

    text = str(raw).strip()
    if text.isdigit():
        idx = int(text)
        mapped = class_id_map.get(idx)
        if mapped:
            return mapped
        if 0 <= idx < len(labels):
            return labels[idx]
    return text


@dataclass(slots=True)
class Prediction:
    label: str
    confidence: float
    probabilities: Dict[str, float]


class CvAgentClassifier:
    _instance: "CvAgentClassifier | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.model_path = MODEL_DIR / "cv_agent_icon.onnx"
        self.labels_path = MODEL_DIR / "cv_agent_icon.labels.json"
        if not self.model_path.exists():
            raise FileNotFoundError(f"CV model missing: {self.model_path}")
        if not self.labels_path.exists():
            raise FileNotFoundError(f"CV labels missing: {self.labels_path}")
        self.labels, self.class_id_map = _read_labels(self.labels_path)
        self.session = ort.InferenceSession(str(self.model_path), providers=_provider_priority())
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    @classmethod
    def instance(cls) -> "CvAgentClassifier":
        with cls._lock:
            if cls._instance is None:
                cls._instance = CvAgentClassifier()
            return cls._instance

    @classmethod
    def exists(cls) -> bool:
        return (MODEL_DIR / "cv_agent_icon.onnx").exists() and (MODEL_DIR / "cv_agent_icon.labels.json").exists()

    def predict(self, icon: np.ndarray) -> Prediction:
        if icon.ndim == 2:
            icon = cv2.cvtColor(icon, cv2.COLOR_GRAY2BGR)
        resized = cv2.resize(icon, (32, 32), interpolation=cv2.INTER_AREA)
        feature = (resized.astype(np.float32) / 255.0).reshape(1, -1)

        outputs = self.session.run(self.output_names, {self.input_name: feature})
        label: str | None = None
        probability_candidates: list[Dict[str, float]] = []

        for output in outputs:
            if label is None:
                if isinstance(output, np.ndarray) and output.size > 0:
                    label = _map_label_key(output[0], self.labels, self.class_id_map)
                elif isinstance(output, list) and output:
                    label = _map_label_key(output[0], self.labels, self.class_id_map)

            candidate = _extract_probabilities(output, self.labels, self.class_id_map)
            if candidate:
                probability_candidates.append(candidate)

        probabilities: Dict[str, float] = {}
        if probability_candidates:
            probabilities = max(probability_candidates, key=_probability_quality)

        if label is None:
            label = max(probabilities.items(), key=lambda item: item[1])[0] if probabilities else self.labels[0]

        confidence = float(probabilities.get(label, 0.0))
        if confidence <= 0.0 and probabilities:
            confidence = float(max(probabilities.values()))
        if confidence <= 0.0 and not probabilities:
            confidence = 0.51
        return Prediction(label=label, confidence=confidence, probabilities=probabilities)


def template_scores(icon: np.ndarray) -> Dict[str, float]:
    if icon.ndim == 2:
        icon_bgr = cv2.cvtColor(icon, cv2.COLOR_GRAY2BGR)
    else:
        icon_bgr = icon

    scores: Dict[str, float] = {}
    if not TEMPLATE_DIR.exists():
        return scores
    for template_path in TEMPLATE_DIR.glob("*.png"):
        template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
        if template is None:
            continue
        resized = cv2.resize(icon_bgr, (template.shape[1], template.shape[0]), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        scores[template_path.stem] = float(max_val)
    return scores
