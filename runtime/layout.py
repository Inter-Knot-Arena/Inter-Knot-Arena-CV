from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np

_LAYOUT_CANDIDATES: tuple[tuple[str, float, float, float, float], ...] = (
    ("gameplay_top_bar", 0.12, 0.62, 0.0, 0.14),
    ("menu_top_bar", 0.28, 0.78, 0.0, 0.12),
)
_MIN_SLOT_EDGE_RATIO = 0.05
_MIN_SLOT_STDDEV = 40.0
_MIN_ACTIVE_SLOTS = 2


@dataclass(frozen=True, slots=True)
class TeamStripDetection:
    layout_name: str
    strip: np.ndarray
    slot_crops: List[np.ndarray]
    active_slots: int
    score: float


def _crop_relative(frame: np.ndarray, roi: Sequence[float]) -> np.ndarray:
    h, w = frame.shape[:2]
    x0 = max(0, min(w, int(round(w * float(roi[0])))))
    x1 = max(0, min(w, int(round(w * float(roi[1])))))
    y0 = max(0, min(h, int(round(h * float(roi[2])))))
    y1 = max(0, min(h, int(round(h * float(roi[3])))))
    if x1 <= x0 or y1 <= y0:
        return np.empty((0, 0, 3), dtype=frame.dtype)
    return frame[y0:y1, x0:x1]


def _split_horizontal(strip: np.ndarray, slots: int) -> List[np.ndarray]:
    if strip.size <= 0 or slots <= 0:
        return []
    width = strip.shape[1]
    step = max(1, width // slots)
    output: List[np.ndarray] = []
    for idx in range(slots):
        x0 = idx * step
        x1 = width if idx == slots - 1 else min(width, (idx + 1) * step)
        crop = strip[:, x0:x1]
        if crop.size > 0:
            output.append(crop)
    return output


def _slot_metrics(slot: np.ndarray) -> Tuple[float, float]:
    gray = cv2.cvtColor(slot, cv2.COLOR_BGR2GRAY) if slot.ndim == 3 else slot
    edge_ratio = float((cv2.Canny(gray, 50, 150) > 0).mean())
    stddev = float(gray.std())
    return edge_ratio, stddev


def detect_team_strip(frame: np.ndarray, slots: int = 3) -> TeamStripDetection | None:
    if frame is None or frame.size <= 0:
        return None

    best: TeamStripDetection | None = None
    for layout_name, x0, x1, y0, y1 in _LAYOUT_CANDIDATES:
        strip = _crop_relative(frame, (x0, x1, y0, y1))
        slot_crops = _split_horizontal(strip, slots=slots)
        if len(slot_crops) != slots:
            continue

        metrics = [_slot_metrics(crop) for crop in slot_crops]
        active_slots = sum(
            1 for edge_ratio, stddev in metrics if edge_ratio >= _MIN_SLOT_EDGE_RATIO and stddev >= _MIN_SLOT_STDDEV
        )
        if active_slots < _MIN_ACTIVE_SLOTS:
            continue

        score = float(active_slots * 10.0 + sum(edge for edge, _ in metrics) + sum(stddev for _, stddev in metrics) / 100.0)
        candidate = TeamStripDetection(
            layout_name=layout_name,
            strip=strip,
            slot_crops=slot_crops,
            active_slots=active_slots,
            score=score,
        )
        if best is None or candidate.score > best.score:
            best = candidate
    return best


def extract_team_slot_crops(frame: np.ndarray, slots: int = 3) -> List[np.ndarray]:
    detection = detect_team_strip(frame=frame, slots=slots)
    return detection.slot_crops if detection is not None else []


def team_strip_present(frame: np.ndarray, slots: int = 3) -> bool:
    return detect_team_strip(frame=frame, slots=slots) is not None
