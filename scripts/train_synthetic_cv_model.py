from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

RNG = np.random.default_rng(73)
AGENT_LABELS = [
    "agent_anby",
    "agent_nicole",
    "agent_ellen",
    "agent_lycaon",
    "agent_koleda",
    "agent_vivian",
]


def _agent_color(label: str) -> Tuple[int, int, int]:
    digest = abs(hash(f"cv:{label}"))
    return (
        80 + digest % 120,
        80 + (digest // 11) % 120,
        80 + (digest // 19) % 120,
    )


def render_icon(label: str) -> np.ndarray:
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    color = _agent_color(label)
    img[:] = (color[0] // 3, color[1] // 3, color[2] // 3)
    idx = AGENT_LABELS.index(label)
    cv2.circle(img, (16, 16), 5 + idx % 7, color, thickness=-1)
    cv2.line(img, (3, 6 + idx), (28, 25 - idx), (255 - color[0], 255 - color[1], 255 - color[2]), 2)
    cv2.rectangle(img, (2 + idx % 8, 24 - idx % 8), (10 + idx % 8, 30), color[::-1], thickness=-1)
    noise = RNG.normal(0, 8, size=img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def train_model(output_dir: Path) -> Dict[str, float]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    for idx, label in enumerate(AGENT_LABELS):
        for _ in range(1000):
            icon = render_icon(label)
            features.append(icon.astype(np.float32).reshape(-1) / 255.0)
            labels.append(idx)
    x = np.vstack(features).astype(np.float32)
    y = np.array(labels, dtype=np.int64)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = LogisticRegression(max_iter=900, solver="lbfgs")
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    accuracy = float(accuracy_score(y_test, predictions))

    model_path = output_dir / "cv_agent_icon.onnx"
    labels_path = output_dir / "cv_agent_icon.labels.json"
    initial_type = [("input", FloatTensorType([None, x.shape[1]]))]
    onnx_model = convert_sklearn(clf, initial_types=initial_type, target_opset=17)
    with model_path.open("wb") as fh:
        fh.write(onnx_model.SerializeToString())
    with labels_path.open("w", encoding="utf-8") as fh:
        json.dump({"labels": AGENT_LABELS}, fh, ensure_ascii=True, indent=2)
        fh.write("\n")

    return {"accuracy": accuracy}


def export_templates(templates_dir: Path) -> None:
    templates_dir.mkdir(parents=True, exist_ok=True)
    for label in AGENT_LABELS:
        template = render_icon(label)
        cv2.imwrite(str(templates_dir / f"{label}.png"), template)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train synthetic ONNX CV icon model and generate templates.")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--templates-dir", default="assets/templates")
    parser.add_argument("--metrics-file", default="docs/model_metrics.json")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    templates_dir = Path(args.templates_dir).resolve()

    metrics = {"cv_agent_icon_model": train_model(output_dir)}
    export_templates(templates_dir)

    metrics_path = Path(args.metrics_file).resolve()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=True, indent=2)
        fh.write("\n")

    print(f"Model exported to: {output_dir}")
    print(f"Templates exported to: {templates_dir}")
    print(json.dumps(metrics, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
