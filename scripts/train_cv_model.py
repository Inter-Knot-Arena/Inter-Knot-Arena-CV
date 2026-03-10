from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from manifest_lib import hash_file_sha256
from roster_taxonomy import canonicalize_agent_label, current_agent_ids
from train_synthetic_cv_model import export_templates, train_model as train_synthetic_model

DEFAULT_MODEL_VERSION = "cv-agent-head-v1.4"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def _torch_cuda_available() -> bool:
    if not _torch_available():
        return False
    import torch

    return bool(torch.cuda.is_available())


def _is_valid_agent_label(label: str) -> bool:
    value = str(label or "").strip()
    return bool(canonicalize_agent_label(value) or value == "unknown")


def _extract_label(record: Dict[str, Any]) -> str:
    labels = record.get("labels")
    if isinstance(labels, dict):
        for key in ("agentId", "slot_1_agent", "label"):
            value = labels.get(key)
            if isinstance(value, str) and value.strip():
                candidate = value.strip()
                canonical = canonicalize_agent_label(candidate)
                if canonical:
                    return canonical
    for key in ("agentId", "label"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            candidate = value.strip()
            canonical = canonicalize_agent_label(candidate)
            if canonical:
                return canonical
    unknown_flag = record.get("unknownFlag")
    if unknown_flag is True:
        return "unknown"
    return ""


def _slot_crops(frame: np.ndarray, orientation: str, slots: int = 3) -> List[np.ndarray]:
    h, w = frame.shape[:2]
    crops: List[np.ndarray] = []
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


def _extract_slot_labels(record: Dict[str, Any]) -> List[Tuple[int, str]]:
    labels = record.get("labels")
    if not isinstance(labels, dict):
        return []
    slot_labels: List[Tuple[int, str]] = []
    for idx in range(1, 4):
        key = f"slot_{idx}_agent"
        value = labels.get(key)
        if isinstance(value, str) and value.strip():
            candidate = value.strip()
            canonical = canonicalize_agent_label(candidate)
            if canonical:
                slot_labels.append((idx - 1, canonical))
    return slot_labels


def _load_dataset(manifest_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    records = manifest.get("records", [])
    if not isinstance(records, list):
        raise ValueError("manifest.records must be an array")

    features: List[np.ndarray] = []
    labels: List[str] = []
    skipped = 0
    for record in records:
        if not isinstance(record, dict):
            skipped += 1
            continue
        path_value = str(record.get("path") or "")
        if not path_value:
            skipped += 1
            continue
        label = _extract_label(record)
        if not label:
            skipped += 1
            continue
        path = Path(path_value)
        if not path.exists():
            skipped += 1
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            skipped += 1
            continue

        slot_labels = _extract_slot_labels(record)
        if slot_labels:
            state = str(record.get("state") or "other").lower()
            orientation = "horizontal" if state == "precheck" else "vertical"
            crops = _slot_crops(image, orientation=orientation, slots=3)
            added = 0
            for slot_index, slot_label in slot_labels:
                if slot_index < 0 or slot_index >= len(crops):
                    continue
                crop = cv2.resize(crops[slot_index], (32, 32), interpolation=cv2.INTER_AREA)
                features.append(crop.astype(np.float32).reshape(-1) / 255.0)
                labels.append(slot_label)
                added += 1
            if added <= 0:
                skipped += 1
            continue

        image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
        features.append(image.astype(np.float32).reshape(-1) / 255.0)
        labels.append(label)

    if not features:
        return np.empty((0, 3072), dtype=np.float32), np.empty((0,), dtype=np.int64), [], skipped

    label_names = sorted(set(labels))
    index_map = {label: idx for idx, label in enumerate(label_names)}
    y = np.array([index_map[label] for label in labels], dtype=np.int64)
    x = np.vstack(features).astype(np.float32)
    return x, y, label_names, skipped


def _expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, bins: int = 15) -> float:
    if probs.size == 0:
        return 0.0
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correctness = (predictions == y_true).astype(np.float32)

    ece = 0.0
    for idx in range(bins):
        left = idx / bins
        right = (idx + 1) / bins
        mask = (confidences > left) & (confidences <= right)
        if not np.any(mask):
            continue
        bin_conf = float(np.mean(confidences[mask]))
        bin_acc = float(np.mean(correctness[mask]))
        ece += abs(bin_acc - bin_conf) * (np.sum(mask) / len(confidences))
    return float(ece)


def _latency_stats(clf: LogisticRegression, sample: np.ndarray, iterations: int = 120) -> Tuple[float, float]:
    latencies: List[float] = []
    if sample.size == 0:
        return 0.0, 0.0
    count = min(iterations, max(20, sample.shape[0]))
    for idx in range(count):
        row = sample[idx % sample.shape[0] : (idx % sample.shape[0]) + 1]
        started = time.perf_counter()
        _ = clf.predict_proba(row)
        latencies.append((time.perf_counter() - started) * 1000.0)
    p50 = float(statistics.median(latencies))
    p95 = float(np.percentile(np.array(latencies, dtype=np.float32), 95))
    return round(p50, 3), round(p95, 3)


def _onnx_latency_stats(model_path: Path, sample: np.ndarray, iterations: int = 120) -> Tuple[float, float]:
    if sample.size == 0 or not model_path.exists():
        return 0.0, 0.0
    providers = [provider for provider in ("DmlExecutionProvider", "CPUExecutionProvider") if provider in ort.get_available_providers()] or ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(model_path), providers=providers)
    input_name = session.get_inputs()[0].name
    latencies: List[float] = []
    count = min(iterations, max(20, sample.shape[0]))
    for idx in range(count):
        row = sample[idx % sample.shape[0] : (idx % sample.shape[0]) + 1].astype(np.float32, copy=False)
        started = time.perf_counter()
        _ = session.run(None, {input_name: row})
        latencies.append((time.perf_counter() - started) * 1000.0)
    p50 = float(statistics.median(latencies))
    p95 = float(np.percentile(np.array(latencies, dtype=np.float32), 95))
    return round(p50, 3), round(p95, 3)


def _stratify_target(y: np.ndarray) -> np.ndarray | None:
    if y.size <= 1:
        return None
    values, counts = np.unique(y, return_counts=True)
    if values.size <= 1:
        return None
    if int(np.min(counts)) < 2:
        return None
    return y


def _synthetic_fallback_metrics(raw_metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "accuracy": float(raw_metrics.get("accuracy", 0.0)),
        "macroF1": None,
        "precision": None,
        "recall": None,
        "ece": None,
        "latencyMsP50": None,
        "latencyMsP95": None,
        "backgroundCount": int(raw_metrics.get("backgroundCount", 0)),
        "evaluationMode": "synthetic_holdout_only",
        "trainingBackend": "synthetic_baseline",
        "trainingDevice": "cpu",
    }


def _resolve_backend(requested: str, requested_device: str) -> str:
    if requested == "auto":
        if not _torch_available():
            return "sklearn"
        if requested_device == "cuda" and not _torch_cuda_available():
            return "sklearn"
        return "torch"
    if requested == "torch":
        if not _torch_available():
            raise ImportError("Torch backend requested, but torch is not installed.")
        if requested_device == "cuda" and not _torch_cuda_available():
            raise RuntimeError("Torch CUDA backend requested, but CUDA is not available.")
    return requested


def _reshape_for_torch(x: np.ndarray) -> np.ndarray:
    return np.transpose(x.reshape((-1, 32, 32, 3)), (0, 3, 1, 2)).astype(np.float32)


def _train_sklearn_model(
    x: np.ndarray,
    y: np.ndarray,
    label_names: List[str],
    output_dir: Path,
) -> Dict[str, Any]:
    if x.shape[0] < 10:
        raise ValueError("Not enough samples for real training.")

    values, counts = np.unique(y, return_counts=True)
    if values.size < 2:
        raise ValueError("Need at least two classes for real training.")
    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=_stratify_target(y),
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=_stratify_target(y_temp),
    )

    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    probs = clf.predict_proba(x_test)

    accuracy = float(accuracy_score(y_test, preds))
    macro_f1 = float(f1_score(y_test, preds, average="macro", zero_division=0.0))
    precision = float(precision_score(y_test, preds, average="macro", zero_division=0.0))
    recall = float(recall_score(y_test, preds, average="macro", zero_division=0.0))
    ece = _expected_calibration_error(y_test, probs)
    latency_p50, latency_p95 = _latency_stats(clf, x_val)

    model_path = output_dir / "cv_agent_icon.onnx"
    labels_path = output_dir / "cv_agent_icon.labels.json"
    model_label_names = [label_names[int(class_index)] for class_index in clf.classes_]
    initial_type = [("input", FloatTensorType([None, x.shape[1]]))]
    onnx_model = convert_sklearn(clf, initial_types=initial_type, target_opset=17)
    with model_path.open("wb") as fh:
        fh.write(onnx_model.SerializeToString())
    with labels_path.open("w", encoding="utf-8") as fh:
        json.dump({"labels": model_label_names, "classIds": [int(class_index) for class_index in clf.classes_]}, fh, ensure_ascii=True, indent=2)
        fh.write("\n")

    return {
        "accuracy": accuracy,
        "macroF1": macro_f1,
        "precision": precision,
        "recall": recall,
        "ece": ece,
        "latencyMsP50": latency_p50,
        "latencyMsP95": latency_p95,
        "backgroundCount": 0,
        "evaluationMode": "real_holdout",
        "trainingBackend": "sklearn_logreg",
        "trainingDevice": "cpu",
    }


def _train_torch_model(
    x: np.ndarray,
    y: np.ndarray,
    label_names: List[str],
    output_dir: Path,
    requested_device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> Dict[str, Any]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    if x.shape[0] < 10:
        raise ValueError("Not enough samples for real training.")
    values, counts = np.unique(y, return_counts=True)
    if values.size < 2:
        raise ValueError("Need at least two classes for real training.")

    images = _reshape_for_torch(x)
    x_train, x_temp, y_train, y_temp = train_test_split(
        images,
        y,
        test_size=0.2,
        random_state=42,
        stratify=_stratify_target(y),
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=_stratify_target(y_temp),
    )

    device_name = requested_device.strip().lower()
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA backend requested, but torch.cuda.is_available() is false.")
    device = torch.device(device_name)

    num_classes = len(label_names)

    class TinyCvClassifier(nn.Module):
        def __init__(self, classes: int) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(32, 48, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Linear(48, classes)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            features = self.features(inputs)
            flattened = torch.flatten(features, 1)
            return self.classifier(flattened)

    model = TinyCvClassifier(classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=max(1e-5, float(learning_rate)))
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.int64))),
        batch_size=max(8, int(batch_size)),
        shuffle=True,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )

    for _ in range(max(1, int(epochs))):
        model.train()
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            batch_targets = batch_targets.to(device=device, dtype=torch.long, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_inputs)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()

    model.eval()
    probabilities: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x_test.shape[0], max(1, int(batch_size))):
            batch_np = x_test[start : start + max(1, int(batch_size))]
            batch_tensor = torch.from_numpy(batch_np).to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            logits = model(batch_tensor)
            batch_probs = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
            probabilities.append(batch_probs)
    probs = np.vstack(probabilities) if probabilities else np.empty((0, num_classes), dtype=np.float32)
    preds = np.argmax(probs, axis=1) if probs.size else np.empty((0,), dtype=np.int64)

    accuracy = float(accuracy_score(y_test, preds))
    macro_f1 = float(f1_score(y_test, preds, average="macro", zero_division=0.0))
    precision = float(precision_score(y_test, preds, average="macro", zero_division=0.0))
    recall = float(recall_score(y_test, preds, average="macro", zero_division=0.0))
    ece = _expected_calibration_error(y_test, probs)

    model_path = output_dir / "cv_agent_icon.onnx"
    labels_path = output_dir / "cv_agent_icon.labels.json"
    dummy_input = torch.from_numpy(x_train[:1]).to(device=device, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        str(model_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    with labels_path.open("w", encoding="utf-8") as fh:
        json.dump({"labels": label_names, "classIds": list(range(len(label_names)))}, fh, ensure_ascii=True, indent=2)
        fh.write("\n")

    latency_p50, latency_p95 = _onnx_latency_stats(model_path, x_val)
    return {
        "accuracy": accuracy,
        "macroF1": macro_f1,
        "precision": precision,
        "recall": recall,
        "ece": ece,
        "latencyMsP50": latency_p50,
        "latencyMsP95": latency_p95,
        "backgroundCount": 0,
        "evaluationMode": "real_holdout",
        "trainingBackend": "torch_cnn",
        "trainingDevice": device.type,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train CV agent icon model using manifest data with synthetic fallback.")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--templates-dir", default="assets/templates")
    parser.add_argument("--metrics-file", default="docs/model_metrics.json")
    parser.add_argument("--background-dir", default="", help="Optional directory for synthetic fallback augmentation.")
    parser.add_argument("--samples-per-class", type=int, default=1400, help="Synthetic fallback samples per class.")
    parser.add_argument("--min-real-samples", type=int, default=1200)
    parser.add_argument("--backend", choices=["auto", "sklearn", "torch"], default="auto")
    parser.add_argument("--torch-device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--data-version", default="")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    templates_dir = Path(args.templates_dir).resolve()
    metrics_path = Path(args.metrics_file).resolve()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    x, y, labels, skipped = _load_dataset(manifest_path=manifest_path)
    trained_with_real = x.shape[0] >= max(20, args.min_real_samples) and len(labels) >= 2
    selected_backend = _resolve_backend(args.backend, args.torch_device)

    if trained_with_real:
        try:
            if selected_backend == "torch":
                metrics = _train_torch_model(
                    x=x,
                    y=y,
                    label_names=labels,
                    output_dir=output_dir,
                    requested_device=args.torch_device,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                )
            else:
                metrics = _train_sklearn_model(x=x, y=y, label_names=labels, output_dir=output_dir)
            trained_label_names = list(labels)
        except Exception:
            trained_with_real = False
            background_dir = Path(args.background_dir).resolve() if args.background_dir else None
            fallback = train_synthetic_model(
                output_dir=output_dir,
                background_dir=background_dir,
                samples_per_class=max(200, args.samples_per_class),
            )
            metrics = _synthetic_fallback_metrics(fallback)
            trained_label_names = current_agent_ids()
    else:
        background_dir = Path(args.background_dir).resolve() if args.background_dir else None
        fallback = train_synthetic_model(
            output_dir=output_dir,
            background_dir=background_dir,
            samples_per_class=max(200, args.samples_per_class),
        )
        metrics = _synthetic_fallback_metrics(fallback)
        trained_label_names = current_agent_ids()

    export_templates(templates_dir)

    data_version = args.data_version
    if not data_version:
        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest_payload = json.load(fh)
        data_version = str(manifest_payload.get("version") or "unknown")

    model_path = output_dir / "cv_agent_icon.onnx"
    model_manifest_path = output_dir / "model_manifest.json"
    model_manifest = {
        "version": args.model_version,
        "sha256": hash_file_sha256(model_path) if model_path.exists() else "",
        "trainedAt": _utc_now(),
        "dataVersion": data_version,
        "backend": str(metrics.get("trainingBackend") or "unknown"),
    }
    with model_manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(model_manifest, fh, ensure_ascii=True, indent=2)
        fh.write("\n")

    metrics_payload = {
        "cv_agent_icon_model": {
            **metrics,
            "dataVersion": data_version,
            "trainedAt": model_manifest["trainedAt"],
            "recordCount": int(x.shape[0]),
            "skippedRecords": skipped,
            "mode": "real" if trained_with_real else "synthetic_fallback",
            "rosterAgentCount": len(current_agent_ids()),
            "trainedAgentCount": sum(1 for label in trained_label_names if label in set(current_agent_ids())),
            "missingRosterAgents": [agent for agent in current_agent_ids() if agent not in set(trained_label_names)],
        }
    }
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics_payload, fh, ensure_ascii=True, indent=2)
        fh.write("\n")

    print(json.dumps({"metrics": metrics_payload, "modelManifest": model_manifest}, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
