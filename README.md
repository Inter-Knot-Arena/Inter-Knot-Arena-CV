# Inter-Knot-Arena-CV

Computer-vision package for match-time verification in VerifierApp.

## Scope

- Pre-check validation against drafted team.
- In-run periodic icon verification.
- Violation and low-confidence classification.

## Output contract

See `contracts/match-cv-output.schema.json`.

## Runtime pipeline (v1)

- `runtime/matcher.py` exposes `evaluate_detection(...)`.
- Frame-based detection path:
  - screenshot capture (`dxcam` DXGI first, fallback `PIL.ImageGrab`) or `--frame-path`
  - slot crops
  - ONNX icon classifier + template matching
  - temporal smoothing for in-run
- Result always contains:
  - `PASS | VIOLATION | LOW_CONF`
  - per-agent confidence
  - `lowConfReasons[]`
  - `frameHash`, `timingMs`, `resolution`, `locale`
- Missing expected team in payload is treated as `LOW_CONF` (no auto-violation).
- Banned-agent hit is treated as `VIOLATION`.
- In-run mode applies lightweight temporal smoothing from previous detections.

## Quick run

```powershell
pip install -r requirements.txt
python scripts/run_match_check.py --mode PRECHECK --expected agent_anby,agent_nicole,agent_ellen --detected agent_anby,agent_nicole,agent_ellen
```

In-run with history:

```powershell
python scripts/run_match_check.py --mode INRUN --expected agent_anby,agent_nicole,agent_ellen --detected agent_anby,agent_nicole --history agent_anby,agent_nicole,agent_ellen --locale RU --resolution 1440p
```

Train synthetic baseline CV model:

```powershell
pip install -r requirements.txt
python scripts/train_synthetic_cv_model.py --output-dir models --templates-dir assets/templates --metrics-file docs/model_metrics.json
```

Train production model from dataset manifest (with synthetic fallback when data is insufficient):

```powershell
python scripts/train_cv_model.py --manifest dataset_manifest.json --output-dir models --templates-dir assets/templates --metrics-file docs/model_metrics.json --model-version cv-agent-head-v1.3
```

Domain-adaptive training with private live backgrounds:

```powershell
python scripts/train_synthetic_cv_model.py --output-dir models --templates-dir assets/templates --metrics-file docs/model_metrics.json --background-dir D:\IKA_DATA\ika_live_backgrounds\cv --samples-per-class 1600
```

Benchmark runtime:

```powershell
python scripts/benchmark_runtime.py --iterations 100
```

## Dataset workflow (private raw data)

```powershell
python scripts/bootstrap_dataset.py --storage-root D:\IKA_DATA\cv
python scripts/ingest_public_sources.py --manifest dataset_manifest.json --sources-file D:\IKA_DATA\cv_sources.json
python scripts/extract_frames.py --manifest dataset_manifest.json --state precheck --fps 1.0 --scene-aware
python scripts/extract_frames.py --manifest dataset_manifest.json --state inrun --fps 0.5
python scripts/deduplicate_frames.py --manifest dataset_manifest.json --input-dir D:\IKA_DATA\cv\frames_precheck
python scripts/session_capture.py --manifest dataset_manifest.json --mode cv --duration-sec 180 --fps 1.0 --state inrun --locale RU --resolution 1080p
python scripts/prune_manifest.py --manifest dataset_manifest.json --drop-source live_session_1772675681 --drop-source live_session_1772677183
python scripts/prelabel_dataset.py --manifest dataset_manifest.json --confidence-threshold 0.7
python scripts/qa_audit.py --manifest dataset_manifest.json --output-file docs/qa_report.json --double-review-file docs/double_review_samples.json
python scripts/export_review_pack.py --manifest dataset_manifest.json --status needs_review --output-csv docs/review_queue.csv
# after manual edit of docs/review_queue.csv:
python scripts/apply_review_labels.py --manifest dataset_manifest.json --input-csv docs/review_queue.csv --review-round final --reviewer-id qa_operator_1
python scripts/build_sampling_plan.py --manifest dataset_manifest.json --target-per-agent 6000 --target-unknown 8000 --output-file docs/sampling_plan.json
python scripts/split_dataset.py --manifest dataset_manifest.json --seed 42
```

Raw media and crops remain private/local and are not committed to git.

## Fullscreen capture notes

- Runtime prefers DXGI Desktop Duplication via `dxcam` (works for borderless and most fullscreen scenarios).
- Optional: `IKA_CAPTURE_OUTPUT_IDX=1` to target a non-primary monitor.
