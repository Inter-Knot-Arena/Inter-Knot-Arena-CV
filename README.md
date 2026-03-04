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
  - screenshot capture (`PIL.ImageGrab`) or `--frame-path`
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

Benchmark runtime:

```powershell
python scripts/benchmark_runtime.py --iterations 100
```

## Dataset workflow (private raw data)

```powershell
python scripts/bootstrap_dataset.py --storage-root D:\IKA_DATA\cv
python scripts/split_dataset.py --manifest dataset_manifest.json --seed 42
```

Raw media and crops remain private/local and are not committed to git.
