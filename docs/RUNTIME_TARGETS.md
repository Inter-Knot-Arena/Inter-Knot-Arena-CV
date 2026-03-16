# CV Runtime Targets (v1)

## Data policy

- Raw frames, crops, and videos are private local data.
- Git contains only manifest, scripts, configs, metrics, and release artifacts.

## Accuracy targets

- Pre-check team detection accuracy >= 99%.
- In-run icon verification accuracy >= 98%.
- False violation rate <= 0.5%.

## Performance targets

- Per-frame inference <= 50 ms on GTX 970 class GPU.
- Monitoring cadence 10-20 sec with low overhead.
- Product runtime must stay on GPU; silent CPU fallback is not acceptable.

## Baseline implementation in repo

- `scripts/train_cv_model.py` trains from manifest-backed real data on CUDA only.
- Product training is blocked when the dataset does not cover the full current roster, unless `--allow-partial-roster` is set explicitly for exploratory runs.
- `scripts/train_synthetic_cv_model.py` exports synthetic baseline:
  - `models/cv_agent_icon.onnx`
  - `models/cv_agent_icon.labels.json`
  - `models/model_manifest.json`
  - `assets/templates/*.png`
- `scripts/audit_team_strip_dataset.py` measures how much of the manifest actually contains a valid top team-strip.
- `scripts/extract_frames.py` rejects `precheck` and `inrun` frames without a valid team-strip unless explicitly overridden.
- `runtime/matcher.py` combines ONNX probabilities with template matching and temporal smoothing.
- `scripts/benchmark_runtime.py` provides latency percentile benchmark.
- Dataset ingestion pipeline scripts:
  - `scripts/ingest_public_sources.py`
  - `scripts/extract_frames.py`
  - `scripts/deduplicate_frames.py`
  - `scripts/session_capture.py`
  - `scripts/build_sampling_plan.py`

## Operational policy

- `LOW_CONF` never triggers auto-penalty.
- Enforcement is server-side and requires valid violation evidence.
- Missing `expectedAgents` is downgraded to `LOW_CONF`.
- Banned-agent detection is emitted as `VIOLATION`.

## Supported baseline

- Locale: RU, EN.
- Resolutions: 1080p, 1440p.
- Capture: DXGI (`dxcam`) first, PIL fallback.
