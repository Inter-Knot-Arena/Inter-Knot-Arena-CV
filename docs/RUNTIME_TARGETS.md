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
- CPU fallback remains available when DirectML is unavailable.

## Baseline implementation in repo

- `scripts/train_synthetic_cv_model.py` exports:
  - `models/cv_agent_icon.onnx`
  - `models/cv_agent_icon.labels.json`
  - `assets/templates/*.png`
- `runtime/matcher.py` combines ONNX probabilities with template matching and temporal smoothing.
- `scripts/benchmark_runtime.py` provides latency percentile benchmark.

## Operational policy

- `LOW_CONF` never triggers auto-penalty.
- Enforcement is server-side and requires valid violation evidence.
- Missing `expectedAgents` is downgraded to `LOW_CONF`.
- Banned-agent detection is emitted as `VIOLATION`.

## Supported baseline

- Locale: RU, EN.
- Resolutions: 1080p, 1440p.
- Capture: DXGI (`dxcam`) first, PIL fallback.
