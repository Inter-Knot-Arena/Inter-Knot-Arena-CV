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

## Operational policy

- `LOW_CONF` never triggers auto-penalty.
- Enforcement is server-side and requires valid violation evidence.

## Supported baseline

- Locale: RU, EN.
- Resolutions: 1080p, 1440p.
