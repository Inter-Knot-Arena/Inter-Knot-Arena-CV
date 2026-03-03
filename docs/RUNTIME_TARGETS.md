# CV Runtime Targets (Draft)

## Accuracy targets

- Pre-check team detection accuracy >= 99%
- In-run icon verification accuracy >= 98%
- False violation rate <= 0.5%

## Performance targets

- Per-frame inference <= 50 ms on GTX 970
- Monitoring cadence 10-20 sec with negligible FPS impact

## Operational modes

- MODE_A: result + hash only
- MODE_B: crop + result (ranked default)
- MODE_C: crop + event full-frame (opt-in)
