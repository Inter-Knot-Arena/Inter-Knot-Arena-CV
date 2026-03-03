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
- Result always contains:
  - `PASS | VIOLATION | LOW_CONF`
  - per-agent confidence
  - `lowConfReasons[]`
  - `frameHash`, `timingMs`, `resolution`, `locale`
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

## Dataset workflow (private raw data)

```powershell
python scripts/bootstrap_dataset.py --storage-root D:\IKA_DATA\cv
python scripts/split_dataset.py --manifest dataset_manifest.json --seed 42
```

Raw media and crops remain private/local and are not committed to git.
