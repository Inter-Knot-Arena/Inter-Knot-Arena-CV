# Inter-Knot-Arena-CV

Computer-vision package for match-time verification in VerifierApp.

## Scope

- Pre-check validation against drafted team.
- In-run periodic icon verification.
- Violation and low-confidence classification.

## Output contract

See `contracts/match-cv-output.schema.json`.

## Runtime stub (v1)

- `runtime/matcher.py` exposes deterministic hybrid matcher.
- `scripts/run_match_check.py` emits contract-compliant payload for `PRECHECK`/`INRUN`.

## Quick run

```powershell
pip install -r requirements.txt
python scripts/run_match_check.py --mode PRECHECK --expected agent_anby,agent_nicole,agent_ellen --detected agent_anby,agent_nicole,agent_ellen
```
